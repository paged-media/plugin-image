/*
 * This file is part of paged (https://paged.media).
 *
 * paged is free software: you may redistribute it and/or modify it under the
 * terms of the GNU Affero General Public License, version 3, as published by
 * the Free Software Foundation, OR under the Paged Media Enterprise License
 * (PMEL), a commercial license available from And The Next GmbH. Full
 * copyright and license information is available in LICENSE.md, distributed
 * with this source code.
 *
 * paged is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the licenses for details.
 *
 *  @copyright  Copyright (c) And The Next GmbH
 *  @license    AGPL-3.0-only OR Paged Media Enterprise License (PMEL)
 */

/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * This file is part of paged (https://paged.media) and is additionally
 * available under the Paged Media Enterprise License (PMEL). Full
 * copyright and license information is available in LICENSE.md which is
 * distributed with this source code.
 *
 *  @copyright  Copyright (c) And The Next GmbH
 *  @license    MPL-2.0 OR Paged Media Enterprise License (PMEL)
 */

//! Convolution family (T1, spec §11) — handwritten WGSL modules under
//! the ABI v1.1 contract (`abi::assemble` docs). `conv.box` is the
//! amendment proof: the first windowed kernel through the module lane
//! and the windowed parity harness. `conv.gaussian_h`/`conv.gaussian_v`
//! are the separable two-pass Gaussian (spec §9.2); `conv.unsharp` is
//! the binary-point unsharp mask over (original, blurred).
//!
//! Provenance: separable convolution and box filtering are standard
//! literature; unsharp masking is standard (W3C `feGaussianBlur`-style
//! sharpening: out = a + amount·(a − blurred)); no reference reading.

use crate::{KernelClass, KernelDef, ParamField, ParamsLayout, Tolerance};

/// 3×3 box mean — params are the bare ABI pad.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct ConvBoxParams {
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl ConvBoxParams {
    pub fn new() -> Self {
        Self { _abi_pad: 0 }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// out = mean of the 3×3 window (radius 1,1); mask-mixed against the
/// window center per the windowed convention.
pub static CONV_BOX: KernelDef = KernelDef {
    id: "conv.box",
    class: KernelClass::Windowed { radius: (1, 1) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<ConvBoxParams>(),
        fields: &[],
    },
    wgsl: CONV_BOX_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(2),
};

// Summation order (dy outer ascending, dx inner ascending) is part of
// the kernel's determinism contract — the scalar reference mirrors it
// exactly (§6.3 fixed reduction order).
const CONV_BOX_WGSL: &str = "\
// paged.image kernel `conv.box` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    // Window center: output (x, y) maps to in0 (x + rx, y + ry).
    let c = xy + vec2<i32>(1, 1);
    var sum = vec4<f32>(0.0);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            sum = sum + textureLoad(in0, c + vec2<i32>(dx, dy), 0);
        }
    }
    let result = sum / 9.0;
    let center = textureLoad(in0, c, 0);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(center, result, vec4<f32>(m)));
}
";

// ───────────────────────────── gaussian ────────────────────────────
//
// Separable Gaussian (spec §9.2): a 2D Gaussian convolution factors
// into a horizontal 1D pass followed by a vertical 1D pass over the
// intermediate. Each pass is a handwritten windowed module computing
// its weights in-shader:
//
//     w_i = exp(-i² / (2σ²))   for i ∈ -r..=r,   r = p.radius
//
// normalized by their sum S = Σ_{i=-r..=r} w_i. DETERMINISM: both the
// normalization sum S and the weighted convolution are accumulated in
// ASCENDING i order (i from -r to +r); the scalar reference mirrors
// this exactly (§6.3 fixed reduction order). exp() is a transcendental
// — its last-ulp f32 divergence between WGSL and Rust is absorbed by
// the f16 output quantization (tolerance ChannelEpsF16(4)).
//
// MAX RADIUS: the `KernelClass::Windowed` radius is the ROI-planning
// MAX bound; we fix it at 24 at compile time. The module ALWAYS treats
// in0 as `out + 2·(24,0)` (h) / `out + 2·(0,24)` (v) — sample offsets
// for any p.radius ≤ 24 stay inside that fixed, centered window. The
// shader and the reference both guard `p.radius <= 24`.

/// Compile-time MAX Gaussian radius (the windowed ROI bound, §8.3).
pub const GAUSSIAN_MAX_RADIUS: u16 = 24;

/// Gaussian pass params: blur σ and the (clamped ≤ 24) integer radius.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct ConvGaussianParams {
    pub sigma: f32,
    pub radius: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl ConvGaussianParams {
    pub fn new(sigma: f32, radius: u32) -> Self {
        Self {
            sigma,
            radius,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const GAUSSIAN_PARAM_FIELDS: &[ParamField] = &[
    ParamField {
        name: "sigma",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "radius",
        wgsl_ty: "u32",
    },
];

/// Horizontal Gaussian pass — windows in x only (radius (24, 0)).
/// `mip_exact`: σ halves per mip level (radius scales with it); the
/// engine rescales params per §8.3.
pub static CONV_GAUSSIAN_H: KernelDef = KernelDef {
    id: "conv.gaussian_h",
    class: KernelClass::Windowed {
        radius: (GAUSSIAN_MAX_RADIUS, 0),
    },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<ConvGaussianParams>(),
        fields: GAUSSIAN_PARAM_FIELDS,
    },
    wgsl: CONV_GAUSSIAN_H_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

/// Vertical Gaussian pass — windows in y only (radius (0, 24)).
pub static CONV_GAUSSIAN_V: KernelDef = KernelDef {
    id: "conv.gaussian_v",
    class: KernelClass::Windowed {
        radius: (0, GAUSSIAN_MAX_RADIUS),
    },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<ConvGaussianParams>(),
        fields: GAUSSIAN_PARAM_FIELDS,
    },
    wgsl: CONV_GAUSSIAN_V_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

// in0 = out + 2·(24, 0); output (x, y) ↔ window center (x + 24, y).
// Sums (normalization S then convolution) accumulate i = -r..=r
// ascending; weights divide by S.
const CONV_GAUSSIAN_H_WGSL: &str = "\
// paged.image kernel `conv.gaussian_h` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    sigma: f32,
    radius: u32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    // Window center: in0 expanded by the fixed MAX radius 24 in x.
    let c = xy + vec2<i32>(24, 0);
    var r = i32(min(params.radius, 24u));
    let inv2s2 = 1.0 / (2.0 * params.sigma * params.sigma);
    // Normalization sum S, ascending i.
    var s = 0.0;
    for (var i = -r; i <= r; i = i + 1) {
        let fi = f32(i);
        s = s + exp(-(fi * fi) * inv2s2);
    }
    // Weighted convolution, ascending i; weight = w_i / S.
    var acc = vec4<f32>(0.0);
    for (var i = -r; i <= r; i = i + 1) {
        let fi = f32(i);
        let w = exp(-(fi * fi) * inv2s2) / s;
        acc = acc + textureLoad(in0, c + vec2<i32>(i, 0), 0) * w;
    }
    let center = textureLoad(in0, c, 0);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(center, acc, vec4<f32>(m)));
}
";

// in0 = out + 2·(0, 24); output (x, y) ↔ window center (x, y + 24).
const CONV_GAUSSIAN_V_WGSL: &str = "\
// paged.image kernel `conv.gaussian_v` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    sigma: f32,
    radius: u32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    // Window center: in0 expanded by the fixed MAX radius 24 in y.
    let c = xy + vec2<i32>(0, 24);
    var r = i32(min(params.radius, 24u));
    let inv2s2 = 1.0 / (2.0 * params.sigma * params.sigma);
    // Normalization sum S, ascending i.
    var s = 0.0;
    for (var i = -r; i <= r; i = i + 1) {
        let fi = f32(i);
        s = s + exp(-(fi * fi) * inv2s2);
    }
    // Weighted convolution, ascending i; weight = w_i / S.
    var acc = vec4<f32>(0.0);
    for (var i = -r; i <= r; i = i + 1) {
        let fi = f32(i);
        let w = exp(-(fi * fi) * inv2s2) / s;
        acc = acc + textureLoad(in0, c + vec2<i32>(0, i), 0) * w;
    }
    let center = textureLoad(in0, c, 0);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(center, acc, vec4<f32>(m)));
}
";

// ───────────────────────────── unsharp ─────────────────────────────
//
// Unsharp masking (standard): given the original `a` and a blurred copy
// `b`, sharpen by adding back the high-frequency residual —
//
//     delta = a − b
//     out_c = a_c + amount·delta_c   where |delta_c| > threshold, else a_c
//
// per channel. This is a BINARY POINT kernel (in0 = original, in1 =
// blurred): the blurred input is produced upstream (the Gaussian pair),
// so unsharp itself runs on the existing point lane (`execute_tile_once`
// / `parity`). M0 tests use threshold 0.0 (every channel sharpened).
// The module applies the ABI mask itself: `mix(a, result, m)`.

/// Unsharp params: sharpening `amount` and per-channel `threshold`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct ConvUnsharpParams {
    pub amount: f32,
    pub threshold: f32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl ConvUnsharpParams {
    pub fn new(amount: f32, threshold: f32) -> Self {
        Self {
            amount,
            threshold,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// out = a + amount·(a − b) where |a − b| > threshold per channel, else
/// a. Binary point kernel; mip_exact (a pure pointwise blend of two
/// already-mip-correct inputs).
pub static CONV_UNSHARP: KernelDef = KernelDef {
    id: "conv.unsharp",
    class: KernelClass::Point,
    inputs: 2,
    params: ParamsLayout {
        size: ::core::mem::size_of::<ConvUnsharpParams>(),
        fields: &[
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "threshold",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: CONV_UNSHARP_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

// Per-channel: delta = a − b; sharpen channels whose |delta| exceeds
// threshold, pass the rest through. `select(a, sharp, |delta| > thr)`
// is componentwise (vec4<bool> selector). Mask-mixed against `a`.
const CONV_UNSHARP_WGSL: &str = "\
// paged.image kernel `conv.unsharp` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    amount: f32,
    threshold: f32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(0) @binding(1) var in1 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let a = textureLoad(in0, xy, 0);
    let b = textureLoad(in1, xy, 0);
    let delta = a - b;
    let sharp = a + delta * params.amount;
    let above = abs(delta) > vec4<f32>(params.threshold);
    let result = select(a, sharp, above);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

pub static FAMILY: &[&KernelDef] = &[&CONV_BOX, &CONV_GAUSSIAN_H, &CONV_GAUSSIAN_V, &CONV_UNSHARP];
