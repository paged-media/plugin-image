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
//! `conv.shape` (2026-08) is the family's one TWO-INPUT windowed
//! kernel: the convolution weight for each tap comes from a coverage
//! bitmap on `in1` instead of from a formula, which is what makes
//! "pick a shape, set a radius" possible at all. It is the exact
//! generalisation of `conv.lens` — that kernel's flat-topped disc is
//! one particular bitmap — and it inherits `conv.lens`'s `R_MAX`, its
//! derived-halo prologue and its "return the centre pixel rather than
//! divide a one-tap sum" guard.
//!
//! Provenance: separable convolution and box filtering are standard
//! literature; unsharp masking is standard (W3C `feGaussianBlur`-style
//! sharpening: out = a + amount·(a − blurred)); convolution with a
//! user-supplied kernel image is the textbook definition of a discrete
//! 2D convolution and needs no reference reading.

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

// ── Stylize: emboss + find edges ───────────────────────────────────
//
// Both are 3×3 convolutions and they ship TOGETHER because they are the
// same neighbourhood read with a different tap set — writing one and not
// the other would mean two window loops that must stay in step.
//
// EMBOSS keeps luminance and adds a directional derivative, so a flat
// region stays mid-grey rather than going black. That mid-grey bias is
// the whole reason emboss reads as relief instead of as an edge map.
//
// FIND EDGES is the Sobel MAGNITUDE, and it is INVERTED — Photoshop's
// Find Edges draws dark lines on white, not white on black. Getting that
// backwards produces something that looks like an effect and is the
// wrong one.

/// Emboss params: light direction in degrees and relief height.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct ConvEmbossParams {
    pub angle_deg: f32,
    pub height: f32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl ConvEmbossParams {
    pub fn new(angle_deg: f32, height: f32) -> Self {
        Self {
            angle_deg,
            height,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Directional relief: mid-grey plus the derivative along `angle_deg`.
pub static CONV_EMBOSS: KernelDef = KernelDef {
    id: "conv.emboss",
    class: KernelClass::Windowed { radius: (1, 1) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<ConvEmbossParams>(),
        fields: &[
            ParamField {
                name: "angle_deg",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "height",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: CONV_EMBOSS_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const CONV_EMBOSS_WGSL: &str = "\
// paged.image kernel `conv.emboss` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    angle_deg: f32,
    height: f32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

fn tap(xy: vec2<i32>, dims: vec2<i32>) -> vec3<f32> {
    let c = clamp(xy, vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    return textureLoad(in0, c, 0).rgb;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let dims = vec2<i32>(i32(d.x), i32(d.y));
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    // ABI: a Windowed kernel is handed `in0` expanded by the radius, so
    // output (x,y) is window centre (x+rx, y+ry). DERIVE that offset
    // rather than hardcoding it — the one-shot dispatcher
    // (`execute_tile_once_async`) binds `in0` at the OUTPUT dims with no
    // halo, and a hardcoded +r would read shifted there. Deriving gives
    // r under the tiled path and 0 under the one-shot path, so the
    // kernel is correct under both instead of only the one it was
    // written against.
    let wd = textureDimensions(in0);
    let win = vec2<i32>(i32(wd.x), i32(wd.y));
    let halo = (win - dims) / 2;
    let base = xy + halo;
    let a = textureLoad(in0, base, 0);

    // One offset pair along the light direction: the derivative is the
    // difference across the pixel, which is what gives relief a side.
    let r = radians(params.angle_deg);
    let off = vec2<i32>(i32(round(cos(r))), i32(round(-sin(r))));
    let fwd = tap(base + off, win);
    let bwd = tap(base - off, win);
    // 0.5 bias keeps a FLAT region mid-grey instead of black.
    let rel = vec3<f32>(0.5) + (fwd - bwd) * params.height * 0.5;
    let result = vec4<f32>(clamp(rel, vec3<f32>(0.0), vec3<f32>(1.0)), a.a);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

/// Find-edges params: `strength` scales the gradient before inversion.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct ConvFindEdgesParams {
    pub strength: f32,
    pub _abi_pad0: u32,
    pub _abi_pad1: u32,
}

#[allow(clippy::new_without_default)]
impl ConvFindEdgesParams {
    pub fn new(strength: f32) -> Self {
        Self {
            strength,
            _abi_pad0: 0,
            _abi_pad1: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Inverted Sobel magnitude — DARK lines on white, as Photoshop draws.
pub static CONV_FIND_EDGES: KernelDef = KernelDef {
    id: "conv.find_edges",
    class: KernelClass::Windowed { radius: (1, 1) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<ConvFindEdgesParams>(),
        fields: &[ParamField {
            name: "strength",
            wgsl_ty: "f32",
        }],
    },
    wgsl: CONV_FIND_EDGES_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const CONV_FIND_EDGES_WGSL: &str = "\
// paged.image kernel `conv.find_edges` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    strength: f32,
    _abi_pad0: u32,
    _abi_pad1: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

fn tap(xy: vec2<i32>, dims: vec2<i32>) -> vec3<f32> {
    let c = clamp(xy, vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    return textureLoad(in0, c, 0).rgb;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let dims = vec2<i32>(i32(d.x), i32(d.y));
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    // ABI: a Windowed kernel is handed `in0` expanded by the radius, so
    // output (x,y) is window centre (x+rx, y+ry). DERIVE that offset
    // rather than hardcoding it — the one-shot dispatcher
    // (`execute_tile_once_async`) binds `in0` at the OUTPUT dims with no
    // halo, and a hardcoded +r would read shifted there. Deriving gives
    // r under the tiled path and 0 under the one-shot path, so the
    // kernel is correct under both instead of only the one it was
    // written against.
    let wd = textureDimensions(in0);
    let win = vec2<i32>(i32(wd.x), i32(wd.y));
    let halo = (win - dims) / 2;
    let base = xy + halo;
    let a = textureLoad(in0, base, 0);

    // Sobel, per channel. Run on COLOUR rather than luminance so a
    // red-on-green edge at equal luma still registers — a luma-only
    // gradient misses exactly the edges a designer drew deliberately.
    let tl = tap(base + vec2<i32>(-1, -1), win);
    let tc = tap(base + vec2<i32>( 0, -1), win);
    let tr = tap(base + vec2<i32>( 1, -1), win);
    let ml = tap(base + vec2<i32>(-1,  0), win);
    let mr = tap(base + vec2<i32>( 1,  0), win);
    let bl = tap(base + vec2<i32>(-1,  1), win);
    let bc = tap(base + vec2<i32>( 0,  1), win);
    let br = tap(base + vec2<i32>( 1,  1), win);

    let gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    let gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
    let mag = sqrt(gx * gx + gy * gy) * params.strength;
    // INVERTED: Photoshop draws dark edges on white.
    let edges = clamp(vec3<f32>(1.0) - mag, vec3<f32>(0.0), vec3<f32>(1.0));
    let result = vec4<f32>(edges, a.a);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

// ── Blur Gallery: motion + radial ──────────────────────────────────
//
// Both accumulate along a PATH rather than over a box, which is what
// separates them from the Gaussian pair — and why they cannot be a
// separable two-pass like Gaussian is. Motion walks a straight line,
// radial walks an arc or a ray about a centre; the loop is otherwise
// identical, so the two share their sampling and their edge rule.
//
// TAP COUNT IS FIXED, not derived from the length. A length-derived
// count makes cost depend on a slider, and a designer dragging one
// would fall off a performance cliff mid-gesture. A fixed count with a
// varying STEP keeps the cost flat and degrades by undersampling —
// visible as banding at extreme lengths, which is the honest failure.

/// Motion-blur params: direction in degrees and length in PIXELS.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct ConvMotionParams {
    pub angle_deg: f32,
    pub length_px: f32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl ConvMotionParams {
    pub fn new(angle_deg: f32, length_px: f32) -> Self {
        Self {
            angle_deg,
            length_px,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Directional blur: average along a line through the pixel.
///
/// The window radius is the MAXIMUM the kernel will read, not the
/// current length — the ROI must be inflated for the worst case a
/// parameter can ask for, or a long blur reads outside its tile.
pub static CONV_MOTION: KernelDef = KernelDef {
    id: "conv.motion",
    class: KernelClass::Windowed { radius: (32, 32) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<ConvMotionParams>(),
        fields: &[
            ParamField {
                name: "angle_deg",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "length_px",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: CONV_MOTION_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const CONV_MOTION_WGSL: &str = "\
// paged.image kernel `conv.motion` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    angle_deg: f32,
    length_px: f32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

const TAPS : i32 = 17;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let dims = vec2<i32>(i32(d.x), i32(d.y));
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    // ABI: a Windowed kernel is handed `in0` expanded by the radius, so
    // output (x,y) is window centre (x+rx, y+ry). DERIVE that offset
    // rather than hardcoding it — the one-shot dispatcher
    // (`execute_tile_once_async`) binds `in0` at the OUTPUT dims with no
    // halo, and a hardcoded +r would read shifted there. Deriving gives
    // r under the tiled path and 0 under the one-shot path, so the
    // kernel is correct under both instead of only the one it was
    // written against.
    let wd = textureDimensions(in0);
    let win = vec2<i32>(i32(wd.x), i32(wd.y));
    let halo = (win - dims) / 2;
    let base = xy + halo;
    let a = textureLoad(in0, base, 0);

    let r = radians(params.angle_deg);
    let dir = vec2<f32>(cos(r), -sin(r));
    // Centred on the pixel: half the length each way, so the subject
    // smears symmetrically rather than sliding off its own position.
    let half = params.length_px * 0.5;
    var acc = vec4<f32>(0.0);
    for (var i : i32 = 0; i < TAPS; i = i + 1) {
        let t = (f32(i) / f32(TAPS - 1)) * 2.0 - 1.0;
        let p = vec2<f32>(f32(base.x), f32(base.y)) + dir * (t * half);
        let c = clamp(vec2<i32>(i32(round(p.x)), i32(round(p.y))),
                      vec2<i32>(0, 0), win - vec2<i32>(1, 1));
        acc = acc + textureLoad(in0, c, 0);
    }
    let result = acc / f32(TAPS);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

/// Radial-blur params: centre in NORMALISED coords, amount, and mode
/// (0 = spin, 1 = zoom).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct ConvRadialParams {
    pub cx: f32,
    pub cy: f32,
    pub amount: f32,
    pub mode: u32,
}

#[allow(clippy::new_without_default)]
impl ConvRadialParams {
    pub fn new(cx: f32, cy: f32, amount: f32, spin: bool) -> Self {
        Self {
            cx,
            cy,
            amount,
            mode: if spin { 0 } else { 1 },
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Spin / zoom blur about a centre. Photoshop's two Radial Blur modes
/// are ONE kernel: both walk an arc, and zoom is the degenerate arc
/// that keeps the angle and varies the radius.
pub static CONV_RADIAL: KernelDef = KernelDef {
    id: "conv.radial",
    class: KernelClass::Windowed { radius: (32, 32) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<ConvRadialParams>(),
        fields: &[
            ParamField {
                name: "cx",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "cy",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "mode",
                wgsl_ty: "u32",
            },
        ],
    },
    wgsl: CONV_RADIAL_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const CONV_RADIAL_WGSL: &str = "\
// paged.image kernel `conv.radial` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    cx: f32,
    cy: f32,
    amount: f32,
    mode: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

const TAPS : i32 = 17;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let dims = vec2<i32>(i32(d.x), i32(d.y));
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    // ABI: a Windowed kernel is handed `in0` expanded by the radius, so
    // output (x,y) is window centre (x+rx, y+ry). DERIVE that offset
    // rather than hardcoding it — the one-shot dispatcher
    // (`execute_tile_once_async`) binds `in0` at the OUTPUT dims with no
    // halo, and a hardcoded +r would read shifted there. Deriving gives
    // r under the tiled path and 0 under the one-shot path, so the
    // kernel is correct under both instead of only the one it was
    // written against.
    let wd = textureDimensions(in0);
    let win = vec2<i32>(i32(wd.x), i32(wd.y));
    let halo = (win - dims) / 2;
    let base = xy + halo;
    let a = textureLoad(in0, base, 0);

    let centre = vec2<f32>(params.cx * f32(dims.x), params.cy * f32(dims.y))
               + vec2<f32>(f32(halo.x), f32(halo.y));
    let p = vec2<f32>(f32(base.x), f32(base.y));
    let rel = p - centre;
    let rad = length(rel);
    let ang = atan2(rel.y, rel.x);

    var acc = vec4<f32>(0.0);
    for (var i : i32 = 0; i < TAPS; i = i + 1) {
        let t = (f32(i) / f32(TAPS - 1)) * 2.0 - 1.0;
        var q : vec2<f32>;
        if (params.mode == 0u) {
            // SPIN — vary the angle, hold the radius. The arc length
            // scales with radius, so the smear grows outward, which is
            // what makes a spin read as rotation rather than as noise.
            let a2 = ang + t * params.amount;
            q = centre + vec2<f32>(cos(a2), sin(a2)) * rad;
        } else {
            // ZOOM — hold the angle, vary the radius.
            q = centre + rel * (1.0 + t * params.amount);
        }
        let c = clamp(vec2<i32>(i32(round(q.x)), i32(round(q.y))),
                      vec2<i32>(0, 0), win - vec2<i32>(1, 1));
        acc = acc + textureLoad(in0, c, 0);
    }
    let result = acc / f32(TAPS);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

/// Lens-blur (bokeh) params — a DISC, plus the highlight bloom that is
/// the whole reason a lens blur does not look like a Gaussian.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct ConvLensParams {
    pub radius_px: f32,
    pub threshold: f32,
    pub boost: f32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl ConvLensParams {
    pub fn new(radius_px: f32, threshold: f32, boost: f32) -> Self {
        Self {
            radius_px,
            threshold,
            boost,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// LENS blur — a disc (circle-of-confusion) average with highlights
/// weighted UP before the average and pulled back after.
///
/// This is the one blur that genuinely could not be reached by
/// composing the ones already here. A Gaussian averages with a falloff,
/// so a bright speck spreads into a dim smudge. A real lens spreads it
/// into a bright DISC of roughly the source's brightness — the bokeh
/// ball. Reproducing that needs two things a Gaussian has not got: a
/// flat-topped support (so the disc has an edge), and a non-linear
/// weight so bright pixels survive the division by the tap count.
///
/// Provenance: the weight-boost-then-unboost trick is the standard
/// approximation of physical bokeh in real-time graphics; it is not a
/// port of any Adobe code, whose lens blur additionally reads a depth
/// map we have no equivalent for.
pub static CONV_LENS: KernelDef = KernelDef {
    id: "conv.lens",
    class: KernelClass::Windowed { radius: (24, 24) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<ConvLensParams>(),
        fields: &[
            ParamField {
                name: "radius_px",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "threshold",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "boost",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "_abi_pad",
                wgsl_ty: "u32",
            },
        ],
    },
    wgsl: CONV_LENS_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const CONV_LENS_WGSL: &str = "\
// paged.image kernel `conv.lens` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    radius_px: f32,
    threshold: f32,
    boost: f32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

const R_MAX : i32 = 24;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let dims = vec2<i32>(i32(d.x), i32(d.y));
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    // ABI: a Windowed kernel is handed `in0` expanded by the radius, so
    // output (x,y) is window centre (x+rx, y+ry). DERIVE that offset
    // rather than hardcoding it — the one-shot dispatcher
    // (`execute_tile_once_async`) binds `in0` at the OUTPUT dims with no
    // halo, and a hardcoded +r would read shifted there. Deriving gives
    // r under the tiled path and 0 under the one-shot path, so the
    // kernel is correct under both instead of only the one it was
    // written against.
    let wd = textureDimensions(in0);
    let win = vec2<i32>(i32(wd.x), i32(wd.y));
    let halo = (win - dims) / 2;
    let base = xy + halo;
    let a = textureLoad(in0, base, 0);

    let r = clamp(params.radius_px, 0.0, f32(R_MAX));
    let ri = i32(ceil(r));
    // Radius below half a pixel cannot cover a second sample; the disc
    // IS the centre pixel, so return it rather than divide by a
    // one-tap sum and call that a blur.
    if (ri < 1) {
        textureStore(outp, xy, a);
        return;
    }

    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var dy : i32 = -R_MAX; dy <= R_MAX; dy = dy + 1) {
        if (dy < -ri || dy > ri) { continue; }
        for (var dx : i32 = -R_MAX; dx <= R_MAX; dx = dx + 1) {
            if (dx < -ri || dx > ri) { continue; }
            // FLAT-TOPPED support: inside the circle or not at all.
            // That hard edge is what gives a bokeh ball its rim.
            if (f32(dx * dx + dy * dy) > r * r) { continue; }
            let c = clamp(base + vec2<i32>(dx, dy),
                          vec2<i32>(0, 0), win - vec2<i32>(1, 1));
            let s = textureLoad(in0, c, 0);
            let lum = dot(s.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            // Weight highlights UP so they survive the averaging, then
            // divide by the same weights: a bright speck stays bright
            // across the whole disc instead of averaging away.
            var w = 1.0;
            if (lum > params.threshold) {
                w = 1.0 + params.boost * (lum - params.threshold);
            }
            acc = acc + s * w;
            wsum = wsum + w;
        }
    }
    let result = select(a, acc / wsum, wsum > 0.0);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

/// Bilateral / smart-sharpen shared params.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct ConvBilateralParams {
    pub radius_px: f32,
    pub sigma_range: f32,
    pub amount: f32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl ConvBilateralParams {
    pub fn new(radius_px: f32, sigma_range: f32, amount: f32) -> Self {
        Self {
            radius_px,
            sigma_range,
            amount,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// REDUCE NOISE — a bilateral filter: average the neighbourhood, but
/// weight each tap by how close it is to the centre IN COLOUR as well
/// as in space. Taps across an edge differ in colour, so they get
/// almost no weight, and the edge survives a blur strong enough to
/// flatten the noise beside it.
///
/// `sigma_range` is the whole control: large enough and this degrades
/// to a Gaussian (every tap counts), small enough and it is the
/// identity (only the centre counts).
///
/// Provenance: Tomasi & Manduchi's bilateral filter (ICCV 1998) is
/// standard literature.
pub static CONV_BILATERAL: KernelDef = KernelDef {
    id: "conv.bilateral",
    class: KernelClass::Windowed { radius: (8, 8) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<ConvBilateralParams>(),
        fields: &[
            ParamField {
                name: "radius_px",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "sigma_range",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "_abi_pad",
                wgsl_ty: "u32",
            },
        ],
    },
    wgsl: CONV_BILATERAL_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const CONV_BILATERAL_WGSL: &str = "\
// paged.image kernel `conv.bilateral` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    radius_px: f32,
    sigma_range: f32,
    amount: f32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

const R_MAX : i32 = 8;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let dims = vec2<i32>(i32(d.x), i32(d.y));
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    // ABI: a Windowed kernel is handed `in0` expanded by the radius, so
    // output (x,y) is window centre (x+rx, y+ry). DERIVE that offset
    // rather than hardcoding it — the one-shot dispatcher
    // (`execute_tile_once_async`) binds `in0` at the OUTPUT dims with no
    // halo, and a hardcoded +r would read shifted there. Deriving gives
    // r under the tiled path and 0 under the one-shot path, so the
    // kernel is correct under both instead of only the one it was
    // written against.
    let wd = textureDimensions(in0);
    let win = vec2<i32>(i32(wd.x), i32(wd.y));
    let halo = (win - dims) / 2;
    let base = xy + halo;
    let a = textureLoad(in0, base, 0);

    let r = clamp(params.radius_px, 0.0, f32(R_MAX));
    let ri = i32(ceil(r));
    let sr = max(params.sigma_range, 0.0001);
    let ss = max(r * 0.5, 0.0001);

    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var dy : i32 = -R_MAX; dy <= R_MAX; dy = dy + 1) {
        if (dy < -ri || dy > ri) { continue; }
        for (var dx : i32 = -R_MAX; dx <= R_MAX; dx = dx + 1) {
            if (dx < -ri || dx > ri) { continue; }
            let c = clamp(base + vec2<i32>(dx, dy),
                          vec2<i32>(0, 0), win - vec2<i32>(1, 1));
            let s = textureLoad(in0, c, 0);
            let d2 = f32(dx * dx + dy * dy);
            let ws = exp(-d2 / (2.0 * ss * ss));
            // The RANGE term is the bilateral part: colour distance,
            // not spatial distance. This is what refuses to average
            // across an edge.
            let cd = s.rgb - a.rgb;
            let wr = exp(-dot(cd, cd) / (2.0 * sr * sr));
            let w = ws * wr;
            acc = acc + s * w;
            wsum = wsum + w;
        }
    }
    // The centre always contributes (dx=dy=0 gives w=1), so wsum is
    // never zero and the blend below is always defined.
    let filtered = acc / wsum;
    let result = mix(a, filtered, clamp(params.amount, 0.0, 1.0));
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

/// SMART SHARPEN — unsharp masking that only fires where there is an
/// edge to sharpen.
///
/// Plain unsharp adds `amount·(a − blurred)` everywhere, which
/// sharpens the noise in flat areas just as eagerly as it sharpens a
/// real edge, and rings the strong edges into visible halos. This adds
/// two gates over that: local contrast must clear `threshold` before
/// any sharpening applies at all, and the correction is clamped so a
/// high-contrast edge cannot overshoot into a halo.
pub static CONV_SMART_SHARPEN: KernelDef = KernelDef {
    id: "conv.smart_sharpen",
    class: KernelClass::Windowed { radius: (8, 8) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<ConvSmartSharpenParams>(),
        fields: &[
            ParamField {
                name: "radius_px",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "threshold",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "clamp_hi",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: CONV_SMART_SHARPEN_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct ConvSmartSharpenParams {
    pub radius_px: f32,
    pub amount: f32,
    pub threshold: f32,
    pub clamp_hi: f32,
}

#[allow(clippy::new_without_default)]
impl ConvSmartSharpenParams {
    pub fn new(radius_px: f32, amount: f32, threshold: f32, clamp_hi: f32) -> Self {
        Self {
            radius_px,
            amount,
            threshold,
            clamp_hi,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const CONV_SMART_SHARPEN_WGSL: &str = "\
// paged.image kernel `conv.smart_sharpen` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    radius_px: f32,
    amount: f32,
    threshold: f32,
    clamp_hi: f32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

const R_MAX : i32 = 8;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let dims = vec2<i32>(i32(d.x), i32(d.y));
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    // ABI: a Windowed kernel is handed `in0` expanded by the radius, so
    // output (x,y) is window centre (x+rx, y+ry). DERIVE that offset
    // rather than hardcoding it — the one-shot dispatcher
    // (`execute_tile_once_async`) binds `in0` at the OUTPUT dims with no
    // halo, and a hardcoded +r would read shifted there. Deriving gives
    // r under the tiled path and 0 under the one-shot path, so the
    // kernel is correct under both instead of only the one it was
    // written against.
    let wd = textureDimensions(in0);
    let win = vec2<i32>(i32(wd.x), i32(wd.y));
    let halo = (win - dims) / 2;
    let base = xy + halo;
    let a = textureLoad(in0, base, 0);

    let r = clamp(params.radius_px, 0.0, f32(R_MAX));
    let ri = max(i32(ceil(r)), 1);
    let ss = max(r * 0.5, 0.0001);

    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var dy : i32 = -R_MAX; dy <= R_MAX; dy = dy + 1) {
        if (dy < -ri || dy > ri) { continue; }
        for (var dx : i32 = -R_MAX; dx <= R_MAX; dx = dx + 1) {
            if (dx < -ri || dx > ri) { continue; }
            let c = clamp(base + vec2<i32>(dx, dy),
                          vec2<i32>(0, 0), win - vec2<i32>(1, 1));
            let w = exp(-f32(dx * dx + dy * dy) / (2.0 * ss * ss));
            acc = acc + textureLoad(in0, c, 0) * w;
            wsum = wsum + w;
        }
    }
    let blurred = acc / wsum;

    var diff = a.rgb - blurred.rgb;
    // GATE 1 — below the threshold this is noise, not an edge. Leave
    // it alone rather than amplify it.
    let mag = max(max(abs(diff.r), abs(diff.g)), abs(diff.b));
    if (mag < params.threshold) {
        textureStore(outp, xy, a);
        return;
    }
    // GATE 2 — clamp the correction. An unclamped high-contrast edge
    // overshoots into the bright halo that makes oversharpening
    // recognisable at a glance.
    diff = clamp(diff * params.amount,
                 vec3<f32>(-params.clamp_hi), vec3<f32>(params.clamp_hi));
    let result = vec4<f32>(clamp(a.rgb + diff, vec3<f32>(0.0), vec3<f32>(1.0)), a.a);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

// ─────────────────────────── shape blur ────────────────────────────
//
// Filter ▸ Blur ▸ Shape Blur: convolve with an ARBITRARY kernel shape.
// Photoshop ships a shape library and the user picks one plus a radius;
// here the shape is whatever coverage bitmap the caller binds to `in1`,
// so the "library" is a folder of images rather than a hardcoded set.
//
// WHY IT IS NOT `conv.lens` WITH A PARAMETER. `conv.lens` decides
// membership analytically (`dx² + dy² <= r²`). There is no analytic
// membership test for "a hexagon", "a five-pointed star" or "the user's
// logo" — the SHAPE IS THE DATA. That is the whole reason this kernel
// needs a second input, and it is why it is the last blur to be built:
// everything before it could be written as a formula.

/// `conv.shape` params — the SHAPE itself arrives as a coverage bitmap
/// on `in1`; these four numbers say how large to scale it, how much of
/// it to apply, and (because the dispatcher pads that bitmap) how much
/// of `in1` is actually shape.
///
/// - `radius_px` — HALF-EXTENT in DESTINATION pixels. The shape is
///   scaled to fill the box `[-radius_px, +radius_px]²` centred on the
///   output pixel, so radius 12 spans 25 px exactly like a `conv.lens`
///   disc of radius 12 does, and the two sliders mean the same thing.
///   Clamped to `R_MAX`; `< 0.5` is the identity.
/// - `amount` — wet/dry blend in `[0, 1]`. 0 is the identity, > 1
///   saturates at a fully wet result (`conv.bilateral`'s convention).
/// - `shape_w`, `shape_h` — the REAL extent of the shape inside `in1`,
///   in TEXELS. 0 means "the whole `in1` texture"; anything larger than
///   the bound texture clamps to it, so no tap can read outside.
///   This travels in the params for the reason `gen.pattern`'s
///   `tile_w`/`tile_h` do: `execute_tile_once_async` sizes EVERY input
///   texture at the OUTPUT dims, so a shape smaller than the
///   destination arrives zero-padded at the top-left and
///   `textureDimensions(in1)` would report the padded size — which
///   would scale a 64² star down into the corner of a 1024² box and
///   convolve with a mostly-empty field.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct ConvShapeParams {
    pub radius_px: f32,
    pub amount: f32,
    pub shape_w: u32,
    pub shape_h: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl ConvShapeParams {
    pub fn new(radius_px: f32, amount: f32, shape_w: u32, shape_h: u32) -> Self {
        Self {
            radius_px,
            amount,
            shape_w,
            shape_h,
            _abi_pad: 0,
        }
    }

    /// The engine's dispatch-skip predicate: true when the kernel is a
    /// documented no-op and the dispatch can be elided entirely.
    ///
    /// Written as NEGATED POSITIVE TESTS, exactly as the WGSL guards
    /// are. clippy would rewrite `!(x > 0.0)` to `x <= 0.0`, which is a
    /// different function at NaN — `NaN <= 0.0` is false, so a NaN
    /// slider would stop reading as a no-op here while the shader still
    /// treats it as one. A predicate that disagrees with its shader is
    /// worse than no predicate.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn is_identity(&self) -> bool {
        !(self.amount > 0.0) || !(self.radius_px >= 0.5)
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// SHAPE blur — a normalised convolution whose kernel is a bitmap.
///
/// `out = Σ wᵢ·sᵢ / Σ wᵢ` over the taps in the scaled shape box, where
/// `wᵢ` is the shape's RED channel at the tap's position and `sᵢ` is
/// the (premultiplied) image sample there, then `mix(a, that, amount)`.
///
/// RED, not alpha, is the coverage channel. A shape library is a set of
/// silhouettes, and the two ways anyone ships those are white-on-black
/// (alpha ≡ 1 everywhere — reading alpha would turn every shape into a
/// box) and white-on-transparent. In the premultiplied working space
/// the second form has `r == a` wherever the shape is white, so red is
/// correct for BOTH conventions while alpha is correct for only one.
/// It is also the channel the ABI's own selection mask is read from, so
/// "a coverage field lives in `.r`" stays one rule.
///
/// The divisor is the weight sum ACTUALLY ACCUMULATED, never a
/// precomputed constant. Two things make a constant wrong: the tap grid
/// is the destination pixel grid, so the discrete sum over a given
/// shape depends on `radius_px` and is not `∫shape` scaled; and taps
/// whose weight is zero, non-positive or NaN are skipped, so the
/// divisor is only ever the weight of the samples that entered the
/// numerator. Divide by anything else and the result darkens by exactly
/// the fraction of the footprint that did not contribute.
///
/// `R_MAX = 24` matches `conv.lens`, which is the right ceiling because
/// it is the same cost: the worst case is a 49×49 = 2401-tap gather,
/// already the most expensive kernel in the family. 32 would be
/// 65×65 = 4225 taps — 1.76× the work for a blur only 33% wider, which
/// is a bad trade to make on a slider a designer drags.
///
/// Provenance: discrete 2D convolution with a user-supplied kernel
/// image; normalised (weighted-mean) form so a non-unit-sum kernel does
/// not change exposure. Standard literature; no reference reading.
pub static CONV_SHAPE: KernelDef = KernelDef {
    id: "conv.shape",
    // WINDOWED in `in0` (the ROI must inflate by the largest radius the
    // slider can ask for) but BINARY, because `in1` is a whole-texture
    // RESOURCE read at a computed coordinate, not a co-located window —
    // the same asymmetry `gen.pattern` documents. The ABI has no class
    // that says "windowed in one input, resource in the other"; the
    // radius is the part an engine must act on, so that is what the
    // class states.
    class: KernelClass::Windowed { radius: (24, 24) },
    inputs: 2,
    params: ParamsLayout {
        size: ::core::mem::size_of::<ConvShapeParams>(),
        fields: &[
            ParamField {
                name: "radius_px",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "shape_w",
                wgsl_ty: "u32",
            },
            ParamField {
                name: "shape_h",
                wgsl_ty: "u32",
            },
        ],
    },
    wgsl: CONV_SHAPE_WGSL,
    module: true,
    // The shape is scaled in PIXELS, so a mip level must re-derive
    // `radius_px` (§8.3) — the same reason `conv.lens` is not mip-exact.
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const CONV_SHAPE_WGSL: &str = "\
// paged.image kernel `conv.shape` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    radius_px: f32,
    amount: f32,
    shape_w: u32,
    shape_h: u32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(0) @binding(1) var in1 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

const R_MAX : i32 = 24;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let dims = vec2<i32>(i32(d.x), i32(d.y));
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    // ABI: a Windowed kernel is handed `in0` expanded by the radius, so
    // output (x,y) is window centre (x+rx, y+ry). DERIVE that offset
    // rather than hardcoding it — the one-shot dispatcher
    // (`execute_tile_once_async`) binds `in0` at the OUTPUT dims with no
    // halo, and a hardcoded +r would read shifted there. Deriving gives
    // r under the tiled path and 0 under the one-shot path, so the
    // kernel is correct under both instead of only the one it was
    // written against. This kernel in particular is reached through the
    // ONE-SHOT path today (the binary door in image-js), so the derived
    // form is not defensive here — it is the live case.
    let wd = textureDimensions(in0);
    let win = vec2<i32>(i32(wd.x), i32(wd.y));
    let halo = (win - dims) / 2;
    let base = xy + halo;
    let a = textureLoad(in0, base, 0);

    // IDENTITY 1 — a dry mix. The NEGATED positive test also swallows a
    // NaN amount: IEEE comparisons with NaN are false on every lane, so
    // a NaN slider takes this branch deterministically on GPU and in any
    // reference twin. `clamp(amount, 0, 1)` would not — clamp/min of a
    // NaN is implementation-defined in WGSL, and 'the layer silently
    // became NaN' is the one failure an amount slider must not have.
    // The early return writes `a` RAW rather than through the mask
    // epilogue: `mix(a, a, m)` is `a*(1-m) + a*m`, which is only
    // bit-exactly `a` for dyadic m, and a no-op must be bit-exact.
    if (!(params.amount > 0.0)) {
        textureStore(outp, xy, a);
        return;
    }
    // Safe now: the value is known to be a real number > 0, so min()
    // has no NaN to be implementation-defined about. +inf saturates to 1.
    let amt = min(params.amount, 1.0);

    // IDENTITY 2 — a shape scaled to under one pixel across. Below half
    // a pixel the box [-r, +r] cannot reach a second sample, so the
    // convolution IS the centre pixel; return it rather than divide a
    // one-tap sum by its own weight and call that a blur. The threshold
    // is stated as 0.5 rather than left to `ceil(r) < 1`, which is only
    // ever true at exactly 0 and would let radius 0.2 pay for a
    // full-footprint loop to rediscover the input.
    if (!(params.radius_px >= 0.5)) {
        textureStore(outp, xy, a);
        return;
    }
    let r = min(params.radius_px, f32(R_MAX));
    let ri = i32(ceil(r));

    // Shape extent: params win, 0 means 'the whole in1 texture', and the
    // result is clamped to the bound texture so no tap can leave it. The
    // final max(_, 1) keeps the tap→texel scale below finite.
    let sd = textureDimensions(in1);
    var sw = i32(params.shape_w);
    var sh = i32(params.shape_h);
    if (sw <= 0 || sw > i32(sd.x)) { sw = i32(sd.x); }
    if (sh <= 0 || sh > i32(sd.y)) { sh = i32(sd.y); }
    sw = max(sw, 1);
    sh = max(sh, 1);

    // Tap → shape texel. A tap at offset f maps to (f + r)/(2r) of the
    // way across the shape; hoisting f32(sw)/(2r) out of the loop turns
    // that into one multiply per tap. r >= 0.5 by the guard above, so
    // the divisor is >= 1 and cannot blow up.
    let kx = f32(sw) / (2.0 * r);
    let ky = f32(sh) / (2.0 * r);

    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var dy : i32 = -R_MAX; dy <= R_MAX; dy = dy + 1) {
        if (dy < -ri || dy > ri) { continue; }
        let fy = f32(dy);
        // SUPPORT is the box [-r, +r], not [-ri, +ri]. ri = ceil(r)
        // overshoots for a fractional radius, and letting those taps in
        // would smear the shape's border row outward by a pixel — the
        // same reason conv.lens tests dx²+dy² against r² rather than
        // trusting its own loop bound.
        if (fy < -r || fy > r) { continue; }
        // NEAREST, not bilinear — see the long note in the dx loop.
        // min(_, sh-1) catches only the closed right endpoint fy == +r,
        // which maps to exactly sh under the texel-covers-[i, i+1)
        // convention; every interior tap truncates inside the shape.
        let sy = min(i32((fy + r) * ky), sh - 1);
        for (var dx : i32 = -R_MAX; dx <= R_MAX; dx = dx + 1) {
            if (dx < -ri || dx > ri) { continue; }
            let fx = f32(dx);
            if (fx < -r || fx > r) { continue; }
            let sx = min(i32((fx + r) * kx), sw - 1);
            // THE SAMPLING DECISION — nearest, deliberately, and NOT
            // what gen.pattern does with the same kind of second input.
            //
            // gen.pattern's sample is a COLOUR that goes straight to the
            // screen, so nearest would stair-step a visible edge. This
            // sample is a WEIGHT inside a normalised sum over up to 2401
            // taps: smoothing an individual weight moves the quotient by
            // O(1/taps) and is invisible.
            //
            // Where it IS visible is the shape's silhouette, and there
            // the hard edge is the POINT. A shape blur is chosen for its
            // recognisable bokeh — a highlight must smear into a
            // hexagon with a rim, the same flat-topped-support argument
            // conv.lens makes for its disc. Bilinear rounds that rim off
            // by a shape texel, softening the one feature that
            // distinguishes this from a Gaussian.
            //
            // And the antialiasing bilinear would supply is already
            // unavailable: the tap grid IS the destination pixel grid,
            // at most 49 taps across, while a shape asset is 128-256 px.
            // Consecutive taps land more than a shape texel apart, so
            // bilinear degenerates toward nearest anyway — it would cost
            // 4 textureLoads per tap (9604 instead of 2401 at the worst
            // case) to compute a number one load already gives.
            //
            // The honest failure mode: a shape bitmap SMALLER than the
            // tap footprint quantises the weight field into plateaus and
            // the bokeh terraces faintly. That is a shape-asset
            // resolution problem (ship shapes >= 128 px), the same class
            // of degradation as conv.motion's fixed tap count banding at
            // extreme lengths.
            let w = textureLoad(in1, vec2<i32>(sx, sy), 0).r;
            // Skip by a POSITIVE TEST, so zero, negative and NaN
            // coverage all fall out together. A negative weight could
            // drag wsum through zero and detonate the divide; a NaN
            // would poison the whole output — and mix(a, NaN, 0.0) is
            // NaN, so it would survive even a dry blend downstream.
            if (!(w > 0.0)) { continue; }
            // Edge-replicate, like every other windowed kernel here. The
            // clamped tap contributes a REAL sample and keeps its weight
            // in the divisor, so a shape overhanging the image border
            // neither darkens (which reading zeros would) nor gets a
            // smaller divisor than numerator.
            let c = clamp(base + vec2<i32>(dx, dy),
                          vec2<i32>(0, 0), win - vec2<i32>(1, 1));
            acc = acc + textureLoad(in0, c, 0) * w;
            wsum = wsum + w;
        }
    }

    // IDENTITY 3 — a DEGENERATE shape (all zeros, or every tap skipped)
    // gathered no weight at all. Return the input rather than divide by
    // zero. An explicit branch rather than select(): select evaluates
    // both arms, and there is no reason to compute acc/0.0 just to throw
    // it away. Negated test, so a NaN wsum lands here too.
    if (!(wsum > 0.0)) {
        textureStore(outp, xy, a);
        return;
    }
    let blurred = acc / wsum;

    let result = mix(a, blurred, vec4<f32>(amt));
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

pub static FAMILY: &[&KernelDef] = &[
    &CONV_LENS,
    &CONV_BILATERAL,
    &CONV_SMART_SHARPEN,
    &CONV_BOX,
    &CONV_GAUSSIAN_H,
    &CONV_GAUSSIAN_V,
    &CONV_UNSHARP,
    &CONV_EMBOSS,
    &CONV_FIND_EDGES,
    &CONV_MOTION,
    &CONV_RADIAL,
    &CONV_SHAPE,
];
