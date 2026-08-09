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
    let a = textureLoad(in0, xy, 0);

    // One offset pair along the light direction: the derivative is the
    // difference across the pixel, which is what gives relief a side.
    let r = radians(params.angle_deg);
    let off = vec2<i32>(i32(round(cos(r))), i32(round(-sin(r))));
    let fwd = tap(xy + off, dims);
    let bwd = tap(xy - off, dims);
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
    let a = textureLoad(in0, xy, 0);

    // Sobel, per channel. Run on COLOUR rather than luminance so a
    // red-on-green edge at equal luma still registers — a luma-only
    // gradient misses exactly the edges a designer drew deliberately.
    let tl = tap(xy + vec2<i32>(-1, -1), dims);
    let tc = tap(xy + vec2<i32>( 0, -1), dims);
    let tr = tap(xy + vec2<i32>( 1, -1), dims);
    let ml = tap(xy + vec2<i32>(-1,  0), dims);
    let mr = tap(xy + vec2<i32>( 1,  0), dims);
    let bl = tap(xy + vec2<i32>(-1,  1), dims);
    let bc = tap(xy + vec2<i32>( 0,  1), dims);
    let br = tap(xy + vec2<i32>( 1,  1), dims);

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
    let a = textureLoad(in0, xy, 0);

    let r = radians(params.angle_deg);
    let dir = vec2<f32>(cos(r), -sin(r));
    // Centred on the pixel: half the length each way, so the subject
    // smears symmetrically rather than sliding off its own position.
    let half = params.length_px * 0.5;
    var acc = vec4<f32>(0.0);
    for (var i : i32 = 0; i < TAPS; i = i + 1) {
        let t = (f32(i) / f32(TAPS - 1)) * 2.0 - 1.0;
        let p = vec2<f32>(f32(xy.x), f32(xy.y)) + dir * (t * half);
        let c = clamp(vec2<i32>(i32(round(p.x)), i32(round(p.y))),
                      vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
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
    let a = textureLoad(in0, xy, 0);

    let centre = vec2<f32>(params.cx * f32(dims.x), params.cy * f32(dims.y));
    let p = vec2<f32>(f32(xy.x), f32(xy.y));
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
                      vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
        acc = acc + textureLoad(in0, c, 0);
    }
    let result = acc / f32(TAPS);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

pub static FAMILY: &[&KernelDef] = &[
    &CONV_BOX,
    &CONV_GAUSSIAN_H,
    &CONV_GAUSSIAN_V,
    &CONV_UNSHARP,
    &CONV_EMBOSS,
    &CONV_FIND_EDGES,
    &CONV_MOTION,
    &CONV_RADIAL,
];
