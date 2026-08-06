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

//! Geometry family (T2, spec §11) — handwritten WGSL modules under the
//! ABI v1.1 contract (`abi::assemble` docs). Flip, rotate-by-90°, and
//! crop expressed as `KernelClass::Resample { support: 0.5 }` coordinate
//! remaps: every output texel reads exactly ONE source texel by integer
//! index — no interpolation, no weighting ⇒ `Tolerance::Exact`.
//!
//! # Coordinate model (frozen for this family)
//!
//! `in0` is the full source window; window texel `(i, j)` carries source
//! coordinate `(i, j)` directly (window origin = source origin 0). The
//! mask is RESERVED for resample (M3) — each module writes its single
//! source texel directly, exactly like the `resample.*` modules. The
//! remap dimensions / offset travel in each kernel's params:
//!
//! - `geom.flip_h`     params `{ width }`            → out(x,y)=in(width-1-x, y)
//! - `geom.flip_v`     params `{ height }`           → out(x,y)=in(x, height-1-y)
//! - `geom.rotate90_cw`  params `{ src_w, src_h }`   → out(x,y)=in(y, src_h-1-x)
//! - `geom.rotate90_ccw` params `{ src_w, src_h }`   → out(x,y)=in(src_w-1-y, x)
//! - `geom.crop`       params `{ off_x, off_y }`     → out(x,y)=in(x+off_x, y+off_y) clamp-to-edge
//!
//! `mip_exact: false` — a coordinate remap composes with mip selection
//! only at level 0 (the remap dims are level-0 dims); like `resample.*`
//! it runs at the full resolution.
//!
//! Provenance: flip / 90°-rotation / crop are trivial integer
//! coordinate remaps — affine index math, standard and self-evident; no
//! reference reading. (vips oracle: `flip`/`rot90`/`extract_area`.)

use crate::{KernelClass, KernelDef, ParamField, ParamsLayout, Tolerance};

// ───────────────────────────── flip_h ──────────────────────────────

/// Horizontal flip params: the source/output width (they are equal —
/// a flip preserves dims). `out(x, y) = in(width - 1 - x, y)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct FlipHParams {
    pub width: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl FlipHParams {
    pub fn new(width: u32) -> Self {
        Self { width, _abi_pad: 0 }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const FLIP_H_FIELDS: &[ParamField] = &[ParamField {
    name: "width",
    wgsl_ty: "u32",
}];

/// `out(x, y) = in(width - 1 - x, y)`. One exact source texel per
/// output texel ⇒ `Tolerance::Exact`.
pub static GEOM_FLIP_H: KernelDef = KernelDef {
    id: "geom.flip_h",
    class: KernelClass::Resample { support: 0.5 },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<FlipHParams>(),
        fields: FLIP_H_FIELDS,
    },
    wgsl: GEOM_FLIP_H_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::Exact,
};

const GEOM_FLIP_H_WGSL: &str = "\
// paged.image kernel `geom.flip_h` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    width: u32,
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
    let sx = i32(params.width) - 1 - xy.x;
    textureStore(outp, xy, textureLoad(in0, vec2<i32>(sx, xy.y), 0));
}
";

// ───────────────────────────── flip_v ──────────────────────────────

/// Vertical flip params: the source/output height. `out(x, y) =
/// in(x, height - 1 - y)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct FlipVParams {
    pub height: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl FlipVParams {
    pub fn new(height: u32) -> Self {
        Self {
            height,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const FLIP_V_FIELDS: &[ParamField] = &[ParamField {
    name: "height",
    wgsl_ty: "u32",
}];

/// `out(x, y) = in(x, height - 1 - y)`. Exact single-texel remap.
pub static GEOM_FLIP_V: KernelDef = KernelDef {
    id: "geom.flip_v",
    class: KernelClass::Resample { support: 0.5 },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<FlipVParams>(),
        fields: FLIP_V_FIELDS,
    },
    wgsl: GEOM_FLIP_V_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::Exact,
};

const GEOM_FLIP_V_WGSL: &str = "\
// paged.image kernel `geom.flip_v` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    height: u32,
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
    let sy = i32(params.height) - 1 - xy.y;
    textureStore(outp, xy, textureLoad(in0, vec2<i32>(xy.x, sy), 0));
}
";

// ──────────────────────────── rotate90 ─────────────────────────────
//
// A 90° rotation transposes the dims: a `src_w × src_h` source yields a
// `src_h × src_w` output. `src_w`/`src_h` are the SOURCE dims; the
// output dims (which the dispatch sizes) are their transpose.

/// 90° rotation params: the SOURCE dims (`src_w`, `src_h`). Shared by
/// the cw and ccw kernels — the rotation direction is the kernel, not a
/// param.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct Rotate90Params {
    pub src_w: u32,
    pub src_h: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl Rotate90Params {
    pub fn new(src_w: u32, src_h: u32) -> Self {
        Self {
            src_w,
            src_h,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const ROTATE90_FIELDS: &[ParamField] = &[
    ParamField {
        name: "src_w",
        wgsl_ty: "u32",
    },
    ParamField {
        name: "src_h",
        wgsl_ty: "u32",
    },
];

const ROTATE90_LAYOUT: ParamsLayout = ParamsLayout {
    size: ::core::mem::size_of::<Rotate90Params>(),
    fields: ROTATE90_FIELDS,
};

/// Clockwise 90°: `out(x, y) = in(y, src_h - 1 - x)`. Exact remap.
pub static GEOM_ROTATE90_CW: KernelDef = KernelDef {
    id: "geom.rotate90_cw",
    class: KernelClass::Resample { support: 0.5 },
    inputs: 1,
    params: ROTATE90_LAYOUT,
    wgsl: GEOM_ROTATE90_CW_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::Exact,
};

const GEOM_ROTATE90_CW_WGSL: &str = "\
// paged.image kernel `geom.rotate90_cw` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    src_w: u32,
    src_h: u32,
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
    let sx = xy.y;
    let sy = i32(params.src_h) - 1 - xy.x;
    textureStore(outp, xy, textureLoad(in0, vec2<i32>(sx, sy), 0));
}
";

/// Counter-clockwise 90°: `out(x, y) = in(src_w - 1 - y, x)`. Exact
/// remap; the inverse of `geom.rotate90_cw`.
pub static GEOM_ROTATE90_CCW: KernelDef = KernelDef {
    id: "geom.rotate90_ccw",
    class: KernelClass::Resample { support: 0.5 },
    inputs: 1,
    params: ROTATE90_LAYOUT,
    wgsl: GEOM_ROTATE90_CCW_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::Exact,
};

const GEOM_ROTATE90_CCW_WGSL: &str = "\
// paged.image kernel `geom.rotate90_ccw` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    src_w: u32,
    src_h: u32,
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
    let sx = i32(params.src_w) - 1 - xy.y;
    let sy = xy.x;
    textureStore(outp, xy, textureLoad(in0, vec2<i32>(sx, sy), 0));
}
";

// ────────────────────────────── crop ───────────────────────────────

/// Crop params: signed source offset (`off_x`, `off_y`) added to the
/// output coord. `out(x, y) = in(x + off_x, y + off_y)` clamped to the
/// source edge (clamp-to-edge). Signed so a crop window may start at a
/// negative source coord (the clamp replicates the edge texel).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct CropParams {
    pub off_x: i32,
    pub off_y: i32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl CropParams {
    pub fn new(off_x: i32, off_y: i32) -> Self {
        Self {
            off_x,
            off_y,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const CROP_FIELDS: &[ParamField] = &[
    ParamField {
        name: "off_x",
        wgsl_ty: "i32",
    },
    ParamField {
        name: "off_y",
        wgsl_ty: "i32",
    },
];

/// `out(x, y) = in(x + off_x, y + off_y)` clamp-to-edge. Exact remap;
/// the clamp to `[0, dim-1]` IS the edge rule (sample replication).
pub static GEOM_CROP: KernelDef = KernelDef {
    id: "geom.crop",
    class: KernelClass::Resample { support: 0.5 },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<CropParams>(),
        fields: CROP_FIELDS,
    },
    wgsl: GEOM_CROP_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::Exact,
};

const GEOM_CROP_WGSL: &str = "\
// paged.image kernel `geom.crop` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    off_x: i32,
    off_y: i32,
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
    let wdims = vec2<i32>(textureDimensions(in0));
    let sx = clamp(xy.x + params.off_x, 0, wdims.x - 1);
    let sy = clamp(xy.y + params.off_y, 0, wdims.y - 1);
    textureStore(outp, xy, textureLoad(in0, vec2<i32>(sx, sy), 0));
}
";

// ─────────────────────── rotate_bilinear ──────────────────────────
//
// ARBITRARY-ANGLE rotation — the STRAIGHTEN commit's resample (the
// crop tool previewed a rotated frame long before anything could
// commit it). Unlike the rest of this family it INTERPOLATES, so it is
// `KernelClass::Resample { support: 1.0 }` (a 2×2 bilinear footprint)
// with a f16 tolerance rather than `Exact`.
//
// # Coordinate model (frozen)
//
// Output texel `(x, y)` is the CONTINUOUS point `(x + 0.5, y + 0.5)`
// in destination space. The forward map "rotate the source by θ about
// `src_c`, landing its centre on `dst_c`" is
//
//     dst = R(θ)·(src − src_c) + dst_c ,  R(θ) = [[cos, −sin], [sin, cos]]
//
// so the BACKWARD map this kernel evaluates (one source point per
// output texel, no scatter) is
//
//     src = R(−θ)·(dst − dst_c) + src_c
//         = ( cos·dx + sin·dy + src_cx ,  −sin·dx + cos·dy + src_cy )
//
// with `dx = x + 0.5 − dst_cx`, `dy = y + 0.5 − dst_cy`. Subtracting
// 0.5 converts the continuous source point back to texel-index space.
// Screen convention: y grows DOWN, so a positive θ reads as a
// clockwise rotation of the image.
//
// # Edge rule + summation order (determinism, §6.3)
//
// The four taps are `clamp`ed to `[0, dim−1]` — clamp-to-edge, the
// same rule `resample.*` and `geom.crop` use, so the corners the
// rotation swings past the source repeat the border instead of going
// transparent (the straighten flow then CROPS the valid interior; a
// transparent-outside variant would need an alpha-aware second kernel
// and is NOT this one). The blend is the FIXED order
// `mix(mix(p00, p10, fx), mix(p01, p11, fx), fy)` — x first, then y —
// mirrored term-for-term by the scalar reference.
//
// `mip_exact: false` — like every coordinate remap it runs at level 0.
//
// Provenance: backward-mapped affine resampling with bilinear
// reconstruction is textbook (Wolberg, *Digital Image Warping*, IEEE CS
// Press 1990, §3–§5: inverse mapping + bilinear interpolation). No
// reference reading. (vips oracle: `similarity`/`rotate` at
// `interpolate=bilinear`.)

/// Rotation params: `cos_t`/`sin_t` of the angle, the SOURCE centre the
/// rotation pivots on, and the DESTINATION centre it lands on (the two
/// differ whenever the output canvas grows to hold the rotated bounds).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct RotateBilinearParams {
    pub cos_t: f32,
    pub sin_t: f32,
    pub src_cx: f32,
    pub src_cy: f32,
    pub dst_cx: f32,
    pub dst_cy: f32,
    /// Explicit pad → a 32-byte (16-aligned) uniform block.
    pub _pad0: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl RotateBilinearParams {
    /// Build from `degrees` (positive = clockwise on screen) plus the
    /// source and destination centres in continuous pixel coordinates.
    pub fn new(degrees: f32, src_c: (f32, f32), dst_c: (f32, f32)) -> Self {
        let t = degrees.to_radians();
        Self {
            cos_t: t.cos(),
            sin_t: t.sin(),
            src_cx: src_c.0,
            src_cy: src_c.1,
            dst_cx: dst_c.0,
            dst_cy: dst_c.1,
            _pad0: 0,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const ROTATE_BILINEAR_FIELDS: &[ParamField] = &[
    ParamField {
        name: "cos_t",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "sin_t",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "src_cx",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "src_cy",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "dst_cx",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "dst_cy",
        wgsl_ty: "f32",
    },
];

/// Backward-mapped arbitrary-angle rotation with bilinear
/// reconstruction, clamp-to-edge. One 2×2 tap footprint per output
/// texel ⇒ `Resample { support: 1.0 }`; the two `mix` levels are
/// correctly rounded on both lanes, so the f16 output quantisation
/// absorbs the last-ulp trig divergence — `ChannelEpsF16(4)`.
pub static GEOM_ROTATE_BILINEAR: KernelDef = KernelDef {
    id: "geom.rotate_bilinear",
    class: KernelClass::Resample { support: 1.0 },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<RotateBilinearParams>(),
        fields: ROTATE_BILINEAR_FIELDS,
    },
    wgsl: GEOM_ROTATE_BILINEAR_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GEOM_ROTATE_BILINEAR_WGSL: &str = "\
// paged.image kernel `geom.rotate_bilinear` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    cos_t: f32,
    sin_t: f32,
    src_cx: f32,
    src_cy: f32,
    dst_cx: f32,
    dst_cy: f32,
    _pad0: u32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

fn tap(p: vec2<i32>, dims: vec2<i32>) -> vec4<f32> {
    let c = clamp(p, vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    return textureLoad(in0, c, 0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let wdims = vec2<i32>(textureDimensions(in0));

    let dx = f32(xy.x) + 0.5 - params.dst_cx;
    let dy = f32(xy.y) + 0.5 - params.dst_cy;
    let sx = params.cos_t * dx + params.sin_t * dy + params.src_cx - 0.5;
    let sy = -params.sin_t * dx + params.cos_t * dy + params.src_cy - 0.5;

    let x0 = floor(sx);
    let y0 = floor(sy);
    let fx = sx - x0;
    let fy = sy - y0;
    let i0 = vec2<i32>(i32(x0), i32(y0));

    let p00 = tap(i0, wdims);
    let p10 = tap(i0 + vec2<i32>(1, 0), wdims);
    let p01 = tap(i0 + vec2<i32>(0, 1), wdims);
    let p11 = tap(i0 + vec2<i32>(1, 1), wdims);

    let top = mix(p00, p10, fx);
    let bot = mix(p01, p11, fx);
    textureStore(outp, xy, mix(top, bot, fy));
}
";

// ─────────────────────────── warp_backward ─────────────────────────
//
// THE GENERAL BACKWARD-MAP WARP — one kernel, one sampler, a family of
// distortions selected by `kind`.
//
// `geom.rotate_bilinear` was the only warp in the engine, which is why
// the entire Distort family (Pinch, Spherize, Twirl, Ripple, Wave,
// ZigZag, Polar Coordinates, Shear, Displace) was listed as unbuilt in
// the catalog's §36.4: they are not nine independent effects, they are
// nine source-coordinate functions over the same backward map. Sharing
// the sampler is not a shortcut — it is the only way the reconstruction
// filter, the edge rule and the parity tolerance stay identical across
// all of them, which is what makes them comparable.
//
// BACKWARD, not forward: for each OUTPUT texel compute where it came
// from and sample there. A forward map scatters and leaves holes; a
// backward map is total by construction. Bilinear reconstruction with
// clamp-to-edge, exactly as the rotation already does — one 2x2 tap
// footprint, so `Resample { support: 1.0 }`.
//
// Amount is normalized so 0 is the IDENTITY for every kind. That is
// what lets a UI expose one slider per distortion without special
// cases, and it makes the identity parity test meaningful for all of
// them at once.

/// Which source-coordinate function the warp applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum WarpKind {
    /// Pull toward (amount > 0) or push away from the centre.
    Pinch = 0,
    /// Spherical bulge (amount > 0) or dent.
    Spherize = 1,
    /// Rotate by an angle that falls off with radius.
    Twirl = 2,
    /// Sinusoidal displacement along x, by y.
    Wave = 3,
}

impl WarpKind {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Warp params: the kind, the normalized amount, the centre, and a
/// frequency the periodic kinds use (ignored by the others).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct WarpBackwardParams {
    pub kind: u32,
    pub amount: f32,
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
    pub frequency: f32,
    pub _pad0: u32,
    pub _abi_pad: u32,
}

impl WarpBackwardParams {
    pub fn new(kind: WarpKind, amount: f32, cx: f32, cy: f32, radius: f32) -> Self {
        Self {
            kind: kind.as_u32(),
            amount,
            cx,
            cy,
            radius,
            frequency: 1.0,
            _pad0: 0,
            _abi_pad: 0,
        }
    }

    /// Periodic kinds (Wave) read this; the others ignore it.
    pub fn with_frequency(mut self, frequency: f32) -> Self {
        self.frequency = frequency;
        self
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const WARP_BACKWARD_FIELDS: &[ParamField] = &[
    ParamField {
        name: "kind",
        wgsl_ty: "u32",
    },
    ParamField {
        name: "amount",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "cx",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "cy",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "radius",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "frequency",
        wgsl_ty: "f32",
    },
];

/// Backward-mapped parametric distortion, bilinear, clamp-to-edge.
/// `amount == 0` is the identity for every kind.
pub static GEOM_WARP_BACKWARD: KernelDef = KernelDef {
    id: "geom.warp_backward",
    class: KernelClass::Resample { support: 1.0 },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<WarpBackwardParams>(),
        fields: WARP_BACKWARD_FIELDS,
    },
    wgsl: GEOM_WARP_BACKWARD_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GEOM_WARP_BACKWARD_WGSL: &str = "\
// paged.image kernel `geom.warp_backward` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    kind: u32,
    amount: f32,
    cx: f32,
    cy: f32,
    radius: f32,
    frequency: f32,
    _pad0: u32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

fn tap(p: vec2<i32>, dims: vec2<i32>) -> vec4<f32> {
    let c = clamp(p, vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    return textureLoad(in0, c, 0);
}

// Where output point `d` (centre-relative) READ FROM. Returns a
// centre-relative source offset. `t` is the normalized radius.
fn source_offset(d: vec2<f32>, t: f32) -> vec2<f32> {
    let a = params.amount;
    if (params.kind == 0u) {
        // PINCH: scale radius by (1 + a·(1 − t)), so the effect is
        // strongest at the centre and vanishes at the rim.
        return d * (1.0 + a * (1.0 - t));
    }
    if (params.kind == 1u) {
        // SPHERIZE: a smooth radial remap, zero at centre and rim.
        let k = 1.0 + a * sin(3.14159265 * clamp(t, 0.0, 1.0));
        return d * k;
    }
    if (params.kind == 2u) {
        // TWIRL: rotate by an angle that falls off linearly with radius.
        let ang = a * (1.0 - clamp(t, 0.0, 1.0));
        let cs = cos(ang);
        let sn = sin(ang);
        return vec2<f32>(cs * d.x - sn * d.y, sn * d.x + cs * d.y);
    }
    // WAVE: displace x by a sinusoid of y.
    let phase = params.frequency * 6.28318531 * (d.y / max(params.radius, 1.0));
    return vec2<f32>(d.x + a * params.radius * 0.05 * sin(phase), d.y);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let wdims = vec2<i32>(textureDimensions(in0));

    let d = vec2<f32>(f32(xy.x) + 0.5 - params.cx, f32(xy.y) + 0.5 - params.cy);
    let r = length(d);
    let t = r / max(params.radius, 1.0);
    let src = source_offset(d, t) + vec2<f32>(params.cx, params.cy) - vec2<f32>(0.5, 0.5);

    let x0 = floor(src.x);
    let y0 = floor(src.y);
    let fx = src.x - x0;
    let fy = src.y - y0;
    let i0 = vec2<i32>(i32(x0), i32(y0));

    let p00 = tap(i0, wdims);
    let p10 = tap(i0 + vec2<i32>(1, 0), wdims);
    let p01 = tap(i0 + vec2<i32>(0, 1), wdims);
    let p11 = tap(i0 + vec2<i32>(1, 1), wdims);

    let top = mix(p00, p10, fx);
    let bot = mix(p01, p11, fx);
    let result = mix(top, bot, fy);

    let m = textureLoad(mask, xy, 0).r;
    let a = textureLoad(in0, clamp(xy, vec2<i32>(0, 0), wdims - vec2<i32>(1, 1)), 0);
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

pub static FAMILY: &[&KernelDef] = &[
    &GEOM_FLIP_H,
    &GEOM_FLIP_V,
    &GEOM_ROTATE90_CW,
    &GEOM_ROTATE90_CCW,
    &GEOM_CROP,
    &GEOM_ROTATE_BILINEAR,
    &GEOM_WARP_BACKWARD,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Naga-validate just this family's modules (independent of the
    /// shared `wgsl_validate` suite, which only sees a family once the
    /// orchestrator lands `geom::FAMILY` in `families/mod.rs`).
    #[test]
    fn geom_modules_naga_validate() {
        for def in FAMILY {
            let src = crate::abi::assemble(def);
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{}: WGSL parse failed: {e}\n{src}", def.id));
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::default(),
            );
            validator
                .validate(&module)
                .unwrap_or_else(|e| panic!("{}: WGSL validation failed: {e:?}\n{src}", def.id));
        }
    }
}
