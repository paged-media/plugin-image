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

pub static FAMILY: &[&KernelDef] = &[
    &GEOM_FLIP_H,
    &GEOM_FLIP_V,
    &GEOM_ROTATE90_CW,
    &GEOM_ROTATE90_CCW,
    &GEOM_CROP,
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
