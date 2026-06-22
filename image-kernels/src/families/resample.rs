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

//! Resample family (T1, spec §11) — handwritten WGSL modules under the
//! ABI v1.1 contract (`abi::assemble` docs). Three separable resamplers:
//! `resample.nearest`, `resample.mitchell` (B=C=1/3), `resample.lanczos3`.
//!
//! # Coordinate model (frozen for this family)
//!
//! `in0` is the full source window; window texel `(i, j)` carries source
//! coordinate `(i, j)` directly (window origin = source origin 0). The
//! mapping travels in `ResampleParams`: output texel `(x, y)` samples the
//! continuous source coordinate
//!
//! ```text
//! sx = (x + 0.5) * inv_scale_x - 0.5 + src_off_x   (same for sy)
//! ```
//!
//! (`inv_scale = 1/scale`: >1 downscales, <1 upscales; identity is
//! `inv_scale = 1`, `off = 0` ⇒ `sx = x` ⇒ near-passthrough.) The 2-D
//! filter is the separable product `w(sx - i) * w(sy - j)`. Taps span the
//! integer window `[floor(s) - support + 1, floor(s) + support]` (the
//! kernel's half-extent), **clamped to `[0, dim - 1]`** — that clamp IS
//! the edge rule (clamp-to-edge / sample replication). Weights are
//! normalised by their accumulated sum (mandatory at edges where the
//! support window is truncated, and for mitchell/lanczos whose taps do
//! not analytically sum to 1 at arbitrary phase).
//!
//! # Summation order (determinism, §6.3)
//!
//! `j` (source row) outer ascending, `i` (source column) inner
//! ascending. Both `sum` (weighted colour) and `wsum` (weight total)
//! accumulate in that single fused order; the scalar reference mirrors
//! it byte-for-byte. The Mitchell polynomial is exact arithmetic;
//! lanczos uses `sin()` — its last-ulp f32 divergence between GPU and
//! reference is absorbed by the f16 output quantisation (channel_eps_f16
//! tolerance 4).

use crate::{KernelClass, KernelDef, ParamField, ParamsLayout, Tolerance};

/// Resample mapping: continuous source coord
/// `s = (out + 0.5) * inv_scale - 0.5 + off` per axis. Shared by all
/// three resamplers; the kernel choice is the kernel, not a param.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct ResampleParams {
    pub inv_scale_x: f32,
    pub inv_scale_y: f32,
    pub src_off_x: f32,
    pub src_off_y: f32,
    pub _abi_pad: u32,
}

impl ResampleParams {
    pub fn new(inv_scale_x: f32, inv_scale_y: f32, src_off_x: f32, src_off_y: f32) -> Self {
        Self {
            inv_scale_x,
            inv_scale_y,
            src_off_x,
            src_off_y,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// The four named mapping fields (the WGSL struct mirrors them + the
/// trailing `_abi_pad: u32` per the ABI module contract).
const RESAMPLE_FIELDS: &[ParamField] = &[
    ParamField {
        name: "inv_scale_x",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "inv_scale_y",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "src_off_x",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "src_off_y",
        wgsl_ty: "f32",
    },
];

const RESAMPLE_LAYOUT: ParamsLayout = ParamsLayout {
    size: ::core::mem::size_of::<ResampleParams>(),
    fields: RESAMPLE_FIELDS,
};

/// Nearest-neighbour: support 0.5, one tap (the rounded source texel),
/// no interpolation ⇒ exact (tolerance 0). `mip_exact: false` — resample
/// IS scaling, so it must run at level 0.
pub static RESAMPLE_NEAREST: KernelDef = KernelDef {
    id: "resample.nearest",
    class: KernelClass::Resample { support: 0.5 },
    inputs: 1,
    params: RESAMPLE_LAYOUT,
    wgsl: RESAMPLE_NEAREST_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::Exact,
};

/// Mitchell–Netravali (B = C = 1/3), support 2.0.
pub static RESAMPLE_MITCHELL: KernelDef = KernelDef {
    id: "resample.mitchell",
    class: KernelClass::Resample { support: 2.0 },
    inputs: 1,
    params: RESAMPLE_LAYOUT,
    wgsl: RESAMPLE_MITCHELL_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

/// Lanczos-3 (sinc·sinc windowed), support 3.0.
pub static RESAMPLE_LANCZOS3: KernelDef = KernelDef {
    id: "resample.lanczos3",
    class: KernelClass::Resample { support: 3.0 },
    inputs: 1,
    params: RESAMPLE_LAYOUT,
    wgsl: RESAMPLE_LANCZOS3_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

// ---------------------------------------------------------------------
// WGSL modules. Each declares the exact ABI v1 binding interface
// (in0 @group0, params @group1 with the four mapping fields + _abi_pad,
// mask @group2 reserved, outp rgba16float @group3), @workgroup_size
// (16,16,1), and the dims guard. The source-window clamp is the edge
// rule; weights are sum-normalised. Mask is reserved for resample (M3) —
// the module writes `result` directly.
// ---------------------------------------------------------------------

/// Nearest: `round(s)` = `floor(s + 0.5)`, clamped to the window. The
/// summation/normalisation scaffold collapses to a single unit-weight
/// tap, so the result is one exact texel fetch (tolerance 0).
const RESAMPLE_NEAREST_WGSL: &str = "\
// paged.image kernel `resample.nearest` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    inv_scale_x: f32,
    inv_scale_y: f32,
    src_off_x: f32,
    src_off_y: f32,
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

    let sx = (f32(xy.x) + 0.5) * params.inv_scale_x - 0.5 + params.src_off_x;
    let sy = (f32(xy.y) + 0.5) * params.inv_scale_y - 0.5 + params.src_off_y;
    let ix = clamp(i32(floor(sx + 0.5)), 0, wdims.x - 1);
    let iy = clamp(i32(floor(sy + 0.5)), 0, wdims.y - 1);

    textureStore(outp, xy, textureLoad(in0, vec2<i32>(ix, iy), 0));
}
";

/// Mitchell–Netravali weight, B = C = 1/3 (1988). For `x = |t|`:
///   x < 1 : ((12-9B-6C)x³ + (-18+12B+6C)x² + (6-2B)) / 6
///   x < 2 : ((-B-6C)x³ + (6B+30C)x² + (-12B-48C)x + (8B+24C)) / 6
///   else  : 0
const RESAMPLE_MITCHELL_WGSL: &str = "\
// paged.image kernel `resample.mitchell` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    inv_scale_x: f32,
    inv_scale_y: f32,
    src_off_x: f32,
    src_off_y: f32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

fn mitchell(t: f32) -> f32 {
    let b = 1.0 / 3.0;
    let c = 1.0 / 3.0;
    let x = abs(t);
    if (x < 1.0) {
        return ((12.0 - 9.0 * b - 6.0 * c) * x * x * x
              + (-18.0 + 12.0 * b + 6.0 * c) * x * x
              + (6.0 - 2.0 * b)) / 6.0;
    }
    if (x < 2.0) {
        return ((-b - 6.0 * c) * x * x * x
              + (6.0 * b + 30.0 * c) * x * x
              + (-12.0 * b - 48.0 * c) * x
              + (8.0 * b + 24.0 * c)) / 6.0;
    }
    return 0.0;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let wdims = vec2<i32>(textureDimensions(in0));

    let sx = (f32(xy.x) + 0.5) * params.inv_scale_x - 0.5 + params.src_off_x;
    let sy = (f32(xy.y) + 0.5) * params.inv_scale_y - 0.5 + params.src_off_y;
    // Support 2.0: taps span [floor(s)-1 .. floor(s)+2].
    let bx = i32(floor(sx)) - 1;
    let by = i32(floor(sy)) - 1;

    var sum = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var dj = 0; dj < 4; dj = dj + 1) {
        let j = by + dj;
        let wy = mitchell(sy - f32(j));
        let cj = clamp(j, 0, wdims.y - 1);
        for (var di = 0; di < 4; di = di + 1) {
            let i = bx + di;
            let wx = mitchell(sx - f32(i));
            let ci = clamp(i, 0, wdims.x - 1);
            let w = wx * wy;
            sum = sum + textureLoad(in0, vec2<i32>(ci, cj), 0) * w;
            wsum = wsum + w;
        }
    }
    textureStore(outp, xy, sum / wsum);
}
";

/// Lanczos-3 weight: `sinc(t) * sinc(t/3)` for `|t| < 3`, else 0, with
/// `sinc(u) = sin(pi u) / (pi u)` and `sinc(0) = 1`.
const RESAMPLE_LANCZOS3_WGSL: &str = "\
// paged.image kernel `resample.lanczos3` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    inv_scale_x: f32,
    inv_scale_y: f32,
    src_off_x: f32,
    src_off_y: f32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

const PI : f32 = 3.14159265358979323846;

fn sinc(u: f32) -> f32 {
    if (u == 0.0) { return 1.0; }
    let pu = PI * u;
    return sin(pu) / pu;
}

fn lanczos3(t: f32) -> f32 {
    let x = abs(t);
    if (x < 3.0) {
        return sinc(t) * sinc(t / 3.0);
    }
    return 0.0;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let wdims = vec2<i32>(textureDimensions(in0));

    let sx = (f32(xy.x) + 0.5) * params.inv_scale_x - 0.5 + params.src_off_x;
    let sy = (f32(xy.y) + 0.5) * params.inv_scale_y - 0.5 + params.src_off_y;
    // Support 3.0: taps span [floor(s)-2 .. floor(s)+3].
    let bx = i32(floor(sx)) - 2;
    let by = i32(floor(sy)) - 2;

    var sum = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var dj = 0; dj < 6; dj = dj + 1) {
        let j = by + dj;
        let wy = lanczos3(sy - f32(j));
        let cj = clamp(j, 0, wdims.y - 1);
        for (var di = 0; di < 6; di = di + 1) {
            let i = bx + di;
            let wx = lanczos3(sx - f32(i));
            let ci = clamp(i, 0, wdims.x - 1);
            let w = wx * wy;
            sum = sum + textureLoad(in0, vec2<i32>(ci, cj), 0) * w;
            wsum = wsum + w;
        }
    }
    textureStore(outp, xy, sum / wsum);
}
";

pub static FAMILY: &[&KernelDef] = &[&RESAMPLE_NEAREST, &RESAMPLE_MITCHELL, &RESAMPLE_LANCZOS3];

#[cfg(test)]
mod tests {
    use super::*;

    /// Naga-validate just this family's modules (independent of the
    /// shared `wgsl_validate` suite, which stops at the first failing
    /// kernel across all in-flight families).
    #[test]
    fn resample_modules_naga_validate() {
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
