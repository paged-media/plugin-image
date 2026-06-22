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

//! Generator family (T2, spec §11) — `gen.solid`, `gen.checker`,
//! `gen.linear_gradient`. Procedural pixel sources with NO meaningful
//! input: output depends only on the texel's GLOBAL coordinate and the
//! params, not on any sampled tile.
//!
//! M2 ZERO-INPUT CONVENTION (documented). ABI v1 (`abi::assemble` /
//! `KernelPipeline::build` / `execute_tile_once`) assumes `inputs >= 1`
//! and always binds `in0`. Rather than amend the frozen ABI for a
//! dedicated zero-input lane, generators ship as `module: true` UNARY
//! kernels (`inputs: 1`) that DECLARE the v1 `in0`/mask bindings per the
//! contract but NEVER sample them: the output value is a pure function
//! of `gid` + params. The caller passes a dummy `in0` tile (the point
//! lane already uploads one); its contents are ignored by the shader.
//! A dedicated zero-input ABI lane is a clean-up follow-up — once a
//! `KernelClass::Generator` aware `execute_generator_once` exists, the
//! dummy `in0` binding can be dropped. (FOLLOW-UP: zero-input lane.)
//!
//! TILE CONTINUITY. The tile's GLOBAL origin `(ox, oy)` travels in the
//! params; the shader forms the global coordinate `gx = ox + gid.x`,
//! `gy = oy + gid.y`, so gradients/checker are continuous across tile
//! boundaries (each tile is rendered with its own origin). Generators
//! are therefore NOT `mip_exact`: the geometry is coordinate-absolute
//! (a pixel grid / pixel-space gradient direction), so a mip level must
//! re-derive params for its own resolution rather than reuse the level-0
//! block (§8.3) — recorded as `mip_exact: false` in the registry.
//!
//! These are HANDWRITTEN modules (DSL can't express gid-derived coords),
//! so each carries a HANDWRITTEN scalar reference twin in the test file
//! (`image-conformance/tests/family_gen.rs`) that mirrors the same
//! coordinate math. value-noise is DEFERRED past M2: a cross-language
//! deterministic hash/PRNG (bit-identical WGSL ≡ Rust) is the risky
//! part, and is out of scope for this lane (FOLLOW-UP: gen.noise).
//!
//! Provenance: procedural generation — trivial; no reference reading.

use crate::{KernelClass, KernelDef, ParamField, ParamsLayout, Tolerance};

// ─────────────────────────────── solid ─────────────────────────────
//
// Constant premultiplied color. `(ox, oy)` are carried for ABI
// uniformity (and so every generator's param head is identical), though
// solid's output is coordinate-independent.

/// `gen.solid` params: tile origin (carried, unused by the math) + the
/// constant PREMULTIPLIED rgba color written to every texel.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GenSolidParams {
    pub ox: i32,
    pub oy: i32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
#[allow(clippy::too_many_arguments)]
impl GenSolidParams {
    pub fn new(ox: i32, oy: i32, r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            ox,
            oy,
            r,
            g,
            b,
            a,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// out = (r, g, b, a) at every texel (premultiplied color, written
/// verbatim). Exact: no arithmetic, just the param copy.
pub static GEN_SOLID: KernelDef = KernelDef {
    id: "gen.solid",
    class: KernelClass::Generator,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GenSolidParams>(),
        fields: &[
            ParamField {
                name: "ox",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "oy",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "r",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "g",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "b",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "a",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: GEN_SOLID_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(1),
};

const GEN_SOLID_WGSL: &str = "\
// paged.image kernel `gen.solid` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    ox: i32,
    oy: i32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
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
    // Generator: in0/mask are bound per the ABI but never sampled.
    let result = vec4<f32>(params.r, params.g, params.b, params.a);
    textureStore(outp, xy, result);
}
";

// ────────────────────────────── checker ────────────────────────────
//
// Two-color checkerboard at GLOBAL coords. cell = ((gx/size + gy/size)
// & 1); cell 0 → c0, cell 1 → c1. Integer division floors toward zero
// for non-negative coords; the test stimulus keeps gx, gy >= 0 (origins
// are non-negative), matching the engine's tile grid.

/// `gen.checker` params: tile origin, cell `size` (pixels per square),
/// and the two PREMULTIPLIED rgba colors.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GenCheckerParams {
    pub ox: i32,
    pub oy: i32,
    pub size: u32,
    pub c0r: f32,
    pub c0g: f32,
    pub c0b: f32,
    pub c0a: f32,
    pub c1r: f32,
    pub c1g: f32,
    pub c1b: f32,
    pub c1a: f32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
#[allow(clippy::too_many_arguments)]
impl GenCheckerParams {
    pub fn new(ox: i32, oy: i32, size: u32, c0: [f32; 4], c1: [f32; 4]) -> Self {
        Self {
            ox,
            oy,
            size,
            c0r: c0[0],
            c0g: c0[1],
            c0b: c0[2],
            c0a: c0[3],
            c1r: c1[0],
            c1g: c1[1],
            c1b: c1[2],
            c1a: c1[3],
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// out = ((gx/size + gy/size) & 1) ? c1 : c0, gx = ox + x, gy = oy + y.
/// Exact: an integer parity selecting one of two literal colors.
pub static GEN_CHECKER: KernelDef = KernelDef {
    id: "gen.checker",
    class: KernelClass::Generator,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GenCheckerParams>(),
        fields: &[
            ParamField {
                name: "ox",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "oy",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "size",
                wgsl_ty: "u32",
            },
            ParamField {
                name: "c0r",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c0g",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c0b",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c0a",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c1r",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c1g",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c1b",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c1a",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: GEN_CHECKER_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(1),
};

// gx, gy >= 0 (non-negative origins): u32 cell math matches the scalar
// reference's i32 floor-division exactly. cell parity = (cx + cy) & 1.
const GEN_CHECKER_WGSL: &str = "\
// paged.image kernel `gen.checker` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    ox: i32,
    oy: i32,
    size: u32,
    c0r: f32,
    c0g: f32,
    c0b: f32,
    c0a: f32,
    c1r: f32,
    c1g: f32,
    c1b: f32,
    c1a: f32,
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
    // Global coordinate (non-negative): tile origin + local gid.
    let gx = u32(params.ox + i32(gid.x));
    let gy = u32(params.oy + i32(gid.y));
    let cx = gx / params.size;
    let cy = gy / params.size;
    let cell = (cx + cy) & 1u;
    let c0 = vec4<f32>(params.c0r, params.c0g, params.c0b, params.c0a);
    let c1 = vec4<f32>(params.c1r, params.c1g, params.c1b, params.c1a);
    let result = select(c0, c1, cell == 1u);
    textureStore(outp, xy, result);
}
";

// ──────────────────────── linear_gradient ──────────────────────────
//
// Pixel-space linear gradient. p = (gx, gy); endpoints p0 = (x0, y0),
// p1 = (x1, y1). t = clamp(dot(p - p0, p1 - p0) / |p1 - p0|², 0, 1);
// out = mix(c0, c1, t) in PREMULTIPLIED space (c0/c1 are premultiplied
// rgba, like solid). |p1 - p0|² == 0 (degenerate endpoints) yields t = 0
// (the WGSL guard mirrors the reference). Tolerance ChannelEpsF16(4):
// the dot/normalize divide is f32 on both lanes, the f16 output
// quantization absorbs the last-ulp divergence.
//
// PARAM LAYOUT (16-byte aligned, per the unit instruction): two i32
// origin + four f32 endpoints + 4 f32 c0 + 4 f32 c1 = 14 scalars (56
// bytes); one explicit `_pad0` + the trailing `_abi_pad` round the block
// to 64 bytes (a multiple of 16). Both pads are listed in the WGSL
// struct; only `_abi_pad` is the macro-style tail, `_pad0` is explicit.

/// `gen.linear_gradient` params: tile origin, the two endpoints in pixel
/// space, and the two PREMULTIPLIED endpoint colors. 16-byte aligned.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GenLinearGradientParams {
    pub ox: i32,
    pub oy: i32,
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub c0r: f32,
    pub c0g: f32,
    pub c0b: f32,
    pub c0a: f32,
    pub c1r: f32,
    pub c1g: f32,
    pub c1b: f32,
    pub c1a: f32,
    /// Explicit pad → 64-byte (16-aligned) uniform block.
    pub _pad0: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
#[allow(clippy::too_many_arguments)]
impl GenLinearGradientParams {
    pub fn new(
        ox: i32,
        oy: i32,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        c0: [f32; 4],
        c1: [f32; 4],
    ) -> Self {
        Self {
            ox,
            oy,
            x0,
            y0,
            x1,
            y1,
            c0r: c0[0],
            c0g: c0[1],
            c0b: c0[2],
            c0a: c0[3],
            c1r: c1[0],
            c1g: c1[1],
            c1b: c1[2],
            c1a: c1[3],
            _pad0: 0,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// out = mix(c0, c1, t), t = clamp(dot(p-p0, p1-p0) / |p1-p0|², 0, 1),
/// p = (ox + x, oy + y). Premultiplied. ChannelEpsF16(4).
pub static GEN_LINEAR_GRADIENT: KernelDef = KernelDef {
    id: "gen.linear_gradient",
    class: KernelClass::Generator,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GenLinearGradientParams>(),
        fields: &[
            ParamField {
                name: "ox",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "oy",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "x0",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "y0",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "x1",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "y1",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c0r",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c0g",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c0b",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c0a",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c1r",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c1g",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c1b",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "c1a",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "_pad0",
                wgsl_ty: "u32",
            },
        ],
    },
    wgsl: GEN_LINEAR_GRADIENT_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

// Degenerate guard: dd == 0 → t = 0. Reduction is a single fused dot /
// divide / clamp / mix; the scalar reference mirrors it term-for-term.
const GEN_LINEAR_GRADIENT_WGSL: &str = "\
// paged.image kernel `gen.linear_gradient` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    ox: i32,
    oy: i32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    c0r: f32,
    c0g: f32,
    c0b: f32,
    c0a: f32,
    c1r: f32,
    c1g: f32,
    c1b: f32,
    c1a: f32,
    _pad0: u32,
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
    let px = f32(params.ox + i32(gid.x));
    let py = f32(params.oy + i32(gid.y));
    let dx = px - params.x0;
    let dy = py - params.y0;
    let ex = params.x1 - params.x0;
    let ey = params.y1 - params.y0;
    let dd = ex * ex + ey * ey;
    var t = 0.0;
    if (dd > 0.0) {
        t = clamp((dx * ex + dy * ey) / dd, 0.0, 1.0);
    }
    let c0 = vec4<f32>(params.c0r, params.c0g, params.c0b, params.c0a);
    let c1 = vec4<f32>(params.c1r, params.c1g, params.c1b, params.c1a);
    let result = mix(c0, c1, vec4<f32>(t));
    textureStore(outp, xy, result);
}
";

pub static FAMILY: &[&KernelDef] = &[&GEN_SOLID, &GEN_CHECKER, &GEN_LINEAR_GRADIENT];
