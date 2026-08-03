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
//! coordinate math.
//!
//! KERNEL-BREADTH BATCH (2026-08): the remaining gradient shapes
//! (`gen.radial_gradient`, `gen.angular_gradient`,
//! `gen.reflected_gradient`, `gen.diamond_gradient`) and `gen.noise`.
//! The M2 "value-noise deferred" note is RESOLVED: `gen.noise` uses a
//! deterministic PCG integer hash written identically in WGSL and the
//! scalar twin (u32 wrapping arithmetic — bit-identical by
//! construction, no PRNG, no platform entropy). `gen.angular_gradient`
//! does NOT use the builtin `atan2` (its WGSL accuracy bound is too
//! loose near the seam); it carries its own polynomial `atan`
//! (Abramowitz & Stegun 4.4-class minimax over [0,1] + octant folding)
//! mirrored term-for-term by the reference, so both lanes agree to
//! ordinary f32 rounding noise.
//!
//! Provenance: procedural generation — standard analytic-geometry
//! parameterizations (projection / distance / angle / L1-distance
//! contours); PCG hash per Jarzynski & Olano, "Hash Functions for GPU
//! Rendering" (JCGT 2020, public literature). No reference reading.

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

// ──────────────────────── radial_gradient ──────────────────────────
//
// Pixel-space radial gradient. p = (gx, gy); center (cx, cy), radius
// `radius`. t = clamp(dist(p, center) / radius, 0, 1); out = mix(c0,
// c1, t) in PREMULTIPLIED space. radius <= 0 (degenerate) yields t = 0.
// sqrt + the divide are f32 on both lanes; f16 output quantization
// absorbs the last-ulp divergence (ChannelEpsF16(4), like linear).
// Param block padded to 64 bytes (16-aligned) like linear_gradient.

/// `gen.radial_gradient` params: tile origin, center, radius, and the
/// two PREMULTIPLIED endpoint colors (center → edge). 16-byte aligned.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GenRadialGradientParams {
    pub ox: i32,
    pub oy: i32,
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
    pub c0r: f32,
    pub c0g: f32,
    pub c0b: f32,
    pub c0a: f32,
    pub c1r: f32,
    pub c1g: f32,
    pub c1b: f32,
    pub c1a: f32,
    /// Explicit pads → 64-byte (16-aligned) uniform block.
    pub _pad0: u32,
    pub _pad1: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
#[allow(clippy::too_many_arguments)]
impl GenRadialGradientParams {
    pub fn new(
        ox: i32,
        oy: i32,
        cx: f32,
        cy: f32,
        radius: f32,
        c0: [f32; 4],
        c1: [f32; 4],
    ) -> Self {
        Self {
            ox,
            oy,
            cx,
            cy,
            radius,
            c0r: c0[0],
            c0g: c0[1],
            c0b: c0[2],
            c0a: c0[3],
            c1r: c1[0],
            c1g: c1[1],
            c1b: c1[2],
            c1a: c1[3],
            _pad0: 0,
            _pad1: 0,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// out = mix(c0, c1, t), t = clamp(|p − center| / radius, 0, 1),
/// p = (ox + x, oy + y). Premultiplied. ChannelEpsF16(4).
pub static GEN_RADIAL_GRADIENT: KernelDef = KernelDef {
    id: "gen.radial_gradient",
    class: KernelClass::Generator,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GenRadialGradientParams>(),
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
            ParamField {
                name: "_pad1",
                wgsl_ty: "u32",
            },
        ],
    },
    wgsl: GEN_RADIAL_GRADIENT_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

// dist via sqrt(dx² + dy²) in the FIXED order dx*dx + dy*dy; the scalar
// reference mirrors it. Degenerate guard: radius <= 0 → t = 0.
const GEN_RADIAL_GRADIENT_WGSL: &str = "\
// paged.image kernel `gen.radial_gradient` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    ox: i32,
    oy: i32,
    cx: f32,
    cy: f32,
    radius: f32,
    c0r: f32,
    c0g: f32,
    c0b: f32,
    c0a: f32,
    c1r: f32,
    c1g: f32,
    c1b: f32,
    c1a: f32,
    _pad0: u32,
    _pad1: u32,
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
    let dx = px - params.cx;
    let dy = py - params.cy;
    let d = sqrt(dx * dx + dy * dy);
    var t = 0.0;
    if (params.radius > 0.0) {
        t = clamp(d / params.radius, 0.0, 1.0);
    }
    let c0 = vec4<f32>(params.c0r, params.c0g, params.c0b, params.c0a);
    let c1 = vec4<f32>(params.c1r, params.c1g, params.c1b, params.c1a);
    let result = mix(c0, c1, vec4<f32>(t));
    textureStore(outp, xy, result);
}
";

// ──────────────────────── angular_gradient ─────────────────────────
//
// Conic sweep around a center. The delta (dx, dy) is rotated by
// `-angle` (ca = cos(angle), sa = sin(angle); rx = dx·ca + dy·sa,
// ry = dy·ca − dx·sa), then
//   t = clamp((atan2_det(ry, rx) + π) / 2π, 0, 1)
// so t sweeps 0→1 counter-clockwise from the rotated −x seam, and
// `angle` turns the whole sweep. atan2_det is the kernel's OWN
// deterministic atan2 (module doc): octant folding + a fixed odd
// polynomial on [0,1], mirrored term-for-term by the scalar reference —
// the builtin atan2's loose WGSL accuracy bound near the ±π seam would
// otherwise flip texels across the c0/c1 discontinuity. The seam is a
// genuine discontinuity of the shape; parity stimulus keeps it outside
// the tile (center outside, seam pointing away).

/// `gen.angular_gradient` params: tile origin, center, sweep rotation
/// `angle` (radians), and the two PREMULTIPLIED colors (sweep start →
/// end). 16-byte aligned.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GenAngularGradientParams {
    pub ox: i32,
    pub oy: i32,
    pub cx: f32,
    pub cy: f32,
    pub angle: f32,
    pub c0r: f32,
    pub c0g: f32,
    pub c0b: f32,
    pub c0a: f32,
    pub c1r: f32,
    pub c1g: f32,
    pub c1b: f32,
    pub c1a: f32,
    /// Explicit pads → 64-byte (16-aligned) uniform block.
    pub _pad0: u32,
    pub _pad1: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
#[allow(clippy::too_many_arguments)]
impl GenAngularGradientParams {
    pub fn new(ox: i32, oy: i32, cx: f32, cy: f32, angle: f32, c0: [f32; 4], c1: [f32; 4]) -> Self {
        Self {
            ox,
            oy,
            cx,
            cy,
            angle,
            c0r: c0[0],
            c0g: c0[1],
            c0b: c0[2],
            c0a: c0[3],
            c1r: c1[0],
            c1g: c1[1],
            c1b: c1[2],
            c1a: c1[3],
            _pad0: 0,
            _pad1: 0,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// out = mix(c0, c1, t), t = clamp((atan2_det(rot(p − c, −angle)) + π)
/// / 2π, 0, 1). Premultiplied. ChannelEpsF16(4).
pub static GEN_ANGULAR_GRADIENT: KernelDef = KernelDef {
    id: "gen.angular_gradient",
    class: KernelClass::Generator,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GenAngularGradientParams>(),
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
                name: "cx",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "cy",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "angle",
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
            ParamField {
                name: "_pad1",
                wgsl_ty: "u32",
            },
        ],
    },
    wgsl: GEN_ANGULAR_GRADIENT_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

// atan_poly: odd minimax-style polynomial for atan(z), z ∈ [0,1]
// (Abramowitz & Stegun 4.4-class coefficients); atan2_det: octant fold
// (ax ≥ ay → atan(ay/ax); else π/2 − atan(ax/ay)), then quadrant fix
// (x < 0 → π − base; y < 0 → negate). Zero-vector and axis cases hit
// the r = 0 guard → exact 0 / π. The scalar reference mirrors EVERY
// branch and coefficient.
const GEN_ANGULAR_GRADIENT_WGSL: &str = "\
// paged.image kernel `gen.angular_gradient` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    ox: i32,
    oy: i32,
    cx: f32,
    cy: f32,
    angle: f32,
    c0r: f32,
    c0g: f32,
    c0b: f32,
    c0a: f32,
    c1r: f32,
    c1g: f32,
    c1b: f32,
    c1a: f32,
    _pad0: u32,
    _pad1: u32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

fn atan_poly(z: f32) -> f32 {
    let z2 = z * z;
    return z * (0.9998660 + z2 * (-0.3302995 + z2 * (0.1801410 + z2 * (-0.0851330 + z2 * 0.0208351))));
}

fn atan2_det(y: f32, x: f32) -> f32 {
    let ax = abs(x);
    let ay = abs(y);
    var base: f32;
    if (ax >= ay) {
        var r = 0.0;
        if (ax > 0.0) { r = ay / ax; }
        base = atan_poly(r);
    } else {
        base = 1.5707963267948966 - atan_poly(ax / ay);
    }
    if (x < 0.0) { base = 3.141592653589793 - base; }
    if (y < 0.0) { base = -base; }
    return base;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let px = f32(params.ox + i32(gid.x));
    let py = f32(params.oy + i32(gid.y));
    let dx = px - params.cx;
    let dy = py - params.cy;
    let ca = cos(params.angle);
    let sa = sin(params.angle);
    let rx = dx * ca + dy * sa;
    let ry = dy * ca - dx * sa;
    let theta = atan2_det(ry, rx);
    let t = clamp((theta + 3.141592653589793) / 6.283185307179586, 0.0, 1.0);
    let c0 = vec4<f32>(params.c0r, params.c0g, params.c0b, params.c0a);
    let c1 = vec4<f32>(params.c1r, params.c1g, params.c1b, params.c1a);
    let result = mix(c0, c1, vec4<f32>(t));
    textureStore(outp, xy, result);
}
";

// ─────────────────────── reflected_gradient ────────────────────────
//
// The linear gradient mirrored about p0: SIGNED projection s =
// dot(p − p0, p1 − p0) / |p1 − p0|², then t = clamp(|s|, 0, 1) — c0 at
// the p0 line, c1 at distance |p1 − p0| on EITHER side. Same param
// block as gen.linear_gradient; |p1 − p0|² == 0 → t = 0.

/// `gen.reflected_gradient` params: identical shape to
/// `gen.linear_gradient` (origin, two endpoints, two PREMULTIPLIED
/// colors); the gradient mirrors about p0. 16-byte aligned.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GenReflectedGradientParams {
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
impl GenReflectedGradientParams {
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

/// out = mix(c0, c1, t), t = clamp(|dot(p−p0, p1−p0)| / |p1−p0|², 0, 1)
/// — mirrored about p0. Premultiplied. ChannelEpsF16(4).
pub static GEN_REFLECTED_GRADIENT: KernelDef = KernelDef {
    id: "gen.reflected_gradient",
    class: KernelClass::Generator,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GenReflectedGradientParams>(),
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
    wgsl: GEN_REFLECTED_GRADIENT_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GEN_REFLECTED_GRADIENT_WGSL: &str = "\
// paged.image kernel `gen.reflected_gradient` — handwritten under ABI v1.1.
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
        t = clamp(abs((dx * ex + dy * ey) / dd), 0.0, 1.0);
    }
    let c0 = vec4<f32>(params.c0r, params.c0g, params.c0b, params.c0a);
    let c1 = vec4<f32>(params.c1r, params.c1g, params.c1b, params.c1a);
    let result = mix(c0, c1, vec4<f32>(t));
    textureStore(outp, xy, result);
}
";

// ──────────────────────── diamond_gradient ─────────────────────────
//
// Diamond (L1-distance) contours from a center, rotated by `angle`:
// rotate the delta into the diamond frame (rx = dx·ca + dy·sa,
// ry = dy·ca − dx·sa), then t = clamp((|rx| + |ry|) / scale, 0, 1);
// out = mix(c0, c1, t). `scale` is the L1 radius of the c1 contour.
// scale <= 0 → t = 0. |rx| + |ry| is continuous, so lane rounding
// noise cannot flip across a discontinuity anywhere.

/// `gen.diamond_gradient` params: tile origin, center, rotation `angle`
/// (radians), L1 radius `scale`, and the two PREMULTIPLIED colors.
/// 16-byte aligned.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GenDiamondGradientParams {
    pub ox: i32,
    pub oy: i32,
    pub cx: f32,
    pub cy: f32,
    pub angle: f32,
    pub scale: f32,
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
impl GenDiamondGradientParams {
    pub fn new(
        ox: i32,
        oy: i32,
        cx: f32,
        cy: f32,
        angle: f32,
        scale: f32,
        c0: [f32; 4],
        c1: [f32; 4],
    ) -> Self {
        Self {
            ox,
            oy,
            cx,
            cy,
            angle,
            scale,
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

/// out = mix(c0, c1, t), t = clamp((|rx| + |ry|) / scale, 0, 1) with
/// (rx, ry) = the delta rotated by −angle. Premultiplied.
/// ChannelEpsF16(4).
pub static GEN_DIAMOND_GRADIENT: KernelDef = KernelDef {
    id: "gen.diamond_gradient",
    class: KernelClass::Generator,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GenDiamondGradientParams>(),
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
                name: "cx",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "cy",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "angle",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "scale",
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
    wgsl: GEN_DIAMOND_GRADIENT_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GEN_DIAMOND_GRADIENT_WGSL: &str = "\
// paged.image kernel `gen.diamond_gradient` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    ox: i32,
    oy: i32,
    cx: f32,
    cy: f32,
    angle: f32,
    scale: f32,
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
    let dx = px - params.cx;
    let dy = py - params.cy;
    let ca = cos(params.angle);
    let sa = sin(params.angle);
    let rx = dx * ca + dy * sa;
    let ry = dy * ca - dx * sa;
    var t = 0.0;
    if (params.scale > 0.0) {
        t = clamp((abs(rx) + abs(ry)) / params.scale, 0.0, 1.0);
    }
    let c0 = vec4<f32>(params.c0r, params.c0g, params.c0b, params.c0a);
    let c1 = vec4<f32>(params.c1r, params.c1g, params.c1b, params.c1a);
    let result = mix(c0, c1, vec4<f32>(t));
    textureStore(outp, xy, result);
}
";

// ─────────────────────────────── noise ─────────────────────────────
//
// Deterministic hash-based uniform noise. Per texel:
//   h = pcg(gx ^ pcg(gy ^ pcg(seed)))       (u32 wrapping arithmetic)
//   n = f32(h >> 8) / 2²⁴                   (exact — 24-bit mantissa)
//   v = n · amount
//   out = (v, v, v, 1)                      (opaque premultiplied gray)
// pcg is the PCG output-permutation hash (Jarzynski & Olano, JCGT 2020
// — public literature); u32 arithmetic wraps identically in WGSL and in
// the reference's `wrapping_*` ops, so both lanes are BIT-identical
// through `v` (the mul-by-amount is correctly rounded on both). Global
// coords enter as bitcast i32→u32 (two's complement, matches `as u32`).
// NO platform PRNG anywhere — same (seed, x, y) always gives the same
// texel.

/// `gen.noise` params: tile origin, hash `seed`, and `amount` (the
/// noise amplitude; texel value = hash·amount, hash uniform in [0,1)).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GenNoiseParams {
    pub ox: i32,
    pub oy: i32,
    pub seed: u32,
    pub amount: f32,
    /// Explicit pads → 32-byte (16-aligned) uniform block.
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl GenNoiseParams {
    pub fn new(ox: i32, oy: i32, seed: u32, amount: f32) -> Self {
        Self {
            ox,
            oy,
            seed,
            amount,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// out = (v, v, v, 1), v = pcg-hash(gx, gy, seed)·amount — deterministic
/// uniform noise, bit-identical hash on both lanes. ChannelEpsF16(1).
pub static GEN_NOISE: KernelDef = KernelDef {
    id: "gen.noise",
    class: KernelClass::Generator,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GenNoiseParams>(),
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
                name: "seed",
                wgsl_ty: "u32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "_pad0",
                wgsl_ty: "u32",
            },
            ParamField {
                name: "_pad1",
                wgsl_ty: "u32",
            },
            ParamField {
                name: "_pad2",
                wgsl_ty: "u32",
            },
        ],
    },
    wgsl: GEN_NOISE_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(1),
};

const GEN_NOISE_WGSL: &str = "\
// paged.image kernel `gen.noise` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    ox: i32,
    oy: i32,
    seed: u32,
    amount: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

// PCG output-permutation hash (Jarzynski & Olano, JCGT 2020). u32
// arithmetic wraps; mirrored by the reference's wrapping_* ops.
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let gx = bitcast<u32>(params.ox + i32(gid.x));
    let gy = bitcast<u32>(params.oy + i32(gid.y));
    let h = pcg(gx ^ pcg(gy ^ pcg(params.seed)));
    let n = f32(h >> 8u) * 5.9604644775390625e-8;
    let v = n * params.amount;
    let result = vec4<f32>(v, v, v, 1.0);
    textureStore(outp, xy, result);
}
";

pub static FAMILY: &[&KernelDef] = &[
    &GEN_SOLID,
    &GEN_CHECKER,
    &GEN_LINEAR_GRADIENT,
    &GEN_RADIAL_GRADIENT,
    &GEN_ANGULAR_GRADIENT,
    &GEN_REFLECTED_GRADIENT,
    &GEN_DIAMOND_GRADIENT,
    &GEN_NOISE,
];
