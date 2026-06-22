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

//! compose family (T1, spec §8.4 / §11) — the blend/composite operators
//! required by the PSD merged-composite oracle. Every kernel is a BINARY
//! POINT module under ABI v1.1 (`abi::assemble` docs): `in0` = `a` = the
//! backdrop (premultiplied), `in1` = `b` = the source layer
//! (premultiplied), params `{ opacity: f32 }`, output at the same dims.
//! The module composites source-over and applies `mix(a, result, m)`
//! like the generated point template; it runs through the existing
//! `execute_tile_once` / `parity()` lane.
//!
//! Provenance: W3C Compositing and Blending Level 1 (public spec,
//! <https://www.w3.org/TR/compositing-1/>). The 16 operators map 1:1 to
//! PSD layer blend-mode keys (see `image_conformance::compose_ref`),
//! which is why this family is the PSD flatten oracle's spine. The
//! scalar reference (the parity golden AND the flatten oracle) is the
//! handwritten `compose_ref` module; the WGSL below mirrors it exactly,
//! including the fixed source-over summation order (term1+term2+term3)
//! and the W3C §10.3 non-separable pseudo-code branch order.
//!
//! Each module is a complete WGSL source built by `separable_wgsl!` /
//! `nonsep_wgsl!` from a shared preamble (binding interface + unpremul
//! guard), a per-class helper block (separable per-channel fns, or the
//! §10.3 non-separable Lum/Sat helpers), the kernel's `blend(cb, cs)`
//! body, and the shared `main` (fold opacity → unpremul → blend →
//! source-over → mask-mix). The fragments are `concat!`-spliced at
//! compile time, so there is no external dependency and no runtime
//! assembly.

use crate::{KernelClass, KernelDef, ParamField, ParamsLayout, Tolerance};

/// Layer opacity for the composite (folded into the source first).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct ComposeParams {
    pub opacity: f32,
    pub _abi_pad: u32,
}

impl ComposeParams {
    pub fn new(opacity: f32) -> Self {
        Self {
            opacity,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// The shared `ParamsLayout` for every compose kernel — one `opacity`
/// f32 (the WGSL struct appends `_abi_pad` to mirror the Rust block).
const COMPOSE_PARAMS: ParamsLayout = ParamsLayout {
    size: ::core::mem::size_of::<ComposeParams>(),
    fields: &[ParamField {
        name: "opacity",
        wgsl_ty: "f32",
    }],
};

// The shared module regions are exposed as `macro_rules!` returning
// string LITERALS (not `const &str`), because `concat!` accepts only
// literal expressions. Each kernel's full WGSL is then assembled at
// compile time by `separable_wgsl!` / `nonsep_wgsl!` — one source of
// truth for the large shared preamble/main, no runtime work, no external
// crate. The fragments mirror `image_conformance::compose_ref` exactly.

/// File header + the v1.1 binding interface + the params struct (mirrors
/// `ComposeParams` including `_abi_pad`) + the unpremultiply guard
/// (mirrors `abi::unpremul4`: zero alpha → 0 rgb).
macro_rules! preamble_lit {
    () => {
        "// paged.image compose kernel — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    opacity: f32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(0) @binding(1) var in1 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

fn unpremul_rgb(c: vec4<f32>) -> vec3<f32> {
    if (c.a == 0.0) { return vec3<f32>(0.0); }
    return c.rgb / c.a;
}
"
    };
}

macro_rules! separable_lit {
    () => {
        "
fn s_multiply(cb: f32, cs: f32) -> f32 { return cb * cs; }
fn s_screen(cb: f32, cs: f32) -> f32 { return cb + cs - cb * cs; }
fn s_hard_light(cb: f32, cs: f32) -> f32 {
    if (cs <= 0.5) { return s_multiply(cb, 2.0 * cs); }
    return s_screen(cb, 2.0 * cs - 1.0);
}
fn s_color_dodge(cb: f32, cs: f32) -> f32 {
    if (cb == 0.0) { return 0.0; }
    if (cs == 1.0) { return 1.0; }
    return min(1.0, cb / (1.0 - cs));
}
fn s_color_burn(cb: f32, cs: f32) -> f32 {
    if (cb == 1.0) { return 1.0; }
    if (cs == 0.0) { return 0.0; }
    return 1.0 - min(1.0, (1.0 - cb) / cs);
}
fn s_soft_light(cb: f32, cs: f32) -> f32 {
    if (cs <= 0.5) { return cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb); }
    var d: f32;
    if (cb <= 0.25) { d = ((16.0 * cb - 12.0) * cb + 4.0) * cb; }
    else { d = sqrt(cb); }
    return cb + (2.0 * cs - 1.0) * (d - cb);
}
fn s_difference(cb: f32, cs: f32) -> f32 { return abs(cb - cs); }
fn s_exclusion(cb: f32, cs: f32) -> f32 { return cb + cs - 2.0 * cb * cs; }
"
    };
}

macro_rules! nonsep_lit {
    () => {
        "
fn lum(c: vec3<f32>) -> f32 { return 0.3 * c.r + 0.59 * c.g + 0.11 * c.b; }
fn clip_color(c0: vec3<f32>) -> vec3<f32> {
    var c = c0;
    let l = lum(c);
    let n = min(c.r, min(c.g, c.b));
    let x = max(c.r, max(c.g, c.b));
    if (n < 0.0) { c = vec3<f32>(l) + (c - vec3<f32>(l)) * l / (l - n); }
    if (x > 1.0) { c = vec3<f32>(l) + (c - vec3<f32>(l)) * (1.0 - l) / (x - l); }
    return c;
}
fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    let d = l - lum(c);
    return clip_color(c + vec3<f32>(d));
}
fn sat(c: vec3<f32>) -> f32 {
    return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
}
fn ge3(arr: array<f32, 3>, i: i32, j: i32) -> bool {
    return arr[i] > arr[j] || (arr[i] == arr[j] && i >= j);
}
fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    var arr = array<f32, 3>(c.r, c.g, c.b);
    // Total order with index tie-break -> DISTINCT min/mid/max indices
    // (mirrors compose_ref::set_sat exactly).
    var imax: i32;
    if (ge3(arr, 0, 1) && ge3(arr, 0, 2)) { imax = 0; }
    else if (ge3(arr, 1, 0) && ge3(arr, 1, 2)) { imax = 1; }
    else { imax = 2; }
    var imin: i32;
    if (ge3(arr, 1, 0) && ge3(arr, 2, 0)) { imin = 0; }
    else if (ge3(arr, 0, 1) && ge3(arr, 2, 1)) { imin = 1; }
    else { imin = 2; }
    let imid = 3 - imax - imin;
    let cmax = arr[imax];
    let cmin = arr[imin];
    var outv = array<f32, 3>(0.0, 0.0, 0.0);
    if (cmax > cmin) {
        outv[imid] = (arr[imid] - cmin) * s / (cmax - cmin);
        outv[imax] = s;
    } else {
        outv[imid] = 0.0;
        outv[imax] = 0.0;
    }
    outv[imin] = 0.0;
    return vec3<f32>(outv[0], outv[1], outv[2]);
}
"
    };
}

macro_rules! main_lit {
    () => {
        "
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let a = textureLoad(in0, xy, 0);
    let b_raw = textureLoad(in1, xy, 0);
    let bs = b_raw * params.opacity;
    let alpha_s = bs.a;
    let alpha_b = a.a;
    let cs = unpremul_rgb(bs);
    let cb = unpremul_rgb(a);
    let bc = blend(cb, cs);
    let alpha_o = alpha_s + alpha_b * (1.0 - alpha_s);
    let term1 = alpha_s * (1.0 - alpha_b) * cs;
    let term2 = alpha_s * alpha_b * bc;
    let term3 = (1.0 - alpha_s) * alpha_b * cb;
    let co = term1 + term2 + term3;
    let result = vec4<f32>(co, alpha_o);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
"
    };
}

/// Assemble a separable compose module (preamble + separable fns + blend
/// + main) at compile time.
macro_rules! separable_wgsl {
    ($blend:literal) => {
        concat!(preamble_lit!(), separable_lit!(), $blend, main_lit!())
    };
}

/// Assemble a non-separable compose module (preamble + §10.3 fns + blend
/// + main) at compile time.
macro_rules! nonsep_wgsl {
    ($blend:literal) => {
        concat!(preamble_lit!(), nonsep_lit!(), $blend, main_lit!())
    };
}

const NORMAL_WGSL: &str =
    separable_wgsl!("fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> { return cs; }\n");
const MULTIPLY_WGSL: &str = separable_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {\n    return vec3<f32>(s_multiply(cb.r, cs.r), s_multiply(cb.g, cs.g), s_multiply(cb.b, cs.b));\n}\n"
);
const SCREEN_WGSL: &str = separable_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {\n    return vec3<f32>(s_screen(cb.r, cs.r), s_screen(cb.g, cs.g), s_screen(cb.b, cs.b));\n}\n"
);
const OVERLAY_WGSL: &str = separable_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {\n    return vec3<f32>(s_hard_light(cs.r, cb.r), s_hard_light(cs.g, cb.g), s_hard_light(cs.b, cb.b));\n}\n"
);
const DARKEN_WGSL: &str = separable_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> { return min(cb, cs); }\n"
);
const LIGHTEN_WGSL: &str = separable_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> { return max(cb, cs); }\n"
);
const COLOR_DODGE_WGSL: &str = separable_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {\n    return vec3<f32>(s_color_dodge(cb.r, cs.r), s_color_dodge(cb.g, cs.g), s_color_dodge(cb.b, cs.b));\n}\n"
);
const COLOR_BURN_WGSL: &str = separable_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {\n    return vec3<f32>(s_color_burn(cb.r, cs.r), s_color_burn(cb.g, cs.g), s_color_burn(cb.b, cs.b));\n}\n"
);
const HARD_LIGHT_WGSL: &str = separable_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {\n    return vec3<f32>(s_hard_light(cb.r, cs.r), s_hard_light(cb.g, cs.g), s_hard_light(cb.b, cs.b));\n}\n"
);
const SOFT_LIGHT_WGSL: &str = separable_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {\n    return vec3<f32>(s_soft_light(cb.r, cs.r), s_soft_light(cb.g, cs.g), s_soft_light(cb.b, cs.b));\n}\n"
);
const DIFFERENCE_WGSL: &str = separable_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {\n    return vec3<f32>(s_difference(cb.r, cs.r), s_difference(cb.g, cs.g), s_difference(cb.b, cs.b));\n}\n"
);
const EXCLUSION_WGSL: &str = separable_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {\n    return vec3<f32>(s_exclusion(cb.r, cs.r), s_exclusion(cb.g, cs.g), s_exclusion(cb.b, cs.b));\n}\n"
);
const HUE_WGSL: &str = nonsep_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> { return set_lum(set_sat(cs, sat(cb)), lum(cb)); }\n"
);
const SATURATION_WGSL: &str = nonsep_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> { return set_lum(set_sat(cb, sat(cs)), lum(cb)); }\n"
);
const COLOR_WGSL: &str = nonsep_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> { return set_lum(cs, lum(cb)); }\n"
);
const LUMINOSITY_WGSL: &str = nonsep_wgsl!(
    "fn blend(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> { return set_lum(cb, lum(cs)); }\n"
);

/// Build a compose `KernelDef` with the shared params + binary point
/// module shape; only `id`, `wgsl`, and `tolerance` vary.
const fn compose_def(id: &'static str, wgsl: &'static str, eps: u32) -> KernelDef {
    KernelDef {
        id,
        class: KernelClass::Point,
        inputs: 2,
        params: COMPOSE_PARAMS,
        wgsl,
        module: true,
        mip_exact: true,
        gpu_tolerance: Tolerance::ChannelEpsF16(eps),
    }
}

pub static COMPOSE_NORMAL: KernelDef = compose_def("compose.normal", NORMAL_WGSL, 2);
pub static COMPOSE_MULTIPLY: KernelDef = compose_def("compose.multiply", MULTIPLY_WGSL, 4);
pub static COMPOSE_SCREEN: KernelDef = compose_def("compose.screen", SCREEN_WGSL, 4);
pub static COMPOSE_OVERLAY: KernelDef = compose_def("compose.overlay", OVERLAY_WGSL, 4);
pub static COMPOSE_DARKEN: KernelDef = compose_def("compose.darken", DARKEN_WGSL, 2);
pub static COMPOSE_LIGHTEN: KernelDef = compose_def("compose.lighten", LIGHTEN_WGSL, 2);
pub static COMPOSE_COLOR_DODGE: KernelDef = compose_def("compose.color_dodge", COLOR_DODGE_WGSL, 6);
pub static COMPOSE_COLOR_BURN: KernelDef = compose_def("compose.color_burn", COLOR_BURN_WGSL, 6);
pub static COMPOSE_HARD_LIGHT: KernelDef = compose_def("compose.hard_light", HARD_LIGHT_WGSL, 4);
pub static COMPOSE_SOFT_LIGHT: KernelDef = compose_def("compose.soft_light", SOFT_LIGHT_WGSL, 6);
pub static COMPOSE_DIFFERENCE: KernelDef = compose_def("compose.difference", DIFFERENCE_WGSL, 2);
pub static COMPOSE_EXCLUSION: KernelDef = compose_def("compose.exclusion", EXCLUSION_WGSL, 4);
pub static COMPOSE_HUE: KernelDef = compose_def("compose.hue", HUE_WGSL, 6);
pub static COMPOSE_SATURATION: KernelDef = compose_def("compose.saturation", SATURATION_WGSL, 6);
pub static COMPOSE_COLOR: KernelDef = compose_def("compose.color", COLOR_WGSL, 6);
pub static COMPOSE_LUMINOSITY: KernelDef = compose_def("compose.luminosity", LUMINOSITY_WGSL, 6);

pub static FAMILY: &[&KernelDef] = &[
    &COMPOSE_NORMAL,
    &COMPOSE_MULTIPLY,
    &COMPOSE_SCREEN,
    &COMPOSE_OVERLAY,
    &COMPOSE_DARKEN,
    &COMPOSE_LIGHTEN,
    &COMPOSE_COLOR_DODGE,
    &COMPOSE_COLOR_BURN,
    &COMPOSE_HARD_LIGHT,
    &COMPOSE_SOFT_LIGHT,
    &COMPOSE_DIFFERENCE,
    &COMPOSE_EXCLUSION,
    &COMPOSE_HUE,
    &COMPOSE_SATURATION,
    &COMPOSE_COLOR,
    &COMPOSE_LUMINOSITY,
];
