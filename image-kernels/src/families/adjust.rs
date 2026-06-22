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

//! adjust family (T2, spec §11 T2) — the editor-bearing tone/color point
//! kernels. Every kernel operates on UNpremultiplied rgb (the per-color
//! math is meaningless on premultiplied samples) and PRESERVES alpha; the
//! input/output working space is premultiplied, so each kernel unpremuls
//! `a`, does its math, and re-premultiplies by the original alpha (except
//! `adjust.exposure`, whose scalar gain commutes with premultiplication —
//! `(rgb·α)·k = (rgb·k)·α` — so it scales the premultiplied rgb directly).
//!
//! `adjust.invert_rgb` fits the restricted DSL (`kernel_family!`): its
//! body is pure `unpremul4`/`premul4`/`mix`/`pack4` algebra (the alpha is
//! held by a `pack4(0,0,0,1)` mix mask). The other five need a
//! transcendental, a matrix, or multi-statement unpremul→math→re-premul
//! that the single-expression DSL cannot express cleanly, so they are
//! handwritten point modules under the ABI v1.1 module contract
//! (`abi::assemble` docs): exposure (exp2), brightness_contrast,
//! levels (pow), saturation, hue_rotate (cos/sin matrix). exp2/pow/cos/sin
//! are WGSL builtins mirrored EXACTLY by their `f32::*` Rust twins (the
//! scalar references live in `image-conformance/tests/family_adjust.rs`);
//! the last-ulp f32 divergence of these transcendentals is absorbed by the
//! f16 output quantization (per-kernel tolerances below).
//!
//! Provenance: standard image-adjustment literature — exposure stop = ev
//! powers of two; brightness/contrast = the classic pivot-at-0.5 affine;
//! levels = input/gamma/output remap (Photoshop Levels math); saturation
//! = luma-toward-color interpolation with the Lum weights of W3C
//! Compositing §10.3 (mirrored by `image_conformance::compose_ref`);
//! hue_rotate = the W3C Filter Effects `feColorMatrix type="hueRotate"`
//! luminance-preserving rotation matrix
//! (<https://www.w3.org/TR/filter-effects-1/#feColorMatrixElement>);
//! invert_rgb = photometric per-color negate (1 − c) leaving alpha. No
//! reference reading.

use crate::{KernelClass, KernelDef, ParamField, ParamsLayout, Tolerance};

// ───────────────────────────── invert_rgb ──────────────────────────
//
// out = re-premultiply( (1 − c.rgb, c.a) ) where c = unpremul(a). The
// single-expression form: `mix(splat4(1) - u, u, pack4(0,0,0,1))` keeps
// the inverted rgb (`splat4(1) - u`) but restores the unpremultiplied
// alpha (`u.a`) via the per-channel mix mask, then `premul4` re-folds
// alpha. Distinct from `math.invert` (which negates ALL four channels,
// alpha included) — this leaves alpha untouched. Fits the DSL.

kernel_family! {
    /// out = premul( (1 − unpremul(a).rgb, unpremul(a).a) ) — per-color
    /// negate; alpha preserved (cf. `math.invert`, which negates alpha).
    static ADJUST_INVERT_RGB, params AdjustInvertRgbParams, ref adjust_invert_rgb {
        id: "adjust.invert_rgb",
        class: KernelClass::Point,
        inputs: 1,
        params: {},
        eval: |a, b, p| premul4(mix(
            splat4(1.0) - unpremul4(a),
            unpremul4(a),
            pack4(0.0, 0.0, 0.0, 1.0)
        )),
        mip_exact: true,
        tolerance: Tolerance::ChannelEpsF16(1),
    }
}

// ─────────── shared module preamble (binding interface + guard) ─────
//
// Every handwritten adjust module is a unary point kernel: the v1.1
// binding interface (in0 / params / mask / outp), an `unpremul_rgb`
// guard mirroring `abi::unpremul4` (zero alpha → 0 rgb), the kernel's
// own body, and a shared `main` that reads `a`, computes `result`, and
// applies the ABI mask `mix(a, result, m)`. The fragments are
// `concat!`-spliced at compile time — no runtime assembly, no external
// dependency (same pattern as the compose family).

macro_rules! adjust_main_lit {
    () => {
        "
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let a = textureLoad(in0, xy, 0);
    let result = adjust(a);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
"
    };
}

/// File header + the v1.1 unary binding interface + the per-module
/// `Params` struct (mirrors the Rust block INCLUDING `_abi_pad`) + the
/// `unpremul_rgb` guard. `$params` is the struct body text.
macro_rules! adjust_wgsl {
    ($params:literal, $body:literal) => {
        concat!(
            "// paged.image adjust kernel — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

",
            $params,
            "
@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

fn unpremul_rgb(c: vec4<f32>) -> vec3<f32> {
    if (c.a == 0.0) { return vec3<f32>(0.0); }
    return c.rgb / c.a;
}
",
            $body,
            adjust_main_lit!()
        )
    };
}

// ───────────────────────────── exposure ────────────────────────────
//
// Stops of exposure: a scalar gain k = exp2(ev) applied to LINEAR light.
// Because gain commutes with premultiplication ((rgb·α)·k = (rgb·k)·α),
// the module scales the PREMULTIPLIED rgb directly and leaves alpha — no
// unpremul/re-premul round-trip needed. exp2 is a WGSL builtin; the
// scalar reference mirrors `f32::exp2` exactly.

/// Exposure params: `ev` stops (powers of two; exp2(ev) is the gain).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustExposureParams {
    pub ev: f32,
    pub _abi_pad: u32,
}

impl AdjustExposureParams {
    pub fn new(ev: f32) -> Self {
        Self { ev, _abi_pad: 0 }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const EXPOSURE_PARAMS_FIELDS: &[ParamField] = &[ParamField {
    name: "ev",
    wgsl_ty: "f32",
}];

/// out.rgb = a.rgb · exp2(ev); out.a = a.a (scales premultiplied rgb).
pub static ADJUST_EXPOSURE: KernelDef = KernelDef {
    id: "adjust.exposure",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustExposureParams>(),
        fields: EXPOSURE_PARAMS_FIELDS,
    },
    wgsl: EXPOSURE_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const EXPOSURE_WGSL: &str = adjust_wgsl!(
    "struct Params {
    ev: f32,
    _abi_pad: u32,
}",
    "
fn adjust(a: vec4<f32>) -> vec4<f32> {
    let k = exp2(params.ev);
    return vec4<f32>(a.rgb * k, a.a);
}
"
);

// ──────────────────────── brightness_contrast ──────────────────────
//
// On UNpremultiplied rgb, pivot contrast at 0.5 then add brightness:
//   c' = (c − 0.5)·contrast + 0.5 + brightness
// per channel, then re-premultiply by the original alpha. The classic
// brightness/contrast affine (contrast 1, brightness 0 = identity).

/// Brightness/contrast params (identity at brightness 0, contrast 1).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustBrightnessContrastParams {
    pub brightness: f32,
    pub contrast: f32,
    pub _abi_pad: u32,
}

impl AdjustBrightnessContrastParams {
    pub fn new(brightness: f32, contrast: f32) -> Self {
        Self {
            brightness,
            contrast,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const BRIGHTNESS_CONTRAST_PARAMS_FIELDS: &[ParamField] = &[
    ParamField {
        name: "brightness",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "contrast",
        wgsl_ty: "f32",
    },
];

/// c' = (c − 0.5)·contrast + 0.5 + brightness on unpremult rgb;
/// re-premultiplied, alpha preserved.
pub static ADJUST_BRIGHTNESS_CONTRAST: KernelDef = KernelDef {
    id: "adjust.brightness_contrast",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustBrightnessContrastParams>(),
        fields: BRIGHTNESS_CONTRAST_PARAMS_FIELDS,
    },
    wgsl: BRIGHTNESS_CONTRAST_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const BRIGHTNESS_CONTRAST_WGSL: &str = adjust_wgsl!(
    "struct Params {
    brightness: f32,
    contrast: f32,
    _abi_pad: u32,
}",
    "
fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = unpremul_rgb(a);
    let cp = (c - vec3<f32>(0.5)) * params.contrast + vec3<f32>(0.5 + params.brightness);
    return vec4<f32>(cp * a.a, a.a);
}
"
);

// ───────────────────────────── levels ──────────────────────────────
//
// Photoshop-style input/gamma/output remap on UNpremultiplied rgb:
//   t  = clamp((c − in_black) / (in_white − in_black), 0, 1)
//   t  = pow(t, 1 / gamma)
//   c' = out_black + t·(out_white − out_black)
// per channel, then re-premultiply by the original alpha. pow is a WGSL
// builtin (mirrored by `f32::powf`); tolerance allows for the
// transcendental last-ulp (ChannelEpsF16(6)). Identity =
// {0,1,1,0,1}.

/// Levels params: input black/white, midtone gamma, output black/white.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustLevelsParams {
    pub in_black: f32,
    pub in_white: f32,
    pub gamma: f32,
    pub out_black: f32,
    pub out_white: f32,
    pub _abi_pad: u32,
}

impl AdjustLevelsParams {
    pub fn new(in_black: f32, in_white: f32, gamma: f32, out_black: f32, out_white: f32) -> Self {
        Self {
            in_black,
            in_white,
            gamma,
            out_black,
            out_white,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const LEVELS_PARAMS_FIELDS: &[ParamField] = &[
    ParamField {
        name: "in_black",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "in_white",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "gamma",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "out_black",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "out_white",
        wgsl_ty: "f32",
    },
];

/// Levels remap (input→gamma→output) on unpremult rgb; re-premultiplied,
/// alpha preserved.
pub static ADJUST_LEVELS: KernelDef = KernelDef {
    id: "adjust.levels",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustLevelsParams>(),
        fields: LEVELS_PARAMS_FIELDS,
    },
    wgsl: LEVELS_WGSL,
    module: true,
    mip_exact: true,
    // GPU `pow()` is an approximation (Metal: exp2(y·log2(x))), and the
    // levels remap composes it with an unpremultiply that amplifies f16
    // input noise by 1/α (up to 4× at α=0.25). Measured worst 14 ULP on
    // Metal over a low-α gradient; 16 carries ~15% headroom (§6.3,
    // core's threshold-sizing rule). The D-10 hardware-matrix job
    // watches for drivers that exceed it.
    gpu_tolerance: Tolerance::ChannelEpsF16(16),
};

// Per-channel: normalize against the input window, gamma, scale into the
// output window. The vector `pow` applies componentwise. Reduction order
// is irrelevant (no cross-channel terms); WGSL `pow`/Rust `powf` mirror.
const LEVELS_WGSL: &str = adjust_wgsl!(
    "struct Params {
    in_black: f32,
    in_white: f32,
    gamma: f32,
    out_black: f32,
    out_white: f32,
    _abi_pad: u32,
}",
    "
fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = unpremul_rgb(a);
    let t0 = (c - vec3<f32>(params.in_black)) / vec3<f32>(params.in_white - params.in_black);
    let t1 = clamp(t0, vec3<f32>(0.0), vec3<f32>(1.0));
    let t2 = pow(t1, vec3<f32>(1.0 / params.gamma));
    let cp = vec3<f32>(params.out_black) + t2 * vec3<f32>(params.out_white - params.out_black);
    return vec4<f32>(cp * a.a, a.a);
}
"
);

// ───────────────────────────── saturation ──────────────────────────
//
// On UNpremultiplied rgb: lum = 0.3r + 0.59g + 0.11b (the W3C §10.3 Lum
// weights, mirrored by compose_ref), then interpolate each channel
// toward/away from gray: c' = mix(splat(lum), c, sat). sat 1 = identity,
// 0 = full desaturate (gray), >1 oversaturates. Re-premultiplied, alpha
// preserved.

/// Saturation params: `sat` (1 = identity, 0 = grayscale).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustSaturationParams {
    pub sat: f32,
    pub _abi_pad: u32,
}

impl AdjustSaturationParams {
    pub fn new(sat: f32) -> Self {
        Self { sat, _abi_pad: 0 }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const SATURATION_PARAMS_FIELDS: &[ParamField] = &[ParamField {
    name: "sat",
    wgsl_ty: "f32",
}];

/// c' = mix(splat(lum), c, sat) on unpremult rgb, lum = 0.3r+0.59g+0.11b;
/// re-premultiplied, alpha preserved.
pub static ADJUST_SATURATION: KernelDef = KernelDef {
    id: "adjust.saturation",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustSaturationParams>(),
        fields: SATURATION_PARAMS_FIELDS,
    },
    wgsl: SATURATION_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

// lum dot product in the fixed order r,g,b (matches the scalar
// reference's left-to-right f32 summation); mix is the WGSL builtin.
const SATURATION_WGSL: &str = adjust_wgsl!(
    "struct Params {
    sat: f32,
    _abi_pad: u32,
}",
    "
fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = unpremul_rgb(a);
    let lum = 0.3 * c.r + 0.59 * c.g + 0.11 * c.b;
    let cp = mix(vec3<f32>(lum), c, vec3<f32>(params.sat));
    return vec4<f32>(cp * a.a, a.a);
}
"
);

// ───────────────────────────── hue_rotate ──────────────────────────
//
// The W3C Filter Effects `feColorMatrix type="hueRotate"` matrix — a
// luminance-preserving rotation of the rgb vector about the gray axis by
// `degrees`. With θ = degrees·π/180, c = cos θ, s = sin θ, the per-row
// matrix is the documented constant + cos·M_cos + sin·M_sin (luma
// weights 0.213, 0.715, 0.072):
//
//   r' = (0.213 + c·0.787 + s·(−0.213))·r
//      + (0.715 + c·(−0.715) + s·(−0.715))·g
//      + (0.072 + c·(−0.072) + s·0.928)·b
//   g' = (0.213 + c·(−0.213) + s·0.143)·r
//      + (0.715 + c·0.285 + s·0.140)·g
//      + (0.072 + c·(−0.072) + s·(−0.283))·b
//   b' = (0.213 + c·(−0.213) + s·(−0.787))·r
//      + (0.715 + c·(−0.715) + s·0.715)·g
//      + (0.072 + c·0.928 + s·0.072)·b
//
// on UNpremultiplied rgb (each output channel summed in r,g,b order),
// re-premultiplied, alpha preserved. degrees 0 = identity (each row
// reduces to the luma weights summing to 1, which is the gray-preserving
// identity only on the luma axis; the off-luma identity holds because at
// θ=0 the matrix is exactly I — verified by the identity test). cos/sin
// are WGSL builtins mirrored by `f32::cos`/`f32::sin`; tolerance allows
// the transcendental last-ulp (ChannelEpsF16(6)).

/// Hue-rotate params: rotation angle in `degrees`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustHueRotateParams {
    pub degrees: f32,
    pub _abi_pad: u32,
}

impl AdjustHueRotateParams {
    pub fn new(degrees: f32) -> Self {
        Self {
            degrees,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const HUE_ROTATE_PARAMS_FIELDS: &[ParamField] = &[ParamField {
    name: "degrees",
    wgsl_ty: "f32",
}];

/// W3C luminance-preserving hue rotation on unpremult rgb;
/// re-premultiplied, alpha preserved.
pub static ADJUST_HUE_ROTATE: KernelDef = KernelDef {
    id: "adjust.hue_rotate",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustHueRotateParams>(),
        fields: HUE_ROTATE_PARAMS_FIELDS,
    },
    wgsl: HUE_ROTATE_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(6),
};

// θ = degrees·π/180; each output channel is a fixed-order r,g,b dot of
// the per-row coefficients (constant + cos·M_cos + sin·M_sin). The
// scalar reference computes the identical coefficient expressions and
// summation order.
const HUE_ROTATE_WGSL: &str = adjust_wgsl!(
    "struct Params {
    degrees: f32,
    _abi_pad: u32,
}",
    "
fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = unpremul_rgb(a);
    let theta = params.degrees * 3.14159265358979323846 / 180.0;
    let cs = cos(theta);
    let sn = sin(theta);
    let rr = (0.213 + cs * 0.787 + sn * (-0.213)) * c.r
           + (0.715 + cs * (-0.715) + sn * (-0.715)) * c.g
           + (0.072 + cs * (-0.072) + sn * 0.928) * c.b;
    let gg = (0.213 + cs * (-0.213) + sn * 0.143) * c.r
           + (0.715 + cs * 0.285 + sn * 0.140) * c.g
           + (0.072 + cs * (-0.072) + sn * (-0.283)) * c.b;
    let bb = (0.213 + cs * (-0.213) + sn * (-0.787)) * c.r
           + (0.715 + cs * (-0.715) + sn * 0.715) * c.g
           + (0.072 + cs * 0.928 + sn * 0.072) * c.b;
    return vec4<f32>(vec3<f32>(rr, gg, bb) * a.a, a.a);
}
"
);

// ─────────────────────────── white_balance ─────────────────────────
//
// Temperature/tint white balance as per-channel von-Kries-style gains on
// UNpremultiplied rgb. `temp` warms (+R, −B) along the amber↔blue axis;
// `tint` shifts the green↔magenta axis (+G). The per-channel gains:
//   gr = 1 + temp,  gg = 1 + tint,  gb = 1 − temp
// so c' = (gr·r, gg·g, gb·b), then re-premultiply by the original alpha.
// Identity = {0, 0} (all gains 1). A gray-point eyedropper in the panel
// resolves to a (temp, tint) that neutralizes the picked pixel; this
// kernel only consumes the resolved gains (the pick math is panel-side).
// Pure multiply — no transcendental, so the tolerance is the unpremul
// f16-amplification floor (ChannelEpsF16(4), the family default).

/// White-balance params: `temp` (amber↔blue) and `tint` (green↔magenta);
/// both 0 = identity.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustWhiteBalanceParams {
    pub temp: f32,
    pub tint: f32,
    pub _abi_pad: u32,
}

impl AdjustWhiteBalanceParams {
    pub fn new(temp: f32, tint: f32) -> Self {
        Self {
            temp,
            tint,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const WHITE_BALANCE_PARAMS_FIELDS: &[ParamField] = &[
    ParamField {
        name: "temp",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "tint",
        wgsl_ty: "f32",
    },
];

/// Per-channel WB gains (1+temp, 1+tint, 1−temp) on unpremult rgb;
/// re-premultiplied, alpha preserved.
pub static ADJUST_WHITE_BALANCE: KernelDef = KernelDef {
    id: "adjust.white_balance",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustWhiteBalanceParams>(),
        fields: WHITE_BALANCE_PARAMS_FIELDS,
    },
    wgsl: WHITE_BALANCE_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

// Per-channel scalar gain; componentwise multiply. No cross-channel
// terms, so reduction order is irrelevant and WGSL/Rust mirror exactly.
const WHITE_BALANCE_WGSL: &str = adjust_wgsl!(
    "struct Params {
    temp: f32,
    tint: f32,
    _abi_pad: u32,
}",
    "
fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = unpremul_rgb(a);
    let gain = vec3<f32>(1.0 + params.temp, 1.0 + params.tint, 1.0 - params.temp);
    return vec4<f32>(c * gain * a.a, a.a);
}
"
);

pub static FAMILY: &[&KernelDef] = &[
    &ADJUST_EXPOSURE,
    &ADJUST_BRIGHTNESS_CONTRAST,
    &ADJUST_LEVELS,
    &ADJUST_SATURATION,
    &ADJUST_HUE_ROTATE,
    &ADJUST_INVERT_RGB,
    &ADJUST_WHITE_BALANCE,
];
