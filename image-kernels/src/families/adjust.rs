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

// ═══════════════ kernel-breadth batch (2026-08) ═════════════════════
//
// Eight further editor-bearing tone/color point kernels, same module
// contract as above (unpremul → math → re-premul, alpha preserved,
// mask-mixed by the shared `main`). Provenance: the publicly documented
// Photoshop adjustment semantics (Vibrance, Color Balance, Black &
// White, Posterize, Threshold, Photo Filter, Channel Mixer, per-channel
// Levels) re-derived from standard image-processing literature — the
// exact formulas are DEFINED in the per-kernel comments below and are
// mirrored term-for-term by the scalar references in
// `image-conformance/tests/family_adjust.rs`. No reference reading.

// ───────────────────────────── vibrance ────────────────────────────
//
// Saturation-protecting saturation. On UNpremultiplied rgb:
//   lum   = 0.3r + 0.59g + 0.11b            (the family Lum weights)
//   satl  = max(r,g,b) − min(r,g,b)         (0 gray … 1 fully saturated)
//   f     = 1 + saturation + vibrance·(1 − satl)
//   c'    = mix(splat(lum), c, f)
// `vibrance` scales DOWN as the pixel's own saturation rises (already-
// saturated colors are protected); `saturation` is the uniform offset
// (both 0 = identity). Re-premultiplied, alpha preserved. Pure
// min/max/mix algebra — no transcendental.

/// Vibrance params: `vibrance` (saturation-protected boost) and
/// `saturation` (uniform offset); both 0 = identity.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustVibranceParams {
    pub vibrance: f32,
    pub saturation: f32,
    pub _abi_pad: u32,
}

impl AdjustVibranceParams {
    pub fn new(vibrance: f32, saturation: f32) -> Self {
        Self {
            vibrance,
            saturation,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const VIBRANCE_PARAMS_FIELDS: &[ParamField] = &[
    ParamField {
        name: "vibrance",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "saturation",
        wgsl_ty: "f32",
    },
];

/// c' = mix(splat(lum), c, 1 + saturation + vibrance·(1 − sat_level)) on
/// unpremult rgb; re-premultiplied, alpha preserved.
pub static ADJUST_VIBRANCE: KernelDef = KernelDef {
    id: "adjust.vibrance",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustVibranceParams>(),
        fields: VIBRANCE_PARAMS_FIELDS,
    },
    wgsl: VIBRANCE_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

// lum dot in fixed r,g,b order; the per-pixel factor f feeds the same
// mix the saturation kernel uses. The scalar reference mirrors the
// expression order exactly.
const VIBRANCE_WGSL: &str = adjust_wgsl!(
    "struct Params {
    vibrance: f32,
    saturation: f32,
    _abi_pad: u32,
}",
    "
fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = unpremul_rgb(a);
    let lum = 0.3 * c.r + 0.59 * c.g + 0.11 * c.b;
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let satl = mx - mn;
    let f = 1.0 + params.saturation + params.vibrance * (1.0 - satl);
    let cp = mix(vec3<f32>(lum), c, vec3<f32>(f));
    return vec4<f32>(cp * a.a, a.a);
}
"
);

// ─────────────────────────── color_balance ─────────────────────────
//
// Shadows/midtones/highlights × cyan-red/magenta-green/yellow-blue with
// luminosity preservation. On UNpremultiplied rgb:
//   lum = 0.3r + 0.59g + 0.11b
//   w_s = 1 − smoothstep(0, 0.5, lum)     (shadows weight)
//   w_h = smoothstep(0.5, 1, lum)         (highlights weight)
//   w_m = 1 − w_s − w_h                   (midtones weight)
//   r' = r + sh_cr·w_s + mid_cr·w_m + hi_cr·w_h   (cyan↔red axis, +→red)
//   g' = g + sh_mg·w_s + mid_mg·w_m + hi_mg·w_h   (magenta↔green, +→green)
//   b' = b + sh_yb·w_s + mid_yb·w_m + hi_yb·w_h   (yellow↔blue, +→blue)
//   c'' = clamp(c' + splat(lum − lum(c')), 0, 1)  (luminosity restore)
// The three weights form a smooth partition of unity over the tonal
// range (smoothstep is the WGSL builtin 3t²−2t³ Hermite, mirrored
// exactly by the scalar reference). All params 0 = identity (the
// luminosity-restore delta is then 0 and clamp is a no-op on in-gamut
// colors). Re-premultiplied, alpha preserved.

/// Color-balance params: per tonal range (shadows/midtones/highlights)
/// one offset per opponent axis (cyan-red, magenta-green, yellow-blue).
/// All 0 = identity.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustColorBalanceParams {
    pub sh_cr: f32,
    pub sh_mg: f32,
    pub sh_yb: f32,
    pub mid_cr: f32,
    pub mid_mg: f32,
    pub mid_yb: f32,
    pub hi_cr: f32,
    pub hi_mg: f32,
    pub hi_yb: f32,
    pub _abi_pad: u32,
}

#[allow(clippy::too_many_arguments)]
impl AdjustColorBalanceParams {
    pub fn new(shadows: [f32; 3], midtones: [f32; 3], highlights: [f32; 3]) -> Self {
        Self {
            sh_cr: shadows[0],
            sh_mg: shadows[1],
            sh_yb: shadows[2],
            mid_cr: midtones[0],
            mid_mg: midtones[1],
            mid_yb: midtones[2],
            hi_cr: highlights[0],
            hi_mg: highlights[1],
            hi_yb: highlights[2],
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const COLOR_BALANCE_PARAMS_FIELDS: &[ParamField] = &[
    ParamField {
        name: "sh_cr",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "sh_mg",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "sh_yb",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "mid_cr",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "mid_mg",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "mid_yb",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "hi_cr",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "hi_mg",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "hi_yb",
        wgsl_ty: "f32",
    },
];

/// Tonal-range-weighted opponent-axis offsets with luminosity restore on
/// unpremult rgb; re-premultiplied, alpha preserved.
pub static ADJUST_COLOR_BALANCE: KernelDef = KernelDef {
    id: "adjust.color_balance",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustColorBalanceParams>(),
        fields: COLOR_BALANCE_PARAMS_FIELDS,
    },
    wgsl: COLOR_BALANCE_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(8),
};

// Fixed evaluation order: weights first, then the three per-channel
// three-term sums (left to right), then the luminosity restore. The
// scalar reference mirrors every step.
const COLOR_BALANCE_WGSL: &str = adjust_wgsl!(
    "struct Params {
    sh_cr: f32,
    sh_mg: f32,
    sh_yb: f32,
    mid_cr: f32,
    mid_mg: f32,
    mid_yb: f32,
    hi_cr: f32,
    hi_mg: f32,
    hi_yb: f32,
    _abi_pad: u32,
}",
    "
fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = unpremul_rgb(a);
    let lum = 0.3 * c.r + 0.59 * c.g + 0.11 * c.b;
    let ws = 1.0 - smoothstep(0.0, 0.5, lum);
    let wh = smoothstep(0.5, 1.0, lum);
    let wm = 1.0 - ws - wh;
    let rr = c.r + params.sh_cr * ws + params.mid_cr * wm + params.hi_cr * wh;
    let gg = c.g + params.sh_mg * ws + params.mid_mg * wm + params.hi_mg * wh;
    let bb = c.b + params.sh_yb * ws + params.mid_yb * wm + params.hi_yb * wh;
    let lum2 = 0.3 * rr + 0.59 * gg + 0.11 * bb;
    let cp = clamp(
        vec3<f32>(rr, gg, bb) + vec3<f32>(lum - lum2),
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );
    return vec4<f32>(cp * a.a, a.a);
}
"
);

// ─────────────────────────── black_white ───────────────────────────
//
// Six-channel grayscale mix. Any rgb decomposes EXACTLY as
//   c = mn·white + (mid − mn)·secondary + (mx − mid)·primary
// where mn ≤ mid ≤ mx are the sorted channels, `primary` is the pure
// primary of the max channel (red/green/blue) and `secondary` the pure
// secondary shared by the top two channels (yellow/cyan/magenta). The
// gray value weights the two chromatic parts by the user's per-hue
// weights:
//   gray = mn + (mid − mn)·w_secondary + (mx − mid)·w_primary
// then clamp to [0,1]. A six-way branch ladder on channel order picks
// (primary, secondary); at every sector boundary the adjacent formulas
// COINCIDE (the vanishing term switches weight), so the function is
// continuous and branch-tie divergence between lanes is harmless.
// Weights all 1 = luminance-free identity gray = mx. Gray input (r=g=b)
// → gray = r regardless of weights. Output splats gray to rgb;
// re-premultiplied, alpha preserved.

/// Black & White params: grayscale weights for the six hue sectors.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustBlackWhiteParams {
    pub reds: f32,
    pub yellows: f32,
    pub greens: f32,
    pub cyans: f32,
    pub blues: f32,
    pub magentas: f32,
    pub _abi_pad: u32,
}

#[allow(clippy::too_many_arguments)]
impl AdjustBlackWhiteParams {
    pub fn new(
        reds: f32,
        yellows: f32,
        greens: f32,
        cyans: f32,
        blues: f32,
        magentas: f32,
    ) -> Self {
        Self {
            reds,
            yellows,
            greens,
            cyans,
            blues,
            magentas,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const BLACK_WHITE_PARAMS_FIELDS: &[ParamField] = &[
    ParamField {
        name: "reds",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "yellows",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "greens",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "cyans",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "blues",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "magentas",
        wgsl_ty: "f32",
    },
];

/// gray = mn + (mid−mn)·w_secondary + (mx−mid)·w_primary per the hue
/// sector; splatted to rgb, re-premultiplied, alpha preserved.
pub static ADJUST_BLACK_WHITE: KernelDef = KernelDef {
    id: "adjust.black_white",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustBlackWhiteParams>(),
        fields: BLACK_WHITE_PARAMS_FIELDS,
    },
    wgsl: BLACK_WHITE_WGSL,
    module: true,
    mip_exact: true,
    // Pure min/max/compare algebra + two multiplies (no transcendental):
    // the family's unpremul-amplification floor. Measured worst 1 ULP on
    // Metal over the mixed-alpha stimulus.
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

// The six-way ladder, in the FIXED order r≥g≥b, r≥b≥g, g≥r≥b, g≥b≥r,
// b≥g≥r, else (b≥r≥g) — identical in the scalar reference.
const BLACK_WHITE_WGSL: &str = adjust_wgsl!(
    "struct Params {
    reds: f32,
    yellows: f32,
    greens: f32,
    cyans: f32,
    blues: f32,
    magentas: f32,
    _abi_pad: u32,
}",
    "
fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = unpremul_rgb(a);
    let r = c.r;
    let g = c.g;
    let b = c.b;
    var gray: f32;
    if (r >= g && g >= b) {
        gray = b + (g - b) * params.yellows + (r - g) * params.reds;
    } else if (r >= b && b >= g) {
        gray = g + (b - g) * params.magentas + (r - b) * params.reds;
    } else if (g >= r && r >= b) {
        gray = b + (r - b) * params.yellows + (g - r) * params.greens;
    } else if (g >= b && b >= r) {
        gray = r + (b - r) * params.cyans + (g - b) * params.greens;
    } else if (b >= g && g >= r) {
        gray = r + (g - r) * params.cyans + (b - g) * params.blues;
    } else {
        gray = g + (r - g) * params.magentas + (b - r) * params.blues;
    }
    let gc = clamp(gray, 0.0, 1.0);
    return vec4<f32>(vec3<f32>(gc) * a.a, a.a);
}
"
);

// ───────────────────────────── posterize ───────────────────────────
//
// Quantize each channel of the UNpremultiplied rgb into `levels`
// discrete output values evenly spanning [0,1]:
//   n  = max(levels, 2)
//   t  = clamp(c, 0, 1)
//   c' = min(floor(t·n) / (n − 1), 1)
// (t = 1 lands in the top bin via the min guard). floor is a builtin in
// both lanes; multiplication is correctly rounded on both, so the bin
// choice agrees except for stimuli sitting within f32 noise of a bin
// edge — the parity stimulus uses power-of-two alphas so the
// unpremultiply is EXACT and the lanes see identical t. Re-premultiplied,
// alpha preserved.

/// Posterize params: `levels` — number of output values per channel
/// (effective minimum 2).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustPosterizeParams {
    pub levels: f32,
    pub _abi_pad: u32,
}

impl AdjustPosterizeParams {
    pub fn new(levels: f32) -> Self {
        Self {
            levels,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const POSTERIZE_PARAMS_FIELDS: &[ParamField] = &[ParamField {
    name: "levels",
    wgsl_ty: "f32",
}];

/// c' = min(floor(clamp(c,0,1)·n)/(n−1), 1), n = max(levels, 2), on
/// unpremult rgb; re-premultiplied, alpha preserved.
pub static ADJUST_POSTERIZE: KernelDef = KernelDef {
    id: "adjust.posterize",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustPosterizeParams>(),
        fields: POSTERIZE_PARAMS_FIELDS,
    },
    wgsl: POSTERIZE_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const POSTERIZE_WGSL: &str = adjust_wgsl!(
    "struct Params {
    levels: f32,
    _abi_pad: u32,
}",
    "
fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = unpremul_rgb(a);
    let n = max(params.levels, 2.0);
    let t = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let q = min(floor(t * n) / (n - 1.0), vec3<f32>(1.0));
    return vec4<f32>(q * a.a, a.a);
}
"
);

// ───────────────────────────── threshold ───────────────────────────
//
// Luma threshold → black/white. To avoid the unpremultiply division
// entirely (division noise could flip the comparison), the compare runs
// in PREMULTIPLIED space: lum is linear, so
//   lum(unpremul rgb) ≥ threshold  ⇔  lum(premul rgb) ≥ threshold·α.
//   out.rgb = α·1 = α if on, else 0   (premultiplied white/black)
// Both sides are products/sums of identical inputs — deterministic on
// both lanes. Alpha preserved.

/// Threshold params: luma `threshold` in [0,1].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustThresholdParams {
    pub threshold: f32,
    pub _abi_pad: u32,
}

impl AdjustThresholdParams {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const THRESHOLD_PARAMS_FIELDS: &[ParamField] = &[ParamField {
    name: "threshold",
    wgsl_ty: "f32",
}];

/// rgb = α·(lum ≥ threshold ? 1 : 0) via the premultiplied-luma compare;
/// alpha preserved.
pub static ADJUST_THRESHOLD: KernelDef = KernelDef {
    id: "adjust.threshold",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustThresholdParams>(),
        fields: THRESHOLD_PARAMS_FIELDS,
    },
    wgsl: THRESHOLD_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(1),
};

const THRESHOLD_WGSL: &str = adjust_wgsl!(
    "struct Params {
    threshold: f32,
    _abi_pad: u32,
}",
    "
fn adjust(a: vec4<f32>) -> vec4<f32> {
    let lum_p = 0.3 * a.r + 0.59 * a.g + 0.11 * a.b;
    let cutoff = params.threshold * a.a;
    var v = 0.0;
    if (lum_p >= cutoff) { v = a.a; }
    return vec4<f32>(v, v, v, a.a);
}
"
);

// ─────────────────────────── photo_filter ──────────────────────────
//
// A colored-gel filter: the filter color multiplies the light passing
// through (absorption model), `density` fades the effect in, and the
// preserve-luminosity flag restores the original luma:
//   tinted = c · filter_rgb
//   c'     = mix(c, tinted, density)
//   if preserve ≠ 0: c'' = clamp(c' + splat(lum(c) − lum(c')), 0, 1)
// on UNpremultiplied rgb; re-premultiplied, alpha preserved. density 0 =
// identity.

/// Photo-filter params: filter color rgb, `density` in [0,1], and the
/// preserve-luminosity flag (0 = off, else on).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustPhotoFilterParams {
    pub fr: f32,
    pub fg: f32,
    pub fb: f32,
    pub density: f32,
    pub preserve: u32,
    pub _abi_pad: u32,
}

impl AdjustPhotoFilterParams {
    pub fn new(filter: [f32; 3], density: f32, preserve: bool) -> Self {
        Self {
            fr: filter[0],
            fg: filter[1],
            fb: filter[2],
            density,
            preserve: u32::from(preserve),
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const PHOTO_FILTER_PARAMS_FIELDS: &[ParamField] = &[
    ParamField {
        name: "fr",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "fg",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "fb",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "density",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "preserve",
        wgsl_ty: "u32",
    },
];

/// c' = mix(c, c·filter, density) with optional luminosity restore on
/// unpremult rgb; re-premultiplied, alpha preserved.
pub static ADJUST_PHOTO_FILTER: KernelDef = KernelDef {
    id: "adjust.photo_filter",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustPhotoFilterParams>(),
        fields: PHOTO_FILTER_PARAMS_FIELDS,
    },
    wgsl: PHOTO_FILTER_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const PHOTO_FILTER_WGSL: &str = adjust_wgsl!(
    "struct Params {
    fr: f32,
    fg: f32,
    fb: f32,
    density: f32,
    preserve: u32,
    _abi_pad: u32,
}",
    "
fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = unpremul_rgb(a);
    let fcol = vec3<f32>(params.fr, params.fg, params.fb);
    let tinted = c * fcol;
    var cp = mix(c, tinted, vec3<f32>(params.density));
    if (params.preserve != 0u) {
        let lum0 = 0.3 * c.r + 0.59 * c.g + 0.11 * c.b;
        let lum1 = 0.3 * cp.r + 0.59 * cp.g + 0.11 * cp.b;
        cp = clamp(cp + vec3<f32>(lum0 - lum1), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return vec4<f32>(cp * a.a, a.a);
}
"
);

// ─────────────────────────── channel_mixer ─────────────────────────
//
// 3×4 matrix: each output channel is a weighted mix of the three input
// channels plus a constant, on UNpremultiplied rgb:
//   r' = rr·r + rg·g + rb·b + rc     (fixed left-to-right sums)
//   g' = gr·r + gg·g + gb·b + gc
//   b' = br·r + bg·g + bb·b + bc
// Identity = the identity matrix with zero constants. No clamp (the f16
// working space carries over/under-range like the other adjust ops).
// Re-premultiplied, alpha preserved.

/// Channel-mixer params: row-major 3×4 matrix (output channel × input
/// r,g,b + constant).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustChannelMixerParams {
    pub rr: f32,
    pub rg: f32,
    pub rb: f32,
    pub rc: f32,
    pub gr: f32,
    pub gg: f32,
    pub gb: f32,
    pub gc: f32,
    pub br: f32,
    pub bg: f32,
    pub bb: f32,
    pub bc: f32,
    pub _abi_pad: u32,
}

impl AdjustChannelMixerParams {
    /// Rows are `[in_r, in_g, in_b, constant]` for the r, g, b outputs.
    pub fn new(r_row: [f32; 4], g_row: [f32; 4], b_row: [f32; 4]) -> Self {
        Self {
            rr: r_row[0],
            rg: r_row[1],
            rb: r_row[2],
            rc: r_row[3],
            gr: g_row[0],
            gg: g_row[1],
            gb: g_row[2],
            gc: g_row[3],
            br: b_row[0],
            bg: b_row[1],
            bb: b_row[2],
            bc: b_row[3],
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const CHANNEL_MIXER_PARAMS_FIELDS: &[ParamField] = &[
    ParamField {
        name: "rr",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "rg",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "rb",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "rc",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "gr",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "gg",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "gb",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "gc",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "br",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "bg",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "bb",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "bc",
        wgsl_ty: "f32",
    },
];

/// 3×4 channel-mix matrix on unpremult rgb; re-premultiplied, alpha
/// preserved.
pub static ADJUST_CHANNEL_MIXER: KernelDef = KernelDef {
    id: "adjust.channel_mixer",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustChannelMixerParams>(),
        fields: CHANNEL_MIXER_PARAMS_FIELDS,
    },
    wgsl: CHANNEL_MIXER_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const CHANNEL_MIXER_WGSL: &str = adjust_wgsl!(
    "struct Params {
    rr: f32,
    rg: f32,
    rb: f32,
    rc: f32,
    gr: f32,
    gg: f32,
    gb: f32,
    gc: f32,
    br: f32,
    bg: f32,
    bb: f32,
    bc: f32,
    _abi_pad: u32,
}",
    "
fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = unpremul_rgb(a);
    let rr = params.rr * c.r + params.rg * c.g + params.rb * c.b + params.rc;
    let gg = params.gr * c.r + params.gg * c.g + params.gb * c.b + params.gc;
    let bb = params.br * c.r + params.bg * c.g + params.bb * c.b + params.bc;
    return vec4<f32>(vec3<f32>(rr, gg, bb) * a.a, a.a);
}
"
);

// ─────────────────────────── levels_rgb ────────────────────────────
//
// Per-channel Levels input remap (in_black / in_white / gamma per r, g,
// b; the output range stays composite — use `adjust.levels` for output
// remaps). Per channel, on UNpremultiplied rgb:
//   t  = clamp((c − in_black) / (in_white − in_black), 0, 1)
//   c' = pow(t, 1 / gamma)
// Identity = {0, 1, 1} per channel. Same pow-tolerance rationale as
// `adjust.levels` (GPU pow approximation × unpremultiply f16-noise
// amplification): ChannelEpsF16(16).

/// Per-channel levels params: in_black/in_white/gamma for r, g, b.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustLevelsRgbParams {
    pub r_in_black: f32,
    pub r_in_white: f32,
    pub r_gamma: f32,
    pub g_in_black: f32,
    pub g_in_white: f32,
    pub g_gamma: f32,
    pub b_in_black: f32,
    pub b_in_white: f32,
    pub b_gamma: f32,
    pub _abi_pad: u32,
}

impl AdjustLevelsRgbParams {
    /// Per channel: `[in_black, in_white, gamma]`.
    pub fn new(r: [f32; 3], g: [f32; 3], b: [f32; 3]) -> Self {
        Self {
            r_in_black: r[0],
            r_in_white: r[1],
            r_gamma: r[2],
            g_in_black: g[0],
            g_in_white: g[1],
            g_gamma: g[2],
            b_in_black: b[0],
            b_in_white: b[1],
            b_gamma: b[2],
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const LEVELS_RGB_PARAMS_FIELDS: &[ParamField] = &[
    ParamField {
        name: "r_in_black",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "r_in_white",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "r_gamma",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "g_in_black",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "g_in_white",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "g_gamma",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "b_in_black",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "b_in_white",
        wgsl_ty: "f32",
    },
    ParamField {
        name: "b_gamma",
        wgsl_ty: "f32",
    },
];

/// Per-channel input/gamma levels remap on unpremult rgb;
/// re-premultiplied, alpha preserved.
pub static ADJUST_LEVELS_RGB: KernelDef = KernelDef {
    id: "adjust.levels_rgb",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustLevelsRgbParams>(),
        fields: LEVELS_RGB_PARAMS_FIELDS,
    },
    wgsl: LEVELS_RGB_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(16),
};

// The vector pow applies componentwise; per-channel windows/gammas are
// packed into vec3 registers first. WGSL pow / Rust powf mirror.
const LEVELS_RGB_WGSL: &str = adjust_wgsl!(
    "struct Params {
    r_in_black: f32,
    r_in_white: f32,
    r_gamma: f32,
    g_in_black: f32,
    g_in_white: f32,
    g_gamma: f32,
    b_in_black: f32,
    b_in_white: f32,
    b_gamma: f32,
    _abi_pad: u32,
}",
    "
fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = unpremul_rgb(a);
    let ib = vec3<f32>(params.r_in_black, params.g_in_black, params.b_in_black);
    let iw = vec3<f32>(params.r_in_white, params.g_in_white, params.b_in_white);
    let ga = vec3<f32>(params.r_gamma, params.g_gamma, params.b_gamma);
    let t0 = (c - ib) / (iw - ib);
    let t1 = clamp(t0, vec3<f32>(0.0), vec3<f32>(1.0));
    let cp = pow(t1, vec3<f32>(1.0) / ga);
    return vec4<f32>(cp * a.a, a.a);
}
"
);

// ────────────────────────────── lut1d ──────────────────────────────
//
// A 256-entry per-channel LOOKUP TABLE applied to unpremultiplied rgb —
// the class behind Curves and Gradient Map.
//
// The table travels in PARAMS, not as a second input texture, and that
// is forced rather than chosen: the ABI is frozen at M0 and every kernel
// input uploads at the tile's `w`×`h`, so a 256×1 LUT texture would need
// a versioned ABI amendment. 256 entries as 64 `vec4<f32>` is 1 KiB of
// uniform — well inside any uniform-buffer bound — so the table fits the
// existing contract with nothing to amend.
//
// Sampling is LINEARLY INTERPOLATED between table entries, and that was
// measured rather than assumed. Nearest-index sampling — which would
// match `ingest::apply_curve_lut`'s CPU indexing bit for bit — makes the
// kernel a STEP function, and a step function cannot hold an f16-ULP
// parity tolerance: the GPU reads f16 texels, so a value sitting on an
// index boundary rounds to a different entry than the f32 reference
// picks, and one whole LUT step separates the answers. Measured 61 ULP
// on the IDENTITY table and 135 on an inverted one, against a declared
// tolerance of 4.
//
// Interpolating makes the transfer continuous, so a small input
// difference can only produce a small output difference — and it is the
// better curve anyway (a 256-entry table read with nearest quantizes
// output to 1/255 steps). Alpha is never remapped either way.

/// A 256-entry LUT as 64 `vec4<f32>` (WGSL uniform arrays need a 16-byte
/// stride, so four entries share a vector).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustLut1dParams {
    pub lut: [[f32; 4]; 64],
}

impl AdjustLut1dParams {
    /// Build from the same `[u8; 256]` table the panel hands the CPU
    /// pass, normalized to `[0, 1]`.
    pub fn new(lut: &[u8; 256]) -> Self {
        let mut packed = [[0.0f32; 4]; 64];
        for (i, &v) in lut.iter().enumerate() {
            packed[i / 4][i % 4] = f32::from(v) / 255.0;
        }
        Self { lut: packed }
    }

    /// The identity table — `lut[i] = i`.
    pub fn identity() -> Self {
        let mut lut = [0u8; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = i as u8;
        }
        Self::new(&lut)
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const LUT1D_PARAMS_FIELDS: &[ParamField] = &[ParamField {
    name: "lut",
    wgsl_ty: "array<vec4<f32>, 64>",
}];

/// out.rgb = lut[round(c·255)] per channel on unpremultiplied rgb;
/// out.a = a.a. Alpha is never remapped (the curves contract).
pub static ADJUST_LUT1D: KernelDef = KernelDef {
    id: "adjust.lut1d",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustLut1dParams>(),
        fields: LUT1D_PARAMS_FIELDS,
    },
    wgsl: LUT1D_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const LUT1D_WGSL: &str = adjust_wgsl!(
    "struct Params {
    lut: array<vec4<f32>, 64>,
}",
    "
fn lut_at(i: i32) -> f32 {
    let j = clamp(i, 0, 255);
    return params.lut[j / 4][j % 4];
}

fn lut_lerp(v: f32) -> f32 {
    let t = clamp(v, 0.0, 1.0) * 255.0;
    let lo = i32(floor(t));
    let hi = min(lo + 1, 255);
    return mix(lut_at(lo), lut_at(hi), t - floor(t));
}

fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = unpremul_rgb(a);
    let mapped = vec3<f32>(lut_lerp(c.r), lut_lerp(c.g), lut_lerp(c.b));
    return vec4<f32>(mapped * a.a, a.a);
}
"
);

// ────────────────────────────── lut3d ──────────────────────────────
//
// A trilinearly-interpolated 3D colour cube — the class behind Color
// Lookup, and the only honest way to express a transform where the
// output of one channel depends on the other two (a 1D LUT cannot: it
// is three independent transfers).
//
// SIZE 17 is the cube edge, chosen because it is the .CUBE convention
// most colour-grading LUTs ship at and it fits the same params-only
// constraint `adjust.lut1d` documents: 17^3 = 4913 entries. Packed one
// rgb triple per `vec4<f32>` (the w lane unused, WGSL uniform arrays
// need the 16-byte stride) that is 78 KiB — over the 64 KiB uniform
// floor, so this kernel declares its table as four-channel entries at
// EDGE 9 (729 entries, 11.4 KiB) instead. Nine is coarse for a grade
// but it is what the frozen ABI can carry without an amendment, and a
// coarse honest cube beats a fine one that cannot be bound.
//
// Interpolation is trilinear for the same reason lut1d interpolates:
// nearest sampling is a step function and cannot hold an f16-ULP parity
// tolerance.

/// The 3D LUT cube edge. See the module note for why it is 9 and not 17.
pub const LUT3D_EDGE: usize = 9;
const LUT3D_ENTRIES: usize = LUT3D_EDGE * LUT3D_EDGE * LUT3D_EDGE;

/// A 9x9x9 RGB cube, one entry per `vec4<f32>` (w unused).
#[repr(C)]
#[derive(Debug, Clone, Copy, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustLut3dParams {
    pub cube: [[f32; 4]; LUT3D_ENTRIES],
}

impl AdjustLut3dParams {
    /// Build from a closure over the normalized lattice point.
    pub fn from_fn(f: impl Fn(f32, f32, f32) -> [f32; 3]) -> Self {
        let mut cube = [[0.0f32; 4]; LUT3D_ENTRIES];
        let n = (LUT3D_EDGE - 1) as f32;
        for b in 0..LUT3D_EDGE {
            for g in 0..LUT3D_EDGE {
                for r in 0..LUT3D_EDGE {
                    let v = f(r as f32 / n, g as f32 / n, b as f32 / n);
                    let i = (b * LUT3D_EDGE + g) * LUT3D_EDGE + r;
                    cube[i] = [v[0], v[1], v[2], 0.0];
                }
            }
        }
        Self { cube }
    }

    /// The identity cube — out == in at every lattice point.
    pub fn identity() -> Self {
        Self::from_fn(|r, g, b| [r, g, b])
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const LUT3D_PARAMS_FIELDS: &[ParamField] = &[ParamField {
    name: "cube",
    wgsl_ty: "array<vec4<f32>, 729>",
}];

/// out.rgb = trilinear(cube, unpremul(a).rgb); out.a = a.a.
pub static ADJUST_LUT3D: KernelDef = KernelDef {
    id: "adjust.lut3d",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustLut3dParams>(),
        fields: LUT3D_PARAMS_FIELDS,
    },
    wgsl: LUT3D_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(8),
};

const LUT3D_WGSL: &str = adjust_wgsl!(
    "struct Params {
    cube: array<vec4<f32>, 729>,
}",
    "
const EDGE : i32 = 9;

fn cube_at(r: i32, g: i32, b: i32) -> vec3<f32> {
    let rr = clamp(r, 0, EDGE - 1);
    let gg = clamp(g, 0, EDGE - 1);
    let bb = clamp(b, 0, EDGE - 1);
    return params.cube[(bb * EDGE + gg) * EDGE + rr].rgb;
}

fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = clamp(unpremul_rgb(a), vec3<f32>(0.0), vec3<f32>(1.0));
    let t = c * f32(EDGE - 1);
    let i0 = vec3<i32>(floor(t));
    let f = t - floor(t);

    // Trilinear: four edge lerps, two face lerps, one along b.
    let c000 = cube_at(i0.x, i0.y, i0.z);
    let c100 = cube_at(i0.x + 1, i0.y, i0.z);
    let c010 = cube_at(i0.x, i0.y + 1, i0.z);
    let c110 = cube_at(i0.x + 1, i0.y + 1, i0.z);
    let c001 = cube_at(i0.x, i0.y, i0.z + 1);
    let c101 = cube_at(i0.x + 1, i0.y, i0.z + 1);
    let c011 = cube_at(i0.x, i0.y + 1, i0.z + 1);
    let c111 = cube_at(i0.x + 1, i0.y + 1, i0.z + 1);

    let x00 = mix(c000, c100, f.x);
    let x10 = mix(c010, c110, f.x);
    let x01 = mix(c001, c101, f.x);
    let x11 = mix(c011, c111, f.x);
    let y0 = mix(x00, x10, f.y);
    let y1 = mix(x01, x11, f.y);
    let mapped = mix(y0, y1, f.z);

    return vec4<f32>(mapped * a.a, a.a);
}
"
);

// ─────────────────────────── gradient_map ──────────────────────────
//
// Map LUMINANCE through a colour ramp — the Photoshop Gradient Map.
//
// Not expressible with `adjust.lut1d`, and the difference is worth
// stating because the catalog groups them: lut1d applies one table to
// each channel INDEPENDENTLY, so it can never make output red depend on
// input green. A gradient map reads a single luminance and returns an
// rgb TRIPLE, which is a 256-entry RGB ramp indexed by luma. Same
// params-carrying pattern (the ABI is frozen; see lut1d), same
// interpolation for the same measured reason.
//
// Luma is Rec.709 on unpremultiplied rgb — the same coefficients
// `adjust.black_white` defaults to, so a desaturate-then-map and a
// gradient map agree about what "brightness" means.

/// A 256-entry RGB ramp, one entry per `vec4<f32>` (w unused).
#[repr(C)]
#[derive(Debug, Clone, Copy, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct AdjustGradientMapParams {
    pub ramp: [[f32; 4]; 256],
}

impl AdjustGradientMapParams {
    /// Build a ramp by interpolating between two endpoint colours — the
    /// two-stop gradient a UI exposes first.
    pub fn two_stop(shadow: [f32; 3], highlight: [f32; 3]) -> Self {
        Self::from_fn(|t| {
            [
                shadow[0] + (highlight[0] - shadow[0]) * t,
                shadow[1] + (highlight[1] - shadow[1]) * t,
                shadow[2] + (highlight[2] - shadow[2]) * t,
            ]
        })
    }

    /// Build from a closure over the normalized luminance.
    pub fn from_fn(f: impl Fn(f32) -> [f32; 3]) -> Self {
        let mut ramp = [[0.0f32; 4]; 256];
        for (i, e) in ramp.iter_mut().enumerate() {
            let v = f(i as f32 / 255.0);
            *e = [v[0], v[1], v[2], 0.0];
        }
        Self { ramp }
    }

    /// The identity-ish ramp: black to white, which maps an image to its
    /// own luminance (a greyscale conversion).
    pub fn greyscale() -> Self {
        Self::two_stop([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

const GRADIENT_MAP_FIELDS: &[ParamField] = &[ParamField {
    name: "ramp",
    wgsl_ty: "array<vec4<f32>, 256>",
}];

/// out.rgb = ramp[luma709(unpremul(a))]; out.a = a.a.
pub static ADJUST_GRADIENT_MAP: KernelDef = KernelDef {
    id: "adjust.gradient_map",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<AdjustGradientMapParams>(),
        fields: GRADIENT_MAP_FIELDS,
    },
    wgsl: GRADIENT_MAP_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GRADIENT_MAP_WGSL: &str = adjust_wgsl!(
    "struct Params {
    ramp: array<vec4<f32>, 256>,
}",
    "
fn ramp_at(i: i32) -> vec3<f32> {
    return params.ramp[clamp(i, 0, 255)].rgb;
}

fn adjust(a: vec4<f32>) -> vec4<f32> {
    let c = unpremul_rgb(a);
    let luma = clamp(dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
    let t = luma * 255.0;
    let lo = i32(floor(t));
    let mapped = mix(ramp_at(lo), ramp_at(lo + 1), t - floor(t));
    return vec4<f32>(mapped * a.a, a.a);
}
"
);

pub static FAMILY: &[&KernelDef] = &[
    &ADJUST_LUT1D,
    &ADJUST_LUT3D,
    &ADJUST_GRADIENT_MAP,
    &ADJUST_EXPOSURE,
    &ADJUST_BRIGHTNESS_CONTRAST,
    &ADJUST_LEVELS,
    &ADJUST_SATURATION,
    &ADJUST_HUE_ROTATE,
    &ADJUST_INVERT_RGB,
    &ADJUST_WHITE_BALANCE,
    &ADJUST_VIBRANCE,
    &ADJUST_COLOR_BALANCE,
    &ADJUST_BLACK_WHITE,
    &ADJUST_POSTERIZE,
    &ADJUST_THRESHOLD,
    &ADJUST_PHOTO_FILTER,
    &ADJUST_CHANNEL_MIXER,
    &ADJUST_LEVELS_RGB,
];
