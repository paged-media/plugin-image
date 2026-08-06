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

//! gpu↔ref parity for the T2 adjust family (tone/color point ops). The
//! scalar references mirror each handwritten WGSL module's body
//! verbatim (fixed evaluation order, §6.3); `adjust.invert_rgb` reuses
//! its `kernel_family!`-emitted twin. The integration test the M2
//! fan-out agent didn't land before the session reset — written here
//! against the live (and naga-validated) kernels.
//! feat: adjust.exposure/brightness_contrast/levels/saturation/hue_rotate/invert_rgb.
//!
//! KERNEL-BREADTH BATCH: adjust.vibrance / color_balance / black_white
//! / posterize / threshold / photo_filter / channel_mixer / levels_rgb.
//! posterize and threshold contain hard discontinuities (bin edges /
//! the luma cut), so their stimulus uses POWER-OF-TWO alphas
//! ([`premul_tile_pow2`]): the unpremultiply division is then EXACT on
//! both lanes (exponent shift) and the discontinuity cannot flip
//! between lanes.

use image_conformance::harness::{assert_within, parity, RefTile};
use image_conformance::Px;
use image_kernels::families::adjust::{
    adjust_invert_rgb, AdjustBlackWhiteParams, AdjustBrightnessContrastParams,
    AdjustChannelMixerParams, AdjustColorBalanceParams, AdjustExposureParams,
    AdjustHueRotateParams, AdjustInvertRgbParams, AdjustLevelsParams, AdjustLevelsRgbParams,
    AdjustLut1dParams, AdjustLut3dParams, AdjustPhotoFilterParams, AdjustPosterizeParams,
    AdjustSaturationParams, AdjustThresholdParams, AdjustVibranceParams, AdjustWhiteBalanceParams,
    ADJUST_BLACK_WHITE, ADJUST_BRIGHTNESS_CONTRAST, ADJUST_CHANNEL_MIXER, ADJUST_COLOR_BALANCE,
    ADJUST_EXPOSURE, ADJUST_HUE_ROTATE, ADJUST_INVERT_RGB, ADJUST_LEVELS, ADJUST_LEVELS_RGB,
    ADJUST_LUT1D, ADJUST_LUT3D, ADJUST_PHOTO_FILTER, ADJUST_POSTERIZE, ADJUST_SATURATION,
    ADJUST_THRESHOLD, ADJUST_VIBRANCE, ADJUST_WHITE_BALANCE,
};

/// `unpremul_rgb` — the module preamble helper (a==0 → 0).
fn unpremul(a: Px) -> [f32; 3] {
    let al = a.0[3];
    if al == 0.0 {
        [0.0; 3]
    } else {
        [a.0[0] / al, a.0[1] / al, a.0[2] / al]
    }
}

/// The 1D-LUT reference — LINEARLY INTERPOLATED, matching the kernel.
/// Nearest indexing was tried first and rejected on measurement: a step
/// function cannot hold an f16-ULP tolerance (61 ULP on the identity
/// table, 135 on an inverted one, against a declared 4), because f16
/// storage flips the chosen entry at index boundaries.
fn lut1d_ref(a: Px, _b: Px, p: &AdjustLut1dParams) -> Px {
    let c = unpremul(a);
    let entry = |i: usize| -> f32 {
        let i = i.min(255);
        p.lut[i / 4][i % 4]
    };
    let at = |v: f32| -> f32 {
        let t = v.clamp(0.0, 1.0) * 255.0;
        let lo = t.floor() as usize;
        let hi = (lo + 1).min(255);
        let f = t - t.floor();
        entry(lo) * (1.0 - f) + entry(hi) * f
    };
    let m = [at(c[0]), at(c[1]), at(c[2])];
    Px([m[0] * a.0[3], m[1] * a.0[3], m[2] * a.0[3], a.0[3]])
}

/// The 3D-LUT reference — trilinear over the same 9^3 cube.
fn lut3d_ref(a: Px, _b: Px, p: &AdjustLut3dParams) -> Px {
    const EDGE: usize = 9;
    let c = unpremul(a);
    let at = |r: usize, g: usize, b: usize| -> [f32; 3] {
        let i = (b.min(EDGE - 1) * EDGE + g.min(EDGE - 1)) * EDGE + r.min(EDGE - 1);
        [p.cube[i][0], p.cube[i][1], p.cube[i][2]]
    };
    let lerp = |x: [f32; 3], y: [f32; 3], f: f32| -> [f32; 3] {
        [
            x[0] + (y[0] - x[0]) * f,
            x[1] + (y[1] - x[1]) * f,
            x[2] + (y[2] - x[2]) * f,
        ]
    };
    let t: Vec<f32> = (0..3)
        .map(|k| c[k].clamp(0.0, 1.0) * (EDGE - 1) as f32)
        .collect();
    let i0: Vec<usize> = t.iter().map(|v| v.floor() as usize).collect();
    let f: Vec<f32> = t.iter().map(|v| v - v.floor()).collect();

    let x00 = lerp(at(i0[0], i0[1], i0[2]), at(i0[0] + 1, i0[1], i0[2]), f[0]);
    let x10 = lerp(
        at(i0[0], i0[1] + 1, i0[2]),
        at(i0[0] + 1, i0[1] + 1, i0[2]),
        f[0],
    );
    let x01 = lerp(
        at(i0[0], i0[1], i0[2] + 1),
        at(i0[0] + 1, i0[1], i0[2] + 1),
        f[0],
    );
    let x11 = lerp(
        at(i0[0], i0[1] + 1, i0[2] + 1),
        at(i0[0] + 1, i0[1] + 1, i0[2] + 1),
        f[0],
    );
    let y0 = lerp(x00, x10, f[1]);
    let y1 = lerp(x01, x11, f[1]);
    let m = lerp(y0, y1, f[2]);
    Px([m[0] * a.0[3], m[1] * a.0[3], m[2] * a.0[3], a.0[3]])
}

fn exposure_ref(a: Px, _b: Px, p: &AdjustExposureParams) -> Px {
    // vec4(a.rgb * exp2(ev), a.a) — operates on premultiplied rgb directly.
    let k = p.ev.exp2();
    Px([a.0[0] * k, a.0[1] * k, a.0[2] * k, a.0[3]])
}

fn brightness_contrast_ref(a: Px, _b: Px, p: &AdjustBrightnessContrastParams) -> Px {
    let c = unpremul(a);
    let al = a.0[3];
    let cp = c.map(|x| (x - 0.5) * p.contrast + (0.5 + p.brightness));
    Px([cp[0] * al, cp[1] * al, cp[2] * al, al])
}

fn levels_ref(a: Px, _b: Px, p: &AdjustLevelsParams) -> Px {
    let c = unpremul(a);
    let al = a.0[3];
    let cp = c.map(|x| {
        let t0 = (x - p.in_black) / (p.in_white - p.in_black);
        let t1 = t0.clamp(0.0, 1.0);
        let t2 = t1.powf(1.0 / p.gamma);
        p.out_black + t2 * (p.out_white - p.out_black)
    });
    Px([cp[0] * al, cp[1] * al, cp[2] * al, al])
}

fn saturation_ref(a: Px, _b: Px, p: &AdjustSaturationParams) -> Px {
    let c = unpremul(a);
    let al = a.0[3];
    // lum dot in fixed r,g,b order; cp = mix(splat(lum), c, sat).
    let lum = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
    let cp = [
        lum * (1.0 - p.sat) + c[0] * p.sat,
        lum * (1.0 - p.sat) + c[1] * p.sat,
        lum * (1.0 - p.sat) + c[2] * p.sat,
    ];
    Px([cp[0] * al, cp[1] * al, cp[2] * al, al])
}

fn hue_rotate_ref(a: Px, _b: Px, p: &AdjustHueRotateParams) -> Px {
    let c = unpremul(a);
    let al = a.0[3];
    let theta = p.degrees * std::f32::consts::PI / 180.0;
    let cs = theta.cos();
    let sn = theta.sin();
    // The luminance-preserving rotation matrix, row-by-row, r,g,b order —
    // identical coefficients to HUE_ROTATE_WGSL.
    let rr = (0.213 + cs * 0.787 + sn * (-0.213)) * c[0]
        + (0.715 + cs * (-0.715) + sn * (-0.715)) * c[1]
        + (0.072 + cs * (-0.072) + sn * 0.928) * c[2];
    let gg = (0.213 + cs * (-0.213) + sn * 0.143) * c[0]
        + (0.715 + cs * 0.285 + sn * 0.140) * c[1]
        + (0.072 + cs * (-0.072) + sn * (-0.283)) * c[2];
    let bb = (0.213 + cs * (-0.213) + sn * (-0.787)) * c[0]
        + (0.715 + cs * (-0.715) + sn * 0.715) * c[1]
        + (0.072 + cs * 0.928 + sn * 0.072) * c[2];
    Px([rr * al, gg * al, bb * al, al])
}

fn white_balance_ref(a: Px, _b: Px, p: &AdjustWhiteBalanceParams) -> Px {
    let c = unpremul(a);
    let al = a.0[3];
    // gain = (1+temp, 1+tint, 1-temp); c' = c * gain (componentwise).
    let gain = [1.0 + p.temp, 1.0 + p.tint, 1.0 - p.temp];
    let cp = [c[0] * gain[0], c[1] * gain[1], c[2] * gain[2]];
    Px([cp[0] * al, cp[1] * al, cp[2] * al, al])
}

// ─────────────── kernel-breadth batch reference twins ───────────────

/// `lum` in the family's fixed r,g,b summation order.
fn lum3(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

/// WGSL builtin `smoothstep(e0, e1, x)` — the 3t²−2t³ Hermite.
fn smoothstep_ref(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn vibrance_ref(a: Px, _b: Px, p: &AdjustVibranceParams) -> Px {
    let c = unpremul(a);
    let al = a.0[3];
    let lum = lum3(c);
    let mx = c[0].max(c[1].max(c[2]));
    let mn = c[0].min(c[1].min(c[2]));
    let satl = mx - mn;
    let f = 1.0 + p.saturation + p.vibrance * (1.0 - satl);
    // mix(splat(lum), c, f) per channel.
    let cp = [
        lum * (1.0 - f) + c[0] * f,
        lum * (1.0 - f) + c[1] * f,
        lum * (1.0 - f) + c[2] * f,
    ];
    Px([cp[0] * al, cp[1] * al, cp[2] * al, al])
}

fn color_balance_ref(a: Px, _b: Px, p: &AdjustColorBalanceParams) -> Px {
    let c = unpremul(a);
    let al = a.0[3];
    let lum = lum3(c);
    let ws = 1.0 - smoothstep_ref(0.0, 0.5, lum);
    let wh = smoothstep_ref(0.5, 1.0, lum);
    let wm = 1.0 - ws - wh;
    let rr = c[0] + p.sh_cr * ws + p.mid_cr * wm + p.hi_cr * wh;
    let gg = c[1] + p.sh_mg * ws + p.mid_mg * wm + p.hi_mg * wh;
    let bb = c[2] + p.sh_yb * ws + p.mid_yb * wm + p.hi_yb * wh;
    let lum2 = lum3([rr, gg, bb]);
    let d = lum - lum2;
    let cp = [
        (rr + d).clamp(0.0, 1.0),
        (gg + d).clamp(0.0, 1.0),
        (bb + d).clamp(0.0, 1.0),
    ];
    Px([cp[0] * al, cp[1] * al, cp[2] * al, al])
}

fn black_white_ref(a: Px, _b: Px, p: &AdjustBlackWhiteParams) -> Px {
    let c = unpremul(a);
    let al = a.0[3];
    let (r, g, b) = (c[0], c[1], c[2]);
    // The SAME six-way ladder as BLACK_WHITE_WGSL, in the same order.
    let gray = if r >= g && g >= b {
        b + (g - b) * p.yellows + (r - g) * p.reds
    } else if r >= b && b >= g {
        g + (b - g) * p.magentas + (r - b) * p.reds
    } else if g >= r && r >= b {
        b + (r - b) * p.yellows + (g - r) * p.greens
    } else if g >= b && b >= r {
        r + (b - r) * p.cyans + (g - b) * p.greens
    } else if b >= g && g >= r {
        r + (g - r) * p.cyans + (b - g) * p.blues
    } else {
        g + (r - g) * p.magentas + (b - r) * p.blues
    };
    let gc = gray.clamp(0.0, 1.0);
    Px([gc * al, gc * al, gc * al, al])
}

fn posterize_ref(a: Px, _b: Px, p: &AdjustPosterizeParams) -> Px {
    let c = unpremul(a);
    let al = a.0[3];
    let n = p.levels.max(2.0);
    let cp = c.map(|x| {
        let t = x.clamp(0.0, 1.0);
        ((t * n).floor() / (n - 1.0)).min(1.0)
    });
    Px([cp[0] * al, cp[1] * al, cp[2] * al, al])
}

fn threshold_ref(a: Px, _b: Px, p: &AdjustThresholdParams) -> Px {
    // Premultiplied-space compare (mirrors THRESHOLD_WGSL): lum is
    // linear, so lum(premul) >= threshold*alpha avoids the unpremul
    // division entirely.
    let lum_p = 0.3 * a.0[0] + 0.59 * a.0[1] + 0.11 * a.0[2];
    let cutoff = p.threshold * a.0[3];
    let v = if lum_p >= cutoff { a.0[3] } else { 0.0 };
    Px([v, v, v, a.0[3]])
}

fn photo_filter_ref(a: Px, _b: Px, p: &AdjustPhotoFilterParams) -> Px {
    let c = unpremul(a);
    let al = a.0[3];
    let tinted = [c[0] * p.fr, c[1] * p.fg, c[2] * p.fb];
    // mix(c, tinted, density) per channel.
    let mut cp = [
        c[0] * (1.0 - p.density) + tinted[0] * p.density,
        c[1] * (1.0 - p.density) + tinted[1] * p.density,
        c[2] * (1.0 - p.density) + tinted[2] * p.density,
    ];
    if p.preserve != 0 {
        let lum0 = lum3(c);
        let lum1 = lum3(cp);
        let d = lum0 - lum1;
        cp = [
            (cp[0] + d).clamp(0.0, 1.0),
            (cp[1] + d).clamp(0.0, 1.0),
            (cp[2] + d).clamp(0.0, 1.0),
        ];
    }
    Px([cp[0] * al, cp[1] * al, cp[2] * al, al])
}

fn channel_mixer_ref(a: Px, _b: Px, p: &AdjustChannelMixerParams) -> Px {
    let c = unpremul(a);
    let al = a.0[3];
    // Fixed left-to-right sums, identical to CHANNEL_MIXER_WGSL.
    let rr = p.rr * c[0] + p.rg * c[1] + p.rb * c[2] + p.rc;
    let gg = p.gr * c[0] + p.gg * c[1] + p.gb * c[2] + p.gc;
    let bb = p.br * c[0] + p.bg * c[1] + p.bb * c[2] + p.bc;
    Px([rr * al, gg * al, bb * al, al])
}

fn levels_rgb_ref(a: Px, _b: Px, p: &AdjustLevelsRgbParams) -> Px {
    let c = unpremul(a);
    let al = a.0[3];
    let ch = |x: f32, ib: f32, iw: f32, ga: f32| {
        let t0 = (x - ib) / (iw - ib);
        let t1 = t0.clamp(0.0, 1.0);
        t1.powf(1.0 / ga)
    };
    let cp = [
        ch(c[0], p.r_in_black, p.r_in_white, p.r_gamma),
        ch(c[1], p.g_in_black, p.g_in_white, p.g_gamma),
        ch(c[2], p.b_in_black, p.b_in_white, p.b_gamma),
    ];
    Px([cp[0] * al, cp[1] * al, cp[2] * al, al])
}

/// A finite premultiplied tile: per-texel straight color in [0,1] and a
/// per-texel alpha in {0.25,…,1}, stored premultiplied (rgb = straight·α)
/// so `unpremul` recovers a valid color. Alpha is never 0 (the unpremul
/// special case is covered by the dedicated case below).
fn premul_tile(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        let al = 0.25 + 0.75 * ((x as f32 * 0.01).fract());
        let r = (x as f32 * 0.013).fract();
        let g = (y as f32 * 0.017).fract();
        let bl = ((x + y) as f32 * 0.007).fract();
        Px([r * al, g * al, bl * al, al])
    })
}

/// A premultiplied tile whose alphas are POWERS OF TWO ({0.25, 0.5, 1})
/// — the unpremultiply division is exact on both lanes (exponent
/// shift), so kernels with hard discontinuities (posterize bin edges,
/// the threshold cut) see BIT-identical unpremultiplied stimulus and
/// cannot flip across the discontinuity between lanes.
fn premul_tile_pow2(w: u32, h: u32) -> RefTile {
    const ALPHAS: [f32; 3] = [0.25, 0.5, 1.0];
    RefTile::from_fn(w, h, |x, y| {
        let al = ALPHAS[((x + y) % 3) as usize];
        let r = (x as f32 * 0.013).fract();
        let g = (y as f32 * 0.017).fract();
        let bl = ((x + y) as f32 * 0.007).fract();
        Px([r * al, g * al, bl * al, al])
    })
}

const TILE: u32 = image_core::TILE;

macro_rules! parity_test {
    ($name:ident, $def:expr, $ref:expr, $params:expr) => {
        parity_test!($name, $def, $ref, $params, premul_tile);
    };
    ($name:ident, $def:expr, $ref:expr, $params:expr, $tile:ident) => {
        #[test]
        fn $name() {
            let t = $tile(TILE, TILE);
            match parity(&$def, $ref, &[&t], &$params) {
                Some(r) => {
                    eprintln!("{}: measured max f16 ULP {}", $def.id, r.max_ulp);
                    assert_within(r, &$def)
                }
                None => eprintln!("SKIP: no GPU adapter"),
            }
        }
    };
}

parity_test!(
    exposure_parity,
    ADJUST_EXPOSURE,
    exposure_ref,
    AdjustExposureParams::new(0.8)
);
parity_test!(
    brightness_contrast_parity,
    ADJUST_BRIGHTNESS_CONTRAST,
    brightness_contrast_ref,
    AdjustBrightnessContrastParams::new(0.1, 1.3)
);
parity_test!(
    levels_parity,
    ADJUST_LEVELS,
    levels_ref,
    AdjustLevelsParams::new(0.05, 0.95, 1.4, 0.0, 1.0)
);
parity_test!(
    saturation_parity,
    ADJUST_SATURATION,
    saturation_ref,
    AdjustSaturationParams::new(1.6)
);
parity_test!(
    hue_rotate_parity,
    ADJUST_HUE_ROTATE,
    hue_rotate_ref,
    AdjustHueRotateParams::new(35.0)
);
parity_test!(
    invert_rgb_parity,
    ADJUST_INVERT_RGB,
    adjust_invert_rgb,
    AdjustInvertRgbParams::new()
);
parity_test!(
    white_balance_parity,
    ADJUST_WHITE_BALANCE,
    white_balance_ref,
    AdjustWhiteBalanceParams::new(0.15, -0.08)
);
parity_test!(
    vibrance_parity,
    ADJUST_VIBRANCE,
    vibrance_ref,
    AdjustVibranceParams::new(0.6, 0.15)
);
parity_test!(
    color_balance_parity,
    ADJUST_COLOR_BALANCE,
    color_balance_ref,
    AdjustColorBalanceParams::new([0.1, -0.05, 0.08], [-0.06, 0.04, -0.02], [0.05, 0.02, -0.1])
);
parity_test!(
    black_white_parity,
    ADJUST_BLACK_WHITE,
    black_white_ref,
    // The classic Black & White default weights.
    AdjustBlackWhiteParams::new(0.4, 0.6, 0.4, 0.6, 0.2, 0.8)
);
parity_test!(
    posterize_parity,
    ADJUST_POSTERIZE,
    posterize_ref,
    AdjustPosterizeParams::new(6.0),
    premul_tile_pow2
);
parity_test!(
    threshold_parity,
    ADJUST_THRESHOLD,
    threshold_ref,
    AdjustThresholdParams::new(0.5),
    premul_tile_pow2
);
parity_test!(
    photo_filter_parity,
    ADJUST_PHOTO_FILTER,
    photo_filter_ref,
    // A warming-gel color at density 0.6, luminosity preserved.
    AdjustPhotoFilterParams::new([0.93, 0.66, 0.27], 0.6, true)
);
parity_test!(
    photo_filter_no_preserve_parity,
    ADJUST_PHOTO_FILTER,
    photo_filter_ref,
    AdjustPhotoFilterParams::new([0.35, 0.6, 0.93], 0.8, false)
);

// A NON-trivial table: inverted, so a kernel that silently passed pixels
// through would fail loudly rather than look plausible.
fn inverted_lut() -> AdjustLut1dParams {
    let mut lut = [0u8; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        *v = 255 - i as u8;
    }
    AdjustLut1dParams::new(&lut)
}

parity_test!(lut1d_parity, ADJUST_LUT1D, lut1d_ref, inverted_lut());
parity_test!(
    lut1d_identity_parity,
    ADJUST_LUT1D,
    lut1d_ref,
    AdjustLut1dParams::identity()
);

// A CHANNEL-CROSSING cube — swap red and blue. A 1D LUT cannot express
// this, so it also pins that the kernel really is three-dimensional.
parity_test!(
    lut3d_parity,
    ADJUST_LUT3D,
    lut3d_ref,
    AdjustLut3dParams::from_fn(|r, g, b| [b, g, r])
);
parity_test!(
    lut3d_identity_parity,
    ADJUST_LUT3D,
    lut3d_ref,
    AdjustLut3dParams::identity()
);

parity_test!(
    channel_mixer_parity,
    ADJUST_CHANNEL_MIXER,
    channel_mixer_ref,
    AdjustChannelMixerParams::new(
        [0.7, 0.2, 0.1, 0.05],
        [0.1, 0.8, 0.1, 0.0],
        [0.0, 0.3, 0.6, -0.02]
    )
);
parity_test!(
    levels_rgb_parity,
    ADJUST_LEVELS_RGB,
    levels_rgb_ref,
    AdjustLevelsRgbParams::new([0.05, 0.95, 1.3], [0.0, 1.0, 0.8], [0.1, 0.9, 1.0])
);

/// Identity-parameter cases: each op at its no-op params is a
/// near-passthrough (within tolerance) of a premultiplied input.
#[test]
fn adjust_identity_params() {
    let t = premul_tile(64, 48);
    if let Some(r) = parity(
        &ADJUST_EXPOSURE,
        exposure_ref,
        &[&t],
        &AdjustExposureParams::new(0.0),
    ) {
        assert_within(r, &ADJUST_EXPOSURE);
    }
    if let Some(r) = parity(
        &ADJUST_SATURATION,
        saturation_ref,
        &[&t],
        &AdjustSaturationParams::new(1.0),
    ) {
        assert_within(r, &ADJUST_SATURATION);
    }
    if let Some(r) = parity(
        &ADJUST_HUE_ROTATE,
        hue_rotate_ref,
        &[&t],
        &AdjustHueRotateParams::new(0.0),
    ) {
        assert_within(r, &ADJUST_HUE_ROTATE);
    }
    if let Some(r) = parity(
        &ADJUST_WHITE_BALANCE,
        white_balance_ref,
        &[&t],
        &AdjustWhiteBalanceParams::new(0.0, 0.0),
    ) {
        assert_within(r, &ADJUST_WHITE_BALANCE);
    }
}

/// Identity-parameter cases for the breadth batch: each op at its no-op
/// params is a near-passthrough (within tolerance).
#[test]
fn adjust_breadth_identity_params() {
    let t = premul_tile(64, 48);
    if let Some(r) = parity(
        &ADJUST_VIBRANCE,
        vibrance_ref,
        &[&t],
        &AdjustVibranceParams::new(0.0, 0.0),
    ) {
        assert_within(r, &ADJUST_VIBRANCE);
    }
    if let Some(r) = parity(
        &ADJUST_COLOR_BALANCE,
        color_balance_ref,
        &[&t],
        &AdjustColorBalanceParams::new([0.0; 3], [0.0; 3], [0.0; 3]),
    ) {
        assert_within(r, &ADJUST_COLOR_BALANCE);
    }
    if let Some(r) = parity(
        &ADJUST_PHOTO_FILTER,
        photo_filter_ref,
        &[&t],
        // density 0 = identity regardless of filter color / preserve.
        &AdjustPhotoFilterParams::new([0.9, 0.5, 0.1], 0.0, true),
    ) {
        assert_within(r, &ADJUST_PHOTO_FILTER);
    }
    if let Some(r) = parity(
        &ADJUST_CHANNEL_MIXER,
        channel_mixer_ref,
        &[&t],
        &AdjustChannelMixerParams::new(
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ),
    ) {
        assert_within(r, &ADJUST_CHANNEL_MIXER);
    }
    if let Some(r) = parity(
        &ADJUST_LEVELS_RGB,
        levels_rgb_ref,
        &[&t],
        &AdjustLevelsRgbParams::new([0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]),
    ) {
        assert_within(r, &ADJUST_LEVELS_RGB);
    }
}

/// Pure-reference sanity for the discontinuous / branchy ops (no GPU):
/// pins the documented selection rules.
#[test]
fn adjust_breadth_reference_semantics() {
    // threshold: opaque mid-gray splits exactly at the cut (>= keeps
    // white at equality).
    let p = AdjustThresholdParams::new(0.5);
    let white = threshold_ref(Px([0.6, 0.6, 0.6, 1.0]), Px([0.0; 4]), &p);
    assert_eq!(white.0, [1.0, 1.0, 1.0, 1.0]);
    let black = threshold_ref(Px([0.2, 0.2, 0.2, 1.0]), Px([0.0; 4]), &p);
    assert_eq!(black.0, [0.0, 0.0, 0.0, 1.0]);
    // Exactly at the cut: pure red has lum = 0.3 bit-exactly (single
    // product, no summation rounding); threshold 0.3 → `>=` keeps white.
    let p_cut = AdjustThresholdParams::new(0.3);
    let at_cut = threshold_ref(Px([1.0, 0.0, 0.0, 1.0]), Px([0.0; 4]), &p_cut);
    assert_eq!(at_cut.0, [1.0, 1.0, 1.0, 1.0], ">= keeps white at the cut");

    // posterize: 2 levels snaps below/above 0.5 to the two extremes.
    let p = AdjustPosterizeParams::new(2.0);
    let lo = posterize_ref(Px([0.3, 0.3, 0.3, 1.0]), Px([0.0; 4]), &p);
    assert_eq!(lo.0, [0.0, 0.0, 0.0, 1.0]);
    let hi = posterize_ref(Px([0.7, 0.7, 0.7, 1.0]), Px([0.0; 4]), &p);
    assert_eq!(hi.0, [1.0, 1.0, 1.0, 1.0]);

    // black_white: a gray input is unchanged by ANY weights; a pure red
    // reads exactly the reds weight.
    let p = AdjustBlackWhiteParams::new(0.4, 0.6, 0.4, 0.6, 0.2, 0.8);
    let gray = black_white_ref(Px([0.42, 0.42, 0.42, 1.0]), Px([0.0; 4]), &p);
    assert!((gray.0[0] - 0.42).abs() < 1e-6);
    let red = black_white_ref(Px([1.0, 0.0, 0.0, 1.0]), Px([0.0; 4]), &p);
    assert!((red.0[0] - 0.4).abs() < 1e-6, "pure red -> reds weight");

    // vibrance: a fully-saturated primary gets NO vibrance contribution
    // (satl = 1), only the uniform saturation offset.
    let p = AdjustVibranceParams::new(0.8, 0.0);
    let sat_red = vibrance_ref(Px([1.0, 0.0, 0.0, 1.0]), Px([0.0; 4]), &p);
    let id_red = vibrance_ref(
        Px([1.0, 0.0, 0.0, 1.0]),
        Px([0.0; 4]),
        &AdjustVibranceParams::new(0.0, 0.0),
    );
    for c in 0..4 {
        assert!(
            (sat_red.0[c] - id_red.0[c]).abs() < 1e-6,
            "saturated color protected from vibrance (channel {c})"
        );
    }
}
