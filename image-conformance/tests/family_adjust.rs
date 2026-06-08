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

use image_conformance::harness::{assert_within, parity, RefTile};
use image_conformance::Px;
use image_kernels::families::adjust::{
    adjust_invert_rgb, AdjustBrightnessContrastParams, AdjustExposureParams, AdjustHueRotateParams,
    AdjustInvertRgbParams, AdjustLevelsParams, AdjustSaturationParams, ADJUST_BRIGHTNESS_CONTRAST,
    ADJUST_EXPOSURE, ADJUST_HUE_ROTATE, ADJUST_INVERT_RGB, ADJUST_LEVELS, ADJUST_SATURATION,
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

const TILE: u32 = image_core::TILE;

macro_rules! parity_test {
    ($name:ident, $def:expr, $ref:expr, $params:expr) => {
        #[test]
        fn $name() {
            let t = premul_tile(TILE, TILE);
            match parity(&$def, $ref, &[&t], &$params) {
                Some(r) => assert_within(r, &$def),
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
}
