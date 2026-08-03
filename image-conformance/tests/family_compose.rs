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

//! gpu↔ref parity for the compose family (T1, spec §8.4) — every kernel
//! is a binary point module that composites a premultiplied SOURCE layer
//! over a premultiplied BACKDROP under W3C Compositing and Blending
//! Level 1. The scalar golden is `compose_ref::composite` (the SAME fn
//! the PSD merged-composite oracle consumes); the GPU runs the
//! handwritten WGSL module through the `parity()` lane (constant-1 mask,
//! so the module's `mix(a, result, 1)` returns the composite directly).
//!
//! feat: compose.normal / multiply / screen / overlay / darken /
//! lighten / color_dodge / color_burn / hard_light / soft_light /
//! difference / exclusion / hue / saturation / color / luminosity,
//! plus the extended Photoshop set: compose.linear_burn / linear_dodge
//! / darker_color / lighter_color / vivid_light / linear_light /
//! pin_light / hard_mix / subtract / divide (registry/kernels.yaml).
//!
//! DETERMINISM NOTE for the discontinuous extended modes (hard_mix's
//! 0/1 snap, darker/lighter_color's whole-triple select, divide's
//! Cs = 0 guard): the stimulus alphas {0, 0.25, 0.5, 1} and opacities
//! {0.5, 1} are all powers of two, so the opacity fold and the
//! unpremultiply are EXACT on both lanes — the compared quantities are
//! bit-identical and the discontinuity cannot flip between lanes.
//!
//! Stimulus (FINITE, per the harness rule): premultiplied texels whose
//! alphas walk {0, 0.25, 0.5, 1} and whose unpremultiplied colors sweep
//! [0,1] — including the 0/1 extremes that exercise the dodge/burn
//! guards, exact `a == b` texels, and the §10.3 non-separable sorts.

use image_conformance::compose_ref::{composite, Blend};
use image_conformance::harness::{assert_within, parity, RefTile};
use image_conformance::Px;
use image_kernels::families::compose::{
    ComposeParams, COMPOSE_COLOR, COMPOSE_COLOR_BURN, COMPOSE_COLOR_DODGE, COMPOSE_DARKEN,
    COMPOSE_DARKER_COLOR, COMPOSE_DIFFERENCE, COMPOSE_DIVIDE, COMPOSE_EXCLUSION,
    COMPOSE_HARD_LIGHT, COMPOSE_HARD_MIX, COMPOSE_HUE, COMPOSE_LIGHTEN, COMPOSE_LIGHTER_COLOR,
    COMPOSE_LINEAR_BURN, COMPOSE_LINEAR_DODGE, COMPOSE_LINEAR_LIGHT, COMPOSE_LUMINOSITY,
    COMPOSE_MULTIPLY, COMPOSE_NORMAL, COMPOSE_OVERLAY, COMPOSE_PIN_LIGHT, COMPOSE_SATURATION,
    COMPOSE_SCREEN, COMPOSE_SOFT_LIGHT, COMPOSE_SUBTRACT, COMPOSE_VIVID_LIGHT,
};
use image_kernels::KernelDef;

const W: u32 = image_core::TILE;
const H: u32 = image_core::TILE;

/// The four sample alphas the stimulus walks.
const ALPHAS: [f32; 4] = [0.0, 0.25, 0.5, 1.0];

/// Premultiply an unpremultiplied (color, alpha) — the inputs are
/// premultiplied (spec §8.4), so the stimulus builds them that way.
fn premul(color: [f32; 3], alpha: f32) -> Px {
    Px([color[0] * alpha, color[1] * alpha, color[2] * alpha, alpha])
}

/// Backdrop tile `a`: unpremultiplied color sweeps [0,1] over the tile,
/// alpha cycles through {0, 0.25, 0.5, 1} per column. Channel 2 seeds the
/// exact 0.0 and 1.0 columns that exercise the dodge/burn `Cb` guards.
fn backdrop_tile(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        let r = x as f32 / (w - 1).max(1) as f32; // [0,1] inc. exact 0 and 1
        let g = y as f32 / (h - 1).max(1) as f32;
        let b = match x % 3 {
            0 => 0.0,
            1 => 1.0,
            _ => (x + 2 * y) as f32 / (w + 2 * h) as f32,
        };
        let alpha = ALPHAS[(x as usize) % 4];
        premul([r, g, b], alpha)
    })
}

/// Source tile `b`: a different unpremultiplied sweep with alpha cycling
/// on the row index (so the (αs, αb) grid covers the full {0,0.25,0.5,1}²
/// product across the tile). Channel 2 seeds the exact 0.0/1.0 columns
/// for the `Cs` dodge/burn guards.
fn source_tile(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        let r = 1.0 - x as f32 / (w - 1).max(1) as f32;
        let g = (x + y) as f32 / (w + h) as f32;
        let b = match y % 3 {
            0 => 1.0,
            1 => 0.0,
            _ => (2 * x + y) as f32 / (2 * w + h) as f32,
        };
        let alpha = ALPHAS[(y as usize) % 4];
        premul([r, g, b], alpha)
    })
}

/// An `a == b` tile (identical backdrop and source) — exercises the
/// self-blend identities (e.g. difference→0, multiply→Cs², the §10.3
/// fixed-point cases) on both lanes.
fn equal_tile(w: u32, h: u32) -> RefTile {
    backdrop_tile(w, h)
}

/// Run one compose kernel at both opacities {0.5, 1.0} over a stimulus
/// pair, asserting gpu↔ref parity within the kernel's tolerance.
fn check(def: &'static KernelDef, blend: Blend, a: &RefTile, b: &RefTile) {
    for opacity in [0.5f32, 1.0f32] {
        let p = ComposeParams::new(opacity);
        let reference = move |a: Px, b: Px, p: &ComposeParams| composite(a, b, p.opacity, blend);
        match parity(def, reference, &[a, b], &p) {
            Some(r) => {
                eprintln!(
                    "{} (opacity {opacity}): measured max f16 ULP {}",
                    def.id, r.max_ulp
                );
                assert_within(r, def)
            }
            None => {
                eprintln!("SKIP: no GPU adapter");
                return;
            }
        }
    }
}

/// Drive a kernel over the full backdrop×source grid AND the a==b tile.
fn check_all(def: &'static KernelDef, blend: Blend) {
    let a = backdrop_tile(W, H);
    let b = source_tile(W, H);
    check(def, blend, &a, &b);
    let eq = equal_tile(W, H);
    check(def, blend, &eq, &eq);
}

#[test]
fn compose_normal_parity() {
    check_all(&COMPOSE_NORMAL, Blend::Normal);
}

#[test]
fn compose_multiply_parity() {
    check_all(&COMPOSE_MULTIPLY, Blend::Multiply);
}

#[test]
fn compose_screen_parity() {
    check_all(&COMPOSE_SCREEN, Blend::Screen);
}

#[test]
fn compose_overlay_parity() {
    check_all(&COMPOSE_OVERLAY, Blend::Overlay);
}

#[test]
fn compose_darken_parity() {
    check_all(&COMPOSE_DARKEN, Blend::Darken);
}

#[test]
fn compose_lighten_parity() {
    check_all(&COMPOSE_LIGHTEN, Blend::Lighten);
}

#[test]
fn compose_color_dodge_parity() {
    check_all(&COMPOSE_COLOR_DODGE, Blend::ColorDodge);
}

#[test]
fn compose_color_burn_parity() {
    check_all(&COMPOSE_COLOR_BURN, Blend::ColorBurn);
}

#[test]
fn compose_hard_light_parity() {
    check_all(&COMPOSE_HARD_LIGHT, Blend::HardLight);
}

#[test]
fn compose_soft_light_parity() {
    check_all(&COMPOSE_SOFT_LIGHT, Blend::SoftLight);
}

#[test]
fn compose_difference_parity() {
    check_all(&COMPOSE_DIFFERENCE, Blend::Difference);
}

#[test]
fn compose_exclusion_parity() {
    check_all(&COMPOSE_EXCLUSION, Blend::Exclusion);
}

#[test]
fn compose_hue_parity() {
    check_all(&COMPOSE_HUE, Blend::Hue);
}

#[test]
fn compose_saturation_parity() {
    check_all(&COMPOSE_SATURATION, Blend::Saturation);
}

#[test]
fn compose_color_parity() {
    check_all(&COMPOSE_COLOR, Blend::Color);
}

#[test]
fn compose_luminosity_parity() {
    check_all(&COMPOSE_LUMINOSITY, Blend::Luminosity);
}

#[test]
fn compose_linear_burn_parity() {
    check_all(&COMPOSE_LINEAR_BURN, Blend::LinearBurn);
}

#[test]
fn compose_linear_dodge_parity() {
    check_all(&COMPOSE_LINEAR_DODGE, Blend::LinearDodge);
}

#[test]
fn compose_darker_color_parity() {
    check_all(&COMPOSE_DARKER_COLOR, Blend::DarkerColor);
}

#[test]
fn compose_lighter_color_parity() {
    check_all(&COMPOSE_LIGHTER_COLOR, Blend::LighterColor);
}

#[test]
fn compose_vivid_light_parity() {
    check_all(&COMPOSE_VIVID_LIGHT, Blend::VividLight);
}

#[test]
fn compose_linear_light_parity() {
    check_all(&COMPOSE_LINEAR_LIGHT, Blend::LinearLight);
}

#[test]
fn compose_pin_light_parity() {
    check_all(&COMPOSE_PIN_LIGHT, Blend::PinLight);
}

#[test]
fn compose_hard_mix_parity() {
    check_all(&COMPOSE_HARD_MIX, Blend::HardMix);
}

#[test]
fn compose_subtract_parity() {
    check_all(&COMPOSE_SUBTRACT, Blend::Subtract);
}

#[test]
fn compose_divide_parity() {
    check_all(&COMPOSE_DIVIDE, Blend::Divide);
}

/// Blend-level sanity for the extended set (no GPU): pins the
/// documented formulas at hand-checked points.
#[test]
fn compose_extended_blend_semantics() {
    use image_conformance::compose_ref::{
        b_darker_color, b_divide, b_hard_mix, b_lighter_color, b_linear_burn, b_linear_dodge,
        b_linear_light, b_pin_light, b_subtract, b_vivid_light,
    };
    // linear burn / dodge clamp at the extremes.
    assert_eq!(b_linear_burn(0.3, 0.4), 0.0);
    assert!((b_linear_burn(0.8, 0.7) - 0.5).abs() < 1e-6);
    assert_eq!(b_linear_dodge(0.8, 0.7), 1.0);
    assert!((b_linear_dodge(0.3, 0.4) - 0.7).abs() < 1e-6);
    // vivid light is continuous at Cs = 0.5 (both arms give Cb).
    assert!((b_vivid_light(0.37, 0.5) - 0.37).abs() < 1e-6);
    // linear light collapses to clamp(Cb + 2Cs - 1, 0, 1).
    assert!((b_linear_light(0.5, 0.6) - 0.7).abs() < 1e-6);
    // pin light: darken arm below 0.5, lighten arm above.
    assert!((b_pin_light(0.9, 0.2) - 0.4).abs() < 1e-6);
    assert!((b_pin_light(0.1, 0.9) - 0.8).abs() < 1e-6);
    // hard mix snaps to 0/1; the sum-at-1 tie goes to 1.
    assert_eq!(b_hard_mix(0.4, 0.5), 0.0);
    assert_eq!(b_hard_mix(0.5, 0.5), 1.0);
    // subtract floors at 0.
    assert_eq!(b_subtract(0.2, 0.5), 0.0);
    assert!((b_subtract(0.7, 0.2) - 0.5).abs() < 1e-6);
    // divide: clamp at 1, Cs = 0 -> 1.
    assert!((b_divide(0.25, 0.5) - 0.5).abs() < 1e-6);
    assert_eq!(b_divide(0.5, 0.25), 1.0);
    assert_eq!(b_divide(0.0, 0.0), 1.0);
    // darker/lighter color select the WHOLE triple by Lum; ties keep Cb.
    let dark = [0.1, 0.1, 0.9]; // lum 0.188
    let light = [0.5, 0.9, 0.1]; // lum 0.692
    assert_eq!(b_darker_color(light, dark), dark);
    assert_eq!(b_darker_color(dark, light), dark);
    assert_eq!(b_lighter_color(dark, light), light);
    assert_eq!(b_lighter_color(light, dark), light);
    assert_eq!(b_darker_color(dark, dark), dark, "tie keeps backdrop");
}

/// Sanity (no GPU): an OPAQUE source (αs = 1) fully replaces the
/// backdrop under `normal` — `co = Cs`, `αo = 1` — and a TRANSPARENT
/// source (αs = 0) leaves the backdrop untouched — `(co, αo) = a`.
/// Validates the source-over algebra of `compose_ref::composite`
/// directly (the shared golden / PSD-flatten spine).
#[test]
fn compose_normal_replaces_and_preserves() {
    let backdrop = premul([0.2, 0.4, 0.6], 0.5); // premultiplied (0.1,0.2,0.3,0.5)

    // Opaque source replaces the backdrop entirely.
    let opaque = premul([0.7, 0.1, 0.9], 1.0);
    let out = composite(backdrop, opaque, 1.0, Blend::Normal);
    let want = opaque; // co = Cs (premultiplied), αo = 1
    for c in 0..4 {
        assert!(
            (out.0[c] - want.0[c]).abs() <= 1e-6,
            "opaque-over channel {c}: got {} want {}",
            out.0[c],
            want.0[c]
        );
    }

    // Transparent source leaves the backdrop unchanged.
    let clear = premul([0.7, 0.1, 0.9], 0.0);
    let out = composite(backdrop, clear, 1.0, Blend::Normal);
    for c in 0..4 {
        assert!(
            (out.0[c] - backdrop.0[c]).abs() <= 1e-6,
            "clear-over channel {c}: got {} want {}",
            out.0[c],
            backdrop.0[c]
        );
    }

    // Opacity 0 is equivalent to a transparent source (folds αs → 0).
    let out = composite(backdrop, opaque, 0.0, Blend::Normal);
    for c in 0..4 {
        assert!(
            (out.0[c] - backdrop.0[c]).abs() <= 1e-6,
            "opacity-0 channel {c}: got {} want {}",
            out.0[c],
            backdrop.0[c]
        );
    }
}

/// The PSD blend-key mapping is total and round-trips (the flatten unit
/// joins layers on this fourcc). No GPU.
#[test]
fn compose_psd_key_roundtrip() {
    let all = [
        Blend::Normal,
        Blend::Multiply,
        Blend::Screen,
        Blend::Overlay,
        Blend::Darken,
        Blend::Lighten,
        Blend::ColorDodge,
        Blend::ColorBurn,
        Blend::HardLight,
        Blend::SoftLight,
        Blend::Difference,
        Blend::Exclusion,
        Blend::Hue,
        Blend::Saturation,
        Blend::Color,
        Blend::Luminosity,
        Blend::LinearBurn,
        Blend::LinearDodge,
        Blend::DarkerColor,
        Blend::LighterColor,
        Blend::VividLight,
        Blend::LinearLight,
        Blend::PinLight,
        Blend::HardMix,
        Blend::Subtract,
        Blend::Divide,
    ];
    for blend in all {
        let key = blend.psd_key();
        assert_eq!(key.len(), 4, "PSD blend key {key:?} must be 4 chars");
        assert_eq!(
            Blend::from_psd_key(key),
            Some(blend),
            "psd_key round-trip for {blend:?}"
        );
    }
    assert_eq!(Blend::from_psd_key("zzzz"), None, "unknown key → None");
}
