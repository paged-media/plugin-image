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

//! gpu↔ref parity for the minmax family (T0, spec §11): `math.min` /
//! `math.max` / `math.clamp` / `math.min_const` / `math.max_const`.
//! feat: math.min / math.max / math.clamp / math.min_const /
//! math.max_const (registry/kernels.yaml).
//!
//! Selection ops pick an existing input value per channel, so the
//! stimulus deliberately makes `a` and `b` cross over (including exact
//! ties where `a == b`) and straddles the constant thresholds — that is
//! where a buggy `min`/`max`/`clamp` would diverge, not on smooth runs.

use image_conformance::harness::{assert_within, parity, RefTile};
use image_conformance::Px;
use image_kernels::families::minmax::{
    math_clamp, math_max, math_max_const, math_min, math_min_const, MathClampParams,
    MathMaxConstParams, MathMaxParams, MathMinConstParams, MathMinParams, MATH_CLAMP, MATH_MAX,
    MATH_MAX_CONST, MATH_MIN, MATH_MIN_CONST,
};

/// `a` ramps low→high across x; per-texel it sits below, at, and above
/// the constant thresholds (0.5 / 0.25 / 0.75) the const + clamp tests
/// use, so each kernel sees both sides of every cutoff.
fn ramp_a(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        let t = x as f32 / (w - 1).max(1) as f32; // 0.0 ..= 1.0
        Px([t, 1.0 - t, (t + y as f32 / (h - 1).max(1) as f32) * 0.5, t])
    })
}

/// `b` ramps the opposite way so `a` and `b` cross at the tile center,
/// and the diagonal forces exact `a == b` ties on a full column.
fn ramp_b(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        let t = x as f32 / (w - 1).max(1) as f32; // 0.0 ..= 1.0
        Px([
            1.0 - t,                                      // crosses ramp_a.r at the center
            1.0 - t, // exactly equal to ramp_a.g → tie everywhere
            (t + y as f32 / (h - 1).max(1) as f32) * 0.5, // ties ramp_a.b
            0.5,
        ])
    })
}

#[test]
fn min_parity() {
    let a = ramp_a(image_core::TILE, image_core::TILE);
    let b = ramp_b(image_core::TILE, image_core::TILE);
    let p = MathMinParams::new();
    match parity(&MATH_MIN, math_min, &[&a, &b], &p) {
        Some(r) => assert_within(r, &MATH_MIN),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn max_parity() {
    let a = ramp_a(image_core::TILE, image_core::TILE);
    let b = ramp_b(image_core::TILE, image_core::TILE);
    let p = MathMaxParams::new();
    match parity(&MATH_MAX, math_max, &[&a, &b], &p) {
        Some(r) => assert_within(r, &MATH_MAX),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn clamp_parity() {
    // Window straddled by the ramp: texels fall below lo, inside, above hi.
    let a = ramp_a(image_core::TILE, image_core::TILE);
    let p = MathClampParams::new(0.25, 0.75);
    match parity(&MATH_CLAMP, math_clamp, &[&a], &p) {
        Some(r) => assert_within(r, &MATH_CLAMP),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn min_const_parity() {
    let a = ramp_a(image_core::TILE, image_core::TILE);
    let p = MathMinConstParams::new(0.5);
    match parity(&MATH_MIN_CONST, math_min_const, &[&a], &p) {
        Some(r) => assert_within(r, &MATH_MIN_CONST),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn max_const_parity() {
    let a = ramp_a(image_core::TILE, image_core::TILE);
    let p = MathMaxConstParams::new(0.5);
    match parity(&MATH_MAX_CONST, math_max_const, &[&a], &p) {
        Some(r) => assert_within(r, &MATH_MAX_CONST),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

/// Small quick case exercising the exact-tie / cutoff behavior on a
/// 64×64 tile (cheaper to read in failure output than the full TILE).
#[test]
fn min_max_const_quick() {
    let a = ramp_a(64, 64);
    let b = ramp_b(64, 64);
    if let Some(r) = parity(&MATH_MIN, math_min, &[&a, &b], &MathMinParams::new()) {
        assert_within(r, &MATH_MIN);
    }
    if let Some(r) = parity(&MATH_MAX, math_max, &[&a, &b], &MathMaxParams::new()) {
        assert_within(r, &MATH_MAX);
    }
    if let Some(r) = parity(
        &MATH_CLAMP,
        math_clamp,
        &[&a],
        &MathClampParams::new(0.25, 0.75),
    ) {
        assert_within(r, &MATH_CLAMP);
    }
}
