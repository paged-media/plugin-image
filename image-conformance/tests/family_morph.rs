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

//! gpu↔ref parity + behaviour proofs for the morphology + rank family
//! (T3, spec §11). `morph.dilate`/`morph.erode` are per-channel 3×3
//! max/min; `rank.median3` is the per-channel 3×3 median via a fixed
//! 19-comparator selection network mirrored exactly here. All three are
//! windowed (radius (1, 1)) and EXACT — they reproduce existing f16
//! samples bit-for-bit. feat: morph.dilate, morph.erode, rank.median3
//! (registry/kernels.yaml).

use image_conformance::harness::{assert_within, parity_windowed, RefTile};
use image_conformance::Px;
use image_kernels::families::morph::{MorphParams, MORPH_DILATE, MORPH_ERODE, RANK_MEDIAN3};
use image_kernels::KernelClass;

// ───────────────────────── scalar references ───────────────────────
//
// All three references read the SAME 3×3 window the shaders read:
// output texel (ox, oy) ↔ window center (ox + 1, oy + 1), taps in
// raster order (dy outer ascending, dx inner ascending):
//
//     s0 s1 s2
//     s3 s4 s5
//     s6 s7 s8

/// The 9 window taps for output (ox, oy), raster order, over the
/// quantized window.
fn taps(win: &[Px], win_w: u32, ox: u32, oy: u32) -> [Px; 9] {
    // Window center in `in0` coords.
    let cx = ox + 1;
    let cy = oy + 1;
    let mut s = [Px([0.0; 4]); 9];
    let mut k = 0usize;
    for dy in [-1i32, 0, 1] {
        for dx in [-1i32, 0, 1] {
            let sx = (cx as i32 + dx) as u32;
            let sy = (cy as i32 + dy) as u32;
            s[k] = win[(sy * win_w + sx) as usize];
            k += 1;
        }
    }
    s
}

/// Per-channel componentwise compare-exchange: (a, b) := (min, max).
/// `f32::min`/`f32::max` mirror the WGSL `min`/`max` builtins exactly.
fn op2(a: &mut Px, b: &mut Px) {
    let lo = a.zip(*b, f32::min);
    let hi = a.zip(*b, f32::max);
    *a = lo;
    *b = hi;
}

/// dilate reference: per-channel MAX of the 9 taps, folded in raster
/// order (the shader seeds with s0 then maxes s0..s8 dy/dx ascending).
fn dilate_ref(win: &[Px], win_w: u32, _win_h: u32, ox: u32, oy: u32, _p: &MorphParams) -> Px {
    let s = taps(win, win_w, ox, oy);
    let mut acc = s[0];
    for t in &s {
        acc = acc.zip(*t, f32::max);
    }
    acc
}

/// erode reference: per-channel MIN of the 9 taps, same fold order.
fn erode_ref(win: &[Px], win_w: u32, _win_h: u32, ox: u32, oy: u32, _p: &MorphParams) -> Px {
    let s = taps(win, win_w, ox, oy);
    let mut acc = s[0];
    for t in &s {
        acc = acc.zip(*t, f32::min);
    }
    acc
}

/// median3 reference: per-channel median of the 9 taps via the SAME
/// fixed 19-comparator selection network the shader runs, step for step
/// (s4 holds the median after the final compare-exchange).
fn median3_ref(win: &[Px], win_w: u32, _win_h: u32, ox: u32, oy: u32, _p: &MorphParams) -> Px {
    let mut s = taps(win, win_w, ox, oy);
    // Network identical to MEDIAN3_WGSL.
    const NET: &[(usize, usize)] = &[
        (1, 2),
        (4, 5),
        (7, 8),
        (0, 1),
        (3, 4),
        (6, 7),
        (1, 2),
        (4, 5),
        (7, 8),
        (0, 3),
        (5, 8),
        (4, 7),
        (3, 6),
        (1, 4),
        (2, 5),
        (4, 7),
        (4, 2),
        (6, 4),
        (4, 2),
    ];
    for &(i, j) in NET {
        let mut a = s[i];
        let mut b = s[j];
        op2(&mut a, &mut b);
        s[i] = a;
        s[j] = b;
    }
    s[4]
}

// ───────────────────────── window stimulus ─────────────────────────

/// A finite analytic window (FINITE stimulus rule, harness docs). For a
/// `out_w × out_h` output the windowed kernels need `in0` of size
/// `(out_w + 2, out_h + 2)` (radius (1, 1)).
fn window(out_w: u32, out_h: u32) -> RefTile {
    let (rx, ry) = match MORPH_DILATE.class {
        KernelClass::Windowed { radius } => radius,
        _ => unreachable!(),
    };
    RefTile::from_fn(out_w + 2 * rx as u32, out_h + 2 * ry as u32, |x, y| {
        Px([
            (x as f32 * 0.013).fract(),
            (y as f32 * 0.027).fract(),
            ((x + 3 * y) as f32 * 0.007).fract(),
            1.0,
        ])
    })
}

// ─────────────────────────── parity tests ──────────────────────────

#[test]
fn morph_dilate_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    let win = window(w, h);
    match parity_windowed(&MORPH_DILATE, dilate_ref, &win, w, h, &MorphParams::new()) {
        Some(r) => assert_within(r, &MORPH_DILATE),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn morph_dilate_parity_small() {
    let win = window(64, 48);
    match parity_windowed(&MORPH_DILATE, dilate_ref, &win, 64, 48, &MorphParams::new()) {
        Some(r) => assert_within(r, &MORPH_DILATE),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn morph_erode_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    let win = window(w, h);
    match parity_windowed(&MORPH_ERODE, erode_ref, &win, w, h, &MorphParams::new()) {
        Some(r) => assert_within(r, &MORPH_ERODE),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn morph_erode_parity_small() {
    let win = window(64, 48);
    match parity_windowed(&MORPH_ERODE, erode_ref, &win, 64, 48, &MorphParams::new()) {
        Some(r) => assert_within(r, &MORPH_ERODE),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn rank_median3_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    let win = window(w, h);
    match parity_windowed(&RANK_MEDIAN3, median3_ref, &win, w, h, &MorphParams::new()) {
        Some(r) => assert_within(r, &RANK_MEDIAN3),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn rank_median3_parity_small() {
    let win = window(64, 48);
    match parity_windowed(
        &RANK_MEDIAN3,
        median3_ref,
        &win,
        64,
        48,
        &MorphParams::new(),
    ) {
        Some(r) => assert_within(r, &RANK_MEDIAN3),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

// ───────────────────────── behaviour proofs ────────────────────────
//
// These assert the *meaning* of each kernel against the scalar
// reference over a hand-labelled window, independent of the GPU (they
// run with or without an adapter). They build a finite window, run the
// reference for the labelled output texels, and check the morphological
// behaviour: a bright impulse spreads under dilate; a dark impulse
// spreads under erode; the median removes an isolated impulse and is a
// no-op on a flat field.

const FLAT: f32 = 0.5;

/// A flat field of `FLAT` with a single texel at window position
/// `(px, py)` (in `in0` coords) set to `val` across all channels.
fn labelled_window(w: u32, h: u32, px: u32, py: u32, val: f32) -> Vec<Px> {
    let mut v = vec![Px([FLAT, FLAT, FLAT, 1.0]); (w * h) as usize];
    v[(py * w + px) as usize] = Px([val, val, val, 1.0]);
    v
}

#[test]
fn morph_dilate_impulse_spreads_to_8_neighbours() {
    // 5×5 output ⇒ 7×7 window (radius 1). Put a BRIGHT texel at window
    // center (3, 3). Under dilate (per-channel MAX over the 3×3 window),
    // every output whose 3×3 window contains the bright texel becomes
    // bright — i.e. the bright pixel and its 8 neighbours in OUTPUT
    // space. Output (ox, oy) reads window center (ox+1, oy+1); the
    // bright window texel (3, 3) is the center of output (2, 2). So
    // outputs (1..=3, 1..=3) (the 3×3 block around (2,2)) go bright; the
    // rest stay FLAT.
    let (ow, oh) = (5u32, 5u32);
    let (ww, wh) = (ow + 2, oh + 2);
    let bright = 0.9f32;
    let win = labelled_window(ww, wh, 3, 3, bright);

    for oy in 0..oh {
        for ox in 0..ow {
            let got = dilate_ref(&win, ww, wh, ox, oy, &MorphParams::new());
            let in_spread = (1..=3).contains(&ox) && (1..=3).contains(&oy);
            let want = if in_spread { bright } else { FLAT };
            assert_eq!(
                got.0[0], want,
                "dilate output ({ox},{oy}): bright impulse must spread to the 8 neighbours of (2,2)"
            );
        }
    }
}

#[test]
fn morph_erode_impulse_spreads_to_8_neighbours() {
    // Dual of dilate: a DARK texel at window center (3, 3) spreads to
    // the 3×3 output block around (2, 2) under per-channel MIN.
    let (ow, oh) = (5u32, 5u32);
    let (ww, wh) = (ow + 2, oh + 2);
    let dark = 0.1f32;
    let win = labelled_window(ww, wh, 3, 3, dark);

    for oy in 0..oh {
        for ox in 0..ow {
            let got = erode_ref(&win, ww, wh, ox, oy, &MorphParams::new());
            let in_spread = (1..=3).contains(&ox) && (1..=3).contains(&oy);
            let want = if in_spread { dark } else { FLAT };
            assert_eq!(
                got.0[0], want,
                "erode output ({ox},{oy}): dark impulse must spread to the 8 neighbours of (2,2)"
            );
        }
    }
}

#[test]
fn rank_median3_removes_isolated_impulse() {
    // Salt-and-pepper: a single bright outlier in a flat field. The
    // median of the 9 window samples (8 FLAT + 1 outlier) is FLAT, so
    // the impulse is removed AT its own location AND everywhere it
    // appears in a window. The whole output must be flat.
    let (ow, oh) = (5u32, 5u32);
    let (ww, wh) = (ow + 2, oh + 2);
    // Outlier at the center output texel's window center (3, 3) — and
    // also a dark pepper to exercise the low tail.
    let salt = labelled_window(ww, wh, 3, 3, 0.95);
    let pepper = labelled_window(ww, wh, 3, 3, 0.02);

    for win in [&salt, &pepper] {
        for oy in 0..oh {
            for ox in 0..ow {
                let got = median3_ref(win, ww, wh, ox, oy, &MorphParams::new());
                assert_eq!(
                    got.0[0], FLAT,
                    "median3 output ({ox},{oy}): an isolated impulse must be removed"
                );
            }
        }
    }
}

#[test]
fn rank_median3_noop_on_constant_field() {
    // A constant field is a fixed point of the median (and of dilate /
    // erode): every tap equals the field value, so the selected median
    // is that value.
    let (ow, oh) = (8u32, 6u32);
    let (ww, wh) = (ow + 2, oh + 2);
    let win = vec![Px([FLAT, FLAT, FLAT, 1.0]); (ww * wh) as usize];
    for oy in 0..oh {
        for ox in 0..ow {
            let got = median3_ref(&win, ww, wh, ox, oy, &MorphParams::new());
            assert_eq!(
                got,
                Px([FLAT, FLAT, FLAT, 1.0]),
                "median3 must be a no-op on a constant field"
            );
        }
    }
}
