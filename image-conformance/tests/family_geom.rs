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

//! gpu↔ref parity for the geometry family (T2). Flip / rotate90 / crop
//! are exact integer coordinate remaps: every output texel is ONE source
//! texel copied verbatim ⇒ `Tolerance::Exact` (0 ULP). Each op runs over
//! a finite labeled window where a pixel's value encodes its source
//! coordinate, so the scalar reference (mirroring the index math) and
//! the GPU must agree bit-for-bit after the shared f16 quantization.
//!
//! WINDOW SIZING. The remap moves exact texels, so the source window the
//! GPU consumes equals the SOURCE dims:
//!   - flip_h/flip_v: dims are preserved → window = output dims.
//!   - rotate90_*: a `src_w × src_h` source → `src_h × src_w` output, so
//!     window = (src_w, src_h) and out = (src_h, src_w).
//!   - crop: window = source dims; output = crop region; out-of-window
//!     offsets clamp to the source edge.
//!
//! feat: geom.flip_h, geom.flip_v, geom.rotate90_cw, geom.rotate90_ccw,
//! geom.crop (registry/kernels.yaml).

use image_conformance::harness::{assert_within, parity_windowed, RefTile};
use image_conformance::Px;
use image_kernels::families::geom::{
    CropParams, FlipHParams, FlipVParams, Rotate90Params, GEOM_CROP, GEOM_FLIP_H, GEOM_FLIP_V,
    GEOM_ROTATE90_CCW, GEOM_ROTATE90_CW,
};

/// A labeled source window: each texel encodes its own (x, y) source
/// coordinate (x/256, y/256, (x+y)/512, 1.0) — finite, f16-exact for the
/// dims under test, so a coordinate remap is legible per channel.
fn labeled(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        Px([
            x as f32 / 256.0,
            y as f32 / 256.0,
            (x + y) as f32 / 512.0,
            1.0,
        ])
    })
}

/// Fetch a window texel by integer coordinate (row-major, as the harness
/// lays it out).
fn at(win: &[Px], win_w: u32, sx: i32, sy: i32) -> Px {
    win[(sy as u32 * win_w + sx as u32) as usize]
}

// ───────────────────────────── flip_h ──────────────────────────────

/// `out(x, y) = in(width - 1 - x, y)`.
fn flip_h_ref(win: &[Px], win_w: u32, _win_h: u32, ox: u32, oy: u32, p: &FlipHParams) -> Px {
    at(win, win_w, p.width as i32 - 1 - ox as i32, oy as i32)
}

#[test]
fn geom_flip_h_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    let win = labeled(w, h);
    let p = FlipHParams::new(w);
    match parity_windowed(&GEOM_FLIP_H, flip_h_ref, &win, w, h, &p) {
        Some(r) => assert_within(r, &GEOM_FLIP_H),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn geom_flip_h_parity_small() {
    let (w, h) = (50u32, 37u32);
    let win = labeled(w, h);
    let p = FlipHParams::new(w);
    match parity_windowed(&GEOM_FLIP_H, flip_h_ref, &win, w, h, &p) {
        Some(r) => assert_within(r, &GEOM_FLIP_H),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

// ───────────────────────────── flip_v ──────────────────────────────

/// `out(x, y) = in(x, height - 1 - y)`.
fn flip_v_ref(win: &[Px], win_w: u32, _win_h: u32, ox: u32, oy: u32, p: &FlipVParams) -> Px {
    at(win, win_w, ox as i32, p.height as i32 - 1 - oy as i32)
}

#[test]
fn geom_flip_v_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    let win = labeled(w, h);
    let p = FlipVParams::new(h);
    match parity_windowed(&GEOM_FLIP_V, flip_v_ref, &win, w, h, &p) {
        Some(r) => assert_within(r, &GEOM_FLIP_V),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn geom_flip_v_parity_small() {
    let (w, h) = (50u32, 37u32);
    let win = labeled(w, h);
    let p = FlipVParams::new(h);
    match parity_windowed(&GEOM_FLIP_V, flip_v_ref, &win, w, h, &p) {
        Some(r) => assert_within(r, &GEOM_FLIP_V),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

// ──────────────────────────── rotate90 ─────────────────────────────
//
// Source is (src_w, src_h); output is the transpose (src_h, src_w).
// The window the GPU consumes is the SOURCE.

/// `out(x, y) = in(y, src_h - 1 - x)`.
fn rotate90_cw_ref(
    win: &[Px],
    win_w: u32,
    _win_h: u32,
    ox: u32,
    oy: u32,
    p: &Rotate90Params,
) -> Px {
    at(win, win_w, oy as i32, p.src_h as i32 - 1 - ox as i32)
}

/// `out(x, y) = in(src_w - 1 - y, x)`.
fn rotate90_ccw_ref(
    win: &[Px],
    win_w: u32,
    _win_h: u32,
    ox: u32,
    oy: u32,
    p: &Rotate90Params,
) -> Px {
    at(win, win_w, p.src_w as i32 - 1 - oy as i32, ox as i32)
}

#[test]
fn geom_rotate90_cw_parity_small() {
    // Non-square source to exercise the dim transpose.
    let (src_w, src_h) = (40u32, 28u32);
    let win = labeled(src_w, src_h);
    let p = Rotate90Params::new(src_w, src_h);
    // Output dims are the transpose: (src_h, src_w).
    match parity_windowed(&GEOM_ROTATE90_CW, rotate90_cw_ref, &win, src_h, src_w, &p) {
        Some(r) => assert_within(r, &GEOM_ROTATE90_CW),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn geom_rotate90_ccw_parity_small() {
    let (src_w, src_h) = (40u32, 28u32);
    let win = labeled(src_w, src_h);
    let p = Rotate90Params::new(src_w, src_h);
    match parity_windowed(&GEOM_ROTATE90_CCW, rotate90_ccw_ref, &win, src_h, src_w, &p) {
        Some(r) => assert_within(r, &GEOM_ROTATE90_CCW),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

/// cw then ccw == identity. Rotating a `src_w × src_h` source CW yields
/// a `src_h × src_w` intermediate; rotating that CCW (its src dims are
/// `src_h × src_w`) must recover the original window bit-for-bit. We
/// model the GPU chain entirely through the scalar references (the
/// per-op parity tests above already pin each reference to the GPU), so
/// this proves the algebraic round-trip without a second GPU pass.
#[test]
fn geom_rotate90_cw_then_ccw_identity() {
    let (src_w, src_h) = (40u32, 28u32);
    let src = labeled(src_w, src_h);
    let cw_p = Rotate90Params::new(src_w, src_h);

    // CW: source (src_w, src_h) → intermediate (src_h, src_w).
    let inter = RefTile::from_fn(src_h, src_w, |x, y| {
        rotate90_cw_ref(&src.px, src.w, src.h, x, y, &cw_p)
    });

    // CCW over the intermediate: its source dims are (src_h, src_w) →
    // output (src_w, src_h), which must equal the original source.
    let ccw_p = Rotate90Params::new(src_h, src_w);
    let round = RefTile::from_fn(src_w, src_h, |x, y| {
        rotate90_ccw_ref(&inter.px, inter.w, inter.h, x, y, &ccw_p)
    });

    assert_eq!(round.w, src.w);
    assert_eq!(round.h, src.h);
    assert_eq!(round.px, src.px, "cw∘ccw must be the identity remap");
}

// ────────────────────────────── crop ───────────────────────────────
//
// `out(x, y) = in(x + off_x, y + off_y)` clamp-to-edge. Window = source
// dims; output = crop region. We test an interior crop (offsets stay in
// bounds) and an offset that drives the read off the source edge so the
// clamp (sample replication) is exercised.

/// `out(x, y) = in(clamp(x + off_x, 0, w-1), clamp(y + off_y, 0, h-1))`.
fn crop_ref(win: &[Px], win_w: u32, win_h: u32, ox: u32, oy: u32, p: &CropParams) -> Px {
    let sx = (ox as i32 + p.off_x).clamp(0, win_w as i32 - 1);
    let sy = (oy as i32 + p.off_y).clamp(0, win_h as i32 - 1);
    at(win, win_w, sx, sy)
}

#[test]
fn geom_crop_interior_parity() {
    // Source 64×48; crop a 32×24 interior region at offset (10, 6).
    let (src_w, src_h) = (64u32, 48u32);
    let win = labeled(src_w, src_h);
    let (out_w, out_h) = (32u32, 24u32);
    let p = CropParams::new(10, 6);
    match parity_windowed(&GEOM_CROP, crop_ref, &win, out_w, out_h, &p) {
        Some(r) => assert_within(r, &GEOM_CROP),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn geom_crop_negative_offset_clamps() {
    // Negative offset reads past the top-left source edge → clamp
    // replicates the edge texel (the clamp-to-edge rule).
    let (src_w, src_h) = (64u32, 48u32);
    let win = labeled(src_w, src_h);
    let (out_w, out_h) = (40u32, 30u32);
    let p = CropParams::new(-8, -5);
    match parity_windowed(&GEOM_CROP, crop_ref, &win, out_w, out_h, &p) {
        Some(r) => assert_within(r, &GEOM_CROP),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}
