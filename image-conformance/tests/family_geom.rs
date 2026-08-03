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
//! geom.crop, geom.rotate_bilinear (registry/kernels.yaml).
//!
//! ROTATE_BILINEAR NOTE. The arbitrary-angle rotation is the family's
//! only INTERPOLATING member (2×2 bilinear), so its reference mirrors
//! the WGSL term-for-term: the same `dx/dy` centre-relative continuous
//! coords, the same `R(−θ)` backward map, the same `− 0.5` shift into
//! texel-index space, the same clamp-to-edge taps, and the same fixed
//! `mix(mix(p00,p10,fx), mix(p01,p11,fx), fy)` blend order (x then y).
//! The trig runs HOST-side into the param block, so both lanes consume
//! bit-identical `cos_t`/`sin_t`.

use image_conformance::harness::{assert_within, parity_windowed, RefTile};
use image_conformance::Px;
use image_kernels::families::geom::{
    CropParams, FlipHParams, FlipVParams, Rotate90Params, RotateBilinearParams, GEOM_CROP,
    GEOM_FLIP_H, GEOM_FLIP_V, GEOM_ROTATE90_CCW, GEOM_ROTATE90_CW, GEOM_ROTATE_BILINEAR,
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

// ─────────────────────── rotate_bilinear ──────────────────────────

/// Clamp-to-edge tap (the module's `tap`).
fn tap(win: &[Px], win_w: u32, win_h: u32, x: i32, y: i32) -> Px {
    let cx = x.clamp(0, win_w as i32 - 1);
    let cy = y.clamp(0, win_h as i32 - 1);
    at(win, win_w, cx, cy)
}

fn lerp(a: Px, b: Px, t: f32) -> Px {
    // WGSL `mix(a, b, t)` is `a + t*(b - a)` — mirrored exactly.
    Px(std::array::from_fn(|c| a.0[c] + t * (b.0[c] - a.0[c])))
}

/// The scalar twin of `geom.rotate_bilinear` (see the module note).
fn rotate_bilinear_ref(
    win: &[Px],
    win_w: u32,
    win_h: u32,
    ox: u32,
    oy: u32,
    p: &RotateBilinearParams,
) -> Px {
    let dx = ox as f32 + 0.5 - p.dst_cx;
    let dy = oy as f32 + 0.5 - p.dst_cy;
    let sx = p.cos_t * dx + p.sin_t * dy + p.src_cx - 0.5;
    let sy = -p.sin_t * dx + p.cos_t * dy + p.src_cy - 0.5;
    let x0 = sx.floor();
    let y0 = sy.floor();
    let fx = sx - x0;
    let fy = sy - y0;
    let (ix, iy) = (x0 as i32, y0 as i32);
    let p00 = tap(win, win_w, win_h, ix, iy);
    let p10 = tap(win, win_w, win_h, ix + 1, iy);
    let p01 = tap(win, win_w, win_h, ix, iy + 1);
    let p11 = tap(win, win_w, win_h, ix + 1, iy + 1);
    lerp(lerp(p00, p10, fx), lerp(p01, p11, fx), fy)
}

#[test]
fn geom_rotate_bilinear_parity_same_canvas() {
    // A 13.5° straighten about the image centre, output = source dims
    // (the crop-commit case: the valid interior is cut afterwards).
    let (w, h) = (64u32, 48u32);
    let win = labeled(w, h);
    let c = (w as f32 / 2.0, h as f32 / 2.0);
    let p = RotateBilinearParams::new(13.5, c, c);
    match parity_windowed(&GEOM_ROTATE_BILINEAR, rotate_bilinear_ref, &win, w, h, &p) {
        Some(r) => assert_within(r, &GEOM_ROTATE_BILINEAR),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn geom_rotate_bilinear_parity_grown_canvas_negative_angle() {
    // A negative angle onto a LARGER destination canvas (the "hold the
    // whole rotated bounds" case) — dst_c ≠ src_c, and every corner tap
    // exercises the clamp.
    let (w, h) = (40u32, 30u32);
    let win = labeled(w, h);
    let (out_w, out_h) = (56u32, 48u32);
    let p = RotateBilinearParams::new(
        -22.0,
        (w as f32 / 2.0, h as f32 / 2.0),
        (out_w as f32 / 2.0, out_h as f32 / 2.0),
    );
    match parity_windowed(
        &GEOM_ROTATE_BILINEAR,
        rotate_bilinear_ref,
        &win,
        out_w,
        out_h,
        &p,
    ) {
        Some(r) => assert_within(r, &GEOM_ROTATE_BILINEAR),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn geom_rotate_bilinear_zero_angle_is_a_passthrough() {
    // θ = 0 with matching centres ⇒ the identity map: fx = fy = 0 and
    // every texel reads its own source texel. Pure reference (no GPU) —
    // it pins the coordinate convention itself.
    let (w, h) = (8u32, 6u32);
    let win = labeled(w, h);
    let px = win.quantized_px();
    let c = (w as f32 / 2.0, h as f32 / 2.0);
    let p = RotateBilinearParams::new(0.0, c, c);
    for y in 0..h {
        for x in 0..w {
            let got = rotate_bilinear_ref(&px, w, h, x, y, &p);
            let want = at(&px, w, x as i32, y as i32);
            for ch in 0..4 {
                assert!(
                    (got.0[ch] - want.0[ch]).abs() < 1e-6,
                    "({x},{y}) ch{ch}: {} vs {}",
                    got.0[ch],
                    want.0[ch]
                );
            }
        }
    }
}

#[test]
fn geom_rotate_bilinear_180_degrees_matches_the_corner_swap() {
    // θ = 180° about the centre maps (x, y) → (w-1-x, h-1-y) exactly
    // (cos = −1, sin ≈ 0 ⇒ the taps land on integers). Pure reference:
    // it pins the ROTATION DIRECTION, which a parity test alone cannot.
    let (w, h) = (8u32, 6u32);
    let win = labeled(w, h);
    let px = win.quantized_px();
    let c = (w as f32 / 2.0, h as f32 / 2.0);
    let p = RotateBilinearParams::new(180.0, c, c);
    for (x, y) in [(0u32, 0u32), (7, 5), (3, 2)] {
        let got = rotate_bilinear_ref(&px, w, h, x, y, &p);
        let want = at(&px, w, (w - 1 - x) as i32, (h - 1 - y) as i32);
        for ch in 0..4 {
            assert!(
                (got.0[ch] - want.0[ch]).abs() < 1e-3,
                "({x},{y}) ch{ch}: {} vs {}",
                got.0[ch],
                want.0[ch]
            );
        }
    }
}
