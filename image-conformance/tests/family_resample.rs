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

//! gpu↔ref parity for the resample family (T1) — three separable
//! resamplers (`resample.nearest`, `resample.mitchell`,
//! `resample.lanczos3`) through `parity_windowed` (the resample lane:
//! `in0` is the full source window, the mapping travels in
//! `ResampleParams`).
//!
//! The scalar references below mirror the WGSL byte-for-byte: the same
//! `sx/sy = (out+0.5)*inv - 0.5 + off`, the same tap base
//! `floor(s) - (ceil(support)-1)`, the same `j`-outer / `i`-inner fused
//! `sum`/`wsum` accumulation, the same clamp-to-window edge rule, the
//! same sum normalisation. Determinism (§6.3) lives in that mirroring.
//!
//! WINDOW SIZING (documented algebra). Output texel `x ∈ [0, out_w)`
//! samples continuous source `sx = (x+0.5)*inv_x - 0.5 + off_x`, so the
//! largest tap touched is `floor(sx_max) + ceil(support)` where
//! `sx_max = (out_w-0.5)*inv_x - 0.5 + off_x < out_w*inv_x`. Sizing
//! `win_w = ceil(out_w*inv_x) + 2*ceil(support) + 2` covers
//! `[0 .. floor(sx_max)+ceil(support)]` with margin to spare; the low
//! edge (`x = 0`, support kernels) still exercises the clamp. Identity
//! (`inv = 1`, `off = 0`) gives `sx = x`, a near-passthrough whose
//! support taps land on `x` with the off-centre taps weighted ~0.
//! feat: resample.nearest / resample.mitchell / resample.lanczos3
//! (registry/kernels.yaml).

use image_conformance::harness::{assert_within, parity_windowed, RefTile};
use image_conformance::Px;
use image_kernels::families::resample::{
    ResampleParams, RESAMPLE_LANCZOS3, RESAMPLE_MITCHELL, RESAMPLE_NEAREST,
};
use image_kernels::{KernelClass, KernelDef};

/// Continuous source coordinate for output index `o`, mirroring the WGSL
/// `(o + 0.5) * inv - 0.5 + off`.
fn src_coord(o: u32, inv: f32, off: f32) -> f32 {
    (o as f32 + 0.5) * inv - 0.5 + off
}

fn fetch(win: &[Px], w: u32, h: u32, i: i32, j: i32) -> Px {
    let ci = i.clamp(0, w as i32 - 1) as u32;
    let cj = j.clamp(0, h as i32 - 1) as u32;
    win[(cj * w + ci) as usize]
}

// --- Nearest --------------------------------------------------------

fn nearest_ref(win: &[Px], w: u32, h: u32, ox: u32, oy: u32, p: &ResampleParams) -> Px {
    let sx = src_coord(ox, p.inv_scale_x, p.src_off_x);
    let sy = src_coord(oy, p.inv_scale_y, p.src_off_y);
    let ix = (sx + 0.5).floor() as i32;
    let iy = (sy + 0.5).floor() as i32;
    fetch(win, w, h, ix, iy)
}

// --- Mitchell–Netravali (B = C = 1/3) -------------------------------

fn mitchell(t: f32) -> f32 {
    let b = 1.0 / 3.0_f32;
    let c = 1.0 / 3.0_f32;
    let x = t.abs();
    if x < 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * x * x * x
            + (-18.0 + 12.0 * b + 6.0 * c) * x * x
            + (6.0 - 2.0 * b))
            / 6.0
    } else if x < 2.0 {
        ((-b - 6.0 * c) * x * x * x
            + (6.0 * b + 30.0 * c) * x * x
            + (-12.0 * b - 48.0 * c) * x
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

fn mitchell_ref(win: &[Px], w: u32, h: u32, ox: u32, oy: u32, p: &ResampleParams) -> Px {
    separable_ref(win, w, h, ox, oy, p, 2, mitchell)
}

// --- Lanczos-3 ------------------------------------------------------

const PI: f32 = std::f32::consts::PI;

fn sinc(u: f32) -> f32 {
    if u == 0.0 {
        1.0
    } else {
        let pu = PI * u;
        pu.sin() / pu
    }
}

fn lanczos3(t: f32) -> f32 {
    let x = t.abs();
    if x < 3.0 {
        sinc(t) * sinc(t / 3.0)
    } else {
        0.0
    }
}

fn lanczos3_ref(win: &[Px], w: u32, h: u32, ox: u32, oy: u32, p: &ResampleParams) -> Px {
    separable_ref(win, w, h, ox, oy, p, 3, lanczos3)
}

/// Shared separable resampler reference for the support kernels. `ext`
/// is `ceil(support)` (mitchell 2, lanczos3 3); the tap window is
/// `[floor(s) - (ext-1) .. floor(s) + ext]` ⇒ `2*ext` taps per axis,
/// the SAME base/loop count as the WGSL. `j` outer asc, `i` inner asc,
/// fused `sum`/`wsum`, normalise by `wsum` — the kernel's determinism
/// contract, mirrored exactly.
fn separable_ref(
    win: &[Px],
    w: u32,
    h: u32,
    ox: u32,
    oy: u32,
    p: &ResampleParams,
    ext: i32,
    weight: fn(f32) -> f32,
) -> Px {
    let sx = src_coord(ox, p.inv_scale_x, p.src_off_x);
    let sy = src_coord(oy, p.inv_scale_y, p.src_off_y);
    // The filter scale: 1 enlarging, the shrink factor reducing (≤ 16).
    let fx = p.inv_scale_x.clamp(1.0, 16.0);
    let fy = p.inv_scale_y.clamp(1.0, 16.0);
    let e = ext as f32;
    let (x0, x1) = (
        (sx - e * fx).floor() as i32 + 1,
        (sx + e * fx).floor() as i32,
    );
    let (y0, y1) = (
        (sy - e * fy).floor() as i32 + 1,
        (sy + e * fy).floor() as i32,
    );

    let mut sum = Px([0.0; 4]);
    let mut wsum = 0.0_f32;
    for j in y0..=y1 {
        let wy = weight((sy - j as f32) / fy);
        for i in x0..=x1 {
            let wx = weight((sx - i as f32) / fx);
            let wgt = wx * wy;
            sum = sum + Px(fetch(win, w, h, i, j).0.map(|ch| ch * wgt));
            wsum += wgt;
        }
    }
    Px(sum.0.map(|ch| ch / wsum))
}

// --- Stimulus + window sizing ---------------------------------------

/// `ceil(support)` per the contract supports (0.5 ⇒ 1, 2.0 ⇒ 2,
/// 3.0 ⇒ 3).
fn ceil_support(def: &KernelDef) -> u32 {
    match def.class {
        KernelClass::Resample { support } => support.ceil() as u32,
        _ => unreachable!("resample family"),
    }
}

/// A finite gradient source window sized so every tap for every output
/// texel resolves inside it (see module WINDOW SIZING algebra).
fn window(def: &KernelDef, out_w: u32, out_h: u32, inv_x: f32, inv_y: f32) -> RefTile {
    let s = ceil_support(def);
    let win_w = (out_w as f32 * inv_x).ceil() as u32 + 2 * s + 2;
    let win_h = (out_h as f32 * inv_y).ceil() as u32 + 2 * s + 2;
    RefTile::from_fn(win_w, win_h, |x, y| {
        Px([
            (x as f32 * 0.011).fract(),
            (y as f32 * 0.019).fract(),
            ((x + 2 * y) as f32 * 0.005).fract(),
            1.0,
        ])
    })
}

fn run(def: &'static KernelDef, ref_fn: fn(&[Px], u32, u32, u32, u32, &ResampleParams) -> Px) {
    // Five mappings: 4x and 2x downscale (the filter widened 4x / 2x),
    // 1.5x downscale, 2x upscale, identity.
    let cases = [
        (16u32, 12u32, 4.0f32, 4.0f32),
        (32, 24, 2.0, 2.0),
        (40, 30, 1.5, 1.5),
        (48, 36, 0.5, 0.5),
        (32, 32, 1.0, 1.0),
    ];
    for (out_w, out_h, inv_x, inv_y) in cases {
        let p = ResampleParams::new(inv_x, inv_y, 0.0, 0.0);
        let win = window(def, out_w, out_h, inv_x, inv_y);
        match parity_windowed(def, ref_fn, &win, out_w, out_h, &p) {
            Some(r) => assert_within(r, def),
            None => {
                eprintln!("SKIP: no GPU adapter");
                return;
            }
        }
    }
}

#[test]
fn resample_nearest_parity() {
    run(&RESAMPLE_NEAREST, nearest_ref);
}

#[test]
fn resample_mitchell_parity() {
    run(&RESAMPLE_MITCHELL, mitchell_ref);
}

#[test]
fn resample_lanczos3_parity() {
    run(&RESAMPLE_LANCZOS3, lanczos3_ref);
}

/// A REDUCTION AVERAGES what it passes over. A one-pixel checkerboard
/// reduced 2× is grey (its mean) under a filter widened by the shrink
/// factor; evaluated at source scale, it came out as whichever phase the
/// output texels landed on (the libvips oracle recorded the divergence:
/// up to 0.31 for Mitchell, 0.48 for Lanczos-3). Away from the clamped
/// border, every output texel is within 2 % of 0.5.
#[test]
#[allow(non_snake_case)]
fn a_two_times_reduction_of_a_checkerboard_is_grey__feat__image_kernel_family_t1() {
    let (w, h) = (64u32, 64u32);
    let board: Vec<Px> = (0..w * h)
        .map(|k| {
            let v = ((k % w + k / w) % 2) as f32;
            Px([v, v, v, 1.0])
        })
        .collect();
    let p = ResampleParams::new(2.0, 2.0, 0.0, 0.0);
    for (name, f) in [
        (
            "mitchell",
            mitchell_ref as fn(&[Px], u32, u32, u32, u32, &ResampleParams) -> Px,
        ),
        ("lanczos3", lanczos3_ref),
    ] {
        for oy in 4..28 {
            for ox in 4..28 {
                let v = f(&board, w, h, ox, oy, &p).0[0];
                assert!((v - 0.5).abs() < 0.02, "{name} ({ox},{oy}): {v}");
            }
        }
    }
}
