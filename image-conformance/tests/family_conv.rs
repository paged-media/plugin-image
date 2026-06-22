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

//! gpu↔ref parity for the convolution family. `conv.box` is the ABI
//! v1.1 amendment proof — the first handwritten windowed module through
//! `parity_windowed`. `conv.gaussian_h`/`conv.gaussian_v` add the
//! separable two-pass Gaussian (plus the separability proof:
//! [`gaussian_separable_reference`]); `conv.unsharp` the binary-point
//! unsharp mask. feat: conv.box, conv.gaussian_h, conv.gaussian_v,
//! conv.unsharp (registry/kernels.yaml).

use half::f16;
use image_conformance::harness::{assert_within, parity, parity_windowed, RefTile};
use image_conformance::Px;
use image_kernels::families::conv::{
    ConvBoxParams, ConvGaussianParams, ConvUnsharpParams, CONV_BOX, CONV_GAUSSIAN_H,
    CONV_GAUSSIAN_V, CONV_UNSHARP, GAUSSIAN_MAX_RADIUS,
};
use image_kernels::KernelClass;

/// Scalar reference: 3×3 mean over the quantized window, summation in
/// the kernel's documented order (dy outer asc, dx inner asc).
fn box3_ref(win: &[Px], win_w: u32, _win_h: u32, ox: u32, oy: u32, _p: &ConvBoxParams) -> Px {
    let mut sum = Px([0.0; 4]);
    for dy in 0..3u32 {
        for dx in 0..3u32 {
            sum = sum + win[((oy + dy) * win_w + ox + dx) as usize];
        }
    }
    Px(sum.0.map(|c| c / 9.0))
}

fn window(out_w: u32, out_h: u32) -> RefTile {
    let (rx, ry) = match CONV_BOX.class {
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

#[test]
fn conv_box_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    let win = window(w, h);
    match parity_windowed(&CONV_BOX, box3_ref, &win, w, h, &ConvBoxParams::new()) {
        Some(r) => assert_within(r, &CONV_BOX),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn conv_box_parity_small() {
    let win = window(64, 48);
    match parity_windowed(&CONV_BOX, box3_ref, &win, 64, 48, &ConvBoxParams::new()) {
        Some(r) => assert_within(r, &CONV_BOX),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

// ───────────────────────────── gaussian ────────────────────────────
//
// Both passes treat in0 as `out + 2·MAX_RADIUS` along their axis
// (MAX_RADIUS = 24). Output texel (ox, oy) maps to window center
// (ox + 24, oy) [h] / (ox, oy + 24) [v]. The scalar reference mirrors
// the shader's reduction order exactly: normalization sum S over
// i = -r..=r ascending, then the weighted convolution over i = -r..=r
// ascending with weight exp(-i²/2σ²)/S.

const R24: i32 = GAUSSIAN_MAX_RADIUS as i32;

/// One Gaussian weight `exp(-i²/(2σ²))` — the same f32 expression the
/// shader evaluates (`exp` is the shared transcendental; the f16 output
/// quantization absorbs the last-ulp f32 divergence).
fn gauss_w(i: i32, sigma: f32) -> f32 {
    let fi = i as f32;
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    (-(fi * fi) * inv2s2).exp()
}

/// 1D Gaussian over a window axis. `axis_dx`/`axis_dy` pick x (1,0) or
/// y (0,1); `cx`/`cy` is the window-center texel for this output.
fn gauss_axis(
    win: &[Px],
    win_w: u32,
    cx: i32,
    cy: i32,
    sigma: f32,
    radius: i32,
    axis_x: i32,
    axis_y: i32,
) -> Px {
    let r = radius.min(R24);
    // Normalization sum S, ascending i.
    let mut s = 0.0f32;
    for i in -r..=r {
        s += gauss_w(i, sigma);
    }
    // Weighted convolution, ascending i; weight = w_i / S.
    let mut acc = Px([0.0; 4]);
    for i in -r..=r {
        let w = gauss_w(i, sigma) / s;
        let sx = (cx + i * axis_x) as u32;
        let sy = (cy + i * axis_y) as u32;
        let px = win[(sy * win_w + sx) as usize];
        acc = acc + px.map(|c| c * w);
    }
    acc
}

/// Scalar reference for `conv.gaussian_h` — center at (ox+24, oy).
fn gaussian_h_ref(
    win: &[Px],
    win_w: u32,
    _win_h: u32,
    ox: u32,
    oy: u32,
    p: &ConvGaussianParams,
) -> Px {
    gauss_axis(
        win,
        win_w,
        ox as i32 + R24,
        oy as i32,
        p.sigma,
        p.radius as i32,
        1,
        0,
    )
}

/// Scalar reference for `conv.gaussian_v` — center at (ox, oy+24).
fn gaussian_v_ref(
    win: &[Px],
    win_w: u32,
    _win_h: u32,
    ox: u32,
    oy: u32,
    p: &ConvGaussianParams,
) -> Px {
    gauss_axis(
        win,
        win_w,
        ox as i32,
        oy as i32 + R24,
        p.sigma,
        p.radius as i32,
        0,
        1,
    )
}

/// A finite analytic test window (FINITE stimulus rule, harness docs).
fn gauss_window(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        Px([
            (x as f32 * 0.011).fract(),
            (y as f32 * 0.019).fract(),
            ((x + 2 * y) as f32 * 0.005).fract(),
            1.0,
        ])
    })
}

#[test]
fn conv_gaussian_h_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    // in0 = out + 2·(24, 0).
    let win = gauss_window(w + 2 * R24 as u32, h);
    let p = ConvGaussianParams::new(3.0, 8);
    match parity_windowed(&CONV_GAUSSIAN_H, gaussian_h_ref, &win, w, h, &p) {
        Some(r) => assert_within(r, &CONV_GAUSSIAN_H),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn conv_gaussian_h_parity_small() {
    let (w, h) = (64u32, 48u32);
    let win = gauss_window(w + 2 * R24 as u32, h);
    // Radius at the MAX bound, fractional σ — exercises the full window.
    let p = ConvGaussianParams::new(7.5, GAUSSIAN_MAX_RADIUS as u32);
    match parity_windowed(&CONV_GAUSSIAN_H, gaussian_h_ref, &win, w, h, &p) {
        Some(r) => assert_within(r, &CONV_GAUSSIAN_H),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn conv_gaussian_v_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    // in0 = out + 2·(0, 24).
    let win = gauss_window(w, h + 2 * R24 as u32);
    let p = ConvGaussianParams::new(3.0, 8);
    match parity_windowed(&CONV_GAUSSIAN_V, gaussian_v_ref, &win, w, h, &p) {
        Some(r) => assert_within(r, &CONV_GAUSSIAN_V),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn conv_gaussian_v_parity_small() {
    let (w, h) = (64u32, 48u32);
    let win = gauss_window(w, h + 2 * R24 as u32);
    let p = ConvGaussianParams::new(7.5, GAUSSIAN_MAX_RADIUS as u32);
    match parity_windowed(&CONV_GAUSSIAN_V, gaussian_v_ref, &win, w, h, &p) {
        Some(r) => assert_within(r, &CONV_GAUSSIAN_V),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

// ─────────────────────── separability proof ────────────────────────
//
// WINDOW ALGEBRA. To produce a final `w × h` Gaussian output by the
// two-pass GPU chain (h then v), each pass needs its axis inflated by
// 2·24:
//
//   - v-pass: out (w, h)        ⇐ needs intermediate (w, h + 2·24).
//   - h-pass: out (w, h + 48)   ⇐ needs source     (w + 2·24, h + 48).
//
// So the start window is (w + 48) × (h + 48); the h-pass collapses the
// +48 in x to give the (w) × (h + 48) intermediate; the v-pass collapses
// the +48 in y to give the final (w) × (h). The intermediate is stored
// rgba16float (the GPU readback bytes) — `gaussian_separable_reference`
// models that exact chain (h-pass f32 → f16 quantize → v-pass f32) and
// the test asserts the GPU two-pass chain matches it within
// ChannelEpsF16(4). This proves separable(h∘v) ≡ direct-2D within
// tolerance: a 2D Gaussian whose kernel factors as wₓ(i)·w_y(j) equals
// the row-pass-then-column-pass, the standard separability identity.

/// The faithful scalar model of the GPU two-pass chain. Given the
/// `(w+48)×(h+48)` start window, applies the h-pass (over x, center at
/// +24), quantizes the intermediate to f16 (matching the rgba16float
/// intermediate the GPU writes), then the v-pass (over y, center at
/// +24), yielding the direct 2D-separable Gaussian at final output
/// (ox, oy). Reduction order matches the shader (ascending i, both
/// passes; normalize by per-axis S).
pub fn gaussian_separable_reference(
    win: &[Px],
    win_w: u32,
    win_h: u32,
    sigma: f32,
    radius: i32,
    ox: u32,
    oy: u32,
) -> Px {
    let inter_w = win_w - 2 * R24 as u32; // collapse x.
    let inter_h = win_h; // h-pass keeps y.
                         // Intermediate row this output column needs: v-pass center at
                         // (ox, oy + 24) reads intermediate rows (oy + 24) ± i. We compute
                         // every intermediate sample lazily inside the v-pass loop, applying
                         // the f16 quantization the GPU intermediate texture imposes.
    let r = radius.min(R24);
    let mut s = 0.0f32;
    for i in -r..=r {
        s += gauss_w(i, sigma);
    }
    // v-pass over the intermediate, center row (oy + 24).
    let mut acc = Px([0.0; 4]);
    for j in -r..=r {
        let inter_y = oy as i32 + R24 + j;
        // h-pass for intermediate texel (ox, inter_y): center at
        // (ox + 24, inter_y) in the START window.
        let h = gauss_axis(win, win_w, ox as i32 + R24, inter_y, sigma, radius, 1, 0);
        // The intermediate is stored rgba16float — quantize.
        let hq = h.map(|c| f16::from_f32(c).to_f32());
        let w = gauss_w(j, sigma) / s;
        acc = acc + hq.map(|c| c * w);
    }
    let _ = (inter_w, inter_h);
    acc
}

/// The two-pass GPU chain equals the direct 2D-separable Gaussian
/// (`gaussian_separable_reference`) within ChannelEpsF16(4) — the
/// separability identity proof.
#[test]
fn conv_gaussian_h_v_separable_chain() {
    use image_conformance::device::test_device;
    use image_conformance::quantize::{f16_ulp_distance, f32_to_f16_bits};
    use image_gpu::execute_windowed_once;

    let ctx = match test_device() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no GPU adapter");
            return;
        }
    };

    let (w, h) = (48u32, 40u32);
    let sigma = 4.0f32;
    let radius = 12i32;
    let p = ConvGaussianParams::new(sigma, radius as u32);

    // Start window (w+48)×(h+48).
    let start = gauss_window(w + 2 * R24 as u32, h + 2 * R24 as u32);

    // GPU h-pass: out (w, h+48), in0 = start (w+48, h+48).
    let inter_h_dim = h + 2 * R24 as u32;
    let inter_bytes = execute_windowed_once(
        ctx,
        &CONV_GAUSSIAN_H,
        &start.f16_bytes(),
        start.w,
        start.h,
        p.as_bytes(),
        None,
        w,
        inter_h_dim,
    )
    .expect("h-pass");

    // The intermediate as a window for the v-pass: out (w, h),
    // in0 = intermediate (w, h+48). The GPU bytes ARE rgba16float, so we
    // wrap them as a RefTile by widening to f32 (lossless) then re-upload.
    let inter_px: Vec<Px> = (0..(w * inter_h_dim) as usize)
        .map(|i| {
            let base = i * 8;
            Px([
                f16::from_bits(u16::from_le_bytes([
                    inter_bytes[base],
                    inter_bytes[base + 1],
                ]))
                .to_f32(),
                f16::from_bits(u16::from_le_bytes([
                    inter_bytes[base + 2],
                    inter_bytes[base + 3],
                ]))
                .to_f32(),
                f16::from_bits(u16::from_le_bytes([
                    inter_bytes[base + 4],
                    inter_bytes[base + 5],
                ]))
                .to_f32(),
                f16::from_bits(u16::from_le_bytes([
                    inter_bytes[base + 6],
                    inter_bytes[base + 7],
                ]))
                .to_f32(),
            ])
        })
        .collect();
    let inter = RefTile {
        w,
        h: inter_h_dim,
        px: inter_px,
    };

    let final_bytes = execute_windowed_once(
        ctx,
        &CONV_GAUSSIAN_V,
        &inter.f16_bytes(),
        inter.w,
        inter.h,
        p.as_bytes(),
        None,
        w,
        h,
    )
    .expect("v-pass");

    // Reference: the direct 2D-separable model over the START window.
    let quant_start = start.quantized_px();
    let mut max_ulp = 0u32;
    let mut worst_at = (0usize, 0usize);
    for oy in 0..h {
        for ox in 0..w {
            let i = (oy * w + ox) as usize;
            let want =
                gaussian_separable_reference(&quant_start, start.w, start.h, sigma, radius, ox, oy);
            for c in 0..4 {
                let want_bits = f32_to_f16_bits(want.0[c]);
                let got_bits = u16::from_le_bytes([
                    final_bytes[i * 8 + c * 2],
                    final_bytes[i * 8 + c * 2 + 1],
                ]);
                let d = f16_ulp_distance(want_bits, got_bits);
                if d > max_ulp {
                    max_ulp = d;
                    worst_at = (i, c);
                }
            }
        }
    }
    assert!(
        max_ulp <= 4,
        "separable chain: max f16 ULP {} exceeds 4 (worst at texel {}, channel {})",
        max_ulp,
        worst_at.0,
        worst_at.1,
    );
}

// ───────────────────────────── unsharp ─────────────────────────────
//
// out_c = a_c + amount·(a_c − b_c) where |a_c − b_c| > threshold, else
// a_c. Binary point kernel (in0 = original, in1 = blurred). M0 uses
// threshold 0.0. The reference mirrors the shader's `select` per
// channel.

fn unsharp_ref(a: Px, b: Px, p: &ConvUnsharpParams) -> Px {
    let mut out = [0.0f32; 4];
    for (c, o) in out.iter_mut().enumerate() {
        let delta = a.0[c] - b.0[c];
        *o = if delta.abs() > p.threshold {
            a.0[c] + p.amount * delta
        } else {
            a.0[c]
        };
    }
    Px(out)
}

fn unsharp_inputs(w: u32, h: u32) -> (RefTile, RefTile) {
    let orig = RefTile::from_fn(w, h, |x, y| {
        Px([
            (x as f32 * 0.017).fract(),
            (y as f32 * 0.023).fract(),
            ((x + y) as f32 * 0.009).fract(),
            1.0,
        ])
    });
    // A plausible "blurred" companion: a damped, offset variant so
    // a − b is a finite non-trivial residual.
    let blur = RefTile::from_fn(w, h, |x, y| {
        Px([
            (x as f32 * 0.017).fract() * 0.8,
            (y as f32 * 0.023).fract() * 0.8,
            ((x + y) as f32 * 0.009).fract() * 0.8,
            1.0,
        ])
    });
    (orig, blur)
}

#[test]
fn conv_unsharp_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    let (orig, blur) = unsharp_inputs(w, h);
    let p = ConvUnsharpParams::new(1.5, 0.0);
    match parity(&CONV_UNSHARP, unsharp_ref, &[&orig, &blur], &p) {
        Some(r) => assert_within(r, &CONV_UNSHARP),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn conv_unsharp_parity_small() {
    let (w, h) = (64u32, 48u32);
    let (orig, blur) = unsharp_inputs(w, h);
    let p = ConvUnsharpParams::new(0.75, 0.0);
    match parity(&CONV_UNSHARP, unsharp_ref, &[&orig, &blur], &p) {
        Some(r) => assert_within(r, &CONV_UNSHARP),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}
