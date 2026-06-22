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

//! Selection-mask plumbing, end-to-end (spec §6.1 "selection-ready from
//! day one", §15 M3 'selection-mask plumbing surfaced').
//!
//! The kernel ABI applies `out = mix(a, result, mask)` for every
//! pointwise dispatch (`image_kernels::abi`), the mask binds at
//! `@group(2)` as `r16float`, and [`image_gpu::execute_tile_once`] takes
//! `mask: Option<&[u8]>` (None = the constant-1 mask). Every other
//! conformance test passes `None`, so NOTHING exercises a non-trivial
//! mask — that is the gap this file closes.
//!
//! These tests drive `execute_tile_once` DIRECTLY with a real
//! [`image_gpu::SelectionMask`] (the typed surface the editor lowers
//! selections to) and prove the mix contract pixel-by-pixel:
//!   * where mask == 1 → output == kernel(input)
//!   * where mask == 0 → output == input (the backdrop `a`)
//!   * where mask == 0.5 → output == mix(input, result, 0.5)
//!
//! feat: image.selection.mask (state registry row — orchestrator adds).

use half::f16;
use image_conformance::device::test_device;
use image_gpu::{execute_tile_once, SelectionMask, TileInput};
use image_kernels::families::linear::{
    math_invert, math_linear, MathInvertParams, MathLinearParams, MATH_INVERT, MATH_LINEAR,
};
use image_kernels::reference_prelude::Px;
use image_kernels::KernelDef;

// A small tile keeps the proof legible while still spanning >1 workgroup
// edge (16²) in width so the half-and-half split crosses a workgroup
// boundary, not just within a single one.
const W: u32 = 32;
const H: u32 = 8;

/// The unused second input the unary reference twins ignore (their `b`).
const ZERO: Px = Px([0.0; 4]);

/// A deterministic, NON-white input: a per-channel gradient kept away
/// from 1.0 so "forced white" (gain 0, bias 1) is visibly different from
/// the backdrop, and away from 0.0 so "invert" moves every channel.
fn input_tile() -> Vec<Px> {
    let mut px = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            px.push(Px([
                0.10 + 0.5 * (x as f32 / W as f32),
                0.20 + 0.5 * (y as f32 / H as f32),
                0.30 + 0.4 * ((x + y) as f32 / (W + H) as f32),
                0.80,
            ]));
        }
    }
    px
}

/// rgba16float texel bytes for upload — the input is f16-quantized ONCE
/// here so the GPU and our expectations consume identical stimulus.
fn f16_bytes(px: &[Px]) -> Vec<u8> {
    let mut out = Vec::with_capacity(px.len() * 8);
    for p in px {
        for c in p.0 {
            out.extend_from_slice(&f16::from_f32(c).to_bits().to_le_bytes());
        }
    }
    out
}

/// The SAME f16-quantized stimulus widened back to f32 — what the kernel
/// and the mix arithmetic actually see.
fn quantized(px: &[Px]) -> Vec<Px> {
    px.iter()
        .map(|p| Px(p.0.map(|c| f16::from_f32(c).to_f32())))
        .collect()
}

fn out_px(bytes: &[u8], i: usize, c: usize) -> u16 {
    u16::from_le_bytes([bytes[i * 8 + c * 2], bytes[i * 8 + c * 2 + 1]])
}

/// f16 ULP distance between two f16 bit patterns (order-preserving over
/// the full range, mirrors the harness quantize helper).
fn ulp(a_bits: u16, b_bits: u16) -> u32 {
    let m = |bits: u16| -> i32 {
        if bits & 0x8000 != 0 {
            -((bits & 0x7FFF) as i32)
        } else {
            (bits & 0x7FFF) as i32
        }
    };
    (m(a_bits) - m(b_bits)).unsigned_abs()
}

/// Run `def` over the input tile under `mask`, returning the GPU output
/// bytes. `None` when no GPU adapter (caller skips).
fn run_masked<P: bytemuck::Pod>(
    def: &'static KernelDef,
    params: &P,
    mask: &SelectionMask,
) -> Option<Vec<u8>> {
    let ctx = test_device()?;
    let input = input_tile();
    let in_bytes = f16_bytes(&input);
    let out = execute_tile_once(
        ctx,
        def,
        &[TileInput {
            f16_bytes: &in_bytes,
        }],
        bytemuck::bytes_of(params),
        Some(mask.bytes()),
        W,
        H,
    )
    .expect("masked kernel execution");
    Some(out)
}

/// Assert the per-texel mix contract: for every texel, output ==
/// mix(input, kernel_result, mask_weight) within `tol` f16 ULPs.
/// `kernel` maps an input pixel to the kernel's (unmasked) result.
fn assert_mix_contract(out: &[u8], kernel: impl Fn(Px) -> Px, mask: &SelectionMask, tol: u32) {
    let input = quantized(&input_tile());
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            let a = input[i];
            let result = kernel(a);
            let m = mask.weight_at(x, y);
            // WGSL mix(a, result, m) — the ABI's per-texel blend.
            let want = Px([
                a.0[0] * (1.0 - m) + result.0[0] * m,
                a.0[1] * (1.0 - m) + result.0[1] * m,
                a.0[2] * (1.0 - m) + result.0[2] * m,
                a.0[3] * (1.0 - m) + result.0[3] * m,
            ]);
            for c in 0..4 {
                let want_bits = f16::from_f32(want.0[c]).to_bits();
                let got_bits = out_px(out, i, c);
                let d = ulp(want_bits, got_bits);
                assert!(
                    d <= tol,
                    "texel ({x},{y}) ch {c}: want {} got {} (ulp {d} > tol {tol}, mask {m})",
                    f16::from_bits(want_bits).to_f32(),
                    f16::from_bits(got_bits).to_f32(),
                );
            }
        }
    }
}

/// Case 1 — HARD half-and-half mask over math.linear(gain 0, bias 1).
/// Left half mask==1 ⇒ forced white; right half mask==0 ⇒ untouched
/// backdrop. The headline "selection works" proof.
#[test]
fn image_selection_mask_half_split_force_white() {
    let mask = SelectionMask::from_fn(W, H, |x, _| if x < W / 2 { 1.0 } else { 0.0 });
    let p = MathLinearParams::new(0.0, 1.0);
    let Some(out) = run_masked(&MATH_LINEAR, &p, &mask) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };

    let input = quantized(&input_tile());
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            if x < W / 2 {
                // mask == 1: output is exactly white (kernel result).
                for c in 0..4 {
                    assert_eq!(
                        out_px(&out, i, c),
                        f16::from_f32(1.0).to_bits(),
                        "left/selected texel ({x},{y}) ch {c} should be white",
                    );
                }
            } else {
                // mask == 0: output is exactly the backdrop input `a`.
                for c in 0..4 {
                    assert_eq!(
                        out_px(&out, i, c),
                        f16::from_f32(input[i].0[c]).to_bits(),
                        "right/deselected texel ({x},{y}) ch {c} should equal input",
                    );
                }
            }
        }
    }
    // And the full per-texel mix contract holds (tol 0 — mask is {0,1}),
    // oracled by the math.linear scalar reference twin.
    assert_mix_contract(&out, |a| math_linear(a, ZERO, &p), &mask, 0);
}

/// Case 2 — the SAME hard split over math.invert, proving the masking is
/// kernel-agnostic (it lives in the ABI, not the kernel body).
#[test]
fn image_selection_mask_half_split_invert() {
    let mask = SelectionMask::from_fn(W, H, |x, _| if x < W / 2 { 1.0 } else { 0.0 });
    let p = MathInvertParams::new();
    let Some(out) = run_masked(&MATH_INVERT, &p, &mask) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };

    let input = quantized(&input_tile());
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            for c in 0..4 {
                // Selected → math.invert reference (1 - a); deselected →
                // backdrop. invert is exact (1 - a) but the f16 of (1-a)
                // may differ by 1 ULP from the GPU's own subtraction; allow
                // the kernel's declared tolerance.
                let want = if x < W / 2 {
                    math_invert(input[i], ZERO, &p).0[c]
                } else {
                    input[i].0[c]
                };
                let want_bits = f16::from_f32(want).to_bits();
                let got_bits = out_px(&out, i, c);
                let d = ulp(want_bits, got_bits);
                assert!(d <= 1, "texel ({x},{y}) ch {c}: ulp {d}");
            }
        }
    }
}

/// Case 3 — a GRADIENT mask (weight = x / (W-1)) over force-white, with a
/// guaranteed mid texel at exactly 0.5. Proves intermediate weights blend
/// per `mix(input, result, m)`, not just the {0,1} extremes.
#[test]
fn image_selection_mask_gradient_blend() {
    // Use a width where some column lands on exactly 0.5 (W even ⇒
    // column (W/2 - ... )). Build weight from a value set that includes
    // 0.0, 0.5, 1.0 exactly (all exact in f16).
    let mask = SelectionMask::from_fn(W, H, |x, _| {
        // Quantize the ramp to {0.0, 0.25, 0.5, 0.75, 1.0} so every
        // weight is exactly f16-representable and the mid value is 0.5.
        (x * 4 / (W - 1)).min(4) as f32 / 4.0
    });
    // Sanity: at least one texel has weight exactly 0.5.
    let has_half = (0..W).any(|x| (mask.weight_at(x, 0) - 0.5).abs() < 1e-6);
    assert!(has_half, "gradient must include a 0.5 texel");

    let p = MathLinearParams::new(0.0, 1.0);
    let Some(out) = run_masked(&MATH_LINEAR, &p, &mask) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };

    // Spot-check a 0.5 texel explicitly: output == mix(input, white, 0.5)
    // == (input + 1.0) / 2, within f16 tolerance.
    let input = quantized(&input_tile());
    let half_x = (0..W)
        .find(|&x| (mask.weight_at(x, 0) - 0.5).abs() < 1e-6)
        .expect("a 0.5 column");
    let i = half_x as usize; // row 0
    for c in 0..4 {
        let want = (input[i].0[c] + 1.0) * 0.5;
        let want_bits = f16::from_f32(want).to_bits();
        let got_bits = out_px(&out, i, c);
        let d = ulp(want_bits, got_bits);
        assert!(
            d <= 1,
            "0.5 texel x={half_x} ch {c}: want {} got {} (ulp {d})",
            want,
            f16::from_bits(got_bits).to_f32(),
        );
    }

    // And the whole-tile mix contract (tol 1: f16 of the blend vs the
    // GPU's own fma may differ by a single ULP), oracled by math.linear.
    assert_mix_contract(&out, |a| math_linear(a, ZERO, &p), &mask, 1);
}

/// Case 4 — a rectangular marquee selection ([`SelectionMask::from_rect`],
/// the editor's archetypal selection) over force-white: inside the rect
/// is white, outside is the untouched backdrop.
#[test]
fn image_selection_mask_rect_marquee() {
    let (rx, ry, rw, rh) = (4u32, 2u32, 10u32, 4u32);
    let mask = SelectionMask::from_rect(W, H, rx, ry, rw, rh);
    let p = MathLinearParams::new(0.0, 1.0);
    let Some(out) = run_masked(&MATH_LINEAR, &p, &mask) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };

    let input = quantized(&input_tile());
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            let inside = x >= rx && x < rx + rw && y >= ry && y < ry + rh;
            for c in 0..4 {
                // Inside the marquee → math.linear(0,1) = white; outside →
                // untouched backdrop. Mask is {0,1} ⇒ exact.
                let want_bits = if inside {
                    f16::from_f32(math_linear(input[i], ZERO, &p).0[c]).to_bits()
                } else {
                    f16::from_f32(input[i].0[c]).to_bits()
                };
                assert_eq!(
                    out_px(&out, i, c),
                    want_bits,
                    "texel ({x},{y}) ch {c} inside={inside}",
                );
            }
        }
    }
}
