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

//! gpu↔ref parity for the arithmetic family (T0, spec §11). feat:
//! math.add / sub / mul / div / add_const / mul_const / abs / sign /
//! neg (registry/kernels.yaml). Stimulus is FINITE per the harness
//! rule; div divisors stay in [0.25, 2.0]; sign/abs/neg tiles straddle
//! zero and the 0.5 truthiness line and include exact a==b texels.

use image_conformance::harness::{assert_within, parity, RefTile};
use image_conformance::Px;
use image_kernels::families::arithmetic::{
    math_abs, math_add, math_add_const, math_div, math_mul, math_mul_const, math_neg, math_sign,
    math_sub, MathAbsParams, MathAddConstParams, MathAddParams, MathDivParams, MathMulConstParams,
    MathMulParams, MathNegParams, MathSignParams, MathSubParams, MATH_ABS, MATH_ADD,
    MATH_ADD_CONST, MATH_DIV, MATH_MUL, MATH_MUL_CONST, MATH_NEG, MATH_SIGN, MATH_SUB,
};

/// Smooth [0,1] gradient — the workhorse stimulus for the linear ops.
fn gradient_tile(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        Px([
            x as f32 / w as f32,
            y as f32 / h as f32,
            (x + y) as f32 / (w + h) as f32,
            1.0,
        ])
    })
}

/// A signed gradient over [-1, 1] crossing exact zero and the 0.5
/// truthiness boundary — the stimulus that exercises `sign4`'s
/// comparison composition and `abs`/`neg` over both signs.
fn signed_tile(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        let u = 2.0 * (x as f32 / (w - 1).max(1) as f32) - 1.0;
        let v = 2.0 * (y as f32 / (h - 1).max(1) as f32) - 1.0;
        // Channel 2 deliberately seeds exact 0.0 columns; channel 3 sits
        // on the 0.5 line for even rows so the truthiness edge is probed.
        let zeroish = if x % 4 == 0 { 0.0 } else { u };
        let half = if y % 2 == 0 { 0.5 } else { -0.5 };
        Px([u, v, zeroish, half])
    })
}

/// Divisor tile bounded to [0.25, 2.0] (STIMULUS RULE: away from 0 so
/// the quotient stays finite on both lanes).
fn divisor_tile(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        let t = (x + y) as f32 / (w + h) as f32; // [0,1)
        let d = 0.25 + 1.75 * t; // [0.25, 2.0)
        Px([d, 0.25 + 1.75 * (1.0 - t), d, 1.0])
    })
}

const W: u32 = image_core::TILE;
const H: u32 = image_core::TILE;

#[test]
fn math_add_parity() {
    let a = gradient_tile(W, H);
    let b = signed_tile(W, H);
    let p = MathAddParams::new();
    match parity(&MATH_ADD, math_add, &[&a, &b], &p) {
        Some(r) => assert_within(r, &MATH_ADD),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn math_sub_parity() {
    let a = gradient_tile(W, H);
    let b = signed_tile(W, H);
    let p = MathSubParams::new();
    match parity(&MATH_SUB, math_sub, &[&a, &b], &p) {
        Some(r) => assert_within(r, &MATH_SUB),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn math_mul_parity() {
    let a = gradient_tile(W, H);
    let b = signed_tile(W, H);
    let p = MathMulParams::new();
    match parity(&MATH_MUL, math_mul, &[&a, &b], &p) {
        Some(r) => assert_within(r, &MATH_MUL),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn math_div_parity() {
    let a = gradient_tile(W, H);
    let b = divisor_tile(W, H);
    let p = MathDivParams::new();
    match parity(&MATH_DIV, math_div, &[&a, &b], &p) {
        Some(r) => assert_within(r, &MATH_DIV),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn math_add_const_parity() {
    let a = signed_tile(W, H);
    let p = MathAddConstParams::new(0.375);
    match parity(&MATH_ADD_CONST, math_add_const, &[&a], &p) {
        Some(r) => assert_within(r, &MATH_ADD_CONST),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn math_mul_const_parity() {
    let a = signed_tile(W, H);
    let p = MathMulConstParams::new(-1.5);
    match parity(&MATH_MUL_CONST, math_mul_const, &[&a], &p) {
        Some(r) => assert_within(r, &MATH_MUL_CONST),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn math_abs_parity() {
    let a = signed_tile(W, H);
    let p = MathAbsParams::new();
    match parity(&MATH_ABS, math_abs, &[&a], &p) {
        Some(r) => assert_within(r, &MATH_ABS),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn math_sign_parity() {
    // The full signed tile straddles zero (exact 0.0 columns) and the
    // 0.5 line; the small case keeps the negatives/zero/positives mix
    // dense for a quick local check.
    let a = signed_tile(W, H);
    let p = MathSignParams::new();
    match parity(&MATH_SIGN, math_sign, &[&a], &p) {
        Some(r) => assert_within(r, &MATH_SIGN),
        None => eprintln!("SKIP: no GPU adapter"),
    }

    let small = signed_tile(64, 64);
    match parity(&MATH_SIGN, math_sign, &[&small], &p) {
        Some(r) => assert_within(r, &MATH_SIGN),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn math_neg_parity() {
    let a = signed_tile(W, H);
    let p = MathNegParams::new();
    match parity(&MATH_NEG, math_neg, &[&a], &p) {
        Some(r) => assert_within(r, &MATH_NEG),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}
