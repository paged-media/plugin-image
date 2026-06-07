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

//! gpu↔ref parity for the linear family — the phase-0 codegen proof
//! (spec §16.4): the first kernels through dual emission + the parity
//! gate. feat: math.linear / math.invert (registry/kernels.yaml).

use image_conformance::harness::{assert_within, parity, RefTile};
use image_conformance::Px;
use image_kernels::families::linear::{
    math_invert, math_linear, MathInvertParams, MathLinearParams, MATH_INVERT, MATH_LINEAR,
};

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

#[test]
fn linear_parity() {
    let tile = gradient_tile(image_core::TILE, image_core::TILE);
    let p = MathLinearParams::new(1.5, -0.125);
    match parity(&MATH_LINEAR, math_linear, &[&tile], &p) {
        Some(r) => assert_within(r, &MATH_LINEAR),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn invert_parity() {
    let tile = gradient_tile(image_core::TILE, image_core::TILE);
    let p = MathInvertParams::new();
    match parity(&MATH_INVERT, math_invert, &[&tile], &p) {
        Some(r) => assert_within(r, &MATH_INVERT),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}
