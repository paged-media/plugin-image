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
//! `parity_windowed`. Gaussian/unsharp tests join with the M1 fan-out.
//! feat: conv.box (registry/kernels.yaml).

use image_conformance::harness::{assert_within, parity_windowed, RefTile};
use image_conformance::Px;
use image_kernels::families::conv::{ConvBoxParams, CONV_BOX};
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
