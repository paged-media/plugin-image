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

//! DEFECT PINS in the shipping adjust chain (`image_js::ingest::
//! adjust_rgba8` / `adjust_f16`, the body of the wasm adjust doors and of
//! adjustment layers), found by the Photoshop oracle replay
//! (`oracle_photoshop.rs`: Gaussian blur mean 84 levels and unsharp mask
//! mean 42 levels from Photoshop on the colour sweep).
//!
//! The blur stage (`blur_sigma`, two windowed `conv.gaussian_*` nodes in
//! the Engine A pipeline) does not produce a blur: a CONSTANT opaque
//! image — whose blur is itself, under any edge rule — comes back with
//! changed colour and alpha, and beyond the first tile the output is
//! transparent black. The kernels alone pass their parity tests
//! (`family_conv.rs` drives `execute_windowed_once` directly), so the
//! fault is in how the chain windows them. Unsharp (`sharpen_amount`)
//! rides the same blur and inherits it.
//!
//! Fixed: the scheduler gave a windowed kernel its bare tile, while the
//! kernel reads a window `radius` wider on each side. It now gathers the
//! window across tiles and repeats the image edge outside the image
//! (`image-pipeline/src/schedule.rs`, `windows`); these tests keep it so.

use image_js::ingest::{adjust_rgba8, AdjustParams, DecodedImage};

const COLOUR: [u8; 4] = [100, 150, 200, 255];

fn constant(w: u32, h: u32) -> DecodedImage {
    DecodedImage::from_rgba8(w, h, COLOUR.repeat((w * h) as usize)).expect("image")
}

/// Worst per-channel distance from the constant colour, and where.
fn worst(out: &[u8], w: u32) -> (u8, u32, u32) {
    let mut worst = (0u8, 0u32, 0u32);
    for (i, p) in out.chunks_exact(4).enumerate() {
        let d = (0..4).map(|c| p[c].abs_diff(COLOUR[c])).max().unwrap_or(0);
        if d > worst.0 {
            worst = (d, i as u32 % w, i as u32 / w);
        }
    }
    worst
}

fn run(w: u32, h: u32, p: AdjustParams) -> Option<(u8, u32, u32)> {
    let ctx = image_conformance::device::test_device()?;
    let out = pollster::block_on(adjust_rgba8(ctx, &constant(w, h), &p, None)).expect("adjust");
    Some(worst(&out, w))
}

#[test]
#[allow(non_snake_case)]
fn blur_of_a_constant_image_is_the_constant__feat__image_editor_filters() {
    for (w, h) in [(64, 64), (300, 40)] {
        for sigma in [0.5f32, 2.0] {
            let Some((d, x, y)) = run(
                w,
                h,
                AdjustParams {
                    blur_sigma: sigma,
                    ..Default::default()
                },
            ) else {
                eprintln!("SKIP: no GPU adapter");
                return;
            };
            assert!(
                d <= 1,
                "{w}x{h} blur σ{sigma}: {d} levels off the constant at ({x},{y})"
            );
        }
    }
}

#[test]
#[allow(non_snake_case)]
fn unsharp_of_a_constant_image_is_the_constant__feat__image_editor_filters() {
    let Some((d, x, y)) = run(
        64,
        64,
        AdjustParams {
            sharpen_amount: 1.0,
            ..Default::default()
        },
    ) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    assert!(d <= 1, "unsharp: {d} levels off the constant at ({x},{y})");
}

/// The point stages of the same chain are fine on the same image — the
/// fault is specific to the windowed stages. Always on.
#[test]
#[allow(non_snake_case)]
fn a_point_stage_over_a_constant_image_stays_constant__feat__image_editor_filters() {
    let Some((d, x, y)) = run(
        300,
        40,
        AdjustParams {
            saturation: 1.0001,
            ..Default::default()
        },
    ) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    assert!(d <= 1, "point stage: {d} levels off at ({x},{y})");
}
