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

//! gpu↔ref parity for the cast family — alpha-association casts.
//! feat: cast.premultiply / cast.unpremultiply (registry/kernels.yaml).

use image_conformance::harness::{assert_within, parity, RefTile};
use image_conformance::Px;
use image_kernels::families::cast::{
    cast_premultiply, cast_unpremultiply, CastPremultiplyParams, CastUnpremultiplyParams,
    CAST_PREMULTIPLY, CAST_UNPREMULTIPLY,
};

/// rgb gradient with a non-zero alpha gradient — premultiply's clean
/// stimulus: every channel and alpha sweeps the unit interval, alpha
/// never exactly 0 so the cast is unambiguous.
fn rgb_alpha_gradient(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        Px([
            x as f32 / w as f32,
            y as f32 / h as f32,
            (x + y) as f32 / (w + h) as f32,
            // alpha in (0, 1], never 0.
            (x + 1) as f32 / w as f32,
        ])
    })
}

/// unpremultiply stimulus: alphas held in [0.25, 1.0] (divisors away
/// from 0, the STIMULUS RULE for division-like ops) with premultiplied
/// colour (rgb ≤ a) so the dissociated result stays in range — EXCEPT a
/// deterministic exact-zero-alpha column (x == 0) that must map to
/// all-zero on both lanes.
fn premultiplied_tile(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        if x == 0 {
            // The exact-zero-alpha lane: zero alpha → all-zero output.
            Px([0.0, 0.0, 0.0, 0.0])
        } else {
            // alpha sweeps [0.25, 1.0]; colour is premultiplied (≤ a).
            let alpha = 0.25 + 0.75 * (x as f32 / w as f32);
            Px([
                alpha * (x as f32 / w as f32),
                alpha * (y as f32 / h as f32),
                alpha * ((x + y) as f32 / (w + h) as f32),
                alpha,
            ])
        }
    })
}

#[test]
fn premultiply_parity() {
    let tile = rgb_alpha_gradient(image_core::TILE, image_core::TILE);
    let p = CastPremultiplyParams::new();
    match parity(&CAST_PREMULTIPLY, cast_premultiply, &[&tile], &p) {
        Some(r) => assert_within(r, &CAST_PREMULTIPLY),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn unpremultiply_parity() {
    let tile = premultiplied_tile(image_core::TILE, image_core::TILE);
    let p = CastUnpremultiplyParams::new();
    match parity(&CAST_UNPREMULTIPLY, cast_unpremultiply, &[&tile], &p) {
        Some(r) => assert_within(r, &CAST_UNPREMULTIPLY),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

/// A small 64×64 case that puts the exact-zero-alpha lane next to a
/// dense band of tiny alphas — the brink of the divide-by-zero guard.
#[test]
fn unpremultiply_zero_alpha_quick() {
    let tile = premultiplied_tile(64, 64);
    let p = CastUnpremultiplyParams::new();
    match parity(&CAST_UNPREMULTIPLY, cast_unpremultiply, &[&tile], &p) {
        Some(r) => assert_within(r, &CAST_UNPREMULTIPLY),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}
