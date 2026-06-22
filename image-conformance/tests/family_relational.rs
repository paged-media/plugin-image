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

//! gpu↔ref parity for the relational family (T0, spec §11): the six
//! per-channel comparisons. Each emits an exact 0.0/1.0 mask, so parity
//! must hold at Tolerance::Exact (zero ULP slack). feat: rel.eq /
//! rel.ne / rel.lt / rel.le / rel.gt / rel.ge (registry/kernels.yaml).
//!
//! Stimulus is built so the comparison branch is exercised on BOTH
//! sides per channel: texels where a == b exactly (the GPU and the
//! reference consume the SAME f16-quantized value, so equality is real)
//! and texels straddling — both above and below — the 0.5 truthiness
//! line. Values are FINITE (the M0 stimulus rule); f16-representable
//! tenths keep the a == b lane bit-exact after quantization.

use image_conformance::harness::{assert_within, parity, RefTile};
use image_conformance::Px;
use image_kernels::families::relational::{
    rel_eq, rel_ge, rel_gt, rel_le, rel_lt, rel_ne, RelEqParams, RelGeParams, RelGtParams,
    RelLeParams, RelLtParams, RelNeParams, REL_EQ, REL_GE, REL_GT, REL_LE, REL_LT, REL_NE,
};

/// `a`: a smooth gradient that crosses the 0.5 line within each tile.
fn lhs_tile(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        Px([
            x as f32 / w as f32, // 0 → ~1 across the row
            y as f32 / h as f32, // 0 → ~1 down the column
            (x + y) as f32 / (w + h) as f32,
            1.0,
        ])
    })
}

/// `b`: equal to `a` on a diagonal stripe (the a == b lane), strictly
/// above it on one side and strictly below on the other. The ±0.25
/// step is exactly f16-representable, so where the stripe makes the two
/// tiles agree they agree bit-for-bit after quantization.
fn rhs_tile(w: u32, h: u32) -> RefTile {
    let a = lhs_tile(w, h);
    RefTile::from_fn(w, h, |x, y| {
        let base = a.px[(y * w + x) as usize];
        let delta = match (x + y) % 3 {
            0 => 0.0,   // equal: drives the == / <= / >= true branch
            1 => 0.25,  // b > a: drives < and the false branch of >
            _ => -0.25, // b < a: drives > and the false branch of <
        };
        Px([
            base.0[0] + delta,
            base.0[1] - delta,
            base.0[2] + delta,
            base.0[3], // alpha equal everywhere ⇒ the == lane is always hit
        ])
    })
}

/// Run every relational op over one (lhs, rhs) pair at Exact tolerance.
fn check(w: u32, h: u32) {
    let a = lhs_tile(w, h);
    let b = rhs_tile(w, h);
    let ins: &[&RefTile] = &[&a, &b];

    macro_rules! one {
        ($def:expr, $refn:expr, $params:expr) => {
            match parity(&$def, $refn, ins, &$params) {
                Some(r) => assert_within(r, &$def),
                None => eprintln!("SKIP: no GPU adapter"),
            }
        };
    }

    one!(REL_EQ, rel_eq, RelEqParams::new());
    one!(REL_NE, rel_ne, RelNeParams::new());
    one!(REL_LT, rel_lt, RelLtParams::new());
    one!(REL_LE, rel_le, RelLeParams::new());
    one!(REL_GT, rel_gt, RelGtParams::new());
    one!(REL_GE, rel_ge, RelGeParams::new());
}

#[test]
fn relational_parity() {
    check(image_core::TILE, image_core::TILE);
}

/// Smaller quick case — same straddle/equality coverage, cheaper to run
/// when iterating without the full 256² tile.
#[test]
fn relational_parity_small() {
    check(64, 64);
}
