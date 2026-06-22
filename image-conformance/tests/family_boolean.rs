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

//! gpu↔ref parity for the boolean family (spec §11): `and`/`or`/`xor`/
//! `not` over the `> 0.5` truthiness rule. The stimulus straddles 0.5
//! (the truthiness line) on every channel and includes texels where the
//! two inputs are bit-equal, so each branch of the logic is exercised.
//! Outputs are exact 0.0/1.0 ⇒ Tolerance::Exact must hold to 0 ULP.
//! feat: bool.and / bool.or / bool.xor / bool.not (registry/kernels.yaml).

use image_conformance::harness::{assert_within, parity, RefTile};
use image_conformance::Px;
use image_kernels::families::boolean::{
    bool_and, bool_not, bool_or, bool_xor, BoolAndParams, BoolNotParams, BoolOrParams,
    BoolXorParams, BOOL_AND, BOOL_NOT, BOOL_OR, BOOL_XOR,
};

/// Five canonical truthiness probes: clearly-false, the exact 0.5 line
/// (false: `> 0.5` is strict), just-over the line (true), and clearly-
/// true at the working-space extremes. All f16-exact and finite.
const PROBES: [f32; 5] = [0.0, 0.5, 0.625, 1.0, 0.25];

/// Tile A walks the probes along x; tile B walks them along y. The two
/// agree on the diagonal (a == b — the equal-input case) and disagree
/// off it, so every truth-table cell is covered across the tile. A
/// constant true alpha keeps the channel mix realistic.
fn probe_tile_a(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, _y| {
        let v = PROBES[(x as usize) % PROBES.len()];
        Px([v, PROBES[(x as usize + 1) % PROBES.len()], v, 1.0])
    })
}

fn probe_tile_b(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        let v = PROBES[(y as usize) % PROBES.len()];
        // x feeds one channel so the a==b diagonal (x==y) is hit too.
        Px([v, PROBES[(x as usize) % PROBES.len()], 0.0, 1.0])
    })
}

fn binary<P: bytemuck::Pod>(
    def: &'static image_kernels::KernelDef,
    refn: impl Fn(Px, Px, &P) -> Px,
    p: &P,
) {
    let a = probe_tile_a(image_core::TILE, image_core::TILE);
    let b = probe_tile_b(image_core::TILE, image_core::TILE);
    match parity(def, refn, &[&a, &b], p) {
        Some(r) => assert_within(r, def),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn bool_and_parity() {
    binary(&BOOL_AND, bool_and, &BoolAndParams::new());
}

#[test]
fn bool_or_parity() {
    binary(&BOOL_OR, bool_or, &BoolOrParams::new());
}

#[test]
fn bool_xor_parity() {
    binary(&BOOL_XOR, bool_xor, &BoolXorParams::new());
}

#[test]
fn bool_not_parity() {
    // Unary: a single straddling tile; b is the harness zero-vec.
    let a = probe_tile_a(image_core::TILE, image_core::TILE);
    let p = BoolNotParams::new();
    match parity(&BOOL_NOT, bool_not, &[&a], &p) {
        Some(r) => assert_within(r, &BOOL_NOT),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}
