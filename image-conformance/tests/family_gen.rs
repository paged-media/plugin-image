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

//! gpu↔ref parity for the generator family (T2, spec §11) — `gen.solid`,
//! `gen.checker`, `gen.linear_gradient`. Each is a `module: true` UNARY
//! kernel (the M2 zero-input convention, `families::gen` docs): the
//! shader derives the output texel's GLOBAL coordinate from `gid` +
//! `params.{ox, oy}` and NEVER samples `in0`.
//!
//! HARNESS NOTE. The point `parity()` reference is `Fn(Px, Px, &P) -> Px`
//! — it sees only the (dummy) input pixel, not the texel coordinate. So
//! the dummy `in0` tile is SEEDED with the LOCAL coordinate
//! `Px([x, y, 0, 0])`; the scalar reference recovers `(x, y)` from it and
//! computes `gx = ox + x`, `gy = oy + y` — the SAME global coordinate the
//! shader derives from `gid`. Local coords 0..255 (< TILE) are integers
//! exactly representable in f16, so the seeded values round-trip through
//! the harness's f16 quantization losslessly and the two lanes agree.
//!
//! HANDWRITTEN scalar references (the DSL can't express gid-derived
//! coords) mirror the WGSL coordinate math term-for-term.
//!
//! feat: gen.solid, gen.checker, gen.linear_gradient
//! (registry/kernels.yaml).

use image_conformance::harness::{assert_within, parity, RefTile};
use image_conformance::Px;
use image_kernels::families::gen::{
    GenCheckerParams, GenLinearGradientParams, GenSolidParams, GEN_CHECKER, GEN_LINEAR_GRADIENT,
    GEN_SOLID,
};
use image_kernels::KernelDef;

/// A dummy input tile that ENCODES local coords in (r, g): pixel (x, y) =
/// Px([x, y, 0, 0]). The generators never sample in0 on the GPU; the
/// scalar reference reads (x, y) back from here to mirror gid.
fn coord_tile(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| Px([x as f32, y as f32, 0.0, 0.0]))
}

// ─────────────────────────────── solid ─────────────────────────────

fn solid_ref(_a: Px, _b: Px, p: &GenSolidParams) -> Px {
    Px([p.r, p.g, p.b, p.a])
}

#[test]
fn gen_solid_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    let dummy = coord_tile(w, h);
    let p = GenSolidParams::new(0, 0, 0.25, 0.5, 0.75, 1.0);
    match parity(&GEN_SOLID, solid_ref, &[&dummy], &p) {
        Some(r) => assert_within(r, &GEN_SOLID),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_solid_parity_offset_origin() {
    // A non-zero origin must NOT change a constant fill (coordinate
    // independence) — exact across tiles.
    let (w, h) = (64u32, 48u32);
    let dummy = coord_tile(w, h);
    let p = GenSolidParams::new(1000, -7, 0.1, 0.2, 0.3, 0.4);
    match parity(&GEN_SOLID, solid_ref, &[&dummy], &p) {
        Some(r) => assert_within(r, &GEN_SOLID),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

// ────────────────────────────── checker ────────────────────────────
//
// gx = ox + x, gy = oy + y (all non-negative in the stimulus);
// cell = ((gx/size + gy/size) & 1); cell 0 → c0, cell 1 → c1.

const C0: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const C1: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

fn checker_ref(a: Px, _b: Px, p: &GenCheckerParams) -> Px {
    let gx = (p.ox + a.0[0] as i32) as u32;
    let gy = (p.oy + a.0[1] as i32) as u32;
    let cell = ((gx / p.size) + (gy / p.size)) & 1;
    let c = if cell == 1 { C1 } else { C0 };
    Px(c)
}

#[test]
fn gen_checker_parity_tile_origin() {
    // Origin (0, 0): texel (0, 0) is cell 0 (c0) — the parity anchor.
    let (w, h) = (image_core::TILE, image_core::TILE);
    let dummy = coord_tile(w, h);
    let p = GenCheckerParams::new(0, 0, 8, C0, C1);
    match parity(&GEN_CHECKER, checker_ref, &[&dummy], &p) {
        Some(r) => assert_within(r, &GEN_CHECKER),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_checker_parity_offset_origin() {
    // Offset origin proves ox/oy continuity: an origin that is an ODD
    // multiple of `size` flips the cell parity relative to (0,0), so the
    // GPU must read the SAME global grid the reference does. (ox = 8 =
    // 1·size shifts cell parity by 1.)
    let (w, h) = (64u32, 48u32);
    let dummy = coord_tile(w, h);
    let p = GenCheckerParams::new(8, 16, 8, C0, C1);
    match parity(&GEN_CHECKER, checker_ref, &[&dummy], &p) {
        Some(r) => assert_within(r, &GEN_CHECKER),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_checker_origin_anchor_is_c0() {
    // Pure-reference sanity (no GPU): texel (0,0) at origin (0,0) is cell
    // 0 → c0; the diagonal neighbor cell (size, size) is cell 0 again;
    // (size, 0) is cell 1 → c1. Pins the documented selection rule.
    let p = GenCheckerParams::new(0, 0, 8, C0, C1);
    assert_eq!(
        checker_ref(Px([0.0, 0.0, 0.0, 0.0]), Px([0.0; 4]), &p).0,
        C0
    );
    assert_eq!(
        checker_ref(Px([8.0, 0.0, 0.0, 0.0]), Px([0.0; 4]), &p).0,
        C1
    );
    assert_eq!(
        checker_ref(Px([8.0, 8.0, 0.0, 0.0]), Px([0.0; 4]), &p).0,
        C0
    );
}

// ──────────────────────── linear_gradient ──────────────────────────
//
// p = (ox + x, oy + y); t = clamp(dot(p-p0, p1-p0)/|p1-p0|², 0, 1);
// out = mix(c0, c1, t) (premultiplied). dd == 0 → t = 0.

const G0: [f32; 4] = [0.1, 0.2, 0.3, 0.5];
const G1: [f32; 4] = [0.9, 0.7, 0.5, 1.0];

fn gradient_ref(a: Px, _b: Px, p: &GenLinearGradientParams) -> Px {
    let px = (p.ox + a.0[0] as i32) as f32;
    let py = (p.oy + a.0[1] as i32) as f32;
    let dx = px - p.x0;
    let dy = py - p.y0;
    let ex = p.x1 - p.x0;
    let ey = p.y1 - p.y0;
    let dd = ex * ex + ey * ey;
    let t = if dd > 0.0 {
        ((dx * ex + dy * ey) / dd).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let c0 = Px([p.c0r, p.c0g, p.c0b, p.c0a]);
    let c1 = Px([p.c1r, p.c1g, p.c1b, p.c1a]);
    // WGSL mix(e1, e2, e3) = e1*(1-e3) + e2*e3.
    Px([
        c0.0[0] * (1.0 - t) + c1.0[0] * t,
        c0.0[1] * (1.0 - t) + c1.0[1] * t,
        c0.0[2] * (1.0 - t) + c1.0[2] * t,
        c0.0[3] * (1.0 - t) + c1.0[3] * t,
    ])
}

#[test]
fn gen_linear_gradient_parity_tile() {
    // Horizontal gradient across the tile: p0 = (0,0), p1 = (255,0), so
    // t sweeps 0..1 across x. Endpoints land exactly: texel (0,*) → c0
    // (t=0), texel (255,*) → c1 (t=1).
    let (w, h) = (image_core::TILE, image_core::TILE);
    let dummy = coord_tile(w, h);
    let p = GenLinearGradientParams::new(0, 0, 0.0, 0.0, 255.0, 0.0, G0, G1);
    match parity(&GEN_LINEAR_GRADIENT, gradient_ref, &[&dummy], &p) {
        Some(r) => assert_within(r, &GEN_LINEAR_GRADIENT),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_linear_gradient_parity_offset_diagonal() {
    // Diagonal gradient with a non-zero origin — proves ox/oy feed the
    // dot product (continuity) and exercises the interior t∈(0,1) range.
    let (w, h) = (64u32, 48u32);
    let dummy = coord_tile(w, h);
    let p = GenLinearGradientParams::new(32, 16, 0.0, 0.0, 200.0, 120.0, G0, G1);
    match parity(&GEN_LINEAR_GRADIENT, gradient_ref, &[&dummy], &p) {
        Some(r) => assert_within(r, &GEN_LINEAR_GRADIENT),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_linear_gradient_endpoints_and_degenerate() {
    // Pure-reference sanity (no GPU): t=0 endpoint → c0, t=1 endpoint →
    // c1, beyond-p1 clamps to c1, and degenerate endpoints (p0 == p1)
    // collapse to t=0 → c0. Pins the documented endpoint contract.
    let p = GenLinearGradientParams::new(0, 0, 0.0, 0.0, 100.0, 0.0, G0, G1);
    // Origin texel (0,0): t = 0 → c0.
    assert_eq!(
        gradient_ref(Px([0.0, 0.0, 0.0, 0.0]), Px([0.0; 4]), &p).0,
        G0
    );
    // p1 endpoint (100,0): t = 1 → c1.
    assert_eq!(
        gradient_ref(Px([100.0, 0.0, 0.0, 0.0]), Px([0.0; 4]), &p).0,
        G1
    );
    // Beyond p1 clamps to c1.
    assert_eq!(
        gradient_ref(Px([200.0, 0.0, 0.0, 0.0]), Px([0.0; 4]), &p).0,
        G1
    );
    // Degenerate endpoints → t = 0 → c0.
    let deg = GenLinearGradientParams::new(0, 0, 5.0, 5.0, 5.0, 5.0, G0, G1);
    assert_eq!(
        gradient_ref(Px([9.0, 9.0, 0.0, 0.0]), Px([0.0; 4]), &deg).0,
        G0
    );
}

// Touch the KernelDef import so the `class`/tolerance metadata stays
// linked to the test (and keeps the import non-dead).
#[test]
fn gen_kernels_are_generators_exact_where_declared() {
    use image_kernels::{KernelClass, Tolerance};
    for def in [&GEN_SOLID, &GEN_CHECKER, &GEN_LINEAR_GRADIENT] {
        let _d: &KernelDef = def;
        assert!(matches!(def.class, KernelClass::Generator));
        assert!(!def.mip_exact, "{} is coordinate-absolute", def.id);
    }
    // solid/checker write arbitrary f32 color constants into an
    // rgba16float texture; the f32-uniform→f16-store rounding can differ
    // from the CPU f16 conversion by 1 ULP for non-f16-exact values
    // (§6.3 — GPU output is never byte-golden), so ChannelEpsF16(1), not
    // Exact. (Colors that ARE f16-exact, e.g. 0/0.5/1, round-trip at 0.)
    assert_eq!(GEN_SOLID.gpu_tolerance, Tolerance::ChannelEpsF16(1));
    assert_eq!(GEN_CHECKER.gpu_tolerance, Tolerance::ChannelEpsF16(1));
    assert_eq!(
        GEN_LINEAR_GRADIENT.gpu_tolerance,
        Tolerance::ChannelEpsF16(4)
    );
}
