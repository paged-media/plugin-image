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
//! feat: gen.solid, gen.checker, gen.linear_gradient, plus the breadth
//! batch: gen.radial_gradient, gen.angular_gradient,
//! gen.reflected_gradient, gen.diamond_gradient, gen.noise
//! (registry/kernels.yaml).
//!
//! ANGULAR SEAM NOTE. The angular gradient has a genuine c0/c1
//! discontinuity along its seam ray (rotated −x from the center). The
//! parity stimulus places the CENTER outside/below-left of the tile so
//! every texel's delta is strictly positive in both axes — the seam
//! never crosses the tile and lane rounding noise cannot flip a texel
//! across the discontinuity. The seam semantics themselves are pinned
//! by pure-reference tests (no GPU).

use image_conformance::harness::{assert_within, parity, RefTile};
use image_conformance::Px;
use image_kernels::families::gen::{
    GenAngularGradientParams, GenCheckerParams, GenDiamondGradientParams, GenLinearGradientParams,
    GenNoiseParams, GenRadialGradientParams, GenReflectedGradientParams, GenSolidParams,
    GEN_ANGULAR_GRADIENT, GEN_CHECKER, GEN_DIAMOND_GRADIENT, GEN_LINEAR_GRADIENT, GEN_NOISE,
    GEN_RADIAL_GRADIENT, GEN_REFLECTED_GRADIENT, GEN_SOLID,
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

// ──────────────────── breadth batch: radial ────────────────────────

const R0: [f32; 4] = [0.9, 0.3, 0.1, 1.0];
const R1: [f32; 4] = [0.05, 0.4, 0.85, 0.6];

fn mix4(c0: [f32; 4], c1: [f32; 4], t: f32) -> Px {
    // WGSL mix(e1, e2, e3) = e1*(1-e3) + e2*e3.
    Px([
        c0[0] * (1.0 - t) + c1[0] * t,
        c0[1] * (1.0 - t) + c1[1] * t,
        c0[2] * (1.0 - t) + c1[2] * t,
        c0[3] * (1.0 - t) + c1[3] * t,
    ])
}

fn radial_ref(a: Px, _b: Px, p: &GenRadialGradientParams) -> Px {
    let px = (p.ox + a.0[0] as i32) as f32;
    let py = (p.oy + a.0[1] as i32) as f32;
    let dx = px - p.cx;
    let dy = py - p.cy;
    let d = (dx * dx + dy * dy).sqrt();
    let t = if p.radius > 0.0 {
        (d / p.radius).clamp(0.0, 1.0)
    } else {
        0.0
    };
    mix4(
        [p.c0r, p.c0g, p.c0b, p.c0a],
        [p.c1r, p.c1g, p.c1b, p.c1a],
        t,
    )
}

#[test]
fn gen_radial_gradient_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    let dummy = coord_tile(w, h);
    // Center mid-tile, radius reaching the corners: t spans (0, 1].
    let p = GenRadialGradientParams::new(0, 0, 127.5, 127.5, 180.0, R0, R1);
    match parity(&GEN_RADIAL_GRADIENT, radial_ref, &[&dummy], &p) {
        Some(r) => {
            eprintln!("gen.radial_gradient: measured max f16 ULP {}", r.max_ulp);
            assert_within(r, &GEN_RADIAL_GRADIENT)
        }
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_radial_gradient_parity_offset_origin() {
    // Non-zero origin proves the global-coordinate continuity.
    let (w, h) = (64u32, 48u32);
    let dummy = coord_tile(w, h);
    let p = GenRadialGradientParams::new(200, 100, 220.0, 130.0, 90.0, R0, R1);
    match parity(&GEN_RADIAL_GRADIENT, radial_ref, &[&dummy], &p) {
        Some(r) => {
            eprintln!(
                "gen.radial_gradient (offset): measured max f16 ULP {}",
                r.max_ulp
            );
            assert_within(r, &GEN_RADIAL_GRADIENT)
        }
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_radial_gradient_center_and_degenerate() {
    // Pure-reference sanity: the exact center is c0 (t = 0); far outside
    // the radius clamps to c1; radius <= 0 collapses to c0.
    let p = GenRadialGradientParams::new(0, 0, 10.0, 10.0, 50.0, R0, R1);
    assert_eq!(
        radial_ref(Px([10.0, 10.0, 0.0, 0.0]), Px([0.0; 4]), &p).0,
        R0
    );
    assert_eq!(
        radial_ref(Px([200.0, 10.0, 0.0, 0.0]), Px([0.0; 4]), &p).0,
        R1
    );
    let deg = GenRadialGradientParams::new(0, 0, 10.0, 10.0, 0.0, R0, R1);
    assert_eq!(
        radial_ref(Px([200.0, 10.0, 0.0, 0.0]), Px([0.0; 4]), &deg).0,
        R0
    );
}

// ──────────────────── breadth batch: angular ───────────────────────
//
// The reference mirrors the module's OWN deterministic atan2
// (atan_poly + octant fold + quadrant fix) term-for-term — see the
// module comment in families/gen.rs.

// The f32 std consts are bit-identical to the WGSL literals the module
// uses (1.5707963267948966 / 3.141592653589793 / 6.283185307179586 all
// round to the same nearest f32).
fn atan_poly_ref(z: f32) -> f32 {
    let z2 = z * z;
    z * (0.999_866 + z2 * (-0.3302995 + z2 * (0.180_141 + z2 * (-0.0851330 + z2 * 0.0208351))))
}

fn atan2_det_ref(y: f32, x: f32) -> f32 {
    let ax = x.abs();
    let ay = y.abs();
    let mut base = if ax >= ay {
        let r = if ax > 0.0 { ay / ax } else { 0.0 };
        atan_poly_ref(r)
    } else {
        std::f32::consts::FRAC_PI_2 - atan_poly_ref(ax / ay)
    };
    if x < 0.0 {
        base = std::f32::consts::PI - base;
    }
    if y < 0.0 {
        base = -base;
    }
    base
}

fn angular_ref(a: Px, _b: Px, p: &GenAngularGradientParams) -> Px {
    let px = (p.ox + a.0[0] as i32) as f32;
    let py = (p.oy + a.0[1] as i32) as f32;
    let dx = px - p.cx;
    let dy = py - p.cy;
    let ca = p.angle.cos();
    let sa = p.angle.sin();
    let rx = dx * ca + dy * sa;
    let ry = dy * ca - dx * sa;
    let theta = atan2_det_ref(ry, rx);
    let t = ((theta + std::f32::consts::PI) / std::f32::consts::TAU).clamp(0.0, 1.0);
    mix4(
        [p.c0r, p.c0g, p.c0b, p.c0a],
        [p.c1r, p.c1g, p.c1b, p.c1a],
        t,
    )
}

#[test]
fn gen_angular_gradient_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    let dummy = coord_tile(w, h);
    // Center below-left of the tile → all deltas strictly positive →
    // the seam (−x ray) never crosses the tile (see module doc).
    let p = GenAngularGradientParams::new(0, 0, -20.0, -20.0, 0.0, R0, R1);
    match parity(&GEN_ANGULAR_GRADIENT, angular_ref, &[&dummy], &p) {
        Some(r) => {
            eprintln!("gen.angular_gradient: measured max f16 ULP {}", r.max_ulp);
            assert_within(r, &GEN_ANGULAR_GRADIENT)
        }
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_angular_gradient_parity_offset_rotated() {
    // Offset origin + a rotation small enough that the rotated frame
    // keeps rx > 0 over the whole (first-quadrant) delta range: deltas
    // dx, dy >= 40 and angle 0.35 rad → rx = dx·cos + dy·sin > 0.
    let (w, h) = (64u32, 48u32);
    let dummy = coord_tile(w, h);
    let p = GenAngularGradientParams::new(100, 50, 60.0, 10.0, 0.35, R0, R1);
    match parity(&GEN_ANGULAR_GRADIENT, angular_ref, &[&dummy], &p) {
        Some(r) => {
            eprintln!(
                "gen.angular_gradient (offset+rotated): measured max f16 ULP {}",
                r.max_ulp
            );
            assert_within(r, &GEN_ANGULAR_GRADIENT)
        }
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_angular_gradient_axes_and_seam() {
    // Pure-reference sanity: with angle 0, the +x axis is the sweep
    // midpoint (t = 0.5), +y is 0.75, −y is 0.25, and the −x seam is
    // t = 1 (θ = +π by the dy = 0 branch).
    let p = GenAngularGradientParams::new(0, 0, 100.0, 100.0, 0.0, R0, R1);
    let t_of = |x: f32, y: f32| {
        let out = angular_ref(Px([x, y, 0.0, 0.0]), Px([0.0; 4]), &p);
        // Recover t from the alpha lerp (c0a = 1.0, c1a = 0.6).
        (out.0[3] - 1.0) / (0.6 - 1.0)
    };
    assert!((t_of(150.0, 100.0) - 0.5).abs() < 1e-5, "+x → 0.5");
    assert!((t_of(100.0, 150.0) - 0.75).abs() < 1e-4, "+y → 0.75");
    assert!((t_of(100.0, 50.0) - 0.25).abs() < 1e-4, "−y → 0.25");
    assert!((t_of(50.0, 100.0) - 1.0).abs() < 1e-5, "−x seam → 1");
    // The polynomial is a true atan approximation: 45° diagonal → 0.625.
    assert!(
        (t_of(150.0, 150.0) - 0.625).abs() < 1e-4,
        "diagonal → 0.625"
    );
}

// ─────────────────── breadth batch: reflected ──────────────────────

fn reflected_ref(a: Px, _b: Px, p: &GenReflectedGradientParams) -> Px {
    let px = (p.ox + a.0[0] as i32) as f32;
    let py = (p.oy + a.0[1] as i32) as f32;
    let dx = px - p.x0;
    let dy = py - p.y0;
    let ex = p.x1 - p.x0;
    let ey = p.y1 - p.y0;
    let dd = ex * ex + ey * ey;
    let t = if dd > 0.0 {
        ((dx * ex + dy * ey) / dd).abs().clamp(0.0, 1.0)
    } else {
        0.0
    };
    mix4(
        [p.c0r, p.c0g, p.c0b, p.c0a],
        [p.c1r, p.c1g, p.c1b, p.c1a],
        t,
    )
}

#[test]
fn gen_reflected_gradient_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    let dummy = coord_tile(w, h);
    // Mirror line mid-tile: both signs of the projection occur.
    let p = GenReflectedGradientParams::new(0, 0, 127.0, 0.0, 227.0, 0.0, R0, R1);
    match parity(&GEN_REFLECTED_GRADIENT, reflected_ref, &[&dummy], &p) {
        Some(r) => {
            eprintln!("gen.reflected_gradient: measured max f16 ULP {}", r.max_ulp);
            assert_within(r, &GEN_REFLECTED_GRADIENT)
        }
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_reflected_gradient_parity_offset_diagonal() {
    let (w, h) = (64u32, 48u32);
    let dummy = coord_tile(w, h);
    let p = GenReflectedGradientParams::new(32, 16, 60.0, 40.0, 160.0, 90.0, R0, R1);
    match parity(&GEN_REFLECTED_GRADIENT, reflected_ref, &[&dummy], &p) {
        Some(r) => {
            eprintln!(
                "gen.reflected_gradient (offset): measured max f16 ULP {}",
                r.max_ulp
            );
            assert_within(r, &GEN_REFLECTED_GRADIENT)
        }
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_reflected_gradient_mirrors_about_p0() {
    // Pure-reference sanity: equal distances on either side of p0 give
    // the SAME color (t = |s|); the p0 line itself is c0.
    let p = GenReflectedGradientParams::new(0, 0, 100.0, 0.0, 200.0, 0.0, R0, R1);
    let plus = reflected_ref(Px([150.0, 0.0, 0.0, 0.0]), Px([0.0; 4]), &p);
    let minus = reflected_ref(Px([50.0, 0.0, 0.0, 0.0]), Px([0.0; 4]), &p);
    assert_eq!(plus.0, minus.0, "mirrored points agree");
    assert_eq!(
        reflected_ref(Px([100.0, 0.0, 0.0, 0.0]), Px([0.0; 4]), &p).0,
        R0
    );
    // Both endpoints (±|e| from p0) are c1.
    assert_eq!(
        reflected_ref(Px([200.0, 0.0, 0.0, 0.0]), Px([0.0; 4]), &p).0,
        R1
    );
    assert_eq!(
        reflected_ref(Px([0.0, 0.0, 0.0, 0.0]), Px([0.0; 4]), &p).0,
        R1
    );
}

// ──────────────────── breadth batch: diamond ───────────────────────

fn diamond_ref(a: Px, _b: Px, p: &GenDiamondGradientParams) -> Px {
    let px = (p.ox + a.0[0] as i32) as f32;
    let py = (p.oy + a.0[1] as i32) as f32;
    let dx = px - p.cx;
    let dy = py - p.cy;
    let ca = p.angle.cos();
    let sa = p.angle.sin();
    let rx = dx * ca + dy * sa;
    let ry = dy * ca - dx * sa;
    let t = if p.scale > 0.0 {
        ((rx.abs() + ry.abs()) / p.scale).clamp(0.0, 1.0)
    } else {
        0.0
    };
    mix4(
        [p.c0r, p.c0g, p.c0b, p.c0a],
        [p.c1r, p.c1g, p.c1b, p.c1a],
        t,
    )
}

#[test]
fn gen_diamond_gradient_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    let dummy = coord_tile(w, h);
    let p = GenDiamondGradientParams::new(0, 0, 127.5, 127.5, 0.0, 200.0, R0, R1);
    match parity(&GEN_DIAMOND_GRADIENT, diamond_ref, &[&dummy], &p) {
        Some(r) => {
            eprintln!("gen.diamond_gradient: measured max f16 ULP {}", r.max_ulp);
            assert_within(r, &GEN_DIAMOND_GRADIENT)
        }
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_diamond_gradient_parity_offset_rotated() {
    let (w, h) = (64u32, 48u32);
    let dummy = coord_tile(w, h);
    let p = GenDiamondGradientParams::new(64, 32, 90.0, 55.0, 0.6, 80.0, R0, R1);
    match parity(&GEN_DIAMOND_GRADIENT, diamond_ref, &[&dummy], &p) {
        Some(r) => {
            eprintln!(
                "gen.diamond_gradient (offset+rotated): measured max f16 ULP {}",
                r.max_ulp
            );
            assert_within(r, &GEN_DIAMOND_GRADIENT)
        }
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_diamond_gradient_contours() {
    // Pure-reference sanity (angle 0): the center is c0; the four axis
    // points at L1 distance `scale` are all exactly c1; a diagonal point
    // at (s/2, s/2) has the SAME t as an axis point at s (L1 metric).
    let p = GenDiamondGradientParams::new(0, 0, 100.0, 100.0, 0.0, 40.0, R0, R1);
    assert_eq!(
        diamond_ref(Px([100.0, 100.0, 0.0, 0.0]), Px([0.0; 4]), &p).0,
        R0
    );
    for (x, y) in [(140.0, 100.0), (60.0, 100.0), (100.0, 140.0), (100.0, 60.0)] {
        assert_eq!(diamond_ref(Px([x, y, 0.0, 0.0]), Px([0.0; 4]), &p).0, R1);
    }
    let diag = diamond_ref(Px([120.0, 120.0, 0.0, 0.0]), Px([0.0; 4]), &p);
    assert_eq!(diag.0, R1, "L1 contour: (20, 20) delta == 40 L1 == scale");
}

// ───────────────────── breadth batch: noise ────────────────────────
//
// The reference mirrors the module's PCG hash with wrapping u32 ops —
// BIT-identical to the WGSL lane by construction.

fn pcg_ref(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

fn noise_ref(a: Px, _b: Px, p: &GenNoiseParams) -> Px {
    let gx = (p.ox + a.0[0] as i32) as u32;
    let gy = (p.oy + a.0[1] as i32) as u32;
    // 2⁻²⁴ exactly — the same value the WGSL literal
    // 5.9604644775390625e-8 parses to.
    const INV_2_24: f32 = 1.0 / 16_777_216.0;
    let h = pcg_ref(gx ^ pcg_ref(gy ^ pcg_ref(p.seed)));
    let n = (h >> 8) as f32 * INV_2_24;
    let v = n * p.amount;
    Px([v, v, v, 1.0])
}

#[test]
fn gen_noise_parity_tile() {
    let (w, h) = (image_core::TILE, image_core::TILE);
    let dummy = coord_tile(w, h);
    let p = GenNoiseParams::new(0, 0, 0xC0FF_EE00, 1.0);
    match parity(&GEN_NOISE, noise_ref, &[&dummy], &p) {
        Some(r) => {
            eprintln!("gen.noise: measured max f16 ULP {}", r.max_ulp);
            assert_within(r, &GEN_NOISE)
        }
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_noise_parity_offset_amount() {
    let (w, h) = (64u32, 48u32);
    let dummy = coord_tile(w, h);
    let p = GenNoiseParams::new(1000, -32, 7, 0.35);
    match parity(&GEN_NOISE, noise_ref, &[&dummy], &p) {
        Some(r) => {
            eprintln!("gen.noise (offset): measured max f16 ULP {}", r.max_ulp);
            assert_within(r, &GEN_NOISE)
        }
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn gen_noise_is_deterministic_seeded_and_bounded() {
    // Pure-reference sanity: same (seed, x, y) → same value; a
    // different seed changes the field; values lie in [0, amount).
    let p = GenNoiseParams::new(0, 0, 42, 0.8);
    let p2 = GenNoiseParams::new(0, 0, 43, 0.8);
    let mut any_diff = false;
    for y in 0..16 {
        for x in 0..16 {
            let px = Px([x as f32, y as f32, 0.0, 0.0]);
            let a = noise_ref(px, Px([0.0; 4]), &p);
            let b = noise_ref(px, Px([0.0; 4]), &p);
            assert_eq!(a.0, b.0, "deterministic per (seed, x, y)");
            assert!(a.0[0] >= 0.0 && a.0[0] < 0.8, "bounded by amount");
            assert_eq!(a.0[0], a.0[1]);
            assert_eq!(a.0[1], a.0[2]);
            assert_eq!(a.0[3], 1.0);
            if noise_ref(px, Px([0.0; 4]), &p2).0[0] != a.0[0] {
                any_diff = true;
            }
        }
    }
    assert!(any_diff, "seed changes the field");
}

// Touch the KernelDef import so the `class`/tolerance metadata stays
// linked to the test (and keeps the import non-dead).
#[test]
fn gen_kernels_are_generators_exact_where_declared() {
    use image_kernels::{KernelClass, Tolerance};
    for def in [
        &GEN_SOLID,
        &GEN_CHECKER,
        &GEN_LINEAR_GRADIENT,
        &GEN_RADIAL_GRADIENT,
        &GEN_ANGULAR_GRADIENT,
        &GEN_REFLECTED_GRADIENT,
        &GEN_DIAMOND_GRADIENT,
        &GEN_NOISE,
    ] {
        let _d: &KernelDef = def;
        assert!(matches!(def.class, KernelClass::Generator));
        assert!(!def.mip_exact, "{} is coordinate-absolute", def.id);
    }
    // solid/checker write arbitrary f32 color constants into an
    // rgba16float texture; the f32-uniform→f16-store rounding can differ
    // from the CPU f16 conversion by 1 ULP for non-f16-exact values
    // (§6.3 — GPU output is never byte-golden), so ChannelEpsF16(1), not
    // Exact. (Colors that ARE f16-exact, e.g. 0/0.5/1, round-trip at 0.)
    // noise is products of exact values → the same 1-ULP store bound.
    assert_eq!(GEN_SOLID.gpu_tolerance, Tolerance::ChannelEpsF16(1));
    assert_eq!(GEN_CHECKER.gpu_tolerance, Tolerance::ChannelEpsF16(1));
    assert_eq!(GEN_NOISE.gpu_tolerance, Tolerance::ChannelEpsF16(1));
    // The gradient shapes divide/normalize in f32 → the family's 4-ULP
    // envelope (like linear).
    for def in [
        &GEN_LINEAR_GRADIENT,
        &GEN_RADIAL_GRADIENT,
        &GEN_ANGULAR_GRADIENT,
        &GEN_REFLECTED_GRADIENT,
        &GEN_DIAMOND_GRADIENT,
    ] {
        assert_eq!(def.gpu_tolerance, Tolerance::ChannelEpsF16(4), "{}", def.id);
    }
}
