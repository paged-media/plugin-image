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

//! `gen.pattern` — Photoshop's Edit ▸ Fill ▸ Pattern and the pattern
//! half of the paint bucket: a RASTER fill that tiles a source bitmap
//! into pixels. These are the DEFINITION-level gates that do not need a
//! GPU: the ABI contract (including the `in1` second input and the mask
//! epilogue), the param encoding, the family registration, and a scalar
//! model of the shader that pins every behavioural claim its
//! doc-comment makes — above all THE SEAM.
//!
//! feat: gen.pattern (registry/kernels.yaml — row PENDING, orchestrator
//! wiring; until it lands `image_kernels::lookup("gen.pattern")` is
//! `None` and the crate's `registry_and_definitions_agree` unit test
//! reports it as a code-defined kernel with no row).
//!
//! NOT A SWATCH. This kernel writes pixels; it is not the vector
//! pattern-paint of RFI C-31 (a new engine paint kind + tiling in both
//! renderer backends + an IDML round-trip decision). Nothing here needs
//! a core change, which is why it can exist at all.
//!
//! WHY A SCALAR MODEL LIVES IN THIS FILE. The gpu↔ref parity for this
//! family runs in `image-conformance/tests/family_gen.rs`, which this
//! agent does not own. The model below mirrors the WGSL term-for-term —
//! same centre convention, same inverse phase/rotate/scale order, same
//! float fold, same INDEPENDENT integer wrap of the two tap indices,
//! same `mix(mix(p00,p10,tx), mix(p01,p11,tx), ty)` blend order, same
//! premultiplied source-over and mask epilogue — so the assertions are
//! meaningful now and the same reference transplants into the
//! conformance harness unchanged.

use image_kernels::families::gen::{GenPatternParams, GEN_PATTERN};
use image_kernels::families::{gen, ALL_FAMILIES};
use image_kernels::{KernelClass, Tolerance};

// ─────────────────── scalar model of the shader ───────────────────

type Px = [f32; 4];

/// `mix(a, b, t)` = `a*(1-t) + b*t`, WGSL's definition. At `t == 0.0`
/// this is bit-exactly `a` — the fact the bit-exact-tiling claim rests
/// on.
fn mix(a: Px, b: Px, t: f32) -> Px {
    std::array::from_fn(|i| a[i] * (1.0 - t) + b[i] * t)
}

/// Euclidean modulo — WGSL's `%` on i32 truncates (sign follows the
/// dividend), exactly as Rust's does, so this mirrors `wrapi` in the
/// module character for character.
fn wrapi(v: i32, n: i32) -> i32 {
    ((v % n) + n) % n
}

/// How a tap index is resolved. `Wrap` is the shipped shader; `Clamp`
/// is the WRONG implementation this file exists to rule out — it is
/// modelled so the seam test can show what the hairline would look
/// like rather than merely asserting the right number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tap {
    Wrap,
    Clamp,
}

impl Tap {
    fn resolve(self, v: i32, n: i32) -> i32 {
        match self {
            Tap::Wrap => wrapi(v, n),
            Tap::Clamp => v.clamp(0, n - 1),
        }
    }
}

/// The tile source: a `tw`×`th` pattern living in the top-left of a
/// `tex_w`×`tex_h` texture (the `in1` binding), row-major.
struct Tile<'a> {
    px: &'a [Px],
    tex_w: i32,
    tex_h: i32,
}

impl Tile<'_> {
    fn at(&self, x: i32, y: i32) -> Px {
        self.px[(y * self.tex_w + x) as usize]
    }
}

/// The tile extent the shader actually uses: params win, 0 means "the
/// whole texture", anything larger than the texture is clamped to it,
/// and the divisor is never zero.
fn extent(tile: &Tile<'_>, p: &GenPatternParams) -> (i32, i32) {
    let mut tw = p.tile_w as i32;
    let mut th = p.tile_h as i32;
    if tw <= 0 || tw > tile.tex_w {
        tw = tile.tex_w;
    }
    if th <= 0 || th > tile.tex_h {
        th = tile.tex_h;
    }
    (tw.max(1), th.max(1))
}

/// One bilinear tile sample at the continuous TILE-space coordinate
/// `(ux, uy)`. This is the seam-critical half of the kernel, factored
/// out so the tests can drive it at an arbitrary coordinate instead of
/// only through whole images.
fn sample_tile(tile: &Tile<'_>, tw: i32, th: i32, ux: f32, uy: f32, tap: Tap) -> Px {
    // Fold only to BOUND the magnitude before the i32 conversion.
    let twf = tw as f32;
    let thf = th as f32;
    let wx = ux - (ux / twf).floor() * twf;
    let wy = uy - (uy / thf).floor() * thf;

    // Texel-centre convention, then the two taps resolved INDEPENDENTLY.
    let fx = wx - 0.5;
    let fy = wy - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = fx - x0;
    let ty = fy - y0;
    let ix0 = tap.resolve(x0 as i32, tw);
    let iy0 = tap.resolve(y0 as i32, th);
    let ix1 = tap.resolve(x0 as i32 + 1, tw);
    let iy1 = tap.resolve(y0 as i32 + 1, th);

    let top = mix(tile.at(ix0, iy0), tile.at(ix1, iy0), tx);
    let bot = mix(tile.at(ix0, iy1), tile.at(ix1, iy1), tx);
    mix(top, bot, ty)
}

/// The coordinate chain: destination texel → tile space.
fn to_tile_space(p: &GenPatternParams, s: f32, x: i32, y: i32) -> (f32, f32) {
    let px = (p.ox + x) as f32 + 0.5;
    let py = (p.oy + y) as f32 + 0.5;
    let qx = px - p.offset_x;
    let qy = py - p.offset_y;
    // The module writes π/180 to full f64 digits; this truncation is
    // the SAME f32 (bit pattern 0x3C8EFA35), it is only spelled to the
    // precision f32 can hold so the mirror stays exact.
    let ang = p.angle_deg * 0.017_453_292;
    let ca = ang.cos();
    let sa = ang.sin();
    ((qx * ca + qy * sa) / s, (qy * ca - qx * sa) / s)
}

/// The whole kernel, mask epilogue included. `mask` is the constant
/// selection coverage (the ABI's group-2 texture, `.r`).
///
/// The negated comparisons mirror the shader's guards character for
/// character; clippy's suggested `<= 0.0` rewrite would send NaN down
/// the other branch and desynchronise the reference from the WGSL,
/// which is the one thing a reference twin may never do.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn pattern_model(
    dst: &[Px],
    w: i32,
    h: i32,
    tile: &Tile<'_>,
    p: &GenPatternParams,
    mask: f32,
    tap: Tap,
) -> Vec<Px> {
    let (tw, th) = extent(tile, p);

    // The degenerate guard, mirrored: a non-positive or NaN scale is
    // neutralised into the identity rather than dividing to infinity,
    // and opacity is range-guarded by a POSITIVE TEST rather than by
    // clamp so that a NaN is the identity on both lanes.
    let mut s = p.scale;
    if !(s > 0.0) {
        s = 1.0;
    }
    let mut op = 0.0;
    if p.opacity > 0.0 {
        op = p.opacity.min(1.0);
    }
    if !(p.scale > 0.0) {
        op = 0.0;
    }

    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let (ux, uy) = to_tile_space(p, s, x, y);
            let pat = sample_tile(tile, tw, th, ux, uy, tap);
            let a = dst[(y * w + x) as usize];
            // Premultiplied source-over.
            let src: Px = std::array::from_fn(|i| pat[i] * op);
            let result: Px = std::array::from_fn(|i| src[i] + a[i] * (1.0 - src[3]));
            out.push(mix(a, result, mask));
        }
    }
    out
}

// ───────────────────────── fixtures ────────────────────────────────

const W: i32 = 9;
const H: i32 = 7;

/// A COLUMN-striped 4×4 tile: constant down each column, and the first
/// and last columns are as far apart as the range allows. That contrast
/// is what makes a clamped seam visible as a hairline rather than as
/// rounding noise. Opaque, premultiplied, and every value is f16-exact.
fn striped_tile() -> Vec<Px> {
    const COLS: [f32; 4] = [0.0, 0.25, 0.5, 1.0];
    (0..16)
        .map(|i| {
            let c = COLS[(i % 4) as usize];
            [c, c, c, 1.0]
        })
        .collect()
}

/// A labeled destination: each texel encodes its own coordinate, so
/// "the destination came through unchanged" is checkable per texel.
/// f16-exact values.
fn labeled(w: i32, h: i32) -> Vec<Px> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            [
                x as f32 / 256.0,
                y as f32 / 256.0,
                (x + y) as f32 / 512.0,
                1.0,
            ]
        })
        .collect()
}

// ───────────────────────── IDENTITY ────────────────────────────────

/// THE identity condition from the doc-comment: `opacity == 0` returns
/// the destination UNCHANGED — for every scale, angle, phase and mask
/// value, BIT-exactly, not within a tolerance. This is what lets an
/// opacity slider start at zero without the fill already having damaged
/// the layer.
#[test]
fn gen_pattern_zero_opacity_is_the_identity() {
    let dst = labeled(W, H);
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    for (scale, angle) in [(1.0f32, 0.0f32), (2.5, 37.0), (0.3, -180.0), (17.0, 90.0)] {
        for (offx, offy) in [(0.0f32, 0.0f32), (3.0, -2.0), (0.5, 0.25)] {
            // Mask values chosen so `a*(1-m) + a*m` is exact in binary
            // for the fixture's dyadic texel values — the identity is a
            // bit-equality claim, so it must not be tested through a
            // rounding accident.
            for mask in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
                let p = GenPatternParams::new(0, 0, scale, angle, offx, offy, 0.0, 4, 4);
                assert!(p.is_identity());
                let out = pattern_model(&dst, W, H, &tile, &p, mask, Tap::Wrap);
                assert_eq!(
                    out, dst,
                    "opacity=0 must return the destination unchanged \
                     (scale={scale}, angle={angle}, offset=({offx},{offy}), mask={mask})"
                );
            }
        }
    }
}

/// An OUT-OF-RANGE opacity arriving from JS degrades to the identity,
/// never to an inverse fill and never to NaN. The NaN case is the one
/// that matters: the shader range-guards with `if (opacity > 0)` rather
/// than `clamp`, because `clamp`/`min` of a NaN is implementation-
/// defined in WGSL while the comparison is false on every lane — so a
/// NaN slider value cannot poison the layer.
#[test]
fn gen_pattern_out_of_range_opacity_degrades_to_the_identity() {
    let dst = labeled(W, H);
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    for op in [-0.0f32, -0.5, -1.0, f32::NEG_INFINITY, f32::NAN] {
        let p = GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, op, 4, 4);
        assert!(p.is_identity(), "opacity {op} must read as a no-op");
        assert_eq!(pattern_model(&dst, W, H, &tile, &p, 1.0, Tap::Wrap), dst);
    }

    // Above 1 saturates at a full-strength fill; it must not push the
    // premultiplied source past its own alpha.
    let full = GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 4, 4);
    for op in [1.5f32, 100.0, f32::INFINITY] {
        let over = GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, op, 4, 4);
        assert_eq!(
            pattern_model(&dst, W, H, &tile, &over, 1.0, Tap::Wrap),
            pattern_model(&dst, W, H, &tile, &full, 1.0, Tap::Wrap),
            "opacity {op} must saturate at 1, not overdrive the fill"
        );
    }
}

/// `scale == 0` is DEGENERATE and is neutralised into a no-op — the
/// documented guard. Dividing by it would send the lattice coordinate
/// to infinity, where the shader's `i32` conversion is undefined; a
/// no-op fill is the only answer that is both safe and honest.
/// Negative and NaN scales take the same branch (the guard is a NEGATED
/// comparison precisely so NaN falls into it).
#[test]
fn gen_pattern_non_positive_scale_is_a_no_op() {
    let dst = labeled(W, H);
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    for scale in [0.0f32, -0.0, -1.0, -1e30, f32::NAN] {
        let p = GenPatternParams::new(0, 0, scale, 12.0, 1.5, -0.5, 1.0, 4, 4);
        assert!(p.is_identity(), "scale {scale} must read as a no-op");
        let out = pattern_model(&dst, W, H, &tile, &p, 1.0, Tap::Wrap);
        assert_eq!(
            out, dst,
            "scale={scale} must leave the destination untouched, not produce garbage"
        );
        for v in out.iter().flatten() {
            assert!(v.is_finite(), "scale={scale} produced a non-finite texel");
        }
    }
}

/// `is_identity()` is the engine's dispatch-skip predicate, so it must
/// agree with the shader on EVERY case, in both directions — a false
/// positive silently drops a real fill.
#[test]
fn gen_pattern_is_identity_predicate_agrees_with_the_shader() {
    let dst = labeled(W, H);
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    for scale in [1.0f32, 0.0, -2.0, f32::NAN, 3.5] {
        for op in [0.0f32, -1.0, f32::NAN, 0.5, 1.0] {
            let p = GenPatternParams::new(0, 0, scale, 20.0, 0.0, 0.0, op, 4, 4);
            let unchanged = pattern_model(&dst, W, H, &tile, &p, 1.0, Tap::Wrap) == dst;
            assert_eq!(
                p.is_identity(),
                unchanged,
                "is_identity() disagrees with the shader at scale={scale}, opacity={op}"
            );
        }
    }
    assert!(GenPatternParams::identity(4, 4).is_identity());
}

// ─────────────────────────── THE SEAM ──────────────────────────────

/// THE SEAM, stated as a number. With the phase placing the lattice
/// boundary exactly on a texel centre, the sample there must be the
/// 50/50 blend of the tile's LAST column with its FIRST — 0.5 for this
/// fixture. A clamped tap gives the first column blended with itself
/// (0.0): a one-pixel hairline of edge colour along every lattice line,
/// which is the classic broken pattern fill.
#[test]
fn gen_pattern_seam_blends_the_last_column_with_the_first() {
    let dst = vec![[0.0, 0.0, 0.0, 1.0]; (W * H) as usize];
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    // offset_x = 0.5 puts ux = 0 exactly at x = 0, i.e. the tile
    // boundary lands on that texel's centre.
    let p = GenPatternParams::new(0, 0, 1.0, 0.0, 0.5, 0.0, 1.0, 4, 4);

    let wrapped = pattern_model(&dst, W, H, &tile, &p, 1.0, Tap::Wrap);
    let clamped = pattern_model(&dst, W, H, &tile, &p, 1.0, Tap::Clamp);

    // Columns 0.0 / 0.25 / 0.5 / 1.0 sampled at tx = 0.5 between
    // successive columns, with column 0 following column 3 THROUGH the
    // seam.
    let expected = [0.5f32, 0.125, 0.375, 0.75];
    for y in 0..H {
        for x in 0..W {
            let got = wrapped[(y * W + x) as usize][0];
            let want = expected[(x % 4) as usize];
            assert!(
                (got - want).abs() < 1e-6,
                "({x},{y}) expected {want}, got {got}"
            );
        }
    }

    // …and the clamped implementation is WRONG exactly at the seam
    // columns and right everywhere else — which is why the bug looks
    // like a hairline and not like a broken pattern.
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            if x % 4 == 0 {
                assert!(
                    (clamped[i][0] - 0.0).abs() < 1e-6,
                    "the clamped tap must collapse the seam to the edge colour"
                );
                assert_ne!(
                    wrapped[i], clamped[i],
                    "({x},{y}) is a seam texel; wrap and clamp must differ"
                );
            } else {
                assert_eq!(
                    wrapped[i], clamped[i],
                    "({x},{y}) is interior; the tap policy must not matter"
                );
            }
        }
    }
}

/// The seam is CONTINUOUS, not merely defined: crossing the tile
/// boundary by a hair must change the sample by a hair. This is the
/// stronger statement — a wrap that produced the right value only
/// exactly ON the boundary would still tear beside it.
#[test]
fn gen_pattern_seam_is_continuous_across_the_boundary() {
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    let eps = 1e-4f32;

    let before = sample_tile(&tile, 4, 4, -eps, 2.5, Tap::Wrap)[0];
    let after = sample_tile(&tile, 4, 4, eps, 2.5, Tap::Wrap)[0];
    assert!(
        (before - after).abs() < 1e-3,
        "the wrapped sample must be continuous through the seam: \
         {before} vs {after}"
    );
    // Both sit halfway between the last column (1.0) and the first
    // (0.0) — the sample is genuinely blending across the boundary, not
    // snapping to one side.
    for v in [before, after] {
        assert!((v - 0.5).abs() < 1e-3, "expected ~0.5 at the seam, got {v}");
    }

    // The clamped tap TEARS: a 1e-4 step in the coordinate flips the
    // sample the full width of the range.
    let c_before = sample_tile(&tile, 4, 4, -eps, 2.5, Tap::Clamp)[0];
    let c_after = sample_tile(&tile, 4, 4, eps, 2.5, Tap::Clamp)[0];
    assert!(
        (c_before - c_after).abs() > 0.9,
        "the clamped tap should tear at the seam — this test is only \
         meaningful if it does ({c_before} vs {c_after})"
    );
}

/// The vertical seam behaves identically to the horizontal one: the
/// wrap is applied per axis, so a ROW-striped tile wraps its last row
/// onto its first.
#[test]
fn gen_pattern_seam_wraps_on_both_axes() {
    // Transpose the fixture: constant along each ROW.
    const ROWS: [f32; 4] = [0.0, 0.25, 0.5, 1.0];
    let px: Vec<Px> = (0..16)
        .map(|i| {
            let c = ROWS[(i / 4) as usize];
            [c, c, c, 1.0]
        })
        .collect();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    let v = sample_tile(&tile, 4, 4, 2.5, 0.0, Tap::Wrap)[0];
    assert!(
        (v - 0.5).abs() < 1e-6,
        "the vertical seam must blend row 3 (1.0) with row 0 (0.0); got {v}"
    );
}

/// The lattice is PERIODIC in the tile extent: sliding the phase by a
/// whole tile reproduces the same pixels exactly. A wrap that were even
/// slightly off would drift here.
#[test]
fn gen_pattern_period_is_the_tile_extent() {
    let dst = labeled(W, H);
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    // DYADIC phases (0.25 / −0.75): the claim is a BIT-equality, and a
    // decimal like 0.3 is not the same f32 as 0.3 + 4.0 − 4.0, so a
    // non-dyadic fixture would be testing float rounding, not the wrap.
    let base = GenPatternParams::new(0, 0, 1.0, 0.0, 0.25, -0.75, 1.0, 4, 4);
    let shifted = GenPatternParams::new(0, 0, 1.0, 0.0, 4.25, 3.25, 1.0, 4, 4);
    assert_eq!(
        pattern_model(&dst, W, H, &tile, &base, 1.0, Tap::Wrap),
        pattern_model(&dst, W, H, &tile, &shifted, 1.0, Tap::Wrap),
        "a phase shift of one whole tile must be the identity"
    );
}

// ────────────────────── tiling behaviour ───────────────────────────

/// The case that has to be LOSSLESS: at `scale == 1`, `angle_deg == 0`
/// and integer phase, the half-pixel centre conversions cancel, so
/// `tx == ty == 0.0` exactly and the blend collapses to a single tap.
/// The fill copies the tile texel-for-texel — bit-exactly, not "within
/// tolerance". Without this, placing a 1:1 pattern would soften it.
#[test]
fn gen_pattern_unrotated_integer_phase_tiles_bit_exactly() {
    let dst = vec![[0.0, 0.0, 0.0, 1.0]; (W * H) as usize];
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    for (offx, offy) in [(0.0f32, 0.0f32), (1.0, 3.0), (-5.0, 2.0)] {
        let p = GenPatternParams::new(0, 0, 1.0, 0.0, offx, offy, 1.0, 4, 4);
        let out = pattern_model(&dst, W, H, &tile, &p, 1.0, Tap::Wrap);
        for y in 0..H {
            for x in 0..W {
                let sx = wrapi(x - offx as i32, 4);
                let sy = wrapi(y - offy as i32, 4);
                assert_eq!(
                    out[(y * W + x) as usize],
                    tile.at(sx, sy),
                    "({x},{y}) must be the tile texel verbatim (phase {offx},{offy})"
                );
            }
        }
    }
}

/// A SCALED pattern is bilinear, not rounded. At `scale == 2` a
/// destination texel lands on a half-integer tile coordinate, so the
/// sample must sit strictly BETWEEN two source columns. If it were
/// rounded it would equal one of them, and the lattice edges would come
/// out as stair-steps.
#[test]
fn gen_pattern_scaled_sample_is_bilinear_not_rounded() {
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    // ux = 1.25 → halfway-ish between columns 0 (0.0) and 1 (0.25).
    let v = sample_tile(&tile, 4, 4, 1.25, 2.5, Tap::Wrap)[0];
    let expected = 0.25 * 0.0 + 0.75 * 0.25;
    assert!(
        (v - expected).abs() < 1e-6,
        "expected the 25/75 blend {expected}, got {v}"
    );
    assert!(
        v > 0.0 && v < 0.25,
        "a rounded sample would equal a source column; got {v}"
    );
}

/// `angle_deg` actually rotates the LATTICE. A column-striped tile at
/// 90° must come out row-striped — constant along x, varying along y.
/// The inverse-rotation sign convention is what makes it +90° on screen
/// rather than −90°.
#[test]
fn gen_pattern_angle_rotates_the_lattice() {
    let dst = vec![[0.0, 0.0, 0.0, 1.0]; (W * H) as usize];
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    let straight = GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 4, 4);
    let turned = GenPatternParams::new(0, 0, 1.0, 90.0, 0.0, 0.0, 1.0, 4, 4);
    let a = pattern_model(&dst, W, H, &tile, &straight, 1.0, Tap::Wrap);
    let b = pattern_model(&dst, W, H, &tile, &turned, 1.0, Tap::Wrap);
    assert_ne!(a, b, "the angle must not be decorative");

    for y in 0..H {
        for x in 1..W {
            let here = b[(y * W + x) as usize][0];
            let left = b[(y * W) as usize][0];
            assert!(
                (here - left).abs() < 1e-4,
                "at 90° the stripes must run along x; ({x},{y}) broke that"
            );
        }
    }
    let col: Vec<f32> = (0..H).map(|y| b[(y * W) as usize][0]).collect();
    assert!(
        col.windows(2).any(|w| (w[0] - w[1]).abs() > 0.1),
        "at 90° the stripes must vary along y; got {col:?}"
    );
}

/// The lattice is continuous ACROSS ENGINE TILES. Rendering a region in
/// one dispatch and rendering it as two dispatches with their own
/// `(ox, oy)` origins must produce identical pixels — otherwise every
/// 256² engine tile would restart the pattern and the fill would look
/// shredded at tile boundaries.
#[test]
fn gen_pattern_global_origin_keeps_the_lattice_continuous() {
    let dst = labeled(W, H);
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    let p = GenPatternParams::new(0, 0, 1.7, 23.0, 0.4, -1.1, 1.0, 4, 4);
    let whole = pattern_model(&dst, W, H, &tile, &p, 1.0, Tap::Wrap);

    // The bottom half re-rendered as its own "engine tile" at oy = 3.
    const SPLIT: i32 = 3;
    let lower_dst: Vec<Px> = dst[(SPLIT * W) as usize..].to_vec();
    let mut lower_p = p;
    lower_p.oy = SPLIT;
    let lower = pattern_model(&lower_dst, W, H - SPLIT, &tile, &lower_p, 1.0, Tap::Wrap);

    assert_eq!(
        lower,
        whole[(SPLIT * W) as usize..].to_vec(),
        "the pattern must not restart at an engine-tile boundary"
    );
}

// ─────────────────── composite / tile extent ───────────────────────

/// SOURCE-OVER, NOT A LERP. Where the tile is transparent the
/// destination must show THROUGH, unfaded. A straight
/// `mix(dst, pattern, opacity)` would instead erase the destination
/// toward transparent wherever the pattern has a hole — the difference
/// between filling with a stencil and punching one.
#[test]
fn gen_pattern_holes_let_the_destination_through() {
    let dst = vec![[1.0, 0.0, 0.0, 1.0]; (W * H) as usize];
    // Checker of opaque white / fully transparent, premultiplied.
    let px: Vec<Px> = (0..16)
        .map(|i| {
            let (x, y) = (i % 4, i / 4);
            if (x + y) % 2 == 0 {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.0, 0.0, 0.0, 0.0]
            }
        })
        .collect();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    let p = GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 4, 4);
    let out = pattern_model(&dst, W, H, &tile, &p, 1.0, Tap::Wrap);
    for y in 0..H {
        for x in 0..W {
            let hole = (x % 4 + y % 4) % 2 == 1;
            let got = out[(y * W + x) as usize];
            if hole {
                assert_eq!(
                    got,
                    [1.0, 0.0, 0.0, 1.0],
                    "({x},{y}) is a hole; the destination must survive it"
                );
            } else {
                assert_eq!(got, [1.0, 1.0, 1.0, 1.0]);
            }
        }
    }
}

/// For an OPAQUE tile the source-over composite degenerates to exactly
/// the lerp a user expects from an opacity slider — bit-exactly, so the
/// more-correct formula costs nothing in the common case.
#[test]
fn gen_pattern_opaque_tile_composite_is_the_plain_lerp() {
    let dst = labeled(W, H);
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    let p = GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, 0.5, 4, 4);
    let out = pattern_model(&dst, W, H, &tile, &p, 1.0, Tap::Wrap);
    for y in 0..H {
        for x in 0..W {
            let pat = tile.at(wrapi(x, 4), wrapi(y, 4));
            let a = dst[(y * W + x) as usize];
            assert_eq!(out[(y * W + x) as usize], mix(a, pat, 0.5));
        }
    }
}

/// The mask epilogue is what scopes the fill to a SELECTION: mask 0
/// leaves the destination alone, mask 1 takes the fill whole, and a
/// partial mask lands between the two. Without it "fill the selection
/// with a pattern" would be "fill the layer".
#[test]
fn gen_pattern_mask_scopes_the_fill() {
    let dst = labeled(W, H);
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    let p = GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 4, 4);
    let off = pattern_model(&dst, W, H, &tile, &p, 0.0, Tap::Wrap);
    let on = pattern_model(&dst, W, H, &tile, &p, 1.0, Tap::Wrap);
    let half = pattern_model(&dst, W, H, &tile, &p, 0.5, Tap::Wrap);
    assert_eq!(off, dst, "mask 0 is outside the selection: no fill at all");
    assert_ne!(on, dst);
    for i in 0..(W * H) as usize {
        assert_eq!(half[i], mix(dst[i], on[i], 0.5));
    }
}

/// `tile_w`/`tile_h` of 0 means "the whole `in1` texture" — the
/// degrade-rather-than-fail default for a caller that uploaded the tile
/// at its exact size and has nothing extra to say.
#[test]
fn gen_pattern_zero_extent_means_the_whole_texture() {
    let dst = labeled(W, H);
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    let explicit = GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 4, 4);
    let implied = GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0);
    assert_eq!(
        pattern_model(&dst, W, H, &tile, &explicit, 1.0, Tap::Wrap),
        pattern_model(&dst, W, H, &tile, &implied, 1.0, Tap::Wrap)
    );
}

/// An extent LARGER than the bound texture is clamped to the texture:
/// no tap may leave `in1`, whatever the caller claims about the tile.
#[test]
fn gen_pattern_oversized_extent_clamps_to_the_texture() {
    let dst = labeled(W, H);
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    for (tw, th) in [(9u32, 9u32), (4, 40), (u32::MAX, u32::MAX)] {
        let bogus = GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, 1.0, tw, th);
        assert_eq!(extent(&tile, &bogus), (4, 4), "extent {tw}x{th} must clamp");
        assert_eq!(
            pattern_model(&dst, W, H, &tile, &bogus, 1.0, Tap::Wrap),
            pattern_model(
                &dst,
                W,
                H,
                &tile,
                &GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 4, 4),
                1.0,
                Tap::Wrap
            )
        );
    }
}

/// A SUB-RECT of the texture is a legitimate tile: the pattern is the
/// top-left `tile_w`×`tile_h`, and the wrap uses that extent, not the
/// texture's. This is what lets a tile live in a padded upload.
#[test]
fn gen_pattern_sub_rect_of_the_texture_is_the_tile() {
    let px = striped_tile();
    let tile = Tile {
        px: &px,
        tex_w: 4,
        tex_h: 4,
    };
    // A 2×4 tile: columns 0.0 / 0.25 only, so the seam blends 0.25 → 0.0.
    let v = sample_tile(&tile, 2, 4, 0.0, 2.5, Tap::Wrap)[0];
    assert!(
        (v - 0.125).abs() < 1e-6,
        "the 2-wide sub-rect must wrap column 1 onto column 0; got {v}"
    );
    // …and the period is 2, not 4 (dyadic coordinates, so this is an
    // exact equality rather than a float-rounding coincidence).
    let a = sample_tile(&tile, 2, 4, 0.75, 2.5, Tap::Wrap);
    let b = sample_tile(&tile, 2, 4, 2.75, 2.5, Tap::Wrap);
    assert_eq!(a, b);
}

// ───────────────────── params encoding / layout ────────────────────

/// The uniform block: 48 bytes, `ox | oy | scale | angle_deg |
/// offset_x | offset_y | opacity | tile_w | tile_h | _pad0 | _pad1 |
/// _abi_pad`, and the `ParamsLayout` field list matches the struct
/// field-for-field IN ORDER (the WGSL `struct Params` is handwritten,
/// so this plus the module-text check below is all that holds the two
/// together).
#[test]
fn gen_pattern_params_encoding() {
    assert_eq!(std::mem::size_of::<GenPatternParams>(), 48);
    assert_eq!(
        GEN_PATTERN.params.size,
        std::mem::size_of::<GenPatternParams>()
    );
    assert_eq!(GEN_PATTERN.params.size % 16, 0, "16-aligned uniform block");

    let names: Vec<&str> = GEN_PATTERN.params.fields.iter().map(|f| f.name).collect();
    let types: Vec<&str> = GEN_PATTERN
        .params
        .fields
        .iter()
        .map(|f| f.wgsl_ty)
        .collect();
    assert_eq!(
        names,
        [
            "ox",
            "oy",
            "scale",
            "angle_deg",
            "offset_x",
            "offset_y",
            "opacity",
            "tile_w",
            "tile_h",
            "_pad0",
            "_pad1",
        ]
    );
    assert_eq!(
        types,
        ["i32", "i32", "f32", "f32", "f32", "f32", "f32", "u32", "u32", "u32", "u32"]
    );
    // fields + the trailing `_abi_pad`, all 4-byte scalars.
    assert_eq!(GEN_PATTERN.params.size, 4 * (names.len() + 1));

    let p = GenPatternParams::new(3, -4, 2.5, -90.0, 1.5, -2.25, 0.75, 64, 32);
    let b = p.as_bytes();
    assert_eq!(b.len(), 48);
    assert_eq!(&b[0..4], &3i32.to_le_bytes());
    assert_eq!(&b[4..8], &(-4i32).to_le_bytes());
    assert_eq!(&b[8..12], &2.5f32.to_le_bytes());
    assert_eq!(&b[12..16], &(-90.0f32).to_le_bytes());
    assert_eq!(&b[16..20], &1.5f32.to_le_bytes());
    assert_eq!(&b[20..24], &(-2.25f32).to_le_bytes());
    assert_eq!(&b[24..28], &0.75f32.to_le_bytes());
    assert_eq!(&b[28..32], &64u32.to_le_bytes());
    assert_eq!(&b[32..36], &32u32.to_le_bytes());
    assert_eq!(&b[36..40], &0u32.to_le_bytes(), "_pad0 is always 0");
    assert_eq!(&b[40..44], &0u32.to_le_bytes(), "_pad1 is always 0");
    assert_eq!(&b[44..48], &0u32.to_le_bytes(), "_abi_pad is always 0");

    // Byte identity IS param identity (the op-cache key, spec §6.2), so
    // every knob must participate — a phase change that hashed equal
    // would serve a stale tile from the cache.
    let base = GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 8, 8);
    for other in [
        GenPatternParams::new(1, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 8, 8),
        GenPatternParams::new(0, 1, 1.0, 0.0, 0.0, 0.0, 1.0, 8, 8),
        GenPatternParams::new(0, 0, 2.0, 0.0, 0.0, 0.0, 1.0, 8, 8),
        GenPatternParams::new(0, 0, 1.0, 1.0, 0.0, 0.0, 1.0, 8, 8),
        GenPatternParams::new(0, 0, 1.0, 0.0, 1.0, 0.0, 1.0, 8, 8),
        GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 1.0, 1.0, 8, 8),
        GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, 0.5, 8, 8),
        GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 4, 8),
        GenPatternParams::new(0, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 8, 4),
    ] {
        assert_ne!(base.as_bytes(), other.as_bytes());
    }
}

// ──────────────────── registration + ABI contract ──────────────────

/// Registered in `gen::FAMILY` — and therefore in `ALL_FAMILIES`, which
/// is what `all_defined()` and the registry-drift gate read.
#[test]
fn gen_pattern_is_registered_in_the_family() {
    assert!(
        gen::FAMILY.iter().any(|d| d.id == "gen.pattern"),
        "gen.pattern must be in gen::FAMILY"
    );
    assert!(
        ALL_FAMILIES
            .iter()
            .flat_map(|f| f.iter())
            .any(|d| d.id == "gen.pattern"),
        "gen.pattern must be reachable from ALL_FAMILIES"
    );
    // Exactly once — a duplicated entry would double every count the
    // registry quotes.
    assert_eq!(
        ALL_FAMILIES
            .iter()
            .flat_map(|f| f.iter())
            .filter(|d| d.id == "gen.pattern")
            .count(),
        1
    );
}

/// The metadata the engines dispatch on. `inputs: 2` is the load-bearing
/// one: without a second input there is nowhere for the tile bitmap to
/// live and the kernel could only draw procedural cells.
#[test]
fn gen_pattern_kernel_metadata() {
    assert_eq!(GEN_PATTERN.id, "gen.pattern");
    assert_eq!(
        GEN_PATTERN.inputs, 2,
        "in0 = the destination layer, in1 = the pattern tile"
    );
    assert!(GEN_PATTERN.module, "handwritten ABI v1.1 module");
    assert!(
        !GEN_PATTERN.mip_exact,
        "the lattice is coordinate-absolute, so a mip level must re-derive its params"
    );
    assert_eq!(
        GEN_PATTERN.class,
        KernelClass::Point,
        "the ROI relation for in0 is 1:1; in1 is a whole-texture resource, \
         not an inflated window"
    );
    assert_eq!(GEN_PATTERN.gpu_tolerance, Tolerance::ChannelEpsF16(4));
}

/// The ABI v1.1 binding contract, checked against the module TEXT: the
/// four groups INCLUDING the second input at `@group(0) @binding(1)`,
/// the workgroup size, the bounds prologue, and the mandatory mask
/// epilogue. `abi::assemble` returns a `module: true` kernel's WGSL
/// verbatim, so this is the shipped source.
#[test]
fn gen_pattern_wgsl_honours_the_abi() {
    let src = image_kernels::abi::assemble(&GEN_PATTERN);
    for needle in [
        "@group(0) @binding(0) var in0 : texture_2d<f32>;",
        "@group(0) @binding(1) var in1 : texture_2d<f32>;",
        "@group(1) @binding(0) var<uniform> params : Params;",
        "@group(2) @binding(0) var mask : texture_2d<f32>;",
        "@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;",
        "@compute @workgroup_size(16, 16, 1)",
        "if (gid.x >= dims.x || gid.y >= dims.y) { return; }",
        "let m = textureLoad(mask, xy, 0).r;",
        "textureStore(outp, xy, mix(a, result, vec4<f32>(m)));",
    ] {
        assert!(src.contains(needle), "missing from the module: {needle}");
    }
    assert!(
        !src.contains("textureSample"),
        "inputs are textureLoad-only under the ABI (no samplers); the \
         bilinear footprint is computed, not sampled"
    );
    // The destination is read from in0 and the tile from in1 — swapping
    // them would compile and silently tile the layer over the pattern.
    assert!(src.contains("let a = textureLoad(in0, xy, 0);"));
    assert!(src.contains("textureLoad(in1, vec2<i32>(ix0, iy0), 0)"));
}

/// The handwritten WGSL `struct Params` matches `ParamsLayout` field for
/// field in order, plus the trailing `_abi_pad`. A drift here is a
/// silently misdecoded uniform block, which is why it gets its own gate.
#[test]
fn gen_pattern_wgsl_struct_matches_the_layout() {
    let src = image_kernels::abi::assemble(&GEN_PATTERN);
    let body = src
        .split_once("struct Params {")
        .expect("module declares struct Params")
        .1
        .split_once('}')
        .expect("struct Params closes")
        .0;
    let decls: Vec<String> = body
        .lines()
        .map(|l| l.trim().trim_end_matches(',').to_string())
        .filter(|l| !l.is_empty())
        .collect();

    let mut expected: Vec<String> = GEN_PATTERN
        .params
        .fields
        .iter()
        .map(|f| format!("{}: {}", f.name, f.wgsl_ty))
        .collect();
    expected.push("_abi_pad: u32".to_string());
    assert_eq!(decls, expected);
}

/// Naga parses and validates the module — the real gate for handwritten
/// WGSL (the crate-wide `wgsl_validate` suite covers this too; keeping a
/// local copy means a broken module fails in the file that owns it).
#[test]
fn gen_pattern_wgsl_validates() {
    let src = image_kernels::abi::assemble(&GEN_PATTERN);
    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("gen.pattern: WGSL parse failed: {e}\n{src}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("gen.pattern: WGSL validation failed: {e:?}\n{src}"));
}
