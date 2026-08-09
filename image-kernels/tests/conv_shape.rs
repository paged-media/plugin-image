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

//! `conv.shape` — Photoshop's Filter ▸ Blur ▸ Shape Blur, the last
//! unbuilt row in the Blur family: a convolution whose kernel is an
//! ARBITRARY user-supplied shape rather than a formula. These are the
//! DEFINITION-level gates that do not need a GPU: the ABI contract
//! (including the `in1` second input, the DERIVED halo, and the mask
//! epilogue), the param encoding, the family registration, and a scalar
//! model of the shader that pins every behavioural claim its
//! doc-comment makes — above all THE NORMALISATION and the
//! sample-versus-index decision.
//!
//! feat: conv.shape (registry/kernels.yaml — row PENDING, orchestrator
//! wiring; until it lands `image_kernels::lookup("conv.shape")` is
//! `None` and the crate's `registry_and_definitions_agree` unit test
//! reports it as a code-defined kernel with no row).
//!
//! WHY A SCALAR MODEL LIVES IN THIS FILE. The gpu↔ref parity for this
//! family runs in `image-conformance/tests/family_conv.rs`, which this
//! agent does not own. The model below mirrors the WGSL term-for-term —
//! same derived halo, same guard order, same box support `[-r, +r]`,
//! same hoisted `f32(sw)/(2r)` tap→texel scale, same truncating nearest
//! pick with the closed-endpoint `min(_, sw-1)`, same positive-test
//! weight skip, same edge-replicated tap, same `acc / wsum` and the
//! same `mix(a, blurred, amount)` then mask epilogue — so the
//! assertions are meaningful now and the same reference transplants
//! into the conformance harness unchanged.
//!
//! TWO WRONG IMPLEMENTATIONS ARE MODELLED ALONGSIDE THE SHIPPED ONE
//! ([`Edge::Zero`] and [`Norm::TapCount`]), so the tests can SHOW what
//! the darkening would look like rather than merely assert the right
//! number. Both are plausible drafts; both darken.

use image_kernels::families::conv::{ConvShapeParams, CONV_SHAPE};
use image_kernels::families::{conv, ALL_FAMILIES};
use image_kernels::{KernelClass, Tolerance};

// ─────────────────── scalar model of the shader ───────────────────

type Px = [f32; 4];

/// `mix(a, b, t)` = `a*(1-t) + b*t`, WGSL's definition. At `t == 0.0`
/// this is bit-exactly `a` and at `t == 1.0` bit-exactly `b` — the two
/// facts the identity and the wet/dry claims rest on.
fn mix(a: Px, b: Px, t: f32) -> Px {
    std::array::from_fn(|i| a[i] * (1.0 - t) + b[i] * t)
}

/// The `in1` binding: an `sw`×`sh` COVERAGE silhouette living in the
/// top-left of a `tex_w`×`tex_h` texture (zero-padded by the
/// dispatcher, which sizes every input at the output dims).
struct Shape<'a> {
    px: &'a [Px],
    tex_w: i32,
    tex_h: i32,
}

impl Shape<'_> {
    /// The weight for one shape texel — the RED channel, per the
    /// kernel's documented coverage convention.
    fn coverage_at(&self, x: i32, y: i32) -> f32 {
        self.px[(y * self.tex_w + x) as usize][0]
    }
}

/// The `in0` binding: a window whose CENTRE `out_w`×`out_h` crop is the
/// destination region. Under the tiled dispatcher the window is the
/// output expanded by the radius; under the one-shot dispatcher it IS
/// the output. The shader derives which from the two texture sizes and
/// so does this.
struct Window<'a> {
    px: &'a [Px],
    win_w: i32,
    win_h: i32,
}

impl Window<'_> {
    fn at(&self, x: i32, y: i32) -> Px {
        self.px[(y * self.win_w + x) as usize]
    }
}

/// Where an out-of-window tap reads from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Edge {
    /// SHIPPED: clamp the tap into the window (edge-replicate) so it
    /// contributes a REAL sample and keeps its weight in the divisor.
    Clamp,
    /// WRONG: the "just zero-pad the window" draft — the out-of-window
    /// tap reads black but its weight still lands in the divisor, so the
    /// border fades toward transparent.
    Zero,
}

/// What the accumulated numerator is divided by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Norm {
    /// SHIPPED: the weight sum ACTUALLY gathered.
    Weights,
    /// WRONG: the number of contributing taps — a box filter's
    /// normaliser. Correct only for a binary shape, and silently darkens
    /// by the mean coverage for every other one.
    TapCount,
}

#[derive(Debug, Clone, Copy)]
struct Variant {
    edge: Edge,
    norm: Norm,
}

/// The kernel as it ships.
const SHIPPED: Variant = Variant {
    edge: Edge::Clamp,
    norm: Norm::Weights,
};

/// WRONG DRAFT 1 — the window is zero-padded instead of edge-replicated.
const ZERO_PADDED: Variant = Variant {
    edge: Edge::Zero,
    norm: Norm::Weights,
};

/// WRONG DRAFT 2 — a box filter's divisor on a coverage kernel.
const BOX_DIVISOR: Variant = Variant {
    edge: Edge::Clamp,
    norm: Norm::TapCount,
};

/// The shape extent the shader actually uses: params win, 0 means "the
/// whole texture", anything larger than the texture is clamped to it,
/// and the scale's divisor is never zero.
fn extent(shape: &Shape<'_>, p: &ConvShapeParams) -> (i32, i32) {
    let mut sw = p.shape_w as i32;
    let mut sh = p.shape_h as i32;
    if sw <= 0 || sw > shape.tex_w {
        sw = shape.tex_w;
    }
    if sh <= 0 || sh > shape.tex_h {
        sh = shape.tex_h;
    }
    (sw.max(1), sh.max(1))
}

/// The clamped, NaN-guarded radius, or `None` when the sub-half-pixel
/// identity fires. Mirrors the WGSL guard exactly, negation included.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn radius(p: &ConvShapeParams) -> Option<f32> {
    if !(p.radius_px >= 0.5) {
        return None;
    }
    Some(p.radius_px.min(R_MAX as f32))
}

/// The module's `R_MAX`, mirrored.
const R_MAX: i32 = 24;

/// THE TAP SET: every `(dx, dy, weight)` the convolution gathers, in the
/// shader's iteration order (dy outer ascending, dx inner ascending —
/// part of the determinism contract, §6.3). Factored out so the tests
/// can interrogate the WEIGHTS directly instead of only through pixels.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn taps(shape: &Shape<'_>, p: &ConvShapeParams) -> Vec<(i32, i32, f32)> {
    let Some(r) = radius(p) else {
        return Vec::new();
    };
    let ri = r.ceil() as i32;
    let (sw, sh) = extent(shape, p);

    // The hoisted tap→texel scale. Written in exactly the shader's form
    // (multiply by `sw/(2r)`, not divide by `2r` then multiply by `sw`)
    // because the two are not bit-identical in f32 and a reference twin
    // may not choose its own algebra.
    let kx = sw as f32 / (2.0 * r);
    let ky = sh as f32 / (2.0 * r);

    let mut out = Vec::new();
    for dy in -R_MAX..=R_MAX {
        if dy < -ri || dy > ri {
            continue;
        }
        let fy = dy as f32;
        if fy < -r || fy > r {
            continue;
        }
        let sy = (((fy + r) * ky) as i32).min(sh - 1);
        for dx in -R_MAX..=R_MAX {
            if dx < -ri || dx > ri {
                continue;
            }
            let fx = dx as f32;
            if fx < -r || fx > r {
                continue;
            }
            let sx = (((fx + r) * kx) as i32).min(sw - 1);
            let w = shape.coverage_at(sx, sy);
            // Positive test: zero, negative and NaN coverage all fall
            // out together, on the same side as the shader's.
            if !(w > 0.0) {
                continue;
            }
            out.push((dx, dy, w));
        }
    }
    out
}

/// The whole kernel, mask epilogue included. `mask` is the constant
/// selection coverage (the ABI's group-2 texture, `.r`).
///
/// The negated comparisons mirror the shader's guards character for
/// character; clippy's suggested `<= 0.0` rewrite would send NaN down
/// the other branch and desynchronise the reference from the WGSL,
/// which is the one thing a reference twin may never do.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn shape_model(
    win: &Window<'_>,
    out_w: i32,
    out_h: i32,
    shape: &Shape<'_>,
    p: &ConvShapeParams,
    mask: f32,
    v: Variant,
) -> Vec<Px> {
    // THE DERIVED HALO — `(dims(in0) - dims(outp)) / 2`, which is the
    // radius under the tiled dispatcher and 0 under the one-shot one.
    let hx = (win.win_w - out_w) / 2;
    let hy = (win.win_h - out_h) / 2;

    let tap_set = taps(shape, p);
    let mut out = Vec::with_capacity((out_w * out_h) as usize);

    for y in 0..out_h {
        for x in 0..out_w {
            let a = win.at(x + hx, y + hy);

            // IDENTITY 1 — a dry mix, written as a negated positive test
            // so NaN lands here. The early return bypasses the mask
            // epilogue deliberately: a no-op must be bit-exact.
            if !(p.amount > 0.0) {
                out.push(a);
                continue;
            }
            let amt = p.amount.min(1.0);

            // IDENTITY 2 — sub-half-pixel radius: the footprint is the
            // centre pixel.
            if radius(p).is_none() {
                out.push(a);
                continue;
            }

            let mut acc: Px = [0.0; 4];
            let mut wsum = 0.0f32;
            let mut count = 0u32;
            for &(dx, dy, w) in &tap_set {
                let (tx, ty) = (x + hx + dx, y + hy + dy);
                let inside = tx >= 0 && ty >= 0 && tx < win.win_w && ty < win.win_h;
                let s = match v.edge {
                    Edge::Clamp => win.at(tx.clamp(0, win.win_w - 1), ty.clamp(0, win.win_h - 1)),
                    Edge::Zero => {
                        if inside {
                            win.at(tx, ty)
                        } else {
                            [0.0; 4]
                        }
                    }
                };
                for (c, sc) in acc.iter_mut().zip(s.iter()) {
                    *c += sc * w;
                }
                wsum += w;
                count += 1;
            }

            // IDENTITY 3 — a degenerate shape gathered no weight at all.
            if !(wsum > 0.0) {
                out.push(a);
                continue;
            }
            let div = match v.norm {
                Norm::Weights => wsum,
                Norm::TapCount => count as f32,
            };
            let blurred: Px = std::array::from_fn(|i| acc[i] / div);
            let result = mix(a, blurred, amt);
            out.push(mix(a, result, mask));
        }
    }
    out
}

// ───────────────────────── fixtures ────────────────────────────────

const W: i32 = 21;
const H: i32 = 17;

/// A shape texture: `sw`×`sh` of `cov`, zero-padded into `tex`².
/// Coverage lives in RED (the documented channel); the other channels
/// are filled with a DIFFERENT value so a kernel that read the wrong
/// channel would produce visibly wrong weights instead of coincidentally
/// working.
fn shape_tex(tex: i32, sw: i32, sh: i32, cov: impl Fn(i32, i32) -> f32) -> Vec<Px> {
    (0..tex * tex)
        .map(|i| {
            let (x, y) = (i % tex, i / tex);
            if x < sw && y < sh {
                [cov(x, y), 0.125, 0.375, 1.0]
            } else {
                [0.0, 0.0, 0.0, 0.0]
            }
        })
        .collect()
}

/// A solid square shape — the shape blur's box-blur degenerate case, and
/// the fixture every "does the machinery work" assertion uses.
fn solid_shape(tex: i32, sw: i32, sh: i32) -> Vec<Px> {
    shape_tex(tex, sw, sh, |_, _| 1.0)
}

/// A labeled window: each texel encodes its own coordinate, so "the
/// input came through unchanged" is checkable per texel. f16-exact
/// dyadic values.
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

/// A CONSTANT premultiplied field. Every component is dyadic with at
/// most two mantissa bits, so `Σ w·c` over ≤ 2401 taps and the division
/// that follows are both EXACT in f32 — which lets the normalisation
/// tests assert bit-equality rather than hide behind an epsilon.
const FLAT: Px = [0.5, 0.25, 0.75, 1.0];

fn flat(w: i32, h: i32) -> Vec<Px> {
    vec![FLAT; (w * h) as usize]
}

// ───────────────────────── IDENTITY ────────────────────────────────

/// THE identity condition from the doc-comment: `amount == 0` returns
/// the input UNCHANGED — for every radius, shape and mask value,
/// BIT-exactly, not within a tolerance. This is what lets a shape-blur
/// panel open at zero without the filter having already touched the
/// layer.
#[test]
fn conv_shape_zero_amount_is_the_identity() {
    let src = labeled(W, H);
    let win = Window {
        px: &src,
        win_w: W,
        win_h: H,
    };
    let px = solid_shape(8, 5, 5);
    let shape = Shape {
        px: &px,
        tex_w: 8,
        tex_h: 8,
    };
    for radius_px in [0.5f32, 1.0, 4.0, 7.5, 24.0, 1000.0] {
        // Mask values chosen so `a*(1-m) + a*m` is exact in binary for
        // the fixture's dyadic texel values — the identity is a
        // bit-equality claim and must not be tested through a rounding
        // accident.
        for mask in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let p = ConvShapeParams::new(radius_px, 0.0, 5, 5);
            assert!(p.is_identity());
            let out = shape_model(&win, W, H, &shape, &p, mask, SHIPPED);
            assert_eq!(
                out, src,
                "amount=0 must return the input unchanged (radius={radius_px}, mask={mask})"
            );
        }
    }
}

/// An OUT-OF-RANGE amount arriving from JS degrades to the identity,
/// never to an inverse filter and never to NaN. The NaN case is the one
/// that matters: the shader range-guards with `if (amount > 0)` rather
/// than `clamp`, because `clamp`/`min` of a NaN is implementation-defined
/// in WGSL while the comparison is false on every lane — so a NaN slider
/// value cannot poison the layer.
#[test]
fn conv_shape_out_of_range_amount_degrades_to_the_identity() {
    let src = labeled(W, H);
    let win = Window {
        px: &src,
        win_w: W,
        win_h: H,
    };
    let px = solid_shape(8, 5, 5);
    let shape = Shape {
        px: &px,
        tex_w: 8,
        tex_h: 8,
    };
    for amount in [-0.0f32, -0.5, -1.0, f32::NEG_INFINITY, f32::NAN] {
        let p = ConvShapeParams::new(6.0, amount, 5, 5);
        assert!(p.is_identity(), "amount {amount} must read as a no-op");
        assert_eq!(shape_model(&win, W, H, &shape, &p, 1.0, SHIPPED), src);
    }

    // Above 1 saturates at a fully wet result rather than overdriving
    // the blend past the blurred value.
    let full = ConvShapeParams::new(6.0, 1.0, 5, 5);
    let wet = shape_model(&win, W, H, &shape, &full, 1.0, SHIPPED);
    for amount in [1.5f32, 100.0, f32::INFINITY] {
        let over = ConvShapeParams::new(6.0, amount, 5, 5);
        assert_eq!(
            shape_model(&win, W, H, &shape, &over, 1.0, SHIPPED),
            wet,
            "amount {amount} must saturate at 1, not overdrive the blend"
        );
    }
}

/// The SECOND identity: a shape scaled to under half a pixel across.
/// The box `[-r, +r]` cannot reach a second sample, so the convolution
/// is the centre pixel and the kernel says so directly instead of
/// dividing a one-tap sum by its own weight. Negative and NaN radii take
/// the same branch (the guard is a negated positive test precisely so
/// NaN falls into it).
#[test]
fn conv_shape_sub_half_pixel_radius_is_the_identity() {
    let src = labeled(W, H);
    let win = Window {
        px: &src,
        win_w: W,
        win_h: H,
    };
    let px = solid_shape(8, 5, 5);
    let shape = Shape {
        px: &px,
        tex_w: 8,
        tex_h: 8,
    };
    for radius_px in [0.0f32, -0.0, 0.25, 0.499, -3.0, f32::NAN] {
        let p = ConvShapeParams::new(radius_px, 1.0, 5, 5);
        assert!(p.is_identity(), "radius {radius_px} must read as a no-op");
        let out = shape_model(&win, W, H, &shape, &p, 1.0, SHIPPED);
        assert_eq!(
            out, src,
            "radius={radius_px} must leave the input untouched, not produce garbage"
        );
        for v in out.iter().flatten() {
            assert!(
                v.is_finite(),
                "radius={radius_px} produced a non-finite texel"
            );
        }
    }

    // …and 0.5 is on the OTHER side of the line: it is a real (if
    // minimal) filter, so the threshold is a threshold and not a
    // description of the whole low end.
    let p = ConvShapeParams::new(0.5, 1.0, 5, 5);
    assert!(!p.is_identity());
    assert_eq!(
        taps(&shape, &p).len(),
        1,
        "radius 0.5 gathers the centre tap"
    );
}

/// A DEGENERATE shape — all zeros, so no tap contributes and the weight
/// sum is 0 — must return the input rather than divide by zero. Negative
/// and NaN coverage take the same path, because the weight skip is a
/// positive test: a negative weight could drag the divisor through zero,
/// and a NaN would survive even a dry blend downstream (`mix(a, NaN, 0)`
/// is NaN, not `a`).
#[test]
fn conv_shape_degenerate_shape_returns_the_input() {
    let src = labeled(W, H);
    let win = Window {
        px: &src,
        win_w: W,
        win_h: H,
    };
    for (name, cov) in [
        ("all zero", 0.0f32),
        ("negative", -1.0),
        ("NaN", f32::NAN),
        ("negative zero", -0.0),
    ] {
        let px = shape_tex(8, 5, 5, |_, _| cov);
        let shape = Shape {
            px: &px,
            tex_w: 8,
            tex_h: 8,
        };
        let p = ConvShapeParams::new(6.0, 1.0, 5, 5);
        assert!(
            taps(&shape, &p).is_empty(),
            "{name} coverage must gather no taps"
        );
        let out = shape_model(&win, W, H, &shape, &p, 1.0, SHIPPED);
        assert_eq!(out, src, "{name} shape must return the input");
        for v in out.iter().flatten() {
            assert!(v.is_finite(), "{name} shape produced a non-finite texel");
        }
    }
}

/// `is_identity()` is the engine's dispatch-skip predicate, and the
/// direction that MATTERS is soundness: whenever it says "no-op" the
/// kernel must return the input BIT-exactly, or a skipped dispatch
/// silently drops a real blur.
///
/// It is sound but deliberately INCOMPLETE, and the gap is named rather
/// than hidden: at `0.5 <= radius_px < 1` the box `[-r, +r]` still
/// reaches only the centre tap, so the kernel computes `a·w / w` and a
/// full-strength blend of that with `a` — a no-op in effect, but only up
/// to a rounding (`x·w/w` is not bit-exactly `x` for every `w`). A
/// predicate that claimed that case would be claiming more than is true,
/// so the completeness assertion below starts at radius 1, where the
/// footprint genuinely reaches a second sample.
#[test]
fn conv_shape_identity_predicate_is_sound() {
    let src = labeled(W, H);
    let win = Window {
        px: &src,
        win_w: W,
        win_h: H,
    };
    let px = solid_shape(8, 5, 5);
    let shape = Shape {
        px: &px,
        tex_w: 8,
        tex_h: 8,
    };
    for radius_px in [f32::NAN, -1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 3.0, 24.0, 1e30] {
        for amount in [f32::NAN, -1.0, 0.0, 0.001, 0.5, 1.0, 2.0] {
            let p = ConvShapeParams::new(radius_px, amount, 5, 5);
            let unchanged = shape_model(&win, W, H, &shape, &p, 1.0, SHIPPED) == src;
            if p.is_identity() {
                assert!(
                    unchanged,
                    "FALSE POSITIVE: the predicate would skip a dispatch that changes \
                     pixels (radius={radius_px}, amount={amount})"
                );
            } else if radius_px >= 1.0 && amount > 0.0 {
                assert!(
                    !unchanged,
                    "FALSE NEGATIVE: radius={radius_px}, amount={amount} pays for a \
                     full-footprint gather that changes nothing"
                );
            }
        }
    }
}

/// The named gap, pinned so it cannot drift into something worse: a
/// radius in `[0.5, 1)` gathers EXACTLY the centre tap. That is why the
/// predicate stops short of it, and it is also why the shader's cheap
/// `radius_px < 0.5` early return is worth having — it is the largest
/// prefix of the low end that is a no-op bit-exactly.
#[test]
fn conv_shape_sub_one_pixel_radius_is_a_single_centre_tap() {
    let px = solid_shape(8, 5, 5);
    let shape = Shape {
        px: &px,
        tex_w: 8,
        tex_h: 8,
    };
    for radius_px in [0.5f32, 0.75, 0.999] {
        let p = ConvShapeParams::new(radius_px, 1.0, 5, 5);
        assert!(!p.is_identity());
        assert_eq!(
            taps(&shape, &p)
                .iter()
                .map(|&(dx, dy, _)| (dx, dy))
                .collect::<Vec<_>>(),
            vec![(0, 0)],
            "radius {radius_px} must gather the centre tap and nothing else"
        );
    }
    // Radius 1 is where a second sample first enters — a 3×3 footprint.
    let p = ConvShapeParams::new(1.0, 1.0, 5, 5);
    assert_eq!(taps(&shape, &p).len(), 9);
}

// ─────────────────── normalisation / edge overhang ─────────────────

/// A SHAPE OVERHANGING THE IMAGE EDGE MUST NOT DARKEN THE RESULT.
///
/// Convolve a constant field: every output texel — corners included,
/// where three quarters of the footprint hangs off the image — must come
/// back as that exact constant. The fixture is dyadic and the shape is
/// binary, so `Σ w·c / Σ w` is EXACT and this is a bit-equality claim.
///
/// The mechanism is the edge-replicated tap: the out-of-window sample is
/// a real pixel and its weight stays in the divisor. [`Edge::Zero`] —
/// the "just zero-pad the window" draft, where the tap reads black but
/// its weight is still counted — is modelled alongside so the test can
/// show the fade it would bake into every border instead of merely
/// asserting the number that avoids it.
#[test]
fn conv_shape_edge_overhang_does_not_darken() {
    let src = flat(W, H);
    let win = Window {
        px: &src,
        win_w: W,
        win_h: H,
    };
    let px = solid_shape(16, 9, 9);
    let shape = Shape {
        px: &px,
        tex_w: 16,
        tex_h: 16,
    };
    let p = ConvShapeParams::new(6.0, 1.0, 9, 9);

    let out = shape_model(&win, W, H, &shape, &p, 1.0, SHIPPED);
    for y in 0..H {
        for x in 0..W {
            assert_eq!(
                out[(y * W + x) as usize],
                FLAT,
                "({x},{y}) darkened; a constant field must survive the convolution exactly"
            );
        }
    }

    // The WRONG variant, for contrast: the corner loses most of its
    // light and the centre is untouched — the signature "dark vignette
    // that only appears at the frame" bug.
    let bad = shape_model(&win, W, H, &shape, &p, 1.0, ZERO_PADDED);
    assert!(
        bad[0][0] < FLAT[0] * 0.5,
        "the zero-padded draft must visibly darken the corner; got {}",
        bad[0][0]
    );
    assert_eq!(
        bad[((H / 2) * W + W / 2) as usize],
        FLAT,
        "…and must leave the interior alone, which is why the bug hides"
    );
}

/// THE DIVISOR IS THE WEIGHT SUM, NOT THE TAP COUNT. A shape is a
/// COVERAGE field, not a stencil: at 0.5 coverage every tap contributes
/// half a sample, and dividing by the number of taps returns half the
/// brightness. Constant field in, constant field out — exactly, for both
/// a uniform fractional shape and a mixed one.
#[test]
fn conv_shape_normalises_by_the_weights_it_gathered() {
    let src = flat(W, H);
    let win = Window {
        px: &src,
        win_w: W,
        win_h: H,
    };
    let p = ConvShapeParams::new(5.0, 1.0, 8, 8);

    // Uniform half coverage: the exact-arithmetic case, so the
    // darkening factor is a bit-exact 0.5 rather than "about half".
    let px = shape_tex(16, 8, 8, |_, _| 0.5);
    let shape = Shape {
        px: &px,
        tex_w: 16,
        tex_h: 16,
    };
    let good = shape_model(&win, W, H, &shape, &p, 1.0, SHIPPED);
    let bad = shape_model(&win, W, H, &shape, &p, 1.0, BOX_DIVISOR);
    let mid = ((H / 2) * W + W / 2) as usize;
    assert_eq!(good[mid], FLAT, "the weight sum normalises exactly");
    assert_eq!(
        bad[mid],
        [FLAT[0] * 0.5, FLAT[1] * 0.5, FLAT[2] * 0.5, FLAT[3] * 0.5],
        "the tap-count divisor darkens by exactly the mean coverage"
    );

    // Mixed coverage — the realistic case, where no single constant
    // could have been precomputed: half the shape at full weight, half
    // at a quarter.
    let px = shape_tex(16, 8, 8, |x, _| if x < 4 { 1.0 } else { 0.25 });
    let shape = Shape {
        px: &px,
        tex_w: 16,
        tex_h: 16,
    };
    let good = shape_model(&win, W, H, &shape, &p, 1.0, SHIPPED);
    for y in 0..H {
        for x in 0..W {
            assert_eq!(
                good[(y * W + x) as usize],
                FLAT,
                "({x},{y}): a mixed-coverage shape must still be exposure-neutral"
            );
        }
    }
    let bad = shape_model(&win, W, H, &shape, &p, 1.0, BOX_DIVISOR);
    assert!(
        bad[mid][0] < FLAT[0],
        "the tap-count divisor darkens here too"
    );
}

/// The weight sum is RECOMPUTED per radius, which is why it could not
/// have been a constant in the first place: the same shape gathers a
/// different number of taps — and therefore a different total weight —
/// at every radius, because the tap grid is the destination pixel grid.
#[test]
fn conv_shape_weight_sum_is_radius_dependent() {
    let px = shape_tex(16, 8, 8, |_, _| 0.5);
    let shape = Shape {
        px: &px,
        tex_w: 16,
        tex_h: 16,
    };
    let sums: Vec<f32> = [1.0f32, 2.0, 5.0, 9.0]
        .iter()
        .map(|&r| {
            taps(&shape, &ConvShapeParams::new(r, 1.0, 8, 8))
                .iter()
                .map(|t| t.2)
                .sum()
        })
        .collect();
    assert_eq!(sums, vec![4.5, 12.5, 60.5, 180.5], "(2r+1)² taps × 0.5");
    assert!(
        sums.windows(2).all(|w| w[1] > w[0]),
        "no single precomputed constant can serve every radius: {sums:?}"
    );
}

// ─────────────── the shape is SAMPLED, not indexed ─────────────────

/// THE CENTRAL CLAIM: the shape is RESAMPLED to the radius, so an
/// arbitrary radius works with a fixed-resolution shape asset.
///
/// The fixture is a 2×1 shape `[solid, empty]` — "the left half of the
/// box". An INDEXED implementation could only ever gather two taps from
/// it. A sampled one gathers a footprint that scales with the radius,
/// which is exactly what this asserts: an impulse smears to the RIGHT by
/// precisely `r` pixels (a tap at `dx < 0` reads the pixel to the left,
/// so the pixels that see the impulse are the ones to its right), and
/// the pixel ON the impulse stays black because `dx = 0` maps into the
/// shape's empty half.
#[test]
fn conv_shape_is_resampled_to_the_radius() {
    const N: i32 = 41;
    const C: i32 = 20;
    let mut src = vec![[0.0f32; 4]; (N * N) as usize];
    src[(C * N + C) as usize] = [1.0, 1.0, 1.0, 1.0];
    let win = Window {
        px: &src,
        win_w: N,
        win_h: N,
    };
    // 2×1: left texel solid, right texel empty.
    let px = shape_tex(8, 2, 1, |x, _| if x == 0 { 1.0 } else { 0.0 });
    let shape = Shape {
        px: &px,
        tex_w: 8,
        tex_h: 8,
    };

    for r in [4i32, 12] {
        let p = ConvShapeParams::new(r as f32, 1.0, 2, 1);
        let out = shape_model(&win, N, N, &shape, &p, 1.0, SHIPPED);
        let lit: Vec<i32> = (0..N)
            .filter(|&x| out[(C * N + x) as usize][0] > 0.0)
            .collect();
        assert_eq!(
            lit,
            (C + 1..=C + r).collect::<Vec<i32>>(),
            "radius {r}: the smear must run exactly r pixels to the right of the impulse"
        );
        assert_eq!(
            out[(C * N + C) as usize],
            [0.0; 4],
            "radius {r}: dx = 0 maps into the shape's empty half, so the impulse's own \
             pixel stays black — an off-by-one in the tap→texel map would light it"
        );
    }
}

/// NEAREST, NOT BILINEAR — the documented decision, pinned.
///
/// The shape is a hard 1 → 0 step. Under nearest every gathered weight
/// is exactly 1.0; a bilinear pick would necessarily produce
/// intermediate weights along the step (that is the whole difference),
/// softening the silhouette rim that makes a shape blur read as a shape
/// rather than as a Gaussian.
#[test]
fn conv_shape_samples_nearest_so_the_silhouette_stays_hard() {
    let px = shape_tex(16, 8, 8, |x, y| if x + y < 8 { 1.0 } else { 0.0 });
    let shape = Shape {
        px: &px,
        tex_w: 16,
        tex_h: 16,
    };
    for r in [2.0f32, 5.0, 11.0, 24.0] {
        let p = ConvShapeParams::new(r, 1.0, 8, 8);
        let ws = taps(&shape, &p);
        assert!(!ws.is_empty(), "radius {r} must gather something");
        for &(dx, dy, w) in &ws {
            assert_eq!(
                w, 1.0,
                "radius {r} tap ({dx},{dy}) has weight {w}; a hard-edged shape must \
                 produce hard weights — an intermediate value is bilinear bleed"
            );
        }
    }
}

/// The tap→texel map is bounded by the shape, in both directions: no tap
/// may address outside `[0, sw)×[0, sh)`, and the extremes must actually
/// REACH the first and last texel rather than stopping short. The first
/// half is a memory-safety claim about `in1`; the second is what makes
/// the shape fill its box instead of being cropped.
#[test]
fn conv_shape_tap_map_covers_the_shape_and_nothing_else() {
    let px = solid_shape(16, 7, 5);
    let shape = Shape {
        px: &px,
        tex_w: 16,
        tex_h: 16,
    };
    for r in [0.5f32, 1.0, 3.5, 8.0, 24.0] {
        let p = ConvShapeParams::new(r, 1.0, 7, 5);
        let rr = radius(&p).unwrap();
        let (sw, sh) = extent(&shape, &p);
        let (kx, ky) = (sw as f32 / (2.0 * rr), sh as f32 / (2.0 * rr));
        let mut seen_x = vec![false; sw as usize];
        let mut seen_y = vec![false; sh as usize];
        for (dx, dy, _) in taps(&shape, &p) {
            let sx = (((dx as f32 + rr) * kx) as i32).min(sw - 1);
            let sy = (((dy as f32 + rr) * ky) as i32).min(sh - 1);
            assert!((0..sw).contains(&sx), "radius {r}: sx {sx} left the shape");
            assert!((0..sh).contains(&sy), "radius {r}: sy {sy} left the shape");
            seen_x[sx as usize] = true;
            seen_y[sy as usize] = true;
        }
        // At a radius smaller than the shape the map necessarily skips
        // texels (there are fewer taps than texels); the reachability
        // claim is about radii the footprint can actually resolve.
        if rr >= sw as f32 && rr >= sh as f32 {
            assert!(
                seen_x[0] && seen_x[sw as usize - 1],
                "radius {r}: x extremes"
            );
            assert!(
                seen_y[0] && seen_y[sh as usize - 1],
                "radius {r}: y extremes"
            );
        }
    }
}

// ─────────────── shape extent (the padded `in1` upload) ────────────

/// `shape_w`/`shape_h` of 0 means "the whole `in1` texture" — the
/// degrade-rather-than-fail default for a caller that uploaded the shape
/// at its exact size and has nothing extra to say. Same convention as
/// `gen.pattern`'s `tile_w`/`tile_h`, and for the same reason.
#[test]
fn conv_shape_zero_extent_means_the_whole_texture() {
    let src = labeled(W, H);
    let win = Window {
        px: &src,
        win_w: W,
        win_h: H,
    };
    let px = solid_shape(8, 8, 8);
    let shape = Shape {
        px: &px,
        tex_w: 8,
        tex_h: 8,
    };
    let explicit = ConvShapeParams::new(5.0, 1.0, 8, 8);
    let implied = ConvShapeParams::new(5.0, 1.0, 0, 0);
    assert_eq!(extent(&shape, &implied), (8, 8));
    assert_eq!(
        shape_model(&win, W, H, &shape, &explicit, 1.0, SHIPPED),
        shape_model(&win, W, H, &shape, &implied, 1.0, SHIPPED)
    );
}

/// An extent LARGER than the bound texture is clamped to the texture: no
/// tap may leave `in1`, whatever the caller claims about the shape.
#[test]
fn conv_shape_oversized_extent_clamps_to_the_texture() {
    let src = labeled(W, H);
    let win = Window {
        px: &src,
        win_w: W,
        win_h: H,
    };
    let px = solid_shape(8, 8, 8);
    let shape = Shape {
        px: &px,
        tex_w: 8,
        tex_h: 8,
    };
    let sane = ConvShapeParams::new(5.0, 1.0, 8, 8);
    for (sw, sh) in [(9u32, 9u32), (8, 80), (u32::MAX, u32::MAX)] {
        let bogus = ConvShapeParams::new(5.0, 1.0, sw, sh);
        assert_eq!(
            extent(&shape, &bogus),
            (8, 8),
            "extent {sw}x{sh} must clamp"
        );
        assert_eq!(
            shape_model(&win, W, H, &shape, &bogus, 1.0, SHIPPED),
            shape_model(&win, W, H, &shape, &sane, 1.0, SHIPPED)
        );
    }
}

/// A SUB-RECT of the padded texture is the shape: the dispatcher sizes
/// every input at the OUTPUT dims, so a small shape arrives zero-padded
/// at the top-left and its real extent travels in the params. Reading
/// `textureDimensions(in1)` instead would scale the shape down into one
/// corner of a mostly-empty box — modelled here as the wrong extent, so
/// the failure is visible rather than theoretical.
#[test]
fn conv_shape_sub_rect_of_the_texture_is_the_shape() {
    let src = flat(W, H);
    let win = Window {
        px: &src,
        win_w: W,
        win_h: H,
    };
    // A 4×4 shape padded into a 16×16 upload.
    let px = solid_shape(16, 4, 4);
    let shape = Shape {
        px: &px,
        tex_w: 16,
        tex_h: 16,
    };
    let right = ConvShapeParams::new(6.0, 1.0, 4, 4);
    let padded = ConvShapeParams::new(6.0, 1.0, 16, 16);

    // With the real extent every tap in the box lands on solid coverage.
    assert_eq!(taps(&shape, &right).len(), 13 * 13, "(2r+1)² taps");
    // With the padded extent only the top-left quarter of the box does —
    // an off-centre, quarter-strength kernel. Not a subtle difference.
    let bad = taps(&shape, &padded).len();
    assert!(
        bad < 13 * 13 / 3,
        "reading the padded size must visibly cripple the kernel; got {bad} taps"
    );
    assert_eq!(
        shape_model(&win, W, H, &shape, &right, 1.0, SHIPPED)[((H / 2) * W + W / 2) as usize],
        FLAT
    );
}

// ─────────────────── dispatcher / halo / masking ───────────────────

/// THE DERIVED HALO, end to end: the same convolution driven through the
/// TILED dispatcher (`in0` expanded by the radius) and through the
/// ONE-SHOT dispatcher (`in0` at the output dims) must produce identical
/// pixels. A kernel that hardcoded `xy + (24, 24)` would be right under
/// one and shifted by 24 px under the other — and `conv.shape`'s live
/// path today is the one-shot one.
#[test]
fn conv_shape_derived_halo_agrees_across_dispatchers() {
    let src = labeled(W, H);
    let one_shot = Window {
        px: &src,
        win_w: W,
        win_h: H,
    };

    // The tiled window: the same image expanded by the radius with
    // edge replication — which is exactly what the shipped edge policy
    // synthesises for the taps that leave the one-shot window, so the
    // two must agree texel for texel.
    const HALO: i32 = 24;
    let (ew, eh) = (W + 2 * HALO, H + 2 * HALO);
    let expanded: Vec<Px> = (0..ew * eh)
        .map(|i| {
            let (x, y) = (i % ew - HALO, i / ew - HALO);
            src[(y.clamp(0, H - 1) * W + x.clamp(0, W - 1)) as usize]
        })
        .collect();
    let tiled = Window {
        px: &expanded,
        win_w: ew,
        win_h: eh,
    };

    let px = solid_shape(16, 7, 7);
    let shape = Shape {
        px: &px,
        tex_w: 16,
        tex_h: 16,
    };
    for radius_px in [1.0f32, 6.0, 24.0] {
        let p = ConvShapeParams::new(radius_px, 1.0, 7, 7);
        assert_eq!(
            shape_model(&one_shot, W, H, &shape, &p, 1.0, SHIPPED),
            shape_model(&tiled, W, H, &shape, &p, 1.0, SHIPPED),
            "radius {radius_px}: the two dispatchers must agree"
        );
    }
}

/// `amount` is a WET/DRY lerp, not an on/off switch: half amount is the
/// exact midpoint of the input and the fully blurred result. That is what
/// makes the slider continuous instead of a step at zero.
#[test]
fn conv_shape_amount_is_a_wet_dry_lerp() {
    let src = labeled(W, H);
    let win = Window {
        px: &src,
        win_w: W,
        win_h: H,
    };
    let px = solid_shape(16, 7, 7);
    let shape = Shape {
        px: &px,
        tex_w: 16,
        tex_h: 16,
    };
    let wet = shape_model(
        &win,
        W,
        H,
        &shape,
        &ConvShapeParams::new(5.0, 1.0, 7, 7),
        1.0,
        SHIPPED,
    );
    let half = shape_model(
        &win,
        W,
        H,
        &shape,
        &ConvShapeParams::new(5.0, 0.5, 7, 7),
        1.0,
        SHIPPED,
    );
    assert_ne!(wet, src, "the filter must actually do something");
    for i in 0..(W * H) as usize {
        assert_eq!(half[i], mix(src[i], wet[i], 0.5), "texel {i}");
    }
}

/// The mask epilogue is what scopes the blur to a SELECTION: mask 0
/// leaves the input alone, mask 1 takes the blur whole, and a partial
/// mask lands between the two. Without it "blur the selection" would be
/// "blur the layer".
#[test]
fn conv_shape_mask_scopes_the_blur() {
    let src = labeled(W, H);
    let win = Window {
        px: &src,
        win_w: W,
        win_h: H,
    };
    let px = solid_shape(16, 7, 7);
    let shape = Shape {
        px: &px,
        tex_w: 16,
        tex_h: 16,
    };
    let p = ConvShapeParams::new(5.0, 1.0, 7, 7);
    let off = shape_model(&win, W, H, &shape, &p, 0.0, SHIPPED);
    let on = shape_model(&win, W, H, &shape, &p, 1.0, SHIPPED);
    let half = shape_model(&win, W, H, &shape, &p, 0.5, SHIPPED);
    assert_eq!(off, src, "mask 0 is outside the selection: no blur at all");
    assert_ne!(on, src);
    for i in 0..(W * H) as usize {
        assert_eq!(half[i], mix(src[i], on[i], 0.5));
    }
}

/// The shape DIRECTS the blur — a wide, one-row shape smears
/// horizontally and leaves vertical structure alone, which is the whole
/// user-visible promise of "pick a shape". A kernel that ignored `in1`
/// and fell back to a box would fail this.
#[test]
fn conv_shape_the_shape_directs_the_smear() {
    const N: i32 = 41;
    const C: i32 = 20;
    let mut src = vec![[0.0f32; 4]; (N * N) as usize];
    src[(C * N + C) as usize] = [1.0, 1.0, 1.0, 1.0];
    let win = Window {
        px: &src,
        win_w: N,
        win_h: N,
    };
    // 8×1 solid: a horizontal bar.
    let px = solid_shape(16, 8, 1);
    let shape = Shape {
        px: &px,
        tex_w: 16,
        tex_h: 16,
    };
    let p = ConvShapeParams::new(6.0, 1.0, 8, 1);
    let out = shape_model(&win, N, N, &shape, &p, 1.0, SHIPPED);

    // Every tap maps into the single shape ROW, so the footprint is the
    // full box in y as well — the bar says "all of x, all of y" because
    // one row stretched over the box IS the box. The distinguishing
    // fixture is the 1-wide column below.
    assert!(out[(C * N + C + 5) as usize][0] > 0.0, "smears along x");

    // 1×8 solid: a vertical bar — every tap maps into the single COLUMN,
    // so the smear must differ from the horizontal case in exactly the
    // way the shape does.
    let px = solid_shape(16, 1, 8);
    let col = Shape {
        px: &px,
        tex_w: 16,
        tex_h: 16,
    };
    let p = ConvShapeParams::new(6.0, 1.0, 1, 8);
    let out_col = shape_model(&win, N, N, &col, &p, 1.0, SHIPPED);
    assert_eq!(
        out, out_col,
        "a 1×N and an N×1 SOLID shape both fill the box, so they must agree — \
         the fixture that separates them is coverage, not aspect"
    );

    // The real directional fixture: a shape solid in ONE ROW of its
    // extent, empty elsewhere. Measured as a BOUNDING BOX rather than
    // by probing the impulse's own row and column, because an
    // off-centre solid row puts the smear on a neighbouring row — which
    // is correct (a convolution kernel that is off-centre translates)
    // and would make a fixed probe read zero.
    let px = shape_tex(16, 8, 8, |_, y| if y == 3 { 1.0 } else { 0.0 });
    let bar = Shape {
        px: &px,
        tex_w: 16,
        tex_h: 16,
    };
    let p = ConvShapeParams::new(6.0, 1.0, 8, 8);
    let out_bar = shape_model(&win, N, N, &bar, &p, 1.0, SHIPPED);
    let lit = |i: usize| out_bar[i][0] > 0.0;
    let cols = (0..N)
        .filter(|&x| (0..N).any(|y| lit((y * N + x) as usize)))
        .count();
    let rows = (0..N)
        .filter(|&y| (0..N).any(|x| lit((y * N + x) as usize)))
        .count();
    assert_eq!(
        (cols, rows),
        (13, 1),
        "a single-row shape at radius 6 must smear into a 13×1 bar: the shape, \
         resampled to the radius, IS the point-spread function"
    );
}

// ───────────────────── params encoding / layout ────────────────────

/// The uniform block: 20 bytes, `radius_px | amount | shape_w |
/// shape_h | _abi_pad`, and the `ParamsLayout` field list matches the
/// struct field-for-field IN ORDER (the WGSL `struct Params` is
/// handwritten, so this plus the module-text check below is all that
/// holds the two together).
#[test]
fn conv_shape_params_encoding() {
    assert_eq!(std::mem::size_of::<ConvShapeParams>(), 20);
    assert_eq!(
        CONV_SHAPE.params.size,
        std::mem::size_of::<ConvShapeParams>()
    );

    let names: Vec<&str> = CONV_SHAPE.params.fields.iter().map(|f| f.name).collect();
    let types: Vec<&str> = CONV_SHAPE.params.fields.iter().map(|f| f.wgsl_ty).collect();
    assert_eq!(names, ["radius_px", "amount", "shape_w", "shape_h"]);
    assert_eq!(types, ["f32", "f32", "u32", "u32"]);
    // fields + the trailing `_abi_pad`, all 4-byte scalars.
    assert_eq!(CONV_SHAPE.params.size, 4 * (names.len() + 1));

    let p = ConvShapeParams::new(12.5, 0.75, 64, 32);
    let b = p.as_bytes();
    assert_eq!(b.len(), 20);
    assert_eq!(&b[0..4], &12.5f32.to_le_bytes());
    assert_eq!(&b[4..8], &0.75f32.to_le_bytes());
    assert_eq!(&b[8..12], &64u32.to_le_bytes());
    assert_eq!(&b[12..16], &32u32.to_le_bytes());
    assert_eq!(&b[16..20], &0u32.to_le_bytes(), "_abi_pad is always 0");

    // Byte identity IS param identity (the op-cache key, spec §6.2), so
    // every knob must participate — a radius change that hashed equal
    // would serve a stale tile from the cache.
    let base = ConvShapeParams::new(6.0, 1.0, 8, 8);
    for other in [
        ConvShapeParams::new(6.5, 1.0, 8, 8),
        ConvShapeParams::new(6.0, 0.5, 8, 8),
        ConvShapeParams::new(6.0, 1.0, 4, 8),
        ConvShapeParams::new(6.0, 1.0, 8, 4),
    ] {
        assert_ne!(base.as_bytes(), other.as_bytes());
    }
}

// ──────────────────── registration + ABI contract ──────────────────

/// Registered in `conv::FAMILY` — and therefore in `ALL_FAMILIES`, which
/// is what `all_defined()` and the registry-drift gate read.
#[test]
fn conv_shape_is_registered_in_the_family() {
    assert!(
        conv::FAMILY.iter().any(|d| d.id == "conv.shape"),
        "conv.shape must be in conv::FAMILY"
    );
    assert!(
        ALL_FAMILIES
            .iter()
            .flat_map(|f| f.iter())
            .any(|d| d.id == "conv.shape"),
        "conv.shape must be reachable from ALL_FAMILIES"
    );
    // Exactly once — a duplicated entry would double every count the
    // registry quotes.
    assert_eq!(
        ALL_FAMILIES
            .iter()
            .flat_map(|f| f.iter())
            .filter(|d| d.id == "conv.shape")
            .count(),
        1
    );
}

/// The metadata the engines dispatch on. Two of these are load-bearing
/// and unusual together: `inputs: 2` (without a second input there is
/// nowhere for the shape to live and the kernel could only convolve with
/// a formula) and `Windowed` (the `in0` ROI must inflate by the largest
/// radius the slider can ask for, or a wide blur reads outside its tile).
#[test]
fn conv_shape_kernel_metadata() {
    assert_eq!(CONV_SHAPE.id, "conv.shape");
    assert_eq!(
        CONV_SHAPE.inputs, 2,
        "in0 = the image, in1 = the shape coverage bitmap"
    );
    assert!(CONV_SHAPE.module, "handwritten ABI v1.1 module");
    assert!(
        !CONV_SHAPE.mip_exact,
        "the shape is scaled in pixels, so a mip level must re-derive radius_px"
    );
    assert_eq!(
        CONV_SHAPE.class,
        KernelClass::Windowed { radius: (24, 24) },
        "the same ceiling conv.lens pays: a 49×49 worst-case gather"
    );
    assert_eq!(CONV_SHAPE.gpu_tolerance, Tolerance::ChannelEpsF16(4));

    // The declared radius and the module's own loop bound must be the
    // same number — a `R_MAX` below the declared radius silently crops
    // the blur, above it reads outside the guaranteed window.
    let KernelClass::Windowed { radius: (rx, ry) } = CONV_SHAPE.class else {
        panic!("conv.shape must be Windowed");
    };
    assert_eq!((rx, ry), (24, 24));
    assert!(
        image_kernels::abi::assemble(&CONV_SHAPE).contains(&format!("const R_MAX : i32 = {rx};")),
        "the module's R_MAX must equal the declared window radius"
    );
    assert_eq!(R_MAX, i32::from(rx), "…and so must this file's mirror");
}

/// The ABI v1.1 binding contract, checked against the module TEXT: the
/// four groups INCLUDING the second input at `@group(0) @binding(1)`,
/// the workgroup size, the bounds prologue, and the mandatory mask
/// epilogue. `abi::assemble` returns a `module: true` kernel's WGSL
/// verbatim, so this is the shipped source.
#[test]
fn conv_shape_wgsl_honours_the_abi() {
    let src = image_kernels::abi::assemble(&CONV_SHAPE);
    for needle in [
        "@group(0) @binding(0) var in0 : texture_2d<f32>;",
        "@group(0) @binding(1) var in1 : texture_2d<f32>;",
        "@group(1) @binding(0) var<uniform> params : Params;",
        "@group(2) @binding(0) var mask : texture_2d<f32>;",
        "@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;",
        "@compute @workgroup_size(16, 16, 1)",
        "if (gid.x >= d.x || gid.y >= d.y) { return; }",
        "let m = textureLoad(mask, xy, 0).r;",
        "textureStore(outp, xy, mix(a, result, vec4<f32>(m)));",
    ] {
        assert!(src.contains(needle), "missing from the module: {needle}");
    }
    assert!(
        !src.contains("textureSample"),
        "inputs are textureLoad-only under the ABI (no samplers); the shape pick is \
         computed, not sampled by hardware"
    );
    // The image is read from in0 and the SHAPE from in1 — swapping them
    // would compile and silently convolve the shape by the image.
    assert!(src.contains("let a = textureLoad(in0, base, 0);"));
    assert!(src.contains("let w = textureLoad(in1, vec2<i32>(sx, sy), 0).r;"));
    // …and the coverage channel is RED, stated in the text so a channel
    // change cannot happen without this gate noticing.
    assert!(
        src.contains("textureLoad(in1, vec2<i32>(sx, sy), 0).r"),
        "coverage is the RED channel"
    );
}

/// THE DERIVED-HALO IDIOM must be present verbatim, and the hardcoded
/// form must NOT be. `abi.rs` documents that one dispatcher expands
/// `in0` by the radius and the other does not; deriving the offset from
/// the two texture sizes is the only form that is correct under both.
#[test]
fn conv_shape_wgsl_derives_the_halo() {
    let src = image_kernels::abi::assemble(&CONV_SHAPE);
    for needle in [
        "let wd = textureDimensions(in0);",
        "let win = vec2<i32>(i32(wd.x), i32(wd.y));",
        "let halo = (win - dims) / 2;",
        "let base = xy + halo;",
    ] {
        assert!(
            src.contains(needle),
            "the derived halo is missing: {needle}"
        );
    }
    assert!(
        !src.contains("xy + vec2<i32>(24, 24)"),
        "the halo must be DERIVED, never hardcoded to the radius"
    );
    // Every `in0` read goes through `base` (plus a clamped tap offset),
    // never through the raw output coordinate — a stray `textureLoad(in0,
    // xy, ...)` would be the shifted-by-the-halo bug wearing a disguise.
    assert!(
        !src.contains("textureLoad(in0, xy,"),
        "in0 must be addressed from the derived base, not from xy"
    );
}

/// The handwritten WGSL `struct Params` matches `ParamsLayout` field for
/// field in order, plus the trailing `_abi_pad`. A drift here is a
/// silently misdecoded uniform block, which is why it gets its own gate.
#[test]
fn conv_shape_wgsl_struct_matches_the_layout() {
    let src = image_kernels::abi::assemble(&CONV_SHAPE);
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

    let mut expected: Vec<String> = CONV_SHAPE
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
fn conv_shape_wgsl_validates() {
    let src = image_kernels::abi::assemble(&CONV_SHAPE);
    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("conv.shape: WGSL parse failed: {e}\n{src}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("conv.shape: WGSL validation failed: {e:?}\n{src}"));
}
