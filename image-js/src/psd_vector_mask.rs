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

//! PSD VECTOR MASKS drawn the way Photoshop draws them, into the layer
//! stack's one mask.
//!
//! `image_psd::VectorMask` holds the paths; this draws them. The
//! drawing is mask PREPARATION — rasterizing coverage geometry — which
//! the GPU-only constitution leaves on the CPU (`image_gpu::coverage`'s
//! honesty note): no image pixel is touched here, and the mask is
//! consumed by the GPU fold like any other layer mask.
//!
//! # The sampling, as measured ([OBS], Photoshop 27.10)
//!
//! Photoshop caches its render of every enabled vector mask on a pixel
//! layer in that layer's mask channel, so the rasterizer was fitted
//! against those renders — edges at every 1/16 pixel, diagonals, a
//! self-intersecting star, curves — and then checked against the
//! renders of the private
//! corpus (`image-conformance/tests/psd_vector_mask_corpus.rs`,
//! aggregates only):
//!
//! * a pixel is a grid of **17 × 15 point samples** (17 across, 15
//!   down), so coverage is a sample COUNT, 0–255, with no rounding.
//!   Sample `i` across sits at `(i + 31/64) / 17`, sub-scanline `k` down
//!   at `(k + 31/64) / 15`. The 1/16-pixel edge probes pin both offsets
//!   to (7/16, 1/2] — a left or top edge at `.5` drops the middle sample,
//!   a right or bottom edge at `.5` keeps it — and 31/64 is the value
//!   inside that interval that disagrees least with the probes'
//!   diagonals and the corpus's cached renders;
//! * Beziers are flattened, and Photoshop's flattening is not exact
//!   either: its circles come out about 0.02 px small (chords inside the
//!   curve). [`FLATTEN_TOLERANCE`] reproduces that bias on average, not
//!   chord for chord, so a curved edge can still differ by up to about
//!   one sub-scanline (17 levels) on isolated pixels;
//! * a sample is inside when its sub-scanline crosses the component's
//!   edges an odd number of times (even-odd) or with a non-zero winding
//!   sum (non-zero) — the component's fill rule;
//! * components combine PER SAMPLE, not per pixel: two triangles sharing
//!   an anti-aliased diagonal have no seam under "combine", and an
//!   intersect of two edges a quarter pixel apart keeps exactly the
//!   samples between them. So the boolean runs on sample intervals and
//!   the count happens last.
//!
//! The user mask beside a vector mask multiplies with it: the render
//! Photoshop caches for the pair is their product.

use image_gpu::coverage::SelectionCoverage;
use image_psd::layer_pixels::{LayerPlate, MaskPlate};
use image_psd::vector_mask::{FillRule, PathOp, PathPoint, VectorMask};

/// Point samples per pixel, across.
pub const SAMPLES_X: i64 = 17;
/// Point samples per pixel, down (sub-scanlines).
pub const SAMPLES_Y: u32 = 15;
/// Where sample `i` sits inside its 1/17 column, in sample units.
const OFFSET_X: f64 = 0.484375;
/// Where sub-scanline `k` sits inside its 1/15 row, in sample units.
const OFFSET_Y: f64 = 0.484375;
/// Bezier flattening tolerance: how far a control point may lie from its
/// chord, in pixels. Not a free choice — Photoshop flattens too, and its
/// chords sit measurably INSIDE a convex curve (its circles come out
/// ~0.02 px smaller than the true curve). 0.05 is the value that
/// balances that bias on probed circles of radius 30, 120 and 500 and
/// minimises the disagreement with the corpus's cached renders; a finer
/// flattening is further from Photoshop, not closer.
const FLATTEN_TOLERANCE: f64 = 0.05;
/// Ceiling on the subdivision depth (2^16 segments per Bezier).
const MAX_DEPTH: u32 = 16;

#[derive(Debug, Clone, Copy)]
struct Edge {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    /// Winding contribution: +1 going down, −1 going up.
    dir: i32,
}

impl Edge {
    fn ymin(&self) -> f64 {
        self.y0.min(self.y1)
    }

    fn ymax(&self) -> f64 {
        self.y0.max(self.y1)
    }
}

/// One component, flattened: its edges sorted by top, its rule and op.
struct Flat {
    op: PathOp,
    rule: FillRule,
    edges: Vec<Edge>,
}

fn push_line(edges: &mut Vec<Edge>, a: PathPoint, b: PathPoint) {
    if a.y == b.y {
        return; // horizontal: never crosses a sub-scanline
    }
    edges.push(Edge {
        x0: a.x,
        y0: a.y,
        x1: b.x,
        y1: b.y,
        dir: if b.y > a.y { 1 } else { -1 },
    });
}

/// Flatten one cubic Bezier into line edges by midpoint subdivision,
/// until both control points lie within [`FLATTEN_TOLERANCE`] of the
/// chord. Straight segments (control points on their anchors, which is
/// what corner knots store) stay one edge, so axis-aligned edges keep
/// their exact coordinates.
fn push_cubic(edges: &mut Vec<Edge>, p0: PathPoint, p1: PathPoint, p2: PathPoint, p3: PathPoint) {
    if p1 == p0 && p2 == p3 {
        push_line(edges, p0, p3);
        return;
    }
    let mid = |a: PathPoint, b: PathPoint| PathPoint {
        x: (a.x + b.x) / 2.0,
        y: (a.y + b.y) / 2.0,
    };
    let flat = |q: &[PathPoint; 4]| {
        let (dx, dy) = (q[3].x - q[0].x, q[3].y - q[0].y);
        let len = dx.hypot(dy);
        let off = |p: PathPoint| {
            if len < 1e-12 {
                (p.x - q[0].x).hypot(p.y - q[0].y)
            } else {
                ((p.x - q[0].x) * dy - (p.y - q[0].y) * dx).abs() / len
            }
        };
        off(q[1]).max(off(q[2])) <= FLATTEN_TOLERANCE
    };
    // Depth-first, left half first, so the edges come out in path order.
    let mut stack = vec![([p0, p1, p2, p3], 0u32)];
    while let Some((q, depth)) = stack.pop() {
        if depth >= MAX_DEPTH || flat(&q) {
            push_line(edges, q[0], q[3]);
            continue;
        }
        let (a, b, c) = (mid(q[0], q[1]), mid(q[1], q[2]), mid(q[2], q[3]));
        let (d, e) = (mid(a, b), mid(b, c));
        let m = mid(d, e);
        stack.push(([m, e, c, q[3]], depth + 1));
        stack.push(([q[0], a, d, m], depth + 1));
    }
}

fn flatten(mask: &VectorMask) -> Vec<Flat> {
    mask.components
        .iter()
        .map(|c| {
            let mut edges = Vec::new();
            for sub in &c.subpaths {
                let k = &sub.knots;
                if k.len() < 2 {
                    continue; // a lone anchor encloses nothing
                }
                for i in 0..k.len() - 1 {
                    push_cubic(
                        &mut edges,
                        k[i].anchor,
                        k[i].control_out,
                        k[i + 1].control_in,
                        k[i + 1].anchor,
                    );
                }
                let (last, first) = (&k[k.len() - 1], &k[0]);
                if sub.closed {
                    push_cubic(
                        &mut edges,
                        last.anchor,
                        last.control_out,
                        first.control_in,
                        first.anchor,
                    );
                } else {
                    // [OBS] an open subpath fills as if a STRAIGHT line
                    // closed it, whatever its end handles say.
                    push_line(&mut edges, last.anchor, first.anchor);
                }
            }
            edges.sort_by(|a, b| a.ymin().total_cmp(&b.ymin()));
            Flat {
                op: c.op,
                rule: c.fill_rule,
                edges,
            }
        })
        .collect()
}

/// Sorted, disjoint half-open sample-index intervals on one sub-scanline.
type Spans = Vec<(i64, i64)>;

/// Fold `b` into `a` under `keep(in_a, in_b)`, by sweeping both sets'
/// boundaries.
fn boolean(a: &Spans, b: &Spans, keep: impl Fn(bool, bool) -> bool) -> Spans {
    let mut cuts: Vec<i64> = Vec::with_capacity(2 * (a.len() + b.len()));
    for &(s, e) in a.iter().chain(b.iter()) {
        cuts.push(s);
        cuts.push(e);
    }
    cuts.sort_unstable();
    cuts.dedup();
    let inside = |set: &Spans, x: i64| set.iter().any(|&(s, e)| s <= x && x < e);
    let mut out: Spans = Vec::new();
    for w in cuts.windows(2) {
        let (s, e) = (w[0], w[1]);
        if keep(inside(a, s), inside(b, s)) {
            match out.last_mut() {
                Some(last) if last.1 == s => last.1 = e,
                _ => out.push((s, e)),
            }
        }
    }
    out
}

/// The sample intervals one component covers on the sub-scanline at
/// `sy`, given its edges active there.
fn component_spans(
    active: &[Edge],
    sy: f64,
    rule: FillRule,
    limit: i64,
    xs: &mut Vec<(f64, i32)>,
) -> Spans {
    xs.clear();
    for e in active {
        // Half-open in y, so a vertex exactly on the sub-scanline counts
        // once.
        if (e.y0 < sy && sy <= e.y1) || (e.y1 < sy && sy <= e.y0) {
            let t = (sy - e.y0) / (e.y1 - e.y0);
            xs.push((e.x0 + t * (e.x1 - e.x0), e.dir));
        }
    }
    xs.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Sample `s` sits at x = (s + OFFSET_X) / 17 and is inside a span
    // (xa, xb] — the side the probes put a sample lying on an edge.
    let first_after =
        |x: f64| -> i64 { ((x * SAMPLES_X as f64 - OFFSET_X).floor() as i64 + 1).clamp(0, limit) };
    let mut spans: Spans = Vec::new();
    let mut winding = 0i32;
    let mut start = 0.0f64;
    for &(x, dir) in xs.iter() {
        let was = match rule {
            FillRule::NonZero => winding != 0,
            FillRule::EvenOdd => winding % 2 != 0,
        };
        winding += dir;
        let now = match rule {
            FillRule::NonZero => winding != 0,
            FillRule::EvenOdd => winding % 2 != 0,
        };
        if !was && now {
            start = x;
        } else if was && !now {
            let (s, e) = (first_after(start), first_after(x));
            if s < e {
                match spans.last_mut() {
                    Some(last) if last.1 >= s => last.1 = last.1.max(e),
                    _ => spans.push((s, e)),
                }
            }
        }
    }
    spans
}

/// Draw `mask` at `width`×`height` the way Photoshop does: every
/// component, folded in order, inverted when the mask says so. The
/// `disabled` flag is NOT consulted — whether to apply the result is the
/// caller's decision.
pub fn rasterize(mask: &VectorMask, width: u32, height: u32) -> SelectionCoverage {
    let (w, h) = (width as usize, height as usize);
    let limit = SAMPLES_X * width as i64;
    let mut flats = flatten(mask);
    let starts_full = flats.first().is_some_and(|f| f.op == PathOp::Subtract);
    let mut data = vec![0u8; w * h];
    let mut counts = vec![0u32; w + 1];
    let mut full_runs = vec![0i32; w + 1];
    let mut next: Vec<usize> = vec![0; flats.len()];
    let mut active: Vec<Vec<Edge>> = vec![Vec::new(); flats.len()];
    let mut xs = Vec::new();
    for y in 0..h {
        counts.iter_mut().for_each(|c| *c = 0);
        full_runs.iter_mut().for_each(|c| *c = 0);
        for k in 0..SAMPLES_Y {
            let sy = y as f64 + (f64::from(k) + OFFSET_Y) / f64::from(SAMPLES_Y);
            let mut acc: Spans = if starts_full {
                vec![(0, limit)]
            } else {
                Vec::new()
            };
            for (ci, flat) in flats.iter_mut().enumerate() {
                while next[ci] < flat.edges.len() && flat.edges[next[ci]].ymin() < sy {
                    active[ci].push(flat.edges[next[ci]]);
                    next[ci] += 1;
                }
                active[ci].retain(|e| e.ymax() >= sy);
                let spans = component_spans(&active[ci], sy, flat.rule, limit, &mut xs);
                // The first component's operation only decides the start
                // state (module docs): it is ADDED, or subtracted from a
                // full mask.
                let op = if ci == 0 && flat.op != PathOp::Subtract {
                    PathOp::Combine
                } else {
                    flat.op
                };
                acc = match op {
                    PathOp::Combine => boolean(&acc, &spans, |a, b| a || b),
                    PathOp::Subtract => boolean(&acc, &spans, |a, b| a && !b),
                    PathOp::Intersect => boolean(&acc, &spans, |a, b| a && b),
                    PathOp::Exclude => boolean(&acc, &spans, |a, b| a != b),
                };
            }
            // Count: partial pixels at each interval's ends, whole pixels
            // through a difference array.
            for &(s, e) in &acc {
                let (p0, p1) = ((s / SAMPLES_X) as usize, ((e - 1) / SAMPLES_X) as usize);
                if p0 == p1 {
                    counts[p0] += (e - s) as u32;
                } else {
                    counts[p0] += (SAMPLES_X * (p0 as i64 + 1) - s) as u32;
                    counts[p1] += (e - SAMPLES_X * p1 as i64) as u32;
                    full_runs[p0 + 1] += 1;
                    full_runs[p1] -= 1;
                }
            }
        }
        let row = &mut data[y * w..(y + 1) * w];
        let mut run = 0i32;
        for (x, v) in row.iter_mut().enumerate() {
            run += full_runs[x];
            let c = counts[x] + run as u32 * SAMPLES_X as u32;
            *v = c.min(255) as u8;
        }
    }
    if mask.invert {
        data.iter_mut().for_each(|v| *v = 255 - *v);
    }
    SelectionCoverage::from_data(width, height, data).expect("canvas-extent coverage")
}

/// The ONE mask a plate brings to the layer stack, as Photoshop applies
/// it: the user mask and the vector mask multiplied when both are
/// enabled, whichever is enabled alone, and — when neither is — the one
/// that exists, kept but not applied (the user mask when both exist: the
/// stack holds one mask per layer). `None` when the plate has no mask.
pub fn plate_mask(plate: &LayerPlate, width: u32, height: u32) -> Option<MaskPlate> {
    let vector = plate.vector_mask.as_ref().map(|v| MaskPlate {
        coverage: rasterize(v, width, height).data().to_vec(),
        enabled: !v.disabled,
    });
    match (plate.mask.clone(), vector) {
        (None, None) => None,
        (Some(m), None) | (None, Some(m)) => Some(m),
        (Some(u), Some(v)) => Some(match (u.enabled, v.enabled) {
            (true, true) => MaskPlate {
                coverage: u
                    .coverage
                    .iter()
                    .zip(&v.coverage)
                    .map(|(&a, &b)| ((u32::from(a) * u32::from(b) + 127) / 255) as u8)
                    .collect(),
                enabled: true,
            },
            (false, true) => v,
            _ => u,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image_psd::vector_mask::{Knot, PathComponent, SubPath};

    fn corner(x: f64, y: f64) -> Knot {
        let p = PathPoint { x, y };
        Knot {
            linked: false,
            control_in: p,
            anchor: p,
            control_out: p,
        }
    }

    fn poly(points: &[(f64, f64)]) -> SubPath {
        SubPath {
            closed: true,
            knots: points.iter().map(|&(x, y)| corner(x, y)).collect(),
        }
    }

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> SubPath {
        poly(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    }

    fn comp(op: PathOp, rule: FillRule, subpaths: Vec<SubPath>) -> PathComponent {
        PathComponent {
            op,
            fill_rule: rule,
            subpaths,
        }
    }

    fn mask(components: Vec<PathComponent>) -> VectorMask {
        VectorMask {
            version: 3,
            invert: false,
            not_linked: false,
            disabled: false,
            initial_fill: false,
            components,
        }
    }

    fn at(c: &SelectionCoverage, x: u32, y: u32) -> u8 {
        c.coverage_at(x, y)
    }

    /// The 1/16-pixel edge ramp Photoshop drew (`vector-masks.jsx`
    /// `aa-ramp`): left edges lose 15 per 1/16 with the middle step 30,
    /// top edges lose 17 per 1/15 sub-scanline.
    #[test]
    #[allow(non_snake_case)]
    fn edges_count_samples_on_photoshops_17_by_15_grid__feat__image_psd_layer_import() {
        let left = [
            255u8, 240, 225, 210, 195, 180, 165, 150, 120, 105, 90, 75, 60, 45, 30, 15,
        ];
        let top = [
            255u8, 238, 221, 204, 187, 170, 153, 136, 119, 119, 102, 85, 68, 51, 34, 17,
        ];
        for j in 0..16 {
            let f = f64::from(j) / 16.0;
            let m = mask(vec![comp(
                PathOp::Combine,
                FillRule::EvenOdd,
                vec![rect(1.0 + f, 2.0 + f, 3.0, 6.0)],
            )]);
            let c = rasterize(&m, 4, 8);
            assert_eq!(at(&c, 1, 4), left[j as usize], "left edge at +{j}/16");
            assert_eq!(at(&c, 2, 2), top[j as usize], "top edge at +{j}/16");
            assert_eq!(at(&c, 2, 4), 255, "interior");
            assert_eq!(at(&c, 0, 4), 0, "exterior");
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn fill_rule_decides_a_self_intersecting_star__feat__image_psd_layer_import() {
        let star = [
            (32.0, 4.0),
            (49.0, 58.0),
            (4.0, 24.0),
            (60.0, 24.0),
            (15.0, 58.0),
        ];
        let even = rasterize(
            &mask(vec![comp(
                PathOp::Combine,
                FillRule::EvenOdd,
                vec![poly(&star)],
            )]),
            64,
            64,
        );
        let nonzero = rasterize(
            &mask(vec![comp(
                PathOp::Combine,
                FillRule::NonZero,
                vec![poly(&star)],
            )]),
            64,
            64,
        );
        assert_eq!(at(&even, 32, 34), 0, "even-odd: the centre is a hole");
        assert_eq!(at(&nonzero, 32, 34), 255, "non-zero: filled");
        assert_eq!(at(&even, 32, 12), 255);
    }

    #[test]
    #[allow(non_snake_case)]
    fn components_fold_per_sample_in_order__feat__image_psd_layer_import() {
        let w = 64;
        let r = |op, x0, x1| comp(op, FillRule::EvenOdd, vec![rect(x0, 10.0, x1, 50.0)]);
        // Two shapes sharing an anti-aliased diagonal: no seam.
        let tri = |pts: &[(f64, f64)]| comp(PathOp::Combine, FillRule::EvenOdd, vec![poly(pts)]);
        let seam = rasterize(
            &mask(vec![
                tri(&[(0.0, 0.0), (64.0, 0.0), (0.0, 64.0)]),
                tri(&[(64.0, 0.0), (64.0, 64.0), (0.0, 64.0)]),
            ]),
            w,
            64,
        );
        assert!(seam.is_all_one(), "per-sample union leaves no seam");
        // Photoshop's measured values in column 10 (`b-*` probe cases).
        let one = |m: VectorMask| at(&rasterize(&m, w, 64), 10, 30);
        assert_eq!(
            one(mask(vec![
                r(PathOp::Combine, 10.25, 40.0),
                r(PathOp::Combine, 2.0, 10.75)
            ])),
            255
        );
        assert_eq!(
            one(mask(vec![
                r(PathOp::Combine, 10.25, 40.0),
                r(PathOp::Intersect, 2.0, 10.75)
            ])),
            135
        );
        assert_eq!(
            one(mask(vec![
                r(PathOp::Combine, 10.5, 40.0),
                r(PathOp::Subtract, 10.25, 50.0)
            ])),
            0
        );
        assert_eq!(
            one(mask(vec![
                r(PathOp::Combine, 10.25, 40.0),
                r(PathOp::Exclude, 10.75, 50.0)
            ])),
            135
        );
        // A first component that subtracts starts from a full mask; one
        // that intersects or excludes is simply added.
        let alone = |op| rasterize(&mask(vec![r(op, 16.0, 48.0)]), w, 64);
        let sub = alone(PathOp::Subtract);
        assert_eq!((at(&sub, 2, 2), at(&sub, 30, 30)), (255, 0));
        for op in [PathOp::Intersect, PathOp::Exclude, PathOp::Combine] {
            let c = alone(op);
            assert_eq!((at(&c, 2, 2), at(&c, 30, 30)), (0, 255), "{op:?}");
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn invert_open_subpaths_and_the_user_mask_product__feat__image_psd_layer_import() {
        let mut m = mask(vec![comp(
            PathOp::Combine,
            FillRule::EvenOdd,
            vec![rect(2.0, 2.0, 6.0, 6.0)],
        )]);
        m.invert = true;
        let c = rasterize(&m, 8, 8);
        assert_eq!((at(&c, 0, 0), at(&c, 3, 3)), (255, 0));
        // An open subpath closes with a straight line, ignoring handles.
        let mut open = rect(2.0, 2.0, 6.0, 6.0);
        open.closed = false;
        open.knots[3].control_out = PathPoint { x: -20.0, y: 4.0 };
        open.knots[0].control_in = PathPoint { x: -20.0, y: 4.0 };
        let c = rasterize(
            &mask(vec![comp(PathOp::Combine, FillRule::EvenOdd, vec![open])]),
            8,
            8,
        );
        assert_eq!((at(&c, 1, 4), at(&c, 3, 4)), (0, 255));

        let plate = |user: Option<MaskPlate>, vector_disabled: bool| {
            let mut v = mask(vec![comp(
                PathOp::Combine,
                FillRule::EvenOdd,
                vec![rect(0.0, 0.0, 2.0, 1.0)],
            )]);
            v.disabled = vector_disabled;
            LayerPlate {
                name: String::new(),
                blend_key: *b"norm",
                opacity: 255,
                hidden: false,
                rgba: vec![255; 16],
                clipped: false,
                group: None,
                mask: user,
                vector_mask: Some(v),
                smart: false,
                adjustment: None,
                color_overlay: None,
            }
        };
        let user = |enabled| MaskPlate {
            coverage: vec![128, 255, 255, 255],
            enabled,
        };
        let both = plate_mask(&plate(Some(user(true)), false), 2, 2).expect("mask");
        assert_eq!(
            (both.coverage.clone(), both.enabled),
            (vec![128, 255, 0, 0], true)
        );
        let vec_only = plate_mask(&plate(Some(user(false)), false), 2, 2).expect("mask");
        assert_eq!(vec_only.coverage, vec![255, 255, 0, 0]);
        let user_only = plate_mask(&plate(Some(user(true)), true), 2, 2).expect("mask");
        assert_eq!(user_only.coverage, vec![128, 255, 255, 255]);
        let neither = plate_mask(&plate(None, true), 2, 2).expect("kept");
        assert!(
            !neither.enabled,
            "a disabled vector mask is kept, not applied"
        );
    }
}
