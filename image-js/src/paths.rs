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

//! SELECTION → PATH: trace the selection's coverage into closed
//! polygons, so a raster selection can become a real vector element.
//!
//! This is the raster half of the raster↔vector pair the Photoshop
//! catalog prices under "Paths / shapes". The vector half — pen, shapes,
//! pathfinder — is HOST-owned and already strong; what was missing was
//! any bridge between the two. `path → selection` needs no engine work
//! at all (the host answers `pathAnchors`, and `selection_set_polygon`
//! already takes a polygon), so this module is the direction that does.
//!
//! **The algorithm is a boundary walk, not marching squares.** Coverage
//! is a per-pixel `u8`, and the boundary that matters is the one between
//! selected and unselected PIXELS — so the contour runs along pixel
//! EDGES (integer coordinates) rather than through pixel centres. That
//! makes the output exact for the rectangular and axis-aligned regions
//! the marquee produces: a 3×2 rect traces to a 4-point rectangle, not
//! to a wobbling 10-point approximation of one. Diagonal and feathered
//! edges become staircases, which the simplifier then straightens under
//! an explicit tolerance the caller chooses.
//!
//! Two honesty rules:
//!
//! * **A threshold is a decision, so it is a parameter.** Coverage is
//!   partial by design (feather, luminosity masks). Tracing has to pick
//!   a cut, and hiding that choice would silently discard the anti-
//!   aliased boundary the selection tools worked to produce.
//! * **Holes are reported as separate contours, in their own winding.**
//!   A donut selection is two rings; collapsing them into one outline
//!   would silently fill the hole.

/// A traced contour: a closed polygon in IMAGE pixel coordinates, on
/// pixel edges (so `(0,0)` is the image's top-left corner, not the
/// centre of its first pixel).
#[derive(Debug, Clone, PartialEq)]
pub struct Contour {
    pub points: Vec<(f32, f32)>,
    /// `true` when the contour bounds a SELECTED region, `false` when it
    /// bounds a hole inside one. The panel needs this to build an
    /// even-odd path rather than two overlapping filled shapes.
    pub outer: bool,
}

/// Trace every boundary of `coverage` (one byte per pixel, row-major,
/// `width * height` long) at `threshold` (0–255; a pixel is IN when its
/// coverage is `>= threshold`).
///
/// `tolerance` (in pixels) collapses collinear and near-collinear runs;
/// `0.0` keeps every step of the staircase.
pub fn trace(
    coverage: &[u8],
    width: usize,
    height: usize,
    threshold: u8,
    tolerance: f32,
) -> Vec<Contour> {
    if width == 0 || height == 0 || coverage.len() < width * height {
        return Vec::new();
    }
    let inside = |x: isize, y: isize| -> bool {
        if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
            return false;
        }
        coverage[y as usize * width + x as usize] >= threshold
    };

    // ONE pass over every boundary edge. The walk rule is always the
    // same — keep the inside on the RIGHT of travel — because a hole's
    // boundary is a boundary like any other; what distinguishes it is
    // its WINDING, not where the walk started. Classifying by the start
    // condition instead was the first attempt and it reported a donut as
    // two outer rings: the pixel below a hole satisfies exactly the same
    // "selected, with an unselected pixel above" test as the top of the
    // region itself.
    let mut seen: Vec<bool> = vec![false; (width + 1) * (height + 1) * 4];
    let edge_id = |x: isize, y: isize, dir: usize| -> usize {
        (y as usize * (width + 1) + x as usize) * 4 + dir
    };

    let mut out: Vec<Contour> = Vec::new();
    for y in 0..height as isize {
        for x in 0..width as isize {
            if !edge_is_boundary(x, y, 0, &inside) || seen[edge_id(x, y, 0)] {
                continue;
            }
            if let Some(mut c) = walk(x, y, &inside, &mut seen, edge_id) {
                // Shoelace: with y running DOWN, a contour that keeps the
                // inside on its right closes positive for a filled region
                // and negative for the hole punched out of one.
                c.outer = signed_area(&c.points) > 0.0;
                out.push(simplify_contour(c, tolerance));
            }
        }
    }
    out
}

/// Twice the signed area of a closed polygon (shoelace). The SIGN is
/// what this is for — the magnitude is never used.
fn signed_area(points: &[(f32, f32)]) -> f32 {
    let n = points.len();
    let mut acc = 0.0f32;
    for i in 0..n {
        let (x0, y0) = points[i];
        let (x1, y1) = points[(i + 1) % n];
        acc += x0 * y1 - x1 * y0;
    }
    acc
}

/// One boundary walk from `(sx, sy)`, keeping the inside on the RIGHT of
/// travel. `Contour::outer` is decided afterwards, from the winding.
fn walk<F, E>(sx: isize, sy: isize, inside: &F, seen: &mut [bool], edge_id: E) -> Option<Contour>
where
    F: Fn(isize, isize) -> bool,
    E: Fn(isize, isize, usize) -> usize,
{
    // Directions: 0 = +x (right), 1 = +y (down), 2 = -x (left), 3 = -y (up).
    const STEP: [(isize, isize); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];
    let mut points: Vec<(f32, f32)> = Vec::new();
    let (mut cx, mut cy) = (sx, sy);
    let mut dir = 0usize;
    for _ in 0..(1 << 22) {
        let id = edge_id(cx, cy, dir);
        if id < seen.len() {
            seen[id] = true;
        }
        points.push((cx as f32, cy as f32));
        let (dx, dy) = STEP[dir];
        cx += dx;
        cy += dy;
        if cx == sx && cy == sy && !points.is_empty() {
            break;
        }
        // At the new corner, choose the next direction by the 2×2
        // neighbourhood, preferring the turn that keeps the inside on
        // the walking side. Turning "inward" first is what makes the
        // walk hug concavities instead of cutting across them.
        let turns: [usize; 4] = [(dir + 3) % 4, dir, (dir + 1) % 4, (dir + 2) % 4];
        let mut moved = false;
        for &t in &turns {
            if edge_is_boundary(cx, cy, t, inside) {
                dir = t;
                moved = true;
                break;
            }
        }
        if !moved {
            break;
        }
    }
    if points.len() < 3 {
        return None;
    }
    Some(Contour {
        points,
        outer: true,
    })
}

/// Whether walking from corner `(x, y)` in `dir` runs along a boundary
/// with the inside on the RIGHT of travel.
fn edge_is_boundary<F>(x: isize, y: isize, dir: usize, inside: &F) -> bool
where
    F: Fn(isize, isize) -> bool,
{
    // The two pixels an edge separates, expressed from its start corner.
    let (left_px, right_px) = match dir {
        0 => ((x, y - 1), (x, y)),         // walking right: above / below
        1 => ((x, y), (x - 1, y)),         // walking down: right / left
        2 => ((x - 1, y), (x - 1, y - 1)), // walking left: below / above
        _ => ((x - 1, y - 1), (x, y - 1)), // walking up: left / right
    };
    !inside(left_px.0, left_px.1) && inside(right_px.0, right_px.1)
}

/// Drop points that lie (within `tolerance`) on the segment between
/// their neighbours. With `tolerance == 0` this still removes exactly
/// collinear points, which is what turns a rectangle's 2·(w+h) steps
/// into four corners.
fn simplify_contour(mut c: Contour, tolerance: f32) -> Contour {
    if c.points.len() < 3 {
        return c;
    }
    let tol = tolerance.max(0.0);
    let mut kept: Vec<(f32, f32)> = Vec::with_capacity(c.points.len());
    let n = c.points.len();
    for i in 0..n {
        let prev = *kept.last().unwrap_or(&c.points[(i + n - 1) % n]);
        let cur = c.points[i];
        let next = c.points[(i + 1) % n];
        if point_line_distance(cur, prev, next) > tol {
            kept.push(cur);
        }
    }
    if kept.len() >= 3 {
        c.points = kept;
    }
    c
}

fn point_line_distance(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    if len2 <= f32::EPSILON {
        let (ex, ey) = (p.0 - a.0, p.1 - a.1);
        return (ex * ex + ey * ey).sqrt();
    }
    ((p.0 - a.0) * dy - (p.1 - a.1) * dx).abs() / len2.sqrt()
}

/// The traced contours as JSON: `[{outer, points: [[x, y], …]}]`.
pub fn trace_json(
    coverage: &[u8],
    width: usize,
    height: usize,
    threshold: u8,
    tolerance: f32,
) -> String {
    let rows: Vec<String> = trace(coverage, width, height, threshold, tolerance)
        .iter()
        .map(|c| {
            let pts: Vec<String> = c.points.iter().map(|(x, y)| format!("[{x},{y}]")).collect();
            format!("{{\"outer\":{},\"points\":[{}]}}", c.outer, pts.join(","))
        })
        .collect();
    format!("[{}]", rows.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `w × h` coverage with the rect `[x0, x1) × [y0, y1)` selected.
    fn rect(w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<u8> {
        let mut v = vec![0u8; w * h];
        for y in y0..y1 {
            for x in x0..x1 {
                v[y * w + x] = 255;
            }
        }
        v
    }

    /// The case every marquee produces: a rectangle must trace to FOUR
    /// corners on pixel edges, not to a staircase of unit steps.
    #[test]
    fn image_paths_selection_to_path_traces_a_rect_to_four_corners() {
        let cov = rect(6, 5, 1, 1, 4, 3);
        let cs = trace(&cov, 6, 5, 128, 0.0);
        assert_eq!(cs.len(), 1, "one region, one contour");
        assert!(cs[0].outer);
        let mut pts = cs[0].points.clone();
        pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Pixel EDGES: x from 1 to 4, y from 1 to 3.
        assert_eq!(pts, vec![(1.0, 1.0), (1.0, 3.0), (4.0, 1.0), (4.0, 3.0)]);
    }

    /// An empty selection is no contours — not one degenerate contour.
    #[test]
    fn image_paths_selection_to_path_empty_selection_traces_nothing() {
        assert!(trace(&[0u8; 16], 4, 4, 128, 0.0).is_empty());
        assert!(trace(&[], 0, 0, 128, 0.0).is_empty());
        assert_eq!(trace_json(&[0u8; 16], 4, 4, 128, 0.0), "[]");
    }

    /// A full selection traces the image's own border.
    #[test]
    fn image_paths_selection_to_path_full_selection_is_the_image_border() {
        let cs = trace(&[255u8; 12], 4, 3, 128, 0.0);
        assert_eq!(cs.len(), 1);
        let mut pts = cs[0].points.clone();
        pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(pts, vec![(0.0, 0.0), (0.0, 3.0), (4.0, 0.0), (4.0, 3.0)]);
    }

    /// TWO disjoint regions are TWO contours. Merging them would produce
    /// a path that spans the gap between them.
    #[test]
    fn image_paths_selection_to_path_disjoint_regions_stay_separate() {
        let mut cov = rect(9, 3, 1, 1, 3, 2);
        for x in 6..8 {
            cov[9 + x] = 255;
        }
        let cs = trace(&cov, 9, 3, 128, 0.0);
        assert_eq!(cs.len(), 2, "two islands, two contours");
        assert!(cs.iter().all(|c| c.outer));
    }

    /// A donut is an outer ring AND a hole, flagged apart — collapsing
    /// them would silently fill the hole.
    #[test]
    fn image_paths_selection_to_path_a_hole_is_its_own_contour() {
        let mut cov = rect(5, 5, 0, 0, 5, 5);
        cov[2 * 5 + 2] = 0; // punch the centre out
        let cs = trace(&cov, 5, 5, 128, 0.0);
        assert_eq!(cs.len(), 2, "outer ring + hole");
        assert_eq!(cs.iter().filter(|c| c.outer).count(), 1);
        assert_eq!(cs.iter().filter(|c| !c.outer).count(), 1);
        let hole = cs.iter().find(|c| !c.outer).unwrap();
        // The hole is the unit square at (2,2)–(3,3).
        let mut pts = hole.points.clone();
        pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(pts, vec![(2.0, 2.0), (2.0, 3.0), (3.0, 2.0), (3.0, 3.0)]);
    }

    /// The THRESHOLD is a real parameter, not decoration: the same
    /// feathered coverage traces to different regions under different
    /// cuts, which is why the caller has to choose one.
    #[test]
    fn image_paths_selection_to_path_threshold_changes_what_is_traced() {
        // A 4×1 ramp: 0, 100, 200, 255.
        let cov = vec![0u8, 100, 200, 255];
        let low = trace(&cov, 4, 1, 50, 0.0);
        let high = trace(&cov, 4, 1, 220, 0.0);
        assert_eq!(low.len(), 1);
        assert_eq!(high.len(), 1);
        let width_of = |c: &Contour| {
            let xs: Vec<f32> = c.points.iter().map(|p| p.0).collect();
            xs.iter().cloned().fold(f32::MIN, f32::max)
                - xs.iter().cloned().fold(f32::MAX, f32::min)
        };
        assert_eq!(width_of(&low[0]), 3.0, "coverage >= 50 is three pixels");
        assert_eq!(width_of(&high[0]), 1.0, "coverage >= 220 is one");
    }

    /// The JSON the panel parses.
    #[test]
    fn image_paths_selection_to_path_json_carries_outer_and_points() {
        let json = trace_json(&rect(4, 4, 1, 1, 3, 3), 4, 4, 128, 0.0);
        assert!(json.starts_with("[{\"outer\":true,\"points\":[["), "{json}");
        assert_eq!(json.matches("[1,1]").count(), 1, "{json}");
        assert_eq!(json.matches("[3,3]").count(), 1, "{json}");
    }
}
