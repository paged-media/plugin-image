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

//! Selection COVERAGE — the persistent selection model behind the
//! editor's marquee/lasso/wand tools (spec §6.1 "selection-ready from
//! day one"): a `u8` coverage field at IMAGE resolution (`0` =
//! deselected, `255` = fully selected, intermediate = anti-aliased /
//! feathered edge), plus the shape rasterizers and combine algebra that
//! build it.
//!
//! HONESTY NOTE (GPU-only constitution, spec §6): everything here is
//! mask *preparation* — rasterizing coverage geometry, flood-filling a
//! seed, feathering the mask — which is inherently-CPU orchestration
//! work like codec entropy coding or CMS transform compilation. No
//! image pixel is ever processed here. The mask's CONSUMPTION is
//! GPU-only: [`Self::mask_window_f16`] lowers a window of the coverage
//! to the ABI's r16float `@group(2)` texture and every kernel dispatch
//! applies `out = mix(a, result, mask)` on the GPU (`image_kernels::abi`,
//! proven by `image-conformance/tests/selection_mask.rs`). The feather
//! in particular is a Gaussian on the u8 MASK, not on the image — a
//! GPU mask-blur lane can replace it later without changing this type's
//! contract.

use image_core::Region;

use crate::selection::SelectionMask;

/// How a freshly rasterized shape folds into the existing selection.
/// The algebra runs on coverage (`add` = max, `intersect` = min,
/// `subtract` = multiply by the complement) so anti-aliased edges
/// compose without stair-stepping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineMode {
    /// The shape becomes the selection.
    Replace,
    /// Union: `max(existing, shape)`.
    Add,
    /// Difference: `existing · (1 − shape)`.
    Subtract,
    /// Intersection: `min(existing, shape)`.
    Intersect,
}

impl CombineMode {
    /// Decode the wire discriminant (0 = replace, 1 = add, 2 = subtract,
    /// 3 = intersect) — the `mode` argument of the wasm selection doors.
    pub fn from_u32(v: u32) -> Option<CombineMode> {
        match v {
            0 => Some(CombineMode::Replace),
            1 => Some(CombineMode::Add),
            2 => Some(CombineMode::Subtract),
            3 => Some(CombineMode::Intersect),
            _ => None,
        }
    }
}

/// A selection as a per-pixel `u8` coverage field at image resolution
/// (row-major, `width·height` bytes). The persistent model the session
/// holds; [`SelectionMask`] (r16float tile bytes) is its lowered,
/// per-dispatch form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionCoverage {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl SelectionCoverage {
    /// An all-deselected (all-zero) coverage.
    pub fn empty(width: u32, height: u32) -> Self {
        SelectionCoverage {
            width,
            height,
            data: vec![0; (width as usize) * (height as usize)],
        }
    }

    /// An all-selected (all-255) coverage.
    pub fn full(width: u32, height: u32) -> Self {
        SelectionCoverage {
            width,
            height,
            data: vec![255; (width as usize) * (height as usize)],
        }
    }

    /// Wrap caller-supplied coverage bytes (must be `width·height` long).
    pub fn from_data(width: u32, height: u32, data: Vec<u8>) -> Option<Self> {
        if data.len() != (width as usize) * (height as usize) {
            return None;
        }
        Some(SelectionCoverage {
            width,
            height,
            data,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// The raw coverage bytes (row-major).
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Coverage at `(x, y)`; 0 outside the field.
    pub fn coverage_at(&self, x: u32, y: u32) -> u8 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.data[(y * self.width + x) as usize]
    }

    /// Every pixel fully selected? (The trivial mask — callers skip the
    /// GPU mask bind entirely and fall back to the constant-1 default.)
    pub fn is_all_one(&self) -> bool {
        self.data.iter().all(|&v| v == 255)
    }

    /// Every pixel fully deselected?
    pub fn is_all_zero(&self) -> bool {
        self.data.iter().all(|&v| v == 0)
    }

    // ── rasterizers ──────────────────────────────────────────────────

    /// Rasterize an axis-aligned rectangle `[x0, x0+rw) × [y0, y0+rh)`
    /// (image px, fractional coords allowed) with EXACT anti-aliasing:
    /// each pixel's coverage is the area of its unit square inside the
    /// rect (the product of the two 1-D overlaps), scaled to 0–255.
    pub fn rasterize_rect(width: u32, height: u32, x0: f32, y0: f32, rw: f32, rh: f32) -> Self {
        let mut out = SelectionCoverage::empty(width, height);
        if rw <= 0.0 || rh <= 0.0 {
            return out;
        }
        let (x1, y1) = (x0 + rw, y0 + rh);
        let overlap = |a0: f32, a1: f32, p: f32| -> f32 { (a1.min(p + 1.0) - a0.max(p)).max(0.0) };
        for y in 0..height {
            let cy = overlap(y0, y1, y as f32);
            if cy <= 0.0 {
                continue;
            }
            for x in 0..width {
                let c = overlap(x0, x1, x as f32) * cy;
                if c > 0.0 {
                    out.data[(y * width + x) as usize] = (c * 255.0).round().min(255.0) as u8;
                }
            }
        }
        out
    }

    /// Rasterize an axis-aligned ellipse centered at `(cx, cy)` with
    /// radii `(rx, ry)` (image px). Anti-aliasing is a deterministic
    /// 4×4 supersample of the implicit equation per boundary pixel
    /// (16 sub-samples ⇒ 17 coverage levels — visually smooth for a
    /// selection mask; an analytic-area rasterizer is a refinement,
    /// not a contract change).
    pub fn rasterize_ellipse(width: u32, height: u32, cx: f32, cy: f32, rx: f32, ry: f32) -> Self {
        let mut out = SelectionCoverage::empty(width, height);
        if rx <= 0.0 || ry <= 0.0 {
            return out;
        }
        let inside = |sx: f32, sy: f32| -> bool {
            let dx = (sx - cx) / rx;
            let dy = (sy - cy) / ry;
            dx * dx + dy * dy <= 1.0
        };
        for y in 0..height {
            for x in 0..width {
                // Quick reject/accept via the pixel's corner distances is
                // possible; the straightforward 4×4 sample keeps this
                // simple and deterministic (mask prep, not a hot path).
                let mut hits = 0u32;
                for sy in 0..4 {
                    for sx in 0..4 {
                        let fx = x as f32 + (sx as f32 + 0.5) / 4.0;
                        let fy = y as f32 + (sy as f32 + 0.5) / 4.0;
                        if inside(fx, fy) {
                            hits += 1;
                        }
                    }
                }
                if hits > 0 {
                    out.data[(y * width + x) as usize] =
                        ((hits as f32 / 16.0) * 255.0).round() as u8;
                }
            }
        }
        out
    }

    /// Rasterize a closed polygon (the lasso) by SCANLINE with
    /// anti-aliased coverage: 4 sub-scanlines per pixel row, each
    /// intersected with every edge (even–odd rule); the resulting spans
    /// accumulate ANALYTIC x-coverage per pixel at ¼ weight each
    /// (exact in x, 4× supersampled in y). `points` are `(x, y)` image-px
    /// vertices; the closing edge last→first is implicit. Fewer than 3
    /// vertices rasterize to empty.
    pub fn rasterize_polygon(width: u32, height: u32, points: &[(f32, f32)]) -> Self {
        let mut out = SelectionCoverage::empty(width, height);
        if points.len() < 3 {
            return out;
        }
        let n = points.len();
        // f32 coverage accumulator (¼ per sub-scanline span overlap).
        let mut acc = vec![0.0f32; (width as usize) * (height as usize)];
        let mut xs: Vec<f32> = Vec::with_capacity(8);
        for y in 0..height {
            for sub in 0..4u32 {
                let sy = y as f32 + (sub as f32 * 2.0 + 1.0) / 8.0;
                xs.clear();
                for i in 0..n {
                    let (x0, y0) = points[i];
                    let (x1, y1) = points[(i + 1) % n];
                    // Half-open in y (`[min, max)`) so a vertex exactly on
                    // the scanline counts once — the classic even–odd
                    // crossing rule.
                    if (y0 <= sy && sy < y1) || (y1 <= sy && sy < y0) {
                        let t = (sy - y0) / (y1 - y0);
                        xs.push(x0 + t * (x1 - x0));
                    }
                }
                xs.sort_by(|a, b| a.partial_cmp(b).expect("finite intersections"));
                for pair in xs.chunks_exact(2) {
                    let (xa, xb) = (pair[0].max(0.0), pair[1].min(width as f32));
                    if xb <= xa {
                        continue;
                    }
                    let first = xa.floor() as u32;
                    let last = (xb.ceil() as u32).min(width);
                    for px in first..last {
                        let cover = (xb.min(px as f32 + 1.0) - xa.max(px as f32)).max(0.0);
                        acc[(y * width + px) as usize] += cover * 0.25;
                    }
                }
            }
        }
        for (dst, &a) in out.data.iter_mut().zip(acc.iter()) {
            *dst = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        out
    }

    /// Magic-wand coverage over straight-RGBA8 `rgba` (`width·height·4`
    /// bytes): every pixel whose color distance to the seed pixel is
    /// within `tolerance` — CONTIGUOUS = 4-connected BFS flood from the
    /// seed; non-contiguous = a global threshold over all pixels. The
    /// distance is the Chebyshev (max per-channel absolute difference)
    /// over R, G, B, A — per-channel tolerance 0–255, deterministic.
    /// The result is BINARY coverage (255/0) — wand edges are hard;
    /// [`Self::feather`] softens on request. Out-of-bounds seed or a
    /// mis-sized buffer yields empty.
    pub fn magic_wand(
        width: u32,
        height: u32,
        rgba: &[u8],
        seed_x: u32,
        seed_y: u32,
        tolerance: u8,
        contiguous: bool,
    ) -> Self {
        let mut out = SelectionCoverage::empty(width, height);
        let npx = (width as usize) * (height as usize);
        if rgba.len() != npx * 4 || seed_x >= width || seed_y >= height {
            return out;
        }
        let seed_i = ((seed_y * width + seed_x) as usize) * 4;
        let seed: [u8; 4] = rgba[seed_i..seed_i + 4].try_into().expect("4 bytes");
        let within = |i: usize| -> bool {
            let p = &rgba[i * 4..i * 4 + 4];
            p.iter()
                .zip(seed.iter())
                .all(|(&a, &b)| a.abs_diff(b) <= tolerance)
        };
        if contiguous {
            let mut queue = std::collections::VecDeque::new();
            let start = (seed_y * width + seed_x) as usize;
            out.data[start] = 255;
            queue.push_back((seed_x, seed_y));
            while let Some((x, y)) = queue.pop_front() {
                let neighbors = [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ];
                for (nx, ny) in neighbors {
                    if nx >= width || ny >= height {
                        continue;
                    }
                    let i = (ny * width + nx) as usize;
                    if out.data[i] == 0 && within(i) {
                        out.data[i] = 255;
                        queue.push_back((nx, ny));
                    }
                }
            }
        } else {
            for i in 0..npx {
                if within(i) {
                    out.data[i] = 255;
                }
            }
        }
        out
    }

    // ── algebra ──────────────────────────────────────────────────────

    /// Fold `shape` (same dimensions) into this coverage under `mode`.
    /// Mismatched dimensions are a no-op (defensive — the session always
    /// rasterizes at the bound image's resolution).
    pub fn combine(&mut self, shape: &SelectionCoverage, mode: CombineMode) {
        if shape.width != self.width || shape.height != self.height {
            return;
        }
        match mode {
            CombineMode::Replace => self.data.copy_from_slice(&shape.data),
            CombineMode::Add => {
                for (a, &b) in self.data.iter_mut().zip(shape.data.iter()) {
                    *a = (*a).max(b);
                }
            }
            CombineMode::Subtract => {
                // a · (1 − b): the coverage product keeps AA edges smooth
                // (u16 math with the +127 round).
                for (a, &b) in self.data.iter_mut().zip(shape.data.iter()) {
                    *a = ((*a as u16 * (255 - b as u16) + 127) / 255) as u8;
                }
            }
            CombineMode::Intersect => {
                for (a, &b) in self.data.iter_mut().zip(shape.data.iter()) {
                    *a = (*a).min(b);
                }
            }
        }
    }

    /// Invert the coverage (`255 − v` per pixel).
    pub fn invert(&mut self) {
        for v in &mut self.data {
            *v = 255 - *v;
        }
    }

    /// Feather: a separable Gaussian of `sigma` px over the coverage,
    /// radius `ceil(3σ)`. Pixels outside the image count as coverage 0
    /// (a selection cannot extend past the canvas, so it fades toward
    /// the border rather than clamping). CPU by design — this blurs the
    /// MASK, not the image (mask prep; see the module note). `sigma <= 0`
    /// is a no-op.
    pub fn feather(&mut self, sigma: f32) {
        if sigma <= 0.0 || self.data.is_empty() {
            return;
        }
        let radius = (sigma * 3.0).ceil() as i64;
        let mut weights = Vec::with_capacity((2 * radius + 1) as usize);
        let s2 = 2.0 * sigma * sigma;
        for i in -radius..=radius {
            weights.push((-((i * i) as f32) / s2).exp());
        }
        let norm: f32 = weights.iter().sum();
        for w in &mut weights {
            *w /= norm;
        }

        let (w, h) = (self.width as i64, self.height as i64);
        // Horizontal pass into an f32 buffer, vertical pass back to u8.
        let mut mid = vec![0.0f32; self.data.len()];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0f32;
                for (k, &wt) in weights.iter().enumerate() {
                    let sx = x + (k as i64 - radius);
                    if sx >= 0 && sx < w {
                        acc += wt * self.data[(y * w + sx) as usize] as f32;
                    }
                }
                mid[(y * w + x) as usize] = acc;
            }
        }
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0f32;
                for (k, &wt) in weights.iter().enumerate() {
                    let sy = y + (k as i64 - radius);
                    if sy >= 0 && sy < h {
                        acc += wt * mid[(sy * w + x) as usize];
                    }
                }
                self.data[(y * w + x) as usize] = acc.clamp(0.0, 255.0).round() as u8;
            }
        }
    }

    // ── readouts + lowering ──────────────────────────────────────────

    /// The bounding box of non-zero coverage, or `None` when nothing is
    /// selected.
    pub fn bounds(&self) -> Option<Region> {
        let (mut min_x, mut min_y) = (u32::MAX, u32::MAX);
        let (mut max_x, mut max_y) = (0u32, 0u32);
        let mut any = false;
        for y in 0..self.height {
            let row = &self.data[(y * self.width) as usize..((y + 1) * self.width) as usize];
            if let Some(first) = row.iter().position(|&v| v > 0) {
                let last = row
                    .iter()
                    .rposition(|&v| v > 0)
                    .expect("first implies last");
                any = true;
                min_x = min_x.min(first as u32);
                max_x = max_x.max(last as u32);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
        if !any {
            return None;
        }
        Some(Region::new(
            min_x as i32,
            min_y as i32,
            max_x - min_x + 1,
            max_y - min_y + 1,
        ))
    }

    /// The selected fraction of the image (mean coverage, 0.0–1.0).
    pub fn selected_fraction(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.data.iter().map(|&v| v as u64).sum();
        sum as f64 / (255.0 * self.data.len() as f64)
    }

    /// Lower a WINDOW of the coverage to the ABI's r16float mask bytes
    /// (`region.w · region.h` texels, 2 bytes each — the `mask` argument
    /// of `execute_tile_once`). Texels outside the coverage field read
    /// as 0 (deselected). This is the ONE lowering point, so the f16
    /// quantization matches [`SelectionMask`] exactly.
    pub fn mask_window_f16(&self, region: Region) -> Vec<u8> {
        SelectionMask::from_fn(region.w, region.h, |x, y| {
            let ix = region.x + x as i32;
            let iy = region.y + y as i32;
            if ix < 0 || iy < 0 {
                return 0.0;
            }
            self.coverage_at(ix as u32, iy as u32) as f32 / 255.0
        })
        .into_bytes()
    }

    /// FNV-1a over dimensions + coverage bytes — the cache-key component
    /// the pipeline folds into masked apply nodes (a different selection
    /// must never serve a stale cached tile).
    pub fn content_hash(&self) -> u64 {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = OFFSET;
        let mut eat = |b: u8| {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        };
        for b in self.width.to_le_bytes() {
            eat(b);
        }
        for b in self.height.to_le_bytes() {
            eat(b);
        }
        for &b in &self.data {
            eat(b);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── rect (exact AA) ──────────────────────────────────────────────

    #[test]
    fn rect_integer_bounds_is_hard_edged() {
        let c = SelectionCoverage::rasterize_rect(6, 4, 1.0, 1.0, 3.0, 2.0);
        for y in 0..4 {
            for x in 0..6 {
                let inside = (1..4).contains(&x) && (1..3).contains(&y);
                assert_eq!(
                    c.coverage_at(x, y),
                    if inside { 255 } else { 0 },
                    "({x},{y})"
                );
            }
        }
    }

    #[test]
    fn rect_half_pixel_edge_covers_half() {
        // Rect starting at x=1.5: pixel column 1 is half covered.
        let c = SelectionCoverage::rasterize_rect(4, 1, 1.5, 0.0, 2.5, 1.0);
        assert_eq!(c.coverage_at(0, 0), 0);
        assert_eq!(c.coverage_at(1, 0), 128); // 0.5 * 255 rounds to 128
        assert_eq!(c.coverage_at(2, 0), 255);
        assert_eq!(c.coverage_at(3, 0), 255);
    }

    #[test]
    fn rect_corner_pixel_is_area_product() {
        // Rect (0.5, 0.5)–(1.0, 1.0): pixel (0,0) covered 0.25.
        let c = SelectionCoverage::rasterize_rect(2, 2, 0.5, 0.5, 0.5, 0.5);
        assert_eq!(c.coverage_at(0, 0), 64); // 0.25 * 255 = 63.75 → 64
        assert_eq!(c.coverage_at(1, 1), 0);
    }

    // ── ellipse ──────────────────────────────────────────────────────

    #[test]
    fn ellipse_center_full_outside_zero_edge_partial() {
        let c = SelectionCoverage::rasterize_ellipse(8, 8, 4.0, 4.0, 3.0, 3.0);
        assert_eq!(c.coverage_at(4, 4), 255, "center");
        assert_eq!(c.coverage_at(0, 0), 0, "far corner");
        // A boundary pixel gets intermediate coverage (AA).
        let edge = c.coverage_at(4, 1);
        assert!(
            edge > 0 && edge < 255,
            "edge coverage {edge} should be partial"
        );
    }

    // ── polygon scanline ─────────────────────────────────────────────

    #[test]
    fn polygon_square_matches_rect() {
        let poly = SelectionCoverage::rasterize_polygon(
            6,
            6,
            &[(1.0, 1.0), (4.0, 1.0), (4.0, 4.0), (1.0, 4.0)],
        );
        let rect = SelectionCoverage::rasterize_rect(6, 6, 1.0, 1.0, 3.0, 3.0);
        // Interior pixels agree exactly; y-edges may differ by the 4×
        // sub-scanline quantization, so compare where rect is 0 or 255.
        for y in 0..6 {
            for x in 0..6 {
                let r = rect.coverage_at(x, y);
                if r == 255 {
                    assert_eq!(poly.coverage_at(x, y), 255, "interior ({x},{y})");
                } else if r == 0 {
                    assert_eq!(poly.coverage_at(x, y), 0, "exterior ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn polygon_triangle_covers_half_the_square() {
        // Right triangle over a 16×16 box: coverage ≈ half the box area.
        let c =
            SelectionCoverage::rasterize_polygon(16, 16, &[(0.0, 0.0), (16.0, 0.0), (0.0, 16.0)]);
        let frac = c.selected_fraction();
        assert!(
            (frac - 0.5).abs() < 0.02,
            "triangle fraction {frac} should be ≈ 0.5"
        );
    }

    #[test]
    fn polygon_degenerate_is_empty() {
        let c = SelectionCoverage::rasterize_polygon(4, 4, &[(0.0, 0.0), (3.0, 3.0)]);
        assert!(c.is_all_zero());
    }

    #[test]
    fn polygon_diagonal_edge_is_antialiased() {
        let c = SelectionCoverage::rasterize_polygon(8, 8, &[(0.0, 0.0), (8.0, 8.0), (0.0, 8.0)]);
        // A pixel the diagonal crosses gets partial coverage.
        let v = c.coverage_at(3, 3);
        assert!(v > 0 && v < 255, "diagonal pixel coverage {v}");
        // A pixel fully under the diagonal is solid.
        assert_eq!(c.coverage_at(0, 7), 255);
        // A pixel fully above it is empty.
        assert_eq!(c.coverage_at(7, 0), 0);
    }

    // ── magic wand (flood) ───────────────────────────────────────────

    /// 4×1 RGBA: red, red-ish, blue, red — the wand fixture.
    fn wand_rgba() -> Vec<u8> {
        [
            [255u8, 0, 0, 255],
            [250, 5, 0, 255],
            [0, 0, 255, 255],
            [255, 0, 0, 255],
        ]
        .concat()
    }

    #[test]
    fn wand_contiguous_stops_at_the_color_wall() {
        let c = SelectionCoverage::magic_wand(4, 1, &wand_rgba(), 0, 0, 10, true);
        assert_eq!(c.coverage_at(0, 0), 255);
        assert_eq!(c.coverage_at(1, 0), 255, "within tolerance, connected");
        assert_eq!(c.coverage_at(2, 0), 0, "blue wall");
        assert_eq!(c.coverage_at(3, 0), 0, "matching color but NOT connected");
    }

    #[test]
    fn wand_non_contiguous_is_a_global_threshold() {
        let c = SelectionCoverage::magic_wand(4, 1, &wand_rgba(), 0, 0, 10, false);
        assert_eq!(c.coverage_at(0, 0), 255);
        assert_eq!(c.coverage_at(1, 0), 255);
        assert_eq!(c.coverage_at(2, 0), 0);
        assert_eq!(
            c.coverage_at(3, 0),
            255,
            "global: disconnected match selects"
        );
    }

    #[test]
    fn wand_tolerance_zero_is_exact_match_only() {
        let c = SelectionCoverage::magic_wand(4, 1, &wand_rgba(), 0, 0, 0, true);
        assert_eq!(c.coverage_at(0, 0), 255);
        assert_eq!(c.coverage_at(1, 0), 0, "5 off in G > tolerance 0");
    }

    #[test]
    fn wand_out_of_bounds_seed_is_empty() {
        let c = SelectionCoverage::magic_wand(4, 1, &wand_rgba(), 9, 0, 10, true);
        assert!(c.is_all_zero());
    }

    #[test]
    fn wand_flood_crosses_rows() {
        // 2×2 all-red: contiguous flood from (0,0) reaches everything.
        let rgba = [[255u8, 0, 0, 255]; 4].concat();
        let c = SelectionCoverage::magic_wand(2, 2, &rgba, 0, 0, 0, true);
        assert!(c.is_all_one());
    }

    // ── combine algebra ──────────────────────────────────────────────

    #[test]
    fn combine_add_is_max_subtract_is_product_intersect_is_min() {
        let a0 = SelectionCoverage::from_data(2, 1, vec![100, 200]).unwrap();
        let b = SelectionCoverage::from_data(2, 1, vec![150, 60]).unwrap();

        let mut add = a0.clone();
        add.combine(&b, CombineMode::Add);
        assert_eq!(add.data(), &[150, 200]);

        let mut sub = a0.clone();
        sub.combine(&b, CombineMode::Subtract);
        // 100·(255−150)/255 ≈ 41; 200·(255−60)/255 ≈ 153.
        assert_eq!(sub.data(), &[41, 153]);

        let mut int = a0.clone();
        int.combine(&b, CombineMode::Intersect);
        assert_eq!(int.data(), &[100, 60]);

        let mut rep = a0.clone();
        rep.combine(&b, CombineMode::Replace);
        assert_eq!(rep.data(), b.data());
    }

    #[test]
    fn combine_subtract_extremes_are_exact() {
        let mut a = SelectionCoverage::from_data(2, 1, vec![255, 255]).unwrap();
        let b = SelectionCoverage::from_data(2, 1, vec![255, 0]).unwrap();
        a.combine(&b, CombineMode::Subtract);
        assert_eq!(
            a.data(),
            &[0, 255],
            "full minus full = 0; full minus none = full"
        );
    }

    #[test]
    fn combine_dimension_mismatch_is_a_noop() {
        let mut a = SelectionCoverage::full(2, 2);
        let b = SelectionCoverage::empty(3, 3);
        a.combine(&b, CombineMode::Replace);
        assert!(a.is_all_one());
    }

    #[test]
    fn invert_flips_coverage() {
        let mut c = SelectionCoverage::from_data(3, 1, vec![0, 100, 255]).unwrap();
        c.invert();
        assert_eq!(c.data(), &[255, 155, 0]);
    }

    // ── feather ──────────────────────────────────────────────────────

    #[test]
    fn feather_softens_a_hard_edge_monotonically() {
        // A hard left-half selection on a 16×16 field; after feathering,
        // the mid-row profile (far from the top/bottom borders, so the
        // vertical pass is fully in-bounds) is monotone non-increasing
        // with soft intermediate values around the seam.
        let mut c = SelectionCoverage::rasterize_rect(16, 16, 0.0, 0.0, 8.0, 16.0);
        c.feather(1.5);
        let row: Vec<u8> = (0..16).map(|x| c.coverage_at(x, 8)).collect();
        // From the selection's plateau rightward across the seam (x=0..3
        // also fades — the LEFT canvas border, zero-padded by design).
        for i in 5..16 {
            assert!(
                row[i] <= row[i - 1],
                "monotone non-increasing at {i}: {row:?}"
            );
        }
        let seam = c.coverage_at(8, 8);
        assert!(
            seam > 0 && seam < 255,
            "seam coverage {seam} should be soft"
        );
    }

    #[test]
    fn feather_zero_sigma_is_a_noop() {
        let mut c = SelectionCoverage::rasterize_rect(8, 8, 2.0, 2.0, 4.0, 4.0);
        let before = c.clone();
        c.feather(0.0);
        assert_eq!(c, before);
    }

    #[test]
    fn feather_fades_at_the_image_border() {
        // A full coverage feathers DOWN near the border (outside counts
        // as 0 — the selection cannot extend past the canvas); the deep
        // interior (both passes fully in-bounds) stays solid.
        let mut c = SelectionCoverage::full(9, 9);
        c.feather(1.0);
        assert!(c.coverage_at(0, 0) < 255, "corner fades");
        assert_eq!(c.coverage_at(4, 4), 255, "interior stays solid");
    }

    // ── readouts ─────────────────────────────────────────────────────

    #[test]
    fn bounds_of_a_rect_selection() {
        let c = SelectionCoverage::rasterize_rect(10, 10, 2.0, 3.0, 4.0, 5.0);
        let b = c.bounds().expect("non-empty");
        assert_eq!((b.x, b.y, b.w, b.h), (2, 3, 4, 5));
    }

    #[test]
    fn bounds_of_empty_is_none() {
        assert!(SelectionCoverage::empty(4, 4).bounds().is_none());
    }

    #[test]
    fn selected_fraction_full_and_half() {
        assert_eq!(SelectionCoverage::full(4, 4).selected_fraction(), 1.0);
        let half = SelectionCoverage::rasterize_rect(4, 4, 0.0, 0.0, 2.0, 4.0);
        assert!((half.selected_fraction() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn mask_window_matches_coverage_and_zero_pads() {
        let c = SelectionCoverage::rasterize_rect(4, 4, 1.0, 1.0, 2.0, 2.0);
        // A window straddling the field edge: outside texels are 0.
        let bytes = c.mask_window_f16(Region::new(3, 3, 2, 2));
        let m = SelectionMask::from_bytes(2, 2, bytes).unwrap();
        assert_eq!(m.weight_at(0, 0), 0.0, "coverage(3,3) is outside the rect");
        assert_eq!(m.weight_at(1, 1), 0.0, "off-field texel reads 0");
        // A window inside the selection is all-one.
        let bytes = c.mask_window_f16(Region::new(1, 1, 2, 2));
        let m = SelectionMask::from_bytes(2, 2, bytes).unwrap();
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(m.weight_at(x, y), 1.0);
            }
        }
    }

    #[test]
    fn content_hash_tracks_the_coverage() {
        let a = SelectionCoverage::rasterize_rect(8, 8, 1.0, 1.0, 3.0, 3.0);
        let b = SelectionCoverage::rasterize_rect(8, 8, 2.0, 2.0, 3.0, 3.0);
        assert_ne!(a.content_hash(), b.content_hash());
        assert_eq!(a.content_hash(), a.clone().content_hash());
    }

    #[test]
    fn mode_wire_decoding() {
        assert_eq!(CombineMode::from_u32(0), Some(CombineMode::Replace));
        assert_eq!(CombineMode::from_u32(1), Some(CombineMode::Add));
        assert_eq!(CombineMode::from_u32(2), Some(CombineMode::Subtract));
        assert_eq!(CombineMode::from_u32(3), Some(CombineMode::Intersect));
        assert_eq!(CombineMode::from_u32(4), None);
    }
}
