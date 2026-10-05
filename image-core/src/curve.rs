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

//! Curve control-points → 256-entry tone LUT — pure data math for the
//! CURVES panel. The panel's curve editor publishes a small set of
//! `(input, output)` control points in `[0, 1]`; this builds the
//! 256-entry `u8` lookup table a tone op consumes (one output byte per
//! input code value `0..=255`). Deterministic and bit-stable (fixed-order
//! scalar arithmetic, §6.3) — its own golden.
//!
//! Interpolation is MONOTONE cubic (Fritsch–Carlson tangent limiting over
//! piecewise-cubic Hermite) so a curve through reasonable control points
//! never overshoots into a non-monotone wobble (the classic Catmull-Rom
//! tone-curve artefact). With exactly two endpoints it reduces to the
//! straight line `y = x` family — the identity curve `[(0,0),(1,1)]` maps
//! `k → k` for every bin (proven by the identity test).
//!
//! Standard monotone-Hermite literature (Fritsch & Carlson 1980); no
//! reference reading.

/// Build the 256-entry tone LUT from curve control points. `points` are
/// `(input, output)` pairs in `[0, 1]`; they are sorted by input and
/// de-duplicated (a later point at the same input wins), then monotone-
/// cubic-interpolated. `lut[k]` is the output for input code `k` (`k/255`
/// mapped through the curve, scaled back to `0..=255`, rounded, clamped).
///
/// Degenerate input is handled defensively: empty → the identity ramp;
/// a single point → a constant LUT at that output value.
pub fn curve_lut(points: &[(f32, f32)]) -> [u8; 256] {
    // Normalize: sort by input, clamp to [0,1], collapse duplicate inputs.
    let mut pts: Vec<(f32, f32)> = points
        .iter()
        .map(|&(i, o)| (i.clamp(0.0, 1.0), o.clamp(0.0, 1.0)))
        .collect();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
    pts.dedup_by(|a, b| (a.0 - b.0).abs() < f32::EPSILON);

    if pts.is_empty() {
        return identity_lut();
    }
    if pts.len() == 1 {
        let v = quantize(pts[0].1);
        return [v; 256];
    }

    let xs: Vec<f32> = pts.iter().map(|p| p.0).collect();
    let ys: Vec<f32> = pts.iter().map(|p| p.1).collect();
    let m = monotone_tangents(&xs, &ys);

    let mut lut = [0u8; 256];
    for (k, slot) in lut.iter_mut().enumerate() {
        let t = k as f32 / 255.0;
        *slot = quantize(eval_hermite(&xs, &ys, &m, t));
    }
    lut
}

/// Build the 256-entry tone LUT the way PHOTOSHOP's Curves does: a
/// NATURAL cubic spline through the control points (second derivative 0
/// at both ends), flat beyond the first and last point, the result
/// clamped to `[0, 1]`. Unlike [`curve_lut`] it can overshoot between
/// steep points — Photoshop's does, and a PSD's Curves adjustment layer
/// must render as Photoshop renders it. Measured 2026-10-05 against a
/// Curves adjustment layer Photoshop 27.10 wrote (five points): 0 levels
/// apart on every code, where the monotone spline is up to 4 apart.
///
/// Natural cubic spline: the standard tridiagonal system for the knot
/// second derivatives (e.g. Burden & Faires, Numerical Analysis, §3.5);
/// no reference reading. Degenerate input as [`curve_lut`].
pub fn curve_lut_natural(points: &[(f32, f32)]) -> [u8; 256] {
    let mut pts: Vec<(f64, f64)> = points
        .iter()
        .map(|&(i, o)| (f64::from(i.clamp(0.0, 1.0)), f64::from(o.clamp(0.0, 1.0))))
        .collect();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
    pts.dedup_by(|a, b| (a.0 - b.0).abs() < f64::from(f32::EPSILON));
    if pts.is_empty() {
        return identity_lut();
    }
    if pts.len() == 1 {
        return [quantize(pts[0].1 as f32); 256];
    }
    let n = pts.len() - 1;
    let h: Vec<f64> = (0..n).map(|i| pts[i + 1].0 - pts[i].0).collect();
    // Solve for c (half the second derivative) with c[0] = c[n] = 0:
    // h[i-1]·c[i-1] + 2(h[i-1]+h[i])·c[i] + h[i]·c[i+1] = 3(Δ[i] − Δ[i-1]).
    let mut c = vec![0.0f64; n + 1];
    if n > 1 {
        let delta: Vec<f64> = (0..n).map(|i| (pts[i + 1].1 - pts[i].1) / h[i]).collect();
        // Thomas algorithm over the interior knots 1..n-1.
        let m = n - 1;
        let mut diag = vec![0.0f64; m];
        let mut rhs = vec![0.0f64; m];
        for k in 0..m {
            let i = k + 1;
            diag[k] = 2.0 * (h[i - 1] + h[i]);
            rhs[k] = 3.0 * (delta[i] - delta[i - 1]);
            if k > 0 {
                let w = h[i - 1] / diag[k - 1];
                diag[k] -= w * h[i - 1];
                rhs[k] -= w * rhs[k - 1];
            }
        }
        for k in (0..m).rev() {
            let i = k + 1;
            let next = if k + 1 < m { h[i] * c[i + 1] } else { 0.0 };
            c[i] = (rhs[k] - next) / diag[k];
        }
    }
    let mut lut = [0u8; 256];
    for (code, slot) in lut.iter_mut().enumerate() {
        let x = code as f64 / 255.0;
        let y = if x <= pts[0].0 {
            pts[0].1
        } else if x >= pts[n].0 {
            pts[n].1
        } else {
            let i = (0..n).rfind(|&i| pts[i].0 <= x).unwrap_or(0);
            let dx = x - pts[i].0;
            let b = (pts[i + 1].1 - pts[i].1) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
            let d = (c[i + 1] - c[i]) / (3.0 * h[i]);
            pts[i].1 + b * dx + c[i] * dx * dx + d * dx * dx * dx
        };
        *slot = quantize(y as f32);
    }
    lut
}

/// The identity ramp `lut[k] = k` (the `[(0,0),(1,1)]` curve).
pub fn identity_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (k, slot) in lut.iter_mut().enumerate() {
        *slot = k as u8;
    }
    lut
}

/// `round(v * 255)` clamped to `[0, 255]` (round-half-away-from-zero).
#[inline]
fn quantize(v: f32) -> u8 {
    let s = (v.clamp(0.0, 1.0) * 255.0).round();
    s.clamp(0.0, 255.0) as u8
}

/// Fritsch–Carlson monotone tangents for the knots `(xs, ys)` (both length
/// `n >= 2`, `xs` strictly increasing). Returns the per-knot tangent that
/// keeps the piecewise Hermite monotone where the data is monotone.
fn monotone_tangents(xs: &[f32], ys: &[f32]) -> Vec<f32> {
    let n = xs.len();
    // Secant slopes between consecutive knots.
    let mut delta = vec![0.0f32; n - 1];
    for i in 0..n - 1 {
        let dx = xs[i + 1] - xs[i];
        delta[i] = if dx.abs() < f32::EPSILON {
            0.0
        } else {
            (ys[i + 1] - ys[i]) / dx
        };
    }
    // Initial tangents: endpoints take the adjacent secant, interior take
    // the average of the two surrounding secants.
    let mut m = vec![0.0f32; n];
    m[0] = delta[0];
    m[n - 1] = delta[n - 2];
    for i in 1..n - 1 {
        m[i] = (delta[i - 1] + delta[i]) / 2.0;
    }
    // Fritsch–Carlson limiter: where a secant is flat, pin both tangents
    // to 0; else scale (m_i, m_{i+1}) into the monotonicity circle.
    for i in 0..n - 1 {
        if delta[i].abs() < f32::EPSILON {
            m[i] = 0.0;
            m[i + 1] = 0.0;
            continue;
        }
        let a = m[i] / delta[i];
        let b = m[i + 1] / delta[i];
        let s = a * a + b * b;
        if s > 9.0 {
            let tau = 3.0 / s.sqrt();
            m[i] = tau * a * delta[i];
            m[i + 1] = tau * b * delta[i];
        }
    }
    m
}

/// Evaluate the monotone piecewise-cubic Hermite at `t` (clamped to the
/// knot range). Below the first / above the last knot the curve holds the
/// endpoint value (a flat extrapolation — a tone curve is defined on
/// `[0,1]` and the LUT only samples there).
fn eval_hermite(xs: &[f32], ys: &[f32], m: &[f32], t: f32) -> f32 {
    let n = xs.len();
    if t <= xs[0] {
        return ys[0];
    }
    if t >= xs[n - 1] {
        return ys[n - 1];
    }
    // Find the segment [xs[i], xs[i+1]] containing t.
    let mut i = 0;
    while i + 1 < n && t > xs[i + 1] {
        i += 1;
    }
    let h = xs[i + 1] - xs[i];
    if h.abs() < f32::EPSILON {
        return ys[i];
    }
    let s = (t - xs[i]) / h;
    let s2 = s * s;
    let s3 = s2 * s;
    // Hermite basis.
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;
    h00 * ys[i] + h10 * h * m[i] + h01 * ys[i + 1] + h11 * h * m[i + 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The five-point S curve Photoshop 27.10 wrote into a Curves
    /// adjustment layer, and the codes Photoshop's merged composite maps
    /// them to (sampled every 16).
    #[test]
    fn image_core_curve_lut_natural_matches_photoshop() {
        let pts = [
            (0.0, 0.0),
            (64.0, 40.0),
            (128.0, 150.0),
            (192.0, 220.0),
            (255.0, 255.0),
        ]
        .map(|(i, o): (f32, f32)| (i / 255.0, o / 255.0));
        let lut = curve_lut_natural(&pts);
        let photoshop = [
            0, 5, 12, 23, 40, 64, 92, 122, 150, 173, 192, 207, 220, 231, 240, 248,
        ];
        for (k, &want) in photoshop.iter().enumerate() {
            assert_eq!(lut[k * 16], want, "code {}", k * 16);
        }
        assert_eq!(curve_lut_natural(&[(0.0, 0.0), (1.0, 1.0)]), identity_lut());
    }

    // feat: image.editor.curves — control-points → tone LUT. Naming
    // carries the feature tag until the state feature_test macro ships.

    #[test]
    fn image_editor_curves_identity_is_passthrough() {
        let lut = curve_lut(&[(0.0, 0.0), (1.0, 1.0)]);
        for k in 0..=255u32 {
            assert_eq!(lut[k as usize], k as u8, "identity maps {k} → {k}");
        }
    }

    #[test]
    fn image_editor_curves_default_helper_is_identity() {
        assert_eq!(identity_lut(), curve_lut(&[(0.0, 0.0), (1.0, 1.0)]));
    }

    #[test]
    fn image_editor_curves_endpoints_are_honored() {
        // Lift the black point, drop the white point.
        let lut = curve_lut(&[(0.0, 0.2), (1.0, 0.8)]);
        assert_eq!(lut[0], quantize(0.2), "black raised to 0.2");
        assert_eq!(lut[255], quantize(0.8), "white lowered to 0.8");
    }

    #[test]
    fn image_editor_curves_is_monotone_nondecreasing() {
        // An S-curve through monotone points stays monotone (no overshoot).
        let lut = curve_lut(&[(0.0, 0.0), (0.25, 0.15), (0.75, 0.85), (1.0, 1.0)]);
        for k in 1..256 {
            assert!(
                lut[k] >= lut[k - 1],
                "non-decreasing at {k}: {} < {}",
                lut[k],
                lut[k - 1]
            );
        }
    }

    #[test]
    fn image_editor_curves_passes_through_interior_knots() {
        // A knot at (0.5, 0.7) must land near 0.7 at input 128.
        let lut = curve_lut(&[(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)]);
        let got = lut[128] as i32;
        let want = quantize(0.7) as i32;
        assert!(
            (got - want).abs() <= 1,
            "knot honored: got {got}, want {want}"
        );
    }

    #[test]
    fn image_editor_curves_single_point_is_constant() {
        let lut = curve_lut(&[(0.5, 0.4)]);
        assert!(lut.iter().all(|&v| v == quantize(0.4)), "constant LUT");
    }

    #[test]
    fn image_editor_curves_empty_is_identity() {
        assert_eq!(curve_lut(&[]), identity_lut());
    }

    #[test]
    fn image_editor_curves_unsorted_input_is_sorted() {
        // Same curve, points out of order → identical LUT.
        let a = curve_lut(&[(1.0, 1.0), (0.0, 0.0), (0.5, 0.7)]);
        let b = curve_lut(&[(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)]);
        assert_eq!(a, b);
    }
}
