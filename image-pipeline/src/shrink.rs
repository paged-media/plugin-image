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

//! Shrink-on-load planning (spec §7.2): push as much downscale as
//! possible into the decoder, leave the remainder to a resample kernel.
//!
//! A decoder advertises the native downscale factors it can apply itself
//! (`ImageSource::native_shrink` — `[1]` for PNG, `[1,2,4,8]` for JPEG
//! DCT scaling). A pull that wants the image at `requested_scale` (a
//! downscale FRACTION in `(0, 1]`, e.g. `0.5` = half size) is split into
//! two stages:
//!
//! 1. the decoder reads at the largest native shrink `s` whose implied
//!    fraction `1/s` is still ≥ the requested fraction (i.e. `s ≤
//!    1/requested_scale`) — the most decode-side work we can take without
//!    over-shrinking past the target;
//! 2. a `Resample` kernel covers the residual fraction so the composed
//!    result lands exactly at `requested_scale`.
//!
//! The composition invariant the planner guarantees:
//! `(1.0 / shrink) * residual == requested_scale` (decode produces a
//! `1/shrink` fraction, the residual resample finishes the trip) — i.e.
//! `residual == requested_scale * shrink`.

/// Plan the decode shrink + residual resample for `requested_scale`
/// against a decoder's `native` shrink factors.
///
/// Returns `(shrink, residual)` where `shrink ∈ native` is the largest
/// native downscale that does not overshoot the request, and `residual`
/// is the resample fraction the kernel must still apply so that
/// `(1.0 / shrink) * residual == requested_scale`.
///
/// `native` MUST be ascending and contain `1` (the `ImageSource`
/// contract); `requested_scale` is clamped into `(0, 1]` (an upscale or
/// non-positive request decodes at full resolution — `shrink == 1` — and
/// hands the whole job to the resample kernel).
pub fn plan_shrink(native: &[u32], requested_scale: f32) -> (u32, f32) {
    // Clamp to the meaningful downscale range. A request ≥ 1.0 (full size
    // or upscale) or a non-finite/non-positive value never shrinks at the
    // decoder; the resample kernel owns any upscale.
    let scale = if requested_scale.is_finite() && requested_scale > 0.0 {
        requested_scale.min(1.0)
    } else {
        1.0
    };

    // The decoder may shrink by `s` (fraction `1/s`) only while it stays
    // at or above the requested fraction — i.e. `1/s ≥ scale`, i.e.
    // `s ≤ 1/scale`. Pick the largest such native factor (the list is
    // ascending and contains 1, so this always finds at least 1).
    let max_shrink = 1.0 / scale;
    let shrink = native
        .iter()
        .copied()
        .filter(|&s| s >= 1 && (s as f32) <= max_shrink)
        .max()
        .unwrap_or(1);

    // residual = requested_scale * shrink  ⇒  (1/shrink) * residual == scale.
    let residual = scale * shrink as f32;
    (shrink, residual)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The composition invariant: decoding at `1/shrink` then resampling
    /// by `residual` lands at the requested fraction.
    fn assert_composes(native: &[u32], scale: f32) -> (u32, f32) {
        let (shrink, residual) = plan_shrink(native, scale);
        let composed = (1.0 / shrink as f32) * residual;
        assert!(
            (composed - scale).abs() < 1e-6,
            "scale {scale}: (1/{shrink})*{residual} = {composed} != {scale}"
        );
        (shrink, residual)
    }

    #[test]
    fn plan_shrink_png_single_factor() {
        // PNG advertises only [1]: every request decodes full-res, the
        // resample kernel does all the downscaling.
        assert_eq!(plan_shrink(&[1], 1.0), (1, 1.0));
        assert_eq!(plan_shrink(&[1], 0.5), (1, 0.5));
        assert_eq!(plan_shrink(&[1], 0.1), (1, 0.1));
    }

    #[test]
    fn plan_shrink_jpeg_dct_ladder_matrix() {
        // The spec §7.2 matrix: [1,2,4,8] × {1.0, 0.5, 0.3, 0.1}.
        let native = [1u32, 2, 4, 8];

        let (s, _) = assert_composes(&native, 1.0);
        assert_eq!(s, 1, "full size never shrinks at the decoder");

        let (s, r) = assert_composes(&native, 0.5);
        assert_eq!((s, r), (2, 1.0), "exact 1/2 ⇒ decode shrink 2, no residual");

        let (s, r) = assert_composes(&native, 0.3);
        assert_eq!(s, 2, "0.3 ⇒ largest shrink ≤ 1/0.3≈3.33 is 2");
        assert!((r - 0.6).abs() < 1e-6, "residual {r} != 0.6");

        let (s, r) = assert_composes(&native, 0.1);
        assert_eq!(s, 8, "0.1 ⇒ largest shrink ≤ 1/0.1=10 is 8");
        assert!((r - 0.8).abs() < 1e-6, "residual {r} != 0.8");
    }

    #[test]
    fn plan_shrink_clamps_upscale_and_garbage() {
        // Upscale, zero, negative, NaN all decode full-res with the whole
        // scale handed to the resample stage.
        assert_eq!(plan_shrink(&[1, 2, 4, 8], 2.0), (1, 1.0));
        assert_eq!(plan_shrink(&[1, 2, 4, 8], 0.0), (1, 1.0));
        assert_eq!(plan_shrink(&[1, 2, 4, 8], -0.5), (1, 1.0));
        assert_eq!(plan_shrink(&[1, 2, 4, 8], f32::NAN), (1, 1.0));
    }

    #[test]
    fn plan_shrink_never_overshoots() {
        // Whatever the request, the chosen native fraction 1/shrink is
        // never SMALLER than the requested fraction (no over-shrink): the
        // residual is always ≥ requested_scale, i.e. ≤ 1.0 upscale never
        // needed at the decoder.
        let native = [1u32, 2, 4, 8];
        for &scale in &[0.95f32, 0.51, 0.49, 0.26, 0.13, 0.12] {
            let (shrink, residual) = plan_shrink(&native, scale);
            assert!(
                1.0 / shrink as f32 >= scale - 1e-6,
                "scale {scale}: decode fraction 1/{shrink} overshot the target"
            );
            assert!(
                residual <= 1.0 + 1e-6,
                "scale {scale}: residual {residual} would upscale"
            );
        }
    }
}
