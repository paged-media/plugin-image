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

//! Distance transform conformance (spec §11, a T3 breadth op): the CPU
//! two-pass chamfer transform over a binary-mask tile is its own exact
//! reference (deterministic, fixed sweep order, §6.3). The oracles are
//! hand-computed chamfer distances, not a GPU readback — the production
//! jump-flood GPU version is the M3 follow-up and will be verified BY
//! TOLERANCE against this value.
//!
//! feat: image.kernel.distance-transform.

use half::f16;
use image_gpu::{distance_transform, DistanceParams, MaskChannel};

const BPP_RGBA: usize = 8;

/// Decode the distance (R channel) of texel `i` from the rgba16float
/// output.
fn dist(out: &[u8], i: usize) -> f32 {
    let o = i * BPP_RGBA;
    f16::from_bits(u16::from_le_bytes([out[o], out[o + 1]])).to_f32()
}

/// Build an `r16float` binary-mask tile from a foreground predicate.
fn r16_mask(w: usize, h: usize, fg: impl Fn(usize, usize) -> bool) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 2);
    for y in 0..h {
        for x in 0..w {
            let val = if fg(x, y) { 1.0f32 } else { 0.0f32 };
            v.extend_from_slice(&f16::from_f32(val).to_bits().to_le_bytes());
        }
    }
    v
}

fn r16_params() -> DistanceParams {
    DistanceParams {
        channel: MaskChannel::R16,
        ..DistanceParams::default()
    }
}

/// A single foreground pixel: every other texel's distance is the
/// chamfer-(1,√2) distance to it, asserted at hand-computed cells.
#[test]
fn image_kernel_distance_transform_single_pixel_chamfer_field() {
    let (w, h) = (7usize, 7usize);
    let (fx, fy) = (3usize, 3usize); // center seed
    let mask = r16_mask(w, h, |x, y| x == fx && y == fy);
    let out = distance_transform(&mask, w as u32, h as u32, r16_params());

    let s2 = std::f32::consts::SQRT_2;
    // chamfer-(1,√2): max(|dx|,|dy|) + (√2−1)·min(|dx|,|dy|).
    let chamfer = |x: usize, y: usize| -> f32 {
        let dx = (x as i32 - fx as i32).unsigned_abs() as f32;
        let dy = (y as i32 - fy as i32).unsigned_abs() as f32;
        dx.max(dy) + (s2 - 1.0) * dx.min(dy)
    };

    // Hand-checked spot cells.
    let cells = [
        (3usize, 3usize, 0.0), // seed
        (4, 3, 1.0),           // +x orthogonal
        (3, 5, 2.0),           // +2y orthogonal
        (4, 4, s2),            // pure diagonal
        (5, 5, 2.0 * s2),      // 2 diagonals
        (5, 4, 1.0 + s2),      // dx=2,dy=1
        (6, 6, 3.0 * s2),      // far corner, dx=dy=3
        (0, 3, 3.0),           // 3 left
    ];
    for (x, y, want) in cells {
        let got = dist(&out, y * w + x);
        assert!(
            (got - want).abs() < 1e-2,
            "({x},{y}): got {got}, want hand-chamfer {want}"
        );
    }

    // And the WHOLE field matches the closed-form chamfer expression
    // (the two-pass result for a single seed is exactly chamfer-(1,√2)).
    for y in 0..h {
        for x in 0..w {
            let got = dist(&out, y * w + x);
            let want = chamfer(x, y);
            assert!(
                (got - want).abs() < 1e-2,
                "({x},{y}): got {got}, want closed-form chamfer {want}"
            );
        }
    }
}

/// A fully-foreground field: every texel is a seed → all distances zero.
#[test]
fn image_kernel_distance_transform_fully_foreground_is_all_zero() {
    let (w, h) = (16usize, 16usize);
    let mask = r16_mask(w, h, |_, _| true);
    let out = distance_transform(&mask, w as u32, h as u32, r16_params());
    for i in 0..(w * h) {
        assert_eq!(dist(&out, i), 0.0, "all-foreground tile is zero at {i}");
    }
}

/// An empty (all-background) field: there is no foreground to be near, so
/// every texel saturates at the finite max distance (no `+∞`/NaN).
#[test]
fn image_kernel_distance_transform_empty_field_is_max_distance() {
    let (w, h) = (16usize, 16usize);
    let mask = r16_mask(w, h, |_, _| false);
    let out = distance_transform(&mask, w as u32, h as u32, r16_params());
    let far = dist(&out, 0);
    assert!(far.is_finite(), "empty field is finite, not +inf");
    assert!(
        far > 1000.0,
        "empty field saturates at a large distance ({far})"
    );
    for i in 0..(w * h) {
        assert_eq!(
            dist(&out, i),
            far,
            "uniform max across an empty tile at {i}"
        );
    }
}

/// The alpha channel can carry the mask on an rgba16float tile (R=0
/// everywhere), proving the channel selector reads A, not R.
#[test]
fn image_kernel_distance_transform_alpha_channel_mask() {
    let (w, h) = (5usize, 5usize);
    let mut v = Vec::with_capacity(w * h * BPP_RGBA);
    for y in 0..h {
        for x in 0..w {
            let a = if x == 2 && y == 2 { 1.0f32 } else { 0.0f32 };
            for c in [0.0, 0.0, 0.0, a] {
                v.extend_from_slice(&f16::from_f32(c).to_bits().to_le_bytes());
            }
        }
    }
    let params = DistanceParams {
        channel: MaskChannel::A,
        ..DistanceParams::default()
    };
    let out = distance_transform(&v, w as u32, h as u32, params);
    assert_eq!(dist(&out, 2 * w + 2), 0.0, "A-foreground seeds at (2,2)");
    assert_eq!(
        dist(&out, 2 * w + 3),
        1.0,
        "orthogonal neighbour distance 1"
    );
}
