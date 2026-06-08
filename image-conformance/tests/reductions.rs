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

//! T2 reductions (spec §11): per-channel histogram + statistics over an
//! rgba16float tile. These are NOT KernelDefs (a reduction collapses a
//! tile to a table/scalars, not one texel per texel) — there is no GPU
//! parity lane and no registry kernel row; the M2 CPU value IS the
//! golden (it is fixed-order scalar arithmetic, bit-stable by
//! construction, §6.3). So unlike the family parity tests there is no
//! "SKIP: no GPU adapter" branch: these run unconditionally on the CPU.
//!
//! feat: image.reduce.histogram / image.reduce.statistics — these need
//! STATE-registry rows (stage `plugin.image`), NOT `registry/kernels.yaml`
//! rows, because they are state-bearing editor operations, not per-texel
//! kernels. The orchestrator adds the state rows.

use image_conformance::harness::RefTile;
use image_conformance::Px;
use image_gpu::reduce::{histogram, statistics, Histogram};

/// rgba16float bytes for a `w`×`h` tile of one constant texel.
fn constant_bytes(w: u32, h: u32, px: [f32; 4]) -> Vec<u8> {
    RefTile::from_fn(w, h, |_, _| Px(px)).f16_bytes()
}

/// The single non-empty bin index for a constant channel value, by the
/// reduction's own quantizer (`round(v * 255)` clamped to `[0,255]`).
fn expect_bin(v: f32) -> usize {
    let s = (v * 255.0).round();
    s.clamp(0.0, 255.0) as usize
}

/// Total counts in a channel histogram (must always equal `w*h`).
fn channel_total(h: &Histogram, c: usize) -> u32 {
    h.bins[c].iter().sum()
}

// ── histogram ───────────────────────────────────────────────────────

/// Constant tile → exactly ONE bin per channel holds `w*h`; all others
/// are zero, and the per-channel total is `w*h`.
#[test]
fn image_reduce_histogram_constant_tile_single_bin() {
    let (w, h) = (32u32, 16u32);
    // 0.5 / 0.25 / 0.75 / 1.0 are exactly representable in f16.
    let px = [0.5, 0.25, 0.75, 1.0];
    let bytes = constant_bytes(w, h, px);
    let hist = histogram(&bytes, w, h);

    let n = w * h;
    let expect = [
        expect_bin(px[0]), // round(127.5) = 128
        expect_bin(px[1]), // round(63.75) = 64
        expect_bin(px[2]), // round(191.25) = 191
        expect_bin(px[3]), // 255
    ];
    assert_eq!(expect, [128, 64, 191, 255], "quantizer mapping changed");

    for (c, &bin) in expect.iter().enumerate() {
        assert_eq!(
            channel_total(&hist, c),
            n,
            "channel {c} bins must sum to w*h"
        );
        assert_eq!(
            hist.bins[c][bin], n,
            "channel {c} bin {bin} must hold all texels"
        );
        let nonzero = hist.bins[c].iter().filter(|&&b| b != 0).count();
        assert_eq!(nonzero, 1, "channel {c} must have a single spike");
    }
}

/// Extremes clamp: value 0.0 → bin 0, value ≥ 1.0 → bin 255 (and values
/// above 1.0 do not overflow the table).
#[test]
fn image_reduce_histogram_clamps_extremes() {
    let (w, h) = (8u32, 8u32);
    let bytes = constant_bytes(w, h, [0.0, 1.0, 2.0, -1.0]);
    let hist = histogram(&bytes, w, h);
    let n = w * h;
    assert_eq!(hist.bins[0][0], n, "0.0 → bin 0");
    assert_eq!(hist.bins[1][255], n, "1.0 → bin 255");
    assert_eq!(hist.bins[2][255], n, "2.0 clamps to bin 255");
    assert_eq!(hist.bins[3][0], n, "-1.0 clamps to bin 0");
}

/// Two-value tile → exactly TWO spikes on the channel that varies, each
/// holding its share, summing to `w*h`. A left/right split of the tile.
#[test]
fn image_reduce_histogram_two_value_tile_two_spikes() {
    let (w, h) = (16u32, 16u32);
    let lo = 0.25f32; // → bin 64
    let hi = 0.75f32; // → bin 191
                      // R splits left (lo) / right (hi); other channels held constant.
    let tile = RefTile::from_fn(w, h, |x, _| {
        let r = if x < w / 2 { lo } else { hi };
        Px([r, 0.0, 0.0, 1.0])
    });
    let hist = histogram(&tile.f16_bytes(), w, h);

    let half = w / 2 * h;
    assert_eq!(channel_total(&hist, 0), w * h);
    assert_eq!(hist.bins[0][expect_bin(lo)], half, "low spike");
    assert_eq!(hist.bins[0][expect_bin(hi)], half, "high spike");
    let spikes = hist.bins[0].iter().filter(|&&b| b != 0).count();
    assert_eq!(spikes, 2, "exactly two distinct values → two spikes");
}

// ── statistics ──────────────────────────────────────────────────────

/// Constant tile → min == max == mean == the (f16-quantized) value, on
/// every channel.
#[test]
fn image_reduce_statistics_constant_tile_min_eq_max_eq_mean() {
    let (w, h) = (24u32, 24u32);
    let px = [0.5, 0.25, 0.75, 1.0]; // all exact in f16
    let bytes = constant_bytes(w, h, px);
    let s = statistics(&bytes, w, h);
    for (c, &v) in px.iter().enumerate() {
        assert_eq!(s.min[c], v, "min channel {c}");
        assert_eq!(s.max[c], v, "max channel {c}");
        assert_eq!(s.mean[c], v, "mean channel {c}");
    }
}

/// Ramp with a KNOWN mean: a 4×1 R-ramp over exact-in-f16 values
/// {0, 0.25, 0.5, 0.75}; mean = 1.5/4 = 0.375 exactly. min/max are the
/// ramp endpoints.
#[test]
fn image_reduce_statistics_ramp_known_mean() {
    let vals = [0.0f32, 0.25, 0.5, 0.75];
    let w = vals.len() as u32;
    let h = 1u32;
    let tile = RefTile::from_fn(w, h, |x, _| Px([vals[x as usize], 0.0, 0.0, 1.0]));
    let s = statistics(&tile.f16_bytes(), w, h);

    assert_eq!(s.min[0], 0.0, "ramp min");
    assert_eq!(s.max[0], 0.75, "ramp max");
    assert_eq!(s.mean[0], 0.375, "ramp mean = (0+0.25+0.5+0.75)/4");
    // Held channels.
    assert_eq!(s.mean[1], 0.0);
    assert_eq!(s.mean[3], 1.0);
}

/// Two-value tile → min/max are the two values and the mean is the
/// count-weighted average (here a clean 50/50 split → midpoint).
#[test]
fn image_reduce_statistics_two_value_tile() {
    let (w, h) = (16u32, 16u32);
    let lo = 0.25f32;
    let hi = 0.75f32;
    let tile = RefTile::from_fn(w, h, |x, _| {
        let r = if x < w / 2 { lo } else { hi };
        Px([r, 0.0, 0.0, 1.0])
    });
    let s = statistics(&tile.f16_bytes(), w, h);
    assert_eq!(s.min[0], lo);
    assert_eq!(s.max[0], hi);
    assert_eq!(s.mean[0], 0.5, "50/50 split → midpoint 0.5");
}

/// Empty tile (0×0) → all-zero stats and an all-zero histogram (no
/// panic, no division by zero).
#[test]
fn image_reduce_empty_tile_is_zero() {
    let s = statistics(&[], 0, 0);
    assert_eq!(s.min, [0.0; 4]);
    assert_eq!(s.max, [0.0; 4]);
    assert_eq!(s.mean, [0.0; 4]);
    let hist = histogram(&[], 0, 0);
    for c in 0..4 {
        assert_eq!(channel_total(&hist, c), 0);
    }
}
