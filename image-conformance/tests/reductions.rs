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
use image_gpu::reduce::{
    auto_enhance, histogram, histogram_rgba8, statistics, AutoEnhance, Histogram,
};

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

// ── histogram_rgba8 (the panel-facing RGB + luma readout) ────────────
//
// feat: image.reduce.statistics — the LEVELS/CURVES panel histogram over
// the engine-held straight RGBA8 buffer (one bin per code value; luma is
// Rec.601). Pure fixed-order arithmetic → its own golden.

/// Build a tightly packed straight-RGBA8 buffer from a per-pixel fn.
fn rgba8_from_fn(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            v.extend_from_slice(&f(x, y));
        }
    }
    v
}

/// A constant RGBA8 tile spikes exactly one bin per channel (the byte IS
/// the bin) and one luma bin; every total equals the pixel count.
#[test]
fn image_reduce_histogram_rgba8_constant_single_bin() {
    let (w, h) = (16u32, 8u32);
    let n = w * h;
    // luma = round(0.299*40 + 0.587*80 + 0.114*120) = round(72.6) = 73.
    let px = [40u8, 80, 120, 255];
    let hist = histogram_rgba8(&rgba8_from_fn(w, h, |_, _| px));

    assert_eq!(hist.r[40], n, "R spikes its own code value");
    assert_eq!(hist.g[80], n, "G spikes its own code value");
    assert_eq!(hist.b[120], n, "B spikes its own code value");
    assert_eq!(hist.luma[73], n, "Rec.601 luma of (40,80,120) is 73");
    for (label, ch) in [
        ("r", &hist.r),
        ("g", &hist.g),
        ("b", &hist.b),
        ("luma", &hist.luma),
    ] {
        assert_eq!(ch.iter().sum::<u32>(), n, "{label} total == pixel count");
        assert_eq!(
            ch.iter().filter(|&&c| c != 0).count(),
            1,
            "{label} one spike"
        );
    }
}

/// Pure-channel extremes: black → bin 0 on every channel + luma; white →
/// bin 255 on every channel + luma. A 50/50 black/white split.
#[test]
fn image_reduce_histogram_rgba8_black_white_extremes() {
    let (w, h) = (8u32, 8u32);
    let hist = histogram_rgba8(&rgba8_from_fn(w, h, |x, _| {
        if x < w / 2 {
            [0, 0, 0, 255]
        } else {
            [255, 255, 255, 255]
        }
    }));
    let half = w / 2 * h;
    for ch in [&hist.r, &hist.g, &hist.b, &hist.luma] {
        assert_eq!(ch[0], half, "black half → bin 0");
        assert_eq!(ch[255], half, "white half → bin 255");
        assert_eq!(ch.iter().sum::<u32>(), w * h);
    }
}

/// Luma of a pure primary uses ONLY that primary's weight: a full-red
/// pixel has luma round(0.299*255) = 76; full-green round(0.587*255) =
/// 150; full-blue round(0.114*255) = 29.
#[test]
fn image_reduce_histogram_rgba8_luma_primary_weights() {
    let red = histogram_rgba8(&[255, 0, 0, 255]);
    let green = histogram_rgba8(&[0, 255, 0, 255]);
    let blue = histogram_rgba8(&[0, 0, 255, 255]);
    assert_eq!(red.luma[76], 1, "0.299*255 ≈ 76");
    assert_eq!(green.luma[150], 1, "0.587*255 ≈ 150");
    assert_eq!(blue.luma[29], 1, "0.114*255 ≈ 29");
}

/// An empty / sub-pixel buffer yields an all-zero histogram (a trailing
/// partial pixel is ignored — never a panic).
#[test]
fn image_reduce_histogram_rgba8_empty_is_zero() {
    let empty = histogram_rgba8(&[]);
    let partial = histogram_rgba8(&[1, 2, 3]); // 3 bytes — no whole pixel
    for hist in [&empty, &partial] {
        let flat = hist.to_flat();
        assert_eq!(flat.iter().sum::<u32>(), 0, "no counts");
    }
}

/// The flat layout is `[r…, g…, b…, luma…]` — the wasm-boundary order the
/// panel slices back.
#[test]
fn image_reduce_histogram_rgba8_flat_layout() {
    // One pixel: R=1, G=2, B=3, luma=round(0.299+1.174+0.342)=round(1.815)=2.
    let flat = histogram_rgba8(&[1, 2, 3, 255]).to_flat();
    assert_eq!(flat.len(), 1024);
    assert_eq!(flat[1], 1, "R bin 1");
    assert_eq!(flat[256 + 2], 1, "G bin 2");
    assert_eq!(flat[512 + 3], 1, "B bin 3");
    assert_eq!(flat[768 + 2], 1, "luma bin 2");
}

// ── auto-enhance (orchestration over the panel histogram) ────────────
//
// feat: image.adjust.auto-enhance — the single "auto" adjustment that
// composes the EXISTING levels + white-balance kernels from the RGB+luma
// histogram. Pure readout/orchestration math (no kernels.yaml row, spec
// §6); these tests are its golden.

/// Auto-levels on a LOW-CONTRAST image EXPANDS the input range: a luma
/// span confined to roughly [64, 192] yields `in_black > 0` near 64/255
/// and `in_white < 1` near 192/255, so the levels stretch recovers
/// contrast. (The 0.5%/99.5% clip lands on the populated luma extremes.)
#[test]
fn image_adjust_auto_enhance_levels_expand_low_contrast() {
    // Neutral grays (r==g==b → no WB) spread evenly over [64, 192], so the
    // luma equals the gray value and the populated range is exactly that.
    let (w, h) = (32u32, 32u32);
    let hist = histogram_rgba8(&rgba8_from_fn(w, h, |x, _| {
        let v = 64 + ((x * 128) / w) as u8; // 64..=191 across the width
        [v, v, v, 255]
    }));
    let auto = auto_enhance(&hist);

    assert!(
        auto.in_black > 0.0 && auto.in_black <= 80.0 / 255.0,
        "auto black point lifts off 0 onto the populated low end (got {})",
        auto.in_black
    );
    assert!(
        auto.in_white < 1.0 && auto.in_white >= 180.0 / 255.0,
        "auto white point drops below 1 onto the populated high end (got {})",
        auto.in_white
    );
    assert!(
        auto.in_white - auto.in_black < 0.95,
        "the recovered range is narrower than full (contrast to gain)"
    );
    // Neutral grays → no white-balance correction.
    assert!(
        auto.temp.abs() < 1e-3 && auto.tint.abs() < 1e-3,
        "gray input needs no WB (temp {}, tint {})",
        auto.temp,
        auto.tint
    );
}

/// Gray-world WB NEUTRALIZES a tinted image: a warm cast (R high, B low)
/// yields a COOLING correction — `temp < 0` (pull amber→blue) — and a
/// green cast yields `tint < 0` (pull green→magenta). The sign convention
/// matches the kernel gains `gr = 1+temp, gb = 1−temp`: an over-red image
/// must scale R down (temp negative) and B up.
#[test]
fn image_adjust_auto_enhance_grayworld_neutralizes_warm_cast() {
    // Over-red / under-blue (a warm cast); green mid.
    let warm = histogram_rgba8(&rgba8_from_fn(16, 16, |_, _| [200, 128, 64, 255]));
    let auto = auto_enhance(&warm);
    assert!(
        auto.temp < -1e-2,
        "warm (R>B) cast → cooling temp < 0 (got {})",
        auto.temp
    );

    // A green cast (G well above R≈B) → tint pulls toward magenta (< 0).
    let green = histogram_rgba8(&rgba8_from_fn(16, 16, |_, _| [110, 200, 110, 255]));
    let auto_g = auto_enhance(&green);
    assert!(
        auto_g.tint < -1e-2,
        "green cast → tint < 0 (got {})",
        auto_g.tint
    );
    // R≈B in the green-cast image → near-zero temp.
    assert!(
        auto_g.temp.abs() < 1e-2,
        "balanced R/B → temp ≈ 0 (got {})",
        auto_g.temp
    );
}

/// Applying the gray-world gains to the tinted means brings the three
/// channel means into agreement (the WB fit actually neutralizes). We
/// reconstruct the kernel gains `(1+temp, 1+tint, 1−temp)` from the
/// estimate and check the corrected means converge toward each other far
/// better than the originals.
#[test]
fn image_adjust_auto_enhance_grayworld_corrected_means_converge() {
    let rgb = [180.0f32, 120.0, 70.0];
    let hist = histogram_rgba8(&rgba8_from_fn(16, 16, |_, _| {
        [rgb[0] as u8, rgb[1] as u8, rgb[2] as u8, 255]
    }));
    let auto = auto_enhance(&hist);
    let gains = [1.0 + auto.temp, 1.0 + auto.tint, 1.0 - auto.temp];
    let corrected = [rgb[0] * gains[0], rgb[1] * gains[1], rgb[2] * gains[2]];

    let spread = |m: [f32; 3]| {
        let max = m[0].max(m[1]).max(m[2]);
        let min = m[0].min(m[1]).min(m[2]);
        max - min
    };
    assert!(
        spread(corrected) < spread(rgb) * 0.6,
        "WB-corrected channel means converge (before {:.1}, after {:.1})",
        spread(rgb),
        spread(corrected)
    );
}

/// A FLAT (single-value) image has no contrast to recover and is already
/// neutral → the estimate is the identity no-op `[0, 1, 0, 0]`.
#[test]
fn image_adjust_auto_enhance_flat_image_is_identity() {
    let flat = histogram_rgba8(&rgba8_from_fn(8, 8, |_, _| [128, 128, 128, 255]));
    let auto = auto_enhance(&flat);
    assert_eq!(auto, AutoEnhance::default(), "flat neutral → identity");
    assert!(auto.is_identity());
}

/// An already-neutral, full-range image (gray ramp 0..=255) needs no WB
/// and ~no levels lift (black/white already at the extremes within the
/// clip). The estimate stays at/near identity — auto-enhance never
/// invents a correction.
#[test]
fn image_adjust_auto_enhance_full_range_neutral_no_op() {
    let ramp = histogram_rgba8(&rgba8_from_fn(256, 1, |x, _| {
        let v = x as u8;
        [v, v, v, 255]
    }));
    let auto = auto_enhance(&ramp);
    assert!(
        auto.temp.abs() < 1e-3 && auto.tint.abs() < 1e-3,
        "neutral → no WB"
    );
    // 0.5% of 256 px = ~2 px clipped each end → black ≈ 1/255, white ≈ 254/255.
    assert!(
        auto.in_black <= 3.0 / 255.0,
        "black stays near 0 (got {})",
        auto.in_black
    );
    assert!(
        auto.in_white >= 252.0 / 255.0,
        "white stays near 1 (got {})",
        auto.in_white
    );
}

/// Degenerate inputs are clean no-ops: an empty buffer and a near-black
/// image (means under the 1-code-value WB floor) both yield identity —
/// never a divide-by-zero or a runaway gain.
#[test]
fn image_adjust_auto_enhance_degenerate_is_identity() {
    assert_eq!(auto_enhance(&histogram_rgba8(&[])), AutoEnhance::default());
    let near_black = histogram_rgba8(&rgba8_from_fn(8, 8, |_, _| [0, 0, 0, 255]));
    let auto = auto_enhance(&near_black);
    assert!(
        auto.temp == 0.0 && auto.tint == 0.0,
        "near-black means below floor → no WB gain blow-up"
    );
}
