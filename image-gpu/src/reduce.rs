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

//! T2 reductions (spec §11, `KernelClass::Reduction`): histogram +
//! statistics. A reduction collapses a tile/region to a table or to
//! scalars — `out` is NOT one texel per input texel, so reductions do
//! NOT fit the per-texel WGSL ABI (the four frozen bind groups, §9.2)
//! and are NOT `KernelDef`s: they own no registry kernel row and are
//! never dispatched through the kernel table. They are state-bearing
//! editor operations (`image.reduce.histogram` /
//! `image.reduce.statistics`) that need their own state-registry rows.
//!
//! M2 CUT — CPU over the working tile bytes. The inputs and outputs of
//! this engine are rgba16float (4 × little-endian f16, 8 bytes/texel,
//! tightly packed rows; see [`crate::execute`]); here we decode them on
//! the CPU with the `half` crate and reduce in `f32`/`f64`. This is the
//! correctness path and the deterministic reference value (M2): being
//! pure scalar arithmetic in a fixed order, it is bit-stable across
//! platforms by construction (§6.3), so it doubles as its own golden.
//!
//! M3 PERF PATH (documented, not built here): the production reduction
//! is a WGSL compute shader doing an atomic/segmented two-level reduce
//! (per-workgroup partials in shared memory → a global atomic merge for
//! statistics; per-bin `atomicAdd` for the histogram), reading the same
//! rgba16float storage texture. That shader is verified BY TOLERANCE
//! against this CPU value the same way kernels are (`parity(gpu↔ref)`),
//! except the reduction order on the GPU is not the fixed CPU order, so
//! the declared tolerance must absorb f16/f32 summation reassociation.
//! Until M3 the CPU value here IS the value.

use half::f16;

/// Bytes per rgba16float texel (4 × f16). Mirrors `execute`'s working
/// format; reductions read the very same tile bytes.
const BYTES_PER_PIXEL: usize = 8;

/// A per-channel 256-bin histogram over an rgba16float tile/region.
///
/// `bins[c][k]` counts texels whose channel `c` (0=R,1=G,2=B,3=A)
/// quantizes to bin `k`. Quantization: `round(v * 255)` clamped to
/// `[0, 255]` where `v` is the f16 channel value widened to f32 (so
/// `0.0 → 0`, `1.0 → 255`, `0.5 → 128`). The bin counts always sum to
/// `w * h` per channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Histogram {
    pub bins: [[u32; 256]; 4],
}

impl Default for Histogram {
    fn default() -> Self {
        Histogram {
            bins: [[0u32; 256]; 4],
        }
    }
}

/// The RGB + luma 256-bin histogram an editor LEVELS/CURVES panel reads.
///
/// `r`/`g`/`b` count the straight-8-bit channel values directly (one bin
/// per code value — no quantization round-trip, the byte IS the bin);
/// `luma` is the Rec.601 luma `round(0.299·r + 0.587·g + 0.114·b)` per
/// pixel. Alpha is intentionally absent (a levels/curves panel never
/// plots it). Each of the four totals equals the pixel count. This is the
/// panel-facing readout (the RGBA8 working buffer the wasm surface holds);
/// the f16 working-tile [`histogram`] above is the in-pipeline reduction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaLumaHistogram {
    pub r: [u32; 256],
    pub g: [u32; 256],
    pub b: [u32; 256],
    pub luma: [u32; 256],
}

impl Default for RgbaLumaHistogram {
    fn default() -> Self {
        RgbaLumaHistogram {
            r: [0u32; 256],
            g: [0u32; 256],
            b: [0u32; 256],
            luma: [0u32; 256],
        }
    }
}

impl RgbaLumaHistogram {
    /// Flatten to the `[r…, g…, b…, luma…]` 1024-`u32` row the wasm
    /// surface hands JS (the panel slices it back into four 256-bin
    /// channels). Fixed channel order — never reassociated.
    pub fn to_flat(&self) -> [u32; 1024] {
        let mut out = [0u32; 1024];
        out[0..256].copy_from_slice(&self.r);
        out[256..512].copy_from_slice(&self.g);
        out[512..768].copy_from_slice(&self.b);
        out[768..1024].copy_from_slice(&self.luma);
        out
    }
}

/// Rec.601 luma bin for a straight-8-bit pixel: `round(0.299·r +
/// 0.587·g + 0.114·b)` clamped to `[0, 255]`. Fixed coefficient order,
/// round-half-away-from-zero — bit-stable by construction (§6.3). The
/// weights are the ITU-R BT.601 luma weights (standard literature; no
/// reference reading).
#[inline]
fn luma_bin(r: u8, g: u8, b: u8) -> usize {
    let y = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
    let yr = y.round();
    if yr <= 0.0 {
        0
    } else if yr >= 255.0 {
        255
    } else {
        yr as usize
    }
}

/// Build the RGB + luma 256-bin histogram over a tightly packed straight
/// RGBA8 buffer (`pixels.len()` must be a multiple of 4; a trailing
/// partial pixel is ignored). The panel-facing reduction the LEVELS /
/// CURVES editor renders. Pure fixed-order scalar arithmetic — bit-stable
/// across platforms, its own golden (§6.3). Alpha is not binned.
pub fn histogram_rgba8(pixels: &[u8]) -> RgbaLumaHistogram {
    let mut hist = RgbaLumaHistogram::default();
    for px in pixels.chunks_exact(4) {
        let (r, g, b) = (px[0], px[1], px[2]);
        hist.r[r as usize] += 1;
        hist.g[g as usize] += 1;
        hist.b[b as usize] += 1;
        hist.luma[luma_bin(r, g, b)] += 1;
    }
    hist
}

/// Per-channel min / max / mean over an rgba16float tile/region, in f32.
///
/// `min`/`max` are exact (the smallest/largest f16-decoded channel
/// value, finite). `mean` is the sum of f16-decoded channel values in a
/// fixed row-major order accumulated in `f64`, divided by the texel
/// count, then narrowed to `f32` — fixed reduction order makes it
/// bit-stable (§6.3). An empty tile yields all-zero stats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub min: [f32; 4],
    pub max: [f32; 4],
    pub mean: [f32; 4],
}

impl Default for Stats {
    fn default() -> Self {
        Stats {
            min: [0.0; 4],
            max: [0.0; 4],
            mean: [0.0; 4],
        }
    }
}

/// Decode one rgba16float texel (4 × little-endian f16) at byte offset
/// `o` into straight f32. Caller guarantees `o + 8 <= bytes.len()`.
#[inline]
fn decode_texel(bytes: &[u8], o: usize) -> [f32; 4] {
    let mut px = [0.0f32; 4];
    for (c, slot) in px.iter_mut().enumerate() {
        let lo = bytes[o + c * 2];
        let hi = bytes[o + c * 2 + 1];
        *slot = f16::from_bits(u16::from_le_bytes([lo, hi])).to_f32();
    }
    px
}

/// Quantize a channel value to a `[0, 255]` histogram bin: `round(v *
/// 255)` clamped. `round` is round-half-away-from-zero (`f32::round`);
/// for the non-negative working range the half cases round up
/// (`0.5·(1/255) … 254.5·(1/255)` → the upper bin).
#[inline]
fn quantize_bin(v: f32) -> usize {
    let scaled = (v * 255.0).round();
    if scaled <= 0.0 {
        0
    } else if scaled >= 255.0 {
        255
    } else {
        scaled as usize
    }
}

/// Build the per-channel 256-bin histogram over a `w`×`h` rgba16float
/// tile (tightly packed rows). Texels beyond `w * h` (if `tile_bytes`
/// is longer) are ignored; a short buffer simply contributes fewer
/// texels (defensive — the engine always passes exactly `w*h*8`).
pub fn histogram(tile_bytes: &[u8], w: u32, h: u32) -> Histogram {
    let mut hist = Histogram::default();
    let count = (w as usize).saturating_mul(h as usize);
    let avail = tile_bytes.len() / BYTES_PER_PIXEL;
    let texels = count.min(avail);
    for i in 0..texels {
        let px = decode_texel(tile_bytes, i * BYTES_PER_PIXEL);
        for (c, &v) in px.iter().enumerate() {
            hist.bins[c][quantize_bin(v)] += 1;
        }
    }
    hist
}

/// Compute per-channel min / max / mean over a `w`×`h` rgba16float tile
/// (tightly packed rows). Reduction order is fixed row-major; the mean
/// accumulates in `f64` then narrows to `f32` for determinism (§6.3).
pub fn statistics(tile_bytes: &[u8], w: u32, h: u32) -> Stats {
    let count = (w as usize).saturating_mul(h as usize);
    let avail = tile_bytes.len() / BYTES_PER_PIXEL;
    let texels = count.min(avail);
    if texels == 0 {
        return Stats::default();
    }

    let mut min = [f32::INFINITY; 4];
    let mut max = [f32::NEG_INFINITY; 4];
    let mut sum = [0.0f64; 4];

    for i in 0..texels {
        let px = decode_texel(tile_bytes, i * BYTES_PER_PIXEL);
        for c in 0..4 {
            let v = px[c];
            if v < min[c] {
                min[c] = v;
            }
            if v > max[c] {
                max[c] = v;
            }
            sum[c] += v as f64;
        }
    }

    let n = texels as f64;
    let mut mean = [0.0f32; 4];
    for c in 0..4 {
        mean[c] = (sum[c] / n) as f32;
    }
    Stats { min, max, mean }
}

// ── auto-enhance (orchestration over the panel histogram) ────────────
//
// A single "auto" adjustment that composes the EXISTING adjustment
// kernels (levels + white-balance) from the panel-facing RGB+luma
// histogram. This is READOUT/ORCHESTRATION math (spec §6), not a kernel:
// like the histogram it consumes, it owns NO `registry/kernels.yaml`
// dispatch row and is never executed through the WGSL ABI. It derives
// PARAMETERS; the pixels are still moved only by the levels and
// white-balance KernelDefs that already carry their own rows. Pure,
// fixed-order scalar arithmetic over the histogram counts — deterministic
// and bit-stable, its own golden (§6.3). The wasm surface maps the result
// onto `AdjustParams { levels, temp, tint }`; nothing here touches the GPU.

/// The auto-enhance estimate: an auto-levels input range (the percentile-
/// clipped black/white points, normalized to `[0, 1]`) plus a gray-world
/// white-balance estimate fitted to the `temp`/`tint` axes of the
/// `adjust.white_balance` kernel (`gr = 1+temp, gg = 1+tint, gb = 1−temp`).
///
/// `in_black`/`in_white` feed the levels kernel's input range (output
/// range and gamma stay identity — auto-enhance stretches contrast, it
/// does not tone-curve). When the image is flat or degenerate the estimate
/// collapses to identity (`in_black = 0`, `in_white = 1`, `temp = tint = 0`)
/// so applying it is a guaranteed no-op rather than a wrong-looking image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoEnhance {
    /// Auto black point in `[0, 1]` (the levels input-black).
    pub in_black: f32,
    /// Auto white point in `[0, 1]` (the levels input-white); always `> in_black`.
    pub in_white: f32,
    /// Gray-world `temp` (amber↔blue) for `adjust.white_balance`.
    pub temp: f32,
    /// Gray-world `tint` (green↔magenta) for `adjust.white_balance`.
    pub tint: f32,
}

impl Default for AutoEnhance {
    fn default() -> Self {
        AutoEnhance {
            in_black: 0.0,
            in_white: 1.0,
            temp: 0.0,
            tint: 0.0,
        }
    }
}

impl AutoEnhance {
    /// Does this estimate change anything (vs. the identity no-op)?
    pub fn is_identity(&self) -> bool {
        *self == AutoEnhance::default()
    }
}

/// Find the histogram bin (`0..=255`) at the `fraction` cumulative
/// position from the LOW end (`fraction` in `[0, 1]`). Walks the 256 bins
/// accumulating counts until the running total first reaches
/// `ceil(fraction * total)`, returning that bin index. `total == 0`
/// (empty image) yields 0. Fixed forward scan — deterministic.
#[inline]
fn percentile_bin_low(bins: &[u32; 256], total: u64, fraction: f32) -> u8 {
    if total == 0 {
        return 0;
    }
    // round-half-up target rank in [1, total]; clamp the fraction defensively.
    let f = fraction.clamp(0.0, 1.0) as f64;
    let target = ((f * total as f64).ceil() as u64).max(1).min(total);
    let mut acc: u64 = 0;
    for (k, &c) in bins.iter().enumerate() {
        acc += c as u64;
        if acc >= target {
            return k as u8;
        }
    }
    255
}

/// The default low/high clip fractions for auto-levels: clip the darkest
/// 0.5% and brightest 0.5% of luma (the classic "auto contrast" 0.5/99.5
/// percentile stretch). Public so the call site / tests name the policy.
pub const AUTO_LEVELS_CLIP_LOW: f32 = 0.005;
pub const AUTO_LEVELS_CLIP_HIGH: f32 = 0.995;

/// Mean of a 256-bin channel histogram in `[0, 255]` code-value space:
/// `Σ k·count / Σ count`. `total == 0` yields 0.0.
#[inline]
fn channel_mean(bins: &[u32; 256], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let mut sum: u64 = 0;
    for (k, &c) in bins.iter().enumerate() {
        sum += k as u64 * c as u64;
    }
    (sum as f64 / total as f64) as f32
}

/// Derive an [`AutoEnhance`] estimate from a panel RGB+luma histogram.
///
/// **Auto-levels** (contrast): the black/white input points are the
/// [`AUTO_LEVELS_CLIP_LOW`]/[`AUTO_LEVELS_CLIP_HIGH`] percentiles of the
/// LUMA histogram, normalized `bin/255`. If the clipped range is empty or
/// inverted (a flat / single-value image) the levels collapse to identity
/// (`0 → 1`) — never a divide-by-near-zero blow-up.
///
/// **Gray-world white balance**: with per-channel means `R̄, Ḡ, B̄` and the
/// neutral gray `(R̄+Ḡ+B̄)/3`, the ideal gains are `gc = gray/c̄`. Fitted to
/// the kernel's `(1+temp, 1+tint, 1−temp)`: `tint = gg − 1` (green → tint),
/// and `temp = (gr − gb)/2` (the amber↔blue axis splits red/blue). A
/// degenerate channel mean (≈0) or an already-neutral image yields
/// `temp = tint = 0`. Pure fixed-order arithmetic — deterministic (§6.3).
pub fn auto_enhance(hist: &RgbaLumaHistogram) -> AutoEnhance {
    // Pixel count (each channel histogram sums to the same total).
    let total: u64 = hist.luma.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return AutoEnhance::default();
    }

    // ── auto-levels: percentile-clipped luma black/white points ──
    let lo = percentile_bin_low(&hist.luma, total, AUTO_LEVELS_CLIP_LOW);
    let hi = percentile_bin_low(&hist.luma, total, AUTO_LEVELS_CLIP_HIGH);
    let (in_black, in_white) = if hi > lo {
        (lo as f32 / 255.0, hi as f32 / 255.0)
    } else {
        // flat / single-value image — no contrast to recover.
        (0.0, 1.0)
    };

    // ── gray-world white balance fitted to (temp, tint) ──
    let rbar = channel_mean(&hist.r, total);
    let gbar = channel_mean(&hist.g, total);
    let bbar = channel_mean(&hist.b, total);
    let gray = (rbar + gbar + bbar) / 3.0;
    // Guard degenerate (near-black or a zero channel) images: any mean at
    // or below a 1-code-value floor makes the gain unstable → skip WB.
    let floor = 1.0f32;
    let (temp, tint) = if gray <= floor || rbar <= floor || gbar <= floor || bbar <= floor {
        (0.0, 0.0)
    } else {
        let gr = gray / rbar;
        let gg = gray / gbar;
        let gb = gray / bbar;
        ((gr - gb) / 2.0, gg - 1.0)
    };

    AutoEnhance {
        in_black,
        in_white,
        temp,
        tint,
    }
}
