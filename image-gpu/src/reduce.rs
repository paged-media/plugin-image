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
