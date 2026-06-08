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

//! Distance transform (spec §11, a T3 breadth op) — for every background
//! texel, the distance to the nearest foreground texel.
//!
//! This is the representative breadth op for the residency-hardening
//! unit: a whole-tile, *sequential* transform (two raster sweeps that
//! each depend on their predecessor), which is exactly the shape that
//! does NOT fit the per-texel WGSL ABI (the four frozen bind groups,
//! §9.2). So, like the T2 reductions in [`crate::reduce`], it:
//!
//! * is NOT a `KernelDef` — it owns no `registry/kernels.yaml` row and is
//!   never dispatched through the kernel table;
//! * is a state-bearing editor operation
//!   (`image.kernel.distance-transform`) that needs its own state row;
//! * runs on the CPU over the working tile bytes (the M2 correctness
//!   path) and is its own deterministic golden — the arithmetic is in a
//!   fixed sweep order, so it is bit-stable across platforms (§6.3).
//!
//! ## Algorithm — two-pass chamfer (Rosenfeld–Pfaltz)
//!
//! A two-pass chamfer distance transform. Foreground texels seed at
//! distance `0`; background texels seed at `+∞`. A **forward** sweep
//! (rows top→bottom, within a row left→right) relaxes each texel against
//! its already-visited NW/N/NE/W neighbours, and a **backward** sweep
//! (rows bottom→top, right→left) relaxes against SE/S/SW/E. With the
//! local step weights `(ORTHO, DIAG)` this converges in exactly those two
//! passes for a convex 3×3 neighbourhood.
//!
//! Step weights are chosen by [`DistanceMetric`]:
//! * [`DistanceMetric::Chamfer`] — `(1, √2)`. The classic chamfer-(1,√2)
//!   approximation. For a single foreground texel the field is exactly
//!   `max(|dx|,|dy|) + (√2−1)·min(|dx|,|dy|)`, hand-checkable.
//! * [`DistanceMetric::Manhattan`] — `(1, 2)`, the L1 / city-block
//!   transform (diagonals cost two orthogonal steps).
//!
//! True Euclidean is a follow-up (Felzenszwalb–Huttenlocher two-pass);
//! chamfer-(1,√2) is its cheap, deterministic approximation here, ≤ ~8 %
//! over-estimate on 45° diagonals.
//!
//! ## Output
//!
//! `rgba16float` bytes (8 B/texel, little-endian f16, tightly packed
//! rows — the engine's working format). The **distance is in R**; G and B
//! are `0`, A is `1.0`. By default distances are **raw** (in texel units);
//! pass [`DistanceParams::normalize`] to divide by the tile diagonal so
//! the field lands in `[0, 1]` (handy as a mask/falloff). Unreachable
//! texels (an all-background tile) saturate at the f16 max representable
//! distance rather than `+∞`, so the output is always finite.
//!
//! ## M3 GPU follow-up (documented, not built here)
//!
//! The production distance transform is a **jump-flood** (Rong–Tan): seed
//! each foreground texel with its own coordinate, then `log₂(n)` passes
//! propagate the nearest-seed coordinate at halving step sizes, and a
//! final pass writes `‖p − seed(p)‖`. That is `O(n² log n)` parallel work
//! the GPU eats, and it is *exact Euclidean*. It is verified BY TOLERANCE
//! against this CPU chamfer value the usual way (`parity(gpu↔ref)`), with
//! the tolerance absorbing the chamfer-vs-Euclidean gap. Until M3 the CPU
//! value here IS the value.

use half::f16;

/// Bytes per `rgba16float` texel (4 × f16). Mirrors the engine working
/// format; the distance field is written in the same layout.
const BYTES_PER_PIXEL_RGBA: usize = 8;
/// Bytes per `r16float` texel (1 × f16) — the slim single-channel mask
/// input shape.
const BYTES_PER_PIXEL_R: usize = 2;

/// Which input layout the mask tile is in, and which channel carries the
/// foreground signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskChannel {
    /// `r16float` — 2 bytes/texel; the single channel is the mask.
    R16,
    /// `rgba16float` — 8 bytes/texel; the **R** channel is the mask
    /// (alpha-as-foreground callers can pre-swizzle, or use [`Self::A`]).
    RgbaR,
    /// `rgba16float` — 8 bytes/texel; the **A** channel is the mask.
    A,
}

impl MaskChannel {
    fn bytes_per_pixel(self) -> usize {
        match self {
            MaskChannel::R16 => BYTES_PER_PIXEL_R,
            MaskChannel::RgbaR | MaskChannel::A => BYTES_PER_PIXEL_RGBA,
        }
    }

    /// Byte offset, within a texel, of the channel that carries the mask.
    fn channel_offset(self) -> usize {
        match self {
            MaskChannel::R16 | MaskChannel::RgbaR => 0,
            MaskChannel::A => 6, // 4th f16 channel
        }
    }
}

/// The local step weights used by the two-pass sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Chamfer-(1, √2): the Euclidean approximation. Default.
    Chamfer,
    /// Manhattan / L1 (1, 2): diagonals cost two orthogonal steps.
    Manhattan,
}

impl DistanceMetric {
    /// `(orthogonal_step, diagonal_step)`.
    fn weights(self) -> (f32, f32) {
        match self {
            DistanceMetric::Chamfer => (1.0, std::f32::consts::SQRT_2),
            DistanceMetric::Manhattan => (1.0, 2.0),
        }
    }
}

/// Parameters for [`distance_transform`].
#[derive(Debug, Clone, Copy)]
pub struct DistanceParams {
    /// Where the foreground signal lives in the input tile.
    pub channel: MaskChannel,
    /// A texel is foreground when its mask-channel value is `>= threshold`.
    /// Default `0.5` (the spec's `alpha/r > 0.5` rule, taken inclusive at
    /// the half so a clean `1.0` mask is unambiguously foreground).
    pub threshold: f32,
    /// The step weights.
    pub metric: DistanceMetric,
    /// Divide the output distances by the tile diagonal so the field lands
    /// in `[0, 1]`. `false` (default) keeps raw texel-unit distances.
    pub normalize: bool,
}

impl Default for DistanceParams {
    fn default() -> Self {
        DistanceParams {
            channel: MaskChannel::RgbaR,
            threshold: 0.5,
            metric: DistanceMetric::Chamfer,
            normalize: false,
        }
    }
}

impl DistanceParams {
    /// Convenience: raw chamfer distance from the R channel of an
    /// `rgba16float` tile (the common case).
    pub fn rgba_r() -> Self {
        DistanceParams::default()
    }
}

/// The seed distance for an unreached (all-background) field. The largest
/// finite f16 (65504.0); a real tile distance never approaches it, and it
/// keeps the output finite (no `+∞`/NaN through the f16 write).
const FAR: f32 = 65504.0;

/// Decode the mask-channel value of texel `i` to f32.
#[inline]
fn mask_at(bytes: &[u8], i: usize, params: DistanceParams) -> f32 {
    let bpp = params.channel.bytes_per_pixel();
    let off = i * bpp + params.channel.channel_offset();
    let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
    f16::from_bits(bits).to_f32()
}

/// Two-pass chamfer distance transform of a binary-mask tile.
///
/// `mask_bytes` is a `w`×`h` tile in the layout named by `params.channel`
/// (tightly packed rows). Texels with mask value `>= params.threshold` are
/// foreground (distance `0`); every other texel gets the chamfer distance
/// to the nearest foreground texel. Returns `rgba16float` bytes
/// (`w*h*8`) with the distance in **R**, `G=B=0`, `A=1`.
///
/// Deterministic and platform-stable (fixed sweep order, §6.3); it is its
/// own reference. A short input (fewer than `w*h` texels) is treated as
/// all-background for the missing tail (defensive — the engine always
/// passes exactly the full tile).
pub fn distance_transform(mask_bytes: &[u8], w: u32, h: u32, params: DistanceParams) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let n = w * h;
    let bpp = params.channel.bytes_per_pixel();
    let avail = mask_bytes.len() / bpp;
    let (ortho, diag) = params.metric.weights();

    // Seed: foreground 0.0, background FAR.
    let mut d = vec![FAR; n];
    for (i, slot) in d.iter_mut().enumerate() {
        if i < avail && mask_at(mask_bytes, i, params) >= params.threshold {
            *slot = 0.0;
        }
    }

    // Forward sweep: top→bottom, left→right. Relax against the four
    // already-visited neighbours (NW, N, NE, W).
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let mut best = d[i];
            if y > 0 {
                if x > 0 {
                    best = best.min(d[(y - 1) * w + (x - 1)] + diag); // NW
                }
                best = best.min(d[(y - 1) * w + x] + ortho); // N
                if x + 1 < w {
                    best = best.min(d[(y - 1) * w + (x + 1)] + diag); // NE
                }
            }
            if x > 0 {
                best = best.min(d[i - 1] + ortho); // W
            }
            d[i] = best;
        }
    }

    // Backward sweep: bottom→top, right→left. Relax against SE, S, SW, E.
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            let mut best = d[i];
            if y + 1 < h {
                if x + 1 < w {
                    best = best.min(d[(y + 1) * w + (x + 1)] + diag); // SE
                }
                best = best.min(d[(y + 1) * w + x] + ortho); // S
                if x > 0 {
                    best = best.min(d[(y + 1) * w + (x - 1)] + diag); // SW
                }
            }
            if x + 1 < w {
                best = best.min(d[i + 1] + ortho); // E
            }
            d[i] = best;
        }
    }

    // Optional normalize by the tile diagonal so distances land in [0, 1].
    let scale = if params.normalize {
        let diagonal = ((w * w + h * h) as f32).sqrt();
        if diagonal > 0.0 {
            1.0 / diagonal
        } else {
            1.0
        }
    } else {
        1.0
    };

    // Pack to rgba16float: distance in R, G=B=0, A=1.
    let zero = f16::from_f32(0.0).to_bits().to_le_bytes();
    let one = f16::from_f32(1.0).to_bits().to_le_bytes();
    let mut out = Vec::with_capacity(n * BYTES_PER_PIXEL_RGBA);
    for &dist in &d {
        let r = f16::from_f32(dist * scale).to_bits().to_le_bytes();
        out.extend_from_slice(&r); // R
        out.extend_from_slice(&zero); // G
        out.extend_from_slice(&zero); // B
        out.extend_from_slice(&one); // A
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode the R channel (the distance) of texel `i` from an
    /// `rgba16float` output buffer.
    fn dist_r(out: &[u8], i: usize) -> f32 {
        let o = i * BYTES_PER_PIXEL_RGBA;
        f16::from_bits(u16::from_le_bytes([out[o], out[o + 1]])).to_f32()
    }

    /// Build an `r16float` mask tile from a foreground predicate.
    fn r16_mask(w: usize, h: usize, fg: impl Fn(usize, usize) -> bool) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * BYTES_PER_PIXEL_R);
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

    #[test]
    fn single_foreground_pixel_gives_chamfer_field() {
        // 5×5, single foreground texel at (2,2). chamfer-(1,√2) distance
        // is max(|dx|,|dy|) + (√2−1)·min(|dx|,|dy|).
        let (w, h) = (5usize, 5usize);
        let mask = r16_mask(w, h, |x, y| x == 2 && y == 2);
        let out = distance_transform(&mask, w as u32, h as u32, r16_params());

        let s2 = std::f32::consts::SQRT_2;
        let cases = [
            ((2, 2), 0.0),      // the seed
            ((3, 2), 1.0),      // one right
            ((2, 0), 2.0),      // two up, orthogonal
            ((0, 2), 2.0),      // two left
            ((3, 3), s2),       // one diagonal
            ((4, 4), 2.0 * s2), // two diagonals
            ((4, 2), 2.0),      // two right
            ((0, 0), 2.0 * s2), // corner: dx=dy=2
            ((4, 3), 1.0 + s2), // dx=2,dy=1 → 2 + (√2−1)*1
        ];
        for ((x, y), want) in cases {
            let got = dist_r(&out, y * w + x);
            assert!(
                (got - want).abs() < 1e-2,
                "({x},{y}): got {got}, want {want} (chamfer to (2,2))"
            );
        }
    }

    #[test]
    fn manhattan_metric_uses_l1_steps() {
        // Same 5×5 single seed, but L1: diagonal costs 2.
        let (w, h) = (5usize, 5usize);
        let mask = r16_mask(w, h, |x, y| x == 2 && y == 2);
        let params = DistanceParams {
            metric: DistanceMetric::Manhattan,
            ..r16_params()
        };
        let out = distance_transform(&mask, w as u32, h as u32, params);
        // L1 distance |dx| + |dy|.
        for (x, y) in [(3usize, 3usize), (4, 4), (0, 0), (4, 2)] {
            let want = ((x as i32 - 2).abs() + (y as i32 - 2).abs()) as f32;
            let got = dist_r(&out, y * w + x);
            assert!(
                (got - want).abs() < 1e-2,
                "({x},{y}): got {got}, want L1 {want}"
            );
        }
    }

    #[test]
    fn fully_foreground_field_is_all_zero() {
        let (w, h) = (8usize, 8usize);
        let mask = r16_mask(w, h, |_, _| true);
        let out = distance_transform(&mask, w as u32, h as u32, r16_params());
        for i in 0..(w * h) {
            assert_eq!(
                dist_r(&out, i),
                0.0,
                "all-foreground tile is all zeros at {i}"
            );
        }
    }

    #[test]
    fn empty_field_saturates_at_far() {
        let (w, h) = (8usize, 8usize);
        let mask = r16_mask(w, h, |_, _| false);
        let out = distance_transform(&mask, w as u32, h as u32, r16_params());
        for i in 0..(w * h) {
            assert_eq!(
                dist_r(&out, i),
                FAR,
                "all-background tile saturates at FAR (finite, no +inf) at {i}"
            );
        }
    }

    #[test]
    fn output_is_rgba16float_with_distance_in_r() {
        let (w, h) = (4usize, 4usize);
        let mask = r16_mask(w, h, |x, y| x == 0 && y == 0);
        let out = distance_transform(&mask, w as u32, h as u32, r16_params());
        assert_eq!(out.len(), w * h * BYTES_PER_PIXEL_RGBA);
        // G=B=0, A=1 on every texel.
        for i in 0..(w * h) {
            let o = i * BYTES_PER_PIXEL_RGBA;
            let g = f16::from_bits(u16::from_le_bytes([out[o + 2], out[o + 3]])).to_f32();
            let b = f16::from_bits(u16::from_le_bytes([out[o + 4], out[o + 5]])).to_f32();
            let a = f16::from_bits(u16::from_le_bytes([out[o + 6], out[o + 7]])).to_f32();
            assert_eq!((g, b, a), (0.0, 0.0, 1.0), "texel {i} GBA");
        }
    }

    #[test]
    fn normalize_scales_into_unit_range() {
        let (w, h) = (5usize, 5usize);
        // Foreground at one corner; the far corner is the max distance.
        let mask = r16_mask(w, h, |x, y| x == 0 && y == 0);
        let params = DistanceParams {
            normalize: true,
            ..r16_params()
        };
        let out = distance_transform(&mask, w as u32, h as u32, params);
        for i in 0..(w * h) {
            let v = dist_r(&out, i);
            assert!(
                (0.0..=1.0001).contains(&v),
                "normalized distance {v} in [0,1]"
            );
        }
        // Corner (0,0) is the seed → 0; (4,4) is the largest, < 1.
        assert_eq!(dist_r(&out, 0), 0.0);
        assert!(dist_r(&out, 4 * w + 4) > 0.0);
    }

    #[test]
    fn alpha_channel_mask_selects_foreground() {
        // rgba16float tile: R=0 everywhere, A=1 only at (1,1). Using the A
        // mask must seed (1,1), not read R.
        let (w, h) = (3usize, 3usize);
        let mut v = Vec::with_capacity(w * h * BYTES_PER_PIXEL_RGBA);
        for y in 0..h {
            for x in 0..w {
                let a = if x == 1 && y == 1 { 1.0f32 } else { 0.0f32 };
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
        assert_eq!(dist_r(&out, w + 1), 0.0, "A-foreground seeds at (1,1)");
        assert_eq!(dist_r(&out, w), 1.0, "neighbour (1,0)→(1,1) at distance 1");
    }
}
