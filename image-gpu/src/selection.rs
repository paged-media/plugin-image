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

//! Selection-mask plumbing (spec §6.1 "selection-ready from day one",
//! §15 M3) — the typed surface the editor's selections will build.
//!
//! The kernel ABI applies `out = mix(a, result, mask)` at every
//! pointwise dispatch (`image_kernels::abi`), and the mask binds at
//! `@group(2)` as an `r16float` texture. The execution contract is
//! [`crate::execute_tile_once`]'s `mask: Option<&[u8]>` parameter:
//! `None` is the constant-1 mask (Engine A's default — the whole tile
//! is selected); `Some(bytes)` is a `w·h` grid of `r16float` texels
//! (2 bytes each, tightly packed rows, row-major) where each texel is
//! the per-pixel selection weight in `[0, 1]`.
//!
//! [`SelectionMask`] is the builder that produces exactly those bytes:
//! the editor models a selection as a coverage field and lowers it
//! here. A weight of `1.0` means "kernel result fully applies"; `0.0`
//! means "leave the backdrop `a` untouched"; intermediate weights blend
//! (anti-aliased edges, feathered selections). Building the mask in one
//! place keeps the f16 quantization (and the texel byte order) identical
//! to what the GPU upload in `run_common` consumes.

use half::f16;

/// A selection mask tile: per-texel weights in `[0, 1]`, stored as the
/// `r16float` texel bytes [`crate::execute_tile_once`] consumes for its
/// `mask` argument (group 2, ABI v1 §9.2). `width·height` texels, 2
/// bytes each, tightly packed rows, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionMask {
    width: u32,
    height: u32,
    /// `width·height·2` bytes: each texel is a little-endian f16.
    bytes: Vec<u8>,
}

impl SelectionMask {
    /// A fully-selected `w`×`h` mask (every texel `1.0`). Equivalent to
    /// passing `None` to [`crate::execute_tile_once`], but materialized —
    /// useful when a caller wants an explicit mask to mutate.
    pub fn constant_one(w: u32, h: u32) -> Self {
        Self::from_fn(w, h, |_, _| 1.0)
    }

    /// A fully-deselected `w`×`h` mask (every texel `0.0`). The kernel
    /// is a no-op everywhere: output == backdrop `a`.
    pub fn constant_zero(w: u32, h: u32) -> Self {
        Self::from_fn(w, h, |_, _| 0.0)
    }

    /// Build a mask from a per-texel weight function. Weights are clamped
    /// to `[0, 1]` (a selection coverage is a probability, never out of
    /// range) and quantized to f16 — the same final-step quantization the
    /// GPU upload performs, so the typed mask and the bytes the kernel
    /// sees agree exactly.
    pub fn from_fn(w: u32, h: u32, f: impl Fn(u32, u32) -> f32) -> Self {
        let mut bytes = Vec::with_capacity((w * h * 2) as usize);
        for y in 0..h {
            for x in 0..w {
                let weight = f(x, y).clamp(0.0, 1.0);
                bytes.extend_from_slice(&f16::from_f32(weight).to_bits().to_le_bytes());
            }
        }
        Self {
            width: w,
            height: h,
            bytes,
        }
    }

    /// A hard-edged rectangular selection: weight `1.0` inside the
    /// half-open rect `[x0, x0+rw) × [y0, y0+rh)`, `0.0` outside. The
    /// archetypal marquee selection. Out-of-bounds extents are clipped
    /// by the per-texel test, so an oversized rect simply selects the
    /// whole tile.
    pub fn from_rect(w: u32, h: u32, x0: u32, y0: u32, rw: u32, rh: u32) -> Self {
        let x1 = x0.saturating_add(rw);
        let y1 = y0.saturating_add(rh);
        Self::from_fn(w, h, |x, y| {
            if x >= x0 && x < x1 && y >= y0 && y < y1 {
                1.0
            } else {
                0.0
            }
        })
    }

    /// Wrap caller-supplied `r16float` texel bytes as a mask, validating
    /// the length. Use this when the selection coverage already exists as
    /// an f16 texture (e.g. produced GPU-side). `bytes` must be exactly
    /// `w·h·2` long.
    pub fn from_bytes(w: u32, h: u32, bytes: Vec<u8>) -> Option<Self> {
        if bytes.len() != (w as usize) * (h as usize) * 2 {
            return None;
        }
        Some(Self {
            width: w,
            height: h,
            bytes,
        })
    }

    /// The mask width in texels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The mask height in texels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The r16float texel bytes — pass to [`crate::execute_tile_once`]'s
    /// `mask` argument as `Some(mask.bytes())`.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the mask into its raw byte buffer.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// The f16-quantized weight at `(x, y)` as f32 — the value the kernel
    /// actually mixes with (after the same f16 round-trip the GPU sees).
    /// Panics if out of bounds.
    pub fn weight_at(&self, x: u32, y: u32) -> f32 {
        let i = ((y * self.width + x) * 2) as usize;
        let bits = u16::from_le_bytes([self.bytes[i], self.bytes[i + 1]]);
        f16::from_bits(bits).to_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The r16float bit pattern for `1.0` (fully selected) — the byte
    /// `run_common` writes for the implicit constant-1 mask.
    const ONE_F16_BITS: u16 = 0x3C00;

    #[test]
    fn constant_one_is_all_ones() {
        let m = SelectionMask::constant_one(4, 3);
        assert_eq!(m.width(), 4);
        assert_eq!(m.height(), 3);
        assert_eq!(m.bytes().len(), 4 * 3 * 2);
        // Every texel is the f16 bit pattern for 1.0.
        for chunk in m.bytes().chunks_exact(2) {
            assert_eq!(u16::from_le_bytes([chunk[0], chunk[1]]), ONE_F16_BITS);
        }
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(m.weight_at(x, y), 1.0);
            }
        }
    }

    #[test]
    fn constant_one_matches_execute_default_bytes() {
        // run_common builds None's constant-1 mask as 0x3C00 repeated;
        // constant_one must produce the identical byte stream so the
        // typed surface and the implicit default agree.
        let m = SelectionMask::constant_one(7, 5);
        let one = ONE_F16_BITS.to_le_bytes();
        let expected: Vec<u8> = one
            .iter()
            .copied()
            .cycle()
            .take((7 * 5 * 2) as usize)
            .collect();
        assert_eq!(m.bytes(), expected.as_slice());
    }

    #[test]
    fn constant_zero_is_all_zeroes() {
        let m = SelectionMask::constant_zero(3, 3);
        for chunk in m.bytes().chunks_exact(2) {
            assert_eq!(u16::from_le_bytes([chunk[0], chunk[1]]), 0u16);
        }
        assert_eq!(m.weight_at(1, 1), 0.0);
    }

    #[test]
    fn from_rect_selects_inside_only() {
        // 6x6 tile, a 2x2 rect at (1,1): texels (1,1),(2,1),(1,2),(2,2).
        let m = SelectionMask::from_rect(6, 6, 1, 1, 2, 2);
        for y in 0..6 {
            for x in 0..6 {
                let inside = (1..3).contains(&x) && (1..3).contains(&y);
                let want = if inside { 1.0 } else { 0.0 };
                assert_eq!(m.weight_at(x, y), want, "texel ({x},{y})");
            }
        }
    }

    #[test]
    fn from_rect_oversized_clips_to_tile() {
        // A rect larger than the tile selects everything (per-texel test
        // clips), never panics or over-runs the buffer.
        let m = SelectionMask::from_rect(4, 4, 0, 0, 100, 100);
        assert_eq!(m.bytes().len(), 4 * 4 * 2);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(m.weight_at(x, y), 1.0);
            }
        }
    }

    #[test]
    fn from_fn_left_half_one_right_half_zero() {
        // The canonical half-and-half mask: weight 1.0 on the left half,
        // 0.0 on the right half.
        let (w, h) = (8u32, 4u32);
        let m = SelectionMask::from_fn(w, h, |x, _| if x < w / 2 { 1.0 } else { 0.0 });
        for y in 0..h {
            for x in 0..w {
                let want = if x < w / 2 { 1.0 } else { 0.0 };
                assert_eq!(m.weight_at(x, y), want);
            }
        }
    }

    #[test]
    fn from_fn_clamps_out_of_range() {
        // Weights outside [0,1] are clamped — a coverage is a probability.
        let m = SelectionMask::from_fn(2, 1, |x, _| if x == 0 { -3.0 } else { 5.0 });
        assert_eq!(m.weight_at(0, 0), 0.0);
        assert_eq!(m.weight_at(1, 0), 1.0);
    }

    #[test]
    fn from_fn_half_weight_round_trips() {
        // 0.5 is exactly representable in f16, so weight_at returns it
        // unchanged — the texel a mix(input, result, 0.5) test relies on.
        let m = SelectionMask::from_fn(1, 1, |_, _| 0.5);
        assert_eq!(m.weight_at(0, 0), 0.5);
    }

    #[test]
    fn from_bytes_validates_length() {
        let good = vec![0u8; 2 * 3 * 2];
        assert!(SelectionMask::from_bytes(2, 3, good).is_some());
        let bad = vec![0u8; 5];
        assert!(SelectionMask::from_bytes(2, 3, bad).is_none());
    }

    #[test]
    fn from_bytes_round_trips_through_from_fn() {
        let a = SelectionMask::from_fn(3, 2, |x, y| (x + y) as f32 / 4.0);
        let b = SelectionMask::from_bytes(3, 2, a.bytes().to_vec()).unwrap();
        assert_eq!(a, b);
    }
}
