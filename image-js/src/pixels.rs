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

//! A pixel buffer that knows its own depth.
//!
//! # Why this type exists
//!
//! The layer store and the ingest both used to hold `Arc<[u8]>` — bare
//! bytes. Widening that to 16-bit directly would have been the worst
//! shape a refactor can have: every `i * 4`, `chunks_exact(4)` and
//! `len() / 4` in the codebase keeps COMPILING and starts reading the
//! wrong pixel, so the failure is a silently corrupted image rather
//! than a red build. There were 161 such sites across five files.
//!
//! So the depth travels WITH the bytes and there is deliberately no
//! `Deref<Target = [u8]>`: a caller has to say which view it wants.
//! That turns an invisible refactor into a mechanical one — the
//! compiler finds every site — which is the whole point of the type.
//!
//! `bytes_per_pixel()` exists so no call site writes the literal `4`.

use std::sync::Arc;

use image_core::SampleDepth;

/// Straight (non-premultiplied) RGBA at 8 or 16 bits per sample.
#[derive(Clone, Debug)]
pub struct Pixels {
    /// Tightly packed RGBA. At `U16` the samples are NATIVE-endian
    /// pairs — the codec already resolved wire order (PNG is
    /// big-endian on disk; `zune` hands back assembled `u16`s).
    bytes: Arc<[u8]>,
    depth: SampleDepth,
}

impl Pixels {
    /// Wrap an existing 8-bit buffer. `Arc` is kept, so this is free
    /// and a snapshot stays a pointer copy.
    pub fn from_rgba8(bytes: Arc<[u8]>) -> Self {
        Self {
            bytes,
            depth: SampleDepth::U8,
        }
    }

    /// Wrap 16-bit samples, packed to native-endian byte pairs.
    pub fn from_rgba16(samples: &[u16]) -> Self {
        let mut b = Vec::with_capacity(samples.len() * 2);
        for s in samples {
            b.extend_from_slice(&s.to_ne_bytes());
        }
        Self {
            bytes: Arc::from(b),
            depth: SampleDepth::U16,
        }
    }

    /// Re-wrap bytes that are ALREADY at `depth` — for a caller that
    /// took `raw()` apart and is putting it back, such as the undo
    /// journal. Wrong `depth` here is a silent misread, so this is
    /// deliberately not a `From`.
    pub fn from_raw(bytes: Arc<[u8]>, depth: SampleDepth) -> Self {
        Self { bytes, depth }
    }

    pub fn depth(&self) -> SampleDepth {
        self.depth
    }

    pub fn is_16bit(&self) -> bool {
        self.depth == SampleDepth::U16
    }

    /// 4 at `U8`, 8 at `U16`. Never write the literal.
    pub fn bytes_per_pixel(&self) -> usize {
        match self.depth {
            SampleDepth::U16 => 8,
            _ => 4,
        }
    }

    pub fn pixel_count(&self) -> usize {
        self.bytes.len() / self.bytes_per_pixel()
    }

    /// The raw bytes AT THEIR OWN DEPTH. Only correct for a caller that
    /// has consulted [`depth`](Self::depth) — most want
    /// [`to_rgba8`](Self::to_rgba8).
    pub fn raw(&self) -> &[u8] {
        &self.bytes
    }

    /// The `Arc` itself, for a caller that wants to share the
    /// allocation rather than copy it.
    pub fn raw_arc(&self) -> Arc<[u8]> {
        Arc::clone(&self.bytes)
    }

    /// An 8-bit view, NARROWING by the high byte when the buffer is
    /// 16-bit.
    ///
    /// `>> 8` rather than `/ 257`: the high byte is exactly what an
    /// 8-bit consumer would have read from the same sample, and scaling
    /// would push 16-bit precision into low bits that cannot survive
    /// the narrowing anyway. Borrowed when already 8-bit, so the common
    /// path allocates nothing.
    pub fn to_rgba8(&self) -> std::borrow::Cow<'_, [u8]> {
        match self.depth {
            SampleDepth::U16 => std::borrow::Cow::Owned(
                self.bytes
                    .chunks_exact(2)
                    .map(|p| u16::from_ne_bytes([p[0], p[1]]).to_be_bytes()[0])
                    .collect(),
            ),
            _ => std::borrow::Cow::Borrowed(&self.bytes),
        }
    }

    /// One channel of one pixel, WIDENED to 16 bits whatever the store.
    /// An 8-bit `v` becomes `v << 8 | v`, which maps 255 to 65535
    /// exactly — the property a plain `v << 8` loses, and the reason
    /// white stays white through a widening.
    pub fn sample16(&self, pixel: usize, channel: usize) -> u16 {
        match self.depth {
            SampleDepth::U16 => {
                let i = pixel * 8 + channel * 2;
                u16::from_ne_bytes([self.bytes[i], self.bytes[i + 1]])
            }
            _ => {
                let v = u16::from(self.bytes[pixel * 4 + channel]);
                (v << 8) | v
            }
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// Construction from 8-bit bytes is unambiguous and lossless, so it
/// gets a `From`. The READ side deliberately does not — that is where a
/// silent depth mistake would live, and where the caller must choose.
impl From<Vec<u8>> for Pixels {
    fn from(v: Vec<u8>) -> Self {
        Self::from_rgba8(Arc::from(v))
    }
}

impl From<Arc<[u8]>> for Pixels {
    fn from(v: Arc<[u8]>) -> Self {
        Self::from_rgba8(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_8bit_buffer_is_borrowed_not_copied() {
        let p = Pixels::from_rgba8(Arc::from(vec![1u8, 2, 3, 4]));
        assert!(matches!(p.to_rgba8(), std::borrow::Cow::Borrowed(_)));
        assert_eq!(p.bytes_per_pixel(), 4);
        assert_eq!(p.pixel_count(), 1);
    }

    #[test]
    fn a_16bit_buffer_narrows_by_the_high_byte() {
        let p = Pixels::from_rgba16(&[0x1234, 0x5678, 0x9ABC, 0xFFFF]);
        assert_eq!(p.bytes_per_pixel(), 8);
        assert_eq!(p.pixel_count(), 1);
        assert_eq!(&*p.to_rgba8(), &[0x12, 0x56, 0x9A, 0xFF]);
    }

    #[test]
    fn widening_maps_white_to_white_exactly() {
        // The property `v << 8` alone loses: 255 must become 65535, not
        // 65280, or a widened white is no longer white.
        let p = Pixels::from_rgba8(Arc::from(vec![255u8, 0, 128, 255]));
        assert_eq!(p.sample16(0, 0), 0xFFFF);
        assert_eq!(p.sample16(0, 1), 0x0000);
        assert_eq!(p.sample16(0, 2), 0x8080);
    }

    #[test]
    fn a_16bit_sample_reads_back_unchanged() {
        let p = Pixels::from_rgba16(&[0x1234, 0x5678, 0x9ABC, 0xFFFF]);
        assert_eq!(p.sample16(0, 0), 0x1234);
        assert_eq!(p.sample16(0, 3), 0xFFFF);
    }

    #[test]
    fn narrow_then_widen_is_stable_for_8bit_content() {
        // Round-tripping an 8-bit buffer through the 16-bit view and
        // back must be the identity, or every mixed-depth composite
        // drifts.
        let src: Vec<u8> = (0..=255).collect();
        let p = Pixels::from_rgba8(Arc::from(src.clone()));
        let wide: Vec<u16> = (0..src.len()).map(|i| p.sample16(i / 4, i % 4)).collect();
        let back = Pixels::from_rgba16(&wide);
        assert_eq!(&*back.to_rgba8(), &src[..]);
    }
}
