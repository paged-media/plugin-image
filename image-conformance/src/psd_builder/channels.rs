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

//! Channel image-data encoding (brief §10/§11) — the bridge from an
//! 8-bit plane (`height × width` row-major bytes) to the on-disk
//! compressed channel payload. Two layouts: per-channel (layer image
//! data, each channel framed by its own length) and merged composite
//! (the trailing, unframed image-data section).
//!
//! Compression is reused from image-psd's *model* enum only — a plain
//! `Copy` tag. The actual RLE bytes come from this crate's own
//! [`super::emit::pack_bits_row`], never image-psd's compressor.

use image_psd::container::LenWidth;
use image_psd::model::Compression;

use super::emit::{pack_bits_row, Emit};

/// One channel's planar samples: `height × width` bytes, row-major. The
/// builder hands these in; the depth is fixed at 8 (all M0 fixtures).
#[derive(Debug, Clone)]
pub struct Plane {
    pub width: u32,
    pub height: u32,
    pub samples: Vec<u8>,
}

impl Plane {
    pub fn new(width: u32, height: u32, samples: Vec<u8>) -> Self {
        assert_eq!(
            samples.len(),
            (width as usize) * (height as usize),
            "plane sample count must equal width*height (8-bit)"
        );
        Plane {
            width,
            height,
            samples,
        }
    }

    /// A constant-valued plane (the usual fixture fill).
    pub fn solid(width: u32, height: u32, value: u8) -> Self {
        Plane::new(width, height, vec![value; (width * height) as usize])
    }

    fn row(&self, y: u32) -> &[u8] {
        let w = self.width as usize;
        let start = (y as usize) * w;
        &self.samples[start..start + w]
    }
}

/// Encode ONE channel's body: the 2-byte compression tag followed by the
/// payload (brief §10). RLE prepends this channel's per-row byte-count
/// table (u16 PSD / u32 PSB), then the packed rows concatenated. The
/// returned bytes are exactly what the per-channel length field frames
/// (and that length INCLUDES the compression tag — brief §5).
pub fn encode_channel(plane: &Plane, comp: Compression, w: LenWidth) -> Vec<u8> {
    let mut e = Emit::new();
    e.u16(comp.code());
    match comp {
        Compression::Raw => {
            e.raw(&plane.samples);
        }
        Compression::Rle => {
            let packed: Vec<Vec<u8>> = (0..plane.height)
                .map(|y| pack_bits_row(plane.row(y)))
                .collect();
            for row in &packed {
                e.row_count(w, row.len() as u32);
            }
            for row in &packed {
                e.raw(row);
            }
        }
        // ZIP variants are opaque in M0 (brief §10); fixtures never ask
        // for them, so refuse rather than emit a half-formed body.
        Compression::Zip | Compression::ZipPrediction => {
            panic!("psd_builder does not synthesize ZIP channel data (M0)")
        }
    }
    e.into_bytes()
}

/// Encode the merged composite / image-data section (brief §11): a
/// SINGLE compression tag, then all channels planar-sequential. RLE
/// front-loads the row-count table for ALL `height × channels` rows
/// before any packed bytes. This section is NOT length-framed — it runs
/// to EOF — so the caller appends the result last.
pub fn encode_composite(planes: &[Plane], comp: Compression, w: LenWidth) -> Vec<u8> {
    let mut e = Emit::new();
    e.u16(comp.code());
    match comp {
        Compression::Raw => {
            for p in planes {
                e.raw(&p.samples);
            }
        }
        Compression::Rle => {
            // All rows, channel-major, packed once; counts first.
            let mut packed: Vec<Vec<u8>> = Vec::new();
            for p in planes {
                for y in 0..p.height {
                    packed.push(pack_bits_row(p.row(y)));
                }
            }
            for row in &packed {
                e.row_count(w, row.len() as u32);
            }
            for row in &packed {
                e.raw(row);
            }
        }
        Compression::Zip | Compression::ZipPrediction => {
            panic!("psd_builder does not synthesize ZIP composite data (M0)")
        }
    }
    e.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psd_builder_channel_raw_layout() {
        let p = Plane::new(2, 2, vec![1, 2, 3, 4]);
        let body = encode_channel(&p, Compression::Raw, LenWidth::U32);
        // tag(0x0000) + 4 raw bytes.
        assert_eq!(body, vec![0, 0, 1, 2, 3, 4]);
    }

    #[test]
    fn psd_builder_channel_rle_counts_precede_rows() {
        // 2 rows of a solid color: each row packs to a 2-byte repeat
        // packet (control + value), so both counts are 2.
        let p = Plane::solid(4, 2, 0x7F);
        let body = encode_channel(&p, Compression::Rle, LenWidth::U32);
        // tag(0x0001) + count[2] + count[2] + row0[2] + row1[2].
        assert_eq!(&body[..2], &[0, 1]);
        assert_eq!(&body[2..6], &[0, 2, 0, 2]); // two u16 counts
        assert_eq!(&body[6..8], &[(-3i8) as u8, 0x7F]); // row0: repeat 4
        assert_eq!(&body[8..10], &[(-3i8) as u8, 0x7F]); // row1
    }

    #[test]
    fn psd_builder_channel_rle_psb_uses_u32_counts() {
        let p = Plane::solid(4, 1, 0x7F);
        let body = encode_channel(&p, Compression::Rle, LenWidth::U64);
        // tag + ONE u32 count (4 bytes) + the packed row.
        assert_eq!(&body[..2], &[0, 1]);
        assert_eq!(&body[2..6], &[0, 0, 0, 2]);
    }

    #[test]
    fn psd_builder_composite_raw_planar_sequential() {
        let r = Plane::new(2, 1, vec![10, 11]);
        let g = Plane::new(2, 1, vec![20, 21]);
        let body = encode_composite(&[r, g], Compression::Raw, LenWidth::U32);
        assert_eq!(body, vec![0, 0, 10, 11, 20, 21]);
    }

    #[test]
    fn psd_builder_composite_rle_all_counts_first() {
        let r = Plane::solid(4, 1, 1);
        let g = Plane::solid(4, 1, 2);
        let body = encode_composite(&[r, g], Compression::Rle, LenWidth::U32);
        // tag + 2 counts (one per channel-row) + 2 packed rows.
        assert_eq!(&body[..2], &[0, 1]);
        assert_eq!(&body[2..6], &[0, 2, 0, 2]);
        assert_eq!(&body[6..8], &[(-3i8) as u8, 1]);
        assert_eq!(&body[8..10], &[(-3i8) as u8, 2]);
    }
}
