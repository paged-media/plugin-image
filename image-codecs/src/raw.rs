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

//! In-memory raw source — the test/Engine-A-bring-up adapter: an
//! uncompressed interleaved pixel buffer behind the `ImageSource`
//! contract. No entropy coding, no container; exists so the pipeline
//! has a leaf to pull from before real codecs land (and forever after
//! for synthetic test inputs).

use image_core::{PixelFormat, Region, TileSliceMut};

use crate::{CodecError, ImageSource, Result, SourceInfo};

#[derive(Debug, Clone)]
pub struct RawSource {
    width: u32,
    height: u32,
    format: PixelFormat,
    /// Interleaved, tightly packed (`width * bpp` stride).
    pixels: std::sync::Arc<[u8]>,
}

impl RawSource {
    pub fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        pixels: impl Into<std::sync::Arc<[u8]>>,
    ) -> Result<Self> {
        let pixels = pixels.into();
        let expect = width as usize * height as usize * format.bytes_per_pixel();
        if pixels.len() != expect {
            return Err(CodecError::Malformed {
                format: "raw",
                detail: format!("pixel buffer {} bytes, expected {expect}", pixels.len()),
            });
        }
        Ok(RawSource {
            width,
            height,
            format,
            pixels,
        })
    }
}

impl ImageSource for RawSource {
    fn probe(&mut self) -> Result<SourceInfo> {
        Ok(SourceInfo {
            width: self.width,
            height: self.height,
            format: self.format,
            native_format: "raw",
            icc: None,
            exif: None,
            native_mips: Vec::new(),
        })
    }

    fn native_shrink(&self) -> &[u32] {
        &[1]
    }

    fn read_region(&mut self, roi: Region, shrink: u32, out: &mut TileSliceMut<'_>) -> Result<()> {
        if shrink != 1 {
            return Err(CodecError::Unsupported {
                format: "raw",
                detail: format!("shrink {shrink}"),
            });
        }
        if out.format != self.format {
            return Err(CodecError::Unsupported {
                format: "raw",
                detail: "format mismatch (no implicit conversions)".into(),
            });
        }
        let full = Region::new(0, 0, self.width, self.height);
        if roi.intersect(full) != Some(roi) || !out.validate() || out.region != roi {
            return Err(CodecError::Malformed {
                format: "raw",
                detail: format!("roi {roi:?} out of bounds or slice mismatch"),
            });
        }
        let bpp = self.format.bytes_per_pixel();
        let src_stride = self.width as usize * bpp;
        let row_bytes = roi.w as usize * bpp;
        for row in 0..roi.h {
            let sy = (roi.y as usize + row as usize) * src_stride + roi.x as usize * bpp;
            let dy = row as usize * out.row_stride;
            out.bytes[dy..dy + row_bytes].copy_from_slice(&self.pixels[sy..sy + row_bytes]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image_core::{AlphaMode, ChannelLayout, ColorSpaceRef, NamedSpace, SampleDepth, Transfer};

    const FMT_U8: PixelFormat = PixelFormat {
        channels: ChannelLayout::Rgba,
        depth: SampleDepth::U8,
        alpha: AlphaMode::Straight,
        transfer: Transfer::Linear,
        space: ColorSpaceRef::Named(NamedSpace::LinearSrgb),
    };

    #[test]
    fn read_region_subwindow() {
        // 4x2 RGBA8 image with pixel value = x index in R channel.
        let mut px = vec![0u8; 4 * 2 * 4];
        for y in 0..2 {
            for x in 0..4 {
                px[(y * 4 + x) * 4] = x as u8;
            }
        }
        let mut src = RawSource::new(4, 2, FMT_U8, px.into_boxed_slice()).unwrap();
        assert_eq!(src.probe().unwrap().width, 4);

        let roi = Region::new(1, 0, 2, 2);
        let mut buf = vec![0u8; 2 * 2 * 4];
        let mut out = TileSliceMut {
            region: roi,
            format: FMT_U8,
            row_stride: 2 * 4,
            bytes: &mut buf,
        };
        src.read_region(roi, 1, &mut out).unwrap();
        assert_eq!(buf[0], 1); // first pixel R = x index 1
        assert_eq!(buf[4], 2);
    }
}
