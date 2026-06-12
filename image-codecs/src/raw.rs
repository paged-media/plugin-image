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

//! In-memory raw adapters. [`RawSource`] is the test/Engine-A-bring-up
//! leaf: an uncompressed interleaved pixel buffer behind the
//! `ImageSource` contract — no entropy coding, no container.
//! [`RawTarget`] is its sink mirror: an `ImageTarget` that assembles the
//! `to_encoder` strips into one tightly packed pixel buffer — the
//! structured-readback path for consumers that want PIXELS, not a
//! container (the image-js M4 ingest slice hands these to the C-1 image
//! scene item).

use image_core::{PixelFormat, Region, TileSliceMut, TileSliceRef};

use crate::{CodecError, EncodedStats, ImageSource, ImageTarget, Result, SourceInfo, TargetInfo};

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

/// An `ImageTarget` that assembles strips into one tightly packed,
/// interleaved pixel buffer — `RawSource`'s sink mirror. The
/// "container" is no container at all: `into_pixels()` after a
/// successful `finish` yields `width * height * bpp` bytes in the
/// `begin` format, row-major.
#[derive(Debug, Default)]
pub struct RawTarget {
    state: RawState,
}

#[derive(Debug, Default)]
enum RawState {
    #[default]
    Idle,
    Open {
        info: TargetInfo,
        bpp: usize,
        buffer: Vec<u8>,
        /// Highest exclusive row covered so far (strips arrive in order).
        rows_filled: u32,
    },
    Done {
        info: TargetInfo,
        buffer: Vec<u8>,
    },
}

impl RawTarget {
    pub fn new() -> Self {
        RawTarget::default()
    }

    /// The `begin` info, once begun (survives `finish`).
    pub fn info(&self) -> Option<&TargetInfo> {
        match &self.state {
            RawState::Idle => None,
            RawState::Open { info, .. } | RawState::Done { info, .. } => Some(info),
        }
    }

    /// Take the assembled pixels after a successful `finish`. Empty when
    /// the run never finished (the honest nothing, not a partial frame).
    pub fn into_pixels(self) -> Vec<u8> {
        match self.state {
            RawState::Done { buffer, .. } => buffer,
            _ => Vec::new(),
        }
    }
}

impl ImageTarget for RawTarget {
    fn begin(&mut self, info: TargetInfo) -> Result<()> {
        if !matches!(self.state, RawState::Idle) {
            return Err(CodecError::Sequencing("begin called twice"));
        }
        let bpp = info.format.bytes_per_pixel();
        let len = info
            .width
            .checked_mul(info.height)
            .and_then(|px| (px as usize).checked_mul(bpp))
            .ok_or_else(|| CodecError::Malformed {
                format: "raw",
                detail: "target extent overflows usize".into(),
            })?;
        self.state = RawState::Open {
            info,
            bpp,
            buffer: vec![0u8; len],
            rows_filled: 0,
        };
        Ok(())
    }

    fn write_strip(&mut self, region: Region, data: &TileSliceRef<'_>) -> Result<()> {
        let RawState::Open {
            info,
            bpp,
            buffer,
            rows_filled,
        } = &mut self.state
        else {
            return Err(CodecError::Sequencing(
                "write_strip before begin / after finish",
            ));
        };
        if data.format != info.format {
            return Err(CodecError::Unsupported {
                format: "raw",
                detail: "strip format mismatch (no implicit conversions)".into(),
            });
        }
        // Full-width, in-order strips (the to_encoder contract) — misuse
        // is a clean error, never silent corruption.
        if region.x != 0 || region.w != info.width {
            return Err(CodecError::Sequencing("strip not full-width"));
        }
        if region.y as i64 != *rows_filled as i64 {
            return Err(CodecError::Sequencing("strip out of order / gapped"));
        }
        if region.bottom() > info.height as i64 {
            return Err(CodecError::Sequencing("strip past target extent"));
        }
        let row_bytes = region.w as usize * *bpp;
        for row in 0..region.h as usize {
            let src = row * data.row_stride;
            let dst = (region.y as usize + row) * row_bytes;
            buffer[dst..dst + row_bytes].copy_from_slice(&data.bytes[src..src + row_bytes]);
        }
        *rows_filled = region.bottom() as u32;
        Ok(())
    }

    fn finish(&mut self) -> Result<EncodedStats> {
        // Check BEFORE taking so a misuse never destroys a Done buffer.
        if !matches!(self.state, RawState::Open { .. }) {
            return Err(CodecError::Sequencing("finish before begin / twice"));
        }
        let RawState::Open {
            info,
            buffer,
            rows_filled,
            ..
        } = std::mem::take(&mut self.state)
        else {
            unreachable!("matched Open above")
        };
        if rows_filled != info.height {
            return Err(CodecError::Sequencing("strips did not cover the target"));
        }
        let bytes_written = buffer.len() as u64;
        self.state = RawState::Done { info, buffer };
        Ok(EncodedStats { bytes_written })
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

    #[test]
    fn raw_target_assembles_strips() {
        let mut target = RawTarget::new();
        target
            .begin(TargetInfo {
                width: 2,
                height: 3,
                format: FMT_U8,
                icc: None,
            })
            .unwrap();
        // Two strips: rows 0..2 then row 2, with a padded source stride.
        let strip0: Vec<u8> = (0u8..16).collect(); // 2 rows × 8 bytes
        target
            .write_strip(
                Region::new(0, 0, 2, 2),
                &TileSliceRef {
                    region: Region::new(0, 0, 2, 2),
                    format: FMT_U8,
                    bytes: &strip0,
                    row_stride: 8,
                },
            )
            .unwrap();
        let strip1: Vec<u8> = (100u8..108).collect();
        target
            .write_strip(
                Region::new(0, 2, 2, 1),
                &TileSliceRef {
                    region: Region::new(0, 2, 2, 1),
                    format: FMT_U8,
                    bytes: &strip1,
                    row_stride: 8,
                },
            )
            .unwrap();
        let stats = target.finish().unwrap();
        assert_eq!(stats.bytes_written, 24);
        let px = target.into_pixels();
        assert_eq!(&px[..16], &(0u8..16).collect::<Vec<u8>>()[..]);
        assert_eq!(&px[16..], &(100u8..108).collect::<Vec<u8>>()[..]);
    }

    #[test]
    fn raw_target_rejects_misuse() {
        let mut t = RawTarget::new();
        assert!(t.finish().is_err()); // finish before begin
        t.begin(TargetInfo {
            width: 2,
            height: 2,
            format: FMT_U8,
            icc: None,
        })
        .unwrap();
        // Out-of-order strip.
        let bytes = vec![0u8; 8];
        let slice = TileSliceRef {
            region: Region::new(0, 1, 2, 1),
            format: FMT_U8,
            bytes: &bytes,
            row_stride: 8,
        };
        assert!(t.write_strip(Region::new(0, 1, 2, 1), &slice).is_err());
        // Under-covered finish.
        assert!(t.finish().is_err());
    }
}
