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

//! JPEG encode adapter (`JpegTarget`, jpeg-encoder) — accumulates strips
//! into a single frame buffer and encodes once at `finish`.
//!
//! jpeg-encoder's `Encoder::new(writer, quality)` takes the sink by value
//! and `encode(self, …)` consumes the encoder, so (like `PngTarget`) we
//! buffer the whole frame and build the encoder at `finish`.
//!
//! Scope:
//!  - **RGB + Gray.** `Rgba` strips have their alpha stripped (JPEG has no
//!    alpha — documented no-alpha lane); `Gray` encodes as `Luma`.
//!    CMYK/GrayA encode is out of M0 scope (`Unsupported`).
//!  - **Quality** is the `JpegTarget::new(quality)` constructor param (the
//!    frozen `TargetInfo` carries no codec options).
//!  - **Chroma subsampling defaults to 4:2:0** (`SamplingFactor::R_4_2_0`).
//!  - **ICC**: jpeg-encoder can embed an ICC profile, but `TargetInfo.icc`
//!    embedding is wired through `begin`; if the encoder API path is not
//!    exercised we record the profile and skip embedding (M1 follow-up).

use image_core::{ChannelLayout, Region, SampleDepth, TileSliceRef};
use jpeg_encoder::{ColorType, Encoder, SamplingFactor};

use crate::{CodecError, EncodedStats, ImageTarget, Result, TargetInfo};

use super::JPEG;

pub struct JpegTarget {
    quality: u8,
    sink: Vec<u8>,
    state: State,
}

enum State {
    Idle,
    Open {
        info: TargetInfo,
        color: ColorType,
        /// Bytes per pixel of the encoder input (after alpha strip).
        out_bpp: usize,
        /// Packed full-frame buffer in the encoder's input layout.
        buffer: Vec<u8>,
        rows_filled: u32,
    },
    Done,
}

impl JpegTarget {
    /// `quality` is the JPEG quality (1..=100); the frozen `TargetInfo`
    /// carries no codec options, so it lives on the adapter constructor.
    pub fn new(quality: u8) -> Self {
        JpegTarget {
            quality: quality.clamp(1, 100),
            sink: Vec::new(),
            state: State::Idle,
        }
    }

    /// Take the encoded JPEG bytes after a successful `finish`.
    pub fn into_bytes(self) -> Vec<u8> {
        self.sink
    }
}

impl ImageTarget for JpegTarget {
    fn begin(&mut self, info: TargetInfo) -> Result<()> {
        if !matches!(self.state, State::Idle) {
            return Err(CodecError::Sequencing("begin called twice"));
        }
        if info.format.depth != SampleDepth::U8 {
            return Err(CodecError::Unsupported {
                format: JPEG,
                detail: format!("depth {:?} (JPEG is U8 only here)", info.format.depth),
            });
        }
        // The encoder input layout: RGB (3) for Rgba (alpha stripped),
        // Luma (1) for Gray. The strip stays in `info.format`; we narrow
        // at the strip copy.
        let (color, out_bpp) = match info.format.channels {
            ChannelLayout::Rgba => (ColorType::Rgb, 3),
            ChannelLayout::Gray => (ColorType::Luma, 1),
            other => {
                return Err(CodecError::Unsupported {
                    format: JPEG,
                    detail: format!("{other:?} (encode is RGB/Gray only)"),
                })
            }
        };
        // JPEG dimensions are 16-bit in the SOF marker.
        if info.width == 0 || info.height == 0 {
            return Err(CodecError::Malformed {
                format: JPEG,
                detail: "zero target dimension".into(),
            });
        }
        if info.width > u16::MAX as u32 || info.height > u16::MAX as u32 {
            return Err(CodecError::Unsupported {
                format: JPEG,
                detail: format!(
                    "dimension {}x{} exceeds JPEG's 65535 limit",
                    info.width, info.height
                ),
            });
        }
        let len = (info.width as usize)
            .checked_mul(info.height as usize)
            .and_then(|px| px.checked_mul(out_bpp))
            .ok_or_else(|| CodecError::Malformed {
                format: JPEG,
                detail: "target extent overflows usize".into(),
            })?;
        self.state = State::Open {
            info,
            color,
            out_bpp,
            buffer: vec![0u8; len],
            rows_filled: 0,
        };
        Ok(())
    }

    fn write_strip(&mut self, region: Region, data: &TileSliceRef<'_>) -> Result<()> {
        let State::Open {
            info,
            out_bpp,
            buffer,
            rows_filled,
            ..
        } = &mut self.state
        else {
            return Err(CodecError::Sequencing(
                "write_strip before begin / after finish",
            ));
        };

        if data.format != info.format {
            return Err(CodecError::Unsupported {
                format: JPEG,
                detail: "strip format mismatch (no implicit conversions)".into(),
            });
        }
        if region.x != 0 || region.w != info.width {
            return Err(CodecError::Sequencing("strip not full-width"));
        }
        if region.y as i64 != *rows_filled as i64 {
            return Err(CodecError::Sequencing("strip out of order / gapped"));
        }
        if region.bottom() > info.height as i64 {
            return Err(CodecError::Malformed {
                format: JPEG,
                detail: format!("strip {region:?} exceeds target height {}", info.height),
            });
        }
        if !strip_coherent(region, data) {
            return Err(CodecError::Malformed {
                format: JPEG,
                detail: "strip region/slice mismatch".into(),
            });
        }

        let in_bpp = info.format.bytes_per_pixel();
        let dst_stride = info.width as usize * *out_bpp;
        let w = info.width as usize;
        for row in 0..region.h {
            let src = data.row(row);
            let dy = (region.y as usize + row as usize) * dst_stride;
            let dst = &mut buffer[dy..dy + dst_stride];
            // Copy the leading `out_bpp` channels of each pixel (RGB from
            // RGBA strips the alpha; Gray copies the single channel).
            for x in 0..w {
                let s = x * in_bpp;
                let d = x * *out_bpp;
                dst[d..d + *out_bpp].copy_from_slice(&src[s..s + *out_bpp]);
            }
        }
        *rows_filled += region.h;
        Ok(())
    }

    fn finish(&mut self) -> Result<EncodedStats> {
        let State::Open {
            info,
            color,
            buffer,
            rows_filled,
            ..
        } = &self.state
        else {
            return Err(CodecError::Sequencing("finish before begin / called twice"));
        };
        if *rows_filled != info.height {
            return Err(CodecError::Sequencing("finish before full coverage"));
        }

        self.sink.clear();
        let mut encoder = Encoder::new(&mut self.sink, self.quality);
        // 4:2:0 default (the common web/print baseline subsampling).
        encoder.set_sampling_factor(SamplingFactor::R_4_2_0);
        encoder
            .encode(buffer, info.width as u16, info.height as u16, *color)
            .map_err(|e| CodecError::Io(format!("{e}")))?;

        let written = self.sink.len() as u64;
        self.state = State::Done;
        Ok(EncodedStats {
            bytes_written: written,
        })
    }
}

/// Strip coherence: the slice must declare the same region as the call
/// and hold enough bytes at its stride (mirrors `PngTarget::strip_coherent`
/// — `TileSliceRef` has no `validate` on the frozen type).
fn strip_coherent(region: Region, data: &TileSliceRef<'_>) -> bool {
    let bpp = data.format.bytes_per_pixel();
    let row_bytes = data.region.w as usize * bpp;
    data.region == region
        && data.row_stride >= row_bytes
        && data.bytes.len()
            >= data.row_stride * data.region.h.saturating_sub(1) as usize + row_bytes
}
