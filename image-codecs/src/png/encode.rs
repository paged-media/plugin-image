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

//! PNG encode adapter (`PngTarget`) — accumulates strips into a single
//! frame buffer and encodes once at `finish`.
//!
//! Strips arrive top-to-bottom, non-overlapping, jointly covering the
//! full extent (`ImageTarget` contract); we place each into a packed
//! frame buffer keyed by row, then hand the whole buffer to zune-png's
//! `PngEncoder`. zune-png 0.5 exposes EXIF embedding (`add_exif_segment`)
//! but **no iCCP/ICC chunk writer** — when `TargetInfo.icc` is set we
//! record the gap and drop the profile rather than fail the encode; the
//! ICC-embed lane is BREAKAGE-tracked for M1.

use image_core::{ChannelLayout, Region, SampleDepth, TileSliceRef};
use zune_core::bit_depth::BitDepth;
use zune_core::colorspace::ColorSpace;
use zune_core::options::EncoderOptions;
use zune_png::PngEncoder;

use crate::{CodecError, EncodedStats, ImageTarget, Result, TargetInfo};

use super::PNG;

pub struct PngTarget {
    sink: Vec<u8>,
    state: State,
}

enum State {
    Idle,
    Open {
        info: TargetInfo,
        colorspace: ColorSpace,
        bpp: usize,
        /// Packed full-frame buffer in `info.format`; strips write here.
        buffer: Vec<u8>,
        /// Highest exclusive row covered so far, for the coverage check.
        rows_filled: u32,
    },
    Done,
}

impl PngTarget {
    /// `sink` receives the encoded bytes at `finish` (and is what the
    /// `EncodedStats.bytes_written` count reflects).
    pub fn new() -> Self {
        PngTarget {
            sink: Vec::new(),
            state: State::Idle,
        }
    }

    /// Take the encoded PNG bytes after a successful `finish`.
    pub fn into_bytes(self) -> Vec<u8> {
        self.sink
    }
}

impl Default for PngTarget {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageTarget for PngTarget {
    fn begin(&mut self, info: TargetInfo) -> Result<()> {
        if !matches!(self.state, State::Idle) {
            return Err(CodecError::Sequencing("begin called twice"));
        }
        // M0 is U8 only, matching the decode side.
        if info.format.depth != SampleDepth::U8 {
            return Err(CodecError::Unsupported {
                format: PNG,
                detail: format!("depth {:?} (M0 is U8 only)", info.format.depth),
            });
        }
        let colorspace = map_channels(info.format.channels)?;
        let bpp = info.format.bytes_per_pixel();
        let len = info
            .width
            .checked_mul(info.height)
            .and_then(|px| (px as usize).checked_mul(bpp))
            .ok_or_else(|| CodecError::Malformed {
                format: PNG,
                detail: "target extent overflows usize".into(),
            })?;
        // ICC embedding is unsupported by zune-png 0.5 (no iCCP writer) —
        // record the drop in the comment above; the profile is silently
        // not embedded rather than failing the encode.
        self.state = State::Open {
            info,
            colorspace,
            bpp,
            buffer: vec![0u8; len],
            rows_filled: 0,
        };
        Ok(())
    }

    fn write_strip(&mut self, region: Region, data: &TileSliceRef<'_>) -> Result<()> {
        let State::Open {
            info,
            bpp,
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
                format: PNG,
                detail: "strip format mismatch (no implicit conversions)".into(),
            });
        }
        // Strips are full-width and arrive in order (the contract); we
        // enforce both so a misuse is a clean error, not silent corruption.
        if region.x != 0 || region.w != info.width {
            return Err(CodecError::Sequencing("strip not full-width"));
        }
        if region.y as i64 != *rows_filled as i64 {
            return Err(CodecError::Sequencing("strip out of order / gapped"));
        }
        if region.bottom() > info.height as i64 {
            return Err(CodecError::Malformed {
                format: PNG,
                detail: format!("strip {region:?} exceeds target height {}", info.height),
            });
        }
        if !strip_coherent(region, data) {
            return Err(CodecError::Malformed {
                format: PNG,
                detail: "strip region/slice mismatch".into(),
            });
        }

        let row_bytes = info.width as usize * *bpp;
        let dst_stride = row_bytes;
        for row in 0..region.h {
            let dy = (region.y as usize + row as usize) * dst_stride;
            buffer[dy..dy + row_bytes].copy_from_slice(data.row(row));
        }
        *rows_filled += region.h;
        Ok(())
    }

    fn finish(&mut self) -> Result<EncodedStats> {
        let State::Open {
            info,
            colorspace,
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

        let opts = EncoderOptions::new(
            info.width as usize,
            info.height as usize,
            *colorspace,
            BitDepth::Eight,
        );
        let mut encoder = PngEncoder::new(buffer, opts);
        // `&mut Vec<u8>` is a `ZByteWriterTrait` sink; encode appends.
        self.sink.clear();
        let written = encoder
            .encode(&mut self.sink)
            .map_err(|e| CodecError::Io(format!("{e:?}")))?;

        self.state = State::Done;
        Ok(EncodedStats {
            bytes_written: written as u64,
        })
    }
}

/// Map the spec `ChannelLayout` onto a PNG colorspace. CMYK has no PNG
/// representation; gray/rgba pass straight through.
fn map_channels(channels: ChannelLayout) -> Result<ColorSpace> {
    match channels {
        ChannelLayout::Gray => Ok(ColorSpace::Luma),
        ChannelLayout::GrayA => Ok(ColorSpace::LumaA),
        ChannelLayout::Rgba => Ok(ColorSpace::RGBA),
        ChannelLayout::Cmyk | ChannelLayout::Cmyka => Err(CodecError::Unsupported {
            format: PNG,
            detail: format!("{channels:?} has no PNG representation"),
        }),
    }
}

/// Strip coherence: the slice must declare the same region as the call
/// and hold enough bytes at its stride. Mirrors `TileSliceMut::validate`
/// for the read side, but `TileSliceRef` has no such method on the frozen
/// type, so the check lives here.
fn strip_coherent(region: Region, data: &TileSliceRef<'_>) -> bool {
    let bpp = data.format.bytes_per_pixel();
    let row_bytes = data.region.w as usize * bpp;
    data.region == region
        && data.row_stride >= row_bytes
        && data.bytes.len()
            >= data.row_stride * data.region.h.saturating_sub(1) as usize + row_bytes
}
