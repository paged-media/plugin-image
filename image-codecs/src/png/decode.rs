/*
 * This file is part of paged (https://paged.media).
 *
 * paged is free software: you may redistribute it and/or modify it under the
 * terms of the GNU Affero General Public License, version 3, as published by
 * the Free Software Foundation, OR under the Paged Media Enterprise License
 * (PMEL), a commercial license available from And The Next GmbH. Full
 * copyright and license information is available in LICENSE.md, distributed
 * with this source code.
 *
 * paged is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the licenses for details.
 *
 *  @copyright  Copyright (c) And The Next GmbH
 *  @license    AGPL-3.0-only OR Paged Media Enterprise License (PMEL)
 */

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

//! PNG decode adapter (`PngSource`) over [`ByteSource`].
//!
//! M0 decodes the whole frame once on first read and caches the result
//! in the spec `ChannelLayout` (RGB widened to RGBA, alpha=255). Windows
//! are served by copy from the cache. The streaming follow-up (a
//! row/strip-incremental reader) is a known M1 task — zune-png exposes
//! no public row API, so a chunked deflate reader is the plan.

use image_core::{ChannelLayout, Region, TileSliceMut};
use zune_core::bit_depth::BitDepth;
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_png::PngDecoder;

use crate::{ByteSource, CodecError, ImageSource, Result, SourceInfo};

use super::{png_format, PNG};

/// The fully decoded frame, kept in the probed (spec) format so windows
/// are pure copies. Provenance (native_format/icc/exif) lives on
/// `DecodedInfo` from `probe` — the frame is pixels only.
struct Decoded {
    width: u32,
    height: u32,
    channels: ChannelLayout,
    /// Tightly packed, interleaved, in the spec `channels` layout.
    pixels: Vec<u8>,
}

pub struct PngSource<B: ByteSource> {
    bytes: B,
    /// Populated by `probe`/the first read; the decode happens lazily so
    /// `probe` stays a header-only parse (`decode_headers`).
    info: Option<DecodedInfo>,
    frame: Option<Decoded>,
}

/// Header facts captured by `probe` before any pixel decode.
#[derive(Clone)]
struct DecodedInfo {
    width: u32,
    height: u32,
    channels: ChannelLayout,
    native_format: &'static str,
    icc: Option<Vec<u8>>,
    exif: Option<Vec<u8>>,
}

impl<B: ByteSource> PngSource<B> {
    pub fn new(bytes: B) -> Self {
        PngSource {
            bytes,
            info: None,
            frame: None,
        }
    }

    /// Slurp the whole container into memory. PNG is a single IDAT
    /// stream with no random access, so a codec-level read is whole-file
    /// regardless; the `ByteSource` seam is what lets the same adapter
    /// run over memory/OPFS/file backings (spec §10.2).
    fn slurp(&mut self) -> Result<Vec<u8>> {
        let len = self.bytes.len();
        let mut buf = vec![0u8; len as usize];
        self.bytes.read_at(0, &mut buf)?;
        Ok(buf)
    }

    /// Header parse shared by `probe` and the lazy decode: both need the
    /// same depth/colorspace/dimension facts, only the pixel decode that
    /// follows differs. Memoised in `self.info`.
    fn header(&mut self) -> Result<DecodedInfo> {
        if let Some(i) = &self.info {
            return Ok(i.clone());
        }
        let raw = self.slurp()?;
        let info = parse_headers(&raw)?;
        self.info = Some(info.clone());
        Ok(info)
    }

    fn decode_frame(&mut self) -> Result<()> {
        if self.frame.is_some() {
            return Ok(());
        }
        let DecodedInfo {
            width,
            height,
            channels,
            ..
        } = self.header()?;

        let raw = self.slurp()?;
        let mut dec = PngDecoder::new(ZCursor::new(&raw));
        let result = dec.decode().map_err(malformed)?;
        let src = result.u8().ok_or_else(|| CodecError::Unsupported {
            format: PNG,
            detail: "decoder returned non-U8 samples".into(),
        })?;

        // zune delivers RGB as 3 components; widen to the 4-channel
        // `Rgba` the spec layout demands, filling alpha at 255. Every
        // other mapping (Gray, GrayA, RGBA) is already 1:1 with its
        // `ChannelLayout`, so it passes through untouched.
        let pixels = if channels == ChannelLayout::Rgba
            && src.len() == width as usize * height as usize * 3
        {
            widen_rgb_to_rgba(&src, width as usize, height as usize)?
        } else {
            src
        };

        let expect = width as usize * height as usize * channels.count() as usize;
        if pixels.len() != expect {
            return Err(CodecError::Malformed {
                format: PNG,
                detail: format!("decoded {} bytes, expected {expect}", pixels.len()),
            });
        }

        self.frame = Some(Decoded {
            width,
            height,
            channels,
            pixels,
        });
        Ok(())
    }
}

impl<B: ByteSource> ImageSource for PngSource<B> {
    fn probe(&mut self) -> Result<SourceInfo> {
        let i = self.header()?;
        Ok(SourceInfo {
            width: i.width,
            height: i.height,
            format: png_format(i.channels),
            native_format: i.native_format,
            icc: i.icc,
            exif: i.exif,
            native_mips: Vec::new(),
        })
    }

    fn native_shrink(&self) -> &[u32] {
        // PNG has no DCT-style decoder downscale; the shrink-on-load
        // planner (§7.2) must resample post-decode.
        &[1]
    }

    fn read_region(&mut self, roi: Region, shrink: u32, out: &mut TileSliceMut<'_>) -> Result<()> {
        if shrink != 1 {
            return Err(CodecError::Unsupported {
                format: PNG,
                detail: format!("shrink {shrink}"),
            });
        }
        self.decode_frame()?;
        let frame = self.frame.as_ref().expect("decoded above");
        let fmt = png_format(frame.channels);

        if out.format != fmt {
            return Err(CodecError::Unsupported {
                format: PNG,
                detail: "format mismatch (no implicit conversions)".into(),
            });
        }
        let full = Region::new(0, 0, frame.width, frame.height);
        if roi.intersect(full) != Some(roi) || !out.validate() || out.region != roi {
            return Err(CodecError::Malformed {
                format: PNG,
                detail: format!("roi {roi:?} out of bounds or slice mismatch"),
            });
        }
        let bpp = fmt.bytes_per_pixel();
        let src_stride = frame.width as usize * bpp;
        let row_bytes = roi.w as usize * bpp;
        for row in 0..roi.h {
            let sy = (roi.y as usize + row as usize) * src_stride + roi.x as usize * bpp;
            let dy = row as usize * out.row_stride;
            out.bytes[dy..dy + row_bytes].copy_from_slice(&frame.pixels[sy..sy + row_bytes]);
        }
        Ok(())
    }
}

/// Parse the PNG container headers (no pixel decode) into the facts
/// `probe` reports and `decode_frame` reuses.
fn parse_headers(raw: &[u8]) -> Result<DecodedInfo> {
    let mut dec = PngDecoder::new(ZCursor::new(raw));
    dec.decode_headers().map_err(malformed)?;

    // 16-bit is the M1 lane; refuse rather than silently truncate.
    match dec.depth() {
        Some(BitDepth::Eight) => {}
        Some(BitDepth::Sixteen) => {
            return Err(CodecError::Unsupported {
                format: PNG,
                detail: "16-bit depth (M0 is U8 only)".into(),
            })
        }
        other => {
            return Err(CodecError::Unsupported {
                format: PNG,
                detail: format!("unsupported bit depth {other:?}"),
            })
        }
    }

    let cs = dec.colorspace().ok_or_else(|| CodecError::Malformed {
        format: PNG,
        detail: "no colorspace after header decode".into(),
    })?;
    let (channels, native_format) = map_colorspace(cs)?;
    let (w, h) = dec.dimensions().ok_or_else(|| CodecError::Malformed {
        format: PNG,
        detail: "no dimensions after header decode".into(),
    })?;
    let (icc, exif) = dec
        .info()
        .map(|i| (i.icc_profile.clone(), i.exif.clone()))
        .unwrap_or((None, None));

    Ok(DecodedInfo {
        width: u32::try_from(w).map_err(|_| dim_overflow())?,
        height: u32::try_from(h).map_err(|_| dim_overflow())?,
        channels,
        native_format,
        icc,
        exif,
    })
}

/// Map a zune PNG colorspace to the spec `ChannelLayout` + a provenance
/// string. RGB is the one widened case (it is not in `ChannelLayout`);
/// everything zune can emit from PNG that we don't model is `Unsupported`.
fn map_colorspace(cs: ColorSpace) -> Result<(ChannelLayout, &'static str)> {
    match cs {
        ColorSpace::Luma => Ok((ChannelLayout::Gray, "gray8")),
        ColorSpace::LumaA => Ok((ChannelLayout::GrayA, "graya8")),
        ColorSpace::RGB => Ok((ChannelLayout::Rgba, "rgb8")),
        ColorSpace::RGBA => Ok((ChannelLayout::Rgba, "rgba8")),
        other => Err(CodecError::Unsupported {
            format: PNG,
            detail: format!("colorspace {other:?}"),
        }),
    }
}

fn widen_rgb_to_rgba(rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let px = width.checked_mul(height).ok_or_else(dim_overflow)?;
    if rgb.len() != px * 3 {
        return Err(CodecError::Malformed {
            format: PNG,
            detail: format!("rgb buffer {} bytes, expected {}", rgb.len(), px * 3),
        });
    }
    let mut out = vec![0u8; px * 4];
    for (i, chunk) in rgb.chunks_exact(3).enumerate() {
        let d = i * 4;
        out[d] = chunk[0];
        out[d + 1] = chunk[1];
        out[d + 2] = chunk[2];
        out[d + 3] = 255;
    }
    Ok(out)
}

fn malformed(e: impl std::fmt::Display) -> CodecError {
    CodecError::Malformed {
        format: PNG,
        detail: e.to_string(),
    }
}

fn dim_overflow() -> CodecError {
    CodecError::Malformed {
        format: PNG,
        detail: "image dimensions exceed u32".into(),
    }
}
