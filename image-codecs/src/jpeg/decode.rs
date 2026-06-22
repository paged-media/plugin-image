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

//! JPEG decode adapter (`JpegSource`) over [`ByteSource`].
//!
//! M0 decodes the whole frame once on first read and caches it in the
//! spec `ChannelLayout` (3-channel YCbCr/RGB widened to RGBA alpha=255;
//! CMYK/YCCK delivered as 4-channel ink in `ChannelLayout::Cmyk`). Windows
//! are served by copy. `native_shrink == [1]` — zune-jpeg exposes no
//! public DCT-scaled decode, so no decoder-side downscale (the shrink
//! planner must resample post-decode for JPEG; see `mod.rs`).

use image_core::{ChannelLayout, Region, TileSliceMut};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

use crate::{ByteSource, CodecError, ImageSource, Result, SourceInfo};

use super::{jpeg_format, JPEG};

/// How the four-component raw samples zune hands back must be turned into
/// true CMYK ink (spec §10.3 / the Adobe APP14 convention — see `mod.rs`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum FourChan {
    /// Input colorspace was CMYK: zune passed the stored samples through
    /// untouched. Adobe stores them inverted, so re-invert each channel.
    Cmyk,
    /// Input colorspace was YCCK: zune passed Y,Cb,Cr,K through untouched;
    /// we run the YCbCr→RGB matrix on the first three (yielding stored,
    /// inverted CMY) then invert all four (including K).
    Ycck,
}

/// The fully decoded frame in the probed (spec) layout, so windows are
/// pure copies.
struct Decoded {
    width: u32,
    height: u32,
    channels: ChannelLayout,
    /// Tightly packed, interleaved, in the spec `channels` layout.
    pixels: Vec<u8>,
}

pub struct JpegSource<B: ByteSource> {
    bytes: B,
    info: Option<DecodedInfo>,
    frame: Option<Decoded>,
}

/// Header facts captured by `probe` before any pixel decode.
#[derive(Clone)]
struct DecodedInfo {
    width: u32,
    height: u32,
    /// The spec layout the decoded frame is delivered in.
    channels: ChannelLayout,
    native_format: &'static str,
    /// The colorspace we request from zune for the pixel decode.
    out_cs: ColorSpace,
    /// For 4-channel inputs, how to reconstruct true ink from the raw
    /// samples zune passes through. `None` for Gray/RGB/YCbCr.
    four: Option<FourChan>,
    icc: Option<Vec<u8>>,
    exif: Option<Vec<u8>>,
}

impl<B: ByteSource> JpegSource<B> {
    pub fn new(bytes: B) -> Self {
        JpegSource {
            bytes,
            info: None,
            frame: None,
        }
    }

    /// Slurp the whole container into memory. JPEG entropy-coded data has
    /// no useful random access, so a codec-level read is whole-file; the
    /// `ByteSource` seam is what lets the same adapter run over
    /// memory/OPFS/file backings (spec §10.2).
    fn slurp(&mut self) -> Result<Vec<u8>> {
        let len = self.bytes.len();
        let mut buf = vec![0u8; len as usize];
        self.bytes.read_at(0, &mut buf)?;
        Ok(buf)
    }

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
            out_cs,
            four,
            ..
        } = self.header()?;

        let raw = self.slurp()?;
        // Gray→Luma, YCbCr/RGB→RGB (zune converts/passes through). For the
        // 4-channel inputs we request the *matching* input colorspace
        // (CMYK/YCCK) so zune hands the raw stored samples back without
        // its own CMYK→RGB collapse — we own the ink reconstruction (the
        // Adobe APP14 rule, `mod.rs`).
        let opts = DecoderOptions::default().jpeg_set_out_colorspace(out_cs);
        let mut dec = JpegDecoder::new_with_options(ZCursor::new(&raw), opts);
        let decoded = dec.decode().map_err(malformed)?;

        let px = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(dim_overflow)?;

        let pixels = match channels {
            // Gray: we requested Luma out, so the decode is single-channel
            // and passes through 1:1.
            ChannelLayout::Gray => {
                if decoded.len() != px {
                    return Err(unexpected_len(decoded.len(), px));
                }
                decoded
            }
            ChannelLayout::Rgba => {
                if decoded.len() != px * 3 {
                    return Err(unexpected_len(decoded.len(), px * 3));
                }
                widen_rgb_to_rgba(&decoded, px)
            }
            ChannelLayout::Cmyk => {
                if decoded.len() != px * 4 {
                    return Err(unexpected_len(decoded.len(), px * 4));
                }
                match four {
                    Some(FourChan::Cmyk) => cmyk_from_stored(&decoded, px),
                    Some(FourChan::Ycck) => cmyk_from_ycck(&decoded, px),
                    None => return Err(internal("4-channel layout without a CMYK source kind")),
                }
            }
            other => {
                return Err(CodecError::Unsupported {
                    format: JPEG,
                    detail: format!("layout {other:?}"),
                })
            }
        };

        let expect = px * channels.count() as usize;
        if pixels.len() != expect {
            return Err(CodecError::Malformed {
                format: JPEG,
                detail: format!("post-mapping {} bytes, expected {expect}", pixels.len()),
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

impl<B: ByteSource> ImageSource for JpegSource<B> {
    fn probe(&mut self) -> Result<SourceInfo> {
        let i = self.header()?;
        Ok(SourceInfo {
            width: i.width,
            height: i.height,
            format: jpeg_format(i.channels),
            native_format: i.native_format,
            icc: i.icc,
            exif: i.exif,
            native_mips: Vec::new(),
        })
    }

    fn native_shrink(&self) -> &[u32] {
        // zune-jpeg 0.5.15 has no public DCT-scaled (1/2,1/4,1/8) decode
        // entry; advertise no decoder downscale. Consequence: the
        // shrink-on-load planner (§7.2) resamples post-decode for JPEG,
        // paying a full-resolution IDCT even for an 1/8 request.
        &[1]
    }

    fn read_region(&mut self, roi: Region, shrink: u32, out: &mut TileSliceMut<'_>) -> Result<()> {
        if shrink != 1 {
            return Err(CodecError::Unsupported {
                format: JPEG,
                detail: format!("shrink {shrink} (native_shrink is [1])"),
            });
        }
        self.decode_frame()?;
        let frame = self.frame.as_ref().expect("decoded above");
        let fmt = jpeg_format(frame.channels);

        if out.format != fmt {
            return Err(CodecError::Unsupported {
                format: JPEG,
                detail: "format mismatch (no implicit conversions)".into(),
            });
        }
        let full = Region::new(0, 0, frame.width, frame.height);
        if roi.intersect(full) != Some(roi) || !out.validate() || out.region != roi {
            return Err(CodecError::Malformed {
                format: JPEG,
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

/// Parse JPEG headers (no full pixel decode) into the facts `probe`
/// reports and `decode_frame` reuses. zune's `input_colorspace()` already
/// reflects the Adobe APP14 transform marker (transform 0→CMYK, 2→YCCK,
/// 1→YCbCr) and the "4 components, no APP14 ⇒ CMYK" default.
fn parse_headers(raw: &[u8]) -> Result<DecodedInfo> {
    let mut dec = JpegDecoder::new(ZCursor::new(raw));
    dec.decode_headers().map_err(malformed)?;

    let info = dec.info().ok_or_else(|| CodecError::Malformed {
        format: JPEG,
        detail: "no image info after header decode".into(),
    })?;
    let cs = dec
        .input_colorspace()
        .ok_or_else(|| CodecError::Malformed {
            format: JPEG,
            detail: "no input colorspace after header decode".into(),
        })?;
    let m = map_colorspace(cs)?;

    // icc_profile() concatenates the sequence-numbered APP2 ICC_PROFILE
    // chunks (spec §10.3) — the read path needs no M1.5 follow-up.
    let icc = dec.icc_profile();
    let exif = dec.exif().cloned();

    if info.width == 0 || info.height == 0 {
        return Err(CodecError::Malformed {
            format: JPEG,
            detail: "zero image dimension".into(),
        });
    }

    Ok(DecodedInfo {
        width: u32::from(info.width),
        height: u32::from(info.height),
        channels: m.channels,
        native_format: m.native_format,
        out_cs: m.out_cs,
        four: m.four,
        icc,
        exif,
    })
}

/// The decode strategy for one zune *input* colorspace.
struct Mapping {
    channels: ChannelLayout,
    native_format: &'static str,
    out_cs: ColorSpace,
    four: Option<FourChan>,
}

/// Map a zune JPEG *input* colorspace onto the spec `ChannelLayout`, the
/// colorspace to request from zune, a provenance string, and (for
/// 4-channel) the ink-reconstruction kind.
fn map_colorspace(cs: ColorSpace) -> Result<Mapping> {
    Ok(match cs {
        ColorSpace::Luma => Mapping {
            channels: ChannelLayout::Gray,
            native_format: "gray8",
            out_cs: ColorSpace::Luma,
            four: None,
        },
        ColorSpace::YCbCr => Mapping {
            channels: ChannelLayout::Rgba,
            native_format: "ycbcr8",
            out_cs: ColorSpace::RGB,
            four: None,
        },
        ColorSpace::RGB => Mapping {
            channels: ChannelLayout::Rgba,
            native_format: "rgb8",
            out_cs: ColorSpace::RGB,
            four: None,
        },
        ColorSpace::CMYK => Mapping {
            channels: ChannelLayout::Cmyk,
            native_format: "cmyk8+adobe-app14",
            out_cs: ColorSpace::CMYK,
            four: Some(FourChan::Cmyk),
        },
        ColorSpace::YCCK => Mapping {
            channels: ChannelLayout::Cmyk,
            native_format: "ycck8+adobe-app14",
            out_cs: ColorSpace::YCCK,
            four: Some(FourChan::Ycck),
        },
        other => {
            return Err(CodecError::Unsupported {
                format: JPEG,
                detail: format!("input colorspace {other:?}"),
            })
        }
    })
}

fn widen_rgb_to_rgba(rgb: &[u8], px: usize) -> Vec<u8> {
    let mut out = vec![0u8; px * 4];
    for (i, chunk) in rgb.chunks_exact(3).enumerate() {
        let d = i * 4;
        out[d] = chunk[0];
        out[d + 1] = chunk[1];
        out[d + 2] = chunk[2];
        out[d + 3] = 255;
    }
    out
}

/// CMYK input: zune handed back the *stored* samples. Adobe stores CMYK
/// inverted (`stored = 255 - ink`), so re-invert each of the four channels
/// to recover true ink amounts (the APP14 rule, `mod.rs`). A 4-component
/// JPEG without APP14 is treated the same here — the de-facto convention
/// is that CMYK JPEGs are inverted; the rare non-inverted-no-APP14 file
/// would decode inverted (a documented limitation pending APP14-presence
/// exposure from zune-jpeg).
fn cmyk_from_stored(stored: &[u8], px: usize) -> Vec<u8> {
    let mut out = vec![0u8; px * 4];
    for (o, s) in out.iter_mut().zip(stored.iter()) {
        *o = 255 - *s;
    }
    out
}

/// YCCK input: zune handed back raw Y,Cb,Cr,K. The Y,Cb,Cr triple is the
/// JFIF YCbCr transform of the *stored* (inverted) CMY; run the inverse
/// matrix to get stored CMY, then invert all four channels (CMY + K) to
/// true ink (the same Adobe APP14 rule).
fn cmyk_from_ycck(ycck: &[u8], px: usize) -> Vec<u8> {
    let mut out = vec![0u8; px * 4];
    for (i, chunk) in ycck.chunks_exact(4).enumerate() {
        let (r, g, b) = ycbcr_to_rgb(chunk[0], chunk[1], chunk[2]);
        let d = i * 4;
        // r,g,b == stored (inverted) C,M,Y ⇒ true ink = 255 - stored.
        out[d] = 255 - r;
        out[d + 1] = 255 - g;
        out[d + 2] = 255 - b;
        out[d + 3] = 255 - chunk[3];
    }
    out
}

/// Full-range JFIF / ITU-T T.871 YCbCr→RGB. Rounded fixed-point would
/// match zune bit-for-bit; for the YCCK ink path the float form is fine
/// (the result is re-inverted ink, M1 CMS owns the precise transform).
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y = y as f32;
    let cb = cb as f32 - 128.0;
    let cr = cr as f32 - 128.0;
    let r = y + 1.402 * cr;
    let g = y - 0.344_136 * cb - 0.714_136 * cr;
    let b = y + 1.772 * cb;
    (clamp_u8(r), clamp_u8(g), clamp_u8(b))
}

fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

fn malformed(e: impl std::fmt::Display) -> CodecError {
    CodecError::Malformed {
        format: JPEG,
        detail: e.to_string(),
    }
}

fn dim_overflow() -> CodecError {
    CodecError::Malformed {
        format: JPEG,
        detail: "image dimensions exceed addressable size".into(),
    }
}

fn unexpected_len(got: usize, expected: usize) -> CodecError {
    CodecError::Malformed {
        format: JPEG,
        detail: format!("decoder returned {got} bytes, expected {expected}"),
    }
}

fn internal(detail: &str) -> CodecError {
    CodecError::Malformed {
        format: JPEG,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Adobe CMYK inversion rule (spec §10.3): stored = 255 - ink, so
    /// the adapter recovers ink = 255 - stored. A hand-built stored-sample
    /// run verifies the decode-side math directly (no real corpus needed).
    #[test]
    fn cmyk_inversion_is_complement() {
        // stored: full white-paper (no ink) Adobe-stores all 255s.
        let stored = [255u8, 255, 255, 255];
        let ink = cmyk_from_stored(&stored, 1);
        assert_eq!(ink, [0, 0, 0, 0], "stored 255 ⇒ zero ink");

        // stored 0 ⇒ full ink 255 on every channel.
        let ink = cmyk_from_stored(&[0, 0, 0, 0], 1);
        assert_eq!(ink, [255, 255, 255, 255]);

        // A mixed pixel round-trips by complement.
        let ink = cmyk_from_stored(&[200, 100, 50, 0], 1);
        assert_eq!(ink, [55, 155, 205, 255]);
    }

    /// YCCK: a stored pixel that is neutral grey in YCbCr (Cb=Cr=128)
    /// decodes its CMY from luma; K is inverted independently. Y=255 (max
    /// luma) ⇒ stored CMY = 255 ⇒ ink 0; Y=0 ⇒ ink 255.
    #[test]
    fn ycck_neutral_luma_to_ink() {
        // Y=255, neutral chroma, K stored 255 ⇒ all-zero ink.
        let out = cmyk_from_ycck(&[255, 128, 128, 255], 1);
        assert_eq!(out, [0, 0, 0, 0]);
        // Y=0 ⇒ stored CMY 0 ⇒ ink 255; K stored 0 ⇒ K ink 255.
        let out = cmyk_from_ycck(&[0, 128, 128, 0], 1);
        assert_eq!(out, [255, 255, 255, 255]);
    }

    #[test]
    fn ycbcr_to_rgb_neutral() {
        // Neutral chroma ⇒ R=G=B=Y.
        assert_eq!(ycbcr_to_rgb(128, 128, 128), (128, 128, 128));
        assert_eq!(ycbcr_to_rgb(0, 128, 128), (0, 0, 0));
        assert_eq!(ycbcr_to_rgb(255, 128, 128), (255, 255, 255));
    }
}
