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

//! PNG adapter conformance (D-4 winner: zune-png). Encode→decode
//! round-trips through the `ImageSource`/`ImageTarget` adapters for the
//! spec `ChannelLayout`s PNG covers, probe sanity, and the M0
//! whole-decode-then-window invariant (a window read equals the
//! corresponding sub-rectangle of a full read).
//!
//! Feature linkage: image.codec.png (registry/codecs.yaml). Fn names
//! carry the underscored feature id per the repo test-tag convention
//! (CLAUDE.md §2) until the `#[feature_test]` macro ships from state.

use image_codecs::{
    ImageSource, ImageTarget, MemoryByteSource, PngSource, PngTarget, SourceInfo, TargetInfo,
};
use image_core::{
    AlphaMode, ChannelLayout, ColorSpaceRef, NamedSpace, PixelFormat, Region, SampleDepth,
    TileSliceMut, TileSliceRef, Transfer, TransferCurve,
};

/// The exact format the adapter decodes/encodes at M0 for a given layout.
fn fmt(channels: ChannelLayout) -> PixelFormat {
    PixelFormat {
        channels,
        depth: SampleDepth::U8,
        alpha: AlphaMode::Straight,
        transfer: Transfer::Gamma(TransferCurve::Srgb),
        space: ColorSpaceRef::Named(NamedSpace::Srgb),
    }
}

/// Deterministic synthetic pixels: a per-channel gradient that exercises
/// every byte position so a filter/round-trip mistake shows up.
fn synth(width: u32, height: u32, channels: ChannelLayout) -> Vec<u8> {
    let n = channels.count() as usize;
    let mut px = vec![0u8; width as usize * height as usize * n];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let base = (y * width as usize + x) * n;
            for c in 0..n {
                // Mix x, y, channel index so no two channels share a ramp.
                px[base + c] = ((x * 7 + y * 13 + c * 53) & 0xff) as u8;
            }
        }
    }
    px
}

/// Encode a packed full-frame buffer through `PngTarget` (single strip
/// for the simple cases; multi-strip exercised separately).
fn encode(width: u32, height: u32, channels: ChannelLayout, pixels: &[u8]) -> Vec<u8> {
    let format = fmt(channels);
    let mut target = PngTarget::new();
    target
        .begin(TargetInfo {
            width,
            height,
            format,
            icc: None,
        })
        .unwrap();
    let bpp = format.bytes_per_pixel();
    let region = Region::new(0, 0, width, height);
    let slice = TileSliceRef {
        region,
        format,
        bytes: pixels,
        row_stride: width as usize * bpp,
    };
    target.write_strip(region, &slice).unwrap();
    let stats = target.finish().unwrap();
    let bytes = target.into_bytes();
    assert_eq!(stats.bytes_written as usize, bytes.len());
    bytes
}

/// Full decode of `png` through `PngSource` into a packed buffer.
fn decode_full(png: Vec<u8>) -> (SourceInfo, Vec<u8>) {
    let mut src = PngSource::new(MemoryByteSource::new(png.into_boxed_slice()));
    let info = src.probe().unwrap();
    let bpp = info.format.bytes_per_pixel();
    let region = Region::new(0, 0, info.width, info.height);
    let mut buf = vec![0u8; info.width as usize * info.height as usize * bpp];
    let mut out = TileSliceMut {
        region,
        format: info.format,
        row_stride: info.width as usize * bpp,
        bytes: &mut buf,
    };
    src.read_region(region, 1, &mut out).unwrap();
    (info, buf)
}

#[test]
fn image_codec_png_roundtrip_rgba8() {
    let (w, h, ch) = (37u32, 21u32, ChannelLayout::Rgba);
    let pixels = synth(w, h, ch);
    let (info, decoded) = decode_full(encode(w, h, ch, &pixels));
    assert_eq!(info.width, w);
    assert_eq!(info.height, h);
    assert_eq!(info.format, fmt(ch));
    assert_eq!(info.native_format, "rgba8");
    // PNG is lossless: round-trip is byte-exact.
    assert_eq!(decoded, pixels);
}

#[test]
fn image_codec_png_roundtrip_gray8() {
    let (w, h, ch) = (33u32, 19u32, ChannelLayout::Gray);
    let pixels = synth(w, h, ch);
    let (info, decoded) = decode_full(encode(w, h, ch, &pixels));
    assert_eq!(info.format, fmt(ch));
    assert_eq!(info.native_format, "gray8");
    assert_eq!(decoded, pixels);
}

#[test]
fn image_codec_png_roundtrip_graya8() {
    let (w, h, ch) = (29u32, 23u32, ChannelLayout::GrayA);
    let pixels = synth(w, h, ch);
    let (info, decoded) = decode_full(encode(w, h, ch, &pixels));
    assert_eq!(info.format, fmt(ch));
    assert_eq!(info.native_format, "graya8");
    assert_eq!(decoded, pixels);
}

/// RGB8: PNG's 3-channel RGB is not in the spec `ChannelLayout` set, so
/// the adapter widens it to `Rgba` with alpha synthesised at 255 and
/// records the native truth. We build a real RGB PNG with zune's encoder
/// (our `PngTarget` only emits the 4-channel layouts) and assert the
/// decode side widens correctly.
#[test]
fn image_codec_png_roundtrip_rgb8() {
    use zune_core::bit_depth::BitDepth;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::EncoderOptions;
    use zune_png::PngEncoder;

    let (w, h) = (24u32, 18u32);
    // 3-channel source pixels (a different ramp per channel).
    let mut rgb = vec![0u8; w as usize * h as usize * 3];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let b = (y * w as usize + x) * 3;
            rgb[b] = (x * 9) as u8;
            rgb[b + 1] = (y * 11) as u8;
            rgb[b + 2] = (x + y) as u8;
        }
    }
    let opts = EncoderOptions::new(w as usize, h as usize, ColorSpace::RGB, BitDepth::Eight);
    let mut png = Vec::new();
    PngEncoder::new(&rgb, opts).encode(&mut png).unwrap();

    let (info, decoded) = decode_full(png);
    assert_eq!(info.format, fmt(ChannelLayout::Rgba));
    assert_eq!(info.native_format, "rgb8");
    // RGB widened to RGBA: colour channels preserved, alpha == 255.
    for i in 0..(w as usize * h as usize) {
        assert_eq!(&decoded[i * 4..i * 4 + 3], &rgb[i * 3..i * 3 + 3]);
        assert_eq!(decoded[i * 4 + 3], 255, "synthesised alpha at pixel {i}");
    }
}

#[test]
fn image_codec_png_probe_dimensions() {
    let png = encode(
        64,
        48,
        ChannelLayout::Rgba,
        &synth(64, 48, ChannelLayout::Rgba),
    );
    let mut src = PngSource::new(MemoryByteSource::new(png.into_boxed_slice()));
    let info = src.probe().unwrap();
    assert_eq!((info.width, info.height), (64, 48));
    assert_eq!(src.native_shrink(), &[1]);
    // probe is idempotent (header parse is memoised).
    let again = src.probe().unwrap();
    assert_eq!((again.width, again.height), (64, 48));
}

/// A window read must equal the matching sub-rectangle of a full read —
/// the M0 whole-decode-then-serve-windows invariant.
#[test]
fn image_codec_png_window_matches_full() {
    let (w, h, ch) = (50u32, 40u32, ChannelLayout::Rgba);
    let pixels = synth(w, h, ch);
    let png = encode(w, h, ch, &pixels);

    let (info, full) = decode_full(png.clone());
    let bpp = info.format.bytes_per_pixel();

    let mut src = PngSource::new(MemoryByteSource::new(png.into_boxed_slice()));
    src.probe().unwrap();

    for roi in [
        Region::new(0, 0, 10, 10),
        Region::new(13, 7, 17, 22),
        Region::new(w as i32 - 1, h as i32 - 1, 1, 1),
        Region::new(0, 0, w, h),
    ] {
        let mut buf = vec![0u8; roi.w as usize * roi.h as usize * bpp];
        let mut out = TileSliceMut {
            region: roi,
            format: info.format,
            row_stride: roi.w as usize * bpp,
            bytes: &mut buf,
        };
        src.read_region(roi, 1, &mut out).unwrap();

        // Compare against the same rectangle carved from the full decode.
        for row in 0..roi.h as usize {
            let sy = ((roi.y as usize + row) * w as usize + roi.x as usize) * bpp;
            let dy = row * roi.w as usize * bpp;
            let rb = roi.w as usize * bpp;
            assert_eq!(
                &buf[dy..dy + rb],
                &full[sy..sy + rb],
                "window {roi:?} row {row} mismatch"
            );
        }
    }
}

/// Multi-strip encode: strips arriving top-to-bottom must produce the
/// same image as a single full-frame strip.
#[test]
fn image_codec_png_multistrip_encode() {
    let (w, h, ch) = (40u32, 30u32, ChannelLayout::Rgba);
    let format = fmt(ch);
    let bpp = format.bytes_per_pixel();
    let pixels = synth(w, h, ch);

    let mut target = PngTarget::new();
    target
        .begin(TargetInfo {
            width: w,
            height: h,
            format,
            icc: None,
        })
        .unwrap();
    // Strips of 7 rows, last one short.
    let mut y = 0u32;
    while y < h {
        let sh = (h - y).min(7);
        let region = Region::new(0, y as i32, w, sh);
        let start = y as usize * w as usize * bpp;
        let end = start + sh as usize * w as usize * bpp;
        let slice = TileSliceRef {
            region,
            format,
            bytes: &pixels[start..end],
            row_stride: w as usize * bpp,
        };
        target.write_strip(region, &slice).unwrap();
        y += sh;
    }
    target.finish().unwrap();
    let png = target.into_bytes();

    let (_, decoded) = decode_full(png);
    assert_eq!(decoded, pixels);
}

/// Out-of-order / gapped strips are a sequencing error, not silent
/// corruption.
#[test]
fn image_codec_png_strip_out_of_order_errors() {
    let (w, h, ch) = (16u32, 16u32, ChannelLayout::Rgba);
    let format = fmt(ch);
    let bpp = format.bytes_per_pixel();
    let pixels = synth(w, h, ch);

    let mut target = PngTarget::new();
    target
        .begin(TargetInfo {
            width: w,
            height: h,
            format,
            icc: None,
        })
        .unwrap();
    // Jump straight to row 8 — gap from row 0.
    let region = Region::new(0, 8, w, 8);
    let start = 8 * w as usize * bpp;
    let slice = TileSliceRef {
        region,
        format,
        bytes: &pixels[start..],
        row_stride: w as usize * bpp,
    };
    assert!(target.write_strip(region, &slice).is_err());
}
