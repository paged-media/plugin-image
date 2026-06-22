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

//! JPEG adapter conformance (decode: zune-jpeg; encode: jpeg-encoder).
//! JPEG is **lossy**, so round-trips assert a mean-absolute-error bound
//! on a smooth gradient (the worst case for blocking is high-frequency
//! content, which we avoid for the bound), plus probe/dimension sanity
//! and the M0 whole-decode-then-window invariant. CMYK round-trip is NOT
//! exercised — jpeg-encoder here only emits RGB/Gray and we cannot
//! synthesize a CMYK JPEG through the adapter; the Adobe APP14 inversion
//! math is unit-tested directly in `image-codecs/src/jpeg/decode.rs`
//! (a hand-built stored-sample fixture). A real-CMYK-corpus decode test
//! is a needs-real-corpus follow-up (spec §10.3 corpus rule).
//!
//! Feature linkage: image.codec.jpeg (registry/codecs.yaml). Fn names
//! carry the underscored feature id per the repo test-tag convention
//! (CLAUDE.md §2) until the `#[feature_test]` macro ships from state.

use image_codecs::{
    ImageSource, ImageTarget, JpegSource, JpegTarget, MemoryByteSource, SourceInfo, TargetInfo,
};
use image_core::{
    AlphaMode, ChannelLayout, ColorSpaceRef, NamedSpace, PixelFormat, Region, SampleDepth,
    TileSliceMut, TileSliceRef, Transfer, TransferCurve,
};

/// The exact format the adapter decodes/encodes for a given layout (M0:
/// U8, sRGB-encoded, straight alpha).
fn fmt(channels: ChannelLayout) -> PixelFormat {
    PixelFormat {
        channels,
        depth: SampleDepth::U8,
        alpha: AlphaMode::Straight,
        transfer: Transfer::Gamma(TransferCurve::Srgb),
        space: ColorSpaceRef::Named(NamedSpace::Srgb),
    }
}

/// A smooth per-channel gradient — low spatial frequency so JPEG's DCT +
/// 4:2:0 chroma subsampling stays well inside a tight error bound. Alpha
/// (channel 3 of Rgba) is forced opaque since JPEG drops it on encode and
/// the decode synthesises 255.
fn gradient(width: u32, height: u32, channels: ChannelLayout) -> Vec<u8> {
    let n = channels.count() as usize;
    let mut px = vec![0u8; width as usize * height as usize * n];
    let (w, h) = (width.max(1) as f32, height.max(1) as f32);
    for y in 0..height as usize {
        for x in 0..width as usize {
            let base = (y * width as usize + x) * n;
            let gx = x as f32 / w;
            let gy = y as f32 / h;
            for c in 0..n {
                if channels == ChannelLayout::Rgba && c == 3 {
                    px[base + c] = 255; // opaque alpha
                    continue;
                }
                // Smooth ramp, phase-shifted per channel.
                let t = (gx * 0.6 + gy * 0.4 + c as f32 * 0.13).fract();
                px[base + c] = (t * 255.0).round() as u8;
            }
        }
    }
    px
}

fn encode(width: u32, height: u32, channels: ChannelLayout, pixels: &[u8], quality: u8) -> Vec<u8> {
    let format = fmt(channels);
    let mut target = JpegTarget::new(quality);
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
    assert!(!bytes.is_empty(), "encoder produced no bytes");
    bytes
}

fn decode_full(jpeg: Vec<u8>) -> (SourceInfo, Vec<u8>) {
    let mut src = JpegSource::new(MemoryByteSource::new(jpeg.into_boxed_slice()));
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

/// Mean absolute error per sample, ignoring synthesized alpha (channel 3
/// of an Rgba layout, which JPEG does not carry).
fn mean_abs_err(a: &[u8], b: &[u8], channels: ChannelLayout) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = channels.count() as usize;
    let mut sum = 0u64;
    let mut count = 0u64;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if channels == ChannelLayout::Rgba && i % n == 3 {
            continue; // alpha is synthesized, not a codec-fidelity sample
        }
        sum += (x as i32 - y as i32).unsigned_abs() as u64;
        count += 1;
    }
    sum as f64 / count as f64
}

/// RGB round-trip: encode a smooth gradient at quality 90, decode, and
/// assert the mean absolute error is under 4/255 per sample (JPEG is
/// lossy — this is a PSNR-style bound, not byte-exact).
#[test]
fn image_codec_jpeg_roundtrip_rgb_gradient_mae() {
    let (w, h, ch) = (128u32, 96u32, ChannelLayout::Rgba);
    let pixels = gradient(w, h, ch);
    let (info, decoded) = decode_full(encode(w, h, ch, &pixels, 90));

    assert_eq!(info.width, w);
    assert_eq!(info.height, h);
    assert_eq!(info.format, fmt(ch));
    // jpeg-encoder writes a 3-component (YCbCr-from-RGB) baseline JPEG; the
    // decoder reports the YCbCr native truth, widened to Rgba.
    assert_eq!(info.native_format, "ycbcr8");

    // Measured: MAE ~1.64/255 at q90 on this 128x96 smooth gradient.
    let mae = mean_abs_err(&decoded, &pixels, ch);
    assert!(mae < 4.0, "RGB gradient MAE {mae} >= 4/255 at q90");

    // Synthesized alpha is opaque everywhere.
    for i in 0..(w as usize * h as usize) {
        assert_eq!(decoded[i * 4 + 3], 255, "alpha at pixel {i}");
    }
}

/// Gray round-trip: same bound on a single-channel gradient.
#[test]
fn image_codec_jpeg_roundtrip_gray_gradient_mae() {
    let (w, h, ch) = (100u32, 80u32, ChannelLayout::Gray);
    let pixels = gradient(w, h, ch);
    let (info, decoded) = decode_full(encode(w, h, ch, &pixels, 90));

    assert_eq!((info.width, info.height), (w, h));
    assert_eq!(info.format, fmt(ch));
    assert_eq!(info.native_format, "gray8");

    // Measured: MAE ~0.18/255 at q90 on this 100x80 smooth gradient.
    let mae = mean_abs_err(&decoded, &pixels, ch);
    assert!(mae < 4.0, "Gray gradient MAE {mae} >= 4/255 at q90");
}

/// Higher quality must not increase error vs lower quality on the same
/// content — a monotonicity sanity check that the quality knob is wired.
#[test]
fn image_codec_jpeg_quality_reduces_error() {
    let (w, h, ch) = (96u32, 96u32, ChannelLayout::Rgba);
    let pixels = gradient(w, h, ch);

    let (_, lo) = decode_full(encode(w, h, ch, &pixels, 50));
    let (_, hi) = decode_full(encode(w, h, ch, &pixels, 95));

    // Measured: q50 MAE ~3.52/255, q95 MAE ~1.48/255 (higher q ⇒ lower err).
    let mae_lo = mean_abs_err(&lo, &pixels, ch);
    let mae_hi = mean_abs_err(&hi, &pixels, ch);
    assert!(
        mae_hi <= mae_lo + 1e-9,
        "q95 MAE {mae_hi} should not exceed q50 MAE {mae_lo}"
    );
}

#[test]
fn image_codec_jpeg_probe_dimensions() {
    let jpeg = encode(
        64,
        48,
        ChannelLayout::Rgba,
        &gradient(64, 48, ChannelLayout::Rgba),
        85,
    );
    let mut src = JpegSource::new(MemoryByteSource::new(jpeg.into_boxed_slice()));
    let info = src.probe().unwrap();
    assert_eq!((info.width, info.height), (64, 48));
    // No DCT-scaled decode → no native downscale.
    assert_eq!(src.native_shrink(), &[1]);
    // probe is idempotent (header parse is memoised).
    let again = src.probe().unwrap();
    assert_eq!((again.width, again.height), (64, 48));
}

/// A window read must equal the matching sub-rectangle of a full read —
/// the M0 whole-decode-then-serve-windows invariant. The decode itself is
/// lossy but deterministic, so window == sub-rectangle holds *exactly*.
#[test]
fn image_codec_jpeg_window_matches_full() {
    let (w, h, ch) = (120u32, 90u32, ChannelLayout::Rgba);
    let pixels = gradient(w, h, ch);
    let jpeg = encode(w, h, ch, &pixels, 88);

    let (info, full) = decode_full(jpeg.clone());
    let bpp = info.format.bytes_per_pixel();

    let mut src = JpegSource::new(MemoryByteSource::new(jpeg.into_boxed_slice()));
    src.probe().unwrap();

    for roi in [
        Region::new(0, 0, 10, 10),
        Region::new(33, 17, 40, 50),
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

/// Multi-strip encode must produce the same image as a single full-frame
/// strip (decode is deterministic, so the two encodes are byte-identical
/// when fed identical pixels).
#[test]
fn image_codec_jpeg_multistrip_encode() {
    let (w, h, ch) = (80u32, 64u32, ChannelLayout::Rgba);
    let format = fmt(ch);
    let bpp = format.bytes_per_pixel();
    let pixels = gradient(w, h, ch);

    let mut target = JpegTarget::new(90);
    target
        .begin(TargetInfo {
            width: w,
            height: h,
            format,
            icc: None,
        })
        .unwrap();
    let mut y = 0u32;
    while y < h {
        let sh = (h - y).min(11);
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
    let multi = target.into_bytes();

    let single = encode(w, h, ch, &pixels, 90);
    assert_eq!(
        multi, single,
        "multi-strip encode differs from single-strip"
    );
}

/// Out-of-order / gapped strips are a sequencing error.
#[test]
fn image_codec_jpeg_strip_out_of_order_errors() {
    let (w, h, ch) = (32u32, 32u32, ChannelLayout::Rgba);
    let format = fmt(ch);
    let bpp = format.bytes_per_pixel();
    let pixels = gradient(w, h, ch);

    let mut target = JpegTarget::new(90);
    target
        .begin(TargetInfo {
            width: w,
            height: h,
            format,
            icc: None,
        })
        .unwrap();
    let region = Region::new(0, 16, w, 16); // gap from row 0
    let start = 16 * w as usize * bpp;
    let slice = TileSliceRef {
        region,
        format,
        bytes: &pixels[start..],
        row_stride: w as usize * bpp,
    };
    assert!(target.write_strip(region, &slice).is_err());
}

/// CMYK and GrayA have no M0 JPEG encode path → clean `Unsupported`.
#[test]
fn image_codec_jpeg_encode_rejects_cmyk() {
    let mut target = JpegTarget::new(90);
    let err = target.begin(TargetInfo {
        width: 8,
        height: 8,
        format: fmt(ChannelLayout::Cmyk),
        icc: None,
    });
    assert!(err.is_err(), "CMYK encode must be Unsupported");
}
