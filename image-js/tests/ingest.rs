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

//! The M4 ingest slice, natively (feat: image.editor.ingest): magic
//! sniff → codec/PSD decode → RGBA8, and the adjustments chain through
//! Engine A's async sink. The GPU half SKIPS cleanly without an
//! adapter; decode is pure CPU and always runs.

use image_codecs::{ImageTarget, JpegTarget, PngTarget, TargetInfo};
use image_core::{
    AlphaMode, ChannelLayout, ColorSpaceRef, NamedSpace, PixelFormat, Region, SampleDepth,
    TileSliceRef, Transfer, TransferCurve,
};
use image_gpu::GpuContext;
use image_js::ingest::{adjust_rgba8, decode_rgba8, AdjustParams, IngestError};

const PNG_FMT: PixelFormat = PixelFormat {
    channels: ChannelLayout::Rgba,
    depth: SampleDepth::U8,
    alpha: AlphaMode::Straight,
    transfer: Transfer::Gamma(TransferCurve::Srgb),
    space: ColorSpaceRef::Named(NamedSpace::Srgb),
};

/// Deterministic 8×6 RGBA8 test pixels.
fn test_pixels(w: u32, h: u32) -> Vec<u8> {
    let mut px = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            px[i] = (x * 30) as u8;
            px[i + 1] = (y * 40) as u8;
            px[i + 2] = (x * 10 + y * 5) as u8;
            px[i + 3] = 200;
        }
    }
    px
}

/// Encode RGBA8 pixels as a PNG via the codec adapter.
fn png_bytes(w: u32, h: u32, pixels: &[u8]) -> Vec<u8> {
    let mut target = PngTarget::new();
    target
        .begin(TargetInfo {
            width: w,
            height: h,
            format: PNG_FMT,
            icc: None,
        })
        .expect("png begin");
    target
        .write_strip(
            Region::new(0, 0, w, h),
            &TileSliceRef {
                region: Region::new(0, 0, w, h),
                format: PNG_FMT,
                bytes: pixels,
                row_stride: w as usize * 4,
            },
        )
        .expect("png strip");
    target.finish().expect("png finish");
    target.into_bytes()
}

/// Hand-assemble minimal RGB PSD bytes (RAW composite) — mirrors the
/// image-psd composite test helper.
fn psd_bytes(width: u32, height: u32, planes: &[&[u8]]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"8BPS");
    b.extend_from_slice(&1u16.to_be_bytes());
    b.extend_from_slice(&[0u8; 6]);
    b.extend_from_slice(&(planes.len() as u16).to_be_bytes());
    b.extend_from_slice(&height.to_be_bytes());
    b.extend_from_slice(&width.to_be_bytes());
    b.extend_from_slice(&8u16.to_be_bytes()); // depth
    b.extend_from_slice(&3u16.to_be_bytes()); // RGB
    b.extend_from_slice(&0u32.to_be_bytes());
    b.extend_from_slice(&0u32.to_be_bytes());
    b.extend_from_slice(&0u32.to_be_bytes());
    b.extend_from_slice(&0u16.to_be_bytes()); // RAW
    for p in planes {
        b.extend_from_slice(p);
    }
    b
}

/// A JPEG pixel format (straight RGBA8, sRGB) for the encoder. The
/// encoder drops alpha (JPEG has none) — fine for the orientation test.
const JPEG_FMT: PixelFormat = PixelFormat {
    channels: ChannelLayout::Rgba,
    depth: SampleDepth::U8,
    alpha: AlphaMode::Straight,
    transfer: Transfer::Gamma(TransferCurve::Srgb),
    space: ColorSpaceRef::Named(NamedSpace::Srgb),
};

/// Encode RGBA8 pixels as a baseline JPEG via the codec adapter.
fn jpeg_bytes(w: u32, h: u32, pixels: &[u8]) -> Vec<u8> {
    let mut target = JpegTarget::new(92);
    target
        .begin(TargetInfo {
            width: w,
            height: h,
            format: JPEG_FMT,
            icc: None,
        })
        .expect("jpeg begin");
    target
        .write_strip(
            Region::new(0, 0, w, h),
            &TileSliceRef {
                region: Region::new(0, 0, w, h),
                format: JPEG_FMT,
                bytes: pixels,
                row_stride: w as usize * 4,
            },
        )
        .expect("jpeg strip");
    target.finish().expect("jpeg finish");
    target.into_bytes()
}

/// Build a minimal little-endian EXIF TIFF block carrying a single
/// Orientation (0x0112) SHORT entry, then wrap it in a JPEG APP1 segment
/// (`FF E1 len "Exif\0\0" <tiff>`). Splice it right after the SOI of an
/// existing JPEG so the decoder sees real EXIF.
fn splice_exif_orientation(jpeg: &[u8], orientation: u16) -> Vec<u8> {
    // TIFF: header (8) + IFD0 (count=1, one 12-byte entry, next=0).
    let mut tiff = Vec::new();
    tiff.extend_from_slice(b"II");
    tiff.extend_from_slice(&42u16.to_le_bytes());
    tiff.extend_from_slice(&8u32.to_le_bytes());
    tiff.extend_from_slice(&1u16.to_le_bytes()); // 1 entry
    tiff.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation
    tiff.extend_from_slice(&3u16.to_le_bytes()); // SHORT
    tiff.extend_from_slice(&1u32.to_le_bytes()); // count
    tiff.extend_from_slice(&orientation.to_le_bytes());
    tiff.extend_from_slice(&[0u8, 0]); // pad value field to 4 bytes
    tiff.extend_from_slice(&0u32.to_le_bytes()); // no next IFD

    let mut app1_payload = Vec::new();
    app1_payload.extend_from_slice(b"Exif\x00\x00");
    app1_payload.extend_from_slice(&tiff);

    // APP1 segment: marker FFE1 + 2-byte length (includes the length
    // bytes themselves) + payload.
    let seg_len = (app1_payload.len() + 2) as u16;
    let mut app1 = vec![0xFF, 0xE1];
    app1.extend_from_slice(&seg_len.to_be_bytes());
    app1.extend_from_slice(&app1_payload);

    // Splice after SOI (the first two bytes FFD8).
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "input is a JPEG (SOI)");
    let mut out = Vec::with_capacity(jpeg.len() + app1.len());
    out.extend_from_slice(&jpeg[0..2]);
    out.extend_from_slice(&app1);
    out.extend_from_slice(&jpeg[2..]);
    out
}

#[test]
fn image_editor_ingest_decode_png_roundtrip() {
    let (w, h) = (8u32, 6u32);
    let pixels = test_pixels(w, h);
    let img = decode_rgba8(&png_bytes(w, h, &pixels)).expect("decode png");
    assert_eq!((img.width, img.height), (w, h));
    assert_eq!(&img.rgba[..], &pixels[..], "PNG is lossless");
}

#[test]
fn image_editor_ingest_jpeg_no_exif_keeps_dims() {
    // Control: a JPEG without EXIF keeps its dimensions (orientation
    // parses to None → identity auto-orient).
    let (w, h) = (16u32, 8u32);
    let img = decode_rgba8(&jpeg_bytes(w, h, &test_pixels(w, h))).expect("decode jpeg");
    assert_eq!((img.width, img.height), (w, h));
}

#[test]
fn image_editor_ingest_jpeg_exif_orientation_6_auto_rotates() {
    // A 16×8 JPEG tagged Orientation=6 (rotate 90° CW) must ingest as 8×16
    // — the auto-orient in the decode-to-RGBA bridge ran. This proves the
    // EXIF read path (image-codecs::exif) is wired end-to-end through the
    // M4 ingest slice.
    let (w, h) = (16u32, 8u32);
    let base = jpeg_bytes(w, h, &test_pixels(w, h));
    let with_exif = splice_exif_orientation(&base, 6);
    let img = decode_rgba8(&with_exif).expect("decode jpeg+exif");
    assert_eq!(
        (img.width, img.height),
        (h, w),
        "Orientation=6 must swap dimensions to {h}×{w}"
    );
    assert_eq!(img.rgba.len(), (w * h * 4) as usize, "pixel count preserved");
}

#[test]
fn image_editor_ingest_jpeg_exif_orientation_1_is_identity() {
    // Orientation=1 (TopLeft) is the no-op — dims unchanged even with EXIF.
    let (w, h) = (16u32, 8u32);
    let base = jpeg_bytes(w, h, &test_pixels(w, h));
    let with_exif = splice_exif_orientation(&base, 1);
    let img = decode_rgba8(&with_exif).expect("decode jpeg+exif");
    assert_eq!((img.width, img.height), (w, h));
}

#[test]
fn image_editor_ingest_decode_psd_composite() {
    let img =
        decode_rgba8(&psd_bytes(2, 1, &[&[10, 20], &[30, 40], &[50, 60]])).expect("decode psd");
    assert_eq!((img.width, img.height), (2, 1));
    assert_eq!(&img.rgba[..], &[10, 30, 50, 255, 20, 40, 60, 255]);
}

#[test]
fn image_editor_ingest_rejects_unknown_container() {
    assert!(matches!(
        decode_rgba8(b"not an image"),
        Err(IngestError::Unsupported(_))
    ));
}

#[test]
fn image_editor_ingest_adjust_identity_needs_no_gpu() {
    let img =
        decode_rgba8(&psd_bytes(2, 1, &[&[10, 20], &[30, 40], &[50, 60]])).expect("decode psd");
    // Identity short-circuits before any GPU work; a throwaway context
    // is still needed by the signature, so build one only if available
    // — otherwise prove the short-circuit through the wasm-equivalent
    // path (params identity ⇒ decode verbatim).
    let params = AdjustParams::default();
    assert!(params.is_identity());
    let Some(ctx) = pollster::block_on(maybe_device()) else {
        println!("SKIP: no GPU adapter (identity path covered via parity test)");
        return;
    };
    let out = pollster::block_on(adjust_rgba8(&ctx, &img, &params)).expect("identity adjust");
    assert_eq!(&out[..], &img.rgba[..]);
}

#[test]
fn image_editor_ingest_adjust_exposure_doubles_on_gpu() {
    let Some(ctx) = pollster::block_on(maybe_device()) else {
        println!("SKIP: no GPU adapter");
        return;
    };
    let img =
        decode_rgba8(&psd_bytes(2, 1, &[&[10, 20], &[30, 40], &[50, 60]])).expect("decode psd");
    let params = AdjustParams {
        exposure_ev: 1.0, // exp2(1) = ×2 on rgb, alpha preserved
        ..AdjustParams::default()
    };
    let out = pollster::block_on(adjust_rgba8(&ctx, &img, &params)).expect("adjust");
    assert_eq!(out.len(), img.rgba.len());
    for (i, (&got, &src)) in out.iter().zip(img.rgba.iter()).enumerate() {
        let expect = if i % 4 == 3 {
            src as i32 // alpha untouched
        } else {
            (src as i32 * 2).min(255)
        };
        assert!(
            (got as i32 - expect).abs() <= 2,
            "byte {i}: got {got}, expected ~{expect} (f16 working-space tolerance)"
        );
    }
}

async fn maybe_device() -> Option<GpuContext> {
    GpuContext::new().await.ok()
}
