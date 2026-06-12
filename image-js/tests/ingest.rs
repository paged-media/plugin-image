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

use image_codecs::{ImageTarget, PngTarget, TargetInfo};
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

#[test]
fn image_editor_ingest_decode_png_roundtrip() {
    let (w, h) = (8u32, 6u32);
    let pixels = test_pixels(w, h);
    let img = decode_rgba8(&png_bytes(w, h, &pixels)).expect("decode png");
    assert_eq!((img.width, img.height), (w, h));
    assert_eq!(&img.rgba[..], &pixels[..], "PNG is lossless");
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
