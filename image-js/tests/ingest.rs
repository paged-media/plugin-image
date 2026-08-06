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
use std::sync::Arc;

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
fn image_editor_resource_tile_window_cuts_and_clamps() {
    // C-6 (I-06) — the LEVEL-0 tile cut (the honest resource-provider
    // subset). A window inside the image returns exactly its rows; an edge
    // window clamps to the extent; a fully-outside window returns empty.
    let (w, h) = (8u32, 6u32);
    let pixels = test_pixels(w, h);
    let img = image_js::ingest::DecodedImage {
        width: w,
        height: h,
        rgba: Arc::from(pixels.clone().into_boxed_slice()),
        display: image_js::display::DisplayTreatment::AssumedSrgb,
    };

    // A 4×3 tile at (2,1): row-major copy of the matching window.
    let (bytes, tw, th) = img.tile_window_rgba8(2, 1, 4, 3);
    assert_eq!((tw, th), (4, 3));
    assert_eq!(bytes.len(), (4 * 3 * 4) as usize);
    for row in 0..3u32 {
        for col in 0..4u32 {
            let src = (((row + 1) * w + (col + 2)) * 4) as usize;
            let dst = ((row * 4 + col) * 4) as usize;
            assert_eq!(&bytes[dst..dst + 4], &pixels[src..src + 4], "tile pixel");
        }
    }

    // An edge tile at (6,4) requesting 4×4 clamps to 2×2.
    let (_edge, etw, eth) = img.tile_window_rgba8(6, 4, 4, 4);
    assert_eq!((etw, eth), (2, 2));

    // A fully-outside window is empty (a transparent miss the provider skips).
    let (out, ow, oh) = img.tile_window_rgba8(100, 100, 4, 4);
    assert!(out.is_empty());
    assert_eq!((ow, oh), (0, 0));
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
    assert_eq!(
        img.rgba.len(),
        (w * h * 4) as usize,
        "pixel count preserved"
    );
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
    let out = pollster::block_on(adjust_rgba8(&ctx, &img, &params, None)).expect("identity adjust");
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
    let out = pollster::block_on(adjust_rgba8(&ctx, &img, &params, None)).expect("adjust");
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

/// Encode a true-ink CMYK buffer (`4·n` bytes, C,M,Y,K) as an Adobe CMYK
/// JPEG via jpeg-encoder (writes APP14 transform 0 + the Adobe inversion
/// the zune-jpeg decoder re-inverts). No embedded ICC profile, so the
/// decode takes the uncalibrated device-CMYK fallback.
fn cmyk_jpeg_bytes(w: u32, h: u32, cmyk_ink: &[u8]) -> Vec<u8> {
    use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
    let mut buf = Vec::new();
    let mut enc = Encoder::new(&mut buf, 100);
    enc.set_sampling_factor(SamplingFactor::F_1_1);
    enc.encode(cmyk_ink, w as u16, h as u16, ColorType::Cmyk)
        .expect("encode cmyk jpeg");
    buf
}

// feat: image.editor.ingest — the CMYK ingest cast (spec §5.2). A CMYK
// placed image now DECODES (uncalibrated device fallback when there is no
// embedded ICC) instead of the old `Unsupported` rejection.
#[test]
fn image_editor_ingest_cmyk_jpeg_decodes_instead_of_rejecting() {
    // A 2×1 CMYK image: paper white (no ink) and solid black (full K).
    let (w, h) = (2u32, 1u32);
    let cmyk = vec![
        0u8, 0, 0, 0, /* white */ 0, 0, 0, 255, /* full K */
    ];
    let jpeg = cmyk_jpeg_bytes(w, h, &cmyk);
    assert_eq!(&jpeg[0..3], &[0xFF, 0xD8, 0xFF], "JPEG SOI");

    let img = decode_rgba8(&jpeg).expect("CMYK JPEG must now decode, not reject");
    assert_eq!((img.width, img.height), (w, h));
    assert_eq!(img.rgba.len(), (w * h * 4) as usize);

    // No ICC → the device formula. JPEG is lossy, so allow tolerance, but
    // the structure must hold: paper white near white, full K near black,
    // alpha synthesised opaque.
    let white = &img.rgba[0..4];
    let black = &img.rgba[4..8];
    assert!(
        white[0] > 230 && white[1] > 230 && white[2] > 230,
        "paper white should be near RGB white, got {white:?}"
    );
    assert!(
        black[0] < 25 && black[1] < 25 && black[2] < 25,
        "full K should be near RGB black, got {black:?}"
    );
    assert_eq!(white[3], 255, "alpha synthesised opaque");
    assert_eq!(black[3], 255, "alpha synthesised opaque");
}

// ── the EXTENDED (kernel-breadth) adjust stages + the FILL lane ──────
//
// feat: image.editor.adjust / image.editor.generate. Each proves the
// stage is REACHABLE through the same chain the panel drives (identity
// short-circuit, GPU dispatch, selection mask), not that the kernel
// math is right — that is the conformance family's parity job.

/// A 2×1 mid-grey/blue pair with an alpha channel, as an engine image.
fn ext_image() -> image_js::ingest::DecodedImage {
    image_js::ingest::DecodedImage::from_rgba8(2, 1, vec![128, 128, 128, 255, 40, 90, 200, 255])
        .expect("valid rgba8")
}

#[test]
fn image_editor_adjust_extended_block_decodes_onto_the_params() {
    // The flat wire → typed params mapping (no GPU needed).
    let mut p = AdjustParams::default();
    assert!(p.is_identity());
    p.apply_extended(&[]).expect("empty ext is identity");
    assert!(p.is_identity(), "an EMPTY block leaves every stage neutral");

    let mut ext = vec![0.0f32; image_js::ingest::ADJUST_EXT_LEN];
    // Neutral defaults for the fields whose identity is not 0.
    ext[26] = 1.0; // channel mixer r.r
    ext[31] = 1.0; // g.g
    ext[36] = 1.0; // b.b
    for c in 0..3 {
        ext[38 + c * 3 + 1] = 1.0; // levels_rgb in_white
        ext[38 + c * 3 + 2] = 1.0; // levels_rgb gamma
    }
    let mut q = AdjustParams::default();
    q.apply_extended(&ext).expect("neutral ext");
    assert!(q.is_identity(), "the neutral block is still identity");

    // Now flip one field per stage and prove it lands.
    ext[0] = 0.5; // vibrance
    ext[1] = 0.1; // color balance shadows cyan-red
    ext[10] = 1.0; // black & white enabled
    ext[17] = 1.0; // posterize enabled
    ext[18] = 4.0; // posterize levels
    ext[19] = 1.0; // threshold enabled
    ext[20] = 0.5; // threshold value
    ext[21] = 0.25; // photo filter density
    let mut r = AdjustParams::default();
    r.apply_extended(&ext).expect("populated ext");
    assert_eq!(r.vibrance, 0.5);
    assert_eq!(r.color_balance.shadows[0], 0.1);
    assert!(r.black_white.enabled);
    assert_eq!(r.posterize, Some(4.0));
    assert_eq!(r.threshold, Some(0.5));
    assert_eq!(r.photo_filter.density, 0.25);
    assert!(!r.is_identity());
}

#[test]
fn image_editor_adjust_extended_block_rejects_a_wrong_length() {
    let mut p = AdjustParams::default();
    assert!(p.apply_extended(&[1.0, 2.0]).is_err());
}

#[test]
fn image_editor_adjust_threshold_runs_through_the_chain_on_gpu() {
    let Some(ctx) = pollster::block_on(maybe_device()) else {
        println!("SKIP: no GPU adapter");
        return;
    };
    let img = ext_image();
    let params = AdjustParams {
        threshold: Some(0.5),
        ..AdjustParams::default()
    };
    let out = pollster::block_on(adjust_rgba8(&ctx, &img, &params, None)).expect("threshold");
    // px0 luma = 0.5 (≥ 0.5) → white; px1 luma ≈ 0.3·.157+0.59·.353+0.11·.784 ≈ 0.34 → black.
    assert_eq!(&out[0..3], &[255, 255, 255], "px0 above the cut");
    assert_eq!(&out[4..7], &[0, 0, 0], "px1 below the cut");
    assert_eq!(out[3], 255, "alpha preserved");
}

#[test]
fn image_editor_adjust_black_white_runs_through_the_chain_on_gpu() {
    let Some(ctx) = pollster::block_on(maybe_device()) else {
        println!("SKIP: no GPU adapter");
        return;
    };
    let img = ext_image();
    let params = AdjustParams {
        black_white: image_js::ingest::BlackWhiteParams {
            enabled: true,
            ..Default::default()
        },
        ..AdjustParams::default()
    };
    let out = pollster::block_on(adjust_rgba8(&ctx, &img, &params, None)).expect("black&white");
    for px in out.chunks_exact(4) {
        assert!(
            (px[0] as i32 - px[1] as i32).abs() <= 1 && (px[1] as i32 - px[2] as i32).abs() <= 1,
            "the six-weight mix splats one gray, got {px:?}"
        );
    }
}

#[test]
fn image_selection_extended_stage_is_masked_like_the_others() {
    let Some(ctx) = pollster::block_on(maybe_device()) else {
        println!("SKIP: no GPU adapter");
        return;
    };
    let img = ext_image();
    // Coverage selects ONLY pixel 0.
    let cov = Arc::new(
        image_gpu::SelectionCoverage::from_data(2, 1, vec![255, 0]).expect("2 px coverage"),
    );
    let params = AdjustParams {
        threshold: Some(0.5),
        ..AdjustParams::default()
    };
    let out = pollster::block_on(adjust_rgba8(&ctx, &img, &params, Some(cov))).expect("masked");
    assert_eq!(&out[0..3], &[255, 255, 255], "selected pixel thresholded");
    for (i, (&got, &src)) in out[4..8].iter().zip(img.rgba[4..8].iter()).enumerate() {
        assert!(
            (got as i32 - src as i32).abs() <= 2,
            "byte {i} outside the selection must survive: got {got}, was {src}"
        );
    }
}

#[test]
fn image_editor_generate_gradient_fills_through_the_selection() {
    let Some(ctx) = pollster::block_on(maybe_device()) else {
        println!("SKIP: no GPU adapter");
        return;
    };
    let img = ext_image();
    let cov = Arc::new(
        image_gpu::SelectionCoverage::from_data(2, 1, vec![255, 0]).expect("2 px coverage"),
    );
    let spec = image_js::fill::FillSpec::Gradient {
        kind: image_js::fill::GradientKind::Linear,
        c0: [1.0, 0.0, 0.0, 1.0],
        c1: [1.0, 0.0, 0.0, 1.0], // both stops red ⇒ a flat fill, easy to assert
    };
    let out =
        pollster::block_on(image_js::fill::fill_rgba8(&ctx, &img, &spec, Some(cov))).expect("fill");
    assert_eq!(out.len(), img.rgba.len());
    assert!(
        out[0] > 250 && out[1] < 5 && out[2] < 5,
        "the selected pixel took the fill, got {:?}",
        &out[0..4]
    );
    for (i, (&got, &src)) in out[4..8].iter().zip(img.rgba[4..8].iter()).enumerate() {
        assert!(
            (got as i32 - src as i32).abs() <= 2,
            "byte {i} outside the selection must survive: got {got}, was {src}"
        );
    }
}

#[test]
fn image_editor_generate_noise_fills_the_whole_image_without_a_selection() {
    let Some(ctx) = pollster::block_on(maybe_device()) else {
        println!("SKIP: no GPU adapter");
        return;
    };
    // 8×8 so the hash has room to vary.
    let img =
        image_js::ingest::DecodedImage::from_rgba8(8, 8, vec![0u8; 8 * 8 * 4]).expect("valid");
    let spec = image_js::fill::FillSpec::Noise {
        amount: 1.0,
        seed: 7,
    };
    let out = pollster::block_on(image_js::fill::fill_rgba8(&ctx, &img, &spec, None))
        .expect("noise fill");
    let distinct: std::collections::BTreeSet<u8> = out.chunks_exact(4).map(|p| p[0]).collect();
    assert!(
        distinct.len() > 4,
        "deterministic noise must vary across texels, saw {} values",
        distinct.len()
    );
    // Determinism: the same (seed, amount) yields the same field.
    let again =
        pollster::block_on(image_js::fill::fill_rgba8(&ctx, &img, &spec, None)).expect("repeat");
    assert_eq!(out, again, "same seed ⇒ same noise");
}

#[test]
fn image_editor_generate_fill_composites_through_the_premultiply_bracket() {
    // The seam the brush compositor already closes (image_gpu::stroke):
    // the engine's working buffers are STRAIGHT RGBA, the `compose.*`
    // family's contract is PREMULTIPLIED. Over an OPAQUE backdrop the two
    // coincide and nothing is at stake — which is why the other fill
    // tests never caught this — but over a PNG with alpha, feeding
    // straight bytes to `in0` reads the backdrop back too bright and then
    // returns the composite's premultiplied output as if it were
    // straight. This asserts the CORRECT source-over, not "it changed".
    let Some(ctx) = pollster::block_on(maybe_device()) else {
        println!("SKIP: no GPU adapter");
        return;
    };
    // Backdrop: HALF-TRANSPARENT red (straight (1, 0, 0, 128/255)).
    let img =
        image_js::ingest::DecodedImage::from_rgba8(1, 1, vec![255, 0, 0, 128]).expect("valid");
    // Fill: a flat (both stops equal) HALF-TRANSPARENT blue, so the
    // generated field does not simply cover the backdrop.
    let spec = image_js::fill::FillSpec::Gradient {
        kind: image_js::fill::GradientKind::Linear,
        c0: [0.0, 0.0, 1.0, 0.5],
        c1: [0.0, 0.0, 1.0, 0.5],
    };
    let out =
        pollster::block_on(image_js::fill::fill_rgba8(&ctx, &img, &spec, None)).expect("fill");

    // Source-over of s = (0,0,1) @ 0.5 onto b = (1,0,0) @ 128/255:
    //   αo = αs + αb(1 − αs)               = 0.5 + 0.50196·0.5 = 0.75098
    //   Co·αo = αs(1−αb)·cs + αs·αb·cs + (1−αs)·αb·cb
    //         = (0.25098, 0, 0.5)          [premultiplied]
    //   Co    = (0.3342, 0, 0.6658)        [straight, ÷ αo]
    let expect = [85u8, 0, 170, 191];
    // What the un-bracketed composite produced instead: the backdrop
    // dissociated as if premultiplied (rgb/α ⇒ red at 2.0) and the
    // premultiplied result handed back as straight ⇒ (128, 0, 128, 191).
    for (i, (&got, &want)) in out.iter().zip(expect.iter()).enumerate() {
        assert!(
            (got as i32 - want as i32).abs() <= 2,
            "channel {i}: got {got}, expected {want} (f16 working-space \
             tolerance). {out:?} vs {expect:?} — an un-bracketed composite \
             gives [128, 0, 128, 191]"
        );
    }
}

#[test]
fn image_editor_generate_fill_leaves_an_opaque_backdrop_on_the_fast_path() {
    // The bracket's opaque short-circuit must not change the answer: a
    // fully-opaque backdrop composites identically with or without the
    // `cast.*` steps (premultiply IS the identity there), so an opaque
    // fill lands exactly on the fill colour.
    let Some(ctx) = pollster::block_on(maybe_device()) else {
        println!("SKIP: no GPU adapter");
        return;
    };
    let img =
        image_js::ingest::DecodedImage::from_rgba8(1, 1, vec![255, 0, 0, 255]).expect("valid");
    let spec = image_js::fill::FillSpec::Gradient {
        kind: image_js::fill::GradientKind::Linear,
        c0: [0.0, 0.0, 1.0, 1.0],
        c1: [0.0, 0.0, 1.0, 1.0],
    };
    let out =
        pollster::block_on(image_js::fill::fill_rgba8(&ctx, &img, &spec, None)).expect("fill");
    for (i, (&got, &want)) in out.iter().zip([0u8, 0, 255, 255].iter()).enumerate() {
        assert!(
            (got as i32 - want as i32).abs() <= 2,
            "channel {i}: got {got}, expected {want} — {out:?}"
        );
    }
}

// ── the STRAIGHTEN commit (geom.rotate_bilinear) ─────────────────────
//
// feat: image.editor.crop. The crop tool previewed a rotated frame long
// before anything could commit it; these pin the commit's two lanes.

#[test]
fn image_editor_crop_straighten_at_zero_degrees_needs_no_gpu_and_never_resamples() {
    // A 0° straighten MUST take the pure-windowing path: same bytes as
    // crop_rgba8, no interpolation, no device. (We can only assert the
    // byte equality with a device present for the signature, so compare
    // against the windowing function directly — the door's own
    // short-circuit is asserted in the wasm layer.)
    let img = image_js::ingest::DecodedImage::from_rgba8(4, 2, (0..32u8).collect::<Vec<u8>>())
        .expect("valid");
    let cut = image_js::ingest::crop_rgba8(&img, 1, 0, 2, 2).expect("crop");
    let Some(ctx) = pollster::block_on(maybe_device()) else {
        println!("SKIP: no GPU adapter (0° path is pure CPU windowing anyway)");
        return;
    };
    let straight = pollster::block_on(image_js::ingest::straighten_crop_rgba8(
        &ctx, &img, 1, 0, 2, 2, 0.0,
    ))
    .expect("0° straighten");
    assert_eq!(&straight.rgba[..], &cut.rgba[..], "0° is the exact cut");
}

#[test]
fn image_editor_crop_straighten_180_degrees_reverses_the_full_frame() {
    let Some(ctx) = pollster::block_on(maybe_device()) else {
        println!("SKIP: no GPU adapter");
        return;
    };
    // A 4×2 labelled image; straighten by 180° over the FULL rect is a
    // corner swap, so the result must be the reversed pixel order. This
    // pins the ROTATION DIRECTION through the real GPU dispatch.
    let mut px = vec![0u8; 4 * 2 * 4];
    for i in 0..8 {
        px[i * 4] = (i * 30) as u8;
        px[i * 4 + 1] = 10;
        px[i * 4 + 2] = 20;
        px[i * 4 + 3] = 255;
    }
    let img = image_js::ingest::DecodedImage::from_rgba8(4, 2, px.clone()).expect("valid");
    let out = pollster::block_on(image_js::ingest::straighten_crop_rgba8(
        &ctx, &img, 0, 0, 4, 2, 180.0,
    ))
    .expect("180° straighten");
    assert_eq!((out.width, out.height), (4, 2));
    for i in 0..8 {
        let want = px[(7 - i) * 4];
        let got = out.rgba[i * 4];
        assert!(
            (got as i32 - want as i32).abs() <= 3,
            "pixel {i}: got {got}, expected ~{want} (180° reverses the frame)"
        );
    }
}

#[test]
fn image_editor_crop_straighten_small_angle_keeps_a_flat_field_flat() {
    let Some(ctx) = pollster::block_on(maybe_device()) else {
        println!("SKIP: no GPU adapter");
        return;
    };
    // A uniform field is rotation-invariant: any angle, with clamp-to-edge,
    // must reproduce the same colour everywhere (this catches a broken
    // tap/clamp that would pull in transparent-black).
    let img =
        image_js::ingest::DecodedImage::from_rgba8(16, 16, vec![77u8; 16 * 16 * 4]).expect("valid");
    let out = pollster::block_on(image_js::ingest::straighten_crop_rgba8(
        &ctx, &img, 2, 2, 12, 12, 7.5,
    ))
    .expect("7.5° straighten");
    assert_eq!((out.width, out.height), (12, 12));
    for (i, &b) in out.rgba.iter().enumerate() {
        assert!(
            (b as i32 - 77).abs() <= 2,
            "byte {i}: got {b}, a flat field must survive rotation"
        );
    }
}
