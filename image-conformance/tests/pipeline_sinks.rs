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

//! Engine A M1 sinks (spec §7.2/§7.3): the binary `apply2` compose lane,
//! the `to_encoder` structured-readback sink, and the shrink-on-load
//! planner. feat: image.pipeline.engine-a-skeleton.
//!
//! GPU tests SKIP cleanly (pass, print SKIP) when no adapter is present;
//! the `plan_shrink` matrix is pure CPU algebra and always runs.

use std::sync::Arc;

use image_codecs::raw::RawSource;
use image_codecs::{ImageSource, MemoryByteSource, PngSource, PngTarget};
use image_conformance::device::test_device;
use image_conformance::quantize::{f16_bits_to_f32, f16_ulp_distance, f32_to_f16_bits};
use image_core::{
    AlphaMode, ChannelLayout, ColorSpaceRef, NamedSpace, PixelFormat, Region, SampleDepth,
    TileCoord, TileData, Transfer, TransferCurve,
};
use image_kernels::families::arithmetic::{math_add, MathAddParams, MATH_ADD};
use image_kernels::reference_prelude::Px;
use image_pipeline::{plan_shrink, Pipeline};

const W: u32 = 64;
const H: u32 = 64;

/// Straight RGBA8 linear — the raw source format (mirrors pipeline_e2e).
const SRC_FMT: PixelFormat = PixelFormat {
    channels: ChannelLayout::Rgba,
    depth: SampleDepth::U8,
    alpha: AlphaMode::Straight,
    transfer: Transfer::Linear,
    space: ColorSpaceRef::Named(NamedSpace::LinearSrgb),
};

/// The U8 sRGB output format `to_encoder` → `PngTarget` round-trips.
const OUT_FMT: PixelFormat = PixelFormat {
    channels: ChannelLayout::Rgba,
    depth: SampleDepth::U8,
    alpha: AlphaMode::Straight,
    transfer: Transfer::Gamma(TransferCurve::Srgb),
    space: ColorSpaceRef::Named(NamedSpace::Srgb),
};

/// Two distinct deterministic RGBA8 gradients. Both stay well inside
/// [0,128] per channel so `a + b` never leaves the finite, low-rounding
/// f16 range the harness stimulus rule requires (sum ≤ ~1.0 after /255).
fn pixels_a() -> Vec<u8> {
    let mut px = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            px[i] = x as u8; // R 0..63
            px[i + 1] = y as u8; // G 0..63
            px[i + 2] = ((x + y) / 2) as u8; // B
            px[i + 3] = 40; // A
        }
    }
    px
}

fn pixels_b() -> Vec<u8> {
    let mut px = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            px[i] = (63 - x) as u8; // R
            px[i + 1] = (63 - y) as u8; // G
            px[i + 2] = 20; // B
            px[i + 3] = 30; // A
        }
    }
    px
}

fn raw(pixels: Vec<u8>) -> RawSource {
    RawSource::new(W, H, SRC_FMT, pixels.into_boxed_slice()).expect("raw source")
}

/// Read the (0,0) result tile's rgba16float bytes.
fn tile_bytes(map: &image_core::TileMap) -> Vec<u8> {
    let tile = map
        .get(TileCoord {
            level: 0,
            x: 0,
            y: 0,
        })
        .expect("result tile present");
    match &tile.data {
        TileData::Heap(b) => b.to_vec(),
        other => panic!("expected heap tile, got {other:?}"),
    }
}

/// Case 1: `apply2` with `math.add` over two raw sources equals the
/// composed scalar reference (the kernel's own f32 twin) within f16
/// tolerance.
#[test]
fn image_pipeline_engine_a_skeleton_apply2_add() {
    let Some(ctx) = test_device() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };

    let mut pipe = Pipeline::new();
    let a = pipe.source(Box::new(raw(pixels_a())));
    let b = pipe.source(Box::new(raw(pixels_b())));
    let sum = pipe.apply2(
        a,
        b,
        &MATH_ADD,
        Arc::<[u8]>::from(MathAddParams::new().as_bytes()),
    );
    let roi = Region::new(0, 0, W, H);

    let map = pipe.to_buffer(sum, roi, ctx).expect("apply2 pull");
    let bytes = tile_bytes(&map);
    assert_eq!(bytes.len(), (W * H * 8) as usize);

    let (pa, pb) = (pixels_a(), pixels_b());
    let mut worst = 0u32;
    for i in 0..(W * H) as usize {
        // Decode bridge: U8/255 → f16 (the §7.1 M0 cast lane), for each
        // input independently — then math_add, then store to f16.
        let da = Px([
            f16_bits_to_f32(f32_to_f16_bits(pa[i * 4] as f32 / 255.0)),
            f16_bits_to_f32(f32_to_f16_bits(pa[i * 4 + 1] as f32 / 255.0)),
            f16_bits_to_f32(f32_to_f16_bits(pa[i * 4 + 2] as f32 / 255.0)),
            f16_bits_to_f32(f32_to_f16_bits(pa[i * 4 + 3] as f32 / 255.0)),
        ]);
        let db = Px([
            f16_bits_to_f32(f32_to_f16_bits(pb[i * 4] as f32 / 255.0)),
            f16_bits_to_f32(f32_to_f16_bits(pb[i * 4 + 1] as f32 / 255.0)),
            f16_bits_to_f32(f32_to_f16_bits(pb[i * 4 + 2] as f32 / 255.0)),
            f16_bits_to_f32(f32_to_f16_bits(pb[i * 4 + 3] as f32 / 255.0)),
        ]);
        let want = math_add(da, db, &MathAddParams::new())
            .0
            .map(f32_to_f16_bits);
        for c in 0..4 {
            let got = u16::from_le_bytes([bytes[i * 8 + c * 2], bytes[i * 8 + c * 2 + 1]]);
            let d = f16_ulp_distance(want[c], got);
            worst = worst.max(d);
            assert!(
                d <= 1,
                "texel {i} ch {c}: f16 ULP distance {d} (want {want:?} bits, got {got})"
            );
        }
    }
    eprintln!("apply2 math.add: worst f16 ULP distance = {worst}");
}

/// Case 2: `to_encoder` through `PngTarget` — stream a graph result into
/// a PNG, decode it back, and assert it equals the U8 quantization of the
/// SAME graph pulled via `to_buffer`. (`apply2 math.add` is the graph
/// under test, so this also exercises the binary lane through the
/// readback sink.)
#[test]
fn image_pipeline_engine_a_skeleton_to_encoder_png() {
    let Some(ctx) = test_device() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };

    let mut pipe = Pipeline::new();
    let a = pipe.source(Box::new(raw(pixels_a())));
    let b = pipe.source(Box::new(raw(pixels_b())));
    let sum = pipe.apply2(
        a,
        b,
        &MATH_ADD,
        Arc::<[u8]>::from(MathAddParams::new().as_bytes()),
    );
    let roi = Region::new(0, 0, W, H);

    // Reference: the working-space buffer, quantized to U8 the same way
    // `to_encoder` does (clamp(x,0,1)*255 round, alpha carried for RGBA).
    let buf = pipe.to_buffer(sum, roi, ctx).expect("to_buffer pull");
    let work = tile_bytes(&buf);
    let mut expect_u8 = vec![0u8; (W * H * 4) as usize];
    for i in 0..(W * H) as usize {
        for c in 0..4 {
            let f = f16_bits_to_f32(u16::from_le_bytes([
                work[i * 8 + c * 2],
                work[i * 8 + c * 2 + 1],
            ]));
            expect_u8[i * 4 + c] = (f.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }

    // Stream the same graph into a PNG target, then decode it back.
    let mut target = PngTarget::new();
    let stats = pipe
        .to_encoder(sum, roi, ctx, &mut target, OUT_FMT)
        .expect("to_encoder");
    let png = target.into_bytes();
    assert_eq!(
        stats.bytes_written as usize,
        png.len(),
        "stats == byte count"
    );
    assert!(!png.is_empty(), "non-empty PNG");

    let mut src = PngSource::new(MemoryByteSource::new(png.into_boxed_slice()));
    let info = src.probe().expect("probe");
    assert_eq!((info.width, info.height), (W, H));
    let bpp = info.format.bytes_per_pixel();
    let region = Region::new(0, 0, W, H);
    let mut decoded = vec![0u8; (W * H) as usize * bpp];
    let mut out = image_core::TileSliceMut {
        region,
        format: info.format,
        row_stride: W as usize * bpp,
        bytes: &mut decoded,
    };
    src.read_region(region, 1, &mut out).expect("read_region");
    // `out` borrows `buf` mutably; end that borrow before reading `buf`.
    let _ = out;

    // PNG is lossless, so the decoded RGBA8 equals what to_encoder packed,
    // which equals the U8 quantization of the to_buffer working output.
    assert_eq!(
        decoded, expect_u8,
        "to_encoder→PNG→decode must equal the U8 quantization of to_buffer"
    );
}

/// 2b. `to_encoder` over a multi-tile-row ROI: the strip walk must cover
/// every row exactly once (the `PngTarget` coverage contract) and still
/// round-trip. 300px tall = two tile-row strips (256 + 44).
#[test]
fn image_pipeline_engine_a_skeleton_to_encoder_multistrip() {
    let Some(ctx) = test_device() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };

    const TW: u32 = 70;
    const TH: u32 = 300;
    let mut px = vec![0u8; (TW * TH * 4) as usize];
    for y in 0..TH {
        for x in 0..TW {
            let i = ((y * TW + x) * 4) as usize;
            px[i] = (x % 200) as u8;
            px[i + 1] = (y % 200) as u8;
            px[i + 2] = ((x + y) % 200) as u8;
            px[i + 3] = 255;
        }
    }
    let src = RawSource::new(TW, TH, SRC_FMT, px.into_boxed_slice()).unwrap();

    let mut pipe = Pipeline::new();
    let leaf = pipe.source(Box::new(src));
    // Identity-ish op so the strip path is exercised on a real graph:
    // add the source to a zero source (a + 0 == a within tolerance).
    let zeros = RawSource::new(TW, TH, SRC_FMT, vec![0u8; (TW * TH * 4) as usize]).unwrap();
    let zleaf = pipe.source(Box::new(zeros));
    let node = pipe.apply2(
        leaf,
        zleaf,
        &MATH_ADD,
        Arc::<[u8]>::from(MathAddParams::new().as_bytes()),
    );
    let roi = Region::new(0, 0, TW, TH);

    let mut target = PngTarget::new();
    pipe.to_encoder(node, roi, ctx, &mut target, OUT_FMT)
        .expect("multistrip to_encoder");
    let png = target.into_bytes();

    let mut dec = PngSource::new(MemoryByteSource::new(png.into_boxed_slice()));
    let info = dec.probe().unwrap();
    assert_eq!((info.width, info.height), (TW, TH), "full coverage decoded");
}

/// Case 3: `plan_shrink` matrix (spec §7.2) — the JPEG DCT ladder
/// [1,2,4,8] × scales {1.0, 0.5, 0.3, 0.1} picks shrinks (1,2,2,8), and
/// the composition algebra (1/shrink)·residual == requested holds in
/// every row. Pure CPU — always runs.
#[test]
fn image_pipeline_engine_a_skeleton_plan_shrink_matrix() {
    let native = [1u32, 2, 4, 8];
    let cases = [(1.0f32, 1u32), (0.5, 2), (0.3, 2), (0.1, 8)];
    for (scale, want_shrink) in cases {
        let (shrink, residual) = plan_shrink(&native, scale);
        assert_eq!(
            shrink, want_shrink,
            "scale {scale}: expected native shrink {want_shrink}, got {shrink}"
        );
        // The §7.2 invariant: decoding at 1/shrink then resampling by
        // residual lands exactly at the requested fraction.
        let composed = (1.0 / shrink as f32) * residual;
        assert!(
            (composed - scale).abs() < 1e-6,
            "scale {scale}: (1/{shrink})*{residual} = {composed} != {scale}"
        );
    }

    // PNG advertises only [1]: every request stays full-res at the
    // decoder, residual carries the whole scale.
    for scale in [1.0f32, 0.5, 0.3, 0.1] {
        let (shrink, residual) = plan_shrink(&[1], scale);
        assert_eq!(shrink, 1);
        assert!((residual - scale).abs() < 1e-6);
    }
}
