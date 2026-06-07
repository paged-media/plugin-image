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

//! Engine A end-to-end (spec §7): a `RawSource` gradient through a
//! two-stage pointwise pipeline (`math.linear` → `math.invert`) into a
//! `to_buffer` sink, verified against the composed scalar reference
//! (the kernels' own f32 twins) within f16 tolerance — then pulled a
//! second time to prove the operation cache serves the re-pull without
//! recompute. feat: image.pipeline.engine-a-skeleton.
//!
//! SKIPS cleanly (passes, prints SKIP) when no GPU adapter is present —
//! the merge-gate GPU lane runs where one is guaranteed (§9.3).

use std::sync::Arc;

use image_codecs::raw::RawSource;
use image_conformance::device::test_device;
use image_conformance::quantize::{f16_bits_to_f32, f16_ulp_distance, f32_to_f16_bits};
use image_core::{
    AlphaMode, ChannelLayout, ColorSpaceRef, NamedSpace, PixelFormat, Region, SampleDepth,
    TileCoord, TileData, Transfer,
};
use image_kernels::families::linear::{
    math_invert, math_linear, MathInvertParams, MathLinearParams, MATH_INVERT, MATH_LINEAR,
};
use image_kernels::reference_prelude::Px;
use image_pipeline::Pipeline;

const W: u32 = 64;
const H: u32 = 64;

const SRC_FMT: PixelFormat = PixelFormat {
    channels: ChannelLayout::Rgba,
    depth: SampleDepth::U8,
    alpha: AlphaMode::Straight,
    transfer: Transfer::Linear,
    space: ColorSpaceRef::Named(NamedSpace::LinearSrgb),
};

/// A 64×64 RGBA8 gradient. Channels stay well inside [0,255] so the
/// composed math (gain ×2 then 1−x) never leaves the finite f16 range
/// the harness stimulus rule requires.
fn gradient_pixels() -> Vec<u8> {
    let mut px = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            px[i] = (x * 2) as u8; // R: 0..126
            px[i + 1] = (y * 2) as u8; // G: 0..126
            px[i + 2] = (x + y) as u8; // B: 0..126
            px[i + 3] = 200; // A: constant, < 255
        }
    }
    px
}

/// The reference composition, quantizing to f16 at every storage
/// boundary the GPU crosses: decode bridge → linear store → invert
/// store. `gain`/`bias` mirror the pipeline params.
fn reference_pixel(rgba8: [u8; 4]) -> [u16; 4] {
    // Decode bridge: U8/255 → f16 (the §7.1 M0 cast lane).
    let decoded = Px(rgba8.map(|c| f16_bits_to_f32(f32_to_f16_bits(c as f32 / 255.0))));

    // math.linear (gain 2.0, bias 0.0), then store to f16.
    let lin = math_linear(decoded, Px([0.0; 4]), &MathLinearParams::new(2.0, 0.0));
    let lin_q = Px(lin.0.map(|c| f16_bits_to_f32(f32_to_f16_bits(c))));

    // math.invert (1 − x), then store to f16 — the readback value.
    let inv = math_invert(lin_q, Px([0.0; 4]), &MathInvertParams::new());
    inv.0.map(f32_to_f16_bits)
}

/// Build source → math.linear → math.invert and return the leaf chain
/// (the pipeline owns the source).
fn build(pipe: &mut Pipeline) -> image_pipeline::NodeId {
    let src =
        RawSource::new(W, H, SRC_FMT, gradient_pixels().into_boxed_slice()).expect("raw source");
    let leaf = pipe.source(Box::new(src));
    let lin = pipe.apply(
        leaf,
        &MATH_LINEAR,
        Arc::<[u8]>::from(MathLinearParams::new(2.0, 0.0).as_bytes()),
    );
    pipe.apply(
        lin,
        &MATH_INVERT,
        Arc::<[u8]>::from(MathInvertParams::new().as_bytes()),
    )
}

/// Read the single result tile's rgba16float bytes (ROI is one sub-tile,
/// coord (0,0)).
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

#[test]
fn image_pipeline_engine_a_skeleton() {
    let Some(ctx) = test_device() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };

    let mut pipe = Pipeline::new();
    let sink = build(&mut pipe);
    let roi = Region::new(0, 0, W, H);

    // First pull: full compute (decode → linear → invert).
    let first = pipe.to_buffer(sink, roi, ctx).expect("first pull");
    let first_bytes = tile_bytes(&first);
    assert_eq!(first_bytes.len(), (W * H * 8) as usize);

    // Compare every texel against the composed reference within the
    // linear step's tolerance (2 f16 ULPs covers the GPU f32 multiply
    // rounding; the invert step is exact to 1 ULP, dominated here).
    let gradient = gradient_pixels();
    let mut worst = 0u32;
    for i in 0..(W * H) as usize {
        let rgba8 = [
            gradient[i * 4],
            gradient[i * 4 + 1],
            gradient[i * 4 + 2],
            gradient[i * 4 + 3],
        ];
        let want = reference_pixel(rgba8);
        for c in 0..4 {
            let got =
                u16::from_le_bytes([first_bytes[i * 8 + c * 2], first_bytes[i * 8 + c * 2 + 1]]);
            let d = f16_ulp_distance(want[c], got);
            worst = worst.max(d);
            assert!(
                d <= 2,
                "texel {i} channel {c}: f16 ULP distance {d} (want {want:?} bits, got {got})"
            );
        }
    }
    eprintln!("engine-A e2e: worst f16 ULP distance = {worst}");

    // Second pull of the same sink/ROI: every node's subtree is
    // unchanged, so all three nodes hit the operation cache.
    let hits_before = pipe.cache_hits();
    let second = pipe.to_buffer(sink, roi, ctx).expect("second pull");
    let second_bytes = tile_bytes(&second);

    assert!(
        pipe.cache_hits() > hits_before,
        "second pull must be served from the operation cache (hits {} -> {})",
        hits_before,
        pipe.cache_hits()
    );
    assert_eq!(
        first_bytes, second_bytes,
        "cached re-pull must return byte-identical tiles"
    );
}
