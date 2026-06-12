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

//! Async-sink parity (the M4 ingest slice's readback lane —
//! feat: image.pipeline.async-sink): `to_buffer_async` /
//! `to_encoder_async` produce BYTE-FOR-BYTE the sync sinks' output over
//! the same graph (one recording path, two readback maps), and the op
//! cache is shared across the lanes (an async re-pull of a sync-pulled
//! subtree is a hit). Natively the async lane runs under pollster; on
//! wasm32/WebGPU it is the ONLY correct lane (a blocking poll cannot
//! pump the map callback there).
//!
//! SKIPS cleanly (passes, prints SKIP) when no GPU adapter is present —
//! the merge-gate GPU lane runs where one is guaranteed (§9.3).

use std::sync::Arc;

use image_codecs::raw::{RawSource, RawTarget};
use image_conformance::device::test_device;
use image_core::{
    AlphaMode, ChannelLayout, ColorSpaceRef, NamedSpace, PixelFormat, Region, SampleDepth,
    TileData, Transfer,
};
use image_kernels::families::adjust::{
    AdjustBrightnessContrastParams, AdjustExposureParams, AdjustSaturationParams,
    ADJUST_BRIGHTNESS_CONTRAST, ADJUST_EXPOSURE, ADJUST_SATURATION,
};
use image_pipeline::Pipeline;

const W: u32 = 64;
const H: u32 = 48;

/// Straight RGBA8 linear — the raw source format (mirrors pipeline_e2e).
const SRC_FMT: PixelFormat = PixelFormat {
    channels: ChannelLayout::Rgba,
    depth: SampleDepth::U8,
    alpha: AlphaMode::Straight,
    transfer: Transfer::Linear,
    space: ColorSpaceRef::Named(NamedSpace::LinearSrgb),
};

/// A deterministic RGBA8 gradient inside the low-rounding f16 range.
fn gradient_pixels() -> Vec<u8> {
    let mut px = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            px[i] = (x * 2) as u8;
            px[i + 1] = (y * 2) as u8;
            px[i + 2] = (x + y) as u8;
            px[i + 3] = 200;
        }
    }
    px
}

/// The M4 adjustments chain: source → exposure → brightness/contrast →
/// saturation (exactly the image-js ingest graph).
fn build(pipe: &mut Pipeline) -> image_pipeline::NodeId {
    let src =
        RawSource::new(W, H, SRC_FMT, gradient_pixels().into_boxed_slice()).expect("raw source");
    let leaf = pipe.source(Box::new(src));
    let exp = pipe.apply(
        leaf,
        &ADJUST_EXPOSURE,
        Arc::<[u8]>::from(AdjustExposureParams::new(0.5).as_bytes()),
    );
    let bc = pipe.apply(
        exp,
        &ADJUST_BRIGHTNESS_CONTRAST,
        Arc::<[u8]>::from(AdjustBrightnessContrastParams::new(0.05, 1.2).as_bytes()),
    );
    pipe.apply(
        bc,
        &ADJUST_SATURATION,
        Arc::<[u8]>::from(AdjustSaturationParams::new(1.4).as_bytes()),
    )
}

#[test]
fn image_pipeline_async_sink_to_encoder_matches_sync() {
    let Some(ctx) = test_device() else {
        println!("SKIP: no GPU adapter");
        return;
    };
    let roi = Region::new(0, 0, W, H);

    let mut sync_pipe = Pipeline::new();
    let sync_node = build(&mut sync_pipe);
    let mut sync_target = RawTarget::new();
    sync_pipe
        .to_encoder(sync_node, roi, ctx, &mut sync_target, SRC_FMT)
        .expect("sync to_encoder");
    let sync_px = sync_target.into_pixels();

    let mut async_pipe = Pipeline::new();
    let async_node = build(&mut async_pipe);
    let mut async_target = RawTarget::new();
    pollster::block_on(async_pipe.to_encoder_async(
        async_node,
        roi,
        ctx,
        &mut async_target,
        SRC_FMT,
    ))
    .expect("async to_encoder");
    let async_px = async_target.into_pixels();

    assert_eq!(sync_px.len(), (W * H * 4) as usize);
    assert_eq!(sync_px, async_px, "async readback must match sync exactly");
}

#[test]
fn image_pipeline_async_sink_to_buffer_matches_sync_and_shares_cache() {
    let Some(ctx) = test_device() else {
        println!("SKIP: no GPU adapter");
        return;
    };
    let roi = Region::new(0, 0, W, H);

    // ONE pipeline: pull sync, then async — the second pull must be a
    // cache hit (the lanes share the demand path + op cache).
    let mut pipe = Pipeline::new();
    let node = build(&mut pipe);
    let sync_map = pipe.to_buffer(node, roi, ctx).expect("sync to_buffer");
    let misses_after_sync = pipe.cache_misses();
    let hits_before = pipe.cache_hits();

    let async_map =
        pollster::block_on(pipe.to_buffer_async(node, roi, ctx)).expect("async to_buffer");

    assert_eq!(
        pipe.cache_misses(),
        misses_after_sync,
        "the async re-pull must not recompute"
    );
    assert!(pipe.cache_hits() > hits_before);

    // Tile-for-tile byte equality.
    let coords: Vec<_> = sync_map.iter().map(|(c, _)| *c).collect();
    assert!(!coords.is_empty());
    for c in coords {
        let a = sync_map.get(c).expect("sync tile");
        let b = async_map.get(c).expect("async tile");
        match (&a.data, &b.data) {
            (TileData::Heap(ab), TileData::Heap(bb)) => assert_eq!(ab, bb, "tile {c:?}"),
            other => panic!("expected heap tiles, got {other:?}"),
        }
    }
}
