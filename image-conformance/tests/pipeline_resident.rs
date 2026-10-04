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

//! Resident chains against the scheduler they replaced.
//!
//! The scheduler pulls runs of consecutive point stages tile by tile
//! through every stage on the device, and batches every other node's
//! tiles into one submit. Its claim is that no byte changes: random
//! adjust graphs (point runs, separable blurs, an unsharp mask's
//! two-input stage, with and without a selection) over a multi-tile
//! image produce exactly the bytes of the pre-residency scheduler
//! (every stage of every tile its own upload, dispatch and readback),
//! through both sinks and both readback lanes, on the same device.

#![allow(non_snake_case)]

use std::sync::Arc;

use image_codecs::raw::{RawSource, RawTarget};
use image_conformance::device::test_device;
use image_core::{
    AlphaMode, ChannelLayout, ColorSpaceRef, NamedSpace, PixelFormat, Region, SampleDepth,
    TileData, Transfer,
};
use image_gpu::SelectionCoverage;
use image_kernels::families::adjust::{
    AdjustBrightnessContrastParams, AdjustExposureParams, AdjustHueRotateParams,
    AdjustSaturationParams, ADJUST_BRIGHTNESS_CONTRAST, ADJUST_EXPOSURE, ADJUST_HUE_ROTATE,
    ADJUST_SATURATION,
};
use image_kernels::families::conv::{
    ConvGaussianParams, ConvUnsharpParams, CONV_GAUSSIAN_H, CONV_GAUSSIAN_V, CONV_UNSHARP,
};
use image_pipeline::{NodeId, Pipeline};
use proptest::prelude::*;

const RGBA8: PixelFormat = PixelFormat {
    channels: ChannelLayout::Rgba,
    depth: SampleDepth::U8,
    alpha: AlphaMode::Straight,
    transfer: Transfer::Linear,
    space: ColorSpaceRef::Named(NamedSpace::LinearSrgb),
};
const RGBA16F: PixelFormat = PixelFormat {
    depth: SampleDepth::F16,
    ..RGBA8
};

#[derive(Debug, Clone)]
enum Stage {
    Exposure(f32),
    Contrast(f32, f32),
    Saturation(f32),
    Hue(f32),
    Blur(f32),
    Sharpen(f32),
}

fn stage() -> impl Strategy<Value = Stage> {
    prop_oneof![
        3 => (-1.0f32..1.0).prop_map(Stage::Exposure),
        3 => (-0.2f32..0.2, 0.5f32..1.5).prop_map(|(b, c)| Stage::Contrast(b, c)),
        3 => (0.0f32..2.0).prop_map(Stage::Saturation),
        2 => (-90.0f32..90.0).prop_map(Stage::Hue),
        1 => (0.5f32..2.0).prop_map(Stage::Blur),
        1 => (0.1f32..1.0).prop_map(Stage::Sharpen),
    ]
}

fn image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    (0..w * h * 4)
        .map(|i| (i.wrapping_mul(2654435761) ^ seed) as u8 | 1)
        .collect()
}

fn graph(pipe: &mut Pipeline, w: u32, h: u32, seed: u32, stages: &[Stage]) -> NodeId {
    let src = RawSource::new(w, h, RGBA8, image(w, h, seed).into_boxed_slice()).expect("src");
    let mut node = pipe.source(Box::new(src));
    let gauss = |pipe: &mut Pipeline, n: NodeId, sigma: f32, radius: u32| {
        let p = Arc::<[u8]>::from(ConvGaussianParams::new(sigma, radius).as_bytes());
        let n = pipe.apply(n, &CONV_GAUSSIAN_H, Arc::clone(&p));
        pipe.apply(n, &CONV_GAUSSIAN_V, p)
    };
    for s in stages {
        node = match *s {
            Stage::Exposure(ev) => pipe.apply(
                node,
                &ADJUST_EXPOSURE,
                Arc::<[u8]>::from(AdjustExposureParams::new(ev).as_bytes()),
            ),
            Stage::Contrast(b, c) => pipe.apply(
                node,
                &ADJUST_BRIGHTNESS_CONTRAST,
                Arc::<[u8]>::from(AdjustBrightnessContrastParams::new(b, c).as_bytes()),
            ),
            Stage::Saturation(v) => pipe.apply(
                node,
                &ADJUST_SATURATION,
                Arc::<[u8]>::from(AdjustSaturationParams::new(v).as_bytes()),
            ),
            Stage::Hue(d) => pipe.apply(
                node,
                &ADJUST_HUE_ROTATE,
                Arc::<[u8]>::from(AdjustHueRotateParams::new(d).as_bytes()),
            ),
            Stage::Blur(sigma) => gauss(pipe, node, sigma, (sigma * 3.0).ceil() as u32),
            Stage::Sharpen(amount) => {
                let blurred = gauss(pipe, node, 1.5, 5);
                pipe.apply2(
                    node,
                    blurred,
                    &CONV_UNSHARP,
                    Arc::<[u8]>::from(ConvUnsharpParams::new(amount, 0.0).as_bytes()),
                )
            }
        };
    }
    node
}

/// Pull the graph both ways and compare every byte.
fn compare(
    w: u32,
    h: u32,
    seed: u32,
    stages: &[Stage],
    selection: bool,
) -> Result<(), TestCaseError> {
    let Some(ctx) = test_device() else {
        return Ok(());
    };
    let roi = Region::new(0, 0, w, h);
    let sel = selection.then(|| {
        Arc::new(SelectionCoverage::rasterize_rect(
            w,
            h,
            10.5,
            7.25,
            w as f32 * 0.6,
            h as f32 * 0.5,
        ))
    });
    let run = |unbatched: bool, fmt: PixelFormat, asynchronous: bool| {
        let mut pipe = Pipeline::new();
        pipe.set_unbatched_reference(unbatched);
        pipe.set_selection(sel.clone());
        let node = graph(&mut pipe, w, h, seed, stages);
        let mut target = RawTarget::new();
        if asynchronous {
            pollster::block_on(pipe.to_encoder_async(node, roi, ctx, &mut target, fmt))
                .expect("pull");
        } else {
            pipe.to_encoder(node, roi, ctx, &mut target, fmt)
                .expect("pull");
        }
        target.into_pixels()
    };
    for (fmt, asynchronous) in [(RGBA16F, true), (RGBA8, false)] {
        let want = run(true, fmt, asynchronous);
        let got = run(false, fmt, asynchronous);
        prop_assert!(
            got == want,
            "resident scheduler differs ({:?}, async {asynchronous})",
            fmt.depth
        );
    }
    // The tile-map sink too.
    let tiles = |unbatched: bool| {
        let mut pipe = Pipeline::new();
        pipe.set_unbatched_reference(unbatched);
        pipe.set_selection(sel.clone());
        let node = graph(&mut pipe, w, h, seed, stages);
        let map = pipe.to_buffer(node, roi, ctx).expect("pull");
        let mut v: Vec<_> = map
            .iter()
            .map(|(c, t)| match &t.data {
                TileData::Heap(b) => (*c, b.to_vec()),
                _ => (*c, Vec::new()),
            })
            .collect();
        v.sort_by_key(|(c, _)| *c);
        v
    };
    prop_assert!(tiles(false) == tiles(true), "to_buffer differs");
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 16, ..ProptestConfig::default() })]

    #[test]
    fn resident_chains_equal_the_per_tile_scheduler__feat__image_pipeline_async_sink(
        dims in prop_oneof![Just((300u32, 270u32)), Just((257, 64)), Just((40, 300))],
        seed in any::<u32>(),
        stages in prop::collection::vec(stage(), 1..8),
        selection in any::<bool>(),
    ) {
        compare(dims.0, dims.1, seed, &stages, selection)?;
    }
}

/// The Apply chain the editor runs (exposure, contrast, saturation, hue,
/// blur, sharpen), with and without a selection.
#[test]
fn the_six_stage_apply_chain_is_byte_equal__feat__image_pipeline_async_sink() {
    let stages = [
        Stage::Exposure(0.3),
        Stage::Contrast(0.0, 1.2),
        Stage::Saturation(1.1),
        Stage::Hue(10.0),
        Stage::Blur(1.5),
        Stage::Sharpen(0.5),
    ];
    for selection in [false, true] {
        compare(512, 512, 3, &stages, selection).expect("byte-equal");
    }
}
