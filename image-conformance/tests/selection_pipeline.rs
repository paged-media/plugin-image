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

//! Selection through the ADJUST CHAIN (spec §6.1) — the pipeline-level
//! twin of `selection_mask.rs`. Where that file proves the mask ABI at a
//! single `execute_tile_once` dispatch, THIS one proves the editor's
//! path: `Pipeline::set_selection` binds a full-image
//! [`SelectionCoverage`] and the U8-in → adjust kernel → U8-out chain
//! (the exact `to_encoder` route `image-js::ingest::adjust_rgba8`
//! drives) changes ONLY the selected pixels.
//!
//! Cases: a hard rect marquee (outside byte-identical, inside the
//! kernel result), a half-weight coverage (proportional blend), the
//! selection folding into the OPERATION CACHE key (a changed selection
//! recomputes instead of serving stale tiles), and sync↔async lockstep
//! under a mask (the wasm lane is the async one).
//!
//! feat: image.selection.mask (adjust-chain reach).

use std::sync::Arc;

use image_codecs::raw::{RawSource, RawTarget};
use image_conformance::device::test_device;
use image_core::{
    AlphaMode, ChannelLayout, ColorSpaceRef, NamedSpace, PixelFormat, Region, SampleDepth, Transfer,
};
use image_gpu::SelectionCoverage;
use image_kernels::families::adjust::{AdjustInvertRgbParams, ADJUST_INVERT_RGB};
use image_pipeline::Pipeline;

const W: u32 = 64;
const H: u32 = 64;

const RGBA8: PixelFormat = PixelFormat {
    channels: ChannelLayout::Rgba,
    depth: SampleDepth::U8,
    alpha: AlphaMode::Straight,
    transfer: Transfer::Linear,
    space: ColorSpaceRef::Named(NamedSpace::LinearSrgb),
};

/// A deterministic RGBA8 gradient (mirrors the ingest stimulus).
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

/// Run source → adjust.invert_rgb → RGBA8 readback under `selection`
/// through a FRESH pipeline (the ingest shape: one pipeline per adjust).
fn run_invert(selection: Option<Arc<SelectionCoverage>>) -> Option<Vec<u8>> {
    let ctx = test_device()?;
    let mut pipe = Pipeline::new();
    pipe.set_selection(selection);
    let src = RawSource::new(W, H, RGBA8, gradient_pixels().into_boxed_slice()).expect("source");
    let leaf = pipe.source(Box::new(src));
    let node = pipe.apply(
        leaf,
        &ADJUST_INVERT_RGB,
        Arc::<[u8]>::from(AdjustInvertRgbParams::new().as_bytes()),
    );
    let mut target = RawTarget::new();
    pipe.to_encoder(node, Region::new(0, 0, W, H), ctx, &mut target, RGBA8)
        .expect("masked encode");
    Some(target.into_pixels())
}

/// The value (0–255, f32) the chain produces for an INVERTED channel
/// `v` at alpha `al`: the decode bridge's f16 quantization, the
/// kernel's premul-aware body `premul((1 − unpremul(a).rgb, u.a))`
/// computed in f32, the f16 store, and the `round(clamp·255)` U8
/// conversion. (adjust.invert_rgb negates the UNPREMULTIPLIED color and
/// re-folds alpha — cf. its family doc.)
fn inverted_f(v: u8, al: u8) -> f32 {
    use half::f16;
    let a = f16::from_f32(v as f32 / 255.0).to_f32();
    let alpha = f16::from_f32(al as f32 / 255.0).to_f32();
    let u = if alpha == 0.0 { 0.0 } else { a / alpha };
    let r = f16::from_f32((1.0 - u) * alpha).to_f32();
    (r.clamp(0.0, 1.0) * 255.0).round()
}

/// [`inverted_f`] rounded to the u8 the encoder emits.
fn inverted_u8(v: u8, al: u8) -> u8 {
    inverted_f(v, al) as u8
}

/// Case 1 — the HEADLINE proof: a rect marquee masks the invert stage;
/// pixels outside the selection come back BYTE-IDENTICAL to the input
/// (mask 0 ⇒ `mix(a, r, 0) == a`, and the u8→f16→u8 round-trip is
/// exact), pixels inside are the inverted result (±1 for the f16 hop).
#[test]
fn image_selection_adjust_changes_only_inside_the_rect() {
    let (rx, ry, rw, rh) = (16u32, 8u32, 24u32, 32u32);
    let cov = SelectionCoverage::rasterize_rect(W, H, rx as f32, ry as f32, rw as f32, rh as f32);
    let Some(out) = run_invert(Some(Arc::new(cov))) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let input = gradient_pixels();
    assert_eq!(out.len(), input.len());
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            let inside = x >= rx && x < rx + rw && y >= ry && y < ry + rh;
            if inside {
                for c in 0..3 {
                    let want = inverted_u8(input[i + c], input[i + 3]);
                    let got = out[i + c];
                    assert!(
                        want.abs_diff(got) <= 1,
                        "inside ({x},{y}) ch {c}: want ~{want} got {got}"
                    );
                }
                assert_eq!(out[i + 3], input[i + 3], "invert preserves alpha");
            } else {
                assert_eq!(
                    &out[i..i + 4],
                    &input[i..i + 4],
                    "outside ({x},{y}) must be byte-identical"
                );
            }
        }
    }
}

/// Case 2 — a HALF-WEIGHT coverage blends proportionally:
/// `out ≈ mix(a, 1−a, 128/255)` per channel (the feathered-selection
/// contract at the chain level).
#[test]
fn image_selection_adjust_half_coverage_blends() {
    let cov = SelectionCoverage::from_data(W, H, vec![128u8; (W * H) as usize]).expect("coverage");
    let Some(out) = run_invert(Some(Arc::new(cov))) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let input = gradient_pixels();
    let m = 128.0 / 255.0;
    for i in (0..input.len()).step_by(4) {
        for c in 0..3 {
            let a = input[i + c] as f32;
            let r = inverted_f(input[i + c], input[i + 3]);
            let want = a * (1.0 - m) + r * m;
            let got = out[i + c] as f32;
            assert!(
                (want - got).abs() <= 2.0,
                "pixel {} ch {c}: want ~{want:.1} got {got}",
                i / 4
            );
        }
    }
}

/// Case 3 — the selection folds into the OP-CACHE key: pulling the SAME
/// pipeline under selection A and then selection B yields B's result
/// (a stale-cache bug would replay A's tiles — the hash-fold guard).
#[test]
fn image_selection_adjust_cache_keys_on_the_selection() {
    let Some(ctx) = test_device() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let left = SelectionCoverage::rasterize_rect(W, H, 0.0, 0.0, (W / 2) as f32, H as f32);
    let right =
        SelectionCoverage::rasterize_rect(W, H, (W / 2) as f32, 0.0, (W / 2) as f32, H as f32);

    let mut pipe = Pipeline::new();
    let src = RawSource::new(W, H, RGBA8, gradient_pixels().into_boxed_slice()).expect("source");
    let leaf = pipe.source(Box::new(src));
    let node = pipe.apply(
        leaf,
        &ADJUST_INVERT_RGB,
        Arc::<[u8]>::from(AdjustInvertRgbParams::new().as_bytes()),
    );
    let roi = Region::new(0, 0, W, H);

    pipe.set_selection(Some(Arc::new(left)));
    let mut t1 = RawTarget::new();
    pipe.to_encoder(node, roi, ctx, &mut t1, RGBA8)
        .expect("pull A");
    let out_a = t1.into_pixels();

    pipe.set_selection(Some(Arc::new(right)));
    let mut t2 = RawTarget::new();
    pipe.to_encoder(node, roi, ctx, &mut t2, RGBA8)
        .expect("pull B");
    let out_b = t2.into_pixels();

    let input = gradient_pixels();
    // Probe one pixel per half. Left half: A inverted it, B left it.
    let li = ((10 * W + 10) * 4) as usize;
    assert!(
        out_a[li].abs_diff(inverted_u8(input[li], input[li + 3])) <= 1,
        "A inverts the left"
    );
    assert_eq!(out_b[li], input[li], "B leaves the left untouched");
    // Right half: mirrored.
    let ri = ((10 * W + W - 10) * 4) as usize;
    assert_eq!(out_a[ri], input[ri], "A leaves the right untouched");
    assert!(
        out_b[ri].abs_diff(inverted_u8(input[ri], input[ri + 3])) <= 1,
        "B inverts the right"
    );
}

/// Case 4 — sync ↔ async LOCKSTEP under a mask: the async encoder lane
/// (what wasm's `adjust_rgba8` actually awaits) produces byte-for-byte
/// the sync masked output.
#[test]
fn image_selection_adjust_async_matches_sync() {
    let Some(ctx) = test_device() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let cov = Arc::new(SelectionCoverage::rasterize_rect(
        W, H, 5.0, 7.0, 20.0, 11.0,
    ));

    let sync_out = run_invert(Some(Arc::clone(&cov))).expect("device present");

    let mut pipe = Pipeline::new();
    pipe.set_selection(Some(cov));
    let src = RawSource::new(W, H, RGBA8, gradient_pixels().into_boxed_slice()).expect("source");
    let leaf = pipe.source(Box::new(src));
    let node = pipe.apply(
        leaf,
        &ADJUST_INVERT_RGB,
        Arc::<[u8]>::from(AdjustInvertRgbParams::new().as_bytes()),
    );
    let mut target = RawTarget::new();
    pollster::block_on(pipe.to_encoder_async(
        node,
        Region::new(0, 0, W, H),
        ctx,
        &mut target,
        RGBA8,
    ))
    .expect("async masked encode");
    assert_eq!(target.into_pixels(), sync_out, "async lane in lockstep");
}
