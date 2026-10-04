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

//! WORK BUDGETS for whole user operations, in COUNTS (engine +
//! GPU counters), never wall-clock.
//!
//! Each budget equals the value measured when it was written. It is
//! lowered in the commit that earns it and never raised. `check` asserts
//! EQUALITY: a budget looser than the measurement cannot catch the next
//! regression, so an improvement fails too, with the line to change.
//! Run with `--nocapture` to print every measurement.

#![allow(non_snake_case)]

use std::sync::Arc;

use image_core::Region;
use image_gpu::{GpuContext, GpuCounters};
use image_js::counters;
use image_js::ingest::{adjust_rgba8, AdjustParams, DecodedImage};
use image_js::layers::LayerStack;
use image_js::pixels::Pixels;

fn device() -> Option<&'static GpuContext> {
    image_gpu::test_support::device_or_skip("perf_budgets")
}

/// Compare every named measurement with its budget; fail once, listing
/// all mismatches, so one run reports everything that moved.
fn check(case: &str, got: &[(&str, u64, u64)]) {
    let mut bad = Vec::new();
    for (name, measured, budget) in got {
        println!("{case}: {name} = {measured} (budget {budget})");
        if measured != budget {
            bad.push(format!(
                "{name}: measured {measured}, budget {budget} — {}",
                if measured > budget {
                    "REGRESSION (more work than budgeted)"
                } else {
                    "improved: lower the budget to the measurement in this commit"
                }
            ));
        }
    }
    assert!(bad.is_empty(), "{case}:\n  {}", bad.join("\n  "));
}

fn gpu_rows(g: &GpuCounters) -> [(&'static str, u64); 6] {
    [
        ("pipelines_built", g.pipelines_built),
        ("dispatches", g.dispatches),
        ("submits", g.submits),
        ("textures_created", g.textures_created),
        ("readbacks", g.readbacks),
        ("bytes_uploaded", g.bytes_uploaded),
    ]
}

/// Deterministic non-uniform RGBA8.
fn ramp(w: u32, h: u32, seed: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            v.extend_from_slice(&[
                (x as u8).wrapping_add(seed),
                (y as u8).wrapping_mul(3),
                seed,
                200,
            ]);
        }
    }
    v
}

/// A stack of `n` canvas-sized, non-empty, half-opaque layers.
fn stack(w: u32, h: u32, n: usize) -> LayerStack {
    let mut s = LayerStack::from_image(w, h, Arc::from(ramp(w, h, 1))).expect("stack");
    for i in 1..n {
        s.add(&format!("L{i}"));
        s.edit_active(
            "fill",
            Region::new(0, 0, w, h),
            Pixels::from_rgba8(Arc::from(ramp(w, h, (i * 40) as u8))),
        )
        .expect("edit");
    }
    s
}

// ── 16-bit tile cut ──────────────────────────────────────────────────

/// BUDGET (2026-10-04): cutting ONE 64×64 tile out of a 16-bit
/// 256×256 image narrows the WHOLE image once per row of the tile —
/// `to_rgba8()` sits inside the row loop (`ingest.rs`,
/// `tile_window_rgba8`). Quadratic in the image for every tile.
const TILE16_NARROWINGS: u64 = 64;
const TILE16_NARROWED_BYTES: u64 = 64 * 256 * 256 * 4;

#[test]
fn a_16bit_tile_cut__feat__image_editor_tile_provider() {
    let (w, h) = (256u32, 256u32);
    let samples: Vec<u16> = ramp(w, h, 7)
        .iter()
        .map(|&b| (b as u16) << 8 | b as u16)
        .collect();
    let mut img = DecodedImage::from_rgba8(w, h, vec![0; (w * h * 4) as usize]).expect("img");
    img.rgba = Pixels::from_rgba16(&samples);
    counters::reset();
    let ((bytes, tw, th), e, _) = counters::measure(|| img.tile_window_rgba8(64, 64, 64, 64));
    assert_eq!((tw, th, bytes.len()), (64, 64, 64 * 64 * 4));
    // The bytes are right whatever the cost: the high byte of each sample.
    assert_eq!(
        &bytes[..4],
        &ramp(w, h, 7)[(64 * 256 + 64) * 4..(64 * 256 + 64) * 4 + 4]
    );
    check(
        "16-bit tile cut",
        &[
            ("tiles_cut", e.tiles_cut, 1),
            ("depth_narrowings", e.depth_narrowings, TILE16_NARROWINGS),
            ("narrowed_bytes", e.narrowed_bytes, TILE16_NARROWED_BYTES),
        ],
    );
}

// ── layer composite ──────────────────────────────────────────────────

/// Measured 2026-10-04 on Metal: one pipeline build, submit and readback per
/// dispatch, and the accumulator re-uploaded for every layer.
const COMPOSITE3: [(&str, u64); 6] = [
    ("pipelines_built", 7),
    ("dispatches", 7),
    ("submits", 7),
    ("textures_created", 24),
    ("readbacks", 7),
    ("bytes_uploaded", 24641576),
];

/// One composite of a 3-layer 512² stack.
#[test]
fn a_three_layer_composite__feat__image_editor_layers() {
    let Some(ctx) = device() else { return };
    let s = stack(512, 512, 3);
    counters::reset();
    let (out, e, g) = counters::measure(|| pollster::block_on(s.composite(Some(ctx), None)));
    out.expect("composite");
    let mut rows = vec![
        ("composites", e.composites, 1),
        ("layers_folded", e.layers_folded, 3),
    ];
    for ((name, v), (_, b)) in gpu_rows(&g).into_iter().zip(COMPOSITE3) {
        rows.push((name, v, b));
    }
    check("3-layer composite 512²", &rows);
}

/// A 20-step opacity drag on the top layer: today every step is a full
/// composite.
/// Measured 2026-10-04 on Metal: 20 × the composite above.
const DRAG20: [(&str, u64); 6] = [
    ("pipelines_built", 140),
    ("dispatches", 140),
    ("submits", 140),
    ("textures_created", 480),
    ("readbacks", 140),
    ("bytes_uploaded", 492831520),
];

#[test]
fn a_twenty_step_opacity_drag__feat__image_editor_layers() {
    let Some(ctx) = device() else { return };
    let mut s = stack(512, 512, 3);
    counters::reset();
    let ((), e, g) = counters::measure(|| {
        for step in 0..20 {
            s.set_opacity(2, 1.0 - step as f32 / 40.0).expect("opacity");
            pollster::block_on(s.composite(Some(ctx), None)).expect("composite");
        }
    });
    let mut rows = vec![
        ("composites", e.composites, 20),
        ("layers_folded", e.layers_folded, 60),
    ];
    for ((name, v), (_, b)) in gpu_rows(&g).into_iter().zip(DRAG20) {
        rows.push((name, v, b));
    }
    check("20-step opacity drag 512²", &rows);
}

// ── Apply (the adjust chain) ─────────────────────────────────────────

/// Measured 2026-10-04 on Metal: every stage, every tile, its own pipeline
/// build, upload, submit and readback.
const APPLY6: [(&str, u64); 6] = [
    ("pipelines_built", 276),
    ("dispatches", 276),
    ("submits", 276),
    ("textures_created", 832),
    ("readbacks", 276),
    ("bytes_uploaded", 54978224),
];

/// Apply with six stages on (exposure, contrast, saturation, hue, blur,
/// sharpen) over a 512² image; and the SAME Apply again, which reuses
/// nothing today.
#[test]
fn apply_with_six_stages__feat__image_editor_adjust_breadth() {
    let Some(ctx) = device() else { return };
    let img = DecodedImage::from_rgba8(512, 512, ramp(512, 512, 3)).expect("img");
    let p = AdjustParams {
        exposure_ev: 0.3,
        contrast: 0.2,
        saturation: 0.1,
        hue_degrees: 10.0,
        blur_sigma: 1.5,
        sharpen_amount: 0.5,
        ..AdjustParams::default()
    };
    counters::reset();
    let (out, _e, g) = counters::measure(|| pollster::block_on(adjust_rgba8(ctx, &img, &p, None)));
    out.expect("apply");
    let rows: Vec<(&str, u64, u64)> = gpu_rows(&g)
        .into_iter()
        .zip(APPLY6)
        .map(|((n, v), (_, b))| (n, v, b))
        .collect();
    check("Apply, 6 stages, 512²", &rows);
}
