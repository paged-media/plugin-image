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
//! Each GPU case measures the STEADY STATE: the operation runs once to
//! warm the device, then is measured, so the result does not depend on
//! test order. Run with `--nocapture` to print every measurement.

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

/// BUDGET: cutting ONE 64×64 tile out of a 16-bit 256×256 image
/// narrows nothing but the tile. Was 64 whole-image narrowings (16 MiB)
/// when measured on 2026-10-04: `to_rgba8()` sat inside the row loop.
const TILE16_NARROWINGS: u64 = 0;
const TILE16_NARROWED_BYTES: u64 = 0;

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
    // The bytes equal narrowing the whole image and then cropping.
    let full = img.rgba.to_rgba8();
    let want: Vec<u8> = (64..128usize)
        .flat_map(|y| full[(y * 256 + 64) * 4..(y * 256 + 128) * 4].to_vec())
        .collect();
    assert_eq!(bytes, want, "the tile is the narrowed image's window");
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

/// Recompositing an UNCHANGED 3-layer 512² stack. Measured 2026-10-04 on
/// Metal: 7 dispatches, submits and readbacks, 24 textures, 24.6 MB up —
/// the whole fold again, the accumulator re-uploaded per layer. The
/// resident fold hands back its last result: nothing changed, nothing
/// to fold.
const COMPOSITE3: [(&str, u64); 6] = [
    ("pipelines_built", 0),  // was 7: the per-device pipeline cache
    ("dispatches", 0),       // was 7: the resident fold's last-result cache
    ("submits", 0),          // was 7
    ("textures_created", 0), // was 24
    ("readbacks", 0),        // was 7
    ("bytes_uploaded", 0),   // was 24641576
];

/// One composite of an unchanged 3-layer 512² stack.
#[test]
fn a_three_layer_composite__feat__image_editor_layers() {
    let Some(ctx) = device() else { return };
    let s = stack(512, 512, 3);
    pollster::block_on(s.composite(Some(ctx), None)).expect("warm-up");
    counters::reset();
    let (out, e, g) = counters::measure(|| pollster::block_on(s.composite(Some(ctx), None)));
    out.expect("composite");
    let mut rows = vec![
        ("composites", e.composites, 1),
        ("layers_folded", e.layers_folded, 0), // was 3
    ];
    for ((name, v), (_, b)) in gpu_rows(&g).into_iter().zip(COMPOSITE3) {
        rows.push((name, v, b));
    }
    check("3-layer composite 512²", &rows);
}

/// The FIRST composite of a freshly opened 3-layer 512² stack (the
/// device already warm): every plate uploaded once, the whole fold in
/// one submit with one readback. Before the resident fold this was the
/// 7/7/7/24/24.6 MB of `COMPOSITE3`.
const COMPOSITE3_COLD: [(&str, u64); 6] = [
    ("pipelines_built", 0),
    ("dispatches", 7),
    ("submits", 1),
    ("textures_created", 0),
    ("readbacks", 1),
    ("bytes_uploaded", 6291496),
];

#[test]
fn a_fresh_stacks_first_composite__feat__image_editor_layers() {
    let Some(ctx) = device() else { return };
    // Warm the device (pipelines, scratch textures) on another stack.
    let warm = stack(512, 512, 3);
    pollster::block_on(warm.composite(Some(ctx), None)).expect("warm-up");
    drop(warm);
    let s = stack(512, 512, 3);
    counters::reset();
    let (out, e, g) = counters::measure(|| pollster::block_on(s.composite(Some(ctx), None)));
    out.expect("composite");
    let mut rows = vec![
        ("composites", e.composites, 1),
        ("layers_folded", e.layers_folded, 3),
    ];
    for ((name, v), (_, b)) in gpu_rows(&g).into_iter().zip(COMPOSITE3_COLD) {
        rows.push((name, v, b));
    }
    check("fresh 3-layer composite 512²", &rows);
}

/// An opacity change on the BOTTOM layer: below the checkpoint, so the
/// whole stack re-folds — from plates already on the device.
const COMPOSITE3_BOTTOM: [(&str, u64); 6] = [
    ("pipelines_built", 0),
    ("dispatches", 4),
    ("submits", 1),
    ("textures_created", 0),
    ("readbacks", 1),
    ("bytes_uploaded", 28),
];

#[test]
fn a_bottom_layer_opacity_change__feat__image_editor_layers() {
    let Some(ctx) = device() else { return };
    let mut s = stack(512, 512, 3);
    pollster::block_on(s.composite(Some(ctx), None)).expect("warm-up");
    s.set_opacity(0, 0.5).expect("opacity");
    counters::reset();
    let (out, e, g) = counters::measure(|| pollster::block_on(s.composite(Some(ctx), None)));
    out.expect("composite");
    let mut rows = vec![
        ("composites", e.composites, 1),
        ("layers_folded", e.layers_folded, 3),
    ];
    for ((name, v), (_, b)) in gpu_rows(&g).into_iter().zip(COMPOSITE3_BOTTOM) {
        rows.push((name, v, b));
    }
    check("bottom-layer opacity change 512²", &rows);
}

/// A 20-step opacity drag on the top (active) layer. Measured 2026-10-04
/// on Metal: 20 × the whole fold. Now each step re-folds from the
/// checkpoint below the active layer — one blend and the unpremultiply,
/// one submit, one readback — and the first step (opacity 1.0, which it
/// already had) is the unchanged stack.
const DRAG20: [(&str, u64); 6] = [
    ("pipelines_built", 0),  // was 140: the per-device pipeline cache
    ("dispatches", 38),      // was 140: the resident fold
    ("submits", 19),         // was 140
    ("textures_created", 0), // was 480
    ("readbacks", 19),       // was 140
    ("bytes_uploaded", 228), // was 492831520: params only
];

#[test]
fn a_twenty_step_opacity_drag__feat__image_editor_layers() {
    let Some(ctx) = device() else { return };
    let mut s = stack(512, 512, 3);
    pollster::block_on(s.composite(Some(ctx), None)).expect("warm-up");
    counters::reset();
    let ((), e, g) = counters::measure(|| {
        for step in 0..20 {
            s.set_opacity(2, 1.0 - step as f32 / 40.0).expect("opacity");
            pollster::block_on(s.composite(Some(ctx), None)).expect("composite");
        }
    });
    let mut rows = vec![
        ("composites", e.composites, 20),
        ("layers_folded", e.layers_folded, 19), // was 60
    ];
    for ((name, v), (_, b)) in gpu_rows(&g).into_iter().zip(DRAG20) {
        rows.push((name, v, b));
    }
    check("20-step opacity drag 512²", &rows);
}

// ── Apply (the adjust chain) ─────────────────────────────────────────

/// Measured 2026-10-04 on Metal: every stage, every tile, its own pipeline
/// build, upload, submit and readback. Now each node's tiles go out in
/// one submit, and a run of point stages chains on the device per tile
/// (one upload in, one readback out); the dispatches are the same ones.
const APPLY6: [(&str, u64); 6] = [
    ("pipelines_built", 0),       // was 276: the per-device pipeline cache
    ("dispatches", 276),          // unchanged: the same dispatches
    ("submits", 18),              // was 276: resident chains
    ("textures_created", 0),      // was 832: scratch pool + constant mask
    ("readbacks", 18),            // was 276
    ("bytes_uploaded", 26098352), // was 54978224: no intermediate re-uploads
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
    pollster::block_on(adjust_rgba8(ctx, &img, &p, None)).expect("warm-up");
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

// ── brush (60 samples) ───────────────────────────────────────────────

/// One stroke the way the `brush_stroke_extend` door runs it, sample
/// by sample: extend the session, copy its pixels out, composite the
/// stack with them standing in for the active layer, copy the preview
/// out. Returns the number of samples that painted.
fn brush_stroke(ctx: &GpuContext, s: &LayerStack, samples: usize) -> usize {
    use image_js::stroke::{StrokeParams, StrokeSession, StrokeTool};
    let mut params = StrokeParams::defaults(StrokeTool::Brush);
    params.color = [0.9, 0.2, 0.1, 1.0];
    let mut session = StrokeSession::begin_on(
        1,
        s.width(),
        s.height(),
        s.active().rgba.raw_arc(),
        params,
        None,
    )
    .expect("begin");
    let mut painted = 0;
    for i in 0..samples {
        let sample =
            image_gpu::StrokeSample::new(100.0 + i as f32 * 4.0, 200.0 + i as f32 * 2.0, 1.0);
        if pollster::block_on(session.extend(ctx, sample)).expect("extend") {
            painted += 1;
        }
        let px: Arc<[u8]> = Arc::from(session.pixels().to_vec().into_boxed_slice());
        let preview = pollster::block_on(s.composite(Some(ctx), Some(&px))).expect("preview");
        let _bytes = preview.to_vec();
    }
    painted
}

/// A 60-sample brush stroke on the top layer of a 3-layer 512² stack,
/// previewed through the stack after every sample (the door's work).
/// Measured 2026-10-04 on Metal with the resident fold, before the
/// brush work: every painted sample re-uploaded the whole active plate
/// and re-folded from the checkpoint at full size, and the stroke
/// compositor submitted and read back every dispatch. Now the fold
/// re-derives only the rectangle the sample changed (splicing it into
/// the last result in place), the compositor runs as one batch, and a
/// sample that painted nothing costs nothing.
const BRUSH60: [(&str, u64); 6] = [
    ("pipelines_built", 0),
    ("dispatches", 308),        // was 356
    ("submits", 88),            // was 236
    ("textures_created", 6),    // was 572
    ("readbacks", 88),          // was 236
    ("bytes_uploaded", 848822), // was 127369120
];

#[test]
fn a_sixty_sample_brush_stroke__feat__image_editor_paint() {
    let Some(ctx) = device() else { return };
    let s = stack(512, 512, 3);
    pollster::block_on(s.composite(Some(ctx), None)).expect("warm-up");
    brush_stroke(ctx, &s, 60);
    counters::reset();
    let (painted, e, g) = counters::measure(|| brush_stroke(ctx, &s, 60));
    let mut rows = vec![
        ("painted_samples", painted as u64, 44),
        ("composites", e.composites, 60),
        ("layers_folded", e.layers_folded, 44), // was 60
        ("whole_image_copies", e.whole_image_copies, 0),
    ];
    for ((name, v), (_, b)) in gpu_rows(&g).into_iter().zip(BRUSH60) {
        rows.push((name, v, b));
    }
    check("60-sample brush 512²", &rows);
}
