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

//! The BRUSH/DAB engine on the device — `image_gpu::stroke` +
//! `image_gpu::dab` end to end.
//!
//! `selection_mask.rs` proves the ABI's `mix(a, result, m)` contract for
//! unary point kernels. This file proves the four things the painting
//! lane builds on top of it, all on real GPU output:
//!
//! 1. A dab composited under the accumulated coverage lands the brush
//!    colour, with the tip's antialiased rim actually antialiased (and
//!    the pencil's actually not).
//! 2. **Selection clipping**: with a selection bound, every texel
//!    outside it comes back BIT-IDENTICAL to the backdrop — the claim
//!    "a brush cannot paint outside a selection" checked on bytes.
//! 3. The identity the whole design rests on,
//!    `mix(a, over(a, b, α), m) ≡ over(a, b, α·m)`, for every blend
//!    mode — which is what makes "opacity rides the mask" correct
//!    rather than merely convenient.
//! 4. Windowing exactness: compositing a sub-rectangle equals the
//!    corresponding texels of a whole-image composite, which is what
//!    licenses the dirty-rectangle incremental path.
//!
//! feat: image.editor.paint

use half::f16;
use image_conformance::device::test_device;
use image_core::Region;
use image_gpu::dab::{BrushTip, StrokeAccumulator};
use image_gpu::{composite_stroke_window, GpuContext, PaintMode, SelectionCoverage};
use image_kernels::families::compose::ComposeParams;
use image_kernels::families::compose::{
    COMPOSE_DARKEN, COMPOSE_HUE, COMPOSE_MULTIPLY, COMPOSE_NORMAL, COMPOSE_OVERLAY, COMPOSE_SCREEN,
    COMPOSE_SOFT_LIGHT,
};
use image_kernels::KernelDef;

const W: u32 = 32;
const H: u32 = 32;

/// A deterministic non-uniform backdrop, kept away from 0 and 1 so every
/// blend mode moves it and nothing clips.
fn backdrop(w: u32, h: u32, alpha: f32) -> Vec<u8> {
    let mut px = Vec::with_capacity((w * h) as usize * 8);
    for y in 0..h {
        for x in 0..w {
            let c = [
                0.15 + 0.5 * (x as f32 / w as f32),
                0.25 + 0.4 * (y as f32 / h as f32),
                0.35 + 0.3 * ((x + y) as f32 / (w + h) as f32),
                alpha,
            ];
            for v in c {
                px.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
            }
        }
    }
    px
}

/// One channel of one texel of an rgba16float buffer, as f32.
fn texel(bytes: &[u8], w: u32, x: u32, y: u32, c: usize) -> f32 {
    let i = (((y * w + x) as usize) * 4 + c) * 2;
    f16::from_bits(u16::from_le_bytes([bytes[i], bytes[i + 1]])).to_f32()
}

/// The raw 8 bytes of one texel — for BIT-identity assertions.
fn texel_bytes(bytes: &[u8], w: u32, x: u32, y: u32) -> &[u8] {
    let i = ((y * w + x) as usize) * 8;
    &bytes[i..i + 8]
}

/// A single centred dab's accumulator.
fn dabbed(w: u32, h: u32, tip: BrushTip, flow: f32) -> StrokeAccumulator {
    let mut acc = StrokeAccumulator::new(w, h);
    acc.stamp(&tip, w as f32 / 2.0, h as f32 / 2.0, flow);
    acc
}

fn paint(color: [f32; 4], blend: &'static KernelDef) -> PaintMode {
    PaintMode::Paint { blend, color }
}

// ─────────────────────── 1. the dab lands ────────────────────────────

#[test]
fn image_editor_paint_a_dab_composites_the_brush_colour() {
    let Some(ctx) = test_device() else { return };
    let base = backdrop(W, H, 1.0);
    let acc = dabbed(W, H, BrushTip::soft(16.0, 1.0), 1.0);
    let mask = acc.mask_window_f16(Region::new(0, 0, W, H), 1.0, None);
    let out = pollster::block_on(composite_stroke_window(
        ctx,
        &paint([1.0, 0.0, 0.0, 1.0], &COMPOSE_NORMAL),
        &base,
        &mask,
        W,
        H,
    ))
    .expect("composite");

    // The dab's solid interior IS the brush colour.
    assert!(
        (texel(&out, W, 16, 16, 0) - 1.0).abs() < 2e-3,
        "centre red = {}",
        texel(&out, W, 16, 16, 0)
    );
    assert!(texel(&out, W, 16, 16, 1) < 2e-3, "centre green cleared");

    // Far outside the dab the backdrop survives BIT-for-bit.
    assert_eq!(
        texel_bytes(&out, W, 0, 0),
        texel_bytes(&base, W, 0, 0),
        "a texel the dab never reached must be untouched"
    );
}

#[test]
fn image_editor_paint_a_soft_tip_leaves_an_antialiased_rim() {
    let Some(ctx) = test_device() else { return };
    let base = backdrop(W, H, 1.0);
    let acc = dabbed(W, H, BrushTip::soft(16.0, 0.4), 1.0);
    let mask = acc.mask_window_f16(Region::new(0, 0, W, H), 1.0, None);
    let out = pollster::block_on(composite_stroke_window(
        ctx,
        &paint([1.0, 0.0, 0.0, 1.0], &COMPOSE_NORMAL),
        &base,
        &mask,
        W,
        H,
    ))
    .expect("composite");

    // Scan the dab's horizontal diameter: the red channel must take many
    // distinct intermediate values — that IS the falloff, on the device.
    let mut intermediate = 0;
    for x in 0..W {
        let r = texel(&out, W, x, 16, 0);
        let b = texel(&base, W, x, 16, 0);
        if r > b + 0.02 && r < 1.0 - 0.02 {
            intermediate += 1;
        }
    }
    assert!(
        intermediate >= 6,
        "a soft tip should leave a wide ramp, found {intermediate} intermediate texels"
    );
}

#[test]
fn image_editor_paint_the_pencil_tip_leaves_a_hard_aliased_edge() {
    let Some(ctx) = test_device() else { return };
    let base = backdrop(W, H, 1.0);
    let acc = dabbed(W, H, BrushTip::hard(16.0), 1.0);
    let mask = acc.mask_window_f16(Region::new(0, 0, W, H), 1.0, None);
    let out = pollster::block_on(composite_stroke_window(
        ctx,
        &paint([1.0, 0.0, 0.0, 1.0], &COMPOSE_NORMAL),
        &base,
        &mask,
        W,
        H,
    ))
    .expect("composite");

    // Every texel on the diameter is either untouched backdrop or full
    // paint — no ramp anywhere. That is the pencil.
    for x in 0..W {
        let r = texel(&out, W, x, 16, 0);
        let b = texel(&base, W, x, 16, 0);
        let painted = (r - 1.0).abs() < 2e-3;
        let untouched = (r - b).abs() < 2e-3;
        assert!(
            painted || untouched,
            "pencil texel ({x},16) is a partial {r} (backdrop {b})"
        );
    }
}

// ────────────────── 2. selection clipping, on bytes ──────────────────

#[test]
fn image_editor_paint_a_selection_clips_the_stroke_bit_for_bit() {
    let Some(ctx) = test_device() else { return };
    let base = backdrop(W, H, 1.0);
    // A dab centred on the image, a selection covering only the LEFT
    // half: the dab straddles the edge, so this is the real case.
    let acc = dabbed(W, H, BrushTip::soft(20.0, 1.0), 1.0);
    let sel = SelectionCoverage::rasterize_rect(W, H, 0.0, 0.0, 16.0, H as f32);
    let mask = acc.mask_window_f16(Region::new(0, 0, W, H), 1.0, Some(&sel));
    let out = pollster::block_on(composite_stroke_window(
        ctx,
        &paint([1.0, 0.0, 0.0, 1.0], &COMPOSE_NORMAL),
        &base,
        &mask,
        W,
        H,
    ))
    .expect("composite");

    for y in 0..H {
        for x in 16..W {
            assert_eq!(
                texel_bytes(&out, W, x, y),
                texel_bytes(&base, W, x, y),
                "({x},{y}) is outside the selection and MUST be untouched"
            );
        }
    }
    // …and inside it the brush really did paint.
    assert!(
        texel(&out, W, 10, 16, 0) > texel(&base, W, 10, 16, 0) + 0.2,
        "the dab paints inside the selection"
    );
}

#[test]
fn image_editor_paint_a_feathered_selection_edge_blends_the_stroke_once() {
    let Some(ctx) = test_device() else { return };
    let base = backdrop(W, H, 1.0);
    let acc = dabbed(W, H, BrushTip::soft(24.0, 1.0), 1.0);
    let mut sel = SelectionCoverage::rasterize_rect(W, H, 0.0, 0.0, 16.0, H as f32);
    sel.feather(3.0);
    let mask = acc.mask_window_f16(Region::new(0, 0, W, H), 1.0, Some(&sel));
    let out = pollster::block_on(composite_stroke_window(
        ctx,
        &paint([1.0, 0.0, 0.0, 1.0], &COMPOSE_NORMAL),
        &base,
        &mask,
        W,
        H,
    ))
    .expect("composite");

    // Across the feathered seam the painted red decreases monotonically —
    // one smooth blend, not a double-darkened edge.
    let reds: Vec<f32> = (12..22).map(|x| texel(&out, W, x, 16, 0)).collect();
    for i in 1..reds.len() {
        assert!(
            reds[i] <= reds[i - 1] + 2e-3,
            "seam is not monotone: {reds:?}"
        );
    }
    assert!(
        reds[0] > reds[reds.len() - 1] + 0.2,
        "the seam actually fades"
    );
}

// ───────── 3. the identity: opacity may ride the mask ────────────────

/// `mix(a, over(a, b, α), m) ≡ over(a, b, α·m)`.
///
/// Checked by running the SAME compose kernel two ways — opacity in the
/// param with a coverage mask, versus the product in the opacity param
/// with a constant-1 mask — and requiring the outputs to agree. This is
/// the licence for folding brush opacity into the coverage instead of
/// the compose param, and it must hold for separable AND non-separable
/// blend modes alike.
fn assert_mask_opacity_identity(ctx: &GpuContext, blend: &'static KernelDef) {
    let base = backdrop(W, H, 1.0);
    let alpha = 0.4f32;
    let coverage = 0.6f32;

    // Route A: paint at opacity 1, mask = α·m (what the stroke lane does).
    let acc_bytes = image_gpu::SelectionMask::from_fn(W, H, |_, _| alpha * coverage).into_bytes();
    let via_mask = pollster::block_on(composite_stroke_window(
        ctx,
        &paint([0.9, 0.3, 0.1, 1.0], blend),
        &base,
        &acc_bytes,
        W,
        H,
    ))
    .expect("composite via mask");

    // Route B: the same blend at opacity α·m with a constant-1 mask,
    // driven through the identical premultiply bracket so only the
    // opacity/mask split differs.
    let full = image_gpu::SelectionMask::from_fn(W, H, |_, _| 1.0).into_bytes();
    let paint_field = pollster::block_on(image_gpu::execute_tile_once_async(
        ctx,
        &image_kernels::families::gen::GEN_SOLID,
        &[image_gpu::TileInput { f16_bytes: &base }],
        image_kernels::families::gen::GenSolidParams::new(0, 0, 0.9, 0.3, 0.1, 1.0).as_bytes(),
        None,
        W,
        H,
    ))
    .expect("solid");
    let base_premul = pollster::block_on(image_gpu::execute_tile_once_async(
        ctx,
        &image_kernels::families::cast::CAST_PREMULTIPLY,
        &[image_gpu::TileInput { f16_bytes: &base }],
        image_kernels::families::cast::CastPremultiplyParams::new().as_bytes(),
        None,
        W,
        H,
    ))
    .expect("premul");
    let composed = pollster::block_on(image_gpu::execute_tile_once_async(
        ctx,
        blend,
        &[
            image_gpu::TileInput {
                f16_bytes: &base_premul,
            },
            image_gpu::TileInput {
                f16_bytes: &paint_field,
            },
        ],
        ComposeParams::new(alpha * coverage).as_bytes(),
        Some(&full),
        W,
        H,
    ))
    .expect("compose");
    let via_opacity = pollster::block_on(image_gpu::execute_tile_once_async(
        ctx,
        &image_kernels::families::cast::CAST_UNPREMULTIPLY,
        &[image_gpu::TileInput {
            f16_bytes: &composed,
        }],
        image_kernels::families::cast::CastUnpremultiplyParams::new().as_bytes(),
        None,
        W,
        H,
    ))
    .expect("unpremul");

    for y in (0..H).step_by(7) {
        for x in (0..W).step_by(5) {
            for c in 0..4 {
                let a = texel(&via_mask, W, x, y, c);
                let b = texel(&via_opacity, W, x, y, c);
                assert!(
                    (a - b).abs() < 4e-3,
                    "{}: ({x},{y}) channel {c}: mask route {a} vs opacity route {b}",
                    blend.id
                );
            }
        }
    }
}

#[test]
fn image_editor_paint_opacity_may_ride_the_mask_for_every_blend_class() {
    let Some(ctx) = test_device() else { return };
    for blend in [
        &COMPOSE_NORMAL,
        &COMPOSE_MULTIPLY,
        &COMPOSE_SCREEN,
        &COMPOSE_OVERLAY,
        &COMPOSE_DARKEN,
        &COMPOSE_SOFT_LIGHT,
        // …and a NON-separable one (W3C §10.3), the class most likely to
        // break the identity if the spine were not source-over.
        &COMPOSE_HUE,
    ] {
        assert_mask_opacity_identity(ctx, blend);
    }
}

// ───────────────────────── 4. the eraser ─────────────────────────────

#[test]
fn image_editor_paint_the_eraser_removes_alpha_and_keeps_the_colour() {
    let Some(ctx) = test_device() else { return };
    let base = backdrop(W, H, 1.0);
    let acc = dabbed(W, H, BrushTip::soft(16.0, 1.0), 1.0);
    let mask = acc.mask_window_f16(Region::new(0, 0, W, H), 1.0, None);
    let out = pollster::block_on(composite_stroke_window(
        ctx,
        &PaintMode::Erase,
        &base,
        &mask,
        W,
        H,
    ))
    .expect("erase");

    // Fully covered: alpha gone, RGB preserved (straight-space
    // destination-out — a partially erased pixel must not decay toward
    // black, which is exactly what a premultiplied round-trip would do).
    assert!(
        texel(&out, W, 16, 16, 3) < 2e-3,
        "alpha erased at the centre"
    );
    for c in 0..3 {
        assert!(
            (texel(&out, W, 16, 16, c) - texel(&base, W, 16, 16, c)).abs() < 2e-3,
            "channel {c} colour preserved under a fully erased texel"
        );
    }
    // Untouched outside.
    assert_eq!(texel_bytes(&out, W, 0, 0), texel_bytes(&base, W, 0, 0));
}

#[test]
fn image_editor_paint_a_half_covered_eraser_texel_halves_the_alpha() {
    let Some(ctx) = test_device() else { return };
    let base = backdrop(W, H, 0.8);
    let half = image_gpu::SelectionMask::from_fn(W, H, |_, _| 0.5).into_bytes();
    let out = pollster::block_on(composite_stroke_window(
        ctx,
        &PaintMode::Erase,
        &base,
        &half,
        W,
        H,
    ))
    .expect("erase");
    // a.a · (1 − 0.5) = 0.4
    assert!(
        (texel(&out, W, 8, 8, 3) - 0.4).abs() < 2e-3,
        "alpha {} should be 0.4",
        texel(&out, W, 8, 8, 3)
    );
}

#[test]
fn image_editor_paint_the_eraser_honours_the_selection() {
    let Some(ctx) = test_device() else { return };
    let base = backdrop(W, H, 1.0);
    let acc = dabbed(W, H, BrushTip::soft(20.0, 1.0), 1.0);
    let sel = SelectionCoverage::rasterize_rect(W, H, 0.0, 0.0, 16.0, H as f32);
    let mask = acc.mask_window_f16(Region::new(0, 0, W, H), 1.0, Some(&sel));
    let out = pollster::block_on(composite_stroke_window(
        ctx,
        &PaintMode::Erase,
        &base,
        &mask,
        W,
        H,
    ))
    .expect("erase");
    for y in 0..H {
        for x in 16..W {
            assert_eq!(
                texel_bytes(&out, W, x, y),
                texel_bytes(&base, W, x, y),
                "the eraser must not reach ({x},{y}) outside the selection"
            );
        }
    }
    assert!(
        texel(&out, W, 10, 16, 3) < 0.5,
        "erased inside the selection"
    );
}

// ───────── 5. windowing exactness (the incremental licence) ──────────

#[test]
fn image_editor_paint_a_window_composite_matches_the_whole_image() {
    // The incremental path re-composites only the DIRTY rectangle. That
    // is only sound if a dispatch over a sub-window produces exactly the
    // texels a whole-image dispatch would — i.e. the composite has no
    // window-dependent state. Proven here rather than assumed.
    let Some(ctx) = test_device() else { return };
    let base = backdrop(W, H, 1.0);
    let mut acc = StrokeAccumulator::new(W, H);
    let tip = BrushTip::soft(9.0, 0.5);
    for i in 0..6 {
        acc.stamp(&tip, 8.0 + i as f32 * 2.5, 12.0 + i as f32 * 1.5, 0.4);
    }
    let mode = paint([0.1, 0.8, 0.4, 1.0], &COMPOSE_NORMAL);

    let whole = pollster::block_on(composite_stroke_window(
        ctx,
        &mode,
        &base,
        &acc.mask_window_f16(Region::new(0, 0, W, H), 0.7, None),
        W,
        H,
    ))
    .expect("whole");

    let win = Region::new(6, 10, 20, 14);
    let mut base_win = Vec::new();
    for y in 0..win.h {
        let row = (win.y as u32 + y) as usize;
        let start = (row * W as usize + win.x as usize) * 8;
        base_win.extend_from_slice(&base[start..start + (win.w as usize) * 8]);
    }
    let part = pollster::block_on(composite_stroke_window(
        ctx,
        &mode,
        &base_win,
        &acc.mask_window_f16(win, 0.7, None),
        win.w,
        win.h,
    ))
    .expect("window");

    for y in 0..win.h {
        for x in 0..win.w {
            assert_eq!(
                texel_bytes(&part, win.w, x, y),
                texel_bytes(&whole, W, win.x as u32 + x, win.y as u32 + y),
                "window texel ({x},{y}) diverged from the whole-image composite"
            );
        }
    }
}

#[test]
fn image_editor_paint_the_same_stroke_always_produces_the_same_bytes() {
    // Actions and scripts replay strokes; the device output must be a
    // pure function of the samples.
    let Some(ctx) = test_device() else { return };
    let base = backdrop(W, H, 1.0);
    let mut acc = StrokeAccumulator::new(W, H);
    let tip = BrushTip::soft(7.0, 0.25);
    for i in 0..10 {
        acc.stamp(&tip, 5.0 + i as f32 * 2.25, 6.0 + i as f32 * 1.75, 0.3);
    }
    let mask = acc.mask_window_f16(Region::new(0, 0, W, H), 0.85, None);
    let mode = paint([0.3, 0.1, 0.9, 1.0], &COMPOSE_MULTIPLY);
    let a =
        pollster::block_on(composite_stroke_window(ctx, &mode, &base, &mask, W, H)).expect("first");
    let b = pollster::block_on(composite_stroke_window(ctx, &mode, &base, &mask, W, H))
        .expect("second");
    assert_eq!(a, b, "the same stroke must produce byte-identical pixels");
}

// ─────────── 6. the premultiply bracket actually matters ─────────────

#[test]
fn image_editor_paint_over_a_translucent_backdrop_uses_the_right_colour() {
    // Painting a HALF-OPAQUE white over a translucent backdrop: with the
    // premultiply bracket the backdrop's colour enters the blend
    // undissociated, so the result sits between the two colours. Without
    // it the backdrop would be read as already-associated and the
    // recovered colour would be wrong (too bright) — this is the case the
    // bracket exists for, and the case an eraser stroke creates.
    let Some(ctx) = test_device() else { return };
    let base = backdrop(W, H, 0.5);
    let full = image_gpu::SelectionMask::from_fn(W, H, |_, _| 1.0).into_bytes();
    let out = pollster::block_on(composite_stroke_window(
        ctx,
        &paint([1.0, 1.0, 1.0, 0.5], &COMPOSE_NORMAL),
        &base,
        &full,
        W,
        H,
    ))
    .expect("composite");

    // Source-over of a 0.5-alpha white onto a 0.5-alpha backdrop:
    //   αo = 0.5 + 0.5·0.5 = 0.75
    //   Co·αo = 0.5·1 + 0.5·0.5·cb  ⇒  Co = (0.5 + 0.25·cb) / 0.75
    for (x, y) in [(4u32, 4u32), (20, 9), (30, 25)] {
        let alpha = texel(&out, W, x, y, 3);
        assert!((alpha - 0.75).abs() < 3e-3, "alpha {alpha} at ({x},{y})");
        for c in 0..3 {
            let cb = texel(&base, W, x, y, c);
            let want = (0.5 + 0.25 * cb) / 0.75;
            let got = texel(&out, W, x, y, c);
            assert!(
                (got - want).abs() < 5e-3,
                "({x},{y}) channel {c}: got {got}, source-over says {want}"
            );
        }
    }
}

#[test]
fn image_editor_paint_the_premultiply_bracket_is_the_identity_when_opaque() {
    // The licence for `stroke::window_is_opaque` skipping two dispatches:
    // over a fully opaque window, premultiply→unpremultiply must return
    // the input BIT-for-bit. Checked on the device, not argued.
    let Some(ctx) = test_device() else { return };
    let base = backdrop(W, H, 1.0);
    assert!(image_gpu::stroke::window_is_opaque(&base));

    let premul = pollster::block_on(image_gpu::execute_tile_once_async(
        ctx,
        &image_kernels::families::cast::CAST_PREMULTIPLY,
        &[image_gpu::TileInput { f16_bytes: &base }],
        image_kernels::families::cast::CastPremultiplyParams::new().as_bytes(),
        None,
        W,
        H,
    ))
    .expect("premul");
    assert_eq!(premul, base, "premultiply by 1 is the identity");

    let back = pollster::block_on(image_gpu::execute_tile_once_async(
        ctx,
        &image_kernels::families::cast::CAST_UNPREMULTIPLY,
        &[image_gpu::TileInput { f16_bytes: &premul }],
        image_kernels::families::cast::CastUnpremultiplyParams::new().as_bytes(),
        None,
        W,
        H,
    ))
    .expect("unpremul");
    assert_eq!(back, base, "unpremultiply by 1 is the identity");

    // And the composite over an opaque backdrop stays opaque, which is
    // why the trailing unpremultiply may be dropped as well.
    let acc = dabbed(W, H, BrushTip::soft(16.0, 0.5), 1.0);
    let out = pollster::block_on(composite_stroke_window(
        ctx,
        &paint([1.0, 0.2, 0.0, 0.6], &COMPOSE_NORMAL),
        &base,
        &acc.mask_window_f16(Region::new(0, 0, W, H), 0.7, None),
        W,
        H,
    ))
    .expect("composite");
    assert!(
        image_gpu::stroke::window_is_opaque(&out),
        "an opaque backdrop composites to an opaque result even under a \
         translucent brush colour and a partial mask"
    );
}

// ─────────────────── 7. argument validation ──────────────────────────

#[test]
fn image_editor_paint_a_mis_sized_window_is_rejected_not_dispatched() {
    let Some(ctx) = test_device() else { return };
    let base = backdrop(4, 4, 1.0);
    let mask = image_gpu::SelectionMask::from_fn(4, 4, |_, _| 1.0).into_bytes();
    assert!(pollster::block_on(composite_stroke_window(
        ctx,
        &PaintMode::Erase,
        &base,
        &mask,
        8,
        8
    ))
    .is_err());
    assert!(pollster::block_on(composite_stroke_window(
        ctx,
        &PaintMode::Erase,
        &base,
        &mask[..8],
        4,
        4
    ))
    .is_err());
}
