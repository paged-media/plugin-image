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

//! STROKE TAPES: a stroke applied INCREMENTALLY — one `extend` per
//! pointer sample, each re-compositing only its dirty rectangle — lands
//! on exactly the bytes a FROM-SCRATCH composite of the whole stroke
//! produces.
//!
//! That equality is what licenses the incremental path at all, and the
//! session's docs claim it. Here it is checked over seeded, generated
//! pointer streams (random walks with pressure, every non-sampling
//! tool, every blend mode, random tip/flow/opacity/spacing, with and
//! without a selection) rather than one hand-drawn line.
//!
//! The from-scratch side re-plans the same dabs with the same public
//! pieces the session is built from (`plan_segment`, the accumulator,
//! `composite_stroke_window`), composites the WHOLE canvas once, and
//! writes back wherever the effective coverage is non-zero.
//!
//! GPU-only (the composite is a kernel dispatch); skips without a device
//! unless `REQUIRE_GPU=1`.

use std::sync::Arc;

use image_core::Region;
use image_gpu::{
    composite_stroke_window, plan_segment, Dab, PressureTarget, SelectionCoverage,
    StrokeAccumulator, StrokeSample, StrokeWalk,
};
use image_js::stroke::{blend_kernel, blend_names, StrokeParams, StrokeSession, StrokeTool};

/// The engine's straight-RGBA8 ↔ rgba16float bridge (`/255` up,
/// clamp-and-round down) — crate-private in `image_js::fill`, restated
/// here so the from-scratch side uses the same rounding rule.
fn rgba8_to_f16(rgba: &[u8]) -> Vec<u8> {
    rgba.iter()
        .flat_map(|&b| half::f16::from_f32(b as f32 / 255.0).to_le_bytes())
        .collect()
}

fn f16_to_rgba8(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks_exact(2)
        .map(|p| {
            let v = half::f16::from_le_bytes([p[0], p[1]]).to_f32();
            (v.clamp(0.0, 1.0) * 255.0).round() as u8
        })
        .collect()
}

const W: u32 = 96;
const H: u32 = 64;

/// SplitMix64 — a tape is a seed, so a failure names its seed and can be
/// replayed exactly.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    fn pick(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

struct Tape {
    params: StrokeParams,
    samples: Vec<StrokeSample>,
    selection: Option<Arc<SelectionCoverage>>,
}

fn tape(seed: u64) -> Tape {
    let mut r = Rng(seed);
    let tool = [StrokeTool::Brush, StrokeTool::Pencil, StrokeTool::Eraser][r.pick(3)];
    let names = blend_names();
    let mut params = StrokeParams::defaults(tool);
    params.size = r.range(2.0, 36.0);
    params.hardness = r.unit();
    params.opacity = r.range(0.1, 1.0);
    params.flow = r.range(0.1, 1.0);
    params.spacing = r.range(0.05, 1.5);
    params.blend = blend_kernel(names[r.pick(names.len())]).expect("registered blend");
    params.color = [r.unit(), r.unit(), r.unit(), 1.0];
    params.pressure = [
        PressureTarget::None,
        PressureTarget::Size,
        PressureTarget::Opacity,
        PressureTarget::Both,
    ][r.pick(4)];
    // A random walk that may leave and re-enter the canvas, with a
    // repeated sample now and then (a stationary pointer).
    let n = 2 + r.pick(40);
    let (mut x, mut y) = (r.range(0.0, W as f32), r.range(0.0, H as f32));
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        samples.push(StrokeSample::new(x, y, r.range(0.05, 1.0)));
        if r.pick(6) != 0 {
            x += r.range(-14.0, 14.0);
            y += r.range(-10.0, 10.0);
        }
    }
    let selection = if r.pick(3) == 0 {
        let (sx, sy) = (r.range(0.0, W as f32 / 2.0), r.range(0.0, H as f32 / 2.0));
        let mut s = SelectionCoverage::rasterize_ellipse(W, H, sx + 20.0, sy + 15.0, 26.0, 18.0);
        s.feather(r.range(0.0, 3.0));
        Some(Arc::new(s))
    } else {
        None
    };
    Tape {
        params,
        samples,
        selection,
    }
}

fn base(seed: u64) -> Arc<[u8]> {
    let mut r = Rng(seed ^ 0xBA5E);
    let px: Vec<u8> = (0..W * H)
        .flat_map(|i| {
            let (x, y) = (i % W, i / W);
            let a = if (x / 16 + y / 16) % 3 == 0 {
                (r.next() % 256) as u8
            } else {
                255
            };
            [(x * 2) as u8, (y * 3) as u8, (r.next() % 256) as u8, a]
        })
        .collect();
    Arc::from(px)
}

fn incremental(ctx: &image_gpu::GpuContext, t: &Tape, base: &Arc<[u8]>) -> Vec<u8> {
    let mut s = StrokeSession::begin_on(1, W, H, Arc::clone(base), t.params, t.selection.clone())
        .expect("begin");
    for smp in &t.samples {
        pollster::block_on(s.extend(ctx, *smp)).expect("extend");
    }
    s.commit()
}

fn from_scratch(ctx: &image_gpu::GpuContext, t: &Tape, base: &[u8]) -> Vec<u8> {
    let p = t.params.sanitized();
    let mut acc = StrokeAccumulator::new(W, H);
    let mut walk = StrokeWalk::new();
    let mut last: Option<StrokeSample> = None;
    for smp in &t.samples {
        let mut dabs = Vec::new();
        match last {
            None => dabs.push(Dab {
                x: smp.x,
                y: smp.y,
                pressure: smp.pressure,
            }),
            Some(prev) => plan_segment(&mut walk, prev, *smp, p.step_px(), &mut dabs),
        }
        last = Some(*smp);
        for d in &dabs {
            acc.stamp(&p.tip_at(d.pressure), d.x, d.y, p.flow_at(d.pressure));
        }
    }
    if acc.is_empty() {
        return base.to_vec();
    }
    let all = Region::new(0, 0, W, H);
    let sel = t.selection.as_deref();
    let mask = acc.mask_window_f16(all, p.opacity, sel);
    let mode = p.solid_paint_mode().expect("a generated tool");
    let out = pollster::block_on(composite_stroke_window(
        ctx,
        &mode,
        &rgba8_to_f16(base),
        &mask,
        W,
        H,
    ))
    .expect("composite");
    let out = f16_to_rgba8(&out);
    let mut px = base.to_vec();
    for y in 0..H {
        for x in 0..W {
            if acc.effective_at(x, y, p.opacity, sel) > 0.0 {
                let i = ((y * W + x) * 4) as usize;
                px[i..i + 4].copy_from_slice(&out[i..i + 4]);
            }
        }
    }
    px
}

const TAPES: u64 = 48;

#[test]
#[allow(non_snake_case)]
fn incremental_stroke_equals_from_scratch_over_seeded_tapes__feat__image_editor_paint() {
    let Some(ctx) = image_gpu::test_support::device_or_skip("stroke tapes") else {
        return;
    };
    let mut failures = Vec::new();
    for seed in 0..TAPES {
        let t = tape(seed);
        let b = base(seed);
        let inc = incremental(ctx, &t, &b);
        let full = from_scratch(ctx, &t, &b);
        if inc != full {
            let bad = inc
                .chunks_exact(4)
                .zip(full.chunks_exact(4))
                .filter(|(a, b)| a != b)
                .count();
            let worst = inc
                .iter()
                .zip(&full)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            failures.push(format!(
                "seed {seed} ({} {}, {} samples): {bad} texel(s) differ, worst {worst} level(s)",
                t.params.tool.as_wire(),
                t.params.blend.id,
                t.samples.len()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "incremental != from-scratch on {} of {TAPES} tapes:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
#[allow(non_snake_case)]
fn a_tape_replays_to_identical_bytes_and_never_paints_outside_its_bounds__feat__image_editor_paint()
{
    let Some(ctx) = image_gpu::test_support::device_or_skip("stroke tapes") else {
        return;
    };
    for seed in 100..100 + TAPES / 4 {
        let t = tape(seed);
        let b = base(seed);
        let one = incremental(ctx, &t, &b);
        let two = incremental(ctx, &t, &b);
        assert!(
            one == two,
            "seed {seed}: the same tape replayed differently"
        );
        let mut s = StrokeSession::begin_on(1, W, H, Arc::clone(&b), t.params, t.selection.clone())
            .expect("begin");
        for smp in &t.samples {
            pollster::block_on(s.extend(ctx, *smp)).expect("extend");
        }
        let bounds = s.stroke_bounds();
        let px = s.commit();
        for y in 0..H {
            for x in 0..W {
                let inside = bounds.is_some_and(|r| {
                    (x as i32) >= r.x
                        && (y as i32) >= r.y
                        && (x as i32) < r.x + r.w as i32
                        && (y as i32) < r.y + r.h as i32
                });
                if !inside {
                    let i = ((y * W + x) * 4) as usize;
                    assert_eq!(
                        px[i..i + 4],
                        b[i..i + 4],
                        "seed {seed}: texel ({x},{y}) outside the stroke bounds changed"
                    );
                }
            }
        }
    }
}
