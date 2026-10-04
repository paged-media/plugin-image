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

//! The GPU-resident layer fold against the fold it replaced.
//!
//! The resident fold keeps plates, a checkpoint below the active layer
//! and the last result between composites, keyed by identity. Its
//! claim is that none of that changes a single byte: after ANY sequence
//! of edits, the composite equals the pre-residency fold's composite of
//! the same stack (every blend its own upload/dispatch/readback, nothing
//! cached), on the same device.
//!
//! Random stacks: pixel, smart and adjustment layers; opacity, blend,
//! visibility, masks, clipping, isolated and pass-through groups, a
//! 16-bit background. Random edit sequences over them, compared after
//! every edit.

#![allow(non_snake_case)]

use std::sync::Arc;

use image_core::Region;
use image_gpu::coverage::SelectionCoverage;
use image_gpu::{execute_tile_once, GpuContext, TileInput};
use image_js::ingest::AdjustParams;
use image_js::layers::LayerStack;
use image_js::pixels::Pixels;
use image_kernels::families::cast::{CastUnpremultiplyParams, CAST_UNPREMULTIPLY};
use proptest::prelude::*;

fn device() -> Option<&'static GpuContext> {
    image_gpu::test_support::device_or_skip("prop_layer_cache")
}

const BLENDS: &[&str] = &[
    "normal",
    "multiply",
    "screen",
    "overlay",
    "difference",
    "linear_dodge",
    "color",
];

/// Deterministic pixels: `alpha` 0 = opaque, 1 = translucent, 2 = holes.
fn pixels(w: u32, h: u32, seed: u32, alpha: u8) -> Arc<[u8]> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let s = seed.wrapping_mul(2654435761) ^ (x * 31 + y * 17);
            let a = match alpha {
                0 => 255,
                1 => 60 + (s % 150) as u8,
                _ => {
                    if (x + y).wrapping_add(seed).is_multiple_of(3) {
                        0
                    } else {
                        255
                    }
                }
            };
            v.extend_from_slice(&[
                (s % 251) as u8,
                (s / 7 % 253) as u8,
                (x * 9).wrapping_add(seed) as u8,
                a,
            ]);
        }
    }
    Arc::from(v)
}

fn mask(w: u32, h: u32, kind: u8) -> Arc<SelectionCoverage> {
    Arc::new(match kind {
        0 => SelectionCoverage::rasterize_rect(w, h, 2.0, 1.0, w as f32 / 2.0, h as f32 - 2.0),
        1 => SelectionCoverage::empty(w, h),
        _ => SelectionCoverage::from_data(w, h, (0..w * h).map(|i| (i * 37 % 256) as u8).collect())
            .expect("mask"),
    })
}

#[derive(Debug, Clone)]
enum LayerGen {
    Pixel { seed: u32, alpha: u8, smart: bool },
    Adjust { exposure: f32, saturation: f32 },
}

#[derive(Debug, Clone)]
enum Edit {
    Opacity(usize, f32),
    Blend(usize, usize),
    Visible(usize, bool),
    Paint(usize, u32, u8),
    Mask(usize, u8),
    MaskEnabled(usize, bool),
    Clip(usize, bool),
    Active(usize),
    Reorder(usize, usize),
    Group(usize, usize, bool, f32),
    Recomposite,
    /// A brush stroke on layer k: `samples` previews, each changing a
    /// small rectangle of the in-flight pixels, then the commit.
    Stroke(usize, u32, u8),
}

fn layer_gen() -> impl Strategy<Value = LayerGen> {
    prop_oneof![
        3 => (any::<u32>(), 0u8..3, any::<bool>())
            .prop_map(|(seed, alpha, smart)| LayerGen::Pixel { seed, alpha, smart }),
        1 => (-1.0f32..1.0, 0.0f32..2.0)
            .prop_map(|(exposure, saturation)| LayerGen::Adjust { exposure, saturation }),
    ]
}

fn edit_gen() -> impl Strategy<Value = Edit> {
    prop_oneof![
        (0usize..8, 0.0f32..1.0).prop_map(|(i, o)| Edit::Opacity(i, o)),
        (0usize..8, 0usize..BLENDS.len()).prop_map(|(i, b)| Edit::Blend(i, b)),
        (0usize..8, any::<bool>()).prop_map(|(i, v)| Edit::Visible(i, v)),
        (0usize..8, any::<u32>(), 0u8..3).prop_map(|(i, s, a)| Edit::Paint(i, s, a)),
        (0usize..8, 0u8..3).prop_map(|(i, k)| Edit::Mask(i, k)),
        (0usize..8, any::<bool>()).prop_map(|(i, e)| Edit::MaskEnabled(i, e)),
        (0usize..8, any::<bool>()).prop_map(|(i, c)| Edit::Clip(i, c)),
        (0usize..8).prop_map(Edit::Active),
        (0usize..8, 0usize..8).prop_map(|(a, b)| Edit::Reorder(a, b)),
        (0usize..8, 0usize..8, any::<bool>(), 0.3f32..1.0)
            .prop_map(|(a, b, p, o)| Edit::Group(a, b, p, o)),
        Just(Edit::Recomposite),
        (0usize..8, any::<u32>(), 1u8..6).prop_map(|(i, s, n)| Edit::Stroke(i, s, n)),
    ]
}

fn build(w: u32, h: u32, sixteen: bool, gens: &[LayerGen]) -> LayerStack {
    let mut s = if sixteen {
        let base = pixels(w, h, 7, 0);
        let wide: Vec<u16> = base.iter().map(|&b| (b as u16) << 8 | 0x5A).collect();
        LayerStack::from_image_px(w, h, Pixels::from_rgba16(&wide)).expect("stack")
    } else {
        LayerStack::from_image(w, h, pixels(w, h, 7, 0)).expect("stack")
    };
    for (i, g) in gens.iter().enumerate() {
        match g {
            LayerGen::Pixel { seed, alpha, smart } => {
                s.add(&format!("L{i}"));
                if !sixteen {
                    s.edit_active(
                        "fill",
                        Region::new(0, 0, w, h),
                        Pixels::from_rgba8(pixels(w, h, *seed, *alpha)),
                    )
                    .expect("fill");
                }
                if *smart {
                    let _ = s.make_smart(s.active_index());
                }
            }
            LayerGen::Adjust {
                exposure,
                saturation,
            } => {
                s.add_adjustment(
                    &format!("A{i}"),
                    AdjustParams {
                        exposure_ev: *exposure,
                        saturation: *saturation,
                        ..AdjustParams::default()
                    },
                );
            }
        }
    }
    s
}

fn apply(s: &mut LayerStack, e: &Edit, w: u32, h: u32) {
    let n = s.len();
    let i = |k: usize| k % n;
    // Refusals (locked, adjustment pixels, bad groups) are fine: the
    // stack is then simply unchanged, which is a case too.
    let _ = match e {
        Edit::Opacity(k, o) => s.set_opacity(i(*k), *o),
        Edit::Blend(k, b) => s.set_blend(i(*k), BLENDS[*b]),
        Edit::Visible(k, v) => s.set_visible(i(*k), *v),
        Edit::Paint(k, seed, a) => s.set_active(i(*k)).and_then(|()| {
            if s.active().rgba.is_16bit() {
                return Ok(());
            }
            s.edit_active(
                "paint",
                Region::new(0, 0, w, h),
                Pixels::from_rgba8(pixels(w, h, *seed, *a)),
            )
            .map(|_| ())
        }),
        Edit::Mask(k, kind) => s.set_mask(i(*k), mask(w, h, *kind)),
        Edit::MaskEnabled(k, en) => s.set_mask_enabled(i(*k), *en),
        Edit::Clip(k, c) => s.set_clipped(i(*k), *c),
        Edit::Active(k) => s.set_active(i(*k)),
        Edit::Reorder(a, b) => s.reorder(i(*a), i(*b)),
        Edit::Group(a, b, pass, o) => s.group_range(i(*a), i(*b), "G").and_then(|id| {
            s.set_group_pass_through(id, *pass)?;
            s.set_group_opacity(id, *o)
        }),
        Edit::Recomposite | Edit::Stroke(..) => Ok(()),
    };
}

/// Paint `samples` small rectangles into the active layer's pixels,
/// previewing each through the stack (the brush door's composite with
/// the in-flight pixels standing in for the active layer), then commit.
fn stroke(
    ctx: &GpuContext,
    s: &mut LayerStack,
    k: usize,
    seed: u32,
    samples: u8,
    w: u32,
    h: u32,
) -> Result<(), TestCaseError> {
    let k = k % s.len();
    if s.set_active(k).is_err() || !s.active().is_pixels() || s.active().rgba.is_16bit() {
        return Ok(());
    }
    let mut px: Vec<u8> = s.active().rgba.raw().to_vec();
    let mut painted: Arc<[u8]> = Arc::from(px.clone());
    for n in 0..samples as u32 {
        let r = seed.wrapping_add(n.wrapping_mul(7919));
        let (x0, y0) = (r % w, (r / 7) % h);
        let (rw, rh) = (1 + r % 5, 1 + (r / 3) % 4);
        for y in y0..(y0 + rh).min(h) {
            for x in x0..(x0 + rw).min(w) {
                let i = ((y * w + x) * 4) as usize;
                px[i..i + 4].copy_from_slice(&[
                    (r % 256) as u8,
                    (n * 40) as u8,
                    200,
                    (r % 3 * 120) as u8,
                ]);
            }
        }
        painted = Arc::from(px.clone());
        let got = pollster::block_on(s.composite(Some(ctx), Some(&painted)));
        let want = pollster::block_on(s.composite_reference(Some(ctx), Some(&painted)));
        match (got, want) {
            (Ok(g), Ok(w)) => prop_assert!(g[..] == w[..], "stroke sample {n} differs"),
            (Err(_), Err(_)) => {}
            (g, w) => prop_assert!(false, "stroke sample {n}: {g:?} vs {w:?}"),
        }
    }
    let _ = s.edit_active(
        "stroke",
        Region::new(0, 0, w, h),
        Pixels::from_rgba8(painted),
    );
    Ok(())
}

fn check(ctx: &GpuContext, s: &LayerStack, step: &str) -> Result<(), TestCaseError> {
    let got = pollster::block_on(s.composite(Some(ctx), None));
    let want = pollster::block_on(s.composite_reference(Some(ctx), None));
    match (got, want) {
        (Ok(g), Ok(w)) => prop_assert!(
            g[..] == w[..],
            "{step}: resident fold differs from the reference fold"
        ),
        (Err(_), Err(_)) => {}
        (g, w) => prop_assert!(false, "{step}: resident {g:?} vs reference {w:?}"),
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    #[test]
    fn the_resident_fold_equals_the_reference_fold_after_every_edit__feat__image_editor_layers(
        dims in prop_oneof![Just((20u32, 13u32)), Just((33, 17)), Just((16, 16))],
        sixteen in prop::bool::weighted(0.15),
        gens in prop::collection::vec(layer_gen(), 1..5),
        edits in prop::collection::vec(edit_gen(), 1..10),
    ) {
        let Some(ctx) = device() else { return Ok(()) };
        let (w, h) = dims;
        let mut s = build(w, h, sixteen, &gens);
        check(ctx, &s, "built")?;
        for (n, e) in edits.iter().enumerate() {
            apply(&mut s, e, w, h);
            if let Edit::Stroke(k, seed, samples) = e {
                stroke(ctx, &mut s, *k, *seed, *samples, w, h)?;
            }
            check(ctx, &s, &format!("edit {n} {e:?}"))?;
        }
    }
}

/// The one rewrite in the resident fold that is not a relocation: the
/// final unpremultiply is no longer skipped over an opaque canvas. That
/// is exact only if dividing an opaque texel by its alpha of one is the
/// identity on this device — asserted for EVERY finite f16 colour value
/// (up to the sign of zero, which the RGBA8 narrowing erases).
#[test]
fn unpremultiply_is_the_identity_on_opaque_texels__feat__image_editor_layers() {
    let Some(ctx) = device() else { return };
    let values: Vec<u16> = (0u16..=0xFFFF)
        .filter(|b| half::f16::from_bits(*b).is_finite())
        .collect();
    let texels = values.len().div_ceil(3);
    let w = 256u32;
    let h = (texels as u32).div_ceil(w);
    let mut bytes = Vec::with_capacity((w * h * 8) as usize);
    for t in 0..(w * h) as usize {
        for c in 0..3 {
            let v = values.get(t * 3 + c).copied().unwrap_or(0);
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&half::f16::ONE.to_bits().to_le_bytes());
    }
    let out = execute_tile_once(
        ctx,
        &CAST_UNPREMULTIPLY,
        &[TileInput { f16_bytes: &bytes }],
        CastUnpremultiplyParams::new().as_bytes(),
        None,
        w,
        h,
    )
    .expect("dispatch");
    let bad: Vec<(u16, u16)> = out
        .chunks_exact(2)
        .zip(bytes.chunks_exact(2))
        .map(|(a, b)| {
            (
                u16::from_le_bytes([b[0], b[1]]),
                u16::from_le_bytes([a[0], a[1]]),
            )
        })
        .filter(|(i, o)| i != o)
        .collect();
    // The ABI's `mix(a, result, 1)` turns −0 into +0 — the one value it
    // moves. The composite narrows to RGBA8, where both are 0, so the
    // canvas is unchanged; anything else moving would change it.
    assert!(
        bad.iter().all(|&(i, o)| i == 0x8000 && o == 0x0000),
        "unpremultiply changed {} opaque values: {:x?}",
        bad.len(),
        &bad[..bad.len().min(16)]
    );
}
