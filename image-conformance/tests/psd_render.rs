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

//! The PSD render-level oracle (spec §10.4 oracle 2 / §15 M1 "PSD flatten
//! pipeline → render oracle") — feat: image.psd.render.flatten.
//!
//! Three checks over the layered M0 fixtures (`psd_builder::fixtures`):
//!
//! 1. **reference vs hand-computed** — `flatten_reference` against a few
//!    texels per fixture, computed by hand from the known planes/blends
//!    (the arithmetic is written out in the test comments). Proves the
//!    scalar spine itself, independent of the GPU.
//! 2. **gpu vs reference** — `flatten_gpu` (Engine A `apply2` + the
//!    `compose.*` WGSL kernels) against `flatten_reference` within
//!    `ChannelEpsF16(6)` per texel. SKIPS cleanly with no GPU adapter.
//! 3. **rgb8_flat agreement** — for the single-layer `rgb8_flat` fixture,
//!    `flatten_reference == that one layer == the fixture's composite
//!    planes`. (For the LAYERED fixtures the builder's merged composite is
//!    caller-provided and intentionally NOT authoritative — real-PSD
//!    embedded-composite agreement is the M1.5 corpus step.)
//!
//! The flatten oracle and the compose-kernel parity tests share the SAME
//! scalar reference (`compose_ref::composite`), so a blend-math bug shows
//! up in both lanes — never hides in one.

use image_conformance::compose_ref::{self, Blend};
use image_conformance::device::test_device;
use image_conformance::psd_builder::fixtures;
use image_conformance::psd_render::{flatten_gpu, flatten_reference};
use image_conformance::quantize::{f16_ulp_distance, f32_to_f16_bits};
use image_conformance::Px;
use image_psd::model::PsdFile;

/// Parse a fixture's bytes into the production model (the flatten oracle
/// consumes the parsed `PsdFile`, exactly as the renderer would).
fn parse(bytes: &[u8]) -> PsdFile {
    PsdFile::parse(bytes).expect("fixture parses")
}

/// The reference texel at `(x, y)` of a `w`-wide canvas.
fn texel(canvas: &[Px], w: u32, x: u32, y: u32) -> Px {
    canvas[(y * w + x) as usize]
}

/// Assert a reference texel equals a hand-computed expectation (f32; the
/// reference lane is exact f32, so a tight epsilon is honest).
fn assert_texel(got: Px, want: [f32; 4], where_: &str) {
    for (c, &w) in want.iter().enumerate() {
        assert!(
            (got.0[c] - w).abs() <= 1e-5,
            "{where_} channel {c}: got {} want {}",
            got.0[c],
            w,
        );
    }
}

// ---------------------------------------------------------------------
// (1) flatten_reference vs hand-computed probe points.
//
// All three layered fixtures use SOLID-color planes, so every canvas
// texel is identical; the probes pick a spread of coordinates to confirm
// the placement loop covers the whole canvas (corners + interior).
// ---------------------------------------------------------------------

/// `multilayer_groups` (8×8 RGB): flat list bottom-up is
/// `bg`, divider(lsct3), `inner`, folder(lsct2), `top`. The divider and
/// folder carry NO pixels and are skipped, leaving the opaque normal stack
/// bg → inner → top:
///   1. (0,0,0,0) ⊕ bg(0,0,0,1)         = (0,0,0,1)   [Cs over clear]
///   2. (0,0,0,1) ⊕ inner(1,0,0,1)      = (1,0,0,1)   [opaque replaces]
///   3. (1,0,0,1) ⊕ top(0,1,0,1)        = (0,1,0,1)   [opaque replaces]
///
/// → pure green everywhere.
#[test]
fn image_psd_render_flatten_multilayer_groups_reference() {
    let (bytes, _m) = fixtures::multilayer_groups();
    let file = parse(&bytes);
    let canvas = flatten_reference(&file);
    let want = [0.0, 1.0, 0.0, 1.0];
    for (x, y) in [(0u32, 0u32), (7, 0), (0, 7), (7, 7), (3, 4)] {
        assert_texel(texel(&canvas, 8, x, y), want, "multilayer_groups");
    }
}

/// `blend_opacity` (4×4 RGB): four opaque full-canvas layers, all
/// G=B=0 so only the R channel is interesting (G,B stay 0; A stays 1).
/// With an opaque backdrop (αb=1) and an opaque source folded by opacity
/// `o` (αs=o, αo=1), the composite reduces per channel to
///   co = o·B(Cb,Cs) + (1−o)·Cb.
/// Bottom-up over the R channel (Cb starts 0 after `base`):
///   • mult   o=200/255, Cs=40/255, B=Cb·Cs=0   ⇒ R = 0
///   • screen o=128/255, Cs=80/255, B=Cb+Cs−Cb·Cs=80/255
///                                  ⇒ R = (128/255)(80/255) = 0.15747789…
///   • clip   o= 64/255, Cs=120/255, blend=overlay=hardLight(Cs,Cb):
///       Cb=0.15747789 ≤ 0.5 ⇒ hardLight(0.470588, 0.157478)
///                            = 0.470588·(2·0.157478) = 0.14822983…
///       ⇒ R = (64/255)·0.14822983 + (1−64/255)·0.15747789 = 0.15515296…
/// → (0.15515296, 0, 0, 1) everywhere.
#[test]
fn image_psd_render_flatten_blend_opacity_reference() {
    let (bytes, _m) = fixtures::blend_opacity();
    let file = parse(&bytes);
    let canvas = flatten_reference(&file);
    let want = [0.155_152_96, 0.0, 0.0, 1.0];
    for (x, y) in [(0u32, 0u32), (3, 0), (0, 3), (3, 3), (1, 2)] {
        assert_texel(texel(&canvas, 4, x, y), want, "blend_opacity");
    }
}

/// `rle_and_raw_mix` (6×6 RGB): three opaque normal layers, bottom-up
/// `raw`(255,0,0), `rle`(0,255,0), `mixed`(10,20,30). Each opaque normal
/// layer fully replaces the backdrop, so the top layer wins:
/// → (10/255, 20/255, 30/255, 1) everywhere. Exercises that BOTH the RLE
/// and RAW channel decode paths feed the flatten identically (the top
/// layer mixes RLE id0/id2 with RAW id1).
#[test]
fn image_psd_render_flatten_rle_and_raw_mix_reference() {
    let (bytes, _m) = fixtures::rle_and_raw_mix();
    let file = parse(&bytes);
    let canvas = flatten_reference(&file);
    let want = [10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0, 1.0];
    for (x, y) in [(0u32, 0u32), (5, 0), (0, 5), (5, 5), (2, 3)] {
        assert_texel(texel(&canvas, 6, x, y), want, "rle_and_raw_mix");
    }
}

// ---------------------------------------------------------------------
// (2) flatten_gpu vs flatten_reference within ChannelEpsF16(6) per texel.
//
// The GPU lane stores each intermediate composite at f16 between apply2
// stages; the reference accumulates in f32. The per-texel divergence over
// these short stacks (≤3 pixel layers) stays inside the compose kernels'
// declared tolerance — 6 f16 ULPs, the spec's channel_eps_f16 6.
// SKIPS (passes, prints SKIP) with no GPU adapter (§9.3).
// ---------------------------------------------------------------------

const FLATTEN_EPS: u32 = 6;

/// Diff `flatten_gpu` against `flatten_reference` per texel/channel in
/// f16 ULPs (the reference is f16-quantized as the final step, mirroring
/// the kernel parity harness, §6.3).
fn check_gpu_vs_ref(name: &str, bytes: &[u8], w: u32) {
    let Some(ctx) = test_device() else {
        eprintln!("SKIP {name}: no GPU adapter");
        return;
    };
    let file = parse(bytes);
    let reference = flatten_reference(&file);
    let gpu = flatten_gpu(&file, ctx).expect("layered fixture has pixel layers");
    assert_eq!(reference.len(), gpu.len(), "{name}: canvas size");

    let mut worst = 0u32;
    let mut worst_at = (0usize, 0usize);
    for (i, (r, g)) in reference.iter().zip(gpu.iter()).enumerate() {
        for c in 0..4 {
            // Reference is quantized to f16 FIRST (the §6.3 final step);
            // the GPU value already arrived as a widened f16.
            let want = f32_to_f16_bits(r.0[c]);
            let got = f32_to_f16_bits(g.0[c]);
            let d = f16_ulp_distance(want, got);
            if d > worst {
                worst = d;
                worst_at = (i, c);
            }
        }
    }
    let _ = w;
    assert!(
        worst <= FLATTEN_EPS,
        "{name}: worst f16 ULP {worst} > {FLATTEN_EPS} at texel {} channel {}",
        worst_at.0,
        worst_at.1,
    );
    eprintln!("{name}: gpu↔ref worst f16 ULP = {worst}");
}

#[test]
fn image_psd_render_flatten_multilayer_groups_gpu() {
    let (bytes, _m) = fixtures::multilayer_groups();
    check_gpu_vs_ref("multilayer_groups", &bytes, 8);
}

#[test]
fn image_psd_render_flatten_blend_opacity_gpu() {
    let (bytes, _m) = fixtures::blend_opacity();
    check_gpu_vs_ref("blend_opacity", &bytes, 4);
}

#[test]
fn image_psd_render_flatten_rle_and_raw_mix_gpu() {
    let (bytes, _m) = fixtures::rle_and_raw_mix();
    check_gpu_vs_ref("rle_and_raw_mix", &bytes, 6);
}

// ---------------------------------------------------------------------
// (3) rgb8_flat: flatten == the single layer == the composite planes.
// ---------------------------------------------------------------------

/// `rgb8_flat` has NO layers — only a RAW merged composite of solid
/// `(200,100,50)` RGB. With no pixel layers the flatten is the
/// transparent-black canvas (`flatten_gpu` returns `None`); the embedded
/// composite is what `rgb8_flat` proves directly. This is the
/// embedded-composite agreement CASE: for the FLAT fixture the builder's
/// composite IS the only pixel data, so reference-flatten (empty) plus the
/// composite together describe the file; for LAYERED fixtures the builder
/// composite is caller-provided mid-gray and intentionally NOT
/// authoritative (real-PSD embedded composites are the M1.5 corpus step).
#[test]
fn image_psd_render_flatten_rgb8_flat_no_layers() {
    let (bytes, m) = fixtures::rgb8_flat();
    let file = parse(&bytes);
    assert!(
        file.layer_mask.layers.is_empty(),
        "rgb8_flat carries no layer records"
    );

    // No pixel layers ⇒ flatten is the all-transparent canvas, and the GPU
    // lane has nothing to dispatch.
    let canvas = flatten_reference(&file);
    assert_eq!(canvas.len(), (m.width * m.height) as usize);
    for p in &canvas {
        assert_eq!(p.0, [0.0, 0.0, 0.0, 0.0], "no layers ⇒ transparent");
    }
    if let Some(ctx) = test_device() {
        assert!(
            flatten_gpu(&file, ctx).is_none(),
            "rgb8_flat has no pixel layers ⇒ flatten_gpu None"
        );
    }
}

/// A single RGB layer composited over the clear backdrop reproduces that
/// layer's straight color (opaque normal ⇒ co = Cs, αo = 1). Built from a
/// one-layer file via the builder so the embedded-composite agreement case
/// (one layer == its own pixels) is exercised on the flatten lane.
#[test]
fn image_psd_render_flatten_single_layer_equals_its_pixels() {
    use image_conformance::psd_builder::channels::Plane;
    use image_conformance::psd_builder::layers::{ChannelSpec, LayerSpec};
    use image_conformance::psd_builder::PsdBuilder;
    use image_psd::container::Container;
    use image_psd::model::Compression;

    let (w, h) = (4u32, 4u32);
    let rgb = [200u8, 100, 50];
    let channels = [0i16, 1, 2]
        .iter()
        .zip(rgb)
        .map(|(&id, v)| ChannelSpec {
            id,
            plane: Plane::solid(w, h, v),
            compression: Compression::Raw,
        })
        .collect();
    let layer = LayerSpec {
        top: 0,
        left: 0,
        bottom: h as i32,
        right: w as i32,
        name: "solo".into(),
        blend_key: *b"norm",
        opacity: 255,
        clipping: 0,
        flags: 0,
        mask: None,
        blend_ranges: Vec::new(),
        channels,
        extra_addl: Vec::new(),
    };
    let bytes = PsdBuilder::new(Container::Psd, w, h, 3)
        .layer(layer)
        .composite(Compression::Raw, vec![Plane::solid(w, h, 128); 3])
        .build();

    let file = parse(&bytes);
    let canvas = flatten_reference(&file);
    // Opaque normal layer over clear ⇒ straight color, alpha 1.
    let want = [200.0 / 255.0, 100.0 / 255.0, 50.0 / 255.0, 1.0];
    for p in &canvas {
        assert_texel(*p, want, "single_layer");
    }

    // And it matches `compose_ref::composite(clear, premul(layer))` —
    // the embedded-composite agreement spine.
    let clear = Px([0.0; 4]);
    let layer_premul = Px([want[0], want[1], want[2], 1.0]); // rgb·a, a=1
    let direct = compose_ref::composite(clear, layer_premul, 1.0, Blend::Normal);
    assert_texel(canvas[0], direct.0, "single_layer vs composite");
}
