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

//! PROPERTIES of the 26 `compose.*` blend kernels — laws that hold for
//! every input, checked over generated inputs rather than a hand-picked
//! tile:
//!
//! * **opacity 0 is the identity** — the layer contributes nothing, for
//!   every blend mode;
//! * **a fully transparent source is the identity** at any opacity;
//! * **normal at opacity 1 over an opaque source IS the source**;
//! * **the output alpha is source-over** (`αs + αb·(1 − αs)`) whatever
//!   the blend, because the blend only decides colour;
//! * **commutativity where the formula is symmetric** (multiply, screen,
//!   darken, lighten, difference, exclusion, linear dodge) with both
//!   layers opaque, and the one documented ASYMMETRY that pairs two
//!   modes: `overlay(a, b) = hard_light(b, a)`.
//!
//! The scalar reference (`compose_ref::composite`, the parity golden) is
//! checked on every run; the GPU kernels are checked against the same
//! laws when a device exists (`REQUIRE_GPU=1` turns a missing one into a
//! failure). Seeded and modest in case count, for CI time.

use half::f16;
use image_conformance::compose_ref::{composite, Blend};
use image_conformance::quantize::{f16_ulp_distance, f32_to_f16_bits};
use image_conformance::Px;
use image_gpu::{execute_tile_once, TileInput};
use image_kernels::families::compose::{ComposeParams, FAMILY};
use image_kernels::{KernelDef, Tolerance};
use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

const ALL: [Blend; 26] = [
    Blend::Normal,
    Blend::Multiply,
    Blend::Screen,
    Blend::Overlay,
    Blend::Darken,
    Blend::Lighten,
    Blend::ColorDodge,
    Blend::ColorBurn,
    Blend::HardLight,
    Blend::SoftLight,
    Blend::Difference,
    Blend::Exclusion,
    Blend::Hue,
    Blend::Saturation,
    Blend::Color,
    Blend::Luminosity,
    Blend::LinearBurn,
    Blend::LinearDodge,
    Blend::DarkerColor,
    Blend::LighterColor,
    Blend::VividLight,
    Blend::LinearLight,
    Blend::PinLight,
    Blend::HardMix,
    Blend::Subtract,
    Blend::Divide,
];

/// Blends whose `B(Cb, Cs)` is symmetric in its two arguments.
const SYMMETRIC: [Blend; 7] = [
    Blend::Multiply,
    Blend::Screen,
    Blend::Darken,
    Blend::Lighten,
    Blend::Difference,
    Blend::Exclusion,
    Blend::LinearDodge,
];

fn runner(cases: u32) -> TestRunner {
    TestRunner::new_with_rng(
        Config {
            cases,
            failure_persistence: None,
            ..Config::default()
        },
        TestRng::from_seed(RngAlgorithm::ChaCha, &[0x5e; 32]),
    )
}

fn def_of(b: Blend) -> &'static KernelDef {
    FAMILY
        .iter()
        .copied()
        .find(|d| d.id == b.kernel_id())
        .expect("every Blend has a compose kernel")
}

/// A channel value on the k/256 lattice — exact in f16, so the GPU and
/// the reference see the same stimulus.
fn chan() -> impl Strategy<Value = f32> {
    (0u32..=256).prop_map(|k| k as f32 / 256.0)
}

/// An alpha on the {0, ¼, ½, ¾, 1} lattice: premultiplying a k/256
/// colour by it stays exact in f16.
fn alpha() -> impl Strategy<Value = f32> {
    (0u32..=4).prop_map(|k| k as f32 / 4.0)
}

/// A premultiplied pixel.
fn px() -> impl Strategy<Value = Px> {
    ([chan(), chan(), chan()], alpha()).prop_map(|(c, a)| Px([c[0] * a, c[1] * a, c[2] * a, a]))
}

/// An opaque pixel.
fn opaque() -> impl Strategy<Value = Px> {
    [chan(), chan(), chan()].prop_map(|c| Px([c[0], c[1], c[2], 1.0]))
}

fn blend() -> impl Strategy<Value = Blend> {
    (0usize..ALL.len()).prop_map(|i| ALL[i])
}

fn close(a: Px, b: Px, eps: f32) -> bool {
    a.0.iter()
        .zip(b.0.iter())
        .all(|(x, y)| (x - y).abs() <= eps)
}

const EPS: f32 = 1e-5;

#[test]
#[allow(non_snake_case)]
fn every_blend_at_zero_opacity_is_the_identity__feat__image_kernel_family_t1() {
    runner(512)
        .run(&(blend(), px(), px()), |(b, a, s)| {
            let out = composite(a, s, 0.0, b);
            prop_assert!(close(out, a, EPS), "{b:?}: {out:?} != backdrop {a:?}");
            Ok(())
        })
        .unwrap();
}

#[test]
#[allow(non_snake_case)]
fn a_transparent_source_is_the_identity_at_any_opacity__feat__image_kernel_family_t1() {
    runner(512)
        .run(&(blend(), px(), chan()), |(b, a, o)| {
            let out = composite(a, Px([0.0; 4]), o, b);
            prop_assert!(close(out, a, EPS), "{b:?} @ {o}: {out:?} != {a:?}");
            Ok(())
        })
        .unwrap();
}

#[test]
#[allow(non_snake_case)]
fn normal_at_full_opacity_over_an_opaque_source_is_the_source__feat__image_kernel_family_t1() {
    runner(512)
        .run(&(px(), opaque()), |(a, s)| {
            let out = composite(a, s, 1.0, Blend::Normal);
            prop_assert!(close(out, s, EPS), "{out:?} != source {s:?}");
            Ok(())
        })
        .unwrap();
}

#[test]
#[allow(non_snake_case)]
fn output_alpha_is_source_over_for_every_blend__feat__image_kernel_family_t1() {
    runner(512)
        .run(&(blend(), px(), px(), chan()), |(b, a, s, o)| {
            let out = composite(a, s, o, b);
            let (asrc, ab) = (s.0[3] * o, a.0[3]);
            let want = asrc + ab * (1.0 - asrc);
            prop_assert!(
                (out.0[3] - want).abs() <= EPS,
                "{b:?}: α {} != {want}",
                out.0[3]
            );
            for c in 0..3 {
                prop_assert!(
                    out.0[c] >= -EPS && out.0[c] <= out.0[3] + 1e-4,
                    "{b:?}: premultiplied channel {c} = {} outside [0, α = {}]",
                    out.0[c],
                    out.0[3]
                );
            }
            Ok(())
        })
        .unwrap();
}

#[test]
#[allow(non_snake_case)]
fn symmetric_blends_commute_over_opaque_layers__feat__image_kernel_family_t1() {
    runner(512)
        .run(
            &((0usize..SYMMETRIC.len()), opaque(), opaque()),
            |(i, a, s)| {
                let b = SYMMETRIC[i];
                let ab = composite(a, s, 1.0, b);
                let ba = composite(s, a, 1.0, b);
                prop_assert!(close(ab, ba, 1e-5), "{b:?}: {ab:?} != {ba:?}");
                Ok(())
            },
        )
        .unwrap();
}

#[test]
#[allow(non_snake_case)]
fn overlay_is_hard_light_with_the_layers_swapped__feat__image_kernel_family_t1() {
    runner(512)
        .run(&(opaque(), opaque()), |(a, s)| {
            let o = composite(a, s, 1.0, Blend::Overlay);
            let h = composite(s, a, 1.0, Blend::HardLight);
            prop_assert!(
                close(o, h, 1e-5),
                "overlay {o:?} != swapped hard light {h:?}"
            );
            Ok(())
        })
        .unwrap();
}

// ─────────────────────────────── GPU ─────────────────────────────────

const N: usize = 16 * 16;

fn f16_bytes(px: &[Px]) -> Vec<u8> {
    px.iter()
        .flat_map(|p| p.0)
        .flat_map(|c| f16::from_f32(c).to_bits().to_le_bytes())
        .collect()
}

fn gpu(def: &'static KernelDef, a: &[Px], b: &[Px], opacity: f32) -> Option<Vec<[u16; 4]>> {
    let ctx = image_conformance::device::test_device()?;
    let (ab, bb) = (f16_bytes(a), f16_bytes(b));
    let out = execute_tile_once(
        ctx,
        def,
        &[TileInput { f16_bytes: &ab }, TileInput { f16_bytes: &bb }],
        ComposeParams::new(opacity).as_bytes(),
        None,
        16,
        16,
    )
    .expect("compose dispatch");
    Some(
        out.chunks_exact(8)
            .map(|t| std::array::from_fn(|c| u16::from_le_bytes([t[c * 2], t[c * 2 + 1]])))
            .collect(),
    )
}

fn ulps(def: &KernelDef) -> u32 {
    match def.gpu_tolerance {
        Tolerance::Exact => 0,
        Tolerance::ChannelEpsF16(n) => n,
        Tolerance::PerceptualDeltaE(_) => 8,
    }
}

/// The worst f16-ULP distance between the GPU texels and `want`.
fn worst(got: &[[u16; 4]], want: &[Px]) -> u32 {
    got.iter()
        .zip(want)
        .flat_map(|(g, w)| (0..4).map(move |c| f16_ulp_distance(g[c], f32_to_f16_bits(w.0[c]))))
        .max()
        .unwrap_or(0)
}

fn tile(s: impl Strategy<Value = Px>) -> impl Strategy<Value = Vec<Px>> {
    proptest::collection::vec(s, N)
}

#[test]
#[allow(non_snake_case)]
fn gpu_blends_obey_identity_source_and_symmetry_laws__feat__image_kernel_family_t1() {
    if image_conformance::device::test_device().is_none() {
        eprintln!("SKIP gpu blend properties: no GPU adapter");
        return;
    }
    runner(6)
        .run(
            &(tile(px()), tile(px()), tile(opaque()), tile(opaque())),
            |(a, s, oa, os)| {
                for &b in &ALL {
                    let def = def_of(b);
                    let tol = ulps(def);
                    // opacity 0 → the backdrop.
                    let got = gpu(def, &a, &s, 0.0).expect("device");
                    let w = worst(&got, &a);
                    prop_assert!(
                        w <= tol,
                        "{}: opacity 0 is {w} ULP from the backdrop",
                        def.id
                    );
                    // transparent source → the backdrop.
                    let clear = vec![Px([0.0; 4]); N];
                    let got = gpu(def, &a, &clear, 0.5).expect("device");
                    let w = worst(&got, &a);
                    prop_assert!(w <= tol, "{}: clear source is {w} ULP off", def.id);
                }
                // normal at 1 over opaque → the source.
                let def = def_of(Blend::Normal);
                let got = gpu(def, &a, &os, 1.0).expect("device");
                let w = worst(&got, &os);
                prop_assert!(w <= ulps(def), "normal: {w} ULP from the source");
                // symmetry.
                for &b in &SYMMETRIC {
                    let def = def_of(b);
                    let ab = gpu(def, &oa, &os, 1.0).expect("device");
                    let ba = gpu(def, &os, &oa, 1.0).expect("device");
                    let w = ab
                        .iter()
                        .zip(&ba)
                        .flat_map(|(x, y)| (0..4).map(move |c| f16_ulp_distance(x[c], y[c])))
                        .max()
                        .unwrap_or(0);
                    prop_assert!(w <= ulps(def), "{}: not commutative ({w} ULP)", def.id);
                }
                Ok(())
            },
        )
        .unwrap();
}
