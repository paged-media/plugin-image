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

//! PROPERTIES of the colour conversions, over generated colours:
//!
//! * **premultiply ∘ unpremultiply round-trips** — scalar reference on
//!   every run, the two `cast.*` kernels on the GPU when a device exists;
//! * **the CMS identity transform is the identity** — sRGB → sRGB through
//!   BOTH backends: exact in moxcms (print lane); within one level per
//!   pass and bounded at two over repeated passes in qcms (display lane);
//! * **CIELAB / ΔE00 behave like a metric on what we measure with them**
//!   — ΔE(x, x) = 0, symmetry, black at L* 0, white at L* 100, and L*
//!   monotone along the grey axis;
//! * **the uncalibrated device CMYK fallback is sane** — no ink is white,
//!   full black ink is black, more ink never brightens.

use half::f16;
use image_cms::{CmsEngine, Intent};
use image_conformance::delta_e::{ciede2000, srgb8_to_lab};
use image_conformance::quantize::f16_ulp_distance;
use image_conformance::Px;
use image_gpu::{execute_tile_once, TileInput};
use image_kernels::families::cast::{
    cast_premultiply, cast_unpremultiply, CastPremultiplyParams, CastUnpremultiplyParams,
    CAST_PREMULTIPLY, CAST_UNPREMULTIPLY,
};
use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

fn runner(cases: u32) -> TestRunner {
    TestRunner::new_with_rng(
        Config {
            cases,
            failure_persistence: None,
            ..Config::default()
        },
        TestRng::from_seed(RngAlgorithm::ChaCha, &[0xc0; 32]),
    )
}

fn rgb8() -> impl Strategy<Value = [u8; 3]> {
    [any::<u8>(), any::<u8>(), any::<u8>()]
}

#[test]
#[allow(non_snake_case)]
fn premultiply_then_unpremultiply_round_trips__feat__image_kernel_family_t0() {
    let (pp, up) = (CastPremultiplyParams::new(), CastUnpremultiplyParams::new());
    let zero = Px([0.0; 4]);
    runner(1024)
        .run(
            &([0f32..=1.0, 0f32..=1.0, 0f32..=1.0], 1e-3f32..=1.0),
            |(c, a)| {
                let x = Px([c[0], c[1], c[2], a]);
                let rt = cast_unpremultiply(cast_premultiply(x, zero, &pp), zero, &up);
                for i in 0..4 {
                    prop_assert!(
                        (rt.0[i] - x.0[i]).abs() <= 1e-5 * (1.0 / a).max(1.0),
                        "channel {i}: {} -> {}",
                        x.0[i],
                        rt.0[i]
                    );
                }
                Ok(())
            },
        )
        .unwrap();
    // Zero alpha is a defined case, not a division: everything clears.
    let rt = cast_unpremultiply(
        cast_premultiply(Px([0.7, 0.2, 0.9, 0.0]), zero, &pp),
        zero,
        &up,
    );
    assert_eq!(rt.0, [0.0; 4]);
}

#[test]
#[allow(non_snake_case)]
fn gpu_premultiply_then_unpremultiply_round_trips__feat__image_kernel_family_t0() {
    let Some(ctx) = image_conformance::device::test_device() else {
        eprintln!("SKIP gpu colour round trip: no GPU adapter");
        return;
    };
    // k/256 colours and power-of-two alphas: exact in f16 on input, so
    // the only error is the kernels' own.
    let px = ([0u32..=256, 0u32..=256, 0u32..=256], 0u32..=3).prop_map(|(c, k)| {
        let a = [1.0, 0.5, 0.25, 0.125][k as usize];
        [
            c[0] as f32 / 256.0,
            c[1] as f32 / 256.0,
            c[2] as f32 / 256.0,
            a,
        ]
    });
    runner(8)
        .run(&proptest::collection::vec(px, 256), |tile| {
            let bytes: Vec<u8> = tile
                .iter()
                .flatten()
                .flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes())
                .collect();
            let run = |def, params: &[u8], input: &[u8]| {
                execute_tile_once(
                    ctx,
                    def,
                    &[TileInput { f16_bytes: input }],
                    params,
                    None,
                    16,
                    16,
                )
                .expect("cast dispatch")
            };
            let pre = run(
                &CAST_PREMULTIPLY,
                CastPremultiplyParams::new().as_bytes(),
                &bytes,
            );
            let back = run(
                &CAST_UNPREMULTIPLY,
                CastUnpremultiplyParams::new().as_bytes(),
                &pre,
            );
            let worst = bytes
                .chunks_exact(2)
                .zip(back.chunks_exact(2))
                .map(|(a, b)| {
                    f16_ulp_distance(
                        u16::from_le_bytes([a[0], a[1]]),
                        u16::from_le_bytes([b[0], b[1]]),
                    )
                })
                .max()
                .unwrap_or(0);
            // premultiply (1 ULP) + unpremultiply (2 ULP), amplified by
            // 1/α ≤ 8 on the way back.
            prop_assert!(worst <= 24, "round trip is {worst} f16 ULP off");
            Ok(())
        })
        .unwrap();
}

#[test]
#[allow(non_snake_case)]
fn srgb_identity_transform_is_the_identity_in_both_backends__feat__image_cms_engine() {
    let srgb = image_cms::working_srgb_profile().expect("sRGB profile");
    // MEASURED (all 2^24 / 7 colours, ten passes): moxcms is EXACT;
    // qcms moves 0.78 % of samples by one level on the first pass and
    // settles at two after the second, never further. Its 8-bit LUT
    // makes the identity non-idempotent but bounded — that is the
    // display lane's stated precision, asserted here so it cannot grow.
    let engines: [(&str, &dyn CmsEngine, u8, u8); 2] = [
        ("qcms", &image_cms::qcms_engine::QcmsEngine, 1, 2),
        ("moxcms", &image_cms::moxcms_engine::MoxcmsEngine, 0, 0),
    ];
    for (name, engine, once_tol, many_tol) in engines {
        let t = engine
            .compile(&srgb, &srgb, Intent::RelativeColorimetric, false)
            .unwrap_or_else(|e| panic!("{name}: sRGB → sRGB compiles: {e}"));
        runner(64)
            .run(&proptest::collection::vec(rgb8(), 64), |cols| {
                let src: Vec<u8> = cols.iter().flat_map(|c| [c[0], c[1], c[2], 200]).collect();
                let mut cur = src.clone();
                t.apply_rgba8(&mut cur);
                for (i, (a, b)) in src.iter().zip(&cur).enumerate() {
                    if i % 4 == 3 {
                        prop_assert_eq!(a, b, "{}: alpha must pass through", name);
                    } else {
                        prop_assert!(a.abs_diff(*b) <= once_tol, "{}: {} -> {}", name, a, b);
                    }
                }
                for _ in 0..9 {
                    t.apply_rgba8(&mut cur);
                }
                for (a, b) in src.iter().zip(&cur) {
                    prop_assert!(
                        a.abs_diff(*b) <= many_tol,
                        "{}: drifts {} levels over ten applications",
                        name,
                        a.abs_diff(*b)
                    );
                }
                Ok(())
            })
            .unwrap();
    }
}

#[test]
#[allow(non_snake_case)]
fn lab_and_delta_e_behave_on_srgb__feat__image_conformance_harness() {
    let black = srgb8_to_lab([0, 0, 0]);
    let white = srgb8_to_lab([255, 255, 255]);
    assert!(black[0].abs() < 1e-6, "black L* = {}", black[0]);
    assert!((white[0] - 100.0).abs() < 1e-3, "white L* = {}", white[0]);
    assert!(
        white[1].abs() < 0.01 && white[2].abs() < 0.01,
        "white is neutral: {white:?}"
    );
    let mut last = -1.0;
    for g in 0..=255u8 {
        let l = srgb8_to_lab([g, g, g]);
        assert!(l[0] > last, "L* not increasing at grey {g}");
        assert!(
            l[1].abs() < 0.01 && l[2].abs() < 0.01,
            "grey {g} is not neutral: {l:?}"
        );
        last = l[0];
    }
    runner(1024)
        .run(&(rgb8(), rgb8()), |(a, b)| {
            let (la, lb) = (srgb8_to_lab(a), srgb8_to_lab(b));
            prop_assert!(ciede2000(la, la).abs() < 1e-9);
            let (ab, ba) = (ciede2000(la, lb), ciede2000(lb, la));
            prop_assert!((ab - ba).abs() < 1e-9, "ΔE not symmetric: {} vs {}", ab, ba);
            prop_assert!(ab >= 0.0);
            prop_assert!(
                a == b || ab > 0.0,
                "distinct colours at ΔE 0: {:?} {:?}",
                a,
                b
            );
            Ok(())
        })
        .unwrap();
}

#[test]
#[allow(non_snake_case)]
fn device_cmyk_fallback_is_monotone_in_ink__feat__image_cms_print() {
    use image_js::cmyk::cmyk_device_to_rgba8;
    assert_eq!(
        cmyk_device_to_rgba8(&[0, 0, 0, 0]),
        vec![255, 255, 255, 255]
    );
    assert_eq!(cmyk_device_to_rgba8(&[0, 0, 0, 255])[..3], [0, 0, 0]);
    runner(1024)
        .run(
            &([any::<u8>(); 4], 0usize..4, any::<u8>()),
            |(ink, ch, more)| {
                let mut darker = ink;
                darker[ch] = ink[ch].max(more);
                let (a, b) = (cmyk_device_to_rgba8(&ink), cmyk_device_to_rgba8(&darker));
                for c in 0..3 {
                    prop_assert!(b[c] <= a[c], "more ink brightened channel {}", c);
                }
                prop_assert_eq!(a[3], 255);
                Ok(())
            },
        )
        .unwrap();
}
