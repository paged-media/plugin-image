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

//! The lcms2-vs-qcms differential parity test (spec §10.1, D-11 §F.4).
//!
//! qcms is the shipped display backend; lcms2 is the **CI-only** oracle
//! that never ships (the plugin's no-C-in-default-build rule). This is
//! the inverse of core's `paged-color/tests/parity.rs`, where qcms is
//! the native dev-dep oracle for lcms2; here lcms2 is the dev-dep oracle
//! for qcms.
//!
//! Everything is hermetic: the ICC profiles are synthesised in-test via
//! lcms2 and serialised to bytes — no network, no system / corpus
//! profiles. Both engines consume the *same* bytes, so the comparison
//! measures the two CMMs over identical inputs, not profile skew.
//!
//! The oracle runs lcms2 with `Flags::default()` — Perceptual intent,
//! **no black-point compensation** — because qcms exposes no BPC (audit
//! §A): comparing against an lcms2 transform *with* BPC would measure a
//! capability gap the display lane deliberately doesn't have, not a CMM
//! divergence. The print lane (moxcms, M1) owns BPC fidelity.

use image_cms::qcms_engine::QcmsEngine;
use image_cms::{CmsEngine, Intent, Profile};
use image_core::{ContentHash, IccHash};

use lcms2::{
    CIExyY, CIExyYTRIPLE, Flags, PixelFormat, Profile as LcmsProfile, ToneCurve, Transform,
};

/// Per-channel acceptance bound for the differential oracle, in 8-bit
/// codes. Both CMMs share the same 8-bit endpoints and the same
/// (Perceptual, no-BPC) configuration; the only residual is matrix /
/// shaper rounding inside each engine's RGB→RGB pipeline. 3/255 ≈ 1.2%
/// is comfortably below a just-noticeable difference yet tight enough
/// that a real divergence (e.g. a swapped channel or a missed gamma)
/// trips it immediately. The test prints the measured max so a tightening
/// is data-backed, never guessed.
const MAX_CHANNEL_DEVIATION: u8 = 3;

/// Build an interned-style [`Profile`] from raw ICC bytes, hashing them
/// the way [`image_cms::ProfileInterner`] does so the handle's identity
/// matches the production path.
fn profile_from_bytes(bytes: Vec<u8>) -> Profile {
    let bytes: std::sync::Arc<[u8]> = bytes.into();
    Profile {
        hash: IccHash(ContentHash::of(&bytes).0),
        bytes,
    }
}

/// A canonical sRGB profile, serialised to ICC bytes.
fn srgb_icc() -> Vec<u8> {
    LcmsProfile::new_srgb()
        .icc()
        .expect("serialise sRGB profile")
}

/// A hermetic "swapped-primaries" RGB profile: D65 white, a 2.2 gamma on
/// all three channels, but the red and blue primaries exchanged. This is
/// a deliberately non-identity source so the oracle exercises a real LUT
/// pipeline (not just an sRGB→sRGB near-identity), giving the per-channel
/// bound something to actually bound.
fn swapped_primaries_icc() -> Vec<u8> {
    // D65 white point and the sRGB/Rec.709 primaries, with R and B
    // exchanged. xyY chromaticities are public colour-science constants.
    let d65 = CIExyY {
        x: 0.3127,
        y: 0.3290,
        Y: 1.0,
    };
    let primaries = CIExyYTRIPLE {
        // Red slot carries the blue primary, Blue slot carries the red.
        Red: CIExyY {
            x: 0.15,
            y: 0.06,
            Y: 1.0,
        },
        Green: CIExyY {
            x: 0.30,
            y: 0.60,
            Y: 1.0,
        },
        Blue: CIExyY {
            x: 0.64,
            y: 0.33,
            Y: 1.0,
        },
    };
    let gamma = ToneCurve::new(2.2);
    let curves = [&gamma, &gamma, &gamma];
    LcmsProfile::new_rgb(&d65, &primaries, &curves)
        .expect("build swapped-primaries profile")
        .icc()
        .expect("serialise swapped-primaries profile")
}

/// A patch ramp covering the neutral axis, the primaries / secondaries,
/// and a few mixes — interleaved RGBA8 with varying alpha so the
/// alpha-passthrough contract is exercised.
fn patch_ramp() -> Vec<u8> {
    let rgb: &[[u8; 3]] = &[
        [0, 0, 0],
        [255, 255, 255],
        [128, 128, 128],
        [64, 64, 64],
        [192, 192, 192],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [200, 100, 50],
        [33, 167, 211],
        [17, 17, 17],
        [240, 230, 220],
        [10, 20, 30],
    ];
    let mut out = Vec::with_capacity(rgb.len() * 4);
    for (i, px) in rgb.iter().enumerate() {
        out.extend_from_slice(px);
        // Spread alpha across the ramp; include the 0 and 255 extremes.
        out.push(((i * 255) / (rgb.len() - 1)) as u8);
    }
    out
}

/// (2) sRGB→sRGB compiles and is a near-identity on the ramp, with alpha
/// passed through verbatim.
#[test]
fn image_cms_core_consistency_identity_srgb_near_identity() {
    let srgb = profile_from_bytes(srgb_icc());
    let t = QcmsEngine
        .compile(&srgb, &srgb, Intent::Perceptual, false)
        .expect("sRGB→sRGB must compile");

    let original = patch_ramp();
    let mut pixels = original.clone();
    t.apply_rgba8(&mut pixels);

    let mut max_rgb_dev: u8 = 0;
    for (i, (out, inp)) in pixels
        .chunks_exact(4)
        .zip(original.chunks_exact(4))
        .enumerate()
    {
        for c in 0..3 {
            let dev = out[c].abs_diff(inp[c]);
            max_rgb_dev = max_rgb_dev.max(dev);
            assert!(
                dev <= MAX_CHANNEL_DEVIATION,
                "sRGB→sRGB not near-identity at patch {i} channel {c}: {} vs {} (dev {dev})",
                out[c],
                inp[c]
            );
        }
        // Alpha is contractually untouched.
        assert_eq!(
            out[3], inp[3],
            "alpha mutated at patch {i}: {} → {}",
            inp[3], out[3]
        );
    }
    eprintln!("identity sRGB→sRGB: max per-channel deviation = {max_rgb_dev}/255");
}

/// (3) The differential oracle: the same patches through lcms2 (8-bit
/// RGB_8, Perceptual, no BPC) and through the qcms `CompiledTransform`
/// must agree per channel within [`MAX_CHANNEL_DEVIATION`]. Run for both
/// the identity (sRGB→sRGB) and a real LUT pipeline (swapped→sRGB).
#[test]
fn image_cms_core_consistency_lcms2_qcms_per_channel() {
    let srgb_bytes = srgb_icc();
    let swapped_bytes = swapped_primaries_icc();

    let cases: &[(&str, &[u8], &[u8])] = &[
        ("srgb->srgb", &srgb_bytes, &srgb_bytes),
        ("swapped->srgb", &swapped_bytes, &srgb_bytes),
    ];

    let mut worst: (u8, &str, usize, usize) = (0, "", 0, 0);
    for &(name, src_bytes, dst_bytes) in cases {
        // qcms path (the shipped backend) — RGBA8 in place.
        let src = profile_from_bytes(src_bytes.to_vec());
        let dst = profile_from_bytes(dst_bytes.to_vec());
        let qt = QcmsEngine
            .compile(&src, &dst, Intent::Perceptual, false)
            .unwrap_or_else(|e| panic!("qcms compile {name}: {e}"));
        let mut qcms_out = patch_ramp();
        qt.apply_rgba8(&mut qcms_out);

        // lcms2 oracle — RGB_8 → RGB_8, Perceptual, Flags::default()
        // (NO black-point compensation: qcms has none, audit §A).
        let lsrc = LcmsProfile::new_icc(src_bytes).expect("lcms2 src profile");
        let ldst = LcmsProfile::new_icc(dst_bytes).expect("lcms2 dst profile");
        let lt: Transform<[u8; 3], [u8; 3]> = Transform::new_flags(
            &lsrc,
            PixelFormat::RGB_8,
            &ldst,
            PixelFormat::RGB_8,
            lcms2::Intent::Perceptual,
            Flags::default(),
        )
        .expect("lcms2 transform");

        let ramp = patch_ramp();
        let rgb_in: Vec<[u8; 3]> = ramp.chunks_exact(4).map(|p| [p[0], p[1], p[2]]).collect();
        let mut rgb_out = vec![[0u8; 3]; rgb_in.len()];
        lt.transform_pixels(&rgb_in, &mut rgb_out);

        for (i, (q, l)) in qcms_out.chunks_exact(4).zip(rgb_out.iter()).enumerate() {
            for c in 0..3 {
                let dev = q[c].abs_diff(l[c]);
                if dev > worst.0 {
                    worst = (dev, name, i, c);
                }
                assert!(
                    dev <= MAX_CHANNEL_DEVIATION,
                    "{name} patch {i} channel {c}: qcms {} vs lcms2 {} (dev {dev} > {MAX_CHANNEL_DEVIATION})",
                    q[c],
                    l[c]
                );
            }
        }
    }
    eprintln!(
        "lcms2↔qcms worst per-channel deviation = {}/255 ({} patch {} channel {})",
        worst.0, worst.1, worst.2, worst.3
    );
}

/// (4) `bake_lut(17)` on the near-identity sRGB→sRGB transform: the
/// lattice corners are exact lattice points (no trilinear slack there),
/// so each corner must map to ≈ itself within the engine's own 8-bit
/// rounding. The corner value at lattice index k on a `dim`-point axis
/// is `k*255/(dim-1)` — for dim=17 every corner lands on an integer code.
#[test]
fn image_cms_core_consistency_bake_lut_identity_corners() {
    const DIM: u32 = 17;
    let srgb = profile_from_bytes(srgb_icc());
    let t = QcmsEngine
        .compile(&srgb, &srgb, Intent::Perceptual, false)
        .expect("sRGB→sRGB must compile");
    let lut = t.bake_lut(DIM);

    let n = DIM as usize;
    assert_eq!(lut.dim, DIM);
    assert_eq!(lut.lattice.len(), n * n * n * 4);

    let coord = |k: usize| ((k * 255) / (n - 1)) as u8;
    let texel = |r: usize, g: usize, b: usize| {
        // x-major, r fastest (matches lut::bake_from_exact).
        let idx = ((b * n + g) * n + r) * 4;
        &lut.lattice[idx..idx + 4]
    };

    let mut max_dev: u8 = 0;
    // Sweep the 8 cube corners — the exact lattice points.
    for &b in &[0, n - 1] {
        for &g in &[0, n - 1] {
            for &r in &[0, n - 1] {
                let px = texel(r, g, b);
                for (c, &expected_k) in [r, g, b].iter().enumerate() {
                    let dev = px[c].abs_diff(coord(expected_k));
                    max_dev = max_dev.max(dev);
                    assert!(
                        dev <= MAX_CHANNEL_DEVIATION,
                        "corner ({r},{g},{b}) channel {c}: {} vs {} (dev {dev})",
                        px[c],
                        coord(expected_k)
                    );
                }
                // Baker writes opaque alpha into the lattice.
                assert_eq!(px[3], 255, "corner ({r},{g},{b}) alpha should be 255");
            }
        }
    }
    eprintln!("bake_lut({DIM}) identity corners: max per-channel deviation = {max_dev}/255");
}
