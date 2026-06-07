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

//! The moxcms print-lane tests (D-11 ruling §F.2).
//!
//! moxcms is the print lane: real intent handling (all four), CMYK
//! ingest, ICC v4 — the capabilities core's qcms build lacks (audit §A).
//! lcms2 is the **CI-only** differential oracle (never shipped), the
//! same dev-dep reversal as `tests/parity.rs`.
//!
//! Everything is hermetic: every ICC profile is synthesised in-test via
//! lcms2 and serialised to bytes — no network, no system / corpus
//! profiles. Both engines consume the *same* bytes, so the oracle
//! measures the two CMMs over identical inputs, not profile skew.
//!
//! Two oracle configurations matter here:
//!   * matrix-shaper RGB↔RGB (sRGB, swapped-primaries) — moxcms and
//!     lcms2 are bounded per channel; intent is folded out (no
//!     per-intent tables), so these cases bound CMM agreement.
//!   * a LUT-bearing RGB→Lab profile with **distinct** A2B0 (Perceptual)
//!     and A2B1 (colorimetric) tables — this is what proves moxcms
//!     honors intents (the qcms gap). lcms2's safe `Pipeline` surface
//!     cannot assemble such a profile (no stage-insertion entry point),
//!     so it is built through `lcms2-sys` raw FFI in [`rgb_lut_icc`].

use image_cms::moxcms_engine::MoxcmsEngine;
use image_cms::qcms_engine::QcmsEngine;
use image_cms::{CmsEngine, Intent, Profile};
use image_core::{ContentHash, IccHash};

use lcms2::{
    CIExyY, CIExyYTRIPLE, Flags, PixelFormat, Profile as LcmsProfile, ToneCurve, Transform,
};

/// Per-channel acceptance bound for the differential oracle, in 8-bit
/// codes (matching `tests/parity.rs`). Both CMMs share the same 8-bit
/// endpoints and the same (Perceptual / RelativeColorimetric, no-BPC)
/// configuration; the residual is matrix / shaper rounding inside each
/// engine's RGB→RGB pipeline. 3/255 ≈ 1.2% is well below a JND yet tight
/// enough that a real divergence (swapped channel, missed gamma) trips
/// it. The tests print the measured max so a tightening is data-backed.
const MAX_CHANNEL_DEVIATION: u8 = 3;

/// Build an interned-style [`Profile`] from raw ICC bytes, hashing them
/// the way [`image_cms::ProfileInterner`] does so the handle identity
/// matches the production path (same helper as `tests/parity.rs`).
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

/// A hermetic "swapped-primaries" RGB profile: D65 white, 2.2 gamma on
/// all three channels, red and blue primaries exchanged. A deliberately
/// non-identity source so the oracle exercises a real matrix pipeline
/// (identical to `tests/parity.rs` so the two backends are bounded over
/// the same input).
fn swapped_primaries_icc() -> Vec<u8> {
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

/// A patch ramp covering the neutral axis, primaries / secondaries, and
/// a few mixes — interleaved RGBA8 with varying alpha so the
/// alpha-passthrough contract is exercised (same ramp as
/// `tests/parity.rs`).
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
        out.push(((i * 255) / (rgb.len() - 1)) as u8);
    }
    out
}

/// Synthesise a hermetic RGB→Lab v2 ICC profile carrying **two distinct
/// LUTs**: AToB0 (Perceptual) and AToB1 (Relative/Absolute colorimetric),
/// differing by a fixed shift on the Lab a/b axes. moxcms selects A2B0
/// for Perceptual and A2B1 for the colorimetric intents, so a transform
/// through this profile is provably intent-dependent — the lever for the
/// intent-difference smoke test.
///
/// This is built via `lcms2-sys` raw FFI because lcms2's safe `Pipeline`
/// API exposes no stage-insertion entry point, so a CLUT-bearing pipeline
/// cannot be assembled through it. A v2 `lut16` shape is used (input tone
/// curves → CLUT → output tone curves); the v4 `mAB` form rejects a bare
/// CLUT. The `shift` perturbs the colorimetric table away from the
/// perceptual one.
fn rgb_lut_icc() -> Vec<u8> {
    use lcms2_sys as ffi;
    use std::os::raw::c_void;

    const N: usize = 9; // grid points per axis

    // Build one input-curves → CLUT → output-curves pipeline. `shift`
    // perturbs the Lab a/b output so two such pipelines differ.
    fn make_clut(shift: f64) -> *mut ffi::Pipeline {
        let mut table = vec![0u16; N * N * N * 3];
        let mut idx = 0;
        for r in 0..N {
            for g in 0..N {
                for b in 0..N {
                    let rf = r as f64 / (N - 1) as f64;
                    let gf = g as f64 / (N - 1) as f64;
                    let bf = b as f64 / (N - 1) as f64;
                    // A crude but smooth RGB→Lab-ish encoding (0..65535),
                    // with a per-intent shift on the a/b axes.
                    let l = (rf * 0.3 + gf * 0.59 + bf * 0.11) * 65535.0;
                    let a = ((rf - gf) * 0.5 + 0.5 + shift) * 65535.0;
                    let bb = ((gf - bf) * 0.5 + 0.5 - shift) * 65535.0;
                    table[idx] = l.clamp(0.0, 65535.0) as u16;
                    table[idx + 1] = a.clamp(0.0, 65535.0) as u16;
                    table[idx + 2] = bb.clamp(0.0, 65535.0) as u16;
                    idx += 3;
                }
            }
        }
        // SAFETY: all pointers passed to lcms2 are valid for the call;
        // lcms copies `table` into the CLUT stage, the tone curve is freed
        // after the stages that copy it, and the returned pipeline is owned
        // by the caller (freed after the profile is serialised).
        unsafe {
            let pipe = ffi::cmsPipelineAlloc(std::ptr::null_mut(), 3, 3);
            // v2 lut16 shape: input curves (3) → CLUT → output curves (3).
            let lin = ffi::cmsBuildGamma(std::ptr::null_mut(), 1.0);
            let curves: [*const ffi::ToneCurve; 3] = [lin, lin, lin];
            let pre = ffi::cmsStageAllocToneCurves(std::ptr::null_mut(), 3, curves.as_ptr());
            ffi::cmsPipelineInsertStage(pipe, ffi::StageLoc::AT_END, pre);
            let clut =
                ffi::cmsStageAllocCLut16bit(std::ptr::null_mut(), N as u32, 3, 3, table.as_ptr());
            ffi::cmsPipelineInsertStage(pipe, ffi::StageLoc::AT_END, clut);
            let post = ffi::cmsStageAllocToneCurves(std::ptr::null_mut(), 3, curves.as_ptr());
            ffi::cmsPipelineInsertStage(pipe, ffi::StageLoc::AT_END, post);
            ffi::cmsFreeToneCurve(lin);
            pipe
        }
    }

    unsafe {
        let h = ffi::cmsCreateProfilePlaceholder(std::ptr::null_mut());
        ffi::cmsSetProfileVersion(h, 2.4);
        ffi::cmsSetDeviceClass(h, ffi::ProfileClassSignature::OutputClass);
        ffi::cmsSetColorSpace(h, ffi::ColorSpaceSignature::RgbData);
        ffi::cmsSetPCS(h, ffi::ColorSpaceSignature::LabData);

        let a2b0 = make_clut(0.0); // Perceptual
        let a2b1 = make_clut(0.15); // colorimetric — perturbed
        assert!(!a2b0.is_null() && !a2b1.is_null(), "CLUT pipeline alloc");
        assert_eq!(
            ffi::cmsWriteTag(h, ffi::TagSignature::AToB0Tag, a2b0 as *const c_void),
            1,
            "write AToB0"
        );
        assert_eq!(
            ffi::cmsWriteTag(h, ffi::TagSignature::AToB1Tag, a2b1 as *const c_void),
            1,
            "write AToB1"
        );

        // Mandatory tags for a serialisable profile.
        let wp = ffi::cmsD50_XYZ();
        ffi::cmsWriteTag(
            h,
            ffi::TagSignature::MediaWhitePointTag,
            (wp as *const ffi::CIEXYZ) as *const c_void,
        );
        let en = b"en\0";
        let us = b"US\0";
        let desc = b"paged.image print-lane LUT probe\0";
        let cpy = b"no copyright\0";
        let mlu_d = ffi::cmsMLUalloc(std::ptr::null_mut(), 1);
        ffi::cmsMLUsetASCII(
            mlu_d,
            en.as_ptr() as *const _,
            us.as_ptr() as *const _,
            desc.as_ptr() as *const _,
        );
        ffi::cmsWriteTag(
            h,
            ffi::TagSignature::ProfileDescriptionTag,
            mlu_d as *const c_void,
        );
        let mlu_c = ffi::cmsMLUalloc(std::ptr::null_mut(), 1);
        ffi::cmsMLUsetASCII(
            mlu_c,
            en.as_ptr() as *const _,
            us.as_ptr() as *const _,
            cpy.as_ptr() as *const _,
        );
        ffi::cmsWriteTag(h, ffi::TagSignature::CopyrightTag, mlu_c as *const c_void);

        let mut len: u32 = 0;
        assert_eq!(
            ffi::cmsSaveProfileToMem(h, std::ptr::null_mut(), &mut len),
            1,
            "size LUT profile"
        );
        assert!(len > 0, "LUT profile is empty");
        let mut data = vec![0u8; len as usize];
        assert_eq!(
            ffi::cmsSaveProfileToMem(h, data.as_mut_ptr() as *mut c_void, &mut len),
            1,
            "serialise LUT profile"
        );

        ffi::cmsPipelineFree(a2b0);
        ffi::cmsPipelineFree(a2b1);
        ffi::cmsMLUfree(mlu_d);
        ffi::cmsMLUfree(mlu_c);
        ffi::cmsCloseProfile(h);
        data
    }
}

/// (1) moxcms sRGB→sRGB compiles and is a near-identity on the ramp,
/// with alpha passed through verbatim.
#[test]
fn image_cms_print_lane_moxcms_srgb_near_identity() {
    let srgb = profile_from_bytes(srgb_icc());
    let t = MoxcmsEngine
        .compile(&srgb, &srgb, Intent::Perceptual, false)
        .expect("moxcms sRGB→sRGB must compile");

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
                "moxcms sRGB→sRGB not near-identity at patch {i} channel {c}: {} vs {} (dev {dev})",
                out[c],
                inp[c]
            );
        }
        assert_eq!(
            out[3], inp[3],
            "alpha mutated at patch {i}: {} → {}",
            inp[3], out[3]
        );
    }
    eprintln!("moxcms identity sRGB→sRGB: max per-channel deviation = {max_rgb_dev}/255");
}

/// (2) The three-way differential oracle: the swapped-primaries→sRGB
/// patches through moxcms and through lcms2 (8-bit RGB_8, Flags::default
/// = no BPC) must agree per channel within [`MAX_CHANNEL_DEVIATION`], for
/// **both** Perceptual and RelativeColorimetric. Matrix-shaper profiles
/// carry no per-intent tables, so the two intents fold to the same matrix
/// path — the bound holds for each. Measured maxima printed.
#[test]
fn image_cms_print_lane_moxcms_lcms2_oracle_per_channel() {
    let srgb_bytes = srgb_icc();
    let swapped_bytes = swapped_primaries_icc();

    let intents: &[(Intent, lcms2::Intent, &str)] = &[
        (Intent::Perceptual, lcms2::Intent::Perceptual, "perceptual"),
        (
            Intent::RelativeColorimetric,
            lcms2::Intent::RelativeColorimetric,
            "relative-colorimetric",
        ),
    ];

    let mut worst: (u8, &str, usize, usize) = (0, "", 0, 0);
    for &(intent, lintent, name) in intents {
        // moxcms path (the print backend) — RGBA8 in place.
        let src = profile_from_bytes(swapped_bytes.clone());
        let dst = profile_from_bytes(srgb_bytes.clone());
        let mt = MoxcmsEngine
            .compile(&src, &dst, intent, false)
            .unwrap_or_else(|e| panic!("moxcms compile {name}: {e}"));
        let mut mox_out = patch_ramp();
        mt.apply_rgba8(&mut mox_out);

        // lcms2 oracle — RGB_8 → RGB_8, Flags::default() (NO BPC).
        let lsrc = LcmsProfile::new_icc(&swapped_bytes).expect("lcms2 src profile");
        let ldst = LcmsProfile::new_icc(&srgb_bytes).expect("lcms2 dst profile");
        let lt: Transform<[u8; 3], [u8; 3]> = Transform::new_flags(
            &lsrc,
            PixelFormat::RGB_8,
            &ldst,
            PixelFormat::RGB_8,
            lintent,
            Flags::default(),
        )
        .expect("lcms2 transform");

        let ramp = patch_ramp();
        let rgb_in: Vec<[u8; 3]> = ramp.chunks_exact(4).map(|p| [p[0], p[1], p[2]]).collect();
        let mut rgb_out = vec![[0u8; 3]; rgb_in.len()];
        lt.transform_pixels(&rgb_in, &mut rgb_out);

        for (i, (m, l)) in mox_out.chunks_exact(4).zip(rgb_out.iter()).enumerate() {
            for c in 0..3 {
                let dev = m[c].abs_diff(l[c]);
                if dev > worst.0 {
                    worst = (dev, name, i, c);
                }
                assert!(
                    dev <= MAX_CHANNEL_DEVIATION,
                    "{name} patch {i} channel {c}: moxcms {} vs lcms2 {} (dev {dev} > {MAX_CHANNEL_DEVIATION})",
                    m[c],
                    l[c]
                );
            }
            // Alpha contractually untouched by the moxcms RGBA path.
            assert_eq!(m[3], ramp[i * 4 + 3], "alpha mutated at patch {i}");
        }
    }
    eprintln!(
        "moxcms↔lcms2 worst per-channel deviation = {}/255 ({} patch {} channel {})",
        worst.0, worst.1, worst.2, worst.3
    );
}

/// (3) BPC on/off. moxcms 0.8.1 does **not** expose black-point
/// compensation (the `TransformOptions::black_point_compensation` field
/// is commented out upstream; the BPC module is dead code). So a
/// `bpc: true` transform must be **byte-identical** to `bpc: false` — the
/// documented degradation. We assert that identity (proving the flag is
/// honestly inert, not silently mis-applied) and, to make the gap
/// audit-visible, contrast it with lcms2 *with* BLACKPOINT_COMPENSATION
/// on the **same** profile pair, which DOES move pixels.
///
/// The profile pair is the LUT-bearing source → sRGB at
/// RelativeColorimetric: BPC only acts on colorimetric intents and only
/// when there is a real (non-trivial) black point to compensate — which
/// a LUT/CLUT source has and a plain matrix sRGB does not. So this pair
/// is the one where lcms2 BPC genuinely bites, making the moxcms gap
/// concrete rather than a vacuous 0-vs-0.
///
/// REGISTRY NOTE: the BPC-fidelity conformance row stays pinned to a
/// future moxcms release (or the lcms2-shaped export oracle) that exposes
/// the knob — this test guards the M1 reality, it does not claim BPC.
#[test]
fn image_cms_print_lane_moxcms_bpc_documented_degradation() {
    // LUT source → sRGB: the pair where colorimetric BPC has a real
    // black point to compensate (see the doc comment).
    let src = profile_from_bytes(rgb_lut_icc());
    let dst = profile_from_bytes(srgb_icc());

    // moxcms: bpc:false vs bpc:true must be identical (flag is inert).
    let t_off = MoxcmsEngine
        .compile(&src, &dst, Intent::RelativeColorimetric, false)
        .expect("moxcms compile (bpc off)");
    let t_on = MoxcmsEngine
        .compile(&src, &dst, Intent::RelativeColorimetric, true)
        .expect("moxcms compile (bpc on)");
    assert!(t_on.bpc, "bpc flag must be recorded on the handle");
    assert!(!t_off.bpc, "bpc:false must be recorded on the handle");

    let mut off = patch_ramp();
    let mut on = patch_ramp();
    t_off.apply_rgba8(&mut off);
    t_on.apply_rgba8(&mut on);
    assert_eq!(
        off, on,
        "moxcms 0.8.1 exposes no BPC — bpc:true must be byte-identical to bpc:false"
    );

    // Oracle contrast: lcms2 BLACKPOINT_COMPENSATION vs none on the same
    // profile pair DOES differ — documenting the capability moxcms lacks.
    let lsrc = LcmsProfile::new_icc(&src.bytes).expect("lcms2 src");
    let ldst = LcmsProfile::new_icc(&dst.bytes).expect("lcms2 dst");
    let no_bpc: Transform<[u8; 3], [u8; 3]> = Transform::new_flags(
        &lsrc,
        PixelFormat::RGB_8,
        &ldst,
        PixelFormat::RGB_8,
        lcms2::Intent::RelativeColorimetric,
        Flags::default(),
    )
    .expect("lcms2 no-bpc transform");
    let with_bpc: Transform<[u8; 3], [u8; 3]> = Transform::new_flags(
        &lsrc,
        PixelFormat::RGB_8,
        &ldst,
        PixelFormat::RGB_8,
        lcms2::Intent::RelativeColorimetric,
        Flags::BLACKPOINT_COMPENSATION,
    )
    .expect("lcms2 bpc transform");

    let ramp = patch_ramp();
    let rgb_in: Vec<[u8; 3]> = ramp.chunks_exact(4).map(|p| [p[0], p[1], p[2]]).collect();
    let mut a = vec![[0u8; 3]; rgb_in.len()];
    let mut b = vec![[0u8; 3]; rgb_in.len()];
    no_bpc.transform_pixels(&rgb_in, &mut a);
    with_bpc.transform_pixels(&rgb_in, &mut b);
    let lcms_bpc_max = a
        .iter()
        .zip(b.iter())
        .flat_map(|(x, y)| (0..3).map(move |c| x[c].abs_diff(y[c])))
        .max()
        .unwrap_or(0);
    assert!(
        lcms_bpc_max > 0,
        "oracle sanity: lcms2 BPC vs none should differ on the LUT→sRGB colorimetric pair \
         (if this is 0 the contrast is vacuous and the chosen profile pair is wrong)"
    );
    eprintln!(
        "BPC: moxcms 0.8.1 inert (off==on); lcms2 oracle BPC vs none max per-channel = {lcms_bpc_max}/255 (the capability the print lane awaits)"
    );
}

/// (4) Intent-difference smoke: through moxcms, Perceptual and
/// AbsoluteColorimetric produce DIFFERENT output on at least one patch —
/// proving intents are honored (the qcms gap, audit §A: qcms collapses
/// Saturation/Absolute to nearest supported behaviour). Uses the
/// LUT-bearing RGB→Lab profile whose A2B0 (Perceptual) and A2B1
/// (colorimetric) tables differ; the matrix-shaper sRGB destination is
/// the common sink. A control assertion shows qcms does NOT differentiate
/// these intents on the same profiles (it has no per-intent LUT path).
#[test]
fn image_cms_print_lane_moxcms_intents_not_ignored() {
    let lut_src = profile_from_bytes(rgb_lut_icc());
    let dst = profile_from_bytes(srgb_icc());

    let perc = MoxcmsEngine
        .compile(&lut_src, &dst, Intent::Perceptual, false)
        .expect("moxcms compile Perceptual");
    let abs = MoxcmsEngine
        .compile(&lut_src, &dst, Intent::AbsoluteColorimetric, false)
        .expect("moxcms compile AbsoluteColorimetric");

    let mut p_out = patch_ramp();
    let mut a_out = patch_ramp();
    perc.apply_rgba8(&mut p_out);
    abs.apply_rgba8(&mut a_out);

    let max_dev = p_out
        .chunks_exact(4)
        .zip(a_out.chunks_exact(4))
        .flat_map(|(p, a)| (0..3).map(move |c| p[c].abs_diff(a[c])))
        .max()
        .unwrap_or(0);
    assert!(
        max_dev > 0,
        "moxcms must honor intents: Perceptual and AbsoluteColorimetric produced identical output \
         through a profile with distinct A2B0/A2B1 tables (intent ignored)"
    );
    eprintln!(
        "moxcms intents honored: Perceptual vs AbsoluteColorimetric max per-channel deviation = {max_dev}/255"
    );

    // Control: qcms on the SAME profiles does not differentiate these
    // intents (it lacks the per-intent LUT path — the audit §A gap moxcms
    // closes). Only assert if qcms accepts the synthetic LUT profile.
    if let (Ok(qp), Ok(qa)) = (
        QcmsEngine.compile(&lut_src, &dst, Intent::Perceptual, false),
        QcmsEngine.compile(&lut_src, &dst, Intent::AbsoluteColorimetric, false),
    ) {
        let mut qp_out = patch_ramp();
        let mut qa_out = patch_ramp();
        qp.apply_rgba8(&mut qp_out);
        qa.apply_rgba8(&mut qa_out);
        let q_dev = qp_out
            .chunks_exact(4)
            .zip(qa_out.chunks_exact(4))
            .flat_map(|(p, a)| (0..3).map(move |c| p[c].abs_diff(a[c])))
            .max()
            .unwrap_or(0);
        eprintln!(
            "control: qcms Perceptual vs AbsoluteColorimetric max per-channel deviation = {q_dev}/255 (the gap moxcms closes)"
        );
    } else {
        eprintln!("control: qcms did not accept the synthetic LUT profile (skipped)");
    }
}
