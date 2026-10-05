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

//! PHOTOSHOP-WRITTEN CMYK PSDs, replayed in CI. `cmyk-stacks.jsx` had
//! Photoshop build small CMYK documents in Coated FOGRA39 (the CMYK
//! working space of "Europe General Purpose 3"; the profile is EMBEDDED
//! in every PSD), save them with real merged data, and convert each
//! document — layers merged first, dither off — to sRGB three ways:
//! relative colorimetric + black-point compensation (the Color Settings'
//! default, read from the app and recorded in the fixture's `findings`),
//! relative colorimetric without BPC, and perceptual.
//!
//! Three questions, three kinds of test:
//!
//! 1. INK STORAGE — `patches` fills known ink percentages; the decode must
//!    read them back (the file stores ink INVERTED: 255 = no ink).
//! 2. WHICH CONVERSION — our ink transform (`image_js::cmyk::InkTransform`,
//!    moxcms Perceptual through the embedded profile) applied to the
//!    merged CMYK numbers, against Photoshop's own conversion of the same
//!    numbers. [`CONVERSION`] holds the distances.
//! 3. COMPOSITING SPACE — our layered import converts each plate and
//!    blends in RGB; Photoshop blends in CMYK and converts the result.
//!    Measured against OUR transform of Photoshop's merged CMYK composite
//!    (so the CMM difference of 2. is not charged to the blend) and,
//!    end to end, against Photoshop's sRGB conversion. [`STACKS`] and
//!    [`TILES`] hold the distances; [`TILES`] is what the `cmyk-blend`
//!    blocker rests on.

use std::path::PathBuf;

use image_conformance::abr_corpus::json::Json;
use image_conformance::psd_corpus::{compare_rgba8, merged_data, MergedData, Reference};
use image_js::cmyk::InkTransform;
use image_js::layers::{
    cmyk_flatten_agreement, cmyk_flatten_agrees, CmykFlattenAgreement, LayerStack,
};
use image_psd::PsdFile;
use zune_core::bytestream::ZCursor;
use zune_core::result::DecodingResult;
use zune_png::PngDecoder;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/photoshop/cmyk-stacks")
}

fn psd(id: &str) -> PsdFile {
    let bytes = std::fs::read(dir().join(format!("{id}.psd"))).expect("read fixture PSD");
    PsdFile::parse(&bytes).expect("parse")
}

fn fixture_json() -> Json {
    let p = dir().with_file_name("cmyk-stacks.photoshop.json");
    Json::parse(&std::fs::read_to_string(p).expect("fixture json")).expect("json")
}

/// One of Photoshop's sRGB conversions, as straight RGBA8 (16-bit PNGs
/// are rounded to 8 bits).
fn view(id: &str, view: &str) -> Vec<u8> {
    let bytes = std::fs::read(dir().join(format!("{id}.{view}.png"))).expect("read PNG");
    let mut dec = PngDecoder::new(ZCursor::new(&bytes));
    let px = dec.decode().expect("decode PNG");
    let ch = dec.colorspace().expect("colorspace").num_components();
    let v: Vec<u8> = match px {
        DecodingResult::U8(v) => v,
        DecodingResult::U16(v) => v
            .iter()
            .map(|&s| ((u32::from(s) * 255 + 32767) / 65535) as u8)
            .collect(),
        _ => panic!("unexpected PNG sample type"),
    };
    match ch {
        4 => v,
        3 => v
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        n => panic!("{n}-channel PNG"),
    }
}

/// OUR conversion of Photoshop's merged CMYK composite: the file's
/// embedded profile through the shipping ink transform, alpha re-attached.
fn ours_converted_merged(file: &PsdFile) -> Vec<u8> {
    let c = file.composite_cmyk8().expect("CMYK composite decodes");
    let ink = image_js::ingest::psd_ink_transform(file);
    assert!(
        ink.is_managed(),
        "the embedded FOGRA39 profile must compile"
    );
    let mut rgba = ink.to_rgba8(&c.cmyk);
    if let Some(a) = &c.alpha {
        for (p, &al) in rgba.chunks_exact_mut(4).zip(a) {
            p[3] = al;
        }
    }
    rgba
}

/// Our layered import, flattened in RGB, through the shipping gate:
/// `None` without a GPU; `Err(category)` when the import or the
/// agreement check declines; else the flatten and its agreement with our
/// conversion of Photoshop's merged composite. `measuring` lifts the
/// static `cmyk-blend` blocker AND skips the gate, so the disagreement
/// they rest on stays measured.
#[allow(clippy::type_complexity)]
fn ours_layered(
    file: &PsdFile,
    measuring: bool,
) -> Option<Result<(Vec<u8>, CmykFlattenAgreement), String>> {
    let ctx = image_conformance::device::test_device()?;
    let ink = image_js::ingest::psd_ink_transform(file);
    let convert = |cmyk: &[u8]| ink.to_rgba8(cmyk);
    let import = if measuring {
        file.layer_plates_rgba8_via_measuring_cmyk_blends(&convert)
    } else {
        file.layer_plates_rgba8_via(&convert)
    };
    let import = match import {
        Ok(i) => i,
        Err(e) => {
            let first = file
                .layer_import_blockers()
                .first()
                .map_or("other", |b| b.category);
            return Some(Err(format!("{first}: {e}")));
        }
    };
    assert!(
        import.converted_from_cmyk,
        "the import reports the conversion"
    );
    let stack = LayerStack::from_psd_plates(&import).expect("stack");
    let ours = pollster::block_on(stack.composite(Some(ctx), None))
        .expect("composite")
        .to_vec();
    let theirs = ours_converted_merged(file);
    let agreement = if measuring {
        cmyk_flatten_agreement(&ours, &theirs)
    } else {
        match cmyk_flatten_agrees(&ours, &theirs) {
            Ok(a) => a,
            Err(e) => return Some(Err(format!("cmyk-composite: {e}"))),
        }
    };
    Some(Ok((ours, agreement)))
}

/// (max over R/G/B, mean over R/G/B) in 8-bit levels, RGB over white.
fn dist(a: &[u8], b: &[u8]) -> (u8, f64) {
    let d = compare_rgba8(a, b, Reference::Straight);
    (
        d.channels[..3].iter().map(|c| c.max).max().unwrap_or(0),
        d.channels[..3].iter().map(|c| c.mean).sum::<f64>() / 3.0,
    )
}

const IDS: &[&str] = &[
    "patches",
    "transparent-soft",
    "normal-stack",
    "normal-stack-16",
    "opaque-stack",
    "blend-modes",
];

// ───────────────────────────── 1. ink storage ─────────────────────────

/// The decode reads back the ink Photoshop was told to fill, and reads
/// it the right way round: paper white is NO ink, 100 % cyan is 255 in
/// the cyan slot. Within one level of the percentage (Photoshop rounds
/// 50 % to 127).
#[test]
#[allow(non_snake_case)]
fn cmyk_composite_reads_the_ink_photoshop_filled__feat__image_psd_rendered() {
    let file = psd("patches");
    let c = file.composite_cmyk8().expect("decode");
    assert!(
        c.alpha.is_none(),
        "a saved alpha channel is NOT transparency"
    );
    assert_eq!(file.header.channels, 5, "4 inks + the saved alpha channel");
    let json = fixture_json();
    let case = json
        .get("cases")
        .and_then(Json::as_array)
        .and_then(|a| {
            a.iter()
                .find(|c| c.get("id").and_then(Json::as_str) == Some("patches"))
        })
        .expect("patches case");
    let inks = case
        .get("params")
        .and_then(|p| p.get("inks"))
        .and_then(Json::as_array)
        .expect("recorded inks");
    assert_eq!(inks.len(), 64);
    for (i, ink) in inks.iter().enumerate() {
        let want: Vec<f64> = ink
            .as_array()
            .expect("ink row")
            .iter()
            .map(|v| v.as_f64().expect("percent"))
            .collect();
        let (x, y) = ((i % 8) * 8 + 4, (i / 8) * 8 + 4);
        let o = (y * 64 + x) * 4;
        for (ch, (&pct, &stored)) in want.iter().zip(&c.cmyk[o..o + 4]).enumerate() {
            let expect = pct * 255.0 / 100.0;
            let got = f64::from(stored);
            assert!(
                (got - expect).abs() <= 1.0,
                "patch {i} channel {ch}: {got} for {pct}% ink"
            );
        }
    }
}

/// The LAYER channels store ink the same way (ids 0–3 = C, M, Y, K,
/// inverted), and `layer_plates_rgba8_via` hands the transform true ink:
/// read back through two probing "transforms" that expose the ink
/// itself, the interior of each `opaque-stack` layer is the ink Photoshop
/// filled it with.
#[test]
#[allow(non_snake_case)]
fn cmyk_layer_channels_decode_to_the_filled_ink__feat__image_psd_layer_import() {
    let file = psd("opaque-stack");
    let cmy = |ink: &[u8]| -> Vec<u8> {
        ink.chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect()
    };
    let k = |ink: &[u8]| -> Vec<u8> {
        ink.chunks_exact(4)
            .flat_map(|p| [p[3], p[3], p[3], 255])
            .collect()
    };
    let a = file.layer_plates_rgba8_via(&cmy).expect("imports");
    let b = file.layer_plates_rgba8_via(&k).expect("imports");
    let pct = |v: f64| (v * 255.0 / 100.0).round() as i32;
    // (layer index bottom-first, interior point, ink percent)
    let cases: &[(usize, (usize, usize), [f64; 4])] = &[
        (0, (8, 8), [70.0, 15.0, 0.0, 0.0]),
        (1, (60, 60), [0.0, 30.0, 95.0, 0.0]),
        (2, (200, 200), [60.0, 50.0, 40.0, 80.0]),
        (3, (76, 180), [75.0, 0.0, 90.0, 10.0]),
    ];
    for &(li, (x, y), ink) in cases {
        let o = (y * 256 + x) * 4;
        // Plates are bounded (ADR 464); read them on the canvas.
        let (ca, cb) = (
            a.layers[li].canvas_rgba8(a.width, a.height),
            b.layers[li].canvas_rgba8(b.width, b.height),
        );
        let got = [ca[o], ca[o + 1], ca[o + 2], cb[o]];
        for c in 0..4 {
            assert!(
                (i32::from(got[c]) - pct(ink[c])).abs() <= 1,
                "layer {li} at ({x},{y}): ink {got:?}, filled {ink:?} %"
            );
        }
        assert_eq!(ca[o + 3], 255, "opaque interior");
    }
}

// ─────────────────────────── 2. which conversion ──────────────────────

/// Our transform of Photoshop's merged CMYK numbers vs Photoshop's own
/// conversion of them, per view: (max, mean) levels over R/G/B.
///
/// Recorded 2026-10 against Photoshop 27.10 (Adobe ACE):
/// * Photoshop's DEFAULT view (relcol + BPC) is what we match: max 4,
///   mean 0.44 on the ink patches — the profile's perceptual table (the
///   one we use) and ACE's relcol + BPC agree as closely as lcms2's own
///   relcol + BPC does (max 4, mean 0.17, measured outside this suite).
/// * against relcol WITHOUT BPC we are 28 levels off on the patches —
///   the evidence that Photoshop's view compensates the black point.
/// * the 23-level outlier in `blend-modes` is a saturated cyan-green
///   outside sRGB that ACE clips differently from the ICC CMMs (lcms2
///   gives the same 23).
const CONVERSION: &[(&str, &str, u8, f64)] = &[
    ("patches", "relcol-bpc", 4, 0.45),
    ("patches", "relcol", 28, 5.3),
    ("transparent-soft", "relcol-bpc", 6, 0.9),
    ("normal-stack", "relcol-bpc", 8, 0.55),
    ("normal-stack-16", "relcol-bpc", 9, 0.6),
    ("opaque-stack", "relcol-bpc", 8, 0.91),
    ("blend-modes", "relcol-bpc", 23, 0.6),
];

#[test]
#[allow(non_snake_case)]
fn our_ink_transform_matches_photoshops_default_cmyk_view__feat__image_cms_print() {
    let json = fixture_json();
    let settings = json
        .get("findings")
        .and_then(|f| f.get("conversion_settings"))
        .expect("the probe recorded Photoshop's conversion settings");
    assert_eq!(
        settings.get("intent").and_then(Json::as_str),
        Some("colorimetric"),
        "Photoshop's default CMYK→RGB conversion is relative colorimetric"
    );
    assert_eq!(
        settings.get("mapBlack").and_then(Json::as_bool),
        Some(true),
        "… with black-point compensation"
    );
    let mut report = Vec::new();
    let mut problems = Vec::new();
    for &(id, v, max, mean) in CONVERSION {
        let file = psd(id);
        let ours = ours_converted_merged(&file);
        let theirs = view(id, v);
        let (m, me) = dist(&ours, &theirs);
        report.push(format!("{id:<18} vs {v:<11} max {m:>3} mean {me:.3}"));
        if m > max || me > mean {
            problems.push(format!(
                "{id} vs {v}: max {m} mean {me:.3}, recorded at most {max} / {mean}"
            ));
        }
    }
    // The ordering IS the finding: the default view is the near one.
    let patches = psd("patches");
    let ours = ours_converted_merged(&patches);
    let near = dist(&ours, &view("patches", "relcol-bpc")).1;
    let far = dist(&ours, &view("patches", "relcol")).1;
    assert!(
        near * 5.0 < far,
        "ours should sit on Photoshop's relcol+BPC view ({near:.2}), far from plain relcol ({far:.2})"
    );
    println!("{}", report.join("\n"));
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// The ingest door: a CMYK PSD now OPENS, says it was converted through
/// its profile, and is the same pixels as the transform above.
#[test]
#[allow(non_snake_case)]
fn a_cmyk_psd_opens_converted_through_its_profile__feat__image_cms_print() {
    for id in IDS {
        let bytes = std::fs::read(dir().join(format!("{id}.psd"))).expect("read");
        let img = image_js::ingest::decode_rgba8(&bytes).expect("a CMYK PSD decodes");
        assert_eq!(
            img.display,
            image_js::display::DisplayTreatment::CmykConverted,
            "{id}: the treatment says it was converted through the profile"
        );
        assert_eq!(img.depth_reduced, *id == "normal-stack-16", "{id}: depth");
        let file = psd(id);
        assert_eq!(
            &img.rgba.to_rgba8()[..],
            &ours_converted_merged(&file)[..],
            "{id}: the door and the shipping transform are the same pixels"
        );
    }
}

/// Without a profile the device formula converts, and the treatment says
/// THAT instead of claiming colour management.
#[test]
#[allow(non_snake_case)]
fn a_cmyk_conversion_without_a_profile_says_so__feat__image_cms_print() {
    let ink = InkTransform::for_profile(None);
    assert!(!ink.is_managed());
    assert_eq!(
        ink.treatment(),
        image_js::display::DisplayTreatment::CmykUncalibrated
    );
    assert!(ink.treatment().label().contains("no usable profile"));
    assert_eq!(&ink.to_rgba8(&[0, 0, 0, 0])[..], &[255, 255, 255, 255]);
}

/// A transparent CMYK document: alpha decodes exactly, and the ink is
/// un-matted (the merged image is matted against paper, i.e. no ink),
/// so it converts to Photoshop's straight sRGB colour.
#[test]
#[allow(non_snake_case)]
fn a_transparent_cmyk_composite_is_unmatted_against_paper__feat__image_psd_rendered() {
    let file = psd("transparent-soft");
    assert!(file.layer_mask.transparency_in_merged);
    let ours = ours_converted_merged(&file);
    let theirs = view("transparent-soft", "relcol-bpc");
    let mut worst = 0u8;
    for (a, b) in ours.chunks_exact(4).zip(theirs.chunks_exact(4)) {
        assert_eq!(a[3], b[3], "alpha decodes exactly");
        if b[3] >= 64 {
            worst = worst.max((0..3).map(|c| a[c].abs_diff(b[c])).max().unwrap_or(0));
        }
    }
    // Left matted, the soft edge drifts toward white by up to ~200 levels.
    assert!(
        worst <= 8,
        "{worst} levels from Photoshop's straight colour at α ≥ 64"
    );
}

// ───────────────────────── 3. compositing space ───────────────────────

/// `Ok((max vs our conversion of the merged CMYK, max vs Photoshop's
/// sRGB view))` or `Err(category)`.
type StackOutcome = Result<(u8, u8), &'static str>;

/// Each document's layered import through the shipping path (static
/// blockers, then the composite-agreement gate): `Ok((max vs our
/// conversion of Photoshop's merged CMYK, max vs Photoshop's own sRGB
/// view))`, or `Err(category)` when it declines.
///
/// Recorded 2026-10 against Photoshop 27.10:
/// * `opaque-stack` (hard-edged opaque rectangles + an anti-aliased disc,
///   256 px) IMPORTS: exact everywhere but the disc's rim, where RGB and
///   CMYK mixing differ by up to 25 levels on 0.42 % of the canvas (the
///   gate allows 1 %); 24 levels from Photoshop's own sRGB view at worst;
/// * `transparent-soft` (one feathered layer over transparency — nothing
///   to mix with) IMPORTS within 2 levels;
/// * `normal-stack` is DECLINED by the gate: its 60 % layer, 50 % group
///   and feathered edges mix colours, and RGB mixing of the converted
///   colours lands up to 61 levels from Photoshop's CMYK mixing;
/// * `blend-modes` is DECLINED before decoding (`cmyk-blend`);
/// * `normal-stack-16` has no top-level layer records: Photoshop writes a
///   16-bit document's layers into the `Lr16` block, which the import
///   does not read yet (`no-layers`).
const STACKS: &[(&str, StackOutcome)] = &[
    ("opaque-stack", Ok((25, 24))),
    ("transparent-soft", Ok((2, 1))),
    ("normal-stack", Err("cmyk-composite")),
    ("normal-stack-16", Err("no-layers")),
    ("blend-modes", Err("cmyk-blend")),
];

#[test]
#[allow(non_snake_case)]
fn cmyk_layer_stacks_import_or_decline_as_recorded__feat__image_psd_layer_import() {
    let mut report = Vec::new();
    let mut problems = Vec::new();
    for &(id, want) in STACKS {
        let file = psd(id);
        assert_eq!(merged_data(&file), MergedData::Real, "{id}: 0x0421 vouches");
        let blockers: Vec<&str> = file
            .layer_import_blockers()
            .iter()
            .map(|b| b.category)
            .collect();
        assert!(
            !blockers.contains(&"colour-mode"),
            "{id}: CMYK is no longer a colour-mode blocker"
        );
        let Some(got) = ours_layered(&file, false) else {
            eprintln!("SKIP {id}: no GPU adapter");
            continue;
        };
        let got: Result<(u8, u8), String> = match got {
            Ok((ours, a)) => {
                report.push(format!(
                    "{id:<18} gate: {:.2}% off, mean {:.3}, max {}",
                    a.pct_off, a.mean, a.max
                ));
                Ok((
                    dist(&ours, &ours_converted_merged(&file)).0,
                    dist(&ours, &view(id, "relcol-bpc")).0,
                ))
            }
            Err(e) => {
                report.push(format!("{id:<18} {e}"));
                Err(e.split(':').next().unwrap_or("").to_string())
            }
        };
        report.push(format!("{id:<18} {got:?}"));
        match (want, &got) {
            (Ok((a, b)), Ok((x, y))) if *x <= a && *y <= b => {}
            (Err(w), Err(g)) if g == w => {}
            _ => problems.push(format!("{id}: expected {want:?}, got {got:?}")),
        }
    }
    println!("{}", report.join("\n"));
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// The measurement under the gate's verdict on `normal-stack`: how far
/// RGB mixing of the converted colours lands from Photoshop's CMYK
/// mixing when layers are partly transparent — (pct of pixels more than
/// 8 levels off, max levels).
#[test]
#[allow(non_snake_case)]
fn normal_mixing_of_converted_cmyk_is_measured__feat__image_psd_layer_import() {
    let file = psd("normal-stack");
    let Some(got) = ours_layered(&file, true) else {
        eprintln!("SKIP normal-stack: no GPU adapter");
        return;
    };
    let (_, a) = got.expect("normal-stack has no static blocker");
    println!(
        "normal-stack: {:.2}% off, mean {:.3}, max {}",
        a.pct_off, a.mean, a.max
    );
    assert!(
        a.pct_off > image_js::layers::CMYK_FLATTEN_MAX_PCT,
        "normal-stack now agrees ({:.2}% off) — the gate would import it; re-record STACKS",
        a.pct_off
    );
    assert!(a.max <= NORMAL_STACK_MAX, "worse than recorded: {}", a.max);
}

/// The recorded worst of `normal-stack` (levels): 61.1 % of its pixels
/// are more than 8 levels off, mean 19.9.
const NORMAL_STACK_MAX: u8 = 61;

/// Per blend-mode tile of `blend-modes`: our RGB blend of the converted
/// plates vs our transform of Photoshop's CMYK blend — (max levels, ΔE00
/// p95) — and whether the mode passes the static blocker. The normal
/// tiles pass it and are then judged by the gate; every other mode was
/// measured far off across its footprint, which is the `cmyk-blend`
/// blocker (`image_psd::layer_pixels::CMYK_RGB_SAFE_BLENDS`).
const TILES: &[(&str, u8, f64, bool)] = &[
    ("normal-100", 56, 9.66, true),
    ("normal-50", 49, 13.76, true),
    ("multiply", 74, 20.02, false),
    ("screen", 95, 13.63, false),
    ("overlay", 43, 12.47, false),
    ("darken", 72, 22.65, false),
    ("lighten", 69, 26.77, false),
    ("color-burn", 225, 50.26, false),
    ("color-dodge", 153, 24.50, false),
    ("soft-light", 36, 10.37, false),
    ("hard-light", 104, 19.78, false),
    ("difference", 181, 43.27, false),
    ("exclusion", 183, 49.85, false),
    ("linear-burn", 85, 22.98, false),
    ("linear-dodge", 111, 19.50, false),
    ("hue", 64, 20.12, false),
    ("saturation", 75, 9.88, false),
    ("color", 61, 16.66, false),
    ("luminosity", 82, 18.71, false),
    ("multiply-50", 44, 12.26, false),
];

#[test]
#[allow(non_snake_case)]
fn blending_converted_cmyk_layers_in_rgb_is_measured_per_mode__feat__image_psd_layer_import() {
    let file = psd("blend-modes");
    let Some(got) = ours_layered(&file, true) else {
        eprintln!("SKIP blend-modes: no GPU adapter");
        return;
    };
    let (ours, _) = got.expect("the measuring import lifts only cmyk-blend");
    let theirs = ours_converted_merged(&file);
    let w = file.header.width as usize;
    let pick = |buf: &[u8], x0: usize, y0: usize| -> Vec<u8> {
        (y0..y0 + 32)
            .flat_map(|y| buf[(y * w + x0) * 4..(y * w + x0 + 32) * 4].to_vec())
            .collect()
    };
    let mut report = Vec::new();
    let mut problems = Vec::new();
    for (i, &(id, max, de_p95, normal)) in TILES.iter().enumerate() {
        let (x0, y0) = ((i % 5) * 32, (i / 5) * 32);
        let d = compare_rgba8(
            &pick(&ours, x0, y0),
            &pick(&theirs, x0, y0),
            Reference::Straight,
        );
        let m = d.channels[..3].iter().map(|c| c.max).max().unwrap_or(0);
        report.push(format!(
            "{id:<13} max {m:>3}  dE00 p95 {:>6.2}  {}",
            d.delta_e_p95,
            if normal {
                "-> the gate judges it"
            } else {
                "-> cmyk-blend"
            }
        ));
        if m > max || d.delta_e_p95 > de_p95 + 0.05 {
            problems.push(format!(
                "{id}: max {m} dE00 p95 {:.2}, recorded {max} / {de_p95}",
                d.delta_e_p95
            ));
        }
        if !normal && d.delta_e_p95 < 2.0 {
            problems.push(format!(
                "{id}: now within dE00 p95 {:.2} — reconsider the cmyk-blend blocker for it",
                d.delta_e_p95
            ));
        }
    }
    let blocked: Vec<&str> = file
        .layer_import_blockers()
        .iter()
        .map(|b| b.category)
        .collect();
    // Normal and Multiply are admitted and judged per file against the
    // composite (`CMYK_RGB_SAFE_BLENDS`); every other mode is a blocker.
    assert_eq!(
        blocked.len(),
        TILES
            .iter()
            .filter(|t| !t.3 && !t.0.starts_with("multiply"))
            .count(),
        "one cmyk-blend blocker per layer in a mode other than normal/multiply: {blocked:?}"
    );
    assert!(blocked.iter().all(|b| *b == "cmyk-blend"));
    println!("{}", report.join("\n"));
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
