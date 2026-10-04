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

//! THE PHOTOSHOP ORACLE, replayed. `scripts/photoshop/run-probe.sh`
//! recorded what Photoshop does to fixed 16-bit stimuli (blend modes,
//! adjustments, filters); this file runs OUR engine on the same stimuli
//! and measures the distance, in 8-bit levels and ΔE00.
//!
//! Every case is CLASSIFIED in `fixtures/photoshop/ledger.json`:
//!
//! * `agreement` — within [`AGREE_LEVELS`] of Photoshop everywhere;
//! * `convention` — a documented, deliberate difference (the note says
//!   which, e.g. a textbook formula where Photoshop uses its own);
//! * `defect` — we claim Photoshop's behaviour (the PSD layer import maps
//!   Photoshop's blend keys onto these kernels) and do not deliver it.
//!   `defects_reach_agreement` fails while any case is classed so (none
//!   is, since every recorded defect was fixed); a new one is recorded
//!   as a red test, not hidden;
//! * `no-counterpart` — Photoshop recorded, nothing in the engine to
//!   compare (kept so the gap is visible, not forgotten).
//!
//! The always-on test re-measures every case and holds it to its ledger
//! row: a classification may not be missing, an agreement may not
//! drift, and no case may get WORSE than its recorded distance (an
//! improvement passes and is captured by regenerating:
//! `PAGED_PHOTOSHOP_LEDGER=write cargo test -p image-conformance --test
//! oracle_photoshop`). Scalar references run everywhere; cases whose
//! only engine path is a GPU chain need a device and skip without one
//! (`REQUIRE_GPU=1` makes that a failure).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use image_conformance::abr_corpus::json::Json;
use image_conformance::compose_ref::{composite, Blend};
use image_conformance::delta_e::{ciede2000, srgb_to_lab};
use image_conformance::Px;
use zune_core::bytestream::ZCursor;
use zune_png::PngDecoder;

/// Agreement: no pixel further than this from Photoshop, in 8-bit levels.
const AGREE_LEVELS: f64 = 2.0;
/// A re-measurement may exceed its recorded distance by this much
/// (GPU adapters differ in the last f16 bit).
const SLACK_LEVELS: f64 = 0.6;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/photoshop")
}

/// A 16-bit RGB image as [0, 1] floats.
struct Img {
    w: usize,
    h: usize,
    px: Vec<[f64; 3]>,
}

fn read_png16(path: &Path) -> Img {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut dec = PngDecoder::new(ZCursor::new(&bytes));
    let res = dec
        .decode()
        .unwrap_or_else(|e| panic!("{}: {e:?}", path.display()));
    let (w, h) = dec.dimensions().expect("dimensions");
    let ch = dec.colorspace().expect("colorspace").num_components();
    let samples: Vec<f64> = match res {
        zune_core::result::DecodingResult::U16(v) => {
            v.iter().map(|&s| s as f64 / 65535.0).collect()
        }
        zune_core::result::DecodingResult::U8(v) => v.iter().map(|&s| s as f64 / 255.0).collect(),
        _ => panic!("{}: unexpected sample type", path.display()),
    };
    assert!(ch >= 3, "{}: {ch} channel(s)", path.display());
    let px = samples
        .chunks_exact(ch)
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    Img { w, h, px }
}

/// The distance between our render and Photoshop's.
#[derive(Debug, Clone, Copy, Default)]
struct Dist {
    max: f64,
    mean: f64,
    p99: f64,
    de_p95: f64,
    de_max: f64,
}

fn dist(ours: &[[f64; 3]], theirs: &[[f64; 3]]) -> Dist {
    assert_eq!(ours.len(), theirs.len());
    let mut d: Vec<f64> = Vec::with_capacity(ours.len());
    let mut de: Vec<f64> = Vec::with_capacity(ours.len());
    let mut sum = 0.0;
    for (a, b) in ours.iter().zip(theirs) {
        let m = (0..3)
            .map(|c| (a[c].clamp(0.0, 1.0) - b[c]).abs() * 255.0)
            .fold(0.0, f64::max);
        sum += m;
        d.push(m);
        let c = |p: &[f64; 3]| {
            srgb_to_lab([
                p[0].clamp(0.0, 1.0),
                p[1].clamp(0.0, 1.0),
                p[2].clamp(0.0, 1.0),
            ])
        };
        de.push(ciede2000(c(a), c(b)));
    }
    let q = |v: &mut Vec<f64>, f: f64| {
        v.sort_by(|x, y| x.total_cmp(y));
        v[((v.len() as f64 * f).ceil() as usize).clamp(1, v.len()) - 1]
    };
    Dist {
        max: d.iter().copied().fold(0.0, f64::max),
        mean: sum / ours.len() as f64,
        p99: q(&mut d, 0.99),
        de_p95: q(&mut de, 0.95),
        de_max: de.iter().copied().fold(0.0, f64::max),
    }
}

/// One recorded case: its id and what the engine produced for it
/// (`None` = no engine counterpart / no device for its only path).
type Ours = Option<Vec<[f64; 3]>>;

/// A probe's engine side: case id -> our render.
type Engine = fn(&str) -> Ours;

// ─────────────────────────────── blends ──────────────────────────────

const BLENDS: [(&str, Blend); 26] = [
    ("normal", Blend::Normal),
    ("multiply", Blend::Multiply),
    ("screen", Blend::Screen),
    ("overlay", Blend::Overlay),
    ("darken", Blend::Darken),
    ("lighten", Blend::Lighten),
    ("color_dodge", Blend::ColorDodge),
    ("color_burn", Blend::ColorBurn),
    ("hard_light", Blend::HardLight),
    ("soft_light", Blend::SoftLight),
    ("difference", Blend::Difference),
    ("exclusion", Blend::Exclusion),
    ("hue", Blend::Hue),
    ("saturation", Blend::Saturation),
    ("color", Blend::Color),
    ("luminosity", Blend::Luminosity),
    ("linear_burn", Blend::LinearBurn),
    ("linear_dodge", Blend::LinearDodge),
    ("darker_color", Blend::DarkerColor),
    ("lighter_color", Blend::LighterColor),
    ("vivid_light", Blend::VividLight),
    ("linear_light", Blend::LinearLight),
    ("pin_light", Blend::PinLight),
    ("hard_mix", Blend::HardMix),
    ("subtract", Blend::Subtract),
    ("divide", Blend::Divide),
];

fn blend_case(id: &str) -> Ours {
    let stim = dir().join("stimuli");
    let bottom = read_png16(&stim.join("blend-bottom.png"));
    if id == "identity-bottom" {
        return Some(bottom.px);
    }
    let top = read_png16(&stim.join("blend-top.png"));
    let (mode, op) = id.split_once('@').expect("<mode>@<opacity>");
    let blend = BLENDS
        .iter()
        .find(|(n, _)| *n == mode)
        .expect("known mode")
        .1;
    let opacity = op.parse::<f32>().expect("opacity") / 100.0;
    let px = |p: &[f64; 3]| Px([p[0] as f32, p[1] as f32, p[2] as f32, 1.0]);
    Some(
        bottom
            .px
            .iter()
            .zip(&top.px)
            .map(|(a, b)| {
                let o = composite(px(a), px(b), opacity, blend);
                [o.0[0] as f64, o.0[1] as f64, o.0[2] as f64]
            })
            .collect(),
    )
}

// ───────────────────────── adjustments + filters ─────────────────────

/// The stimulus as straight rgba16float bytes (alpha 1) — what the
/// engine's f16 adjust chain consumes.
fn f16_of(img: &Img) -> Vec<u8> {
    img.px
        .iter()
        .flat_map(|p| [p[0], p[1], p[2], 1.0])
        .flat_map(|v| half::f16::from_f64(v).to_le_bytes())
        .collect()
}

fn from_f16(bytes: &[u8]) -> Vec<[f64; 3]> {
    bytes
        .chunks_exact(8)
        .map(|t| {
            std::array::from_fn(|c| half::f16::from_le_bytes([t[c * 2], t[c * 2 + 1]]).to_f64())
        })
        .collect()
}

fn sweep() -> Img {
    read_png16(&dir().join("stimuli/colour-sweep.png"))
}

/// The shipping adjust chain (`image_js::ingest::adjust_f16`) over the
/// sweep. Photoshop's parameters are mapped onto the engine's units the
/// way the panel exposes them; each mapping is stated in the ledger note.
fn adjust_case(id: &str) -> Ours {
    use image_js::ingest::{
        AdjustParams, BlackWhiteParams, ChannelMixerParams, ColorBalanceParams, LevelsParams,
        PhotoFilterParams,
    };
    let img = sweep();
    if id == "identity" {
        return Some(img.px);
    }
    let ctx = image_conformance::device::test_device()?;
    let mut p = AdjustParams::default();
    match id {
        "invert" => p.invert = true,
        "posterize-4" => p.posterize = Some(4.0),
        "threshold-128" => p.threshold = Some(128.0 / 255.0),
        "levels" => {
            p.levels = LevelsParams {
                in_black: 20.0 / 255.0,
                in_white: 230.0 / 255.0,
                gamma: 1.4,
                out_black: 10.0 / 255.0,
                out_white: 245.0 / 255.0,
            }
        }
        "curves" => {
            p.curve_lut = Some(image_core::curve_lut(
                &[
                    (0.0, 0.0),
                    (64.0, 40.0),
                    (128.0, 150.0),
                    (192.0, 220.0),
                    (255.0, 255.0),
                ]
                .map(|(i, o)| (i / 255.0, o / 255.0)),
            ))
        }
        "exposure+1" => p.exposure_ev = 1.0,
        "exposure-0.5" => p.exposure_ev = -0.5,
        "brightness-contrast-legacy" | "brightness-contrast" => {
            p.brightness = 30.0 / 255.0;
            p.contrast = 1.4;
        }
        "hue-saturation" => {
            p.hue_degrees = 40.0;
            p.saturation = 1.3;
        }
        "hue-only" => p.hue_degrees = 40.0,
        "vibrance" => p.vibrance = 0.5,
        "color-balance" => {
            p.color_balance = ColorBalanceParams {
                shadows: [0.0; 3],
                midtones: [0.3, -0.2, 0.1],
                highlights: [0.0; 3],
            }
        }
        "channel-mixer" => {
            p.channel_mixer = ChannelMixerParams {
                r: [0.8, 0.3, -0.1, 0.0],
                g: [0.0, 1.0, 0.0, 0.0],
                b: [0.0, 0.2, 0.8, 0.0],
            }
        }
        "photo-filter" => {
            p.photo_filter = PhotoFilterParams {
                color: [236.0 / 255.0, 138.0 / 255.0, 0.0],
                density: 0.5,
                preserve_luminosity: true,
            }
        }
        "black-white" => {
            p.black_white = BlackWhiteParams {
                enabled: true,
                weights: [0.4, 0.6, 0.4, 0.6, 0.2, 0.8],
            }
        }
        _ => panic!("adjustments: unknown case {id}"),
    }
    let out = pollster::block_on(image_js::ingest::adjust_f16(
        ctx,
        img.w as u32,
        img.h as u32,
        &f16_of(&img),
        &p,
        None,
    ))
    .expect("adjust chain");
    Some(from_f16(&out))
}

/// One registered kernel over the sweep, dispatched the way the wasm
/// filter doors do it (`apply_point_kernel`): windowed kernels read an
/// edge-clamped padded window, the rest run as point/resample modules.
fn kernel(def: &'static image_kernels::KernelDef, params: &[u8]) -> Ours {
    let ctx = image_conformance::device::test_device()?;
    let img = sweep();
    let (w, h) = (img.w as u32, img.h as u32);
    let src = f16_of(&img);
    let out = match def.class {
        image_kernels::KernelClass::Windowed { radius: (rx, ry) } => {
            let (rx, ry) = (u32::from(rx), u32::from(ry));
            let (ww, wh) = (w + 2 * rx, h + 2 * ry);
            let mut win = Vec::with_capacity((ww * wh * 8) as usize);
            for wy in 0..wh {
                let sy = (wy as i64 - ry as i64).clamp(0, h as i64 - 1) as usize;
                for wx in 0..ww {
                    let sx = (wx as i64 - rx as i64).clamp(0, w as i64 - 1) as usize;
                    let i = (sy * w as usize + sx) * 8;
                    win.extend_from_slice(&src[i..i + 8]);
                }
            }
            image_gpu::execute_windowed_once(ctx, def, &win, ww, wh, params, None, w, h)
        }
        _ => image_gpu::execute_tile_once(
            ctx,
            def,
            &[image_gpu::TileInput { f16_bytes: &src }],
            params,
            None,
            w,
            h,
        ),
    }
    .expect("kernel dispatch");
    Some(from_f16(&out))
}

fn filter_case(id: &str) -> Ours {
    use image_kernels::families::conv::{
        ConvEmbossParams, ConvFindEdgesParams, ConvMotionParams, CONV_EMBOSS, CONV_FIND_EDGES,
        CONV_MOTION,
    };
    use image_kernels::families::geom::{
        EdgePolicy, MosaicParams, OffsetParams, GEOM_MOSAIC, GEOM_OFFSET,
    };
    use image_kernels::families::morph::{MorphParams, RANK_MEDIAN3};
    let blur = |p: image_js::ingest::AdjustParams| -> Ours {
        let ctx = image_conformance::device::test_device()?;
        let img = sweep();
        let out = pollster::block_on(image_js::ingest::adjust_f16(
            ctx,
            img.w as u32,
            img.h as u32,
            &f16_of(&img),
            &p,
            None,
        ))
        .expect("adjust chain");
        Some(from_f16(&out))
    };
    match id {
        "identity" => Some(sweep().px),
        "gaussian-r2" | "gaussian-r5" => blur(image_js::ingest::AdjustParams {
            blur_sigma: if id == "gaussian-r2" { 2.0 } else { 5.0 },
            ..Default::default()
        }),
        "unsharp-100-r1.5" => blur(image_js::ingest::AdjustParams {
            sharpen_amount: 1.0,
            ..Default::default()
        }),
        "motion-30-r10" => kernel(&CONV_MOTION, ConvMotionParams::new(30.0, 10.0).as_bytes()),
        "emboss" => kernel(&CONV_EMBOSS, ConvEmbossParams::new(135.0, 3.0).as_bytes()),
        "find-edges" => kernel(&CONV_FIND_EDGES, ConvFindEdgesParams::new(1.0).as_bytes()),
        "mosaic-8" => kernel(&GEOM_MOSAIC, MosaicParams::new(8.0).as_bytes()),
        "median-r1" => kernel(&RANK_MEDIAN3, MorphParams::new().as_bytes()),
        "offset-20-10-wrap" => kernel(
            &GEOM_OFFSET,
            OffsetParams::new(20.0, 10.0, EdgePolicy::Wrap).as_bytes(),
        ),
        _ => panic!("filters: unknown case {id}"),
    }
}

// ─────────────────────────────── replay ──────────────────────────────

/// Every probe this file replays, with its engine side.
fn probes() -> Vec<(&'static str, Engine)> {
    vec![
        ("blend-modes", blend_case as Engine),
        ("adjustments", adjust_case),
        ("filters", filter_case),
    ]
}

struct Measured {
    key: String,
    dist: Option<Dist>,
}

fn measure_all() -> Vec<Measured> {
    let mut out = Vec::new();
    for (probe, engine) in probes() {
        let fx = std::fs::read_to_string(dir().join(format!("{probe}.photoshop.json")))
            .unwrap_or_else(|e| panic!("{probe}.photoshop.json: {e}"));
        let fx = Json::parse(&fx).expect("fixture JSON");
        for case in fx.get("cases").and_then(Json::as_array).expect("cases") {
            let id = case.get("id").and_then(Json::as_str).expect("id");
            let Some(output) = case.get("output").and_then(Json::as_str) else {
                continue;
            };
            let theirs = read_png16(&dir().join(output));
            let dist = engine(id).map(|ours| {
                assert_eq!(ours.len(), theirs.w * theirs.h, "{probe}/{id}: extent");
                dist(&ours, &theirs.px)
            });
            out.push(Measured {
                key: format!("{probe}/{id}"),
                dist,
            });
        }
    }
    out
}

fn r2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// The committed ledger: key → (class, note, recorded max).
fn ledger() -> BTreeMap<String, (String, String, Option<f64>)> {
    let Ok(text) = std::fs::read_to_string(dir().join("ledger.json")) else {
        return BTreeMap::new();
    };
    let j = Json::parse(&text).expect("ledger JSON");
    j.get("cases")
        .and_then(Json::as_object)
        .expect("cases")
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                (
                    v.get("class")
                        .and_then(Json::as_str)
                        .unwrap_or("")
                        .to_string(),
                    v.get("note")
                        .and_then(Json::as_str)
                        .unwrap_or("")
                        .to_string(),
                    v.get("max_levels").and_then(Json::as_f64),
                ),
            )
        })
        .collect()
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn write_ledger(measured: &[Measured], old: &BTreeMap<String, (String, String, Option<f64>)>) {
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut rows = String::new();
    for (i, m) in measured.iter().enumerate() {
        let (class, note) = match old.get(&m.key) {
            Some((c, n, _)) if !c.is_empty() => (c.clone(), n.clone()),
            _ => match m.dist {
                Some(d) if d.max <= AGREE_LEVELS => ("agreement".to_string(), String::new()),
                Some(_) => ("unclassified".to_string(), String::new()),
                None => ("no-counterpart".to_string(), String::new()),
            },
        };
        *counts.entry(class.clone()).or_default() += 1;
        let metrics = match m.dist {
            Some(d) => format!(
                ", \"max_levels\": {}, \"mean_levels\": {}, \"p99_levels\": {}, \"de00_p95\": {}, \"de00_max\": {}",
                r2(d.max),
                r2(d.mean),
                r2(d.p99),
                r2(d.de_p95),
                r2(d.de_max)
            ),
            None => String::new(),
        };
        let _ = writeln!(
            rows,
            "    \"{}\": {{\"class\": \"{class}\"{metrics}, \"note\": \"{}\"}}{}",
            m.key,
            esc(&note),
            if i + 1 < measured.len() { "," } else { "" }
        );
    }
    let summary = counts
        .iter()
        .map(|(k, v)| format!("\"{k}\": {v}"))
        .collect::<Vec<_>>()
        .join(", ");
    let text = format!(
        "{{\n  \"about\": \"Photoshop oracle ledger: every recorded case, classified, with our measured distance (8-bit levels, CIEDE2000). Regenerate the numbers with PAGED_PHOTOSHOP_LEDGER=write; classes and notes are kept.\",\n  \"agree_levels\": {AGREE_LEVELS},\n  \"summary\": {{{summary}}},\n  \"cases\": {{\n{rows}  }}\n}}\n"
    );
    std::fs::write(dir().join("ledger.json"), text).expect("write ledger");
}

#[test]
#[allow(non_snake_case)]
fn every_photoshop_case_holds_its_ledger_row__feat__image_conformance_harness() {
    let measured = measure_all();
    let old = ledger();
    if std::env::var_os("PAGED_PHOTOSHOP_LEDGER").is_some_and(|v| v == "write") {
        write_ledger(&measured, &old);
        for m in &measured {
            println!("{:<40} {:?}", m.key, m.dist.map(|d| r2(d.max)));
        }
        return;
    }
    let mut problems = Vec::new();
    for m in &measured {
        let Some((class, _, recorded)) = old.get(&m.key) else {
            problems.push(format!("{}: not in the ledger", m.key));
            continue;
        };
        match (class.as_str(), m.dist) {
            ("unclassified", _) | ("", _) => problems.push(format!("{}: unclassified", m.key)),
            ("agreement", Some(d)) if d.max > AGREE_LEVELS => problems.push(format!(
                "{}: agreement drifted to {:.2} levels",
                m.key, d.max
            )),
            (_, Some(d)) => {
                if let Some(r) = recorded {
                    if d.max > r + SLACK_LEVELS {
                        problems.push(format!(
                            "{}: {:.2} levels, worse than the recorded {r:.2}",
                            m.key, d.max
                        ));
                    }
                }
            }
            ("no-counterpart", None) => {}
            (_, None) => {
                if image_conformance::device::test_device().is_some() {
                    problems.push(format!(
                        "{}: classified {class} but nothing was measured",
                        m.key
                    ));
                }
            }
        }
    }
    assert!(
        problems.is_empty(),
        "{} problem(s):\n  {}",
        problems.len(),
        problems.join("\n  ")
    );
}

#[test]
#[allow(non_snake_case)]
fn defects_reach_agreement__feat__image_conformance_harness() {
    let led = ledger();
    let bad: Vec<String> = measure_all()
        .into_iter()
        .filter(|m| led.get(&m.key).is_some_and(|(c, _, _)| c == "defect"))
        .filter_map(|m| {
            m.dist
                .filter(|d| d.max > AGREE_LEVELS)
                .map(|d| format!("{}: {:.2} levels", m.key, d.max))
        })
        .collect();
    assert!(
        bad.is_empty(),
        "defects still open:\n  {}",
        bad.join("\n  ")
    );
}
