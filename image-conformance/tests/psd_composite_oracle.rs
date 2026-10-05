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

//! THE CORPUS COMPOSITE ORACLE: Photoshop's own merged composite, which
//! every real PSD saved with "maximize compatibility" carries, against
//! the flatten OUR layer stack produces from the same file's layers.
//!
//! For each corpus file (selected by the `8BPS` magic, never by
//! extension) the lane:
//!
//! 1. classifies it — colour mode, depth, and the layer features it uses
//!    (groups, masks, clipping, adjustment and fill layers, effects,
//!    smart objects, text, blend-if, …);
//! 2. requires REAL merged data — resource 0x0421's `hasRealMergedData`.
//!    A placeholder composite is not an oracle, and a file without the
//!    resource has nothing vouching for its composite;
//! 3. imports the layers through the shipping path
//!    (`PsdFile::layer_plates_rgba8` → `LayerStack::from_psd_plates`, the
//!    native body of the `layers_open_from_psd` door) and composites them
//!    on the GPU;
//! 4. compares: per-channel absolute difference (max, mean, p99, % of
//!    pixels over 2 levels; RGB over white, alpha raw) and ΔE00 (p95,
//!    max).
//!
//! CMYK documents: the plates go through `layer_plates_rgba8_via` with
//! the shipping ink transform (`image_js::ingest::psd_ink_transform`, the
//! file's embedded profile), and `theirs` is Photoshop's merged CMYK
//! composite converted by THAT SAME transform — so the comparison is
//! ours-vs-Photoshop in one space and does not charge the CMM (measured
//! separately against Photoshop's own conversion in
//! `psd_cmyk_photoshop.rs`) to the layer decode. The shipping door's
//! blend-space gate (`layers::cmyk_flatten_agrees`) is applied too, so a
//! CMYK file is `compared` only when the door would import it, and the
//! row records the gate's numbers (`cmyk_gate`) either way.
//!
//! Outcomes: `compared`, `refused` (the engine's own reason, reduced to
//! a category) or `no-merged-data`. The aggregate ledger
//! `fixtures/psd-corpus/composite-ledger.json` is keyed by SHA-256 and
//! carries ONLY aggregates and the classification — the corpus is
//! private, so no name, path or pixel leaves it.
//!
//! RATCHET: a run fails, and leaves the ledger untouched, when fewer
//! files are compared than the committed ledger records, or when a file
//! the ledger records as compared is no longer compared. The always-on
//! `composite_ledger_*` tests hold the committed ledger to the same
//! floor in CI.
//!
//! OPT-IN (needs the private corpus and a GPU; run optimised — the
//! files total several GB):
//!
//! ```text
//! (A file that is now REFUSED on purpose — the import learned it was
//! approximating — is accepted with `PAGED_PSD_LEDGER=accept-refusals`.)
//!
//! PAGED_PSD_CORPUS=1 cargo test --release -p image-conformance \
//!   --test psd_composite_oracle -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use image_conformance::abr_corpus::json::Json;
use image_conformance::abr_corpus::sha256::sha256_hex;
use image_conformance::psd_corpus::{
    classify, compare_rgba8, corpus_root, merged_data, psds_by_magic, CompositeDiff, Features,
    MergedData, Reference,
};
use image_js::layers::LayerStack;
use image_psd::PsdFile;

/// The committed ledger may never record fewer compared files than
/// this. Raise it in the commit that earns it; never lower it.
const FLOOR_COMPARED: u64 = 2;

fn ledger_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/psd-corpus/composite-ledger.json")
}

/// One file's row.
struct Row {
    width: u32,
    height: u32,
    features: Features,
    merged: MergedData,
    outcome: &'static str,
    reason: Option<(String, String)>,
    diff: Option<CompositeDiff>,
    /// EVERY blocker's category, deduplicated, in file order — the
    /// refusal names only the first; this says how far the file is.
    blockers: Vec<&'static str>,
    /// Smart objects checked against the composite: (count, worst % of a
    /// footprint off, worst mean levels).
    smart: Option<(usize, f64, f64)>,
    /// A CMYK document's blend-space gate, when it was reached: (% of
    /// pixels more than 8 levels off, mean, max).
    cmyk_gate: Option<(f64, f64, u8)>,
    /// What the flattened open (`ingest::decode_rgba8`, the merged
    /// composite) does with the file: its colour treatment, or
    /// `refused`.
    opens: &'static str,
}

/// Reduce an engine refusal to a stable category, and drop anything
/// quoted (layer names are file content and stay in the corpus).
fn categorize(reason: &str) -> (String, String) {
    let mut clean = String::new();
    let mut quoted = false;
    for ch in reason.chars() {
        if ch == '"' {
            quoted = !quoted;
            if quoted {
                clean.push_str("\"…\"");
            }
            continue;
        }
        if !quoted {
            clean.push(ch);
        }
    }
    let cat = if reason.contains("CMYK document with a") {
        "cmyk-blend"
    } else if reason.contains("CMYK document without real merged data") {
        "cmyk-unverified"
    } else if reason.contains("CMYK document whose layers") {
        "cmyk-composite"
    } else if reason.contains("GROUP with a mask") {
        "group-mask"
    } else if reason.contains("VECTOR MASK") || reason.contains("vector-derived") {
        "vector-mask"
    } else if reason.contains("density/feather") {
        "mask-parameters"
    } else if reason.contains("layer effects") {
        "effects"
    } else if reason.contains("adjustment layer") {
        "adjustment-layers"
    } else if reason.contains("smart object") {
        "smart-objects"
    } else if reason.contains("artboard") {
        "artboards"
    } else if reason.contains("FILL OPACITY") {
        "fill-opacity"
    } else if reason.contains("color mode") || reason.contains("colour mode") {
        "colour-mode"
    } else if reason.contains("no layer records") || reason.contains("no layers") {
        "no-layers"
    } else if reason.contains("budget") {
        "budget"
    } else if reason.contains("16-bit RLE") {
        "16-bit-rle"
    } else if reason.contains("depth") {
        "depth"
    } else if reason.starts_with("gpu:") {
        "gpu"
    } else if reason.starts_with("merged composite:") {
        "merged-decode"
    } else {
        "other"
    };
    (cat.to_string(), clean)
}

fn evaluate(bytes: &[u8], ctx: &image_gpu::GpuContext) -> Result<Row, String> {
    let psd = PsdFile::parse(bytes).map_err(|e| format!("parse: {e}"))?;
    let features = classify(&psd);
    let merged = merged_data(&psd);
    let mut row = Row {
        width: psd.header.width,
        height: psd.header.height,
        features,
        merged,
        outcome: "no-merged-data",
        reason: None,
        diff: None,
        blockers: {
            let mut b: Vec<&'static str> = Vec::new();
            for x in psd.layer_import_blockers() {
                if !b.contains(&x.category) {
                    b.push(x.category);
                }
            }
            b
        },
        smart: None,
        cmyk_gate: None,
        opens: match image_js::ingest::decode_rgba8(bytes) {
            Ok(img) => {
                use image_js::display::DisplayTreatment as D;
                match img.display {
                    D::Managed => "managed",
                    D::AssumedSrgb => "assumed-srgb",
                    D::ProfileRejected => "profile-rejected",
                    D::CmykConverted => "cmyk-converted",
                    D::CmykUncalibrated => "cmyk-uncalibrated",
                }
            }
            Err(_) => "refused",
        },
    };
    if merged != MergedData::Real {
        return Ok(row);
    }
    row.outcome = "refused";
    // A CMYK document goes through the shipping conversion: its plates
    // AND the merged composite through the SAME ink transform (the file's
    // embedded profile), so `theirs` is Photoshop's CMYK composite in our
    // RGB and the comparison is ours-vs-Photoshop in one space — the CMM
    // is not on trial here, only the layer decode and the blend space.
    let cmyk = psd.header.color_mode == image_psd::model::ColorMode::Cmyk;
    let ink = image_js::ingest::psd_ink_transform(&psd);
    let convert = |c: &[u8]| ink.to_rgba8(c);
    let import = match if cmyk {
        psd.layer_plates_rgba8_via(&convert)
    } else {
        psd.layer_plates_rgba8()
    } {
        Ok(i) => i,
        Err(e) => {
            row.reason = Some(categorize(&e.to_string()));
            return Ok(row);
        }
    };
    let theirs = match merged_rgba8(&psd, &ink) {
        Ok(c) => c,
        Err(e) => {
            row.reason = Some(categorize(&format!("merged composite: {e}")));
            return Ok(row);
        }
    };
    let stack = match LayerStack::from_psd_plates(&import) {
        Ok(s) => s,
        Err(e) => {
            row.reason = Some(categorize(&e.to_string()));
            return Ok(row);
        }
    };
    let ours = match pollster::block_on(stack.composite(Some(ctx), None)) {
        Ok(px) => px,
        Err(e) => {
            row.reason = Some(categorize(&format!("gpu: {e}")));
            return Ok(row);
        }
    };
    // The shipping gate: every smart object's stored render must agree
    // with the composite inside its own footprint.
    let checked = image_js::layers::smart_render_agreement(&import, &ours, &theirs);
    if !checked.is_empty() {
        for a in &checked {
            println!(
                "    smart footprint {:>9} px  off {:>6.2}%  mean {:>6.2}",
                a.footprint, a.pct_off, a.mean
            );
        }
        row.smart = Some((
            checked.len(),
            checked.iter().map(|a| a.pct_off).fold(0.0, f64::max),
            checked.iter().map(|a| a.mean).fold(0.0, f64::max),
        ));
    }
    if let Err(e) = image_js::layers::smart_renders_agree(&import, &ours, &theirs) {
        row.reason = Some(categorize(&e.to_string()));
        return Ok(row);
    }
    // The CMYK gate the shipping door applies: the RGB blend of the
    // converted plates must agree with the converted composite.
    if cmyk {
        let a = image_js::layers::cmyk_flatten_agreement(&ours, &theirs);
        println!(
            "    cmyk gate: {:.2}% off, mean {:.2}, max {}",
            a.pct_off, a.mean, a.max
        );
        row.cmyk_gate = Some((a.pct_off, a.mean, a.max));
        if let Err(e) = image_js::layers::cmyk_flatten_agrees(&ours, &theirs) {
            row.reason = Some(categorize(&e.to_string()));
            return Ok(row);
        }
    }
    drop(import);
    // The decoder un-mattes a transparent document's merged composite,
    // so both sides are straight colour.
    let reference = Reference::Straight;
    row.diff = Some(compare_rgba8(&ours, &theirs, reference));
    row.outcome = "compared";
    Ok(row)
}

/// Photoshop's merged composite as straight RGBA8: decoded for RGB and
/// Grayscale; for CMYK the ink through `ink` — the same transform the
/// plates took.
fn merged_rgba8(psd: &PsdFile, ink: &image_js::cmyk::InkTransform) -> Result<Vec<u8>, String> {
    if psd.header.color_mode == image_psd::model::ColorMode::Cmyk {
        let c = psd.composite_cmyk8().map_err(|e| e.to_string())?;
        let mut rgba = ink.to_rgba8(&c.cmyk);
        if let Some(a) = &c.alpha {
            for (p, &al) in rgba.chunks_exact_mut(4).zip(a) {
                p[3] = al;
            }
        }
        Ok(rgba)
    } else {
        psd.composite_rgba8()
            .map(|c| c.rgba)
            .map_err(|e| e.to_string())
    }
}

// ───────────────────────────── the ledger ─────────────────────────────

fn num(v: f64) -> String {
    let r = (v * 1e4).round() / 1e4;
    if r == 0.0 {
        "0".into()
    } else {
        format!("{r}")
    }
}

fn features_json(f: &Features) -> String {
    format!(
        "{{\"color_mode\": \"{}\", \"depth\": {}, \"psb\": {}, \"layer_records\": {}, \"tags\": [{}]}}",
        f.color_mode,
        f.depth,
        f.psb,
        f.layer_records,
        f.tags()
            .iter()
            .map(|t| format!("\"{t}\""))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn diff_json(d: &CompositeDiff) -> String {
    let ch = ["r", "g", "b", "a"]
        .iter()
        .zip(d.channels.iter())
        .map(|(n, c)| {
            format!(
                "\"{n}\": {{\"max\": {}, \"mean\": {}, \"p99\": {}, \"pct_over_2\": {}}}",
                c.max,
                num(c.mean),
                c.p99,
                num(c.pct_over_2)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{{{ch}, \"delta_e00\": {{\"mean\": {}, \"p95\": {}, \"max\": {}, \"samples\": {}}}}}",
        num(d.delta_e_mean),
        num(d.delta_e_p95),
        num(d.delta_e_max),
        d.delta_e_samples
    )
}

fn ledger_json(rows: &BTreeMap<String, Row>) -> String {
    let mut by_outcome: BTreeMap<&str, u64> = BTreeMap::new();
    let mut by_reason: BTreeMap<String, u64> = BTreeMap::new();
    let mut by_merged: BTreeMap<&str, u64> = BTreeMap::new();
    let mut by_blocker: BTreeMap<String, u64> = BTreeMap::new();
    let mut by_opens: BTreeMap<&str, u64> = BTreeMap::new();
    for r in rows.values() {
        *by_opens.entry(r.opens).or_default() += 1;
        for b in &r.blockers {
            *by_blocker.entry(b.to_string()).or_default() += 1;
        }
        *by_outcome.entry(r.outcome).or_default() += 1;
        *by_merged.entry(r.merged.as_str()).or_default() += 1;
        if let Some((cat, _)) = &r.reason {
            *by_reason.entry(cat.clone()).or_default() += 1;
        }
    }
    for o in ["compared", "refused", "no-merged-data"] {
        by_outcome.entry(o).or_default();
    }
    let obj = |m: Vec<(String, u64)>| {
        m.iter()
            .map(|(k, v)| format!("\"{k}\": {v}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str("  \"about\": \"Merged-composite oracle over the private PSD corpus: Photoshop's own merged composite vs our layer-stack flatten (a CMYK document: both sides through the same conversion of its embedded profile, so the comparison is in one space). Keyed by SHA-256; aggregates and feature classification only. Generated by image-conformance/tests/psd_composite_oracle.rs.\",\n");
    let _ = writeln!(s, "  \"files\": {},", rows.len());
    let _ = writeln!(
        s,
        "  \"outcomes\": {{{}}},",
        obj(by_outcome
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect())
    );
    let _ = writeln!(
        s,
        "  \"merged_data\": {{{}}},",
        obj(by_merged.iter().map(|(k, v)| (k.to_string(), *v)).collect())
    );
    let _ = writeln!(
        s,
        "  \"refusals\": {{{}}},",
        obj(by_reason.into_iter().collect())
    );
    let _ = writeln!(
        s,
        "  \"blockers\": {{{}}},",
        obj(by_blocker.into_iter().collect())
    );
    let _ = writeln!(
        s,
        "  \"opens\": {{{}}},",
        obj(by_opens.iter().map(|(k, v)| (k.to_string(), *v)).collect())
    );
    s.push_str("  \"rows\": {\n");
    let n = rows.len();
    for (i, (sha, r)) in rows.iter().enumerate() {
        let reason = match &r.reason {
            Some((cat, text)) => format!(
                ", \"reason\": {{\"category\": \"{cat}\", \"engine\": \"{}\"}}",
                text.replace('\\', "\\\\").replace('"', "\\\"")
            ),
            None => String::new(),
        };
        let diff = match &r.diff {
            Some(d) => format!(", \"diff\": {}", diff_json(d)),
            None => String::new(),
        };
        let blockers = format!(
            ", \"blockers\": [{}]",
            r.blockers
                .iter()
                .map(|b| format!("\"{b}\""))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let smart = match r.smart {
            Some((n, pct, mean)) => format!(
                ", \"smart\": {{\"checked\": {n}, \"worst_pct_off\": {}, \"worst_mean\": {}}}",
                num(pct),
                num(mean)
            ),
            None => String::new(),
        };
        let cmyk_gate = match r.cmyk_gate {
            Some((pct, mean, max)) => format!(
                ", \"cmyk_gate\": {{\"pct_off\": {}, \"mean\": {}, \"max\": {max}}}",
                num(pct),
                num(mean)
            ),
            None => String::new(),
        };
        let _ = write!(
            s,
            "    \"{sha}\": {{\"outcome\": \"{}\", \"opens\": \"{}\", \"merged_data\": \"{}\", \"width\": {}, \"height\": {}, \"features\": {}{blockers}{smart}{cmyk_gate}{reason}{diff}}}",
            r.outcome,
            r.opens,
            r.merged.as_str(),
            r.width,
            r.height,
            features_json(&r.features)
        );
        s.push_str(if i + 1 < n { ",\n" } else { "\n" });
    }
    s.push_str("  }\n}\n");
    s
}

/// The committed ledger, parsed (`None` when there is none yet).
fn committed_ledger() -> Option<Json> {
    let text = std::fs::read_to_string(ledger_path()).ok()?;
    Some(Json::parse(&text).expect("committed composite ledger is JSON"))
}

fn outcome_count(ledger: &Json, outcome: &str) -> u64 {
    ledger
        .get("outcomes")
        .and_then(|o| o.get(outcome))
        .and_then(Json::as_u64)
        .unwrap_or(0)
}

// ───────────────────────────── the lane ──────────────────────────────

#[test]
#[allow(non_snake_case)]
#[ignore = "psd corpus oracle: opt-in (PAGED_PSD_CORPUS=1 + the private corpus + a GPU)"]
fn photoshop_merged_composite_vs_our_layer_flatten__feat__image_psd_layer_import() {
    let Some(root) = corpus_root("PAGED_PSD_CORPUS") else {
        return;
    };
    let Some(ctx) = image_conformance::device::test_device() else {
        eprintln!("SKIP psd composite oracle: no GPU adapter");
        return;
    };
    let files = psds_by_magic(&root);
    println!(
        "psd composite oracle: {} file(s) by 8BPS magic",
        files.len()
    );
    assert!(!files.is_empty(), "no PSD content under {}", root.display());

    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    let mut failures = Vec::new();
    for f in &files {
        let bytes = std::fs::read(&f.path).expect("read corpus file");
        let sha = sha256_hex(&bytes);
        match evaluate(&bytes, ctx) {
            Ok(row) => {
                let detail = match (&row.diff, &row.reason) {
                    (Some(d), _) => format!(
                        "dE00 p95 {:.2} max {:.2}; max|d| r{} g{} b{} a{}; >2: {:.1}%",
                        d.delta_e_p95,
                        d.delta_e_max,
                        d.channels[0].max,
                        d.channels[1].max,
                        d.channels[2].max,
                        d.channels[3].max,
                        d.channels[0]
                            .pct_over_2
                            .max(d.channels[1].pct_over_2)
                            .max(d.channels[2].pct_over_2)
                    ),
                    (None, Some((cat, _))) => cat.clone(),
                    (None, None) => row.merged.as_str().to_string(),
                };
                println!(
                    "  {:<15} {} [{}] {}",
                    row.outcome,
                    f.rel,
                    row.features.tags().join(","),
                    detail
                );
                rows.insert(sha, row);
            }
            Err(e) => failures.push(format!("{}: {e}", f.rel)),
        }
    }
    assert!(
        failures.is_empty(),
        "{} corpus file(s) failed before an outcome could be assigned:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );

    let text = ledger_json(&rows);
    let fresh = Json::parse(&text).expect("generated ledger parses");
    if let Some(old) = committed_ledger() {
        let (was, now) = (
            outcome_count(&old, "compared"),
            outcome_count(&fresh, "compared"),
        );
        assert!(
            now >= was
                || std::env::var_os("PAGED_PSD_LEDGER").is_some_and(|v| v == "accept-refusals"),
            "RATCHET: {now} file(s) compared, the committed ledger records {was}"
        );
        if let Some(old_rows) = old.get("rows").and_then(Json::as_object) {
            let lost: Vec<&str> = old_rows
                .iter()
                .filter(|(_, r)| r.get("outcome").and_then(Json::as_str) == Some("compared"))
                .filter(|(sha, _)| rows.get(sha.as_str()).map(|r| r.outcome) != Some("compared"))
                .map(|(sha, _)| &sha[..12])
                .collect();
            // A DELIBERATE new refusal (the import learned it was
            // approximating) is accepted only when asked for, and only
            // when every lost file is now refused with a reason.
            let accept = std::env::var_os("PAGED_PSD_LEDGER")
                .is_some_and(|v| v == "accept-refusals")
                && lost.iter().all(|short| {
                    rows.iter()
                        .any(|(sha, r)| sha.starts_with(short) && r.outcome == "refused")
                });
            assert!(
                lost.is_empty() || accept,
                "RATCHET: file(s) the ledger records as compared no longer are: {lost:?} \
                 (if they are now refused on purpose, re-run with \
                 PAGED_PSD_LEDGER=accept-refusals)"
            );
        }
    }
    std::fs::create_dir_all(ledger_path().parent().expect("parent")).expect("mkdir");
    std::fs::write(ledger_path(), &text).expect("write ledger");
    println!(
        "psd composite oracle: compared {}, refused {}, no-merged-data {} -> {}",
        outcome_count(&fresh, "compared"),
        outcome_count(&fresh, "refused"),
        outcome_count(&fresh, "no-merged-data"),
        ledger_path().display()
    );
}

// ─────────────────────── always on: the committed ledger ──────────────

#[test]
#[allow(non_snake_case)]
fn composite_ledger_is_well_formed_and_private__feat__image_psd_layer_import() {
    let Some(ledger) = committed_ledger() else {
        panic!("{} is missing", ledger_path().display());
    };
    let rows = ledger
        .get("rows")
        .and_then(Json::as_object)
        .expect("rows object");
    assert_eq!(
        ledger.get("files").and_then(Json::as_u64),
        Some(rows.len() as u64),
        "the file count disagrees with the rows"
    );
    let mut tally: BTreeMap<String, u64> = BTreeMap::new();
    for (sha, row) in rows {
        assert!(
            sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()),
            "row key {sha:?} is not a SHA-256 — the ledger must not name corpus files"
        );
        let outcome = row.get("outcome").and_then(Json::as_str).expect("outcome");
        assert!(
            matches!(outcome, "compared" | "refused" | "no-merged-data"),
            "unknown outcome {outcome}"
        );
        assert_eq!(
            outcome == "compared",
            row.get("diff").is_some(),
            "a compared row carries a diff, and only a compared row does"
        );
        assert_eq!(
            outcome == "refused",
            row.get("reason").is_some(),
            "a refused row carries the engine's reason, and only a refused row does"
        );
        *tally.entry(outcome.to_string()).or_default() += 1;
    }
    for o in ["compared", "refused", "no-merged-data"] {
        assert_eq!(
            outcome_count(&ledger, o),
            tally.get(o).copied().unwrap_or(0),
            "summary count for {o} disagrees with the rows"
        );
    }
    let text = std::fs::read_to_string(ledger_path()).expect("read");
    for needle in [".psd", ".psb", ".PSD", "/Users/", "packs/"] {
        assert!(
            !text.contains(needle),
            "the ledger contains {needle:?} — it must carry hashes and aggregates only"
        );
    }
}

#[test]
#[allow(non_snake_case)]
fn composite_ledger_holds_the_compared_floor__feat__image_psd_layer_import() {
    let ledger = committed_ledger().expect("committed ledger");
    let compared = outcome_count(&ledger, "compared");
    assert!(
        compared >= FLOOR_COMPARED,
        "RATCHET: the committed ledger compares {compared} file(s), below the floor \
         {FLOOR_COMPARED} — a regenerated ledger may only improve"
    );
}
