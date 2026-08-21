/*
 * This file is part of paged (https://paged.media), the commercial editor
 * for the paged IDML engine.
 *
 * paged is free software: you may redistribute it and/or modify it under the
 * terms of the GNU Affero General Public License, version 3, as published by
 * the Free Software Foundation, OR under the Paged Media Enterprise License
 * (PMEL), a commercial license available from And The Next GmbH. Full
 * copyright and license information is available in LICENSE.md, distributed
 * with this source code.
 *
 *  @copyright  Copyright (c) And The Next GmbH
 *  @license    AGPL-3.0-only OR Paged Media Enterprise License (PMEL)
 */

//! Real Photoshop files — the ones a designer saved.
//!
//! Every PSD this crate tests is built by `psd_builder` (11 fixtures,
//! ~2,200 lines of emitter). That is the right default: they are precise,
//! diffable, and carry no binary blobs. But they are also all OUR shape,
//! and a synthesised file cannot surprise the parser the way a designer's
//! can — the corpus PSDs carry real layer trees and run to 5,032 × 33,321
//! px (a 168-megapixel annual-report spread), against fixtures that fit
//! in a diff.
//!
//! Until the 2026-08 corpus campaign, plugin-image had ZERO real rasters
//! of any kind. The first pass extracted 21 fixture-sized PSDs under a
//! per-file 25 MB cap and a per-pack count cap. Both caps were lifted on
//! 2026-08-21 when the source archive was extracted in full and deleted,
//! so this lane now walks **157 files, 9.0 GB** — every group's
//! `assets/psd` plus each pack's own `primary.psd`/`.psb`, including the
//! 213 MB `.psb` and the 105-285 MB mockup primaries the caps had held
//! out. The single largest is a 639 MB poster source.
//!
//! SIZE POSTURE, measured rather than assumed: the whole population runs
//! in **3.3 s warm, ~10 s cold**. Both tests below walk the whole list,
//! so a run reads ~18 GB; each file is read WHOLE and handed to
//! `PsdFile::parse`, which walks the header and section table and does
//! not decode pixel data. Peak memory is therefore one file per test
//! thread — bounded by twice the largest, ~1.3 GB — which is why no size
//! threshold is applied here. If this lane ever grows a pixel-decoding
//! assertion, or a third whole-population test, it needs one.
//!
//! `CV.psd` was the file that first mattered here: **CMYK, 4×8-bit**, when
//! the crate's own JPEG test says outright that "we cannot synthesize a
//! CMYK JPEG through the adapter" — so CMYK raster input had never been
//! exercised at all. It is no longer alone: **77 of the 157 are CMYK**,
//! 80 RGB, every one of them 8-bit.
//!
//! OPT-IN — the assets live in the private corpus checkout:
//!
//! ```text
//! PAGED_PSD_CORPUS=1 cargo test -p image-conformance --test real_psd_corpus -- --ignored --nocapture
//! ```

use std::collections::BTreeSet;
use std::path::PathBuf;

use image_psd::{model::ColorMode, PsdFile};

/// Every extracted `assets/psd/*` file, or `None` with a printed reason.
fn corpus_psds() -> Option<Vec<PathBuf>> {
    let Some(switch) = std::env::var_os("PAGED_PSD_CORPUS") else {
        eprintln!(
            "SKIP psd corpus lane: PAGED_PSD_CORPUS unset \
             (set it to 1, or to a corpus root, and run with --ignored)"
        );
        return None;
    };
    let switch = switch.to_string_lossy().into_owned();
    let root = if switch == "1" || switch.is_empty() {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../corpus")
    } else {
        PathBuf::from(switch)
    };
    // Every group's packs, not just idml's. Since 2026-08-20 packs are
    // filed by PRIMARY format, so the 16 Photoshop packs live at
    // `psd/packs/<pack>/primary.psd` — nine of them 105-285 MB mockups
    // with layer trees psd_builder cannot synthesise.
    let mut out = Vec::new();
    let mut any_group = false;
    for group in ["psd", "idml", "docx", "vector", "html", "pptx"] {
        let Ok(entries) = std::fs::read_dir(root.join(group).join("packs")) else {
            continue;
        };
        any_group = true;
        for pack in entries.flatten() {
            let dir = pack.path();
            for cand in ["primary.psd", "primary.psb"] {
                let p = dir.join(cand);
                if p.is_file() {
                    out.push(p);
                }
            }
            let Ok(files) = std::fs::read_dir(dir.join("assets").join("psd")) else {
                continue;
            };
            for f in files.flatten() {
                let p = f.path();
                let is_psd = p.extension().is_some_and(|e| {
                    matches!(e.to_string_lossy().to_lowercase().as_str(), "psd" | "psb")
                });
                if p.is_file() && is_psd {
                    out.push(p);
                }
            }
        }
    }
    if !any_group {
        eprintln!(
            "SKIP psd corpus lane: no <group>/packs under {}",
            root.display()
        );
        return None;
    }
    out.sort();
    if out.is_empty() {
        eprintln!(
            "SKIP psd corpus lane: no PSDs under {} — run corpus/harness/unpack.sh",
            root.display()
        );
        return None;
    }
    Some(out)
}

#[test]
#[ignore = "psd corpus lane: opt-in (PAGED_PSD_CORPUS=1 + the private corpus mount)"]
fn every_real_psd_parses_with_a_sane_header() {
    let Some(files) = corpus_psds() else {
        return;
    };
    println!("psd corpus: {} file(s)", files.len());

    let mut modes: BTreeSet<String> = BTreeSet::new();
    let mut depths: BTreeSet<u16> = BTreeSet::new();
    let mut failures: Vec<String> = Vec::new();

    for path in &files {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let bytes = std::fs::read(path).expect("read corpus psd");
        match PsdFile::parse(&bytes) {
            Ok(psd) => {
                let h = &psd.header;
                // Photoshop's own limits. A parser that mis-reads the
                // header (endianness, offset drift) lands outside these
                // long before it produces a wrong pixel.
                assert!(
                    h.width > 0 && h.height > 0,
                    "{name}: zero-sized canvas {}x{}",
                    h.width,
                    h.height
                );
                assert!(
                    h.width <= 300_000 && h.height <= 300_000,
                    "{name}: implausible canvas {}x{} — header mis-read",
                    h.width,
                    h.height
                );
                assert!(
                    (1..=56).contains(&h.channels),
                    "{name}: {} channels is outside Photoshop's 1..=56",
                    h.channels
                );
                assert!(
                    matches!(h.depth, 1 | 8 | 16 | 32),
                    "{name}: {}-bit depth is not a Photoshop depth",
                    h.depth
                );
                modes.insert(format!("{:?}", h.color_mode));
                depths.insert(h.depth);
                println!(
                    "  ok  {name:<40} {}x{} {:?} {}ch {}bit",
                    h.width, h.height, h.color_mode, h.channels, h.depth
                );
            }
            Err(e) => failures.push(format!("{name}: {e:?}")),
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} real PSDs failed to parse:\n  {}",
        failures.len(),
        files.len(),
        failures.join("\n  ")
    );
    println!("psd corpus: colour modes {modes:?}, depths {depths:?}");
}

#[test]
#[ignore = "psd corpus lane: opt-in (PAGED_PSD_CORPUS=1 + the private corpus mount)"]
fn the_corpus_covers_cmyk_which_no_synthesised_fixture_does() {
    let Some(files) = corpus_psds() else {
        return;
    };
    // The reason this lane exists rather than another psd_builder fixture.
    // If the corpus ever loses its CMYK file, that is a COVERAGE
    // regression and must be loud — CMYK is a print format and this is a
    // print engine.
    let mut cmyk = Vec::new();
    for path in &files {
        let bytes = std::fs::read(path).expect("read corpus psd");
        if let Ok(psd) = PsdFile::parse(&bytes) {
            if matches!(psd.header.color_mode, ColorMode::Cmyk) {
                cmyk.push((path.clone(), psd.header.channels, psd.header.depth));
            }
        }
    }
    assert!(
        !cmyk.is_empty(),
        "no CMYK PSD in the corpus — the only real CMYK raster coverage this \
         project has just disappeared (psd_builder cannot synthesise one, and \
         codec_jpeg.rs documents the same gap for JPEG)"
    );
    for (path, channels, depth) in &cmyk {
        println!(
            "  cmyk {} — {channels} channels, {depth}-bit",
            path.file_name().unwrap_or_default().to_string_lossy()
        );
        assert!(
            *channels >= 4,
            "a CMYK PSD needs at least 4 channels, got {channels}"
        );
    }
}
