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

use image_conformance::psd_corpus::{corpus_root, psds_by_magic, CorpusFile};
use image_psd::{model::ColorMode, PsdFile};
use std::collections::BTreeSet;

/// Every corpus file whose CONTENT is a PSD/PSB (`8BPS` magic), or
/// `None` with a printed reason. Selected by content, never by
/// extension, and named by its path relative to the corpus root.
fn corpus_psds() -> Option<Vec<CorpusFile>> {
    let root = corpus_root("PAGED_PSD_CORPUS")?;
    let out = psds_by_magic(&root);
    if out.is_empty() {
        eprintln!(
            "SKIP psd corpus lane: no PSD content under {} — run corpus/harness/unpack.sh",
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

    for file in &files {
        let name = &file.rel;
        let bytes = std::fs::read(&file.path).expect("read corpus psd");
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
    for file in &files {
        let bytes = std::fs::read(&file.path).expect("read corpus psd");
        if let Ok(psd) = PsdFile::parse(&bytes) {
            if matches!(psd.header.color_mode, ColorMode::Cmyk) {
                cmyk.push((file.rel.clone(), psd.header.channels, psd.header.depth));
            }
        }
    }
    assert!(
        !cmyk.is_empty(),
        "no CMYK PSD in the corpus — the only real CMYK raster coverage this \
         project has just disappeared (psd_builder cannot synthesise one, and \
         codec_jpeg.rs documents the same gap for JPEG)"
    );
    for (rel, channels, depth) in &cmyk {
        println!("  cmyk {rel} — {channels} channels, {depth}-bit");
        assert!(
            *channels >= 4,
            "a CMYK PSD needs at least 4 channels, got {channels}"
        );
    }
}
