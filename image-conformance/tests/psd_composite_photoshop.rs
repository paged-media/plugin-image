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

//! PHOTOSHOP-WRITTEN LAYERED PSDs, replayed in CI. `layer-stacks.jsx`
//! had Photoshop build small RGB documents (transparency, opacity and
//! blend stacks, clipping, a layer mask, groups) and save each as a PSD
//! with REAL merged data, plus a PNG of the same document as the
//! straight-alpha colour reference. These are public fixtures, so the
//! private corpus oracle's findings can be pinned where CI sees them.
//!
//! For each PSD: resource 0x0421 must vouch for the merged composite;
//! the layers go through the shipping import (`layer_plates_rgba8` →
//! `LayerStack::from_psd_plates`) and the GPU fold; the result is
//! compared with Photoshop's merged composite. [`EXPECT`] holds the
//! outcome and the measured distance of every case — an improvement
//! passes, a regression fails, and a changed refusal is a diff to
//! review. Defects are pinned by the `defect_*` tests, `#[ignore]`d so
//! CI stays green and failing until fixed.

use std::path::{Path, PathBuf};

use image_conformance::psd_corpus::{classify, compare_rgba8, merged_data, MergedData, Reference};
use image_js::layers::LayerStack;
use image_psd::PsdFile;
use zune_core::bytestream::ZCursor;
use zune_png::PngDecoder;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/photoshop/layer-stacks")
}

/// What a case is expected to do: `Ok(max_levels)` = compared, at most
/// that far from Photoshop's merged composite (8-bit levels, RGB over
/// white); `Err(category)` = refused for that reason.
///
/// Recorded 2026-10 against Photoshop 27.10:
/// * `transparent-soft` and `opacity-blend-stack` AGREE (the merged
///   composite compared through its white matte);
/// * `clip-base-100` / `clip-base-50` are the CLIPPING-GROUP defect:
///   clipped layers blend against everything below instead of against
///   the clip base alone, so the base's anti-aliased edge (27 levels) and
///   its opacity (64 levels) leak the clipped blend onto the backdrop;
///   pinned by `defect_clipping_groups_match_photoshop` below and, without
///   a PSD, in `composite_defects.rs`;
/// * groups and layer masks are REFUSED by the import, by design today.
const EXPECT: &[(&str, Result<u8, &str>)] = &[
    ("clip-base-100", Ok(27)),
    ("clip-base-50", Ok(64)),
    ("group-isolated-50", Err("groups")),
    ("group-pass-through", Err("groups")),
    ("layer-mask", Err("layer-mask")),
    ("opacity-blend-stack", Ok(1)),
    ("transparent-soft", Ok(0)),
];

fn psd(id: &str) -> PsdFile {
    let bytes = std::fs::read(dir().join(format!("{id}.psd"))).expect("read fixture PSD");
    PsdFile::parse(&bytes).expect("parse")
}

/// Photoshop's PNG export of the document: straight RGBA8.
fn png_rgba8(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).expect("read PNG");
    let mut dec = PngDecoder::new(ZCursor::new(&bytes));
    let px = dec.decode_raw().expect("decode PNG");
    let ch = dec.colorspace().expect("colorspace").num_components();
    match ch {
        4 => px,
        3 => px
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        n => panic!("{n}-channel PNG"),
    }
}

/// Our flatten, or the engine's refusal reduced to a category.
fn flatten(file: &PsdFile) -> Result<Vec<u8>, String> {
    let import = file.layer_plates_rgba8().map_err(|e| {
        let s = e.to_string();
        if s.contains("GROUPED") {
            "groups".into()
        } else if s.contains("LAYER MASK") {
            "layer-mask".into()
        } else {
            s
        }
    })?;
    let ctx = image_conformance::device::test_device().ok_or("no-gpu")?;
    let stack = LayerStack::from_psd_plates(&import).map_err(|e| e.to_string())?;
    pollster::block_on(stack.composite(Some(ctx), None))
        .map(|p| p.to_vec())
        .map_err(|e| e.to_string())
}

fn ids() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir())
        .expect("fixtures dir")
        .flatten()
        .filter_map(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.strip_suffix(".psd").map(str::to_string)
        })
        .collect();
    v.sort();
    v
}

#[test]
#[allow(non_snake_case)]
fn photoshop_layer_stacks_flatten_as_recorded__feat__image_psd_layer_import() {
    let have_gpu = image_conformance::device::test_device().is_some();
    let mut report = Vec::new();
    let mut problems = Vec::new();
    for id in ids() {
        let file = psd(&id);
        assert_eq!(
            merged_data(&file),
            MergedData::Real,
            "{id}: Photoshop wrote it with maximize compatibility, so 0x0421 must vouch"
        );
        let theirs = file.composite_rgba8().expect("merged composite decodes");
        let reference = if file.layer_mask.transparency_in_merged {
            Reference::MattedWhite
        } else {
            Reference::Straight
        };
        let got: Result<u8, String> = match flatten(&file) {
            Ok(ours) => Ok(compare_rgba8(&ours, &theirs.rgba, reference).channels[..3]
                .iter()
                .map(|c| c.max)
                .max()
                .unwrap_or(0)),
            Err(e) if e == "no-gpu" => {
                eprintln!("SKIP {id}: no GPU adapter");
                continue;
            }
            Err(e) => Err(e),
        };
        report.push(format!(
            "{id:<22} [{}] {got:?}",
            classify(&file).tags().join(",")
        ));
        match EXPECT.iter().find(|(e, _)| *e == id) {
            None => problems.push(format!("{id}: no expectation recorded ({got:?})")),
            Some((_, want)) => match (want, &got) {
                (Ok(w), Ok(g)) if g <= w => {}
                (Err(w), Err(g)) if g == w => {}
                _ => problems.push(format!("{id}: expected {want:?}, got {got:?}")),
            },
        }
    }
    println!("{}", report.join("\n"));
    if have_gpu {
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }
}

/// MERGED-COMPOSITE MATTE. A transparent document's merged composite
/// stores its colour MATTED against white (`c·α + 255·(1 − α)`) — the
/// corpus shows it on every transparent file and Photoshop's own PNG
/// export of the same document, which is straight, confirms it. The
/// decoder (`PsdFile::composite_rgba8`, the flattened-ingest path)
/// returns those matted values as STRAIGHT colour, so every semi-
/// transparent edge of a placed PSD comes in lightened toward white
/// (a halo over anything dark). Up to ~255 levels as α → 0; ~9 levels at
/// α = 225, measured on the corpus.
#[test]
#[allow(non_snake_case)]
#[ignore = "DEFECT: composite_rgba8 does not un-matte the white-matted merged colour of a transparent PSD"]
fn defect_merged_composite_of_a_transparent_psd_decodes_to_straight_colour__feat__image_psd_rendered(
) {
    let file = psd("transparent-soft");
    assert!(
        file.layer_mask.transparency_in_merged,
        "fixture is transparent"
    );
    let ours = file.composite_rgba8().expect("decode").rgba;
    let theirs = png_rgba8(&dir().join("transparent-soft.png"));
    let mut worst = (0u8, 0u8);
    for (a, b) in ours.chunks_exact(4).zip(theirs.chunks_exact(4)) {
        if b[3] < 16 {
            continue; // colour is ill-conditioned this close to clear
        }
        let d = (0..3).map(|c| a[c].abs_diff(b[c])).max().unwrap_or(0);
        if d > worst.0 {
            worst = (d, b[3]);
        }
    }
    assert!(
        worst.0 <= 2,
        "merged colour is {} levels from Photoshop's straight export (at alpha {})",
        worst.0,
        worst.1
    );
}

/// The same file with the matte undone agrees — which is what proves the
/// matte is the whole story, and keeps a fix honest.
#[test]
#[allow(non_snake_case)]
fn unmatting_the_merged_composite_recovers_photoshops_colour__feat__image_psd_rendered() {
    let file = psd("transparent-soft");
    let ours = file.composite_rgba8().expect("decode").rgba;
    let theirs = png_rgba8(&dir().join("transparent-soft.png"));
    for (a, b) in ours.chunks_exact(4).zip(theirs.chunks_exact(4)) {
        assert_eq!(a[3], b[3], "alpha decodes exactly");
        if b[3] < 64 {
            continue;
        }
        let al = b[3] as f64 / 255.0;
        for c in 0..3 {
            let un = ((a[c] as f64 - 255.0 * (1.0 - al)) / al).clamp(0.0, 255.0);
            assert!(
                (un - b[c] as f64).abs() <= 3.0,
                "un-matted {un:.1} vs Photoshop {} at alpha {}",
                b[c],
                b[3]
            );
        }
    }
}

#[test]
#[allow(non_snake_case)]
#[ignore = "DEFECT: clipped layers blend against everything below, not against the clip base alone (27 levels at the base's edge, 64 at 50 % base opacity)"]
fn defect_clipping_groups_match_photoshop__feat__image_layers_clipping() {
    for id in ["clip-base-100", "clip-base-50"] {
        let file = psd(id);
        let theirs = file.composite_rgba8().expect("merged").rgba;
        let Ok(ours) = flatten(&file) else {
            eprintln!("SKIP {id}: no GPU adapter or refused");
            return;
        };
        let d = compare_rgba8(&ours, &theirs, Reference::Straight);
        let max = d.channels[..3].iter().map(|c| c.max).max().unwrap_or(0);
        assert!(
            max <= 2,
            "{id}: {max} levels from Photoshop's merged composite"
        );
    }
}
