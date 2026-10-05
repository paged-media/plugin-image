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

//! PHOTOSHOP-WRITTEN LAYER EFFECTS, replayed in CI.
//! `scripts/photoshop/probes/layer-effects.jsx` had Photoshop draw small
//! documents with a Color Overlay (each mode the corpus uses, masked, on
//! a clip base, under layer opacity, on soft edges) or a Stroke (off and
//! on) and save each as a PSD with REAL merged data.
//!
//! For each PSD: the effects go through the shipping import (`lfx2` →
//! `image_psd::effects`; an overlay becomes a solid layer clipped to its
//! base in `LayerStack::from_psd_plates`) and the GPU fold; the result is
//! compared with Photoshop's merged composite.
//! [`EXPECT`] holds each case's outcome and measured distance: an
//! improvement passes, a regression fails, a changed refusal is a diff
//! to review.

use std::path::PathBuf;

use image_conformance::psd_corpus::{compare_rgba8, merged_data, MergedData, Reference};
use image_js::layers::LayerStack;
use image_psd::PsdFile;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/photoshop/layer-effects")
}

/// `Ok(max_levels)` = imported and at most that far from Photoshop's
/// merged composite (8-bit levels, any channel); `Err(text)` = refused,
/// the refusal containing `text`.
///
/// Recorded 2026-10-05 against Photoshop 27.10. A Color Overlay is a
/// solid layer clipped to its base, AFTER the base's own clipped layers
/// (Photoshop's overlay covers them: a normal 100 % overlay hides a
/// clipped multiply layer completely — placing it directly above the base
/// was 100 levels off). Every overlay case agrees within 1 level; a
/// switched-off Stroke draws nothing and imports; an enabled one is
/// refused.
const EXPECT: &[(&str, Result<u8, &str>)] = &[
    ("overlay-clip-base", Ok(1)),
    ("overlay-layer-opacity-60", Ok(1)),
    ("overlay-linear-burn-100", Ok(1)),
    ("overlay-masked", Ok(1)),
    ("overlay-multiply-50", Ok(1)),
    ("overlay-normal-100", Ok(1)),
    ("overlay-normal-90", Ok(1)),
    ("overlay-overlay-45", Ok(1)),
    ("overlay-soft-edge", Ok(1)),
    ("stroke-disabled", Ok(1)),
    ("stroke-enabled", Err("layer effects FrFX")),
];

fn flatten(file: &PsdFile) -> Result<Vec<u8>, String> {
    let import = file.layer_plates_rgba8().map_err(|e| e.to_string())?;
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
fn photoshop_layer_effects_composite_as_recorded__feat__image_psd_layer_import() {
    let have_gpu = image_conformance::device::test_device().is_some();
    let mut report = Vec::new();
    let mut problems = Vec::new();
    for id in ids() {
        let bytes = std::fs::read(dir().join(format!("{id}.psd"))).expect("read fixture");
        let file = PsdFile::parse(&bytes).expect("parse");
        assert_eq!(
            merged_data(&file),
            MergedData::Real,
            "{id}: 0x0421 must vouch"
        );
        let theirs = file.composite_rgba8().expect("merged composite decodes");
        let got: Result<u8, String> = match flatten(&file) {
            Ok(ours) => {
                let d = compare_rgba8(&ours, &theirs.rgba, Reference::Straight);
                report.push(format!(
                    "{id:<24} max {:>3} p99 {:>3} mean {:.2}",
                    d.channels[..3].iter().map(|c| c.max).max().unwrap_or(0),
                    d.channels[..3].iter().map(|c| c.p99).max().unwrap_or(0),
                    d.channels[..3].iter().map(|c| c.mean).fold(0.0, f64::max)
                ));
                Ok(d.channels[..3].iter().map(|c| c.max).max().unwrap_or(0))
            }
            Err(e) if e == "no-gpu" => {
                eprintln!("SKIP {id}: no GPU adapter");
                continue;
            }
            Err(e) => {
                report.push(format!("{id:<24} refused: {e}"));
                Err(e)
            }
        };
        match EXPECT.iter().find(|(e, _)| *e == id) {
            None => problems.push(format!("{id}: no expectation recorded ({got:?})")),
            Some((_, want)) => match (want, &got) {
                (Ok(w), Ok(g)) if g <= w => {}
                (Err(w), Err(g)) if g.contains(w) => {}
                _ => problems.push(format!("{id}: expected {want:?}, got {got:?}")),
            },
        }
    }
    println!("{}", report.join("\n"));
    if have_gpu {
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }
}
