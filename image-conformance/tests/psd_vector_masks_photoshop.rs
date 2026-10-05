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

//! PHOTOSHOP-WRITTEN VECTOR MASKS, replayed in CI.
//! `scripts/photoshop/probes/vector-masks.jsx` had Photoshop build small
//! RGB documents whose layers carry vector masks — anti-aliased edges,
//! curves, every path operation, the fill rules, a user mask beside the
//! vector mask, disabled masks, shape layers — and save each as a PSD
//! with REAL merged data. Cases the scripting DOM cannot express (the
//! non-zero rule, a hole inside one component, the invert flag) were
//! patched in the saved bytes and then OPENED and SAVED again by
//! Photoshop, so every PSD and composite here is Photoshop's own.
//!
//! Two checks per case:
//!
//! * **the layer import against the merged composite** — the shipping
//!   path (`layer_plates_rgba8` → `LayerStack::from_psd_plates`, the GPU
//!   fold), RGB over white, max 8-bit level; or the refusal category;
//! * **the rasterizer against Photoshop's cached render** — wherever
//!   Photoshop cached its render of an enabled vector mask (times the
//!   user mask) in channel −2, our `psd_vector_mask::rasterize` (times
//!   the same user mask) must reproduce it: this needs no GPU.
//!
//! [`EXPECT`] holds both measured distances; an improvement passes, a
//! regression fails, a changed refusal is a diff to review.

use std::path::PathBuf;

use image_conformance::psd_corpus::{compare_rgba8, merged_data, MergedData, Reference};
use image_js::layers::LayerStack;
use image_js::psd_vector_mask::rasterize;
use image_psd::PsdFile;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/photoshop/vector-masks")
}

/// `(case, composite, raster)`: `composite` is `Ok(max levels)` from
/// Photoshop's merged composite or `Err(refusal category)`; `raster` is
/// the max distance of our rasterization from Photoshop's cached render
/// (`None` where Photoshop cached none: disabled masks, shape layers).
///
/// Recorded 2026-10 against Photoshop 27.10. Straight edges agree to two
/// levels (an anti-aliased diagonal's samples can land one or two either
/// side); CURVES are the measured gap — Photoshop flattens its Beziers
/// into chords we reproduce on average but not chord for chord, so a
/// curved edge pixel can be ~10 levels off (`circle`, `open-curve`). The
/// user mask times the vector mask is rounded to nearest, which is the
/// rounding that leaves only the vector mask's own edge differences.
const EXPECT: &[(&str, Result<u8, &str>, Option<u8>)] = &[
    ("add-subtract-add", Ok(0), Some(0)),
    ("circle", Ok(11), Some(12)),
    ("component-hole", Ok(1), Some(0)),
    ("edge-ramp", Ok(1), Some(0)),
    ("exclude", Ok(0), Some(0)),
    ("first-subtract", Ok(0), Some(0)),
    // Group masks (user or vector) are not modelled.
    ("group-vector-mask", Err("group-mask"), Some(0)),
    ("intersect", Ok(0), Some(0)),
    ("inverted", Ok(1), Some(2)),
    ("open-curve", Ok(7), Some(9)),
    ("partial-layer", Ok(1), Some(2)),
    ("per-sample-intersect", Ok(0), Some(0)),
    ("per-sample-seam", Ok(0), Some(0)),
    ("rect-fractional", Ok(1), Some(0)),
    // Shape layers: the stored pixels already are the shape; the vector
    // mask is not applied again, and Photoshop caches no render.
    ("shape-ellipse", Ok(1), None),
    ("shape-fill", Ok(1), None),
    ("star-even-odd", Ok(2), Some(2)),
    ("star-non-zero", Ok(2), Some(2)),
    ("subtract", Ok(0), Some(0)),
    ("triangle", Ok(2), Some(2)),
    ("user-and-vector", Ok(3), Some(3)),
    ("user-disabled", Ok(3), Some(4)),
    // Disabled: kept, not applied — and Photoshop caches no render.
    ("vector-disabled", Ok(0), None),
    ("vector-disabled-with-user", Ok(1), None),
    // The cache holds the paths WITHOUT the feather (raster 0): Photoshop
    // applies feather and density live, which is not modelled.
    ("vector-feather", Ok(3), Some(0)),
];

fn psd(id: &str) -> PsdFile {
    let bytes = std::fs::read(dir().join(format!("{id}.psd"))).expect("read fixture PSD");
    PsdFile::parse(&bytes).expect("parse")
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

/// The flatten's distance from the merged composite, or the refusal.
fn composite(file: &PsdFile) -> Option<Result<u8, String>> {
    if let Some(b) = file.layer_import_blockers().first() {
        return Some(Err(b.category.to_string()));
    }
    let import = file
        .layer_plates_rgba8()
        .expect("no blockers, so it imports");
    let ctx = image_conformance::device::test_device()?;
    let stack = LayerStack::from_psd_plates(&import).expect("stack");
    let ours = pollster::block_on(stack.composite(Some(ctx), None)).expect("composite");
    let theirs = file.composite_rgba8().expect("merged composite decodes");
    let d = compare_rgba8(&ours, &theirs.rgba, Reference::Straight);
    Some(Ok(d.channels[..3].iter().map(|c| c.max).max().unwrap_or(0)))
}

/// Our rasterization against every cached render in the file.
fn raster(file: &PsdFile) -> Option<u8> {
    let (w, h) = (file.header.width, file.header.height);
    let mut worst: Option<u8> = None;
    for layer in &file.layer_mask.layers {
        let masks = file.layer_masks(layer, w, h).expect("masks decode");
        let (Some(v), Some(cache)) = (masks.vector, masks.cached_render) else {
            continue;
        };
        let mut ours = rasterize(&v, w, h).data().to_vec();
        if let Some(u) = masks.user.as_ref().filter(|u| u.enabled) {
            for (o, &a) in ours.iter_mut().zip(&u.coverage) {
                *o = ((u32::from(*o) * u32::from(a) + 127) / 255) as u8;
            }
        }
        let max = ours
            .iter()
            .zip(&cache.coverage)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        worst = Some(worst.map_or(max, |w| w.max(max)));
    }
    worst
}

#[test]
#[allow(non_snake_case)]
fn photoshop_vector_masks_import_as_recorded__feat__image_psd_layer_import() {
    let mut report = Vec::new();
    let mut problems = Vec::new();
    let mut gpu = true;
    for id in ids() {
        let file = psd(&id);
        assert_eq!(
            merged_data(&file),
            MergedData::Real,
            "{id}: written with maximize compatibility, so 0x0421 must vouch"
        );
        let got_raster = raster(&file);
        let got = composite(&file);
        report.push(format!("{id:<28} composite {got:?} raster {got_raster:?}"));
        let Some((_, want, want_raster)) = EXPECT.iter().find(|(e, _, _)| *e == id) else {
            problems.push(format!("{id}: no expectation recorded"));
            continue;
        };
        match (want_raster, got_raster) {
            (Some(w), Some(g)) if g <= *w => {}
            (None, None) => {}
            _ => problems.push(format!(
                "{id}: raster expected {want_raster:?}, got {got_raster:?}"
            )),
        }
        match (want, got) {
            (_, None) => gpu = false,
            (Ok(w), Some(Ok(g))) if g <= *w => {}
            (Err(w), Some(Err(g))) if g == *w => {}
            (_, Some(g)) => problems.push(format!("{id}: composite expected {want:?}, got {g:?}")),
        }
    }
    println!("{}", report.join("\n"));
    if !gpu {
        eprintln!("SKIP the composite half: no GPU adapter");
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
