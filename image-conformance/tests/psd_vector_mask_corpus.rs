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

//! THE VECTOR-MASK RASTERIZER AGAINST PHOTOSHOP'S OWN RENDERS, over the
//! private corpus. Photoshop caches its render of every enabled vector
//! mask (times the user mask beside it, when there is one) in the
//! layer's mask channel −2, flagged "rendered from other data". That
//! cache is the exact answer to "what does Photoshop draw for these
//! paths", so every one of them is an oracle case for
//! `image_js::psd_vector_mask` — independent of colour mode, which is
//! why this lane reads CMYK files the composite oracle cannot.
//!
//! Skipped: masks with density/feather parameters (the cache holds the
//! paths' render WITHOUT them; Photoshop applies them live) and disabled
//! ones (Photoshop writes no cache). Only aggregates are printed — the
//! corpus is private.
//!
//! ```text
//! PAGED_PSD_CORPUS=1 cargo test --release -p image-conformance \
//!   --test psd_vector_mask_corpus -- --ignored --nocapture
//! ```

use image_conformance::psd_corpus::{corpus_root, psds_by_magic};
use image_js::psd_vector_mask::rasterize;
use image_psd::vector_mask::{FillRule, PathOp};
use image_psd::PsdFile;

#[derive(Default)]
struct Tally {
    masks: u64,
    pixels: u64,
    over: [u64; 4],         // pixels more than 0 / 1 / 2 / 8 levels off
    per_mask_max: [u64; 5], // masks whose max is 0 / 1 / 2 / ≤8 / >8
    worst: u8,
}

impl Tally {
    fn add(&mut self, max: u8, over: [u64; 4], pixels: u64) {
        self.masks += 1;
        self.pixels += pixels;
        for (t, o) in self.over.iter_mut().zip(over) {
            *t += o;
        }
        let bucket = match max {
            0 => 0,
            1 => 1,
            2 => 2,
            3..=8 => 3,
            _ => 4,
        };
        self.per_mask_max[bucket] += 1;
        self.worst = self.worst.max(max);
    }

    fn line(&self, label: &str) -> String {
        let pct = |n: u64| 100.0 * n as f64 / self.pixels.max(1) as f64;
        format!(
            "{label:<28} masks {:>5} | per-mask max 0:{} 1:{} 2:{} ≤8:{} >8:{} | worst {} | \
             pixels >0 {:.4}% >1 {:.4}% >2 {:.4}% >8 {:.5}%",
            self.masks,
            self.per_mask_max[0],
            self.per_mask_max[1],
            self.per_mask_max[2],
            self.per_mask_max[3],
            self.per_mask_max[4],
            self.worst,
            pct(self.over[0]),
            pct(self.over[1]),
            pct(self.over[2]),
            pct(self.over[3]),
        )
    }
}

enum ShapeAlpha {
    Render,
    OpaqueOutside,
    Other,
}

/// Compare a shape layer's stored transparency with our render of its
/// path, over the layer's rectangle.
fn shape_alpha_matches(
    psd: &PsdFile,
    layer: &image_psd::model::LayerRecord,
    vector: &image_psd::VectorMask,
) -> Option<ShapeAlpha> {
    let (w, h) = (psd.header.width, psd.header.height);
    let lw = (layer.right - layer.left).max(0) as u32;
    let lh = (layer.bottom - layer.top).max(0) as u32;
    let ai = layer.channels.iter().position(|c| c.id == -1)?;
    if lw == 0 || lh == 0 {
        return None;
    }
    let alpha = layer.channel_data[ai]
        .decode(psd.container, lh, lw, psd.header.depth)
        .ok()?;
    let ours = rasterize(vector, w, h);
    let (mut n, mut close, mut outside) = (0u64, 0u64, 0u64);
    for y in 0..lh as i64 {
        for x in 0..lw as i64 {
            let (cx, cy) = (i64::from(layer.left) + x, i64::from(layer.top) + y);
            if cx < 0 || cy < 0 || cx >= i64::from(w) || cy >= i64::from(h) {
                continue;
            }
            let a = alpha[(y * i64::from(lw) + x) as usize];
            let m = ours.coverage_at(cx as u32, cy as u32);
            n += 1;
            if a.abs_diff(m) <= 8 {
                close += 1;
            }
            if m == 0 && a == 255 {
                outside += 1;
            }
        }
    }
    Some(if n > 0 && close * 1000 >= n * 995 {
        ShapeAlpha::Render
    } else if outside > 0 {
        ShapeAlpha::OpaqueOutside
    } else {
        ShapeAlpha::Other
    })
}

#[test]
#[allow(non_snake_case)]
#[ignore = "vector-mask corpus oracle: opt-in (PAGED_PSD_CORPUS=1 + the private corpus)"]
fn vector_masks_draw_as_photoshops_cached_renders__feat__image_psd_layer_import() {
    let Some(root) = corpus_root("PAGED_PSD_CORPUS") else {
        return;
    };
    let files = psds_by_magic(&root);
    let mut all = Tally::default();
    let mut with_user = Tally::default();
    let mut curved = Tally::default();
    let mut nonzero = Tally::default();
    let mut multi = Tally::default();
    let mut open = Tally::default();
    let mut ops = Tally::default();
    let (mut skipped_params, mut skipped_disabled, mut no_cache, mut unparsed) = (0, 0, 0, 0);
    // Shape layers: is the stored transparency the path's render?
    let (mut shapes, mut shapes_match, mut shapes_outside) = (0u64, 0u64, 0u64);
    for f in &files {
        let bytes = std::fs::read(&f.path).expect("read corpus file");
        let Ok(psd) = PsdFile::parse(&bytes) else {
            continue;
        };
        let (w, h) = (psd.header.width, psd.header.height);
        for layer in &psd.layer_mask.layers {
            if !layer
                .addl
                .iter()
                .any(|a| &a.key == b"vmsk" || &a.key == b"vsms")
            {
                continue;
            }
            let Ok(masks) = psd.layer_masks(layer, w, h) else {
                unparsed += 1;
                continue;
            };
            let Some(vector) = masks.vector else {
                continue;
            };
            if vector.disabled {
                skipped_disabled += 1;
                continue;
            }
            if layer
                .addl
                .iter()
                .any(|a| &a.key == b"SoCo" || &a.key == b"vscg")
            {
                if let Some(verdict) = shape_alpha_matches(&psd, layer, &vector) {
                    shapes += 1;
                    match verdict {
                        ShapeAlpha::Render => shapes_match += 1,
                        ShapeAlpha::OpaqueOutside => shapes_outside += 1,
                        ShapeAlpha::Other => {}
                    }
                }
            }
            if layer.mask.as_ref().is_some_and(|m| m.flags & 0x10 != 0) {
                skipped_params += 1;
                continue;
            }
            let Some(cache) = masks.cached_render else {
                no_cache += 1;
                continue;
            };
            let mut ours = rasterize(&vector, w, h).data().to_vec();
            if let Some(u) = masks.user.as_ref().filter(|u| u.enabled) {
                for (o, &a) in ours.iter_mut().zip(&u.coverage) {
                    *o = ((u32::from(*o) * u32::from(a) + 127) / 255) as u8;
                }
            }
            let mut over = [0u64; 4];
            let mut max = 0u8;
            for (&a, &b) in ours.iter().zip(&cache.coverage) {
                let d = a.abs_diff(b);
                max = max.max(d);
                for (o, t) in over.iter_mut().zip([0u8, 1, 2, 8]) {
                    if d > t {
                        *o += 1;
                    }
                }
            }
            let px = ours.len() as u64;
            all.add(max, over, px);
            if masks.user.is_some() {
                with_user.add(max, over, px);
            }
            let subs = || vector.components.iter().flat_map(|c| c.subpaths.iter());
            if subs()
                .flat_map(|s| s.knots.iter())
                .any(|k| k.control_in != k.anchor || k.control_out != k.anchor)
            {
                curved.add(max, over, px);
            }
            if vector
                .components
                .iter()
                .any(|c| c.fill_rule == FillRule::NonZero)
            {
                nonzero.add(max, over, px);
            }
            if subs().count() > 1 {
                multi.add(max, over, px);
            }
            if subs().any(|s| !s.closed) {
                open.add(max, over, px);
            }
            if vector
                .components
                .iter()
                .skip(1)
                .any(|c| c.op != PathOp::Combine)
                || vector
                    .components
                    .first()
                    .is_some_and(|c| c.op == PathOp::Subtract)
            {
                ops.add(max, over, px);
            }
        }
    }
    println!("vector-mask corpus oracle over {} PSD(s)", files.len());
    for (label, t) in [
        ("all", &all),
        ("with a user mask", &with_user),
        ("curved", &curved),
        ("non-zero component", &nonzero),
        ("several subpaths", &multi),
        ("open subpaths", &open),
        ("subtract/intersect/exclude", &ops),
    ] {
        println!("{}", t.line(label));
    }
    println!(
        "shape layers (SoCo/vscg): {shapes}; transparency = the path's render (8 levels, \
         99.5 % of the rect) {shapes_match}; opaque outside the path (strokes) {shapes_outside}"
    );
    println!(
        "skipped: {skipped_params} with mask parameters, {skipped_disabled} disabled, \
         {no_cache} without a cached render, {unparsed} not parsed"
    );
    assert!(
        all.masks > 0,
        "no vector mask with a cached render in the corpus"
    );
}
