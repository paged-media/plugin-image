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

//! `image.psd.layer.pixel-import` — the PRODUCTION layer-pixel decode
//! (`image_psd::layer_pixels`), checked against the two things that can
//! keep it honest:
//!
//! 1. **The M1 flatten oracle.** Folding the imported plates with the
//!    SAME scalar `compose_ref` spine the flatten reference uses must
//!    reproduce `flatten_reference` exactly — i.e. the import hands the
//!    layer stack the identical stimulus the render oracle already
//!    validates against `psd_composite`. If the import mis-decoded a
//!    channel, mis-placed a rect, or dropped alpha, this diverges.
//! 2. **The refusals.** Opening a PSD as layers REPLACES Photoshop's own
//!    composite with ours, so every structure the model does not
//!    reproduce (groups, clipping, masks, non-8-bit-RGB, no layers, an
//!    over-budget canvas) must decline with a stated reason rather than
//!    quietly produce a different-looking file.

use image_conformance::compose_ref::{self, Blend};
use image_conformance::psd_builder::fixtures;
use image_conformance::psd_render::flatten_reference;
use image_conformance::Px;
use image_psd::model::PsdFile;

fn parse(bytes: &[u8]) -> PsdFile {
    PsdFile::parse(bytes).expect("fixture parses")
}

/// Straight RGBA8 → premultiplied `Px` (the working-space convention the
/// flatten reference composites in).
fn premultiplied(rgba: &[u8]) -> Vec<Px> {
    rgba.chunks_exact(4)
        .map(|t| {
            let a = t[3] as f32 / 255.0;
            Px([
                (t[0] as f32 / 255.0) * a,
                (t[1] as f32 / 255.0) * a,
                (t[2] as f32 / 255.0) * a,
                a,
            ])
        })
        .collect()
}

#[test]
fn image_psd_layer_pixel_import_folds_to_the_flatten_reference() {
    // Three layers, RAW + RLE + mixed-per-channel compression — so a
    // decode slip in either lane shows up as a pixel difference.
    let (bytes, m) = fixtures::rle_and_raw_mix();
    let file = parse(&bytes);
    let import = file
        .layer_plates_rgba8()
        .expect("flat, unclipped, unmasked");
    assert_eq!((import.width, import.height), (m.width, m.height));
    assert_eq!(import.layers.len(), 3);

    let n = (m.width * m.height) as usize;
    let mut canvas = vec![Px([0.0; 4]); n];
    for plate in &import.layers {
        assert_eq!(plate.rgba.len(), n * 4, "plates are canvas-extent");
        let src = premultiplied(&plate.rgba);
        let blend = Blend::from_psd_key(std::str::from_utf8(&plate.blend_key).unwrap_or("norm"))
            .unwrap_or(Blend::Normal);
        let opacity = plate.opacity as f32 / 255.0;
        for (bd, &s) in canvas.iter_mut().zip(src.iter()) {
            *bd = compose_ref::composite(*bd, s, opacity, blend);
        }
    }

    let golden = flatten_reference(&file);
    assert_eq!(canvas.len(), golden.len());
    for (i, (got, want)) in canvas.iter().zip(golden.iter()).enumerate() {
        for c in 0..4 {
            assert!(
                (got.0[c] - want.0[c]).abs() < 1e-6,
                "texel {i} channel {c}: import fold {} vs flatten oracle {}",
                got.0[c],
                want.0[c]
            );
        }
    }
}

#[test]
fn image_psd_layer_pixel_import_carries_the_record_properties() {
    let (bytes, _m) = fixtures::layer_ids();
    let import = parse(&bytes)
        .layer_plates_rgba8()
        .expect("flat, unclipped, unmasked");
    let names: Vec<&str> = import.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, vec!["one", "two"], "BOTTOM-first, as stored");
    for l in &import.layers {
        assert_eq!(&l.blend_key, b"norm");
        assert_eq!(l.opacity, 255);
        assert!(!l.hidden);
    }
    // The fixture paints solid [1,1,1] / [2,2,2] with no alpha channel,
    // so a plate is opaque inside its (full-canvas) rect.
    assert_eq!(&import.layers[0].rgba[..4], &[1, 1, 1, 255]);
    assert_eq!(&import.layers[1].rgba[..4], &[2, 2, 2, 255]);
}

/// GROUPS ARE NO LONGER A REFUSAL — they are imported. The stack gained
/// groups (pass-through and isolated, nested) before the import did; the
/// refusal outlived the reason for it, as the clipping one below had.
/// Each member names its group, and each group is read from its folder
/// record (the bounding divider below the members opens it).
#[test]
fn image_psd_layer_pixel_import_carries_groups_across() {
    let (bytes, _m) = fixtures::multilayer_groups();
    let import = parse(&bytes)
        .layer_plates_rgba8()
        .expect("a grouped PSD imports now");
    assert!(!import.groups.is_empty(), "the fixture's groups arrived");
    assert!(
        import.layers.iter().any(|l| l.group.is_some()),
        "members name their group"
    );
    for g in &import.groups {
        if let Some(p) = g.parent {
            assert!(p < import.groups.len(), "a parent is a group of the file");
        }
    }
}

/// CLIPPING IS NO LONGER A REFUSAL — it is imported.
///
/// This test used to assert the opposite, and it was right to: the layer
/// model had no clipping, so accepting a clipped PSD would have
/// composited it wrong. The model gained clipping on 2026-08-06, so the
/// refusal became a lie about our own capability and the plate now
/// carries the record's `clipping` byte across.
#[test]
fn image_psd_layer_pixel_import_carries_clipping_across() {
    let (bytes, _m) = fixtures::blend_opacity();
    let import = parse(&bytes)
        .layer_plates_rgba8()
        .expect("a clipped PSD imports now");
    assert!(
        import.layers.iter().any(|l| l.clipped),
        "the fixture's clipped layer arrived marked as clipped"
    );
}

/// MASKS ARE NO LONGER A REFUSAL — a user mask (channel −2) imports as
/// canvas-extent coverage: the record's default colour outside the mask
/// rectangle, the decoded mask inside it.
#[test]
fn image_psd_layer_pixel_import_carries_masks_across() {
    let (bytes, _m) = fixtures::raster_masks();
    let import = parse(&bytes)
        .layer_plates_rgba8()
        .expect("a masked PSD imports now");
    let masked: Vec<_> = import
        .layers
        .iter()
        .filter_map(|l| l.mask.as_ref())
        .collect();
    assert!(!masked.is_empty(), "the fixture's mask arrived");
    let n = (import.width * import.height) as usize;
    for m in masked {
        assert_eq!(m.coverage.len(), n, "canvas extent, one byte per pixel");
    }
}

#[test]
fn image_psd_layer_pixel_import_declines_a_file_with_no_layer_records() {
    let (bytes, _m) = fixtures::rgb8_flat();
    let err = parse(&bytes)
        .layer_plates_rgba8()
        .expect_err("nothing to import");
    assert!(err.to_string().contains("no layer records"), "{err}");
}

#[test]
fn image_psd_layer_pixel_import_declines_over_the_memory_budget() {
    // Plates are canvas-extent, so the budget is the honest guard on a
    // many-layer big-canvas file. Fake the extent by editing the parsed
    // header: the gate must fire BEFORE any channel is decoded.
    let (bytes, _m) = fixtures::rle_and_raw_mix();
    let mut file = parse(&bytes);
    file.header.width = 8000;
    file.header.height = 8000;
    let err = file.layer_plates_rgba8().expect_err("over budget");
    let msg = err.to_string();
    assert!(msg.contains("over the"), "{msg}");
    assert!(msg.contains("MiB budget"), "{msg}");
    assert!(
        msg.contains("merged composite is kept"),
        "the refusal says what happens instead: {msg}"
    );
}
