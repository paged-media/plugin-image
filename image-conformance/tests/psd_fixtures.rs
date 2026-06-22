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

//! The PSD fixture round-trip suite (spec §10.4 oracle 1) — the M0 exit
//! "image.psd.* preserved across the corpus". For every one of the 11
//! synthesized fixtures this drives four independent checks against the
//! production parser/writer:
//!
//! 1. **Parse** — `PsdFile::parse` succeeds on the builder's bytes.
//! 2. **Manifest** — the parsed model matches the fixture's answer key
//!    (names incl. Unicode, group `lsct` kinds + nesting, blend keys,
//!    opacities, clipping, layer ids, mask presence, unknown addl blocks
//!    surfaced as `Opaque` with matching key + payload length, unknown
//!    resources preserved, container, dims, RLE channels still
//!    `Compression::Rle` with verbatim bytes).
//! 3. **Byte-identity** — `file.write()` reproduces the source bytes
//!    EXACTLY (the strong zero-edit preservation target, §10.4).
//! 4. **Re-encode** — null every lazy-verbatim guard the model carries,
//!    re-emit, REPARSE, and assert the typed model survives unchanged —
//!    proving the canonical writer agrees with the parser independently
//!    of the verbatim fast path.
//!
//! The builder (`psd_builder`) and the parser share NO code, so a shared
//! bug cannot hide: feat image.psd.roundtrip.

use image_conformance::psd_builder::fixtures::{self, ExpectedLayer, FixtureManifest};
use image_psd::container::Container;
use image_psd::model::resources::ResourceBody;
use image_psd::model::{AddlBody, Compression, PsdFile};

// ---------------------------------------------------------------------
// Manifest verification.
// ---------------------------------------------------------------------

/// Walk the flat layer list bottom-first and assign each entry its group
/// nesting depth, matching the manifest convention (brief §7): a kind-3
/// bounding divider OPENS a group from below (it and the members above it
/// sit one level deeper); a kind-1|2 folder CLOSES it from above. Divider
/// and folder markers report the depth of their CONTENTS.
fn derived_depths(file: &PsdFile) -> Vec<u32> {
    let mut depths = Vec::with_capacity(file.layer_mask.layers.len());
    let mut current = 0u32;
    for layer in &file.layer_mask.layers {
        match layer.addl.iter().find_map(|a| a.lsct()).map(|d| d.kind) {
            // Bounding divider: contents live one level deeper; it reports
            // that deeper level, then the group is open below it.
            Some(image_psd::model::SectionKind::BoundingDivider) => {
                current += 1;
                depths.push(current);
            }
            // Folder record: reports its contents' depth, then closes.
            Some(image_psd::model::SectionKind::OpenFolder)
            | Some(image_psd::model::SectionKind::ClosedFolder) => {
                depths.push(current);
                current = current.saturating_sub(1);
            }
            // A plain layer sits at the running depth.
            _ => depths.push(current),
        }
    }
    depths
}

/// The `lsct` kind code carried by a layer record, if any (brief §7).
fn lsct_kind(layer: &image_psd::model::LayerRecord) -> Option<u32> {
    layer
        .addl
        .iter()
        .find_map(|a| a.lsct())
        .map(|d| d.kind.code())
}

/// Assert one parsed layer against its `ExpectedLayer` answer key.
fn check_layer(
    fixture: &str,
    idx: usize,
    got: &image_psd::model::LayerRecord,
    depth: u32,
    want: &ExpectedLayer,
) {
    let ctx = format!("{fixture} layer[{idx}]");
    assert_eq!(got.name_legacy.text_lossy(), want.name, "{ctx} legacy name");
    assert_eq!(
        got.addl.iter().find_map(|a| a.unicode_name()),
        want.unicode_name,
        "{ctx} unicode name"
    );
    assert_eq!(depth, want.depth, "{ctx} group depth");
    assert_eq!(got.blend_key, want.blend_key, "{ctx} blend key");
    assert_eq!(got.opacity, want.opacity, "{ctx} opacity");
    assert_eq!(got.clipping, want.clipping, "{ctx} clipping");
    assert_eq!(
        got.addl.iter().find_map(|a| a.layer_id()),
        want.layer_id,
        "{ctx} layer id"
    );
    assert_eq!(got.mask.is_some(), want.has_mask, "{ctx} mask presence");
    assert_eq!(lsct_kind(got), want.lsct_kind, "{ctx} lsct kind");
}

/// Verify the parsed model recovers every fact the manifest pins.
fn check_manifest(file: &PsdFile, m: &FixtureManifest) {
    assert_eq!(file.container, m.container, "{} container", m.name);
    assert_eq!(file.header.width, m.width, "{} width", m.name);
    assert_eq!(file.header.height, m.height, "{} height", m.name);
    assert_eq!(file.header.channels, m.channels, "{} channels", m.name);
    assert_eq!(
        file.layer_mask.transparency_in_merged, m.transparency_in_merged,
        "{} transparency-in-merged flag",
        m.name
    );

    // The merged composite's compression tag (brief §11).
    assert_eq!(
        file.composite.compression,
        m.composite_compression.code(),
        "{} composite compression",
        m.name
    );

    // Layers: count, then field-by-field with derived depth.
    assert_eq!(
        file.layer_mask.layers.len(),
        m.layers.len(),
        "{} layer count",
        m.name
    );
    let depths = derived_depths(file);
    for (i, (got, want)) in file.layer_mask.layers.iter().zip(&m.layers).enumerate() {
        check_layer(m.name, i, got, depths[i], want);
    }

    // Resources: order + id (the modeled ids get a typed body, the
    // unknown one stays Opaque but is retained — brief §3/§10.4).
    assert_eq!(
        file.resources.blocks.len(),
        m.resources.len(),
        "{} resource count",
        m.name
    );
    for (i, (got, want)) in file.resources.blocks.iter().zip(&m.resources).enumerate() {
        assert_eq!(got.id, want.id, "{} resource[{i}] id", m.name);
    }

    // Unknown additional-layer-info blocks: surfaced as Opaque, matched by
    // (sig, key, stored payload length). The stored length excludes the
    // even pad, so re-derive it from raw_block (block - header - pad).
    let mut opaque_addl: Vec<(_, _, usize)> = Vec::new();
    for layer in &file.layer_mask.layers {
        for a in &layer.addl {
            if matches!(a.body, AddlBody::Opaque) {
                opaque_addl.push((a.sig, a.key, addl_stored_len(file.container, a)));
            }
        }
    }
    for a in &file.layer_mask.addl_global {
        if matches!(a.body, AddlBody::Opaque) {
            opaque_addl.push((a.sig, a.key, addl_stored_len(file.container, a)));
        }
    }
    assert_eq!(
        opaque_addl.len(),
        m.unknown_addl.len(),
        "{} unknown addl count",
        m.name
    );
    for (i, want) in m.unknown_addl.iter().enumerate() {
        let got = &opaque_addl[i];
        assert_eq!(got.0, want.sig, "{} unknown addl[{i}] sig", m.name);
        assert_eq!(got.1, want.key, "{} unknown addl[{i}] key", m.name);
        assert_eq!(got.2, want.len, "{} unknown addl[{i}] payload len", m.name);
    }
}

/// Read an addl block's STORED data length straight from the on-disk
/// length field captured in `raw_block` (the field excludes padding, so
/// it equals the manifest's recorded length). Layout: sig(4) + key(4) +
/// length(4|8, wide only for the PSB keys the container marks).
fn addl_stored_len(container: Container, a: &image_psd::model::AdditionalLayerInfo) -> usize {
    let raw = a.raw_block.as_ref().expect("parsed block keeps raw");
    let width = if container.addl_len_is_wide(a.key) {
        8
    } else {
        4
    };
    read_be(&raw[8..8 + width])
}

/// Decode a big-endian length (4 or 8 bytes) as usize.
fn read_be(b: &[u8]) -> usize {
    let mut v = 0u64;
    for &byte in b {
        v = (v << 8) | byte as u64;
    }
    v as usize
}

// ---------------------------------------------------------------------
// Re-encode lane: normalize away the lazy-verbatim guards so the
// canonical writer path is exercised, then prove it agrees with parse.
// ---------------------------------------------------------------------

/// Null every lazy-verbatim guard that has a canonical typed re-encode.
/// `raw_block` on an `Opaque`-bodied block is NOT nulled: an opaque block
/// has no typed body, so its raw bytes ARE its semantic content (the
/// model's own contract — `emit` for a constructed Opaque block writes an
/// empty payload). Nulling those would destroy preserved data, which is
/// the opposite of what this lane proves.
fn strip_verbatim(file: &mut PsdFile) {
    file.layer_mask.section_raw = None;
    for layer in &mut file.layer_mask.layers {
        layer.extra_raw = None;
        for a in &mut layer.addl {
            if !matches!(a.body, AddlBody::Opaque) {
                a.raw_block = None;
            }
        }
    }
    for a in &mut file.layer_mask.addl_global {
        if !matches!(a.body, AddlBody::Opaque) {
            a.raw_block = None;
        }
    }
    for b in &mut file.resources.blocks {
        if !matches!(b.body, ResourceBody::Opaque) {
            b.raw_block = None;
        }
    }
}

/// A semantic comparison view: both models are stripped of EVERY Raw
/// guard (including Opaque raw_blocks) so equality ignores byte-level
/// padding and verbatim spans, comparing only the typed structure.
fn normalize(file: &PsdFile) -> PsdFile {
    let mut f = file.clone();
    f.layer_mask.section_raw = None;
    for layer in &mut f.layer_mask.layers {
        layer.extra_raw = None;
        for a in &mut layer.addl {
            a.raw_block = None;
        }
    }
    for a in &mut f.layer_mask.addl_global {
        a.raw_block = None;
    }
    for b in &mut f.resources.blocks {
        b.raw_block = None;
    }
    f
}

// ---------------------------------------------------------------------
// The corpus-wide driver.
// ---------------------------------------------------------------------

/// Run all four checks for one fixture.
fn run_fixture(bytes: Vec<u8>, m: FixtureManifest) {
    // (1) Parse.
    let file = PsdFile::parse(&bytes).unwrap_or_else(|e| panic!("{} parse failed: {e}", m.name));

    // (2) Manifest expectations.
    check_manifest(&file, &m);

    // RLE channels survive as Compression::Rle with verbatim bytes — the
    // parser never decodes them (preservation + streaming budget). Spot
    // this on any fixture declaring an RLE composite or RLE channels.
    for layer in &file.layer_mask.layers {
        for cd in &layer.channel_data {
            if cd.compression == Compression::Rle {
                assert!(!cd.bytes.is_empty(), "{} RLE channel kept empty", m.name);
            }
        }
    }

    // (3) Byte-identity: zero-edit re-emit reproduces the source exactly.
    let reemit = file
        .write()
        .unwrap_or_else(|e| panic!("{} write failed: {e}", m.name));
    assert_eq!(
        reemit,
        bytes,
        "{} zero-edit re-emit is not byte-identical (len {} vs {})",
        m.name,
        reemit.len(),
        bytes.len()
    );

    // (4) Re-encode lane: strip the verbatim guards, re-emit through the
    // CANONICAL writer, reparse, and compare the typed model. This proves
    // the writer's re-encode path agrees with the parser independently of
    // the byte-identity fast path.
    let mut stripped = file.clone();
    strip_verbatim(&mut stripped);
    let canonical = stripped
        .write()
        .unwrap_or_else(|e| panic!("{} canonical write failed: {e}", m.name));
    let reparsed = PsdFile::parse(&canonical)
        .unwrap_or_else(|e| panic!("{} canonical reparse failed: {e}", m.name));
    assert_eq!(
        normalize(&reparsed),
        normalize(&file),
        "{} canonical re-encode diverges from parse",
        m.name
    );
}

/// Every fixture in the corpus passes all four round-trip checks.
#[test]
fn image_psd_roundtrip_all_fixtures() {
    for (bytes, m) in fixtures::all() {
        run_fixture(bytes, m);
    }
}

// Per-fixture entry points so a failure names the offending fixture and
// each block-handler registry row has its own green test pointer.

#[test]
fn image_psd_roundtrip_rgb8_flat() {
    let (b, m) = fixtures::rgb8_flat();
    run_fixture(b, m);
}

#[test]
fn image_psd_roundtrip_rgb8_flat_rle() {
    let (b, m) = fixtures::rgb8_flat_rle();
    run_fixture(b, m);
}

#[test]
fn image_psd_roundtrip_multilayer_groups() {
    let (b, m) = fixtures::multilayer_groups();
    run_fixture(b, m);
}

#[test]
fn image_psd_roundtrip_unicode_names() {
    let (b, m) = fixtures::unicode_names();
    run_fixture(b, m);
}

#[test]
fn image_psd_roundtrip_layer_ids() {
    let (b, m) = fixtures::layer_ids();
    run_fixture(b, m);
}

#[test]
fn image_psd_roundtrip_blend_opacity() {
    let (b, m) = fixtures::blend_opacity();
    run_fixture(b, m);
}

#[test]
fn image_psd_roundtrip_raster_masks() {
    let (b, m) = fixtures::raster_masks();
    run_fixture(b, m);
}

#[test]
fn image_psd_roundtrip_rle_and_raw_mix() {
    let (b, m) = fixtures::rle_and_raw_mix();
    run_fixture(b, m);
}

#[test]
fn image_psd_roundtrip_unknown_addl() {
    let (b, m) = fixtures::unknown_addl();
    run_fixture(b, m);
}

#[test]
fn image_psd_roundtrip_unknown_resource() {
    let (b, m) = fixtures::unknown_resource();
    run_fixture(b, m);
}

#[test]
fn image_psd_roundtrip_psb_wide() {
    let (b, m) = fixtures::psb_wide();
    run_fixture(b, m);
}
