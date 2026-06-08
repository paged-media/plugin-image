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

//! Mutatable-tier suite (spec §10.4 "write semantics under editing", §15
//! M2). Each edit op is asserted on three axes:
//!
//!   1. **the edit took** — reparse the written bytes, the touched field
//!      reflects the mutation;
//!   2. **untouched layers are intact** — names / opacities / channel
//!      bytes of every OTHER layer survive unchanged (the dirtied subtree
//!      is the only thing the writer re-encodes);
//!   3. **canonical form is idempotent** — write → reparse → write yields
//!      byte-identical output the second time (the re-encoded subtree is
//!      stable, so a re-edit pipeline never drifts).
//!
//! Fixtures are hand-built byte vectors IN this file (no image-conformance
//! dependency — that crate dev-depends on this one; a cycle).

use image_psd::container::Container;
use image_psd::edit;
use image_psd::model::{AddlBody, Compression, PsdFile, SectionKind};

// ---------------------------------------------------------------------------
// Byte-vector builders (big-endian, spec §10.4 layout) — mirror roundtrip.rs.
// ---------------------------------------------------------------------------

fn be16(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}
fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

/// 26-byte file header.
fn header(channels: u16, height: u32, width: u32) -> Vec<u8> {
    let mut h = Vec::new();
    h.extend_from_slice(b"8BPS");
    h.extend_from_slice(&be16(1)); // PSD
    h.extend_from_slice(&[0u8; 6]); // reserved
    h.extend_from_slice(&be16(channels));
    h.extend_from_slice(&be32(height));
    h.extend_from_slice(&be32(width));
    h.extend_from_slice(&be16(8)); // depth
    h.extend_from_slice(&be16(3)); // RGB
    h
}

/// An additional-layer-info block, padded to `align`. Stored length
/// excludes the pad.
fn addl_block(key: &[u8; 4], data: &[u8], align: usize) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"8BIM");
    b.extend_from_slice(key);
    b.extend_from_slice(&be32(data.len() as u32));
    b.extend_from_slice(data);
    let rem = data.len() % align;
    if rem != 0 {
        b.resize(b.len() + (align - rem), 0);
    }
    b
}

fn luni_block(name: &str) -> Vec<u8> {
    let units: Vec<u16> = name.encode_utf16().collect();
    let mut d = be32(units.len() as u32).to_vec();
    for u in units {
        d.extend_from_slice(&u.to_be_bytes());
    }
    addl_block(b"luni", &d, 2)
}

/// One layer record (no mask, empty blend ranges). `channels` is a list
/// of (id, payload-excluding-tag, compression-code). `opacity` is the
/// record's opacity byte.
fn layer_record(
    rect: (i32, i32, i32, i32),
    channels: &[(i16, Vec<u8>, u16)],
    blend_key: &[u8; 4],
    opacity: u8,
    name: &[u8],
    addl: &[Vec<u8>],
) -> Vec<u8> {
    let mut rec = Vec::new();
    rec.extend_from_slice(&rect.0.to_be_bytes());
    rec.extend_from_slice(&rect.1.to_be_bytes());
    rec.extend_from_slice(&rect.2.to_be_bytes());
    rec.extend_from_slice(&rect.3.to_be_bytes());
    rec.extend_from_slice(&be16(channels.len() as u16));
    for (id, payload, _comp) in channels {
        rec.extend_from_slice(&id.to_be_bytes());
        rec.extend_from_slice(&be32((payload.len() + 2) as u32)); // + 2-byte tag
    }
    rec.extend_from_slice(b"8BIM");
    rec.extend_from_slice(blend_key);
    rec.push(opacity);
    rec.push(0); // clipping
    rec.push(0); // flags
    rec.push(0); // filler

    let mut extra = Vec::new();
    extra.extend_from_slice(&be32(0)); // no layer mask
    extra.extend_from_slice(&be32(0)); // empty blending ranges
    extra.push(name.len() as u8); // legacy name, pad4 incl. length byte
    extra.extend_from_slice(name);
    let field = 1 + name.len();
    let pad = (4 - (field % 4)) % 4;
    extra.resize(extra.len() + pad, 0);
    for a in addl {
        extra.extend_from_slice(a);
    }
    rec.extend_from_slice(&be32(extra.len() as u32));
    rec.extend_from_slice(&extra);
    rec
}

/// Channel image data: compression tag + payload.
fn channel_data(comp: u16, payload: &[u8]) -> Vec<u8> {
    let mut c = be16(comp).to_vec();
    c.extend_from_slice(payload);
    c
}

fn assemble(
    channels: u16,
    height: u32,
    width: u32,
    resources: Vec<u8>,
    layer_mask: Vec<u8>,
    composite: Vec<u8>,
) -> Vec<u8> {
    let mut f = header(channels, height, width);
    f.extend_from_slice(&be32(0)); // color mode data: empty (RGB)
    f.extend_from_slice(&resources);
    f.extend_from_slice(&layer_mask);
    f.extend_from_slice(&composite);
    f
}

/// Assemble a layered file from already-built records + their channel
/// image data, in PSD bottom-first order.
fn assemble_layers(
    records: &[Vec<u8>],
    chan_data: Vec<u8>,
    count: i16,
    composite: Vec<u8>,
) -> Vec<u8> {
    let resources = {
        let mut s = be32(0).to_vec();
        s.truncate(4);
        s
    };

    let mut layer_info_content = count.to_be_bytes().to_vec();
    for r in records {
        layer_info_content.extend_from_slice(r);
    }
    layer_info_content.extend_from_slice(&chan_data);
    if !layer_info_content.len().is_multiple_of(2) {
        layer_info_content.push(0);
    }
    let mut layer_info = be32(layer_info_content.len() as u32).to_vec();
    layer_info.extend_from_slice(&layer_info_content);
    layer_info.extend_from_slice(&be32(0)); // global layer mask: zero length

    let mut layer_mask = be32(layer_info.len() as u32).to_vec();
    layer_mask.extend_from_slice(&layer_info);

    assemble(4, 1, 2, resources, layer_mask, composite)
}

// ---------------------------------------------------------------------------
// Fixture: two 1x2 content pixel layers ("Lower", "Upper"), RAW channels.
// Each layer is R/G/B + alpha; a 1x2 plane is 2 bytes per channel.
// ---------------------------------------------------------------------------

/// Distinct per-channel payloads so untouched-layer assertions are sharp.
fn lower_payloads() -> [Vec<u8>; 4] {
    [
        vec![0x10, 0x11],
        vec![0x12, 0x13],
        vec![0x14, 0x15],
        vec![0xFF, 0xFF],
    ]
}
fn upper_payloads() -> [Vec<u8>; 4] {
    [
        vec![0x20, 0x21],
        vec![0x22, 0x23],
        vec![0x24, 0x25],
        vec![0xFE, 0xFE],
    ]
}

fn channels_from(payloads: &[Vec<u8>; 4]) -> Vec<(i16, Vec<u8>, u16)> {
    vec![
        (0, payloads[0].clone(), 0),
        (1, payloads[1].clone(), 0),
        (2, payloads[2].clone(), 0),
        (-1, payloads[3].clone(), 0),
    ]
}

/// Two pixel layers, no groups. Bottom-first: "Lower" (opacity 128, luni),
/// then "Upper" (opacity 200, luni).
fn two_layer_psd() -> Vec<u8> {
    let lower = layer_record(
        (0, 0, 1, 2),
        &channels_from(&lower_payloads()),
        b"norm",
        128,
        b"Lower",
        &[luni_block("Lower")],
    );
    let upper = layer_record(
        (0, 0, 1, 2),
        &channels_from(&upper_payloads()),
        b"norm",
        200,
        b"Upper",
        &[luni_block("Upper")],
    );

    let mut chan_data = Vec::new();
    for p in lower_payloads() {
        chan_data.extend_from_slice(&channel_data(0, &p));
    }
    for p in upper_payloads() {
        chan_data.extend_from_slice(&channel_data(0, &p));
    }

    let composite = channel_data(0, &[1, 2, 3, 4, 5, 6]); // 1x2 RGB
    assemble_layers(&[lower, upper], chan_data, 2, composite)
}

/// One open-folder group (folder above, bounding divider below) wrapping a
/// single pixel layer. Bottom-first: divider, leaf, folder.
fn grouped_psd() -> Vec<u8> {
    let leaf = layer_record(
        (0, 0, 1, 2),
        &channels_from(&lower_payloads()),
        b"norm",
        255,
        b"Leaf",
        &[luni_block("Leaf")],
    );
    let lsct_div = {
        let d = be32(SectionKind::BoundingDivider.code()).to_vec();
        addl_block(b"lsct", &d, 2)
    };
    let divider = layer_record(
        (0, 0, 0, 0),
        &[],
        b"norm",
        255,
        b"</Layer group>",
        &[lsct_div],
    );
    let lsct_open = addl_block(b"lsct", &be32(SectionKind::OpenFolder.code()), 2);
    let folder = layer_record((0, 0, 0, 0), &[], b"norm", 255, b"Group 1", &[lsct_open]);

    let mut chan_data = Vec::new();
    for p in lower_payloads() {
        chan_data.extend_from_slice(&channel_data(0, &p));
    }

    let composite = channel_data(0, &[1, 2, 3, 4, 5, 6]);
    assemble_layers(&[divider, leaf, folder], chan_data, 3, composite)
}

// ---------------------------------------------------------------------------
// Helpers: the three axes.
// ---------------------------------------------------------------------------

/// Decode every channel of a layer into planar buffers (for unchanged-bytes
/// assertions). 1x2 plane ⇒ rows=1, cols=2.
fn decoded_channels(file: &PsdFile, layer_idx: usize) -> Vec<Vec<u8>> {
    let layer = &file.layer_mask.layers[layer_idx];
    layer
        .channel_data
        .iter()
        .map(|cd| cd.decode(file.container, 1, 2, 8).unwrap())
        .collect()
}

/// Write → reparse → write; assert the second write equals the first
/// (canonical form is idempotent), and return the reparsed model.
fn write_reparse_stable(file: &PsdFile) -> PsdFile {
    let out1 = file.write().unwrap();
    let reparsed = PsdFile::parse(&out1).unwrap();
    let out2 = reparsed.write().unwrap();
    assert_eq!(
        out1, out2,
        "canonical re-encode must be idempotent (out2 == out1)"
    );
    reparsed
}

// ---------------------------------------------------------------------------
// set_layer_opacity
// ---------------------------------------------------------------------------

#[test]
fn image_psd_mutatable_set_opacity() {
    let mut file = PsdFile::parse(&two_layer_psd()).unwrap();

    // Untouched-layer reference (the OTHER layer's name + opacity + pixels).
    let lower_pixels_before = decoded_channels(&file, 0);

    edit::set_layer_opacity(&mut file, 1, 64).unwrap();

    // The edited record dirtied its guards; the untouched record kept them.
    assert!(
        file.layer_mask.layers[1].extra_raw.is_none(),
        "edited record dirtied"
    );
    assert!(file.layer_mask.section_raw.is_none(), "section dirtied");
    assert!(
        file.layer_mask.layers[0].extra_raw.is_some(),
        "untouched record stays verbatim"
    );

    let reparsed = write_reparse_stable(&file);

    // (1) the edit took.
    assert_eq!(reparsed.layer_mask.layers[1].opacity, 64);
    assert_eq!(reparsed.layer_mask.layers[1].name(), "Upper");
    // (2) untouched layer intact: name, opacity, pixels.
    assert_eq!(reparsed.layer_mask.layers[0].name(), "Lower");
    assert_eq!(reparsed.layer_mask.layers[0].opacity, 128);
    assert_eq!(decoded_channels(&reparsed, 0), lower_pixels_before);
}

// ---------------------------------------------------------------------------
// set_layer_name
// ---------------------------------------------------------------------------

#[test]
fn image_psd_mutatable_set_name_existing_luni() {
    let mut file = PsdFile::parse(&two_layer_psd()).unwrap();
    let lower_name_before = file.layer_mask.layers[0].name();
    let lower_pixels_before = decoded_channels(&file, 0);

    edit::set_layer_name(&mut file, 1, "Renamed").unwrap();

    // Both names updated in the live model.
    assert_eq!(
        file.layer_mask.layers[1].name_legacy.text_lossy(),
        "Renamed"
    );
    assert!(file.layer_mask.layers[1]
        .addl
        .iter()
        .any(|a| matches!(&a.body, AddlBody::UnicodeName(s) if s == "Renamed")));

    let reparsed = write_reparse_stable(&file);

    // (1) the edit took — both the luni (canonical) and legacy name.
    assert_eq!(reparsed.layer_mask.layers[1].name(), "Renamed");
    assert_eq!(
        reparsed.layer_mask.layers[1].name_legacy.text_lossy(),
        "Renamed"
    );
    // (2) untouched layer intact.
    assert_eq!(reparsed.layer_mask.layers[0].name(), lower_name_before);
    assert_eq!(decoded_channels(&reparsed, 0), lower_pixels_before);
}

#[test]
fn image_psd_mutatable_set_name_inserts_luni() {
    // A layer with NO luni block: only the legacy name. set_layer_name must
    // insert a luni block so name() (which prefers luni) returns the new name.
    let bare = layer_record(
        (0, 0, 1, 2),
        &channels_from(&lower_payloads()),
        b"norm",
        255,
        b"OldLegacy",
        &[], // no addl blocks
    );
    let mut chan_data = Vec::new();
    for p in lower_payloads() {
        chan_data.extend_from_slice(&channel_data(0, &p));
    }
    let composite = channel_data(0, &[1, 2, 3, 4, 5, 6]);
    let bytes = assemble_layers(&[bare], chan_data, 1, composite);

    let mut file = PsdFile::parse(&bytes).unwrap();
    assert!(file.layer_mask.layers[0].addl.is_empty());
    assert_eq!(file.layer_mask.layers[0].name(), "OldLegacy");

    edit::set_layer_name(&mut file, 0, "Fresh").unwrap();
    let reparsed = write_reparse_stable(&file);

    // (1) luni inserted + canonical name is the new one.
    assert_eq!(reparsed.layer_mask.layers[0].name(), "Fresh");
    assert!(reparsed.layer_mask.layers[0]
        .addl
        .iter()
        .any(|a| matches!(&a.body, AddlBody::UnicodeName(s) if s == "Fresh")));
    assert_eq!(
        reparsed.layer_mask.layers[0].name_legacy.text_lossy(),
        "Fresh"
    );
}

// ---------------------------------------------------------------------------
// replace_channel_pixels
// ---------------------------------------------------------------------------

#[test]
fn image_psd_mutatable_replace_channel_raw() {
    let mut file = PsdFile::parse(&two_layer_psd()).unwrap();
    let lower_pixels_before = decoded_channels(&file, 0);

    let new_plane = vec![0x99u8, 0x88];
    edit::replace_channel_pixels(&mut file, 1, 0, &new_plane, Compression::Raw, 1, 2).unwrap();

    // data_len updated: 2-byte tag + 2-byte RAW plane.
    assert_eq!(file.layer_mask.layers[1].channels[0].data_len, 4);

    let reparsed = write_reparse_stable(&file);

    // (1) the edit took — channel 0 of the upper layer now the new plane.
    assert_eq!(decoded_channels(&reparsed, 1)[0], new_plane);
    // Other channels of the EDITED layer unchanged.
    assert_eq!(decoded_channels(&reparsed, 1)[1], upper_payloads()[1]);
    // (2) untouched layer intact.
    assert_eq!(decoded_channels(&reparsed, 0), lower_pixels_before);
    assert_eq!(reparsed.layer_mask.layers[0].name(), "Lower");
}

#[test]
fn image_psd_mutatable_replace_channel_rle() {
    let mut file = PsdFile::parse(&two_layer_psd()).unwrap();
    let lower_pixels_before = decoded_channels(&file, 0);

    let new_plane = vec![0x77u8, 0x66];
    edit::replace_channel_pixels(&mut file, 1, 2, &new_plane, Compression::Rle, 1, 2).unwrap();

    assert_eq!(reparse_channel_compression(&file, 1, 2), Compression::Rle);

    let reparsed = write_reparse_stable(&file);

    // (1) RLE re-encode decodes back to the new plane.
    assert_eq!(decoded_channels(&reparsed, 1)[2], new_plane);
    assert_eq!(
        reparsed.layer_mask.layers[1].channel_data[2].compression,
        Compression::Rle
    );
    // (2) untouched layer intact.
    assert_eq!(decoded_channels(&reparsed, 0), lower_pixels_before);
}

fn reparse_channel_compression(file: &PsdFile, layer: usize, ch: usize) -> Compression {
    file.layer_mask.layers[layer].channel_data[ch].compression
}

// ---------------------------------------------------------------------------
// remove_layer
// ---------------------------------------------------------------------------

#[test]
fn image_psd_mutatable_remove_plain_layer() {
    let mut file = PsdFile::parse(&two_layer_psd()).unwrap();
    let lower_pixels_before = decoded_channels(&file, 0);

    edit::remove_layer(&mut file, 1).unwrap(); // drop "Upper"
    assert_eq!(file.layer_mask.layers.len(), 1);
    assert!(file.layer_mask.section_raw.is_none());

    let reparsed = write_reparse_stable(&file);

    // (1) the edit took — only "Lower" remains, count word re-framed.
    assert_eq!(reparsed.layer_mask.layers.len(), 1);
    assert_eq!(reparsed.layer_mask.layers[0].name(), "Lower");
    // (2) the surviving layer's pixels + opacity are intact.
    assert_eq!(reparsed.layer_mask.layers[0].opacity, 128);
    assert_eq!(decoded_channels(&reparsed, 0), lower_pixels_before);
}

#[test]
fn image_psd_mutatable_remove_folder_drops_divider() {
    // grouped: [0]=divider(kind3), [1]=leaf, [2]=folder(open). Removing the
    // folder also drops its matching bounding divider — the leaf survives.
    let mut file = PsdFile::parse(&grouped_psd()).unwrap();
    assert_eq!(file.layer_mask.layers.len(), 3);
    let leaf_pixels_before = decoded_channels(&file, 1);

    edit::remove_layer(&mut file, 2).unwrap(); // remove "Group 1" folder

    // Folder + its divider gone; only the leaf remains.
    assert_eq!(file.layer_mask.layers.len(), 1);

    let reparsed = write_reparse_stable(&file);

    // (1) only the leaf survives, no orphan divider.
    assert_eq!(reparsed.layer_mask.layers.len(), 1);
    assert_eq!(reparsed.layer_mask.layers[0].name(), "Leaf");
    let has_divider = reparsed.layer_mask.layers[0]
        .addl
        .iter()
        .filter_map(|a| a.lsct())
        .any(|d| matches!(d.kind, SectionKind::BoundingDivider));
    assert!(
        !has_divider,
        "no dangling bounding divider after folder removal"
    );
    // (2) leaf pixels intact.
    assert_eq!(decoded_channels(&reparsed, 0), leaf_pixels_before);
}

// ---------------------------------------------------------------------------
// Cross-cutting: untouched layer re-emits VERBATIM (not just semantically).
// ---------------------------------------------------------------------------

#[test]
fn image_psd_mutatable_untouched_layer_extra_is_verbatim() {
    // After editing layer 1, layer 0's extra_raw guard is still Some — the
    // writer takes its verbatim path for the untouched record. Compare the
    // captured extra-data bytes before/after the edit.
    let mut file = PsdFile::parse(&two_layer_psd()).unwrap();
    let lower_extra_before = file.layer_mask.layers[0].extra_raw.clone();
    assert!(lower_extra_before.is_some());

    edit::set_layer_opacity(&mut file, 1, 10).unwrap();
    assert_eq!(
        file.layer_mask.layers[0].extra_raw, lower_extra_before,
        "untouched layer extra_raw unchanged (verbatim guard intact)"
    );

    // Sanity: the file still writes + reparses stably and the container held.
    let reparsed = write_reparse_stable(&file);
    assert_eq!(reparsed.container, Container::Psd);
}
