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

//! Round-trip suite for the image-psd structural parser + preservation
//! writer (spec §10.4). Fixtures are hand-built byte vectors IN this file
//! — no dependency on image-conformance (that crate dev-depends on this
//! one; consuming its builder would be a cycle).
//!
//! Three lanes per the constitution:
//!   1. parse → write byte-identity (the zero-edit verbatim guard);
//!   2. model field assertions (the typed views are correct);
//!   3. re-encode lane: clear every `Raw` to `None`, write, REPARSE, and
//!      assert semantic equality — proving canonical re-encoding is
//!      structurally faithful even without the verbatim shortcut.

use image_psd::container::Container;
use image_psd::model::{AddlBody, ColorMode, GlobalLayerMask, PsdFile, ResourceBody, SectionKind};

// ---------------------------------------------------------------------------
// Byte-vector builders (big-endian, spec §10.4 layout)
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

/// One `8BIM` image-resource block with the canonical padding: empty name
/// → 2-byte name field; data padded to even (size excludes the pad byte).
fn resource_block(id: u16, name: &[u8], data: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"8BIM");
    b.extend_from_slice(&be16(id));
    // Pascal name: length byte + content, padded so the field is even.
    b.push(name.len() as u8);
    b.extend_from_slice(name);
    if !(1 + name.len()).is_multiple_of(2) {
        b.push(0);
    }
    b.extend_from_slice(&be32(data.len() as u32));
    b.extend_from_slice(data);
    if !data.len().is_multiple_of(2) {
        b.push(0);
    }
    b
}

/// The image-resources section: u32 total length + concatenated blocks.
fn resources_section(blocks: &[Vec<u8>]) -> Vec<u8> {
    let body: Vec<u8> = blocks.iter().flatten().copied().collect();
    let mut s = be32(body.len() as u32).to_vec();
    s.extend_from_slice(&body);
    s
}

/// An additional-layer-info block, padded to `align` (2 inside a record, 4
/// at document level). The stored length excludes the pad.
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

/// One layer record (no mask, empty blend ranges). `channels` is a list of
/// (id, channel_data_payload_excluding_tag, compression).
fn layer_record(
    rect: (i32, i32, i32, i32),
    channels: &[(i16, Vec<u8>, u16)],
    blend_key: &[u8; 4],
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
        // data length INCLUDES the 2-byte compression tag.
        rec.extend_from_slice(&be32((payload.len() + 2) as u32));
    }
    rec.extend_from_slice(b"8BIM");
    rec.extend_from_slice(blend_key);
    rec.push(255); // opacity
    rec.push(0); // clipping
    rec.push(0); // flags
    rec.push(0); // filler

    // Extra data: mask(0) + blend ranges(0) + name(pad4) + addl blocks.
    let mut extra = Vec::new();
    extra.extend_from_slice(&be32(0)); // no layer mask
    extra.extend_from_slice(&be32(0)); // empty blending ranges
                                       // Legacy name, padded to a multiple of 4 INCLUDING the length byte.
    extra.push(name.len() as u8);
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

/// Channel image data for one channel: compression tag + payload.
fn channel_data(comp: u16, payload: &[u8]) -> Vec<u8> {
    let mut c = be16(comp).to_vec();
    c.extend_from_slice(payload);
    c
}

/// Assemble the full file.
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

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Minimal flat RGB PSD: no layers, one resolution resource, RAW composite.
fn flat_psd() -> Vec<u8> {
    // ResolutionInfo (id 1005, 16 bytes): 72 ppi h/v.
    let mut res_data = Vec::new();
    res_data.extend_from_slice(&be32(72 << 16)); // h res 16.16
    res_data.extend_from_slice(&be16(1)); // ppi
    res_data.extend_from_slice(&be16(1)); // inches
    res_data.extend_from_slice(&be32(72 << 16)); // v res
    res_data.extend_from_slice(&be16(1));
    res_data.extend_from_slice(&be16(1));
    let resources = resources_section(&[resource_block(1005, b"", &res_data)]);

    // No layers, no global mask: layer & mask info total length 0.
    let layer_mask = be32(0).to_vec();

    // 2x2 RGB RAW composite: 3 planar channels × 4 bytes.
    let comp_payload: Vec<u8> = (0..12u8).collect();
    let composite = channel_data(0, &comp_payload);

    assemble(3, 2, 2, resources, layer_mask, composite)
}

/// Layered RGB PSD: one folder group around one pixel layer, with luni /
/// lyid / lsct addl blocks. Exercises the flat group encoding (brief §7).
fn layered_psd() -> Vec<u8> {
    let resources = resources_section(&[]);

    // Bottom-most first (PSD order): divider record, then the content
    // layer, then the open-folder record above it.
    // For a 1x1 layer each channel payload is 1 RAW byte.
    let one = vec![0xABu8];

    // Content layer "Leaf": channels R/G/B/alpha, with luni + lyid.
    let luni = {
        // u32 count + UTF-16BE, no trailing null (one variant).
        let units: Vec<u16> = "Leaf".encode_utf16().collect();
        let mut d = be32(units.len() as u32).to_vec();
        for u in units {
            d.extend_from_slice(&u.to_be_bytes());
        }
        addl_block(b"luni", &d, 2)
    };
    let lyid = addl_block(b"lyid", &be32(42), 2);
    let leaf = layer_record(
        (0, 0, 1, 1),
        &[
            (0, one.clone(), 0),
            (1, one.clone(), 0),
            (2, one.clone(), 0),
            (-1, one.clone(), 0),
        ],
        b"norm",
        b"Leaf",
        &[luni, lyid],
    );

    // Divider record (kind 3, '</Layer group>'), below the members.
    let lsct_div = {
        let mut d = be32(SectionKind::BoundingDivider.code()).to_vec();
        // length 4 only (just the kind word).
        d.truncate(4);
        addl_block(b"lsct", &d, 2)
    };
    let divider = layer_record((0, 0, 0, 0), &[], b"norm", b"</Layer group>", &[lsct_div]);

    // Folder record (kind 1, open folder), above the members.
    let lsct_open = addl_block(b"lsct", &be32(SectionKind::OpenFolder.code()), 2);
    let folder = layer_record((0, 0, 0, 0), &[], b"norm", b"Group 1", &[lsct_open]);

    // PSD stores bottom-most first: divider, leaf, folder.
    let records: Vec<u8> = [divider, leaf.clone(), folder].concat();

    // Channel image data follows all records, in layer/channel order. Only
    // the leaf has channels (divider/folder have zero).
    let mut chan_data = Vec::new();
    for _ in 0..4 {
        chan_data.extend_from_slice(&channel_data(0, &one));
    }

    // Layer info content: i16 count (3) + records + channel data.
    let mut layer_info_content = (3i16).to_be_bytes().to_vec();
    layer_info_content.extend_from_slice(&records);
    layer_info_content.extend_from_slice(&chan_data);
    // Pad content to even.
    if !layer_info_content.len().is_multiple_of(2) {
        layer_info_content.push(0);
    }

    // Layer info section = u32 length + content.
    let mut layer_info = be32(layer_info_content.len() as u32).to_vec();
    layer_info.extend_from_slice(&layer_info_content);
    // Global layer mask info: zero length.
    layer_info.extend_from_slice(&be32(0));

    // Layer & mask info section = u32 total length + (layer info + gmask).
    let mut layer_mask = be32(layer_info.len() as u32).to_vec();
    layer_mask.extend_from_slice(&layer_info);

    // 1x1 RGB RAW composite: 3 channels × 1 byte.
    let composite = channel_data(0, &[1, 2, 3]);

    assemble(4, 1, 1, resources, layer_mask, composite)
}

// ---------------------------------------------------------------------------
// Lane 1: parse → write byte-identity
// ---------------------------------------------------------------------------

#[test]
fn image_psd_roundtrip_flat_byte_identical() {
    let bytes = flat_psd();
    let file = PsdFile::parse(&bytes).unwrap();
    let out = file.write().unwrap();
    assert_eq!(
        out, bytes,
        "zero-edit flat round-trip must be byte-identical"
    );
}

#[test]
fn image_psd_roundtrip_layered_byte_identical() {
    let bytes = layered_psd();
    let file = PsdFile::parse(&bytes).unwrap();
    let out = file.write().unwrap();
    assert_eq!(
        out, bytes,
        "zero-edit layered round-trip must be byte-identical"
    );
}

// ---------------------------------------------------------------------------
// Lane 2: model field assertions
// ---------------------------------------------------------------------------

#[test]
fn image_psd_roundtrip_flat_model_fields() {
    let file = PsdFile::parse(&flat_psd()).unwrap();
    assert_eq!(file.container, Container::Psd);
    assert_eq!(file.header.channels, 3);
    assert_eq!(file.header.width, 2);
    assert_eq!(file.header.height, 2);
    assert_eq!(file.header.depth, 8);
    assert_eq!(file.header.color_mode, ColorMode::Rgb);
    assert!(file.color_mode.raw.is_empty());
    // Typed resolution view.
    let res = file.resources.resolution().expect("1005 resolution view");
    assert_eq!(res.h_ppi(), 72.0);
    assert_eq!(res.v_ppi(), 72.0);
    assert!(file.layer_mask.layers.is_empty());
    assert_eq!(file.composite.compression, 0);
    assert_eq!(file.composite.raw.len(), 12);
}

#[test]
fn image_psd_roundtrip_layered_model_fields() {
    let file = PsdFile::parse(&layered_psd()).unwrap();
    assert_eq!(file.layer_mask.layers.len(), 3);

    // Flat order: divider (bottom), leaf, folder (top).
    let divider = &file.layer_mask.layers[0];
    let leaf = &file.layer_mask.layers[1];
    let folder = &file.layer_mask.layers[2];

    // Divider lsct kind 3.
    assert_eq!(
        divider.addl[0].lsct().unwrap().kind,
        SectionKind::BoundingDivider
    );
    // Folder lsct kind 1.
    assert_eq!(folder.addl[0].lsct().unwrap().kind, SectionKind::OpenFolder);

    // Leaf: 4 channels, luni name "Leaf", lyid 42.
    assert_eq!(leaf.channels.len(), 4);
    assert_eq!(leaf.channels[3].id, -1); // transparency alpha
    assert_eq!(leaf.channel_data.len(), 4);
    assert_eq!(leaf.name(), "Leaf");
    let lyid = leaf.addl.iter().find_map(|a| a.layer_id());
    assert_eq!(lyid, Some(42));
    // luni view present.
    assert!(leaf
        .addl
        .iter()
        .any(|a| matches!(&a.body, AddlBody::UnicodeName(s) if s == "Leaf")));

    // Global mask present and empty.
    assert_eq!(
        file.layer_mask.global_mask,
        Some(GlobalLayerMask::default())
    );
}

// ---------------------------------------------------------------------------
// Lane 3: re-encode (clear all Raw → None, write, REPARSE, semantic-equal)
// ---------------------------------------------------------------------------

/// Strip every verbatim guard so the writer takes the canonical re-encode
/// path everywhere, then prove the result re-parses to the same model.
fn clear_raw(file: &mut PsdFile) {
    file.resources
        .blocks
        .iter_mut()
        .for_each(|b| b.raw_block = None);
    for layer in &mut file.layer_mask.layers {
        layer.extra_raw = None;
        for a in &mut layer.addl {
            a.raw_block = None;
        }
    }
    for a in &mut file.layer_mask.addl_global {
        a.raw_block = None;
    }
    file.layer_mask.section_raw = None;
}

fn assert_semantic_eq(a: &PsdFile, b: &PsdFile) {
    assert_eq!(a.container, b.container);
    assert_eq!(a.header, b.header);
    assert_eq!(a.color_mode, b.color_mode);
    // Resource bodies + ids + names (raw_block may legitimately differ).
    assert_eq!(a.resources.blocks.len(), b.resources.blocks.len());
    for (x, y) in a.resources.blocks.iter().zip(&b.resources.blocks) {
        assert_eq!(x.id, y.id);
        assert_eq!(x.name, y.name);
        assert_eq!(x.body, y.body);
    }
    assert_eq!(a.layer_mask.layers.len(), b.layer_mask.layers.len());
    for (x, y) in a.layer_mask.layers.iter().zip(&b.layer_mask.layers) {
        assert_eq!(
            (x.top, x.left, x.bottom, x.right),
            (y.top, y.left, y.bottom, y.right)
        );
        assert_eq!(x.channels, y.channels);
        assert_eq!(x.blend_key, y.blend_key);
        assert_eq!(x.opacity, y.opacity);
        assert_eq!(x.name_legacy, y.name_legacy);
        assert_eq!(x.channel_data, y.channel_data);
        assert_eq!(x.addl.len(), y.addl.len());
        for (p, q) in x.addl.iter().zip(&y.addl) {
            assert_eq!(p.key, q.key);
            assert_eq!(p.body, q.body);
        }
    }
    assert_eq!(
        a.layer_mask.transparency_in_merged,
        b.layer_mask.transparency_in_merged
    );
    assert_eq!(a.composite, b.composite);
}

#[test]
fn image_psd_roundtrip_flat_reencode_semantic() {
    let mut file = PsdFile::parse(&flat_psd()).unwrap();
    let original = file.clone();
    clear_raw(&mut file);
    let out = file.write().unwrap();
    let reparsed = PsdFile::parse(&out).unwrap();
    assert_semantic_eq(&original, &reparsed);
}

#[test]
fn image_psd_roundtrip_layered_reencode_semantic() {
    let mut file = PsdFile::parse(&layered_psd()).unwrap();
    let original = file.clone();
    clear_raw(&mut file);
    let out = file.write().unwrap();
    let reparsed = PsdFile::parse(&out).unwrap();
    assert_semantic_eq(&original, &reparsed);
}

#[test]
fn image_psd_roundtrip_unknown_resource_opaque() {
    // An unmodeled resource id stays Opaque but round-trips byte-identical.
    let blk = resource_block(2999, b"meta", &[0xDE, 0xAD, 0xBE, 0xEF, 0x01]);
    let resources = resources_section(&[blk]);
    let layer_mask = be32(0).to_vec();
    let composite = channel_data(0, &(0..3u8).collect::<Vec<_>>());
    let bytes = assemble(1, 1, 3, resources, layer_mask, composite);

    let file = PsdFile::parse(&bytes).unwrap();
    assert!(matches!(
        file.resources.blocks[0].body,
        ResourceBody::Opaque
    ));
    assert_eq!(file.resources.blocks[0].id, 2999);
    assert_eq!(file.resources.blocks[0].name.text_lossy(), "meta");
    assert_eq!(file.write().unwrap(), bytes);
}
