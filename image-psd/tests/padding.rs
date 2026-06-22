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

//! Padding-rule fixtures (brief §3/§5/§6). These pin the three distinct
//! alignment boundaries PSD mixes:
//!   - image resources: name field padded to EVEN, data padded to EVEN
//!     (the size field excludes the pad byte);
//!   - layer-record legacy name: padded to a multiple of 4 INCLUDING the
//!     length byte;
//!   - additional-layer-info: EVEN inside a record, multiple-of-4 at the
//!     document level (stored length excludes the pad).
//!
//! Each case verifies the re-encode path (Raw cleared) reproduces the
//! canonical padding exactly — the verbatim path is covered in roundtrip.

use image_psd::model::PsdFile;

fn be16(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}
fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

fn header(channels: u16, h: u32, w: u32) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"8BPS");
    out.extend_from_slice(&be16(1));
    out.extend_from_slice(&[0u8; 6]);
    out.extend_from_slice(&be16(channels));
    out.extend_from_slice(&be32(h));
    out.extend_from_slice(&be32(w));
    out.extend_from_slice(&be16(8));
    out.extend_from_slice(&be16(3));
    out
}

fn resource_block(id: u16, name: &[u8], data: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"8BIM");
    b.extend_from_slice(&be16(id));
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

fn resources_section(blocks: &[Vec<u8>]) -> Vec<u8> {
    let body: Vec<u8> = blocks.iter().flatten().copied().collect();
    let mut s = be32(body.len() as u32).to_vec();
    s.extend_from_slice(&body);
    s
}

fn assemble(channels: u16, h: u32, w: u32, resources: Vec<u8>, composite_raw: &[u8]) -> Vec<u8> {
    let mut f = header(channels, h, w);
    f.extend_from_slice(&be32(0)); // color mode
    f.extend_from_slice(&resources);
    f.extend_from_slice(&be32(0)); // empty layer & mask info
    f.extend_from_slice(&be16(0)); // composite compression RAW
    f.extend_from_slice(composite_raw);
    f
}

/// Clear the resource verbatim guards so the writer must re-derive the pad.
fn clear_resource_raw(file: &mut PsdFile) {
    file.resources
        .blocks
        .iter_mut()
        .for_each(|b| b.raw_block = None);
}

#[test]
fn image_psd_padding_resource_even_data() {
    // 5-byte ICC data (odd) ⇒ one pad byte; size field stays 5.
    let icc = vec![0x10, 0x20, 0x30, 0x40, 0x50];
    let bytes = assemble(
        1,
        1,
        1,
        resources_section(&[resource_block(1039, b"", &icc)]),
        &[0],
    );
    // Verbatim round-trip.
    let file = PsdFile::parse(&bytes).unwrap();
    assert_eq!(file.write().unwrap(), bytes);
    assert_eq!(file.resources.icc_profile(), Some(icc.as_slice()));

    // Re-encode must reproduce the same padded layout byte-for-byte.
    let mut re = file.clone();
    clear_resource_raw(&mut re);
    assert_eq!(re.write().unwrap(), bytes);
}

#[test]
fn image_psd_padding_resource_even_name() {
    // 4-char name "icc!" ⇒ name field = 1 + 4 = 5 (odd) ⇒ one pad byte so
    // the field is even. Even-length data needs no pad.
    let bytes = assemble(
        1,
        1,
        1,
        resources_section(&[resource_block(1039, b"icc!", &[0xAA, 0xBB])]),
        &[0],
    );
    let file = PsdFile::parse(&bytes).unwrap();
    assert_eq!(file.resources.blocks[0].name.text_lossy(), "icc!");
    assert_eq!(file.write().unwrap(), bytes);

    let mut re = file.clone();
    clear_resource_raw(&mut re);
    assert_eq!(re.write().unwrap(), bytes);
}

#[test]
fn image_psd_padding_resource_empty_name_two_bytes() {
    // Empty name occupies exactly 2 bytes (length 0 + 1 pad).
    let blk = resource_block(1005, b"", &[0u8; 16]);
    // signature(4)+id(2)+name(2)+size(4)+data(16) = 28
    assert_eq!(blk.len(), 28);
    let bytes = assemble(1, 1, 1, resources_section(&[blk]), &[0]);
    let file = PsdFile::parse(&bytes).unwrap();
    assert!(file.resources.blocks[0].name.text_lossy().is_empty());
    assert_eq!(file.write().unwrap(), bytes);
}

#[test]
fn image_psd_padding_layer_name_pad4_and_addl_even() {
    // A single layer whose legacy name forces pad-4 and whose addl block
    // forces even-pad inside the extra-data span.
    // Name "ab" ⇒ field 1+2 = 3 ⇒ pad to 4 (one pad byte).
    // lyid data is 4 bytes (even already); use an odd-length opaque addl
    // ('shmd'-style) to force the even pad.
    let mut extra = Vec::new();
    extra.extend_from_slice(&be32(0)); // no mask
    extra.extend_from_slice(&be32(0)); // empty blend ranges
    extra.push(2); // name length
    extra.extend_from_slice(b"ab");
    extra.push(0); // pad to multiple of 4 (field was 3)
                   // Opaque addl with 3-byte data ⇒ even pad adds 1 byte.
    extra.extend_from_slice(b"8BIM");
    extra.extend_from_slice(b"xxYY");
    extra.extend_from_slice(&be32(3));
    extra.extend_from_slice(&[1, 2, 3]);
    extra.push(0); // even pad

    let mut rec = Vec::new();
    rec.extend_from_slice(&0i32.to_be_bytes()); // top
    rec.extend_from_slice(&0i32.to_be_bytes()); // left
    rec.extend_from_slice(&1i32.to_be_bytes()); // bottom
    rec.extend_from_slice(&1i32.to_be_bytes()); // right
    rec.extend_from_slice(&be16(1)); // one channel
    rec.extend_from_slice(&0i16.to_be_bytes()); // channel id 0
    rec.extend_from_slice(&be32(3)); // channel data length: tag(2) + 1 byte
    rec.extend_from_slice(b"8BIM");
    rec.extend_from_slice(b"norm");
    rec.push(255);
    rec.push(0);
    rec.push(0);
    rec.push(0);
    rec.extend_from_slice(&be32(extra.len() as u32));
    rec.extend_from_slice(&extra);

    // Channel data for the one channel: RAW tag + 1 byte.
    let mut chan = be16(0).to_vec();
    chan.push(0x99);

    let mut layer_info_content = (1i16).to_be_bytes().to_vec();
    layer_info_content.extend_from_slice(&rec);
    layer_info_content.extend_from_slice(&chan);
    if !layer_info_content.len().is_multiple_of(2) {
        layer_info_content.push(0);
    }
    let mut layer_info = be32(layer_info_content.len() as u32).to_vec();
    layer_info.extend_from_slice(&layer_info_content);
    layer_info.extend_from_slice(&be32(0)); // global mask

    let mut layer_mask = be32(layer_info.len() as u32).to_vec();
    layer_mask.extend_from_slice(&layer_info);

    let mut f = header(1, 1, 1);
    f.extend_from_slice(&be32(0)); // color mode
    f.extend_from_slice(&resources_section(&[]));
    f.extend_from_slice(&layer_mask);
    f.extend_from_slice(&be16(0)); // composite
    f.push(0x77);

    let file = PsdFile::parse(&f).unwrap();
    assert_eq!(file.layer_mask.layers.len(), 1);
    assert_eq!(file.layer_mask.layers[0].name_legacy.text_lossy(), "ab");
    assert_eq!(file.layer_mask.layers[0].addl.len(), 1);
    // Verbatim path.
    assert_eq!(file.write().unwrap(), f);

    // Re-encode path: clear the structural framing (extra_raw, section_raw)
    // so the writer re-derives the extra-data span and its pad-4 name pad.
    // The Opaque addl block keeps its `raw_block` — opaque-verbatim (§10.4
    // preservation strategy 2) is its ONLY representation; that retained
    // span already carries the canonical even pad.
    let mut re = file.clone();
    re.layer_mask.section_raw = None;
    for l in &mut re.layer_mask.layers {
        l.extra_raw = None;
    }
    assert_eq!(re.write().unwrap(), f);
}

#[test]
fn image_psd_padding_addl_document_level_pad4() {
    // A document-level addl block ('lnk2'-like, opaque) with 5-byte data
    // ⇒ padded to a multiple of 4 (3 pad bytes), length stored as 5.
    let mut addl = Vec::new();
    addl.extend_from_slice(b"8BIM");
    addl.extend_from_slice(b"Patt");
    addl.extend_from_slice(&be32(5));
    addl.extend_from_slice(&[1, 2, 3, 4, 5]);
    addl.extend_from_slice(&[0, 0, 0]); // pad to multiple of 4

    // Layer info length 0 (no layers), then global mask 0, then the addl.
    let mut section = be32(0).to_vec(); // layer info length 0
    section.extend_from_slice(&be32(0)); // global mask
    section.extend_from_slice(&addl);

    let mut layer_mask = be32(section.len() as u32).to_vec();
    layer_mask.extend_from_slice(&section);

    let mut f = header(1, 1, 1);
    f.extend_from_slice(&be32(0));
    f.extend_from_slice(&resources_section(&[]));
    f.extend_from_slice(&layer_mask);
    f.extend_from_slice(&be16(0));
    f.push(0x55);

    let file = PsdFile::parse(&f).unwrap();
    assert_eq!(file.layer_mask.addl_global.len(), 1);
    assert_eq!(&file.layer_mask.addl_global[0].key, b"Patt");
    assert_eq!(file.write().unwrap(), f);

    // Re-encode path: clear only the section framing. The Opaque 'Patt'
    // block keeps its `raw_block` (opaque-verbatim is its only form); the
    // re-encoder re-derives the section length + layer-info/global-mask
    // framing around it, and the retained block carries its pad-4 tail.
    let mut re = file.clone();
    re.layer_mask.section_raw = None;
    assert_eq!(re.write().unwrap(), f);
}
