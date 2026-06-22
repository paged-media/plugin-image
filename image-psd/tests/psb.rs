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

//! PSB (large-document) fixtures. PSB widens specific length fields to 8
//! bytes (Container::section_len_width / addl_len_is_wide): the layer &
//! mask info section length, the layer info length, the per-channel data
//! length, and the addl-block length for the enumerated wide keys
//! (`Layr`, `Lr16`, …). Narrow keys (`lyid`, `luni`) keep their 4-byte
//! length even in PSB. A hand-built PSB vector pins all four.

use image_psd::container::Container;
use image_psd::model::PsdFile;

fn be16(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}
fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}
fn be64(v: u64) -> [u8; 8] {
    v.to_be_bytes()
}

fn psb_header(channels: u16, h: u32, w: u32) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"8BPS");
    out.extend_from_slice(&be16(2)); // PSB
    out.extend_from_slice(&[0u8; 6]);
    out.extend_from_slice(&be16(channels));
    out.extend_from_slice(&be32(h));
    out.extend_from_slice(&be32(w));
    out.extend_from_slice(&be16(8));
    out.extend_from_slice(&be16(3)); // RGB
    out
}

/// Narrow-key addl (4-byte length), even-padded (inside a layer record).
fn addl_narrow(key: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"8BIM");
    b.extend_from_slice(key);
    b.extend_from_slice(&be32(data.len() as u32));
    b.extend_from_slice(data);
    if !data.len().is_multiple_of(2) {
        b.push(0);
    }
    b
}

/// Wide-key addl (8-byte length in PSB), even-padded.
fn addl_wide(key: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"8BIM");
    b.extend_from_slice(key);
    b.extend_from_slice(&be64(data.len() as u64));
    b.extend_from_slice(data);
    if !data.len().is_multiple_of(2) {
        b.push(0);
    }
    b
}

fn psb_layer_record(name: &[u8], channels: &[(i16, usize)], addl: &[Vec<u8>]) -> Vec<u8> {
    let mut rec = Vec::new();
    rec.extend_from_slice(&0i32.to_be_bytes());
    rec.extend_from_slice(&0i32.to_be_bytes());
    rec.extend_from_slice(&1i32.to_be_bytes());
    rec.extend_from_slice(&1i32.to_be_bytes());
    rec.extend_from_slice(&be16(channels.len() as u16));
    for (id, payload_len) in channels {
        rec.extend_from_slice(&id.to_be_bytes());
        // PSB per-channel data length is u64; INCLUDES the 2-byte tag.
        rec.extend_from_slice(&be64((payload_len + 2) as u64));
    }
    rec.extend_from_slice(b"8BIM");
    rec.extend_from_slice(b"norm");
    rec.push(255);
    rec.push(0);
    rec.push(0);
    rec.push(0);

    let mut extra = Vec::new();
    extra.extend_from_slice(&be32(0)); // no mask
    extra.extend_from_slice(&be32(0)); // empty blend ranges
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

fn psb_fixture() -> Vec<u8> {
    // One layer, one channel (composite id 0), RAW 1-byte payload.
    let lyid = addl_narrow(b"lyid", &be32(7)); // narrow key keeps u32 length
    let layr = addl_wide(b"Layr", &[0xCA, 0xFE, 0xBA, 0xBE]); // wide → u64 length
    let rec = psb_layer_record(b"L", &[(0, 1)], &[lyid, layr]);

    // Channel data: RAW tag + 1 byte.
    let mut chan = be16(0).to_vec();
    chan.push(0x42);

    // Layer info content: i16 count(1) + record + channel data.
    let mut content = (1i16).to_be_bytes().to_vec();
    content.extend_from_slice(&rec);
    content.extend_from_slice(&chan);
    if !content.len().is_multiple_of(2) {
        content.push(0);
    }
    // Layer info length is u64 in PSB.
    let mut layer_info = be64(content.len() as u64).to_vec();
    layer_info.extend_from_slice(&content);
    // Global layer mask info: u32 length 0 (NOT widened).
    layer_info.extend_from_slice(&be32(0));

    // Layer & mask info section length is u64 in PSB.
    let mut layer_mask = be64(layer_info.len() as u64).to_vec();
    layer_mask.extend_from_slice(&layer_info);

    let mut f = psb_header(1, 1, 1);
    f.extend_from_slice(&be32(0)); // color mode
    f.extend_from_slice(&be32(0)); // resources (empty)
    f.extend_from_slice(&layer_mask);
    f.extend_from_slice(&be16(0)); // composite compression
    f.push(0x99); // 1x1 grayscale-as-rgb composite single channel byte
    f
}

#[test]
fn image_psd_psb_widened_fields_roundtrip() {
    let bytes = psb_fixture();
    let file = PsdFile::parse(&bytes).unwrap();
    assert_eq!(file.container, Container::Psb);
    assert_eq!(file.layer_mask.layers.len(), 1);

    let layer = &file.layer_mask.layers[0];
    // Per-channel data length carried the u64 width.
    assert_eq!(layer.channels.len(), 1);
    assert_eq!(layer.channels[0].data_len, 3); // tag(2) + 1
    assert_eq!(layer.channel_data.len(), 1);
    assert_eq!(layer.channel_data[0].bytes, vec![0x42]);

    // Narrow key parsed (lyid → u32 length).
    assert_eq!(layer.addl.iter().find_map(|a| a.layer_id()), Some(7));
    // Wide key present (Layr → u64 length), preserved opaquely.
    assert!(layer.addl.iter().any(|a| &a.key == b"Layr"));
    assert_eq!(layer.name_legacy.text_lossy(), "L");

    // Zero-edit verbatim round-trip is byte-identical.
    assert_eq!(file.write().unwrap(), bytes);
}

#[test]
fn image_psd_psb_reencode_semantic() {
    // Clear the structural framing so the writer re-derives the u64 widths;
    // typed addl (lyid) re-encodes from its body, the opaque wide-key block
    // re-emits from its retained verbatim span.
    let bytes = psb_fixture();
    let original = PsdFile::parse(&bytes).unwrap();

    let mut re = original.clone();
    re.layer_mask.section_raw = None;
    for l in &mut re.layer_mask.layers {
        l.extra_raw = None;
        for a in &mut l.addl {
            // Keep opaque blocks' raw; clear only typed ones (lyid).
            if a.layer_id().is_some() {
                a.raw_block = None;
            }
        }
    }
    let out = re.write().unwrap();
    let reparsed = PsdFile::parse(&out).unwrap();

    assert_eq!(reparsed.container, Container::Psb);
    let a = &original.layer_mask.layers[0];
    let b = &reparsed.layer_mask.layers[0];
    assert_eq!(a.channels, b.channels);
    assert_eq!(a.channel_data, b.channel_data);
    assert_eq!(a.name_legacy, b.name_legacy);
    assert_eq!(a.addl.iter().find_map(|x| x.layer_id()), Some(7));
    assert_eq!(b.addl.iter().find_map(|x| x.layer_id()), Some(7));
    assert!(b.addl.iter().any(|x| &x.key == b"Layr"));
    assert_eq!(original.composite, reparsed.composite);
}
