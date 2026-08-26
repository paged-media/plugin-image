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

//! The Paged Annual's layered-PSD fixture — `tests/fixtures/annual-layers.psd`.
//!
//! The editor showcase places a byte-identical copy of this file
//! (`editor/apps/canvas/tests/showcase/assets/annual-layers.psd`), so unlike
//! the sibling suites' in-memory byte vectors this fixture is COMMITTED, and
//! this test is its generator, its byte-stability proof, and its rendered-tier
//! contract in one:
//!
//!   1. `build_annual_layers_psd()` assembles the file deterministically with
//!      the same spec-§10.4 byte builders the round-trip suite uses (plus the
//!      crate's own `ChannelData::encode_rle` for the pixel planes);
//!   2. the committed fixture must equal the freshly built bytes (re-running
//!      the builder is byte-stable; regenerating = delete + rerun);
//!   3. the file must parse, its three layers carry their distinct blend
//!      modes/opacities, and BOTH rendered tiers decode: the merged composite
//!      (`composite_rgba8`, the placement lane) and the per-layer plates
//!      (`layer_plates_rgba8`, the layer-stack lane);
//!   4. a zero-edit write round-trips byte-identical (preservation invariant).
//!
//! Content: 400x300 RGB-8. Bottom-first: "Backdrop" (full-canvas vertical
//! blue gradient, `norm`, 255), "Plate" (solid orange block, `mul `, 200),
//! "Signal" (solid paper block, `scrn`, 160). DELIBERATELY MASK-FREE: the
//! record builder below supports the 20-byte mask section (and the parser +
//! preservation writer carry masks fine), but `layer_plates_rgba8` REFUSES a
//! file with a layer mask — "a mask changes what the layer covers, which is
//! not modeled" — and the showcase asset must open in the layer-stack tier,
//! so the fixture stays inside that documented envelope. Self-authored; no
//! third-party bytes.

use std::path::PathBuf;

use image_psd::container::Container;
use image_psd::model::{ChannelData, PsdFile};

// ---------------------------------------------------------------------------
// Byte-vector builders (big-endian, spec §10.4) — the round-trip suite's
// conventions, extended with an opacity byte and an optional 20-byte mask.
// ---------------------------------------------------------------------------

fn be16(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}
fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

fn header(channels: u16, height: u32, width: u32) -> Vec<u8> {
    let mut h = Vec::new();
    h.extend_from_slice(b"8BPS");
    h.extend_from_slice(&be16(1)); // PSD
    h.extend_from_slice(&[0u8; 6]);
    h.extend_from_slice(&be16(channels));
    h.extend_from_slice(&be32(height));
    h.extend_from_slice(&be32(width));
    h.extend_from_slice(&be16(8)); // depth
    h.extend_from_slice(&be16(3)); // RGB
    h
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

/// One layer record. `mask` is the optional 20-byte layer-mask section:
/// (rect TLBR, default_color, flags) + 2 pad bytes under a u32(20) frame.
#[allow(clippy::too_many_arguments)]
fn layer_record(
    rect: (i32, i32, i32, i32),
    channels: &[(i16, Vec<u8>, u16)],
    blend_key: &[u8; 4],
    opacity: u8,
    mask: Option<MaskSpec>,
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
        rec.extend_from_slice(&be32((payload.len() + 2) as u32)); // + tag
    }
    rec.extend_from_slice(b"8BIM");
    rec.extend_from_slice(blend_key);
    rec.push(opacity);
    rec.push(0); // clipping
    rec.push(0); // flags
    rec.push(0); // filler

    let mut extra = Vec::new();
    match mask {
        Some((mrect, default_color, flags)) => {
            extra.extend_from_slice(&be32(20));
            extra.extend_from_slice(&mrect.0.to_be_bytes());
            extra.extend_from_slice(&mrect.1.to_be_bytes());
            extra.extend_from_slice(&mrect.2.to_be_bytes());
            extra.extend_from_slice(&mrect.3.to_be_bytes());
            extra.push(default_color);
            extra.push(flags);
            extra.extend_from_slice(&[0, 0]); // pad to the 20-byte variant
        }
        None => extra.extend_from_slice(&be32(0)),
    }
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

fn channel_data(comp: u16, payload: &[u8]) -> Vec<u8> {
    let mut c = be16(comp).to_vec();
    c.extend_from_slice(payload);
    c
}

// ---------------------------------------------------------------------------
// The fixture
// ---------------------------------------------------------------------------

const W: u32 = 400;
const H: u32 = 300;

/// RLE-encode one planar 8-bit channel via the crate's own encoder; returns
/// the payload EXCLUDING the 2-byte compression tag (count table + rows).
fn rle(planar: &[u8], rows: u32, cols: u32) -> Vec<u8> {
    ChannelData::encode_rle(planar, Container::Psd, rows, cols)
        .expect("RLE encode")
        .bytes
}

/// A solid plane.
fn solid(rows: u32, cols: u32, v: u8) -> Vec<u8> {
    vec![v; (rows * cols) as usize]
}

/// The backdrop's vertical gradient plane: each row constant, ramping
/// `top..=bottom` over the canvas height (deterministic integer math).
fn vgrad(rows: u32, cols: u32, top: u8, bottom: u8) -> Vec<u8> {
    let mut p = Vec::with_capacity((rows * cols) as usize);
    for r in 0..rows {
        let v = top as i32 + (bottom as i32 - top as i32) * r as i32 / (rows as i32 - 1);
        p.extend(std::iter::repeat_n(v as u8, cols as usize));
    }
    p
}

/// (channel id, payload excluding the 2-byte tag, compression code).
type ChannelSpec = (i16, Vec<u8>, u16);

/// An optional 20-byte layer-mask section: (rect TLBR, default color, flags).
type MaskSpec = ((i32, i32, i32, i32), u8, u8);

/// RGBA channel set (ids -1,0,1,2) for a layer whose planes are given
/// per-channel, all RLE.
fn rgba_channels(rows: u32, cols: u32, r: &[u8], g: &[u8], b: &[u8], a: &[u8]) -> Vec<ChannelSpec> {
    vec![
        (0, rle(r, rows, cols), 1),
        (1, rle(g, rows, cols), 1),
        (2, rle(b, rows, cols), 1),
        (-1, rle(a, rows, cols), 1),
    ]
}

/// Assemble `annual-layers.psd`: 400x300 RGB-8, three layers, RLE composite.
fn build_annual_layers_psd() -> Vec<u8> {
    // ResolutionInfo (id 1005): 72 ppi.
    let mut res_data = Vec::new();
    res_data.extend_from_slice(&be32(72 << 16));
    res_data.extend_from_slice(&be16(1));
    res_data.extend_from_slice(&be16(1));
    res_data.extend_from_slice(&be32(72 << 16));
    res_data.extend_from_slice(&be16(1));
    res_data.extend_from_slice(&be16(1));
    let resources = resources_section(&[resource_block(1005, b"", &res_data)]);

    // ── layer 0 (bottom): "Backdrop" — full-canvas vertical blue gradient.
    let bd_r = vgrad(H, W, 0x10, 0x1c);
    let bd_g = vgrad(H, W, 0x24, 0x3f);
    let bd_b = vgrad(H, W, 0x52, 0x94);
    let bd_a = solid(H, W, 0xFF);
    let backdrop = layer_record(
        (0, 0, H as i32, W as i32),
        &rgba_channels(H, W, &bd_r, &bd_g, &bd_b, &bd_a),
        b"norm",
        255,
        None,
        b"Backdrop",
        &[luni_block("Backdrop")],
    );

    // ── layer 1: "Plate" — a solid orange block, multiply at 200/255.
    let (pt, pl, pb, pr) = (60, 60, 200, 260); // 140 rows x 200 cols
    let (ph, pw) = ((pb - pt) as u32, (pr - pl) as u32);
    let plate = layer_record(
        (pt, pl, pb, pr),
        &rgba_channels(
            ph,
            pw,
            &solid(ph, pw, 0xD9),
            &solid(ph, pw, 0x4F),
            &solid(ph, pw, 0x2B),
            &solid(ph, pw, 0xFF),
        ),
        b"mul ",
        200,
        None,
        b"Plate",
        &[luni_block("Plate")],
    );

    // ── layer 2 (top): "Signal" — a solid paper block, screen at 160/255.
    //    No mask (see the module docs: the layer-import tier refuses masked
    //    files, and this asset must open there).
    let (st, sl, sb, sr) = (120, 180, 240, 340); // 120 rows x 160 cols
    let (sh, sw) = ((sb - st) as u32, (sr - sl) as u32);
    let signal_channels = rgba_channels(
        sh,
        sw,
        &solid(sh, sw, 0xF4),
        &solid(sh, sw, 0xF1),
        &solid(sh, sw, 0xEA),
        &solid(sh, sw, 0xFF),
    );
    let signal = layer_record(
        (st, sl, sb, sr),
        &signal_channels,
        b"scrn",
        160,
        None,
        b"Signal",
        &[luni_block("Signal")],
    );

    // Channel image data, record order then channel order (tag + payload).
    let mut chan_data = Vec::new();
    for rec_channels in [
        &rgba_channels(H, W, &bd_r, &bd_g, &bd_b, &bd_a),
        &rgba_channels(
            ph,
            pw,
            &solid(ph, pw, 0xD9),
            &solid(ph, pw, 0x4F),
            &solid(ph, pw, 0x2B),
            &solid(ph, pw, 0xFF),
        ),
        &signal_channels,
    ] {
        for (_id, payload, comp) in rec_channels.iter() {
            chan_data.extend_from_slice(&channel_data(*comp, payload));
        }
    }

    // Layer info: i16 count (+3 — no transparency in the merged composite),
    // records bottom-first, channel data, pad to even.
    let mut layer_info_content = 3i16.to_be_bytes().to_vec();
    for rec in [&backdrop, &plate, &signal] {
        layer_info_content.extend_from_slice(rec);
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

    // Merged composite: RLE — ONE count table covering all 3 channels'
    // rows, then the packed scanlines channel-major. The flattened look:
    // the backdrop gradient (good RLE), enough for the placement oracle.
    let comp_r = vgrad(H, W, 0x10, 0x1c);
    let comp_g = vgrad(H, W, 0x24, 0x3f);
    let comp_b = vgrad(H, W, 0x52, 0x94);
    let mut table = Vec::new();
    let mut rows_packed = Vec::new();
    for plane in [&comp_r, &comp_g, &comp_b] {
        for r in 0..H as usize {
            let row = &plane[r * W as usize..(r + 1) * W as usize];
            let packed = image_psd::compression::packbits::encode(row);
            table.extend_from_slice(&be16(packed.len() as u16));
            rows_packed.extend_from_slice(&packed);
        }
    }
    let mut comp_payload = table;
    comp_payload.extend_from_slice(&rows_packed);
    let composite = channel_data(1, &comp_payload);

    let mut f = header(3, H, W);
    f.extend_from_slice(&be32(0)); // color mode data: empty (RGB)
    f.extend_from_slice(&resources);
    f.extend_from_slice(&layer_mask);
    f.extend_from_slice(&composite);
    f
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("annual-layers.psd")
}

#[test]
fn image_psd_annual_layers_fixture_builds_parses_and_renders() {
    let built = build_annual_layers_psd();

    // 1+2. The committed fixture IS the builder's output (byte-stable;
    // first run materializes it, every later run proves it).
    let path = fixture_path();
    if !path.exists() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &built).unwrap();
    }
    let committed = std::fs::read(&path).unwrap();
    assert_eq!(
        committed, built,
        "tests/fixtures/annual-layers.psd must equal the builder's output \
         (delete the file and rerun to regenerate)"
    );

    // 3a. Structural parse: three layers, bottom-first, with their blend
    // modes, opacities, names, and the top layer's mask.
    let file = PsdFile::parse(&built).expect("annual-layers.psd parses");
    let layers = &file.layer_mask.layers;
    assert_eq!(layers.len(), 3);
    assert_eq!(&layers[0].blend_key, b"norm");
    assert_eq!(layers[0].opacity, 255);
    assert_eq!(&layers[1].blend_key, b"mul ");
    assert_eq!(layers[1].opacity, 200);
    assert_eq!(&layers[2].blend_key, b"scrn");
    assert_eq!(layers[2].opacity, 160);
    // Mask-free by design: layer_plates_rgba8 refuses masked files (its
    // documented not-modeled envelope), and this asset must open there.
    assert!(layers.iter().all(|l| l.mask.is_none()));

    // 3b. The placement rendered tier: the merged composite decodes to
    // 400x300 straight RGBA8 (opaque — 3 channels, positive layer count).
    let comp = file.composite_rgba8().expect("composite decodes");
    assert_eq!((comp.width, comp.height), (W, H));
    assert_eq!(comp.rgba.len(), (W * H * 4) as usize);
    assert!(!comp.depth_reduced);
    // Top-left pixel is the gradient's top (0x10, 0x24, 0x52, opaque).
    assert_eq!(&comp.rgba[0..4], &[0x10, 0x24, 0x52, 0xFF]);

    // 3c. The layer-stack rendered tier: all three plates decode at canvas
    // extent with their names + stacking order intact.
    let plates = file.layer_plates_rgba8().expect("layer plates decode");
    assert_eq!((plates.width, plates.height), (W, H));
    let names: Vec<&str> = plates.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["Backdrop", "Plate", "Signal"]);
    // Record opacity rides the plate as its own field (not baked into alpha).
    assert_eq!(plates.layers[1].opacity, 200);
    // The Plate layer's pixels land inside its rect: sample (100, 100).
    let idx = ((100 * W + 100) * 4) as usize;
    assert_eq!(
        &plates.layers[1].rgba[idx..idx + 4],
        &[0xD9, 0x4F, 0x2B, 0xFF]
    );

    // 4. Preservation: a zero-edit write is byte-identical.
    assert_eq!(file.write().expect("re-emit"), built);
}
