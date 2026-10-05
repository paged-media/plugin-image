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

//! Merged-composite decode (registry row
//! `image.psd.global.merged-composite`, the M4 ingest slice): RAW and
//! RLE composites in RGB/Grayscale at 8 bit decode to the exact RGBA8
//! planes; the unsupported set answers `PsdError::Unsupported` cleanly.

use image_psd::compression::packbits;
use image_psd::model::{
    ColorMode, ColorModeData, FileHeader, GlobalImageData, ImageResources, LayerAndMaskInfo,
    PsdFile,
};
use image_psd::{Container, PsdError};

/// Hand-assemble minimal PSD bytes: 26-byte header + three empty
/// sections + the composite (compression tag + payload). Goes through
/// the REAL parser so the test covers parse → decode end to end.
fn psd_bytes(
    channels: u16,
    width: u32,
    height: u32,
    depth: u16,
    mode: u16,
    compression: u16,
    payload: &[u8],
) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"8BPS");
    b.extend_from_slice(&1u16.to_be_bytes()); // version 1 = PSD
    b.extend_from_slice(&[0u8; 6]); // reserved
    b.extend_from_slice(&channels.to_be_bytes());
    b.extend_from_slice(&height.to_be_bytes());
    b.extend_from_slice(&width.to_be_bytes());
    b.extend_from_slice(&depth.to_be_bytes());
    b.extend_from_slice(&mode.to_be_bytes());
    b.extend_from_slice(&0u32.to_be_bytes()); // color mode data: empty
    b.extend_from_slice(&0u32.to_be_bytes()); // image resources: empty
    b.extend_from_slice(&0u32.to_be_bytes()); // layer & mask info: empty
    b.extend_from_slice(&compression.to_be_bytes());
    b.extend_from_slice(payload);
    b
}

/// A model-constructed PsdFile around just a header + composite (the
/// sections decode never touches stay `Default`).
fn psd_model(
    channels: u16,
    width: u32,
    height: u32,
    color_mode: ColorMode,
    transparency_in_merged: bool,
    compression: u16,
    raw: Vec<u8>,
) -> PsdFile {
    PsdFile {
        container: Container::Psd,
        header: FileHeader {
            channels,
            height,
            width,
            depth: 8,
            color_mode,
        },
        color_mode: ColorModeData::default(),
        resources: ImageResources::default(),
        layer_mask: LayerAndMaskInfo {
            transparency_in_merged,
            ..Default::default()
        },
        composite: GlobalImageData { compression, raw },
    }
}

#[test]
fn image_psd_global_merged_composite_decode_rgb8_raw() {
    // 2×2 RGB, RAW planar: R plane, G plane, B plane.
    let payload: Vec<u8> = [
        [10u8, 20, 30, 40],    // R
        [50u8, 60, 70, 80],    // G
        [90u8, 100, 110, 120], // B
    ]
    .concat();
    let bytes = psd_bytes(3, 2, 2, 8, 3, 0, &payload);
    let file = PsdFile::parse(&bytes).expect("parse");
    let img = file.composite_rgba8().expect("decode");
    assert_eq!((img.width, img.height), (2, 2));
    assert_eq!(
        img.rgba,
        vec![
            10, 50, 90, 255, // (0,0)
            20, 60, 100, 255, // (1,0)
            30, 70, 110, 255, // (0,1)
            40, 80, 120, 255, // (1,1)
        ]
    );
}

#[test]
fn image_psd_global_merged_composite_decode_gray8_raw() {
    let bytes = psd_bytes(1, 2, 1, 8, 1, 0, &[7, 200]);
    let file = PsdFile::parse(&bytes).expect("parse");
    let img = file.composite_rgba8().expect("decode");
    assert_eq!(img.rgba, vec![7, 7, 7, 255, 200, 200, 200, 255]);
}

#[test]
fn image_psd_global_merged_composite_decode_rgba8_rle_with_transparency() {
    // 3×2 RGBA, RLE. One count table covering all 4 channels' rows
    // (u16 entries for PSD), then the packed rows channel-major. The
    // transparency flag marks channel 3 as the merged alpha, and the
    // colour of a transparent merged image is stored matted against
    // white, so the decode un-mattes it: 177 at α 128 is 100, 191 at
    // α 64 is 0, and a clear pixel has no colour.
    let planes: [[u8; 6]; 4] = [
        [1, 2, 3, 177, 191, 6],      // R
        [11, 12, 13, 127, 255, 16],  // G
        [21, 22, 23, 255, 191, 26],  // B
        [255, 255, 255, 128, 64, 0], // A
    ];
    let mut table = Vec::new();
    let mut packed = Vec::new();
    for plane in &planes {
        for row in plane.chunks(3) {
            let enc = packbits::encode(row);
            table.extend_from_slice(&(enc.len() as u16).to_be_bytes());
            packed.extend_from_slice(&enc);
        }
    }
    let mut raw = table;
    raw.extend_from_slice(&packed);

    let file = psd_model(4, 3, 2, ColorMode::Rgb, true, 1, raw);
    let img = file.composite_rgba8().expect("decode");
    assert_eq!(
        img.rgba,
        vec![
            1, 11, 21, 255, //
            2, 12, 22, 255, //
            3, 13, 23, 255, //
            100, 0, 255, 128, //
            0, 255, 0, 64, //
            0, 0, 0, 0,
        ]
    );
}

#[test]
fn image_psd_global_merged_composite_extra_channel_without_flag_is_opaque() {
    // Same 4-channel layout but transparency_in_merged = false: the 4th
    // plane is a spot/alpha channel, NOT merged transparency — opaque.
    let payload: Vec<u8> = [[1u8], [2u8], [3u8], [9u8]].concat();
    let file = psd_model(4, 1, 1, ColorMode::Rgb, false, 0, payload);
    let img = file.composite_rgba8().expect("decode");
    assert_eq!(img.rgba, vec![1, 2, 3, 255]);
}

#[test]
fn image_psd_global_merged_composite_unsupported_answers_cleanly() {
    // Depth 16 is no longer here — it is ACCEPTED and reported (see
    // `…_sixteen_bit_is_reduced_and_reported`). Depth 1 still is not:
    // a bitmap composite is a different unpacking problem, not a
    // precision one.
    let file1 = {
        let mut f = psd_model(3, 1, 1, ColorMode::Rgb, false, 0, vec![0; 3]);
        f.header.depth = 1;
        f
    };
    assert!(matches!(
        file1.composite_rgba8(),
        Err(PsdError::Unsupported(_))
    ));

    // CMYK mode.
    let cmyk = psd_model(4, 1, 1, ColorMode::Cmyk, false, 0, vec![0; 4]);
    assert!(matches!(
        cmyk.composite_rgba8(),
        Err(PsdError::Unsupported(_))
    ));

    // ZIP composite (compression 2).
    let zip = psd_model(3, 1, 1, ColorMode::Rgb, false, 2, vec![]);
    assert!(matches!(
        zip.composite_rgba8(),
        Err(PsdError::Unsupported(_))
    ));

    // RAW size mismatch is Malformed, not a wrong image.
    let short = psd_model(3, 2, 2, ColorMode::Rgb, false, 0, vec![0; 5]);
    assert!(matches!(
        short.composite_rgba8(),
        Err(PsdError::Malformed { .. })
    ));
}

/// 16-BIT IS ACCEPTED, REDUCED AND REPORTED.
///
/// It used to be refused, and the refusal meant a 16-bit scan could not
/// be opened at all. The layer stack downstream is 8-bit, so the extra
/// precision cannot survive it either way — which makes "opens, and says
/// it was reduced" strictly better than "does not open".
#[test]
fn image_psd_global_merged_composite_sixteen_bit_is_reduced_and_reported() {
    // One RGB pixel, big-endian 16-bit: 0xAABB, 0xCCDD, 0xEEFF.
    let raw = vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
    let mut f = psd_model(3, 1, 1, ColorMode::Rgb, false, 0, raw);
    f.header.depth = 16;
    let out = f.composite_rgba8().expect("16-bit opens now");
    assert!(out.depth_reduced, "and it SAYS it was reduced");
    // The HIGH byte is the correct nearest-8-bit reading.
    assert_eq!(&out.rgba[0..4], &[0xAA, 0xCC, 0xEE, 255]);
}

/// An 8-bit file must not claim a reduction that did not happen.
#[test]
fn image_psd_global_merged_composite_eight_bit_reports_no_reduction() {
    let f = psd_model(3, 1, 1, ColorMode::Rgb, false, 0, vec![10, 20, 30]);
    let out = f.composite_rgba8().expect("8-bit");
    assert!(!out.depth_reduced);
    assert_eq!(&out.rgba[0..4], &[10, 20, 30, 255]);
}

/// 16-bit RLE stays REFUSED, and the refusal cites its evidence: no
/// corpus fixture is 16-bit, and the row-table semantics are unverified.
/// A guess would decode to a wrong-looking image, which is the one
/// outcome worse than a refusal.
#[test]
fn image_psd_global_merged_composite_sixteen_bit_rle_is_still_refused() {
    let mut f = psd_model(3, 1, 1, ColorMode::Rgb, false, 1, vec![0; 8]);
    f.header.depth = 16;
    let err = f.composite_rgba8().expect_err("unverified");
    let msg = err.to_string();
    assert!(msg.contains("16-bit RLE"), "{msg}");
    assert!(msg.contains("unverified"), "the refusal cites why: {msg}");
}

// ─────────────────────────────── CMYK ────────────────────────────────
//
// A CMYK document's merged composite is decoded to INK by
// `composite_cmyk8` (the conversion to RGB needs the file's profile and a
// CMS, which this crate does not have). The polarity and the matte were
// measured on Photoshop-written files
// (`image-conformance/fixtures/photoshop/cmyk-stacks`, replayed in
// `image-conformance/tests/psd_cmyk_photoshop.rs`); these pin the decode
// on hand-built bytes.

/// The planes are stored INVERTED (255 = no ink); the decode flips them
/// to ink amounts, interleaved C, M, Y, K.
#[test]
#[allow(non_snake_case)]
fn a_cmyk_composite_decodes_to_ink_not_to_stored_values__feat__image_psd_rendered() {
    // 2×1 CMYK RAW planar. Pixel 0 = paper (stored 255 everywhere);
    // pixel 1 = 100 % cyan + 40 % black (stored 0 / 255 / 255 / 153).
    let payload: Vec<u8> = [[255u8, 0], [255, 255], [255, 255], [255, 153]].concat();
    let bytes = psd_bytes(4, 2, 1, 8, 4, 0, &payload);
    let file = PsdFile::parse(&bytes).expect("parse");
    let c = file.composite_cmyk8().expect("decode");
    assert_eq!((c.width, c.height), (2, 1));
    assert_eq!(c.cmyk, vec![0, 0, 0, 0, 255, 0, 0, 102]);
    assert!(c.alpha.is_none());
    assert!(!c.depth_reduced);
}

/// The RGBA8 door does not pretend: a CMYK composite needs a colour
/// transform, and the refusal says where the ink is.
#[test]
#[allow(non_snake_case)]
fn a_cmyk_composite_is_not_handed_out_as_rgb__feat__image_psd_rendered() {
    let cmyk = psd_model(4, 1, 1, ColorMode::Cmyk, false, 0, vec![0; 4]);
    let err = cmyk.composite_rgba8().expect_err("needs a transform");
    assert!(err.to_string().contains("composite_cmyk8"), "{err}");
    let rgb = psd_model(3, 1, 1, ColorMode::Rgb, false, 0, vec![0; 3]);
    assert!(
        rgb.composite_cmyk8().is_err(),
        "and the ink door is CMYK-only"
    );
}

/// With the layer-count sign flag, the fifth channel is transparency and
/// the stored ink is un-matted against the paper (stored 255) before the
/// flip; without it, a fifth channel is a spot/alpha channel and the
/// image is opaque.
#[test]
#[allow(non_snake_case)]
fn a_transparent_cmyk_composite_is_unmatted_against_paper__feat__image_psd_rendered() {
    // One pixel, 100 % magenta at α = 128: stored M = 0·α + 255·(1 − α).
    let a = 128u8;
    let stored_m = (255.0 * (1.0 - f32::from(a) / 255.0)).round() as u8;
    let raw = vec![255, stored_m, 255, 255, a];
    let with_flag = psd_model(5, 1, 1, ColorMode::Cmyk, true, 0, raw.clone());
    let c = with_flag.composite_cmyk8().expect("decode");
    assert_eq!(c.alpha.as_deref(), Some(&[a][..]));
    assert_eq!(c.cmyk, vec![0, 255, 0, 0], "un-matted to full magenta");

    let without = psd_model(5, 1, 1, ColorMode::Cmyk, false, 0, raw);
    let c = without.composite_cmyk8().expect("decode");
    assert!(c.alpha.is_none(), "an extra channel is not transparency");
    assert_eq!(
        c.cmyk,
        vec![0, 255 - stored_m, 0, 0],
        "and nothing is un-matted"
    );
}

/// RLE: one count table for every channel's rows, channel-major — the
/// same layout as RGB, four ink planes long.
#[test]
#[allow(non_snake_case)]
fn a_cmyk_rle_composite_decodes__feat__image_psd_rendered() {
    // 3×1, each plane one row packed as a run of 3 (header 0xFE = repeat
    // 3 times), stored C=200, M=255, Y=100, K=255.
    let rows: [u8; 4] = [200, 255, 100, 255];
    let mut payload = Vec::new();
    for _ in 0..4 {
        payload.extend_from_slice(&2u16.to_be_bytes());
    }
    for v in rows {
        payload.extend_from_slice(&[0xFE, v]);
    }
    let bytes = psd_bytes(4, 3, 1, 8, 4, 1, &payload);
    let file = PsdFile::parse(&bytes).expect("parse");
    let c = file.composite_cmyk8().expect("decode");
    assert_eq!(c.cmyk, [55, 0, 155, 0].repeat(3));
    // Sanity on the codec: the same run decodes through packbits.
    let mut out = [0u8; 3];
    packbits::decode(&[0xFE, 7], &mut out).expect("run");
    assert_eq!(out, [7, 7, 7]);
}

/// 16-bit CMYK is accepted, reduced to the high byte, inverted, and
/// reported — what Photoshop writes for a 16-bit document's composite is
/// RAW (measured on the `normal-stack-16` fixture).
#[test]
#[allow(non_snake_case)]
fn a_sixteen_bit_cmyk_composite_is_reduced_and_reported__feat__image_psd_rendered() {
    // Stored big-endian: C 0x00FF, M 0xFFFF, Y 0x8000, K 0xFF00.
    let raw = vec![0x00, 0xFF, 0xFF, 0xFF, 0x80, 0x00, 0xFF, 0x00];
    let mut f = psd_model(4, 1, 1, ColorMode::Cmyk, false, 0, raw);
    f.header.depth = 16;
    let c = f.composite_cmyk8().expect("16-bit opens");
    assert!(c.depth_reduced);
    assert_eq!(c.cmyk, vec![255, 0, 0x7F, 0]);
}

/// A CMYK document is no longer a `colour-mode` blocker; what replaces it
/// is precise: without real merged data nothing can vouch for an RGB
/// blend of the converted layers (`cmyk-unverified`), and the RGBA8-only
/// layer door refuses without a transform.
#[test]
#[allow(non_snake_case)]
fn a_cmyk_document_is_judged_by_its_own_blockers__feat__image_psd_layer_import() {
    let f = psd_model(4, 1, 1, ColorMode::Cmyk, false, 0, vec![0; 4]);
    let cats: Vec<&str> = f
        .layer_import_blockers()
        .iter()
        .map(|b| b.category)
        .collect();
    assert!(!cats.contains(&"colour-mode"), "{cats:?}");
    assert!(cats.contains(&"cmyk-unverified"), "{cats:?}");
    let err = f.layer_plates_rgba8().expect_err("needs the ink transform");
    assert!(err.to_string().contains("layer_plates_rgba8_via"), "{err}");

    // Lab is still a colour-mode blocker.
    let lab = psd_model(3, 1, 1, ColorMode::Lab, false, 0, vec![0; 3]);
    let cats: Vec<&str> = lab
        .layer_import_blockers()
        .iter()
        .map(|b| b.category)
        .collect();
    assert!(cats.contains(&"colour-mode"), "{cats:?}");
}
