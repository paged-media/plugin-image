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

//! `ChannelData::decode`/`encode_*` — the `rendered` tier data feed
//! (spec §10.4). Self-contained byte-level fixtures: a hand-built RLE
//! count table, a miniz_oxide-compressed ZIP stream, a hand-applied
//! horizontal delta, and the error paths. Feature id:
//! `image.psd.layer.channel-data.zip` (and `…raw-rle`).

use image_psd::compression::packbits;
use image_psd::model::{ChannelData, Compression};
use image_psd::{Container, PsdError};

// ---- RAW -----------------------------------------------------------------

#[test]
fn image_psd_layer_channel_data_raw_rle_raw_roundtrip() {
    // RAW is the identity: encode_raw stores the plane verbatim, decode
    // copies it straight back.
    let plane: Vec<u8> = (0..(3u32 * 4)).map(|n| (n * 17) as u8).collect();
    let cd = ChannelData::encode_raw(&plane);
    assert_eq!(cd.compression, Compression::Raw);
    assert_eq!(cd.bytes, plane);
    let out = cd.decode(Container::Psd, 3, 4, 8).unwrap();
    assert_eq!(out, plane);
}

#[test]
fn image_psd_layer_channel_data_raw_rle_raw_wrong_size_errors() {
    // RAW plane that does not match rows*cols must be rejected.
    let cd = ChannelData {
        compression: Compression::Raw,
        bytes: vec![0u8; 10],
    };
    let err = cd.decode(Container::Psd, 3, 4, 8).unwrap_err(); // expects 12
    assert!(matches!(err, PsdError::Malformed { .. }));
}

// ---- RLE -----------------------------------------------------------------

/// Build an RLE payload by hand: a `u16`-per-row (PSD) count table, then
/// the PackBits-packed rows concatenated.
fn hand_rle_psd(rows: &[&[u8]]) -> Vec<u8> {
    let packed: Vec<Vec<u8>> = rows.iter().map(|r| packbits::encode(r)).collect();
    let mut bytes = Vec::new();
    for p in &packed {
        bytes.extend_from_slice(&(p.len() as u16).to_be_bytes());
    }
    for p in &packed {
        bytes.extend_from_slice(p);
    }
    bytes
}

#[test]
fn image_psd_layer_channel_data_raw_rle_rle_handbuilt_count_table() {
    // Two rows, 5 cols each. Row 0 has a replicate run, row 1 is literal.
    let row0: &[u8] = b"AABBB"; // -> literal AA + replicate B×3
    let row1: &[u8] = b"wxyz!";
    let bytes = hand_rle_psd(&[row0, row1]);

    // Sanity: the table is two big-endian u16 counts up front.
    let c0 = u16::from_be_bytes([bytes[0], bytes[1]]) as usize;
    let c1 = u16::from_be_bytes([bytes[2], bytes[3]]) as usize;
    assert_eq!(c0, packbits::encode(row0).len());
    assert_eq!(c1, packbits::encode(row1).len());
    assert_eq!(bytes.len(), 4 + c0 + c1);

    let cd = ChannelData {
        compression: Compression::Rle,
        bytes,
    };
    let out = cd.decode(Container::Psd, 2, 5, 8).unwrap();
    let mut expected = row0.to_vec();
    expected.extend_from_slice(row1);
    assert_eq!(out, expected);
}

#[test]
fn image_psd_layer_channel_data_raw_rle_rle_encode_decode_roundtrip_psd() {
    // encode_rle -> decode is the identity for an arbitrary plane.
    let rows = 4u32;
    let cols = 9u32;
    let plane: Vec<u8> = (0..(rows * cols))
        .map(|n| ((n * 31 + 7) % 256) as u8)
        .collect();
    let cd = ChannelData::encode_rle(&plane, Container::Psd, rows, cols).unwrap();
    assert_eq!(cd.compression, Compression::Rle);
    let out = cd.decode(Container::Psd, rows, cols, 8).unwrap();
    assert_eq!(out, plane);
}

#[test]
fn image_psd_layer_channel_data_raw_rle_rle_roundtrip_with_runs() {
    // A plane mixing long replicate runs and literal stretches across
    // multiple rows, to exercise both PackBits packets through the table.
    let rows = 3u32;
    let cols = 12u32;
    let mut plane = Vec::new();
    plane.extend(std::iter::repeat_n(0x55u8, 12)); // all-same row
    plane.extend((0..12u8).map(|n| n.wrapping_mul(9))); // distinct row
    plane.extend([1, 1, 1, 2, 2, 3, 3, 3, 3, 9, 9, 0]); // mixed row
    let cd = ChannelData::encode_rle(&plane, Container::Psd, rows, cols).unwrap();
    let out = cd.decode(Container::Psd, rows, cols, 8).unwrap();
    assert_eq!(out, plane);
}

#[test]
fn image_psd_container_psb_rle_u32_counts_roundtrip() {
    // PSB widens the count table to u32 per row. encode_rle must emit
    // 4-byte counts and decode must read them back.
    let rows = 2u32;
    let cols = 6u32;
    let plane: Vec<u8> = (0..(rows * cols)).map(|n| (n * 13) as u8).collect();
    let cd = ChannelData::encode_rle(&plane, Container::Psb, rows, cols).unwrap();

    // The first 8 bytes are two big-endian u32 counts.
    let c0 = u32::from_be_bytes([cd.bytes[0], cd.bytes[1], cd.bytes[2], cd.bytes[3]]) as usize;
    let c1 = u32::from_be_bytes([cd.bytes[4], cd.bytes[5], cd.bytes[6], cd.bytes[7]]) as usize;
    assert_eq!(cd.bytes.len(), 8 + c0 + c1);

    let out = cd.decode(Container::Psb, rows, cols, 8).unwrap();
    assert_eq!(out, plane);

    // A PSB payload decoded as PSD (narrow counts) must NOT silently
    // produce the right plane.
    assert!(cd.decode(Container::Psd, rows, cols, 8).is_err());
}

#[test]
fn image_psd_layer_channel_data_raw_rle_rle_bad_count_table_errors() {
    // Count table claims more packed bytes than the payload carries.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&999u16.to_be_bytes()); // row 0 claims 999 bytes
    bytes.extend_from_slice(&[0x00, b'A']); // only 2 bytes follow
    let cd = ChannelData {
        compression: Compression::Rle,
        bytes,
    };
    let err = cd.decode(Container::Psd, 1, 1, 8).unwrap_err();
    assert!(matches!(err, PsdError::Malformed { .. }));
}

#[test]
fn image_psd_layer_channel_data_raw_rle_rle_truncated_table_errors() {
    // Payload shorter than the count table itself.
    let cd = ChannelData {
        compression: Compression::Rle,
        bytes: vec![0x00], // 1 byte, but PSD needs 2 per row * 2 rows = 4
    };
    let err = cd.decode(Container::Psd, 2, 4, 8).unwrap_err();
    assert!(matches!(err, PsdError::Malformed { .. }));
}

#[test]
fn image_psd_layer_channel_data_raw_rle_rle_row_decodes_to_wrong_size_errors() {
    // A row whose PackBits stream unpacks to fewer than `cols` bytes
    // surfaces packbits' underrun error through decode_rle.
    let short_row = packbits::encode(b"AB"); // unpacks to 2 bytes
    let mut bytes = (short_row.len() as u16).to_be_bytes().to_vec();
    bytes.extend_from_slice(&short_row);
    let cd = ChannelData {
        compression: Compression::Rle,
        bytes,
    };
    // Claim cols=5 -> the row only yields 2 bytes -> error.
    let err = cd.decode(Container::Psd, 1, 5, 8).unwrap_err();
    assert!(matches!(err, PsdError::Malformed { .. }));
}

// ---- ZIP (2) -------------------------------------------------------------

#[test]
fn image_psd_layer_channel_data_zip_inflate_roundtrip() {
    // Compress a known plane with miniz_oxide, then ChannelData::decode
    // must inflate it back exactly.
    let rows = 3u32;
    let cols = 8u32;
    let plane: Vec<u8> = (0..(rows * cols)).map(|n| (n * 11) as u8).collect();
    let zlib = miniz_oxide::deflate::compress_to_vec_zlib(&plane, 6);
    let cd = ChannelData {
        compression: Compression::Zip,
        bytes: zlib,
    };
    let out = cd.decode(Container::Psd, rows, cols, 8).unwrap();
    assert_eq!(out, plane);
}

#[test]
fn image_psd_layer_channel_data_zip_inflate_any_depth() {
    // Method 2 inflates raw bytes for any depth (no prediction applied);
    // the plane is treated as raw bytes (rows*cols byte count).
    let plane: Vec<u8> = (0..32u32).map(|n| (n * 3) as u8).collect();
    let zlib = miniz_oxide::deflate::compress_to_vec_zlib(&plane, 6);
    let cd = ChannelData {
        compression: Compression::Zip,
        bytes: zlib,
    };
    let out = cd.decode(Container::Psd, 4, 8, 16).unwrap();
    assert_eq!(out, plane);
}

#[test]
fn image_psd_layer_channel_data_zip_wrong_decoded_size_errors() {
    // Inflated length must match rows*cols.
    let plane = vec![7u8; 20];
    let zlib = miniz_oxide::deflate::compress_to_vec_zlib(&plane, 6);
    let cd = ChannelData {
        compression: Compression::Zip,
        bytes: zlib,
    };
    let err = cd.decode(Container::Psd, 3, 8, 8).unwrap_err(); // expects 24
    assert!(matches!(err, PsdError::Malformed { .. }));
}

#[test]
fn image_psd_layer_channel_data_zip_bad_stream_errors() {
    // Not a valid zlib stream -> inflate fails.
    let cd = ChannelData {
        compression: Compression::Zip,
        bytes: vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01],
    };
    let err = cd.decode(Container::Psd, 1, 6, 8).unwrap_err();
    assert!(matches!(err, PsdError::Malformed { .. }));
}

// ---- ZIP with prediction (3) ---------------------------------------------

/// Apply the per-row horizontal delta (the inverse of decode's undo):
/// `delta[i] = sample[i] - sample[i-1]`, first sample per row unchanged.
fn apply_prediction_8bit(plane: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let mut out = plane.to_vec();
    for r in 0..rows {
        let base = r * cols;
        // Walk right-to-left so we read original neighbours.
        for i in (1..cols).rev() {
            out[base + i] = plane[base + i].wrapping_sub(plane[base + i - 1]);
        }
    }
    out
}

#[test]
fn image_psd_layer_channel_data_zip_prediction_handapplied_delta() {
    // Hand-apply the horizontal delta, compress, decode -> the original
    // plane must come back (decode undoes the delta after inflate).
    let rows = 2u32;
    let cols = 6u32;
    let plane: Vec<u8> = vec![
        10, 12, 15, 15, 200, 199, // row 0 (wraps at the 199-200 step)
        0, 1, 3, 6, 10, 250, // row 1
    ];
    let deltaed = apply_prediction_8bit(&plane, rows as usize, cols as usize);
    // Sanity: the hand delta differs from the plane (it really encoded).
    assert_ne!(deltaed, plane);
    let zlib = miniz_oxide::deflate::compress_to_vec_zlib(&deltaed, 6);
    let cd = ChannelData {
        compression: Compression::ZipPrediction,
        bytes: zlib,
    };
    let out = cd.decode(Container::Psd, rows, cols, 8).unwrap();
    assert_eq!(out, plane);
}

#[test]
fn image_psd_layer_channel_data_zip_prediction_non8bit_unsupported() {
    // Depth != 8 on method 3 is deferred to M2 -> Unsupported.
    let zlib = miniz_oxide::deflate::compress_to_vec_zlib(&[0u8; 12], 6);
    let cd = ChannelData {
        compression: Compression::ZipPrediction,
        bytes: zlib,
    };
    let err = cd.decode(Container::Psd, 2, 6, 16).unwrap_err();
    assert!(matches!(err, PsdError::Unsupported(_)));
}
