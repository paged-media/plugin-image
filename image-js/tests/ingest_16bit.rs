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

//! 16-bit PNG through the INGEST lane — it opens, and it says so.
//!
//! The byte-order proof lives in `image-conformance`'s
//! `codec_png_16bit.rs`, at the codec boundary where it belongs. What
//! is asserted here is the half that is about honesty rather than
//! correctness: a narrowed file must announce the narrowing, because a
//! 16-bit image that opens silently looks like an 8-bit original, and
//! that is the same quiet lie the PSD lane already refuses to tell.

use zune_core::bit_depth::BitDepth;
use zune_core::colorspace::ColorSpace;
use zune_core::options::EncoderOptions;
use zune_png::PngEncoder;

fn png16(width: u32, height: u32, samples: &[u16]) -> Vec<u8> {
    let mut be = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        be.extend_from_slice(&s.to_be_bytes());
    }
    let opts = EncoderOptions::new(
        width as usize,
        height as usize,
        ColorSpace::RGBA,
        BitDepth::Sixteen,
    );
    let mut sink = Vec::new();
    PngEncoder::new(&be, opts).encode(&mut sink).unwrap();
    sink
}

fn png8(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let opts = EncoderOptions::new(
        width as usize,
        height as usize,
        ColorSpace::RGBA,
        BitDepth::Eight,
    );
    let mut sink = Vec::new();
    PngEncoder::new(rgba, opts).encode(&mut sink).unwrap();
    sink
}

#[test]
fn a_16bit_png_opens_at_all() {
    // It used to be an `Unsupported` error: the file would not open.
    let bytes = png16(
        2,
        1,
        &[
            0x1234, 0x5678, 0x9ABC, 0xFFFF, 0xDE01, 0x02FE, 0x8001, 0x7FFF,
        ],
    );
    let img = image_js::ingest::decode_rgba8(&bytes).expect("16-bit PNG must open");
    assert_eq!(img.width, 2);
    assert_eq!(img.height, 1);
    assert_eq!(img.rgba.len(), 8);
}

#[test]
fn the_narrowing_is_announced_rather_than_hidden() {
    let bytes = png16(
        2,
        1,
        &[
            0x1234, 0x5678, 0x9ABC, 0xFFFF, 0xDE01, 0x02FE, 0x8001, 0x7FFF,
        ],
    );
    let img = image_js::ingest::decode_rgba8(&bytes).unwrap();
    assert!(
        img.depth_reduced,
        "a narrowed 16-bit PNG must announce the narrowing — the panel's \
         Depth row reads this flag"
    );
}

#[test]
fn an_8bit_png_does_not_claim_a_reduction_it_did_not_make() {
    // The other direction, which is what stops the flag becoming a
    // constant that says `true` and means nothing.
    let src = [10u8, 20, 30, 255, 40, 50, 60, 128];
    let img = image_js::ingest::decode_rgba8(&png8(2, 1, &src)).unwrap();
    assert!(!img.depth_reduced, "an 8-bit PNG lost nothing");
    assert_eq!(img.rgba.raw(), &src);
}

#[test]
fn the_high_byte_survives_the_ingest_lane_too() {
    // The codec test proves the byte order at its own boundary; this
    // proves nothing downstream re-orders it on the way to RGBA8.
    let bytes = png16(1, 1, &[0x1234, 0x5678, 0x9ABC, 0xFFFF]);
    let img = image_js::ingest::decode_rgba8(&bytes).unwrap();
    assert_eq!(img.rgba.raw(), &[0x12, 0x56, 0x9A, 0xFF]);
}
