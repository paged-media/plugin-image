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

//! 16-bit PNG through the INGEST lane — it opens, and it KEEPS its bits.
//!
//! The byte-order proof lives in `image-conformance`'s
//! `codec_png_16bit.rs`, at the codec boundary where it belongs. What
//! is asserted here is that the depth survives the whole ingest, which
//! is what makes a 16-bit pipeline worth having: every kernel already
//! computed in f32 and stored rgba16float, so the only place precision
//! was ever lost was the round trip back to a byte — once per
//! operation.
//!
//! These tests previously asserted the opposite (that the file was
//! narrowed and SAID so). That was the honest contract while narrowing
//! was the behaviour; it is not the contract any more, and the tests
//! failing when the capability landed is them working.

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

const PROBE: [u16; 8] = [
    0x1234, 0x5678, 0x9ABC, 0xFFFF, 0xDE01, 0x02FE, 0x8001, 0x7FFF,
];

#[test]
fn a_16bit_png_opens_and_keeps_its_depth() {
    // It used to be `Unsupported` — the file would not open at all.
    // Then it opened NARROWED. It now opens at full depth.
    let img = image_js::ingest::decode_rgba8(&png16(2, 1, &PROBE)).expect("must open");
    assert_eq!(img.width, 2);
    assert_eq!(img.height, 1);
    assert!(img.rgba.is_16bit(), "the depth must survive the ingest");
    assert_eq!(img.rgba.pixel_count(), 2);
    assert_eq!(img.rgba.len(), 16, "2 px x 4 ch x 2 bytes");
}

#[test]
fn nothing_was_narrowed_so_nothing_claims_it_was() {
    // `depth_reduced` drives the panel's Depth row. With the bits now
    // surviving there is nothing to announce, and announcing it anyway
    // would be its own small lie.
    let img = image_js::ingest::decode_rgba8(&png16(2, 1, &PROBE)).unwrap();
    assert!(
        !img.depth_reduced,
        "a 16-bit PNG that keeps its bits was not reduced"
    );
}

#[test]
fn every_bit_survives_the_ingest_lane() {
    // The codec test proves the byte order at its own boundary; this
    // proves nothing downstream reorders or narrows it.
    let src = [0x1234u16, 0x5678, 0x9ABC, 0xFFFF];
    let img = image_js::ingest::decode_rgba8(&png16(1, 1, &src)).unwrap();
    for (c, want) in src.iter().enumerate() {
        assert_eq!(img.rgba.sample16(0, c), *want, "channel {c}");
    }
}

#[test]
fn the_values_an_8bit_lane_could_not_tell_apart_stay_distinct() {
    // The whole point. 0xFFFF and 0xFF00 both narrow to 255, so if they
    // come back equal the depth did not really survive.
    let img =
        image_js::ingest::decode_rgba8(&png16(1, 1, &[0xFFFF, 0xFF00, 0x8000, 0xFFFF])).unwrap();
    assert_ne!(img.rgba.sample16(0, 0), img.rgba.sample16(0, 1));
    assert_eq!(img.rgba.sample16(0, 0), 0xFFFF);
    assert_eq!(img.rgba.sample16(0, 1), 0xFF00);
}

#[test]
fn an_8bit_png_is_neither_widened_nor_marked_reduced() {
    // The other direction, which stops both flags becoming constants.
    let src = [10u8, 20, 30, 255, 40, 50, 60, 128];
    let img = image_js::ingest::decode_rgba8(&png8(2, 1, &src)).unwrap();
    assert!(!img.depth_reduced, "an 8-bit PNG lost nothing");
    assert!(!img.rgba.is_16bit(), "an 8-bit file must not be widened");
    assert_eq!(&img.rgba.to_rgba8()[..], &src[..]);
}

// ─────────────── the layered path (plan step 4) ───────────────
//
// The journal turned out to be depth-agnostic already: it stores opaque
// tile byte handles, and `FlatImage` always took bytes-per-pixel as a
// parameter rather than baking in 4. So nothing in `image-graph` had to
// change — the only bug to avoid was narrowing on the way IN or OUT.
//
// Undo is the sharp end. Restoring through an 8-bit view would lose the
// precision the edit had kept, and it would only show up after an undo,
// which is the worst place to find a bug.

use image_core::Region;
use image_js::layers::LayerStack;
use std::sync::Arc;

fn px16(w: u32, h: u32, v: u16) -> image_js::pixels::Pixels {
    image_js::pixels::Pixels::from_rgba16(&vec![v; (w * h * 4) as usize])
}

#[test]
fn a_16bit_layer_edit_journals_and_undoes_at_16_bits() {
    let mut s = LayerStack::from_image_px(4, 4, px16(4, 4, 0x1234)).unwrap();
    assert!(s.active().rgba.is_16bit(), "the stack must start 16-bit");

    s.edit_active("paint", Region::new(0, 0, 4, 4), px16(4, 4, 0xABCD))
        .expect("unlocked");
    assert_eq!(s.active().rgba.sample16(0, 0), 0xABCD);
    assert!(s.active().rgba.is_16bit(), "the edit must not narrow");

    // THE assertion. An undo that restored through an 8-bit view would
    // give 0x1200 here — the high byte preserved, the low byte gone.
    s.undo().expect("something to undo");
    assert_eq!(
        s.active().rgba.sample16(0, 0),
        0x1234,
        "undo must restore the FULL sample, not its high byte"
    );
    assert!(s.active().rgba.is_16bit(), "undo must not narrow either");
}

#[test]
fn a_mismatched_depth_edit_is_refused_rather_than_narrowed() {
    // The journal snapshots tiles in one geometry and restores them in
    // another, so a depth mismatch corrupts an UNDO rather than merely
    // degrading the edit. Refuse loudly.
    let mut s = LayerStack::from_image_px(4, 4, px16(4, 4, 0x1234)).unwrap();
    let eight = image_js::pixels::Pixels::from_rgba8(Arc::from(vec![9u8; 4 * 4 * 4]));
    let err = s
        .edit_active("paint", Region::new(0, 0, 4, 4), eight)
        .expect_err("a depth mismatch must be refused");
    assert!(
        format!("{err}").contains("depths must match"),
        "the refusal must say why: {err}"
    );
}

#[test]
fn an_8bit_stack_is_untouched_by_any_of_this() {
    let mut s = LayerStack::from_image(4, 4, Arc::from(vec![10u8; 4 * 4 * 4])).unwrap();
    assert!(!s.active().rgba.is_16bit());
    s.edit_active(
        "paint",
        Region::new(0, 0, 4, 4),
        image_js::pixels::Pixels::from_rgba8(Arc::from(vec![20u8; 4 * 4 * 4])),
    )
    .expect("unlocked");
    assert_eq!(s.active().rgba.to_rgba8()[0], 20);
    s.undo().expect("something to undo");
    assert_eq!(s.active().rgba.to_rgba8()[0], 10);
    assert!(!s.active().rgba.is_16bit(), "must not be widened");
}
