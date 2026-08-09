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

//! The export-optimisation lane (RFI E-3, plugin half).
//!
//! The claim under test is the one that matters: `reduce` is LOSSLESS.
//! A size win that quietly altered pixels would be worse than no win,
//! so every reduction here is decoded back and compared byte for byte
//! against the input rather than merely being smaller.

use image_js::saveback::{
    encode_rgba8, encode_rgba8_opt, lossless_shape, LosslessShape, RasterFormat,
};

fn grey_opaque(w: u32, h: u32) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| {
            let v = (i % 256) as u8;
            [v, v, v, 255]
        })
        .collect()
}

fn grey_with_alpha(w: u32, h: u32) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| {
            let v = (i % 256) as u8;
            [v, v, v, (i % 251) as u8]
        })
        .collect()
}

fn colourful(w: u32, h: u32) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| [(i % 256) as u8, (i % 97) as u8, (i % 31) as u8, 255])
        .collect()
}

/// Colour AND translucent — the only shape with nothing to reduce.
fn colourful_with_alpha(w: u32, h: u32) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| {
            [
                (i % 256) as u8,
                (i % 97) as u8,
                (i % 31) as u8,
                (i % 251) as u8,
            ]
        })
        .collect()
}

#[test]
fn the_classifier_names_what_a_buffer_really_is() {
    assert_eq!(lossless_shape(&grey_opaque(16, 16)), LosslessShape::Gray);
    assert_eq!(
        lossless_shape(&grey_with_alpha(16, 16)),
        LosslessShape::GrayA
    );
    // Colour but OPAQUE: the alpha plane is dead weight and goes.
    assert_eq!(lossless_shape(&colourful(16, 16)), LosslessShape::Rgb);
    // Colour AND translucent: genuinely nothing to reduce.
    assert_eq!(
        lossless_shape(&colourful_with_alpha(16, 16)),
        LosslessShape::Rgba
    );
}

#[test]
fn one_off_grey_pixel_defeats_the_reduction() {
    // The guard that matters: a single colour pixel in an otherwise
    // grey image must force RGBA. A sampling classifier would miss it,
    // and the result would be a silently WRONG file.
    let mut buf = grey_opaque(32, 32);
    let last = buf.len() - 4;
    buf[last] = 200;
    buf[last + 1] = 10;
    // Still opaque, so it falls back to Rgb rather than all the way to
    // Rgba — the grey claim is defeated, the opacity claim is not.
    assert_eq!(lossless_shape(&buf), LosslessShape::Rgb);

    // And with one translucent pixel too, both claims are defeated.
    let mut both = grey_opaque(32, 32);
    both[3] = 128;
    let n = both.len();
    both[n - 4] = 200;
    both[n - 3] = 10;
    assert_eq!(lossless_shape(&both), LosslessShape::Rgba);
}

#[test]
fn a_reduced_png_decodes_back_bit_for_bit() {
    for (name, buf) in [
        ("grey opaque", grey_opaque(64, 64)),
        ("grey with alpha", grey_with_alpha(64, 64)),
        ("colour opaque", colourful(64, 64)),
        ("colour translucent", colourful_with_alpha(64, 64)),
    ] {
        let png = encode_rgba8_opt(&buf, 64, 64, RasterFormat::Png, 90, true)
            .unwrap_or_else(|e| panic!("{name}: encode failed: {e}"));
        assert_eq!(&png[0..4], &[0x89, 0x50, 0x4e, 0x47], "{name}: not a PNG");

        let back = image_js::ingest::decode_rgba8(&png)
            .unwrap_or_else(|e| panic!("{name}: decode failed: {e}"));
        assert_eq!(
            back.rgba.as_ref(),
            buf.as_slice(),
            "{name}: the reduction is supposed to be LOSSLESS, and this buffer \
             came back different"
        );
    }
}

#[test]
fn dropping_a_dead_alpha_plane_actually_saves_bytes() {
    // The most common export there is: an opaque colour image carrying
    // a pointless 255 alpha plane. A quarter of the raw bytes.
    let buf = colourful(128, 128);
    let plain = encode_rgba8(&buf, 128, 128, RasterFormat::Png).unwrap();
    let reduced = encode_rgba8_opt(&buf, 128, 128, RasterFormat::Png, 90, true).unwrap();
    assert!(
        reduced.len() < plain.len(),
        "reduced {} bytes vs plain {} — no saving",
        reduced.len(),
        plain.len()
    );
}

#[test]
fn reducing_a_grey_image_actually_saves_bytes() {
    // Not just correct — worth doing. If this ever stops holding the
    // knob is costing a pass over the buffer for nothing.
    let buf = grey_opaque(128, 128);
    let plain = encode_rgba8(&buf, 128, 128, RasterFormat::Png).unwrap();
    let reduced = encode_rgba8_opt(&buf, 128, 128, RasterFormat::Png, 90, true).unwrap();
    assert!(
        reduced.len() < plain.len(),
        "reduced {} bytes vs plain {} — no saving",
        reduced.len(),
        plain.len()
    );
}

#[test]
fn reduce_off_is_byte_identical_to_the_plain_door() {
    // The knob must be inert when off, or every existing caller's
    // output silently changes.
    let buf = grey_opaque(32, 32);
    let plain = encode_rgba8(&buf, 32, 32, RasterFormat::Png).unwrap();
    let opt = encode_rgba8_opt(&buf, 32, 32, RasterFormat::Png, 90, false).unwrap();
    assert_eq!(plain, opt, "reduce=false must not change the output");
}

#[test]
fn jpeg_quality_is_honoured_and_the_default_matches_the_plain_door() {
    let buf = colourful(64, 64);
    let low = encode_rgba8_opt(&buf, 64, 64, RasterFormat::Jpeg, 20, false).unwrap();
    let high = encode_rgba8_opt(&buf, 64, 64, RasterFormat::Jpeg, 95, false).unwrap();
    assert_eq!(&low[0..2], &[0xFF, 0xD8], "not a JPEG");
    assert!(
        low.len() < high.len(),
        "quality 20 ({} bytes) should beat quality 95 ({})",
        low.len(),
        high.len()
    );
    // The plain door rides a fixed 90; asking for 90 must reproduce it,
    // or the two lanes have quietly diverged.
    let plain = encode_rgba8(&buf, 64, 64, RasterFormat::Jpeg).unwrap();
    let ninety = encode_rgba8_opt(&buf, 64, 64, RasterFormat::Jpeg, 90, false).unwrap();
    assert_eq!(
        plain, ninety,
        "quality 90 must equal the fixed-quality door"
    );
}

#[test]
fn quality_is_clamped_rather_than_passed_through() {
    // 0 and 255 are both outside JPEG's 1..100. Clamping keeps a
    // caller's slider bug from reaching the encoder as garbage.
    let buf = colourful(16, 16);
    assert!(encode_rgba8_opt(&buf, 16, 16, RasterFormat::Jpeg, 0, false).is_ok());
    assert!(encode_rgba8_opt(&buf, 16, 16, RasterFormat::Jpeg, 255, false).is_ok());
}
