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

//! 16-bit PNG ingest — and the fixture whose absence was the blocker.
//!
//! The catalog recorded 16-bit as refused because "their 16-bit byte
//! order is unverified and no fixture exists to check a guess, and a
//! wrong guess decodes to a plausible-looking wrong image". Both halves
//! were wrong for PNG. The endianness is normative (PNG §7.1 — samples
//! are network byte order, MSB first), and we never touch the wire
//! bytes anyway: zune hands back `DecodingResult::U16(Vec<u16>)`,
//! already assembled into native values.
//!
//! A fixture also costs nothing to make, which is what this file does.
//! And the byte order is not asserted — it is MEASURED: the samples are
//! chosen so that reading them the wrong way round gives a DIFFERENT
//! answer, so a byte-swap fails loudly rather than producing the
//! plausible-looking wrong image the catalog feared.

use image_codecs::{ImageSource, MemoryByteSource, PngSource, SourceInfo};
use image_core::{ChannelLayout, Region, SampleDepth, TileSliceMut};
use zune_core::bit_depth::BitDepth;
use zune_core::colorspace::ColorSpace;
use zune_core::options::EncoderOptions;
use zune_png::PngEncoder;

/// Encode a 16-bit RGBA PNG from native `u16` samples.
///
/// The wire is big-endian by the spec, so the samples are written MSB
/// first here — this function IS the "guess" the catalog said could not
/// be checked, and the round trip below checks it.
fn encode_rgba16(width: u32, height: u32, samples: &[u16]) -> Vec<u8> {
    assert_eq!(samples.len(), (width * height * 4) as usize);
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

fn decode_full(png: Vec<u8>) -> (SourceInfo, Vec<u8>) {
    let mut src = PngSource::new(MemoryByteSource::new(png.into_boxed_slice()));
    let info = src.probe().unwrap();
    let bpp = info.format.bytes_per_pixel();
    let region = Region::new(0, 0, info.width, info.height);
    let mut buf = vec![0u8; info.width as usize * info.height as usize * bpp];
    let mut out = TileSliceMut {
        region,
        format: info.format,
        row_stride: info.width as usize * bpp,
        bytes: &mut buf,
    };
    src.read_region(region, 1, &mut out).unwrap();
    (info, buf)
}

/// Samples whose two bytes DIFFER, so a byte-swap is detectable.
/// `0x1234` narrows to `0x12`; read the wrong way round it is `0x34`.
const PROBE: [u16; 8] = [
    0x1234, 0x5678, 0x9ABC, 0xFFFF, // pixel 0
    0xDE01, 0x02FE, 0x8001, 0x7FFF, // pixel 1
];

#[test]
fn image_codec_png_16bit_opens_and_keeps_its_depth() {
    // It used to error `Unsupported` — the file would not open at all.
    // Then it opened NARROWED. It now opens at full depth: the source
    // reports U16 and the buffer is eight bytes per pixel.
    let png = encode_rgba16(2, 1, &PROBE);
    let (info, buf) = decode_full(png);
    assert_eq!(info.width, 2);
    assert_eq!(info.height, 1);
    assert_eq!(info.format.channels, ChannelLayout::Rgba);
    assert_eq!(info.format.depth, SampleDepth::U16);
    assert_eq!(buf.len(), 16, "2 px x 4 ch x 2 bytes");
}

#[test]
fn image_codec_png_16bit_byte_order_is_measured_not_assumed() {
    let png = encode_rgba16(2, 1, &PROBE);
    let (_, buf) = decode_full(png);

    // Reassemble the native-endian pairs the decoder emits and compare
    // against the samples that went in. A byte-swap anywhere in the
    // chain turns 0x1234 into 0x3412, so this is the order check — and
    // it is now EXACT rather than a comparison of narrowed high bytes,
    // because nothing is narrowed any more.
    let got: Vec<u16> = buf
        .chunks_exact(2)
        .map(|p| u16::from_ne_bytes([p[0], p[1]]))
        .collect();
    assert_eq!(
        got,
        PROBE.to_vec(),
        "16-bit samples came back reordered. The probe values have \
         DIFFERING bytes precisely so a swap cannot hide — which is the \
         'plausible-looking wrong image' the catalog warned a guess \
         could produce."
    );
    // State the negative too, so this cannot pass by an accident that
    // also matches a byte-swapped read.
    let swapped: Vec<u16> = PROBE.iter().map(|s| s.swap_bytes()).collect();
    assert_ne!(got, swapped, "read with the bytes reversed");
}

#[test]
fn image_codec_png_16bit_extremes_survive_intact() {
    // The values an 8-bit round trip cannot tell apart. 0xFF00 and
    // 0xFFFF both narrow to 255; keeping them DISTINCT is the whole
    // point of the depth surviving.
    let samples = [0xFFFFu16, 0x0000, 0x8000, 0xFF00];
    let png = encode_rgba16(1, 1, &samples);
    let (_, buf) = decode_full(png);
    let got: Vec<u16> = buf
        .chunks_exact(2)
        .map(|p| u16::from_ne_bytes([p[0], p[1]]))
        .collect();
    assert_eq!(got, samples.to_vec());
    assert_ne!(
        got[0], got[3],
        "0xFFFF and 0xFF00 must stay distinguishable"
    );
}

#[test]
fn image_codec_png_8bit_is_untouched_by_the_16bit_lane() {
    // The regression that would matter most: accepting 16-bit must not
    // perturb the 8-bit path every existing file rides.
    let opts = EncoderOptions::new(2, 1, ColorSpace::RGBA, BitDepth::Eight);
    let src = [10u8, 20, 30, 255, 40, 50, 60, 128];
    let mut png = Vec::new();
    PngEncoder::new(&src, opts).encode(&mut png).unwrap();
    let (info, pixels) = decode_full(png);
    assert_eq!(info.format.depth, SampleDepth::U8);
    assert_eq!(pixels, src.to_vec());
}
