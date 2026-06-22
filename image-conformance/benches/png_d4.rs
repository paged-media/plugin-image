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

//! D-4: zune-png vs image-rs/png decode/encode throughput over
//! synthetic PNGs (256² and 1024², RGBA8 + RGB8, mixed filters). The
//! winner (zune-png) is recorded in registry/codecs.yaml with the
//! measured numbers; it is re-confirmed against the Links corpus before
//! the M1 freeze (spec §10.3 corpus rule). Run: `cargo bench -p
//! image-conformance --bench png_d4 -- --quick`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Channel shape under test.
#[derive(Clone, Copy)]
enum Kind {
    Rgba8,
    Rgb8,
}

impl Kind {
    fn components(self) -> usize {
        match self {
            Kind::Rgba8 => 4,
            Kind::Rgb8 => 3,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Kind::Rgba8 => "rgba8",
            Kind::Rgb8 => "rgb8",
        }
    }
}

/// Synthetic pixels with structure that defeats trivial filtering (so
/// both encoders do real adaptive-filter + deflate work, not a
/// degenerate all-zero case): per-channel gradients plus a coarse
/// checkerboard.
fn synth(w: usize, h: usize, k: Kind) -> Vec<u8> {
    let n = k.components();
    let mut px = vec![0u8; w * h * n];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * n;
            let check = (((x / 8) ^ (y / 8)) & 1) as u8 * 40;
            px[base] = ((x * 3) as u8).wrapping_add(check);
            px[base + 1] = ((y * 5) as u8).wrapping_add(check);
            px[base + 2] = ((x + y) as u8).wrapping_add(check);
            if n == 4 {
                px[base + 3] = (255 - ((x ^ y) & 0xff)) as u8;
            }
        }
    }
    px
}

// ---- zune-png ----

fn zune_color(k: Kind) -> zune_core::colorspace::ColorSpace {
    match k {
        Kind::Rgba8 => zune_core::colorspace::ColorSpace::RGBA,
        Kind::Rgb8 => zune_core::colorspace::ColorSpace::RGB,
    }
}

fn zune_encode(w: usize, h: usize, k: Kind, px: &[u8]) -> Vec<u8> {
    use zune_core::bit_depth::BitDepth;
    use zune_core::options::EncoderOptions;
    use zune_png::PngEncoder;
    let opts = EncoderOptions::new(w, h, zune_color(k), BitDepth::Eight);
    let mut out = Vec::new();
    PngEncoder::new(px, opts).encode(&mut out).unwrap();
    out
}

fn zune_decode(bytes: &[u8]) -> Vec<u8> {
    use zune_core::bytestream::ZCursor;
    use zune_png::PngDecoder;
    let mut dec = PngDecoder::new(ZCursor::new(bytes));
    dec.decode().unwrap().u8().unwrap()
}

// ---- image-rs/png ----

fn imagers_color(k: Kind) -> png::ColorType {
    match k {
        Kind::Rgba8 => png::ColorType::Rgba,
        Kind::Rgb8 => png::ColorType::Rgb,
    }
}

fn imagers_encode(w: usize, h: usize, k: Kind, px: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
        enc.set_color(imagers_color(k));
        enc.set_depth(png::BitDepth::Eight);
        // Adaptive filtering — the apples-to-apples setting against
        // zune's default adaptive filter selection.
        enc.set_filter(png::Filter::Adaptive);
        let mut writer = enc.write_header().unwrap();
        writer.write_image_data(px).unwrap();
    }
    out
}

fn imagers_decode(bytes: &[u8]) -> Vec<u8> {
    // png 0.18 wants `BufRead + Seek`; a Cursor over the slice gives both.
    let dec = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = dec.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    buf
}

fn bench(c: &mut Criterion) {
    let dims = [(256usize, 256usize), (1024, 1024)];
    let kinds = [Kind::Rgba8, Kind::Rgb8];

    // Decode: each crate decodes a PNG the OTHER pre-encoded? No — decode
    // its own encode, so both read a valid, comparably-filtered stream.
    let mut dec_group = c.benchmark_group("png_decode");
    for &(w, h) in &dims {
        for &k in &kinds {
            let px = synth(w, h, k);
            let zbytes = zune_encode(w, h, k, &px);
            let ibytes = imagers_encode(w, h, k, &px);
            let pixels = (w * h) as u64;
            dec_group.throughput(Throughput::Elements(pixels));
            let id = format!("{}x{}_{}", w, h, k.label());

            dec_group.bench_with_input(BenchmarkId::new("zune", &id), &zbytes, |b, d| {
                b.iter(|| zune_decode(std::hint::black_box(d)))
            });
            dec_group.bench_with_input(BenchmarkId::new("imagers", &id), &ibytes, |b, d| {
                b.iter(|| imagers_decode(std::hint::black_box(d)))
            });
        }
    }
    dec_group.finish();

    let mut enc_group = c.benchmark_group("png_encode");
    for &(w, h) in &dims {
        for &k in &kinds {
            let px = synth(w, h, k);
            let pixels = (w * h) as u64;
            enc_group.throughput(Throughput::Elements(pixels));
            let id = format!("{}x{}_{}", w, h, k.label());

            enc_group.bench_with_input(BenchmarkId::new("zune", &id), &px, |b, p| {
                b.iter(|| zune_encode(w, h, k, std::hint::black_box(p)))
            });
            enc_group.bench_with_input(BenchmarkId::new("imagers", &id), &px, |b, p| {
                b.iter(|| imagers_encode(w, h, k, std::hint::black_box(p)))
            });
        }
    }
    enc_group.finish();
}

criterion_group!(d4, bench);
criterion_main!(d4);
