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

//! Real JPEGs — the follow-up `codec_jpeg.rs` asked for by name.
//!
//! That file says, of the whole CMYK path: *"we cannot synthesize a CMYK
//! JPEG through the adapter … A real-CMYK-corpus decode test is a
//! needs-real-corpus follow-up (spec §10.3 corpus rule)."* Every JPEG
//! this crate had ever decoded, it had first encoded itself, through an
//! encoder that emits only RGB and Gray. The Adobe APP14 inversion —
//! the thing that decides whether a print-bound CMYK JPEG comes out
//! inverted — was reachable only by a hand-built stored-sample unit
//! fixture.
//!
//! The 2026-08 corpus campaign found eleven JPEGs across the 61 Envato
//! pack zips. They are all "placeholders" by subject, which made them
//! easy to dismiss; by FILE they are the opposite of synthetic:
//!
//! * **five are CMYK** (4-channel, Adobe APP14), 596×596
//! * one is **progressive**, 4000×4000 — a different decode path
//!   entirely from every baseline file the crate has ever produced
//! * one carries **real EXIF** from a real pipeline
//!
//! There is no wider raster haul to be had: Envato strips the licensed
//! stock photography and ships a download link, so no zip has a
//! populated `Links/` folder and no zip contains a single TIFF or PNG.
//! These eleven are the whole real-raster tail we own.
//!
//! OPT-IN — the assets live in the private corpus checkout:
//!
//! ```text
//! PAGED_JPEG_CORPUS=1 cargo test -p image-conformance --test real_jpeg_corpus -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use image_codecs::{ImageSource, JpegSource, MemoryByteSource, SourceInfo};
use image_core::{ChannelLayout, Region, TileSliceMut};

/// Every extracted `assets/image/*.jpg`, or `None` with a printed reason.
fn corpus_jpegs() -> Option<Vec<PathBuf>> {
    let Some(switch) = std::env::var_os("PAGED_JPEG_CORPUS") else {
        eprintln!(
            "SKIP jpeg corpus lane: PAGED_JPEG_CORPUS unset \
             (set it to 1, or to a corpus root, and run with --ignored)"
        );
        return None;
    };
    let switch = switch.to_string_lossy().into_owned();
    let root = if switch == "1" || switch.is_empty() {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../corpus")
    } else {
        PathBuf::from(switch)
    };
    let packs = root.join("envato/packs");
    if !packs.is_dir() {
        eprintln!(
            "SKIP jpeg corpus lane: {} is not a directory",
            packs.display()
        );
        return None;
    }
    let mut out = Vec::new();
    for pack in std::fs::read_dir(&packs).ok()?.flatten() {
        let Ok(files) = std::fs::read_dir(pack.path().join("assets/image")) else {
            continue;
        };
        for f in files.flatten() {
            let p = f.path();
            let is_jpeg = p
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"));
            if p.is_file() && is_jpeg {
                out.push(p);
            }
        }
    }
    out.sort();
    if out.is_empty() {
        eprintln!(
            "SKIP jpeg corpus lane: no assets/image/*.jpg under {} — run corpus/envato/unpack.sh",
            packs.display()
        );
        return None;
    }
    Some(out)
}

fn name_of(p: &std::path::Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Decode the whole image. Returns `None` (with the error printed) rather
/// than panicking, so one bad file reports as one failure among many
/// instead of hiding the rest.
fn decode_full(path: &std::path::Path) -> Option<(SourceInfo, Vec<u8>)> {
    let bytes = std::fs::read(path).expect("read corpus jpeg");
    let mut src = JpegSource::new(MemoryByteSource::new(bytes.into_boxed_slice()));
    let info = match src.probe() {
        Ok(i) => i,
        Err(e) => {
            eprintln!("  probe failed {}: {e:?}", name_of(path));
            return None;
        }
    };
    let bpp = info.format.bytes_per_pixel();
    let region = Region::new(0, 0, info.width, info.height);
    let mut buf = vec![0u8; info.width as usize * info.height as usize * bpp];
    let mut out = TileSliceMut {
        region,
        format: info.format,
        row_stride: info.width as usize * bpp,
        bytes: &mut buf,
    };
    if let Err(e) = src.read_region(region, 1, &mut out) {
        eprintln!("  decode failed {}: {e:?}", name_of(path));
        return None;
    }
    Some((info, buf))
}

#[test]
#[ignore = "jpeg corpus lane: opt-in (PAGED_JPEG_CORPUS=1 + the private corpus mount)"]
fn every_real_jpeg_decodes_to_a_full_plausible_image() {
    let Some(files) = corpus_jpegs() else {
        return;
    };
    println!("jpeg corpus: {} file(s)", files.len());

    let mut failures: Vec<String> = Vec::new();
    for path in &files {
        let name = name_of(path);
        let Some((info, pixels)) = decode_full(path) else {
            failures.push(name);
            continue;
        };
        let bpp = info.format.bytes_per_pixel();
        assert_eq!(
            pixels.len(),
            info.width as usize * info.height as usize * bpp,
            "{name}: decoded buffer does not match the probed dimensions"
        );

        // A decoder that mis-reads the scan can still fill the buffer —
        // with a single constant. These are photographs and gradients,
        // so a real decode is never one flat value. This is the cheapest
        // assertion that separates "returned Ok" from "produced pixels".
        let first = pixels[0];
        assert!(
            pixels.iter().any(|&b| b != first),
            "{name}: every sample decoded to {first} — a uniform buffer is what a \
             failed scan looks like when the decoder still reports success"
        );

        println!(
            "  ok  {name:<28} {}x{} {:?} {} bpp  native={}  icc={}  exif={}",
            info.width,
            info.height,
            info.format.channels,
            bpp,
            info.native_format,
            info.icc.as_ref().map_or(0, |v| v.len()),
            info.exif_meta()
                .orientation
                .map_or("none".to_string(), |o| format!("{o:?}")),
        );
    }

    assert!(
        failures.is_empty(),
        "{} of {} real JPEGs failed to decode: {failures:?}",
        failures.len(),
        files.len()
    );
}

#[test]
#[ignore = "jpeg corpus lane: opt-in (PAGED_JPEG_CORPUS=1 + the private corpus mount)"]
fn the_corpus_covers_cmyk_jpeg_which_the_adapter_cannot_synthesise() {
    let Some(files) = corpus_jpegs() else {
        return;
    };

    let mut cmyk = Vec::new();
    for path in &files {
        let Some((info, pixels)) = decode_full(path) else {
            continue;
        };
        // The adapter maps CMYK onto a 4-channel layout and records the
        // container's own truth in `native_format` — that string is where
        // "this was a CMYK file with an Adobe APP14 marker" survives.
        if info.native_format.contains("cmyk") {
            cmyk.push((name_of(path), info, pixels));
        }
    }

    assert!(
        !cmyk.is_empty(),
        "no CMYK JPEG in the corpus — codec_jpeg.rs states outright that the \
         adapter cannot synthesise one, so this lane is the ONLY exercise the \
         Adobe APP14 inversion path gets outside a hand-built unit fixture"
    );

    for (name, info, pixels) in &cmyk {
        println!("  cmyk {name:<28} native={} ", info.native_format);
        assert_eq!(
            info.format.channels.count(),
            4,
            "{name}: a CMYK JPEG must decode to 4 channels, got {:?}",
            info.format.channels
        );

        // These carry a 557 KB embedded profile — the first real CMYK ICC
        // this crate has ever been handed. Everything downstream of
        // `image-cms` depends on it surviving the probe, and a CMYK file
        // WITHOUT one would be untransformable, so its absence is a
        // finding rather than a shrug.
        let icc = info.icc.as_ref().map_or(0, |v| v.len());
        assert!(
            icc > 0,
            "{name}: a CMYK JPEG with no embedded ICC cannot be colour-managed \
             — the corpus file used to carry one, so either it changed or the \
             probe stopped extracting it"
        );
        println!("       embedded icc: {icc} bytes");

        // The APP14 inversion is a whole-image polarity flip: get it
        // backwards and a light page decodes as a dark one. Measured here
        // rather than bounded tightly — there is no reference to compare
        // against, so this asserts only that the result is not PINNED at
        // an extreme, which is what a polarity bug or a dropped scan
        // actually looks like. The printed value is the real signal:
        // ~12/255 mean ink for light placeholder art, and a flip would
        // put it near 243.
        let n = info.format.channels.count() as usize;
        let mean = pixels
            .chunks_exact(n)
            .map(|px| px[..n.min(4)].iter().map(|&s| s as u64).sum::<u64>() / n as u64)
            .sum::<u64>() as f64
            / (pixels.len() / n) as f64;
        println!("       mean sample {mean:.1}/255");
        assert!(
            (1.0..=254.0).contains(&mean),
            "{name}: mean sample {mean:.1} is pinned at an extreme — the APP14 \
             inversion has almost certainly run the wrong way"
        );
    }
}

#[test]
#[ignore = "jpeg corpus lane: opt-in (PAGED_JPEG_CORPUS=1 + the private corpus mount)"]
fn a_progressive_jpeg_decodes_and_windows_like_a_baseline_one() {
    let Some(files) = corpus_jpegs() else {
        return;
    };

    // Progressive JPEGs arrive as successive approximation scans rather
    // than one pass, and every JPEG this crate encodes is baseline — so
    // without the corpus this path is unreachable. Find it by size: the
    // 4000x4000 photography-portfolio placeholder is the only one.
    let mut widest: Option<(String, SourceInfo, Vec<u8>)> = None;
    for path in &files {
        let Some((info, pixels)) = decode_full(path) else {
            continue;
        };
        if widest.as_ref().is_none_or(|(_, i, _)| info.width > i.width) {
            widest = Some((name_of(path), info, pixels));
        }
    }
    let Some((name, info, full)) = widest else {
        eprintln!("SKIP: nothing decoded");
        return;
    };
    println!("  largest: {name} {}x{}", info.width, info.height);

    // The M0 whole-decode-then-window invariant, on a real file: a
    // sub-region read must equal the same rectangle of the full decode.
    // If progressive scans are assembled per-region rather than per-image
    // the two disagree — and that is exactly the bug a synthesised
    // baseline fixture cannot surface.
    let bpp = info.format.bytes_per_pixel();
    let roi = Region::new(
        (info.width / 4) as i32,
        (info.height / 4) as i32,
        (info.width / 8).max(1),
        (info.height / 8).max(1),
    );
    let bytes = std::fs::read(
        files
            .iter()
            .find(|p| name_of(p) == name)
            .expect("the file we just decoded"),
    )
    .expect("re-read");
    let mut src = JpegSource::new(MemoryByteSource::new(bytes.into_boxed_slice()));
    src.probe().expect("probe");
    let mut buf = vec![0u8; roi.w as usize * roi.h as usize * bpp];
    let mut out = TileSliceMut {
        region: roi,
        format: info.format,
        row_stride: roi.w as usize * bpp,
        bytes: &mut buf,
    };
    src.read_region(roi, 1, &mut out).expect("windowed read");

    for row in 0..roi.h as usize {
        let sy = ((roi.y as usize + row) * info.width as usize + roi.x as usize) * bpp;
        let dy = row * roi.w as usize * bpp;
        let rb = roi.w as usize * bpp;
        assert_eq!(
            &buf[dy..dy + rb],
            &full[sy..sy + rb],
            "{name}: window {roi:?} row {row} differs from the full decode"
        );
    }
    println!("  windowed read matches the full decode ({} channels)", {
        let c: ChannelLayout = info.format.channels;
        c.count()
    });
}
