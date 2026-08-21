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

const ENV_SWITCH: &str = "PAGED_JPEG_CORPUS";
const LANE: &str = "jpeg";

/// PNG magic, JPEG SOI, and the RIFF/WEBP container — enough to tell
/// what a file ACTUALLY is.
///
/// Selecting corpus files by extension is not safe here. The full 2026-08
/// extraction brought 1,288 pack images out of the archive and **267 of
/// them are misnamed**: 266 files called `.jpg` are PNG, and one called
/// `.png` is WebP. These are real vendor web templates, and an export
/// pipeline renaming a file without re-encoding it is evidently routine.
/// A lane that trusted the extension would report our decoder as broken
/// on 267 files that it reads exactly correctly.
fn sniff(path: &std::path::Path) -> &'static str {
    let mut buf = [0u8; 12];
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return "unreadable";
    };
    let n = f.read(&mut buf).unwrap_or(0);
    let b = &buf[..n];
    if b.starts_with(b"\x89PNG\r\n\x1a\n") {
        "png"
    } else if b.starts_with(&[0xFF, 0xD8, 0xFF]) {
        "jpeg"
    } else if b.starts_with(b"RIFF") && b.len() >= 12 && &b[8..12] == b"WEBP" {
        "webp"
    } else {
        "other"
    }
}

/// Every `assets/image/*` file across every group, with what it really
/// is. Returns `None` (printing why) when the corpus is not mounted.
fn corpus_images() -> Option<Vec<(std::path::PathBuf, &'static str)>> {
    let Some(switch) = std::env::var_os(ENV_SWITCH) else {
        eprintln!(
            "SKIP {LANE} corpus lane: {ENV_SWITCH} unset \
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
    let mut out = Vec::new();
    let mut any_group = false;
    for group in ["idml", "docx", "psd", "html", "vector", "pptx"] {
        let Ok(entries) = std::fs::read_dir(root.join(group).join("packs")) else {
            continue;
        };
        any_group = true;
        for pack in entries.flatten() {
            let Ok(files) = std::fs::read_dir(pack.path().join("assets").join("image")) else {
                continue;
            };
            for f in files.flatten() {
                let p = f.path();
                if p.is_file() {
                    let kind = sniff(&p);
                    out.push((p, kind));
                }
            }
        }
    }
    // ALSO the standalone raster tier. `raster/image-rs` is the image-rs
    // decoder suite (MIT, pinned) — 31 PNGs and 6 JPEGs of deliberate
    // format-edge coverage, plus 157 files in eleven formats this
    // workspace has no codec for. The pack tier is real-world output;
    // this tier is adversarial by construction, and neither substitutes
    // for the other.
    fn walk_raster(dir: &std::path::Path, out: &mut Vec<(PathBuf, &'static str)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_symlink() {
                continue;
            }
            if p.is_dir() {
                if !p
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with('.'))
                {
                    walk_raster(&p, out);
                }
            } else if p.is_file() {
                let kind = sniff(&p);
                out.push((p, kind));
            }
        }
    }
    let raster = root.join("raster");
    if raster.is_dir() {
        any_group = true;
        walk_raster(&raster, &mut out);
    }

    if !any_group {
        eprintln!(
            "SKIP {LANE} corpus lane: no <group>/packs or raster/ under {}",
            root.display()
        );
        return None;
    }
    out.sort();
    Some(out)
}

/// Files whose CONTENT is this lane's format, whatever they are called.
fn corpus_of_kind(kind: &str) -> Option<Vec<PathBuf>> {
    let all = corpus_images()?;
    let out: Vec<PathBuf> = all
        .iter()
        .filter(|(_, k)| *k == kind)
        .map(|(p, _)| p.clone())
        .collect();
    if out.is_empty() {
        eprintln!("SKIP {LANE} corpus lane: no {kind} content — run corpus/harness/unpack.sh");
        return None;
    }
    Some(out)
}

/// Files CALLED this lane's extension whose content is something else.
fn corpus_misnamed(exts: &[&str], kind: &str) -> Vec<(PathBuf, &'static str)> {
    let Some(all) = corpus_images() else {
        return Vec::new();
    };
    all.into_iter()
        .filter(|(p, k)| {
            *k != kind
                && p.extension().is_some_and(|e| {
                    let e = e.to_string_lossy().to_lowercase();
                    exts.contains(&e.as_str())
                })
        })
        .collect()
}

/// Every file whose CONTENT is JPEG, whatever it is called.
fn corpus_jpegs() -> Option<Vec<PathBuf>> {
    corpus_of_kind("jpeg")
}

/// Does this file carry an SOF2 (progressive) frame header?
///
/// Walks the marker chain rather than scanning for the byte pair, so an
/// 0xFFC2 occurring inside entropy-coded data or a thumbnail cannot be
/// mistaken for a frame header.
fn is_progressive(path: &std::path::Path) -> bool {
    let Ok(d) = std::fs::read(path) else {
        return false;
    };
    if d.len() < 4 || d[0] != 0xFF || d[1] != 0xD8 {
        return false;
    }
    let mut i = 2usize;
    while i + 3 < d.len() {
        if d[i] != 0xFF {
            i += 1;
            continue;
        }
        let m = d[i + 1];
        match m {
            0xC2 => return true,                       // SOF2 — progressive
            0xC0 | 0xC1 | 0xC3 | 0xDA => return false, // a non-progressive frame, or the scan
            0xD8 | 0xD9 => i += 2,
            0xD0..=0xD7 | 0x01 | 0xFF => i += 2,
            _ => {
                let len = u16::from_be_bytes([d[i + 2], d[i + 3]]) as usize;
                if len < 2 {
                    return false;
                }
                i += 2 + len;
            }
        }
    }
    false
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
    let mut flat: Vec<String> = Vec::new();
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
        // with a single constant, so a flat result is what "returned Ok"
        // looks like when nothing was actually decoded.
        //
        // This was a per-file assertion while the corpus was eleven
        // photographs. It cannot stay one: `raster/image-rs` is a
        // decoder TEST SUITE, and some of its fixtures are legitimately
        // flat — `exif-xmp-metadata.jpg` is 5x5 pixels in 4,263 bytes — 99%
        // metadata — and decodes flat
        // because its job is to carry EXIF and XMP, not pixels. Failing
        // on it would be asserting that purpose-built fixtures must look
        // like photographs.
        //
        // So flat files are COLLECTED, and the guard becomes a ratio
        // below. That keeps what the assertion was actually worth — a
        // regression that flattens the decoder shows up as a flood, not
        // as one file — without punishing a fixture for being deliberate.
        let first = pixels[0];
        if !pixels.iter().any(|&b| b != first) {
            flat.push(name.clone());
            continue;
        }

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

    if !flat.is_empty() {
        println!(
            "  note: {} uniform-value JPEG(s) — legitimate for metadata-only \
             fixtures: {:?}",
            flat.len(),
            &flat[..flat.len().min(5)]
        );
    }

    assert!(
        failures.is_empty(),
        "{} of {} real JPEGs failed to decode: {failures:?}",
        failures.len(),
        files.len()
    );

    // The ratio guard the per-file assertion turned into. A handful of
    // deliberately flat fixtures is expected; a tenth of the corpus
    // decoding flat is a decoder that has stopped decoding.
    let flat_share = flat.len() * 100 / files.len().max(1);
    assert!(
        flat_share < 10,
        "{}% of {} real JPEGs decoded to a single uniform value ({} files) — a \
         few purpose-built fixtures are legitimately flat, but this many means \
         the scan is not being read: {:?}",
        flat_share,
        files.len(),
        flat.len(),
        &flat[..flat.len().min(10)]
    );
}

#[test]
#[ignore = "jpeg corpus lane: opt-in (PAGED_JPEG_CORPUS=1 + the private corpus mount)"]
fn the_corpus_covers_cmyk_jpeg_which_the_adapter_cannot_synthesise() {
    let Some(files) = corpus_jpegs() else {
        return;
    };

    let mut cmyk = Vec::new();
    let mut with_icc = 0usize;
    let mut without_icc = 0usize;
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

        // COUNTED, not asserted per file. This started life as
        // `assert!(icc > 0)` back when the corpus held exactly five CMYK
        // JPEGs, all from one pack, all carrying the same 557 KB
        // profile — and generalising "these five have an ICC" into "all
        // CMYK JPEGs must" was overreach. The 2026-08-20 packs brought
        // `Resume Page 1-01.jpg`: CMYK, Adobe APP14, and NO embedded
        // profile. That is legitimate and common — the consumer has to
        // fall back to a default, which is precisely the
        // FOGRA39-vs-SWOP trap that costs ~4 dE on every solid fill.
        //
        // So an un-profiled CMYK file is COVERAGE, not a failure. What
        // must hold is that the corpus contains at least one of each, so
        // both paths stay exercised.
        let icc = info.icc.as_ref().map_or(0, |v| v.len());
        if icc > 0 {
            with_icc += 1;
        } else {
            without_icc += 1;
        }
        println!(
            "       embedded icc: {}",
            if icc > 0 {
                format!("{icc} bytes")
            } else {
                "NONE (defaults apply)".into()
            }
        );

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

    println!(
        "cmyk: {} file(s) — {with_icc} with an embedded ICC, {without_icc} without",
        cmyk.len()
    );
    // Both paths must stay covered: a profiled CMYK JPEG exercises the
    // ICC extraction, an un-profiled one exercises the default-profile
    // fallback. Losing either is a coverage regression, and until
    // 2026-08-20 the corpus had only the first.
    assert!(
        with_icc > 0,
        "no CMYK JPEG carries an embedded ICC — the profile-extraction path is untested"
    );
    assert!(
        without_icc > 0,
        "no CMYK JPEG lacks an ICC — the default-profile fallback is untested, and that \
         fallback is where the FOGRA39-vs-SWOP mismatch bites"
    );
}

#[test]
#[ignore = "jpeg corpus lane: opt-in (PAGED_JPEG_CORPUS=1 + the private corpus mount)"]
fn a_progressive_jpeg_decodes_and_windows_like_a_baseline_one() {
    let Some(files) = corpus_jpegs() else {
        return;
    };

    // Progressive JPEGs arrive as successive approximation scans rather
    // than one pass, and every JPEG this crate encodes is baseline — so
    // without the corpus this path is unreachable.
    //
    // This used to find one BY SIZE, on the reasoning that "the 4000x4000
    // photography-portfolio placeholder is the only one". True of eleven
    // files; false of 546. The widest file in the corpus is now a
    // baseline 6476x3643, so the test was no longer testing progressive
    // decoding at all — it just happened to still pass. Select on the
    // property the test is actually about.
    let progressive: Vec<&PathBuf> = files.iter().filter(|p| is_progressive(p)).collect();
    if progressive.is_empty() {
        eprintln!("SKIP: no progressive JPEG in the corpus");
        return;
    }
    println!("progressive JPEGs in the corpus: {}", progressive.len());

    // Keep the PATH, never re-find by file name. Two different packs
    // ship a `p1.jpg` (370x460 and 6476x3643), so a basename lookup
    // silently returned the wrong file: the ROI was computed from one
    // image and read from the other, which surfaced as a bogus
    // "roi out of bounds" against the decoder. Same bare-name-collision
    // class as the three bugs this corpus extraction found in the
    // harness.
    let path = progressive
        .iter()
        .max_by_key(|p| decode_full(p).map(|(i, _)| i.width).unwrap_or(0))
        .expect("non-empty");
    let Some((info, full)) = decode_full(path) else {
        eprintln!("SKIP: the progressive file did not decode");
        return;
    };
    let name = name_of(path);
    println!("  progressive: {name} {}x{}", info.width, info.height);

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
    let bytes = std::fs::read(path).expect("re-read");
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

#[test]
#[ignore = "jpeg corpus lane: opt-in (PAGED_JPEG_CORPUS=1 + the private corpus mount)"]
fn a_file_called_jpg_that_is_not_one_is_refused_by_name() {
    let misnamed = corpus_misnamed(&["jpg", "jpeg"], "jpeg");
    if misnamed.is_empty() {
        eprintln!("SKIP: no misnamed .jpg in the corpus");
        return;
    }

    // 266 of the corpus's `.jpg` files are PNG. That is not an oddity to
    // route around, it is the single largest misnaming population we
    // have, and it makes this the best-exercised refusal path in the
    // crate: the decoder must reject them by signature rather than start
    // parsing a JPEG scan out of PNG chunks.
    let mut wrongly_accepted = Vec::new();
    let mut by_actual: std::collections::BTreeMap<&str, usize> = Default::default();
    for (path, actual) in &misnamed {
        *by_actual.entry(actual).or_default() += 1;
        let bytes = std::fs::read(path).expect("read misnamed fixture");
        let mut src = JpegSource::new(MemoryByteSource::new(bytes.into_boxed_slice()));
        if src.probe().is_ok() {
            wrongly_accepted.push(format!("{} (really {actual})", name_of(path)));
        }
    }

    println!("misnamed .jpg files: {}", misnamed.len());
    for (actual, n) in &by_actual {
        println!("    really {actual:<6} {n:>4}");
    }

    assert!(
        wrongly_accepted.is_empty(),
        "{} file(s) named .jpg but holding another format were ACCEPTED by the \
         JPEG decoder: {:?} — a half-read image is worse than a refused one",
        wrongly_accepted.len(),
        wrongly_accepted
    );
}
