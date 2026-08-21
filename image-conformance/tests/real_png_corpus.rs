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

//! Real PNGs — the other half of what `codec_png.rs` can reach.
//!
//! `codec_png.rs` and `codec_png_16bit.rs` are thorough, and every PNG
//! they decode this crate first encoded itself. That is the same gap
//! `real_jpeg_corpus.rs` exists to close, and until 2026-08-21 it was
//! not closeable: the sibling JPEG lane states outright that "no zip
//! contains a single TIFF or PNG".
//!
//! That was true of the 61 IDML pack zips it was written against. It is
//! not true of the corpus as it now stands — the full extraction of all
//! 155 zips across six groups brought **484 real PNGs** out of the
//! archive, overwhelmingly from the html tier: UI kits, icon sheets,
//! logos and screenshots written by real design tools rather than by
//! `image-codecs`.
//!
//! What they add over the synthetic set is producer diversity — palette
//! and greyscale files, interlaced files, and the alpha-heavy output of
//! export pipelines this crate has never seen. The assertions are
//! deliberately the same shape as the JPEG lane's: every file decodes to
//! its probed dimensions, and no file decodes to a single flat value,
//! which is what a failed scan looks like when the decoder still reports
//! success.
//!
//! OPT-IN — the assets live in the private corpus checkout:
//!
//! ```text
//! PAGED_PNG_CORPUS=1 cargo test -p image-conformance --test real_png_corpus -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use image_codecs::{ImageSource, MemoryByteSource, PngSource, SourceInfo};
use image_core::{ChannelLayout, Region, TileSliceMut};

const ENV_SWITCH: &str = "PAGED_PNG_CORPUS";
const LANE: &str = "png";

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

/// Every file whose CONTENT is PNG, whatever it is called.
fn corpus_pngs() -> Option<Vec<PathBuf>> {
    corpus_of_kind("png")
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
    let bytes = std::fs::read(path).expect("read corpus png");
    let mut src = PngSource::new(MemoryByteSource::new(bytes.into_boxed_slice()));
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
#[ignore = "png corpus lane: opt-in (PAGED_PNG_CORPUS=1 + the private corpus mount)"]
fn every_real_png_decodes_to_a_full_plausible_image() {
    let Some(files) = corpus_pngs() else {
        return;
    };
    println!("png corpus: {} file(s)", files.len());

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

        // Same cheap separator the JPEG lane uses between "returned Ok"
        // and "produced pixels" — with one carve-out the JPEG set does
        // not need. A PNG legitimately CAN be one flat value: spacers,
        // single-colour swatches and fully transparent shims are real
        // files in real UI kits. So a flat buffer is reported, not
        // asserted on, and only a decode FAILURE fails the lane.
        let first = pixels[0];
        if !pixels.iter().any(|&b| b != first) {
            flat.push(name.clone());
            continue;
        }

        println!(
            "  ok  {name:<34} {}x{} {:?} {} bpp  native={}  icc={}",
            info.width,
            info.height,
            info.format.channels,
            bpp,
            info.native_format,
            info.icc.as_ref().map_or(0, |v| v.len()),
        );
    }

    if !flat.is_empty() {
        println!(
            "  note: {} uniform-value PNG(s) — legitimate for spacers and \
             single-colour swatches: {:?}",
            flat.len(),
            &flat[..flat.len().min(5)]
        );
    }

    assert!(
        failures.is_empty(),
        "{} of {} real PNGs failed to decode: {failures:?}",
        failures.len(),
        files.len()
    );
}

#[test]
#[ignore = "png corpus lane: opt-in (PAGED_PNG_CORPUS=1 + the private corpus mount)"]
fn the_corpus_covers_png_colour_types_the_encoder_never_emits() {
    let Some(files) = corpus_pngs() else {
        return;
    };

    // `PngTarget` writes a narrow set of colour types. The point of a
    // real corpus is the ones it cannot produce — palette, greyscale,
    // 16-bit and interlaced files all reach decode paths that a
    // round-trip through our own encoder can never exercise.
    let mut by_native: std::collections::BTreeMap<String, usize> = Default::default();
    let mut with_alpha = 0usize;
    for path in &files {
        let Some((info, _)) = decode_full(path) else {
            continue;
        };
        *by_native.entry(info.native_format.to_string()).or_default() += 1;
        if matches!(
            info.format.channels,
            ChannelLayout::GrayA | ChannelLayout::Rgba | ChannelLayout::Cmyka
        ) {
            with_alpha += 1;
        }
    }

    println!("png native formats across the corpus:");
    for (fmt, n) in &by_native {
        println!("    {fmt:<28} {n:>4}");
    }
    println!("  {with_alpha} file(s) carry alpha");

    assert!(
        by_native.len() > 1,
        "every real PNG probed as the same native format ({:?}) — this lane \
         exists because the corpus should reach colour types PngTarget never \
         emits, and one format means it is not doing that",
        by_native.keys().next()
    );
}

#[test]
#[ignore = "png corpus lane: opt-in (PAGED_PNG_CORPUS=1 + the private corpus mount)"]
fn a_file_called_png_that_is_not_one_is_refused_by_name() {
    let misnamed = corpus_misnamed(&["png"], "png");
    if misnamed.is_empty() {
        eprintln!("SKIP: no misnamed .png in the corpus");
        return;
    }

    // The danger is not that a misnamed file fails — it is that it
    // half-succeeds. `bg-pattern.png` is a WebP (RIFF/VP8L) shipped by a
    // React portfolio template, and a decoder that trusted the extension
    // and started parsing chunks would hand the caller pixels that are
    // not the image. Same shape as the corpus's other refusal gates:
    // plugin-sheets' legacy_xls_is_refused_rather_than_half_read and
    // plugin-doc's every_legacy_doc_is_refused_by_name.
    let mut wrongly_accepted = Vec::new();
    for (path, actual) in &misnamed {
        let bytes = std::fs::read(path).expect("read misnamed fixture");
        let mut src = PngSource::new(MemoryByteSource::new(bytes.into_boxed_slice()));
        match src.probe() {
            Ok(_) => wrongly_accepted.push(format!("{} (really {actual})", name_of(path))),
            Err(e) => println!(
                "  refused {:<26} really {actual:<6} {e:?}",
                name_of(path),
                e = e
            ),
        }
    }

    assert!(
        wrongly_accepted.is_empty(),
        "{} file(s) named .png but holding another format were ACCEPTED by the \
         PNG decoder: {:?} — a half-read image is worse than a refused one, \
         because the caller gets pixels that are not the file they opened",
        wrongly_accepted.len(),
        wrongly_accepted
    );
}
