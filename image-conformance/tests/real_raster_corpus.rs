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

//! The formats this workspace does NOT read — and must refuse by name.
//!
//! `corpus/raster/image-rs` is the image-rs decoder suite (MIT, pinned at
//! `44ce9226`): 192 files across thirteen formats. `image-codecs` has a
//! codec for exactly two of them, PNG and JPEG, which are gated by
//! `real_png_corpus.rs` and `real_jpeg_corpus.rs`.
//!
//! The other **eleven formats are the point of this lane.** BMP, GIF,
//! TIFF, WebP, TGA, OpenEXR, Radiance HDR, QOI, PBM, farbfeld and ICO
//! are all things a user can plausibly drag onto a canvas, and the
//! property that matters at that boundary is not that we read them — it
//! is that a file we cannot read is **refused by name rather than
//! half-decoded into pixels that are not the image**.
//!
//! That failure mode is not hypothetical here. The pack tier ships 266
//! files named `.jpg` that are really PNG and one `.png` that is really
//! WebP, so "the extension said so" is demonstrably not a safe basis for
//! choosing a decoder. A decoder that trusts it and starts parsing will
//! produce *something*.
//!
//! The corpus already gates this shape elsewhere — plugin-sheets'
//! `legacy_xls_is_refused_rather_than_half_read` and plugin-doc's
//! `every_legacy_doc_is_refused_by_name`. This is the raster case, and
//! it is by some distance the largest sample any of them has.
//!
//! OPT-IN — the assets live in the private corpus checkout:
//!
//! ```text
//! PAGED_RASTER_CORPUS=1 cargo test -p image-conformance --test real_raster_corpus -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use image_codecs::{ImageSource, JpegSource, MemoryByteSource, PngSource};

/// Documentation and licence texts that travel with the fixtures.
const NON_IMAGE: &[&str] = &["md", "txt", "license", "json", "toml"];

/// What a file actually is, by signature. Extensions are not trusted
/// anywhere in this crate's corpus lanes — see the module docs.
fn sniff(path: &Path) -> &'static str {
    let mut buf = [0u8; 16];
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
    } else if b.starts_with(b"GIF8") {
        "gif"
    } else if b.starts_with(b"BM") {
        "bmp"
    } else if b.starts_with(b"II*\x00") || b.starts_with(b"MM\x00*") {
        "tiff"
    } else if b.starts_with(&[0x76, 0x2F, 0x31, 0x01]) {
        "exr"
    } else if b.starts_with(b"#?RADIANCE") || b.starts_with(b"#?RGBE") {
        "hdr"
    } else if b.starts_with(b"qoif") {
        "qoi"
    } else if b.starts_with(b"farbfeld") {
        "farbfeld"
    } else if b.len() > 1 && b[0] == b'P' && (b'1'..=b'6').contains(&b[1]) {
        "netpbm"
    } else {
        // TGA and ICO have no leading magic worth trusting.
        "unmagicked"
    }
}

fn raster_files() -> Option<Vec<PathBuf>> {
    let Some(switch) = std::env::var_os("PAGED_RASTER_CORPUS") else {
        eprintln!(
            "SKIP raster corpus lane: PAGED_RASTER_CORPUS unset \
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
    let dir = root.join("raster");
    if !dir.is_dir() {
        eprintln!("SKIP raster corpus lane: {} not readable", dir.display());
        return None;
    }
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
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
                    walk(&p, out);
                }
            } else if p.is_file() {
                let ext = p
                    .extension()
                    .map(|e| e.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                let name = p.file_name().unwrap_or_default().to_string_lossy();
                if !NON_IMAGE.contains(&ext.as_str()) && !name.eq_ignore_ascii_case("license") {
                    out.push(p);
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(&dir, &mut out);
    out.sort();
    if out.is_empty() {
        eprintln!(
            "SKIP raster corpus lane: no fixtures under {}",
            dir.display()
        );
        return None;
    }
    Some(out)
}

fn name_of(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

#[test]
#[ignore = "raster corpus lane: opt-in (PAGED_RASTER_CORPUS=1 + the private corpus mount)"]
fn a_format_we_cannot_read_is_refused_by_both_decoders() {
    let Some(files) = raster_files() else {
        return;
    };

    let mut census: std::collections::BTreeMap<&str, usize> = Default::default();
    let mut accepted_by_png = Vec::new();
    let mut accepted_by_jpeg = Vec::new();
    let mut unreadable = 0usize;

    for path in &files {
        let kind = sniff(path);
        *census.entry(kind).or_default() += 1;
        if kind == "png" || kind == "jpeg" {
            continue; // gated by the format's own lane
        }
        let Ok(bytes) = std::fs::read(path) else {
            unreadable += 1;
            continue;
        };
        let boxed = bytes.into_boxed_slice();

        let mut p = PngSource::new(MemoryByteSource::new(boxed.clone()));
        if p.probe().is_ok() {
            accepted_by_png.push(format!("{} (really {kind})", name_of(path)));
        }
        let mut j = JpegSource::new(MemoryByteSource::new(boxed));
        if j.probe().is_ok() {
            accepted_by_jpeg.push(format!("{} (really {kind})", name_of(path)));
        }
    }

    println!("raster corpus: {} fixture(s)", files.len());
    for (kind, n) in &census {
        println!("    {kind:<12}{n:>5}");
    }
    if unreadable > 0 {
        println!("  {unreadable} unreadable file(s)");
    }

    assert!(
        accepted_by_png.is_empty(),
        "the PNG decoder ACCEPTED {} file(s) that are not PNG: {:?} — a \
         half-read image is worse than a refused one, because the caller \
         gets pixels that are not the file they opened",
        accepted_by_png.len(),
        &accepted_by_png[..accepted_by_png.len().min(8)]
    );
    assert!(
        accepted_by_jpeg.is_empty(),
        "the JPEG decoder ACCEPTED {} file(s) that are not JPEG: {:?}",
        accepted_by_jpeg.len(),
        &accepted_by_jpeg[..accepted_by_jpeg.len().min(8)]
    );
}

#[test]
#[ignore = "raster corpus lane: opt-in (PAGED_RASTER_CORPUS=1 + the private corpus mount)"]
fn the_suite_actually_spans_the_formats_it_claims_to() {
    let Some(files) = raster_files() else {
        return;
    };
    let kinds: std::collections::BTreeSet<&str> = files.iter().map(|p| sniff(p)).collect();

    // Guard against the tier silently emptying out — a refusal lane over
    // two formats proves much less than one over eleven, and a partial
    // re-download would look identical to a pass.
    assert!(
        kinds.len() >= 8,
        "only {} distinct raster format(s) in the corpus ({kinds:?}) — this \
         lane exists to face the breadth of things we cannot decode, so a \
         narrow set means the tier did not materialise: re-copy \
         corpus/raster/image-rs (see its PROVENANCE.md)",
        kinds.len()
    );
    println!("raster formats present: {kinds:?}");
}
