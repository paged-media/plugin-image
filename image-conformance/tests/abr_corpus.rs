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

//! **LANE B** — the corpus gate (spec §14.3.1). **ANALYST-owned**,
//! opt-in, `#[ignore]`d, and it needs the clean-room mount.
//!
//! ```text
//! PAGED_ABR_CORPUS=1 cargo test -p image-conformance --test abr_corpus \
//!     -- --ignored --nocapture
//! ```
//!
//! It parses the nine licensed `.abr` files at `references/abr-fixtures/`
//! and compares them, row by row, against the artifacts an ANALYST
//! published from that same corpus — `fixtures/abr/corpus-profile.json`
//! and `corpus-record-ledger.tsv`. Mirrors the existing opt-in
//! `PAGED_PSD_ORACLE` lane (`tests/psd_ecosystem.rs`) in shape and in
//! posture: `#[ignore]` keeps it out of the ordinary run, and each test
//! additionally no-ops with a clean `eprintln!` skip when the switch is
//! off or the mount is absent.
//!
//! **It is skipped on every machine that is not an analyst's, and that is
//! the design, not a defect.** `references/` is gitignored, CI has no
//! copy, and the IMPLEMENTER role may not read it at all (repo
//! `CLAUDE.md` §3.1). The always-on half of the gate — which needs no
//! corpus because it drives synthesised fixtures from the published
//! tables — is `tests/abr_lane_a.rs`.
//!
//! # The four gates
//!
//! | # | Gate |
//! |---|---|
//! | B1 | Each file's row of the profile: version, minor version, the section kind/size list, brush count, tip count, descriptor version. |
//! | B2 | All 3,202 ledger rows: id, the four bounds, `w`/`h`, depth, compression, the record's declared/pad lengths and plane count, decoded byte count, and the **SHA-256 of the decoded coverage**. The §2.2 structural self-check, the RLE decoder and the polarity, pinned at once. |
//! | B3 | The 238 rows carrying a `png_oracle`: the tip digest equals the SHA-256 of that PNG's ALPHA channel. The polarity oracle of §2.5, re-run rather than remembered. |
//! | B4 | The corpus aggregates: 81,926 descriptor items, 45,706 long-form vs 36,220 4-byte keys, the whole OSType histogram, 102 distinct keys, and the class-id inventory. |
//!
//! # THE OWNERSHIP RULE
//!
//! `corpus-profile.json` and `corpus-record-ledger.tsv` are regenerated
//! **only by an ANALYST session with the mount**, from a parser written
//! against the behaviour spec rather than against anything in
//! `references/`. A change to either is a reviewable diff, and that is
//! the entire point: a Lane-B failure is then unambiguous — **either the
//! reader regressed, or the format understanding legitimately changed and
//! the analyst must re-publish.**
//!
//! **An implementer must never "fix" a Lane-B failure by editing an
//! expectation file.** Doing so converts a measurement into a tautology
//! and destroys the only independent check this reader has. If a lane
//! fails and the reader looks right, the correct move is to report it and
//! let the role that owns the mount re-measure.
//!
//! # Mount layout
//!
//! The corpus directory is `references/abr-fixtures/` by default and
//! holds the nine `.abr` files under the names in the profile's `files[]`.
//! The 238 published PNGs of the Mercator set are looked for in
//! `<corpus>/png-oracle/`. Both can be redirected —
//! `PAGED_ABR_CORPUS_DIR` and `PAGED_ABR_PNG_DIR` — because the mount's
//! internal layout is not something the published artifacts describe, and
//! an implementer cannot look. If the PNG directory is absent, B3 skips
//! by itself while B1/B2/B4 still run.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use image_conformance::abr_corpus::sha256::sha256_hex;
use image_conformance::abr_corpus::{ledger, profile, LedgerRow};
use image_psd::abr::{AbrFile, AbrWarning};
use image_psd::descriptor::{read_versioned_descriptor, Descriptor, DescriptorValue};
use image_psd::reader::ByteReader;

const SWITCH: &str = "PAGED_ABR_CORPUS";

/// `<workspace>/references/abr-fixtures`, or `PAGED_ABR_CORPUS_DIR`.
fn corpus_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("PAGED_ABR_CORPUS_DIR") {
        return PathBuf::from(dir);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("image-conformance has a workspace parent")
        .join("references")
        .join("abr-fixtures")
}

/// `<corpus>/png-oracle`, or `PAGED_ABR_PNG_DIR`.
fn png_dir() -> PathBuf {
    match std::env::var_os("PAGED_ABR_PNG_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => corpus_dir().join("png-oracle"),
    }
}

/// The nine corpus files, in profile order, or `None` after printing a
/// visible skip. Every path this lane needs is checked up front so the
/// skip says exactly what is missing.
fn corpus(gate: &str) -> Option<Vec<(String, Vec<u8>)>> {
    if std::env::var_os(SWITCH).as_deref() != Some(std::ffi::OsStr::new("1")) {
        eprintln!(
            "SKIP {gate}: {SWITCH} != 1. This lane is ANALYST-owned and needs the \
             clean-room mount; the always-on half of the gate is tests/abr_lane_a.rs. \
             Enable with: {SWITCH}=1 cargo test -p image-conformance --test abr_corpus \
             -- --ignored --nocapture"
        );
        return None;
    }
    let dir = corpus_dir();
    if !dir.is_dir() {
        eprintln!(
            "SKIP {gate}: the corpus mount {} is absent (gitignored by design; \
             point PAGED_ABR_CORPUS_DIR elsewhere if it lives somewhere else)",
            dir.display()
        );
        return None;
    }
    let mut out = Vec::new();
    for shape in profile().files() {
        let path = dir.join(&shape.file);
        match std::fs::read(&path) {
            Ok(bytes) => {
                assert_eq!(
                    bytes.len() as u64,
                    shape.bytes,
                    "{}: the mounted file is not the one the profile was measured from",
                    shape.file
                );
                out.push((shape.file.clone(), bytes));
            }
            Err(e) => {
                eprintln!(
                    "SKIP {gate}: the mount is incomplete — {} could not be read ({e}). \
                     A partial corpus would gate on a subset and report green for it.",
                    path.display()
                );
                return None;
            }
        }
    }
    Some(out)
}

// ── a reader-independent walk of the container ───────────────────────
//
// The reader is what is under test, so the section list and the record
// framing are re-derived from the bytes rather than taken from it.

fn sections(bytes: &[u8]) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    let mut i = 4usize; // the two version words
    while i + 12 <= bytes.len() {
        assert_eq!(&bytes[i..i + 4], b"8BIM", "section signature at offset {i}");
        let kind = String::from_utf8_lossy(&bytes[i + 4..i + 8]).into_owned();
        let size = u32::from_be_bytes(bytes[i + 8..i + 12].try_into().unwrap()) as usize;
        out.push((kind, size, i + 12));
        i += 12 + size + (4 - (size % 4)) % 4;
    }
    out
}

/// One sampled-tip record, as framed on disk.
struct RawRecord {
    declared: usize,
    pad: usize,
    array_count: u32,
    written_planes: u32,
    compression: u8,
}

fn be_u32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(b[at..at + 4].try_into().expect("4 bytes"))
}

/// Walk the `samp` section structurally — the same self-describing
/// virtual-memory-array-list the spec's §2.2 describes, and emphatically
/// not a 264-byte constant.
fn raw_records(bytes: &[u8]) -> Vec<RawRecord> {
    let mut out = Vec::new();
    for (kind, size, at) in sections(bytes) {
        if kind != "samp" {
            continue;
        }
        let (mut i, end) = (at, at + size);
        while i + 4 <= end {
            let declared = be_u32(bytes, i) as usize;
            let rounded = declared.div_ceil(4) * 4;
            let body = i + 4;
            assert!(body + rounded <= end, "record at {i} overruns the section");

            let id_len = bytes[body] as usize;
            // vm list: unknown, version, length, rect, array_count.
            let mut p = body + 1 + id_len;
            let array_count = be_u32(bytes, p + 4 + 4 + 4 + 16);
            p += 4 + 4 + 4 + 16 + 4;
            let mut written_planes = 0u32;
            let mut compression = u8::MAX;
            for _ in 0..array_count + 2 {
                let is_written = be_u32(bytes, p);
                p += 4;
                if is_written == 0 {
                    continue;
                }
                written_planes += 1;
                let array_length = be_u32(bytes, p) as usize;
                // pixel_depth(4) + rect(16) + depth(2) + compression(1)
                compression = bytes[p + 4 + 4 + 16 + 2];
                p += 4 + array_length;
            }
            assert_eq!(
                p,
                body + declared,
                "THE STRUCTURAL SELF-CHECK (spec §2.2): the array list must end exactly \
                 on the DECLARED extent, with the 0..=3 pad outside the body"
            );
            out.push(RawRecord {
                declared,
                pad: rounded - declared,
                array_count,
                written_planes,
                compression,
            });
            i = body + rounded;
        }
    }
    out
}

/// The ledger rows belonging to one file, in index order.
fn rows_for(file: &str) -> Vec<&'static LedgerRow> {
    let mut v: Vec<&LedgerRow> = ledger().iter().filter(|r| r.file == file).collect();
    v.sort_by_key(|r| r.index);
    v
}

// ── B1 — the per-file profile row ────────────────────────────────────

/// Warnings that the published artifacts say cannot occur: the join
/// resolved 3,205/3,205 with no duplicates and no orphans, and every one
/// of the 3,202 records decoded with the self-check clean.
fn is_structural(w: &AbrWarning) -> bool {
    matches!(
        w,
        AbrWarning::SampleRecordOvershoot { .. }
            | AbrWarning::SampleRecordTrailingBytes { .. }
            | AbrWarning::MultiplePlanes { .. }
            | AbrWarning::SampleHasNoPlane { .. }
            | AbrWarning::PlaneDepthDisagrees { .. }
            | AbrWarning::SampleDecodeFailed { .. }
            | AbrWarning::DuplicateSampleId { .. }
            | AbrWarning::OrphanSample { .. }
            | AbrWarning::UnresolvedSampleReference { .. }
            | AbrWarning::UnexpectedDescriptorVersion { .. }
            | AbrWarning::SectionTrailingBytes { .. }
            | AbrWarning::UnknownSection { .. }
    )
}

#[test]
#[ignore = "ABR corpus gate: ANALYST-owned, opt-in (PAGED_ABR_CORPUS=1 + the clean-room mount)"]
fn image_abr_corpus_b1_every_file_matches_its_profile_row() {
    let Some(files) = corpus("B1 (per-file profile rows)") else {
        return;
    };
    for (shape, (name, bytes)) in profile().files().into_iter().zip(&files) {
        assert_eq!(&shape.file, name);
        let f = AbrFile::parse(bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(f.version, shape.version, "{name}: major version");
        assert_eq!(
            f.minor_version, shape.minor_version,
            "{name}: minor version"
        );

        let got: Vec<(String, usize)> = sections(bytes)
            .into_iter()
            .map(|(k, s, _)| (k, s))
            .collect();
        let want: Vec<(String, usize)> = shape
            .sections
            .iter()
            .map(|(k, s)| (k.clone(), *s as usize))
            .collect();
        assert_eq!(got, want, "{name}: the section kind/size list");

        assert_eq!(f.brushes.len(), shape.brushes, "{name}: brush presets");
        assert_eq!(f.samples.len(), shape.sampled_tips, "{name}: sampled tips");

        // The descriptor version is not surfaced on the model; the reader
        // reports any value other than 16, so silence IS the assertion.
        let unexpected: Vec<&AbrWarning> = f
            .warnings
            .iter()
            .filter(|w| matches!(w, AbrWarning::UnexpectedDescriptorVersion { .. }))
            .collect();
        if shape.descriptor_version == 16 {
            assert!(unexpected.is_empty(), "{name}: {unexpected:?}");
        } else {
            assert!(!unexpected.is_empty(), "{name}: expected a version report");
        }

        let structural: Vec<&AbrWarning> = f.warnings.iter().filter(|w| is_structural(w)).collect();
        assert!(
            structural.is_empty(),
            "{name}: the artifacts say these cannot happen: {structural:?}"
        );
        eprintln!(
            "OK B1 {name:<48} v{}.{} sections={} brushes={} tips={} warnings={:?}",
            shape.version,
            shape.minor_version,
            got.len(),
            shape.brushes,
            shape.sampled_tips,
            f.warnings
        );
    }
}

// ── B2 — the record ledger, digest included ──────────────────────────

#[test]
#[ignore = "ABR corpus gate: ANALYST-owned, opt-in (PAGED_ABR_CORPUS=1 + the clean-room mount)"]
fn image_abr_corpus_b2_every_record_matches_the_ledger_including_its_digest() {
    let Some(files) = corpus("B2 (record ledger + digests)") else {
        return;
    };
    let mut checked = 0usize;
    for (name, bytes) in &files {
        let f = AbrFile::parse(bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let rows = rows_for(name);
        assert_eq!(
            f.samples.len(),
            rows.len(),
            "{name}: the ledger has {} rows for {} parsed tips",
            rows.len(),
            f.samples.len()
        );
        let raw = raw_records(bytes);
        assert_eq!(
            raw.len(),
            rows.len(),
            "{name}: framed records vs ledger rows"
        );

        for ((s, r), raw) in f.samples.iter().zip(&rows).zip(&raw) {
            let ctx = format!("{name}[{}] {}", r.index, r.id);
            assert_eq!(s.id, r.id, "{ctx}: id");
            assert_eq!(
                (s.bounds.top, s.bounds.left, s.bounds.bottom, s.bounds.right),
                (r.top, r.left, r.bottom, r.right),
                "{ctx}: the bounds are stored y-first"
            );
            assert_eq!((s.width, s.height), (r.w, r.h), "{ctx}: w × h");
            assert_eq!(s.depth, r.depth, "{ctx}: depth");
            assert_eq!(s.coverage.len(), r.decoded_bytes, "{ctx}: decoded bytes");

            // The framing the ledger publishes so the §2.2 identity is
            // checkable: leftover == rounded − declared, in 0..=3.
            assert_eq!(raw.declared, r.declared_len, "{ctx}: declared length");
            assert_eq!(raw.pad, r.pad_len, "{ctx}: pad length");
            assert_eq!(raw.pad, (4 - (raw.declared % 4)) % 4, "{ctx}: the identity");
            assert_eq!(raw.array_count, r.array_count, "{ctx}: array_count");
            assert_eq!(raw.written_planes, r.written_planes, "{ctx}: planes");
            assert_eq!(raw.compression, r.compression, "{ctx}: compression");

            // The decode itself, pinned by a one-way digest: coverage,
            // row-major, one byte per pixel, NOT inverted.
            assert_eq!(r.depth, 8, "{ctx}: the ledger's digests are 8-bit coverage");
            assert_eq!(
                sha256_hex(&s.coverage8()),
                r.sha256,
                "{ctx}: the decoded coverage changed. This is the reader, the RLE \
                 decoder or the polarity — it is NOT the ledger, which only an \
                 analyst may regenerate."
            );
            checked += 1;
        }
    }
    assert_eq!(checked, ledger().len(), "every ledger row was visited");
    eprintln!("OK B2 {checked} records matched the ledger, digests included");
}

// ── B3 — the published-PNG polarity oracle ───────────────────────────

/// The alpha channel of a PNG, one byte per pixel, row-major.
fn png_alpha(bytes: &[u8], ctx: &str) -> (u32, u32, Vec<u8>) {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace;
    use zune_png::PngDecoder;

    let mut dec = PngDecoder::new(ZCursor::new(bytes));
    dec.decode_headers()
        .unwrap_or_else(|e| panic!("{ctx}: PNG headers: {e:?}"));
    let cs = dec
        .colorspace()
        .unwrap_or_else(|| panic!("{ctx}: no colorspace"));
    let (w, h) = dec
        .dimensions()
        .unwrap_or_else(|| panic!("{ctx}: no dimensions"));
    let px = dec
        .decode()
        .unwrap_or_else(|e| panic!("{ctx}: PNG decode: {e:?}"))
        .u8()
        .unwrap_or_else(|| panic!("{ctx}: not 8-bit samples"));
    let (stride, alpha_at) = match cs {
        ColorSpace::RGBA => (4usize, 3usize),
        ColorSpace::LumaA => (2, 1),
        other => panic!("{ctx}: the polarity oracle needs an ALPHA channel; this PNG is {other:?}"),
    };
    let alpha = px.iter().skip(alpha_at).step_by(stride).copied().collect();
    (w as u32, h as u32, alpha)
}

#[test]
#[ignore = "ABR corpus gate: ANALYST-owned, opt-in (PAGED_ABR_CORPUS=1 + the clean-room mount)"]
fn image_abr_corpus_b3_the_published_png_alpha_settles_the_polarity() {
    let Some(_files) = corpus("B3 (published-PNG polarity oracle)") else {
        return;
    };
    let dir = png_dir();
    if !dir.is_dir() {
        eprintln!(
            "SKIP B3 (published-PNG polarity oracle): {} is absent. The 238 published \
             PNGs are part of the mount, not of this repo; point PAGED_ABR_PNG_DIR at \
             them to run this gate. B1/B2/B4 are unaffected.",
            dir.display()
        );
        return;
    }
    let rows: Vec<&LedgerRow> = ledger().iter().filter(|r| r.png_oracle.is_some()).collect();
    assert_eq!(
        rows.len(),
        238,
        "the oracle covers 238 of the 3,202 records"
    );

    for r in &rows {
        let name = r.png_oracle.as_deref().expect("filtered");
        let path = dir.join(name);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let (w, h, alpha) = png_alpha(&bytes, name);
        assert_eq!(
            (w, h),
            (r.w, r.h),
            "{name}: the PNG is exactly w × h, no margin"
        );
        assert_eq!(
            sha256_hex(&alpha),
            r.sha256,
            "{name}: the tip's decoded coverage is no longer byte-identical to the \
             independently published alpha channel. The stored byte IS coverage — \
             255 painted, 0 unpainted — and an inversion here would look like a \
             broken blend mode rather than a decode bug."
        );
    }
    eprintln!(
        "OK B3 {} tips are byte-identical to published PNG alpha",
        rows.len()
    );
}

// ── B4 — the corpus aggregates ───────────────────────────────────────

/// Counted over every descriptor reachable from a `desc` or `phry`
/// section, the section root included.
///
/// The scope matters and is legible in the published numbers: `Objc` is
/// 9,874 while there are 16,611 class ids, because a descriptor that is a
/// LIST ELEMENT (a brush preset, a hierarchy node) has no key and so is
/// not a keyed item — but its own class id and its own items still count.
#[derive(Default)]
struct Tally {
    items: u64,
    ostype: BTreeMap<String, u64>,
    key_form: BTreeMap<String, u64>,
    key_ostype: BTreeMap<String, u64>,
    key_unit: BTreeMap<String, u64>,
    class_ids: BTreeMap<String, u64>,
    keys: BTreeSet<String>,
}

impl Tally {
    fn descriptor(&mut self, d: &Descriptor) {
        *self.class_ids.entry(d.class_id.text_lossy()).or_default() += 1;
        for (k, v) in &d.items {
            let key = k.text_lossy();
            self.items += 1;
            self.keys.insert(key.clone());
            *self
                .key_form
                .entry(if k.is_long_form() { "long" } else { "4cc" }.to_string())
                .or_default() += 1;
            let ostype = String::from_utf8_lossy(&v.ostype()).into_owned();
            *self.ostype.entry(ostype.clone()).or_default() += 1;
            *self
                .key_ostype
                .entry(format!("{key}|{ostype}"))
                .or_default() += 1;
            if let DescriptorValue::UnitFloat { unit, .. } = v {
                let unit = String::from_utf8_lossy(unit).into_owned();
                *self.key_unit.entry(format!("{key}|{unit}")).or_default() += 1;
            }
            self.value(v);
        }
    }

    fn value(&mut self, v: &DescriptorValue) {
        match v {
            DescriptorValue::Descriptor(d) | DescriptorValue::GlobalObject(d) => self.descriptor(d),
            DescriptorValue::List(items) => items.iter().for_each(|i| self.value(i)),
            _ => {}
        }
    }
}

#[test]
#[ignore = "ABR corpus gate: ANALYST-owned, opt-in (PAGED_ABR_CORPUS=1 + the clean-room mount)"]
fn image_abr_corpus_b4_the_aggregates_match_the_published_tables() {
    let Some(files) = corpus("B4 (corpus aggregates)") else {
        return;
    };
    let mut t = Tally::default();
    for (name, bytes) in &files {
        for (kind, size, at) in sections(bytes) {
            if kind != "desc" && kind != "phry" {
                continue;
            }
            let mut r = ByteReader::new(&bytes[at..at + size]);
            let (_version, root) =
                read_versioned_descriptor(&mut r).unwrap_or_else(|e| panic!("{name}/{kind}: {e}"));
            t.descriptor(&root);
        }
    }

    let p = profile();
    let scope = "counted over every descriptor reachable from a `desc`/`phry` section, \
                 root included; list elements are descriptors but not keyed items";

    assert_eq!(
        t.items,
        p.total("descriptor_values"),
        "descriptor items — {scope}"
    );
    assert_eq!(
        t.keys.len() as u64,
        p.total("distinct_keys"),
        "distinct keys — a reader that stops understanding a key changes this"
    );
    assert_eq!(
        t.key_form,
        p.counts("key_form_counts")
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
        "the long-form vs 4-byte key split — {scope}"
    );
    assert_eq!(
        t.ostype,
        p.counts("ostype_counts")
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
        "the OSType histogram — {scope}"
    );
    assert_eq!(
        t.key_ostype,
        p.counts("key_ostype_counts")
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
        "the key × OSType table — {scope}"
    );
    assert_eq!(
        t.key_unit,
        p.counts("key_unit_counts")
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
        "the key × unit table — {scope}"
    );
    assert_eq!(
        t.class_ids,
        p.counts("class_id_counts")
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
        "the class-id inventory — {scope}"
    );
    eprintln!(
        "OK B4 {} descriptor items, {} distinct keys, {} class ids — all tables matched",
        t.items,
        t.keys.len(),
        t.class_ids.values().sum::<u64>()
    );
}
