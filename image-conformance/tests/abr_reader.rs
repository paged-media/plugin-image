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

//! `.abr` brush-preset reader conformance.
//!
//! Fixtures come from [`image_conformance::abr_builder`], an INDEPENDENT
//! byte emitter that shares no code with the reader. Each test is named
//! after the trap in
//! `thoughts/docs/paged/plugin-image/abr-brush-format-spec.md` §10 that
//! it exists to catch.
//!
//! What is tested ELSEWHERE, and why. Revision 2 of the behaviour spec
//! asked for the 9-file, 3,215-preset licensed corpus at
//! `plugin-image/references/abr-fixtures/` to be wired into this suite.
//! That corpus lives under `references/`, which the clean-room protocol
//! (repo CLAUDE.md §3.1) forbids an implementer from reading, so the
//! instruction had no owner; §14.3.1 resolves it by splitting the gate.
//!
//! * `tests/abr_lane_a.rs` — LANE A: the same checks, driven from the
//!   ANALYST-published `fixtures/abr/corpus-profile.json` and
//!   `corpus-record-ledger.tsv` against synthesised fixtures. Always on,
//!   no corpus needed. Where THIS file is one fixture per named trap,
//!   Lane A is exhaustive over the published vocabulary.
//! * `tests/abr_corpus.rs` — LANE B: the real files, `#[ignore]`d and
//!   opt-in behind `PAGED_ABR_CORPUS=1`, and the analyst's to run.

use image_conformance::abr_builder::*;
use image_gpu::dab::{SampledTip, StrokeAccumulator};
use image_psd::abr::{AbrFile, AbrTip, AbrWarning, BlendMode, ControlSource};
use image_psd::descriptor::{read_descriptor, DescriptorValue};
use image_psd::reader::ByteReader;
use image_psd::vm_array::VmSamples;
use image_psd::PsdError;

const UUID_A: &str = "0b0c4a97-2b53-4d3a-9b1f-6f2c9a1d5e70";
const UUID_B: &str = "5f31a0e6-9c44-4f0b-8a2d-1e7b3c9d0a55";

/// A 4×3 tip: a diagonal ramp so a transposed or inverted read is
/// visible, and so no two rows are identical.
fn ramp_tip() -> Vec<u8> {
    let mut v = Vec::new();
    for y in 0..3u32 {
        for x in 0..4u32 {
            v.push((x * 40 + y * 60).min(255) as u8);
        }
    }
    v
}

// ── the descriptor value tree (§4.1) ─────────────────────────────────

#[test]
fn image_psd_descriptor_reads_every_framable_ostype() {
    let d = BDesc::new(klong("root"))
        .with_class_name("Root")
        .item(k4("bool"), BValue::Bool(true))
        .item(klong("integer"), BValue::Long(-42))
        .item(klong("large"), BValue::Comp(1 << 40))
        .item(klong("double"), BValue::Doub(1.5))
        .item(klong("percent"), BValue::UntF(*b"#Prc", 62.0))
        .item(klong("text"), BValue::Text("hello".into()))
        .item(klong("textNul"), BValue::TextNulTerminated("hello".into()))
        .item(klong("enumeration"), BValue::Enum(k4("BlnM"), k4("Nrml")))
        .item(klong("cls"), BValue::Class("Class".into(), k4("Ptrn")))
        .item(klong("alias"), BValue::Alis(vec![1, 2, 3]))
        .item(klong("data"), BValue::Tdta(vec![9, 8, 7, 6]))
        .item(
            klong("nested"),
            BValue::Obj(BDesc::new(k4("brVr")).item(k4("bVTy"), BValue::Long(2))),
        )
        .item(klong("global"), BValue::GlobalObj(BDesc::new(k4("null"))))
        .item(
            klong("mixedList"),
            // A list is NOT homogeneous by construction: each element
            // carries its own OSType (§4.1 trap).
            BValue::List(vec![
                BValue::Long(1),
                BValue::Text("two".into()),
                BValue::Bool(false),
                BValue::Doub(4.0),
            ]),
        );

    let bytes = descriptor_bytes(&d);
    let mut r = ByteReader::new(&bytes);
    let got = read_descriptor(&mut r, 0).unwrap();
    assert_eq!(r.remaining(), 0, "the tree must consume exactly its bytes");

    assert_eq!(got.class_name, "Root");
    assert!(got.class_id.matches(b"root"));
    assert_eq!(got.items.len(), 14);
    assert_eq!(got.bool(b"bool"), Some(true));
    assert_eq!(got.i32(b"integer"), Some(-42));
    assert_eq!(got.number(b"large"), Some((1i64 << 40) as f64));
    assert_eq!(got.number(b"double"), Some(1.5));
    assert_eq!(got.unit(b"percent", *b"#Prc"), Some(62.0));
    assert_eq!(got.text(b"text"), Some("hello"));
    // The trailing NUL some producers include is trimmed, so both
    // variants decode to the same string.
    assert_eq!(got.text(b"textNul"), Some("hello"));
    let (tk, vk) = got.enum_value(b"enumeration").unwrap();
    assert!(tk.matches(b"BlnM") && vk.matches(b"Nrml"));
    assert_eq!(got.raw_data(b"data"), Some(&[9u8, 8, 7, 6][..]));
    assert_eq!(got.raw_data(b"alias"), Some(&[1u8, 2, 3][..]));
    assert_eq!(
        got.descriptor(b"nested").and_then(|n| n.i32(b"bVTy")),
        Some(2)
    );
    assert!(got.descriptor(b"global").is_some(), "GlbO is a descriptor");
    let list = got.list(b"mixedList").unwrap();
    assert_eq!(list.len(), 4);
    assert_eq!(list[0].as_i32(), Some(1));
    assert_eq!(list[1].as_text(), Some("two"));
    assert_eq!(list[2].as_bool(), Some(false));
    assert_eq!(list[3].as_number(), Some(4.0));
    match &got.get(b"cls").unwrap() {
        DescriptorValue::Class { name, id } => {
            assert_eq!(name, "Class");
            assert!(id.matches(b"Ptrn"));
        }
        other => panic!("expected Class, got {other:?}"),
    }
}

#[test]
fn image_psd_descriptor_accepts_both_key_dialects_in_one_descriptor() {
    // 45,706 long-form item keys against 36,220 4-byte ones, in the same
    // files: a reader that handles one form fails on every fixture.
    let d = BDesc::new(k4("null"))
        .item(k4("Nm  "), BValue::Text("four".into()))
        .item(klong("Nm  "), BValue::Text("long".into()))
        .item(klong("sampledData"), BValue::Text(UUID_A.into()));
    let bytes = descriptor_bytes(&d);
    let mut r = ByteReader::new(&bytes);
    let got = read_descriptor(&mut r, 0).unwrap();
    assert_eq!(r.remaining(), 0);
    // Both spellings of `Nm  ` land on the same key; order decides which
    // `get` returns, and both are retained.
    assert_eq!(got.items.len(), 3);
    assert_eq!(got.text(b"Nm  "), Some("four"));
    assert_eq!(
        got.items.iter().filter(|(k, _)| k.matches(b"Nm  ")).count(),
        2
    );
    assert!(!got.items[0].0.is_long_form(), "first arrived as a 4cc");
    assert!(got.items[1].0.is_long_form(), "second arrived long-form");
    assert_eq!(got.text(b"sampledData"), Some(UUID_A));
}

#[test]
fn image_psd_descriptor_trailing_spaces_in_keys_are_significant() {
    let d = BDesc::new(k4("null"))
        .item(k4("Nm  "), BValue::Text("name".into()))
        .item(k4("Mnm "), BValue::Doub(0.0))
        .item(k4("H   "), BValue::Doub(1.0));
    let bytes = descriptor_bytes(&d);
    let got = read_descriptor(&mut ByteReader::new(&bytes), 0).unwrap();
    assert!(got.contains(b"Nm  ") && !got.contains(b"Nm"));
    assert!(got.contains(b"Mnm ") && !got.contains(b"Mnm"));
    assert!(got.contains(b"H   ") && !got.contains(b"H"));
}

#[test]
fn image_psd_descriptor_localisation_token_is_unwrapped_but_the_raw_survives() {
    let raw = "$$$/Presets/Patterns/Patterns_pat/GrayGranite=Gray Granite";
    let d = BDesc::new(k4("Ptrn")).item(k4("Nm  "), BValue::Text(raw.into()));
    let bytes = descriptor_bytes(&d);
    let got = read_descriptor(&mut ByteReader::new(&bytes), 0).unwrap();
    assert_eq!(got.text(b"Nm  "), Some(raw), "the raw value is preserved");
    assert_eq!(got.text_display(b"Nm  "), Some("Gray Granite"));
}

#[test]
fn image_psd_descriptor_unknown_ostype_is_refused_by_name_not_skipped() {
    // A descriptor item is not length-delimited, so a value the reader
    // cannot decode is a value it cannot step over either. Refusing by
    // name beats desynchronising.
    let d = BDesc::new(k4("null")).item(klong("ref"), BValue::Unknown(*b"obj ", vec![0; 8]));
    let bytes = descriptor_bytes(&d);
    match read_descriptor(&mut ByteReader::new(&bytes), 0).unwrap_err() {
        PsdError::Unsupported(m) => assert!(m.contains("obj "), "{m}"),
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

// ── the container and the section stream (§1) ────────────────────────

#[test]
fn image_abr_container_reads_a_file_with_every_section_type() {
    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp_tip()))
        .patt(vec![1, 2, 3, 4, 5])
        .brush(brush_preset("Tip A", sampled_tip("Tip A", UUID_A, 4.0)))
        .hierarchy(vec![
            BDesc::new(k4("Grup"))
                .item(k4("Nm  "), BValue::Text("Buildings".into()))
                .item(klong("zuid"), BValue::Text("folder-uuid".into())),
            BDesc::new(klong("preset")),
            BDesc::new(klong("groupEnd")),
        ])
        .build();

    let f = AbrFile::parse(&bytes).unwrap();
    assert_eq!((f.version, f.minor_version), (6, 2));
    assert_eq!(f.samples.len(), 1);
    assert_eq!(f.brushes.len(), 1);
    assert_eq!(f.hierarchy.len(), 3);
    assert_eq!(f.patterns_raw.as_deref(), Some(&[1u8, 2, 3, 4, 5][..]));
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);
}

#[test]
fn image_abr_container_survives_a_missing_pad_after_the_last_section() {
    // In 5 of 9 real files the file ends on the UNPADDED section end. A
    // loop that applies the pad and then reads a header walks past EOF
    // and reports a good file as corrupt.
    let padded = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp_tip()))
        .brush(brush_preset("Tip A", sampled_tip("Tip A", UUID_A, 4.0)))
        .pad_last_section(true)
        .build();
    let unpadded = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp_tip()))
        .brush(brush_preset("Tip A", sampled_tip("Tip A", UUID_A, 4.0)))
        .pad_last_section(false)
        .build();
    // The fixture must actually differ, or the test proves nothing.
    assert_ne!(padded.len(), unpadded.len(), "no pad was needed at all");

    for (label, bytes) in [("padded", padded), ("unpadded", unpadded)] {
        let f = AbrFile::parse(&bytes).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(f.brushes.len(), 1, "{label}");
        assert_eq!(f.samples.len(), 1, "{label}");
        assert!(f.warnings.is_empty(), "{label}: {:?}", f.warnings);
    }
}

#[test]
fn image_abr_container_empty_samp_and_patt_sections_are_normal() {
    let bytes = AbrBuilder::new()
        .empty_samp()
        .patt(Vec::new())
        .brush(brush_preset("Computed", computed_tip()))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    assert!(f.samples.is_empty());
    assert_eq!(
        f.patterns_raw, None,
        "a zero-length patt is nothing to keep"
    );
    assert_eq!(f.brushes.len(), 1);
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);
}

#[test]
fn image_abr_container_unknown_section_is_retained_not_fatal() {
    let bytes = AbrBuilder::new()
        .brush(brush_preset("Computed", computed_tip()))
        .extra_section(b"zzzz", vec![7; 9])
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    assert_eq!(f.brushes.len(), 1, "the brush list survived");
    assert_eq!(f.unknown_sections.len(), 1);
    assert_eq!(&f.unknown_sections[0].kind, b"zzzz");
    assert_eq!(f.unknown_sections[0].body, vec![7; 9]);
    assert!(f
        .warnings
        .iter()
        .any(|w| matches!(w, AbrWarning::UnknownSection { .. })));
}

#[test]
fn image_abr_container_accumulates_repeated_sections() {
    // Do not assume one `samp` and one `desc`.
    let second_desc = versioned_descriptor_bytes(
        16,
        &BDesc::new(k4("null")).item(
            k4("Brsh"),
            BValue::List(vec![BValue::Obj(brush_preset("Second", computed_tip()))]),
        ),
    );
    let bytes = AbrBuilder::new()
        .brush(brush_preset("First", computed_tip()))
        .extra_section(b"desc", second_desc)
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    assert_eq!(f.brushes.len(), 2);
    assert_eq!(f.brushes[0].name, "First");
    assert_eq!(f.brushes[1].name, "Second");
    assert_eq!((f.brushes[0].index, f.brushes[1].index), (0, 1));
}

#[test]
fn image_abr_container_version_10_parses_identically_to_6() {
    let make = |major: i16| {
        AbrBuilder::new()
            .version(major, 2)
            .sample(SampleSpec::new(UUID_A, 4, 3, ramp_tip()))
            .brush(brush_preset("Tip A", sampled_tip("Tip A", UUID_A, 4.0)))
            .build()
    };
    let six = AbrFile::parse(&make(6)).unwrap();
    let ten = AbrFile::parse(&make(10)).unwrap();
    assert_eq!(six.brushes, ten.brushes);
    assert_eq!(six.samples, ten.samples);
}

#[test]
fn image_abr_container_descriptor_version_other_than_16_warns_but_parses() {
    let bytes = AbrBuilder::new()
        .descriptor_version(17)
        .brush(brush_preset("Computed", computed_tip()))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    assert_eq!(f.brushes.len(), 1);
    assert!(f.warnings.iter().any(|w| matches!(
        w,
        AbrWarning::UnexpectedDescriptorVersion { version: 17, .. }
    )));
}

// ── sampled tips (§2) ────────────────────────────────────────────────

#[test]
fn image_abr_samp_record_is_a_vm_array_list_not_a_264_byte_skip() {
    // The real-file layout: array_count 56, plane in slot 55, whose
    // header measures exactly 264 bytes.
    let real = SampleSpec::new(UUID_A, 4, 3, ramp_tip()).with_layout(56, 55);
    // …and two layouts that break the constant while staying valid.
    let odd_slot = SampleSpec::new(UUID_A, 4, 3, ramp_tip()).with_layout(56, 7);
    let odd_count = SampleSpec::new(UUID_A, 4, 3, ramp_tip()).with_layout(3, 1);

    for (label, spec) in [
        ("array_count 56 / slot 55", real),
        ("array_count 56 / slot 7", odd_slot),
        ("array_count 3 / slot 1", odd_count),
    ] {
        let bytes = AbrBuilder::new().sample(spec).build();
        let f = AbrFile::parse(&bytes).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(f.samples.len(), 1, "{label}");
        assert_eq!(
            f.samples[0].coverage,
            VmSamples::Eight(ramp_tip()),
            "{label}"
        );
        // The structural self-check: a correct parse leaves NOTHING
        // unaccounted for, which is what makes the trailing two empty
        // slots stop being "8 mystery bytes".
        assert!(
            !f.warnings
                .iter()
                .any(|w| matches!(w, AbrWarning::SampleRecordTrailingBytes { .. })),
            "{label}: {:?}",
            f.warnings
        );
    }
}

#[test]
fn image_abr_samp_bounds_are_y_first_and_the_tip_is_not_transposed() {
    // 4 wide, 3 tall. Read x-first this would be 3×4.
    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp_tip()))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    let s = &f.samples[0];
    assert_eq!((s.width, s.height), (4, 3));
    assert_eq!((s.bounds.top, s.bounds.left), (0, 0));
    assert_eq!((s.bounds.bottom, s.bounds.right), (3, 4));
}

#[test]
fn image_abr_samp_origin_is_provenance_not_a_stamping_offset() {
    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp_tip()).at_origin(1731, 908))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    let s = &f.samples[0];
    assert_eq!(s.origin(), (1731, 908), "preserved…");
    // …and it changes nothing about the mask's size or content.
    assert_eq!((s.width, s.height), (4, 3));
    assert_eq!(s.coverage8(), ramp_tip());
}

#[test]
fn image_abr_samp_values_are_coverage_and_are_never_inverted() {
    // 255 = fully painted, 0 = no paint. The emitter wrote a known
    // pattern; the reader must return it byte-for-byte.
    let mut art = vec![0u8; 16];
    art[5] = 255;
    art[6] = 128;
    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 4, art.clone()))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    assert_eq!(f.samples[0].coverage8(), art);
    // The inverted reading, stated explicitly so the intent survives a
    // future refactor.
    let inverted: Vec<u8> = art.iter().map(|v| 255 - v).collect();
    assert_ne!(f.samples[0].coverage8(), inverted);
}

#[test]
fn image_abr_samp_rle_row_lengths_are_an_up_front_table() {
    // Rows that compress to DIFFERENT lengths: a reader that expects a
    // 2-byte prefix per row decodes row 0 and garbles the rest.
    let mut art = Vec::new();
    art.extend_from_slice(&[7u8; 8]); // one long run
    art.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]); // all literal
    art.extend_from_slice(&[9, 9, 9, 1, 2, 9, 9, 9]); // mixed
    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 8, 3, art.clone()).rle())
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    assert_eq!(f.samples[0].coverage8(), art);
    // The tip has no preset referring to it, so an OrphanSample notice
    // is the correct and only warning.
    assert!(
        f.warnings
            .iter()
            .all(|w| matches!(w, AbrWarning::OrphanSample { .. })),
        "{:?}",
        f.warnings
    );
}

#[test]
fn image_abr_samp_raw_and_rle_produce_identical_pixels() {
    let art = ramp_tip();
    let raw = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, art.clone()))
        .build();
    let rle = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, art.clone()).rle())
        .build();
    assert_ne!(raw, rle, "the two encodings must differ on the wire");
    assert_eq!(
        AbrFile::parse(&raw).unwrap().samples[0].coverage8(),
        AbrFile::parse(&rle).unwrap().samples[0].coverage8()
    );
}

#[test]
fn image_abr_samp_record_lengths_are_rounded_up_before_the_end_is_computed() {
    // Two records back to back whose bodies are not multiples of 4: the
    // second only starts in the right place if the first's declared
    // length was rounded UP.
    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 3, 3, vec![11u8; 9]))
        .sample(SampleSpec::new(UUID_B, 5, 1, vec![22u8; 5]))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    assert_eq!(f.samples.len(), 2);
    assert_eq!(f.samples[0].id, UUID_A);
    assert_eq!(f.samples[1].id, UUID_B);
    assert_eq!(f.samples[1].coverage8(), vec![22u8; 5]);
    // Both tips are unreferenced here; nothing else may be reported —
    // in particular no SampleRecordTrailingBytes, which is the §2.2
    // structural self-check.
    assert!(
        f.warnings
            .iter()
            .all(|w| matches!(w, AbrWarning::OrphanSample { .. })),
        "{:?}",
        f.warnings
    );
}

// ── the join (§3.3) ──────────────────────────────────────────────────

#[test]
fn image_abr_join_is_exact_string_equality() {
    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp_tip()))
        .sample(SampleSpec::new(UUID_B, 2, 2, vec![1, 2, 3, 4]))
        .brush(brush_preset("Second", sampled_tip("Second", UUID_B, 2.0)))
        .brush(brush_preset("First", sampled_tip("First", UUID_A, 4.0)))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    // Note the deliberate crossing: brush 0 refers to sample 1.
    assert_eq!(
        f.brushes[0].tip.as_sampled().unwrap().sample_index,
        Some(1),
        "resolved by id, not by position"
    );
    assert_eq!(f.brushes[1].tip.as_sampled().unwrap().sample_index, Some(0));
    assert_eq!(f.sample_for(&f.brushes[0]).unwrap().id, UUID_B);
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);
}

#[test]
fn image_abr_join_shares_one_sample_between_several_presets() {
    // Many-to-one: resolve by lookup, never by consuming as you go.
    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp_tip()))
        .brush(brush_preset("One", sampled_tip("One", UUID_A, 4.0)))
        .brush(brush_preset("Two", sampled_tip("Two", UUID_A, 8.0)))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    assert_eq!(f.samples.len(), 1);
    assert_eq!(f.brushes.len(), 2);
    for b in &f.brushes {
        assert_eq!(b.tip.as_sampled().unwrap().sample_index, Some(0));
    }
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);
}

#[test]
fn image_abr_join_has_no_positional_fallback() {
    // ONE sample, ONE brush — but the brush names an id that does not
    // exist. Positional pairing would "helpfully" bind them; that is
    // exactly the silent mis-pairing the spec forbids.
    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp_tip()))
        .brush(brush_preset("Ghost", sampled_tip("Ghost", UUID_B, 4.0)))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    assert_eq!(f.brushes[0].tip.as_sampled().unwrap().sample_index, None);
    assert!(f.sample_for(&f.brushes[0]).is_none());
    assert!(f.warnings.iter().any(|w| matches!(
        w,
        AbrWarning::UnresolvedSampleReference { brush_index: 0, .. }
    )));
    // …and the unreferenced sample is reported too, not silently kept.
    assert!(f
        .warnings
        .iter()
        .any(|w| matches!(w, AbrWarning::OrphanSample { .. })));
}

#[test]
fn image_abr_join_runs_over_the_dual_brush_tip_as_well() {
    let dual = BDesc::new(klong("dualBrush"))
        .item(klong("useDualBrush"), BValue::Bool(true))
        .item(k4("Brsh"), BValue::Obj(sampled_tip("Dual", UUID_B, 2.0)))
        .item(k4("Flip"), BValue::Bool(false))
        .item(k4("BlnM"), BValue::Enum(k4("BlnM"), k4("Mltp")))
        .item(klong("useScatter"), BValue::Bool(false))
        .item(k4("Spcn"), BValue::UntF(*b"#Prc", 25.0))
        .item(k4("Cnt "), BValue::Doub(2.0))
        .item(klong("bothAxes"), BValue::Bool(true));

    let mut preset = brush_preset("Main", sampled_tip("Main", UUID_A, 4.0));
    preset.items.retain(|(k, _)| !k.matches(b"dualBrush"));
    preset = preset.item(klong("dualBrush"), BValue::Obj(dual));

    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp_tip()))
        .sample(SampleSpec::new(UUID_B, 2, 2, vec![1, 2, 3, 4]))
        .brush(preset)
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    let b = &f.brushes[0];
    assert_eq!(b.sampled_ids(), vec![UUID_A, UUID_B]);
    assert_eq!(b.tip.as_sampled().unwrap().sample_index, Some(0));
    let db = b.dual_brush.as_ref().unwrap();
    assert!(db.enabled);
    assert_eq!(db.blend_mode, Some(BlendMode::Multiply));
    assert_eq!(db.count, Some(2.0));
    assert_eq!(
        db.tip.as_ref().unwrap().as_sampled().unwrap().sample_index,
        Some(1)
    );
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);
}

// ── tips (§5) ────────────────────────────────────────────────────────

fn computed_tip() -> BDesc {
    BDesc::new(klong("computedBrush"))
        .item(k4("Dmtr"), BValue::UntF(*b"#Pxl", 30.0))
        .item(k4("Angl"), BValue::UntF(*b"#Ang", 139.0))
        .item(k4("Spcn"), BValue::UntF(*b"#Prc", 25.0))
        .item(k4("Intr"), BValue::Bool(true))
        .item(klong("flipX"), BValue::Bool(false))
        .item(klong("flipY"), BValue::Bool(false))
        .item(k4("Rndn"), BValue::UntF(*b"#Prc", 60.0))
        .item(k4("Hrdn"), BValue::UntF(*b"#Prc", 0.0))
}

#[test]
fn image_abr_brush_model_tip_variant_is_selected_by_class_id_alone() {
    let bristle = BDesc::new(klong("dBrush"))
        .item(k4("Dmtr"), BValue::UntF(*b"#Pxl", 20.0))
        .item(k4("Angl"), BValue::UntF(*b"#Ang", 0.0))
        .item(k4("Spcn"), BValue::UntF(*b"#Prc", 2.0))
        .item(k4("Intr"), BValue::Bool(true))
        .item(klong("flipX"), BValue::Bool(false))
        .item(klong("flipY"), BValue::Bool(false))
        .item(k4("Shp "), BValue::Long(6))
        .item(k4("Dnst"), BValue::UntF(*b"#Prc", 0.27))
        .item(k4("Lngt"), BValue::UntF(*b"#Prc", 0.33))
        .item(klong("clumping"), BValue::UntF(*b"#Prc", 0.25))
        .item(klong("thickness"), BValue::UntF(*b"#Prc", 0.23))
        .item(klong("stiffness"), BValue::UntF(*b"#Prc", 0.80))
        .item(klong("physics"), BValue::Bool(true));

    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp_tip()))
        .brush(brush_preset("Computed", computed_tip()))
        .brush(brush_preset("Sampled", sampled_tip("Sampled", UUID_A, 4.0)))
        .brush(brush_preset("Flat Blunt Short Stiff", bristle))
        .brush(brush_preset(
            "Alien",
            BDesc::new(klong("noSuchTipClass"))
                .item(k4("Dmtr"), BValue::UntF(*b"#Pxl", 5.0))
                .item(k4("Angl"), BValue::UntF(*b"#Ang", 0.0))
                .item(k4("Spcn"), BValue::UntF(*b"#Prc", 25.0))
                .item(k4("Intr"), BValue::Bool(true))
                .item(klong("flipX"), BValue::Bool(false))
                .item(klong("flipY"), BValue::Bool(false)),
        ))
        .build();

    let f = AbrFile::parse(&bytes).unwrap();
    assert!(matches!(f.brushes[0].tip, AbrTip::Computed(_)));
    assert!(matches!(f.brushes[1].tip, AbrTip::Sampled(_)));
    assert!(matches!(f.brushes[2].tip, AbrTip::Bristle(_)));
    // An unrecognised tip class degrades; it does NOT poison the file.
    match &f.brushes[3].tip {
        AbrTip::Unsupported { class_id } => assert_eq!(class_id, "noSuchTipClass"),
        other => panic!("expected Unsupported, got {other:?}"),
    }
    assert_eq!(f.brushes[3].name, "Alien", "the preset is still metadata");
    assert!(f
        .warnings
        .iter()
        .any(|w| matches!(w, AbrWarning::UnsupportedTipClass { brush_index: 3, .. })));
}

#[test]
fn image_abr_brush_model_tip_bristle_percents_are_fractions_while_siblings_are_0_to_100() {
    // Stock `Flat Blunt Short Stiff` values, exactly as they sit in a
    // real file. Divided by 100 every one falls below its documented
    // minimum — this is the §4.2 CONTRADICTION.
    let bristle = BDesc::new(klong("dBrush"))
        .item(k4("Dmtr"), BValue::UntF(*b"#Pxl", 20.0))
        .item(k4("Angl"), BValue::UntF(*b"#Ang", 0.0))
        .item(k4("Spcn"), BValue::UntF(*b"#Prc", 2.0))
        .item(k4("Intr"), BValue::Bool(true))
        .item(klong("flipX"), BValue::Bool(false))
        .item(klong("flipY"), BValue::Bool(false))
        .item(k4("Shp "), BValue::Long(6))
        .item(k4("Dnst"), BValue::UntF(*b"#Prc", 0.27))
        .item(k4("Lngt"), BValue::UntF(*b"#Prc", 1.37))
        .item(klong("clumping"), BValue::UntF(*b"#Prc", 0.25))
        .item(klong("thickness"), BValue::UntF(*b"#Prc", 0.23))
        .item(klong("stiffness"), BValue::UntF(*b"#Prc", 0.80))
        .item(klong("physics"), BValue::Bool(true));
    let bytes = AbrBuilder::new()
        .brush(brush_preset("Flat Blunt Short Stiff", bristle))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    let AbrTip::Bristle(t) = &f.brushes[0].tip else {
        panic!("expected a bristle tip");
    };
    // Fractions, NOT divided again.
    assert_eq!(t.density, 0.27);
    assert_eq!(t.length, 1.37, "137% length — above 1.0 and correct");
    assert_eq!(t.clumping, 0.25);
    assert_eq!(t.thickness, 0.23);
    assert_eq!(t.stiffness, 0.80);
    // …while the shape-level spacing beside them IS on the 0..100 scale.
    assert_eq!(t.common.spacing, 0.02);
    // The name-corroborated ordinal.
    assert_eq!(t.shape_ordinal, 6);
    assert_eq!(
        t.shape,
        Some(image_psd::abr::BristleShape::FlatBlunt),
        "index 6 is `flat blunt`, and the preset says so"
    );
}

#[test]
fn image_abr_brush_model_tip_computed_carries_roundness_and_hardness_as_0_to_100() {
    let bytes = AbrBuilder::new()
        .brush(brush_preset("Soft Round", computed_tip()))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    let AbrTip::Computed(t) = &f.brushes[0].tip else {
        panic!("expected a computed tip");
    };
    assert_eq!(t.roundness, 0.6);
    assert_eq!(t.hardness, 0.0, "a `Soft Round` really is Hrdn 0");
    assert_eq!(t.common.angle_deg, 139.0);
    assert_eq!(t.common.diameter, 30.0);
    assert_eq!(&t.common.diameter_unit, b"#Pxl");
}

#[test]
fn image_abr_height_map_erodible_tip_retains_both_ordinals_and_the_le_map() {
    // `Square Charcoal`: Shp 3, dtipsType 0, grid 11.
    let values: Vec<f32> = (0..25).map(|i| i as f32 / 24.0).collect();
    let erodible = BDesc::new(klong("dTips"))
        .item(k4("Dmtr"), BValue::UntF(*b"#Pxl", 40.0))
        .item(k4("Angl"), BValue::UntF(*b"#Ang", 0.0))
        .item(k4("Spcn"), BValue::UntF(*b"#Prc", 25.0))
        .item(k4("Intr"), BValue::Bool(true))
        .item(klong("flipX"), BValue::Bool(false))
        .item(klong("flipY"), BValue::Bool(false))
        .item(k4("Shp "), BValue::Long(3))
        .item(klong("physics"), BValue::Bool(true))
        .item(klong("dtipsType"), BValue::Long(0))
        .item(klong("dtipsLengthRatio"), BValue::UntF(*b"#Prc", 100.0))
        .item(klong("dtipsHardness"), BValue::UntF(*b"#Prc", 48.0))
        .item(klong("dtipsGridSize"), BValue::Long(5))
        .item(
            klong("dtipsErodibleTipHeightMap"),
            BValue::Tdta(height_map_tdta(&values)),
        )
        // A bare `doub`, NOT a #Ang unit float — applying the angle rule
        // here would reject the file.
        .item(klong("dtipsAirbrushCutoffAngle"), BValue::Doub(15.0))
        .item(
            klong("dtipsAirbrushGranularity"),
            BValue::UntF(*b"#Prc", 0.0),
        )
        .item(
            klong("dtipsAirbrushStreakiness"),
            BValue::UntF(*b"#Prc", 1.0),
        )
        .item(klong("dtipsAirbrushSplatSize"), BValue::UntF(*b"#Prc", 1.0))
        .item(klong("dtipsAirbrushSplatCount"), BValue::Long(200));

    let bytes = AbrBuilder::new()
        .brush(brush_preset("Square Charcoal", erodible))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    let AbrTip::Erodible(t) = &f.brushes[0].tip else {
        panic!("expected an erodible tip");
    };
    // Both integers retained; NEITHER resolved to a name, because which
    // table each indexes is still undetermined.
    assert_eq!(t.shape_ordinal, 3);
    assert_eq!(t.tips_type, Some(0));
    assert_eq!(t.grid_size, Some(5));
    assert_eq!(t.hardness, Some(0.48));
    let hm = t.height_map.as_ref().unwrap();
    assert_eq!(hm.grid_size, 5);
    assert_eq!(hm.values.len(), 25);
    for (got, want) in hm.values.iter().zip(values.iter()) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
    assert_eq!(hm.raw.len(), 100, "gridSize² × 4 bytes");
    // The airbrush scale is undetermined, so the values are RAW.
    assert_eq!(t.airbrush.cutoff_angle_deg, Some(15.0));
    assert_eq!(t.airbrush.granularity_raw, Some(0.0));
    assert_eq!(t.airbrush.streakiness_raw, Some(1.0));
    assert_eq!(t.airbrush.splat_count, Some(200));
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);
}

// ── dynamics (§6) ────────────────────────────────────────────────────

#[test]
fn image_abr_brush_model_scatter_count_is_a_double_and_reading_four_bytes_would_desync() {
    // `Cnt ` as an 8-byte doub, with keys AFTER it. A 4-byte read
    // corrupts the siblings rather than just this field — which is what
    // makes it the most destructive type error in the format.
    let mut preset = brush_preset("Scattered", computed_tip());
    preset.items.retain(|(k, _)| !k.matches(b"useScatter"));
    preset = preset
        .item(klong("useScatter"), BValue::Bool(true))
        .item(
            klong("scatterDynamics"),
            BValue::Obj(dynamics(2, 25, 368.0, None)),
        )
        .item(
            klong("countDynamics"),
            BValue::Obj(dynamics(0, 25, 0.0, Some(0.0))),
        )
        .item(k4("Cnt "), BValue::Doub(5.0))
        .item(klong("bothAxes"), BValue::Bool(true))
        // The canaries: if `Cnt ` were read as 4 bytes, everything from
        // here on is garbage.
        .item(klong("canaryText"), BValue::Text("intact".into()))
        .item(klong("canaryLong"), BValue::Long(1234));

    let bytes = AbrBuilder::new().brush(preset).build();
    let f = AbrFile::parse(&bytes).unwrap();
    let sc = f.brushes[0].scatter.as_ref().unwrap();
    assert_eq!(sc.count, Some(5.0));
    assert_eq!(sc.both_axes, Some(true));
    let d = sc.scatter.as_ref().unwrap();
    assert_eq!(d.control, ControlSource::PenPressure);
    assert_eq!(d.fade_steps, 25);
    assert!((d.jitter - 3.68).abs() < 1e-12, "368% jitter, unclamped");
    // Jitter and the control source are independent: `off` with jitter.
    let cd = sc.count_dynamics.as_ref().unwrap();
    assert_eq!(cd.control, ControlSource::Off);
    assert_eq!(cd.minimum, Some(0.0));
    // The canaries survived — proof the double did not desynchronise.
    assert_eq!(
        f.brushes[0].descriptor.text(b"canaryText"),
        Some("intact"),
        "a 4-byte read of `Cnt ` would have wrecked this"
    );
    assert_eq!(f.brushes[0].descriptor.i32(b"canaryLong"), Some(1234));
}

#[test]
fn image_abr_brush_model_gate_false_means_the_group_is_absent_entirely() {
    let bytes = AbrBuilder::new()
        .brush(brush_preset("Plain", computed_tip()))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    let b = &f.brushes[0];
    assert!(b.shape_dynamics.is_none());
    assert!(b.scatter.is_none());
    assert!(b.texture.is_none());
    assert!(b.color_dynamics.is_none());
    assert!(b.transfer.is_none());
    assert!(b.brush_pose.is_none());
    // dualBrush is present on EVERY brush; only its inner gate speaks.
    assert!(!b.dual_brush.as_ref().unwrap().enabled);
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);
}

#[test]
fn image_abr_brush_model_flip_flags_do_not_collide_across_the_two_levels() {
    // Shape-level flipX = static mirror ON; brush-level flipX = Flip X
    // Jitter OFF. A reader that merges the namespaces picks up the wrong
    // one on exactly the brushes that have both.
    let mut tip = computed_tip();
    tip.items.retain(|(k, _)| !k.matches(b"flipX"));
    tip = tip.item(klong("flipX"), BValue::Bool(true));

    let mut preset = brush_preset("Both levels", tip);
    preset.items.retain(|(k, _)| !k.matches(b"useTipDynamics"));
    preset = preset
        .item(klong("useTipDynamics"), BValue::Bool(true))
        .item(k4("szVr"), BValue::Obj(dynamics(2, 25, 0.0, Some(0.0))))
        .item(
            klong("angleDynamics"),
            BValue::Obj(dynamics(0, 25, 10.0, None)),
        )
        .item(
            klong("roundnessDynamics"),
            BValue::Obj(dynamics(0, 25, 0.0, None)),
        )
        .item(klong("minimumDiameter"), BValue::UntF(*b"#Prc", 5.0))
        .item(klong("minimumRoundness"), BValue::UntF(*b"#Prc", 25.0))
        .item(klong("tiltScale"), BValue::UntF(*b"#Prc", 200.0))
        .item(klong("flipX"), BValue::Bool(false))
        .item(klong("flipY"), BValue::Bool(true))
        .item(klong("brushProjection"), BValue::Bool(false));

    let bytes = AbrBuilder::new().brush(preset).build();
    let f = AbrFile::parse(&bytes).unwrap();
    let b = &f.brushes[0];
    assert!(b.tip.common().unwrap().flip_x, "shape-level mirror is ON");
    assert!(!b.tip.common().unwrap().flip_y);
    let sd = b.shape_dynamics.as_ref().unwrap();
    assert_eq!(sd.flip_x_jitter, Some(false), "brush-level JITTER is OFF");
    assert_eq!(sd.flip_y_jitter, Some(true));
    assert_eq!(sd.tilt_scale, Some(2.0), "200% is not clamped to 1.0");
    assert_eq!(sd.minimum_roundness, Some(0.25));
    assert_eq!(
        sd.size.as_ref().unwrap().control,
        ControlSource::PenPressure
    );
}

#[test]
fn image_abr_brush_model_texture_identity_nests_while_its_parameters_do_not() {
    let mut preset = brush_preset("Textured", computed_tip());
    preset.items.retain(|(k, _)| !k.matches(b"useTexture"));
    preset = preset
        .item(klong("useTexture"), BValue::Bool(true))
        .item(
            k4("Txtr"),
            BValue::Obj(
                BDesc::new(k4("Ptrn"))
                    .item(k4("Idnt"), BValue::Text("pattern-uuid".into()))
                    .item(
                        k4("Nm  "),
                        BValue::Text(
                            "$$$/Presets/Patterns/Patterns_pat/GrayGranite=Gray Granite".into(),
                        ),
                    ),
            ),
        )
        // …every other texture parameter sits on the PARENT.
        .item(
            klong("textureBlendMode"),
            BValue::Enum(k4("BlnM"), klong("linearHeight")),
        )
        .item(klong("textureScale"), BValue::UntF(*b"#Prc", 62.0))
        .item(klong("textureDepth"), BValue::UntF(*b"#Prc", 100.0))
        .item(klong("minimumDepth"), BValue::UntF(*b"#Prc", 0.0))
        .item(
            klong("textureDepthDynamics"),
            BValue::Obj(dynamics(0, 25, 0.0, None)),
        )
        .item(k4("InvT"), BValue::Bool(false))
        .item(klong("textureBrightness"), BValue::Long(-7))
        .item(klong("textureContrast"), BValue::Long(14))
        .item(k4("TxtC"), BValue::Bool(true));

    let bytes = AbrBuilder::new().brush(preset).build();
    let f = AbrFile::parse(&bytes).unwrap();
    let t = f.brushes[0].texture.as_ref().unwrap();
    assert_eq!(t.pattern_id.as_deref(), Some("pattern-uuid"));
    assert_eq!(t.pattern_name.as_deref(), Some("Gray Granite"));
    assert!(t.pattern_name_raw.as_deref().unwrap().starts_with("$$$/"));
    assert_eq!(t.blend_mode, Some(BlendMode::LinearHeight));
    assert_eq!(t.scale, Some(0.62));
    assert_eq!(t.depth, Some(1.0));
    // Signed and centred on zero: an unsigned store loses half the range.
    assert_eq!(t.brightness, Some(-7));
    assert_eq!(t.contrast, Some(14));
    assert_eq!(t.per_tip, Some(true));
}

#[test]
fn image_abr_brush_model_blend_modes_arrive_in_both_dialects_in_one_file() {
    let textured = |mode: BValue| {
        let mut p = brush_preset("T", computed_tip());
        p.items.retain(|(k, _)| !k.matches(b"useTexture"));
        p.item(klong("useTexture"), BValue::Bool(true))
            .item(klong("textureBlendMode"), mode)
    };
    let bytes = AbrBuilder::new()
        .brush(textured(BValue::Enum(k4("BlnM"), k4("Sbtr"))))
        .brush(textured(BValue::Enum(k4("BlnM"), klong("linearHeight"))))
        .brush(textured(BValue::Enum(k4("BlnM"), klong("neverHeardOfIt"))))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    let mode = |i: usize| f.brushes[i].texture.as_ref().unwrap().blend_mode;
    assert_eq!(mode(0), Some(BlendMode::SubtractionTexture), "4cc form");
    assert_eq!(mode(1), Some(BlendMode::LinearHeight), "long form");
    // An unknown identifier substitutes Normal and is REPORTED.
    assert_eq!(mode(2), Some(BlendMode::Normal));
    assert!(f
        .warnings
        .iter()
        .any(|w| matches!(w, AbrWarning::UnrecognisedBlendMode { .. })));
}

#[test]
fn image_abr_brush_model_color_dynamics_parses_but_announces_that_it_is_unverified() {
    let mut preset = brush_preset("Colourful", computed_tip());
    preset
        .items
        .retain(|(k, _)| !k.matches(b"useColorDynamics"));
    preset = preset
        .item(klong("useColorDynamics"), BValue::Bool(true))
        .item(k4("clVr"), BValue::Obj(dynamics(2, 25, 20.0, None)))
        .item(k4("H   "), BValue::UntF(*b"#Prc", 30.0))
        .item(k4("Strt"), BValue::UntF(*b"#Prc", 40.0))
        .item(k4("Brgh"), BValue::UntF(*b"#Prc", 50.0))
        .item(klong("purity"), BValue::UntF(*b"#Prc", 60.0))
        .item(klong("colorDynamicsPerTip"), BValue::Bool(true));
    let bytes = AbrBuilder::new().brush(preset).build();
    let f = AbrFile::parse(&bytes).unwrap();
    let cd = f.brushes[0].color_dynamics.as_ref().unwrap();
    assert_eq!(cd.hue, Some(0.30));
    assert_eq!(cd.saturation, Some(0.40));
    assert_eq!(cd.per_tip, Some(true));
    // The whole group rests on a single third-party channel — the gate
    // was false on all 3,215 corpus brushes, so nothing here has ever
    // been seen in a real file. Say so, loudly.
    assert!(f
        .warnings
        .iter()
        .any(|w| matches!(w, AbrWarning::ColorDynamicsUnverified { brush_index: 0 })));
}

#[test]
fn image_abr_brush_model_transfer_is_gated_by_use_paint_dynamics() {
    let mut preset = brush_preset("Transfer", computed_tip());
    preset
        .items
        .retain(|(k, _)| !k.matches(b"usePaintDynamics"));
    preset = preset
        .item(klong("usePaintDynamics"), BValue::Bool(true))
        .item(k4("prVr"), BValue::Obj(dynamics(2, 25, 0.0, Some(0.0))))
        .item(k4("opVr"), BValue::Obj(dynamics(0, 25, 0.0, None)));
    let bytes = AbrBuilder::new().brush(preset).build();
    let f = AbrFile::parse(&bytes).unwrap();
    let t = f.brushes[0].transfer.as_ref().unwrap();
    assert!(t.flow.is_some(), "prVr is FLOW, not pressure");
    assert!(t.opacity.is_some());
    // Mixer-only keys are simply absent for a non-mixer brush.
    assert!(t.wetness.is_none() && t.mix.is_none());
}

#[test]
fn image_abr_brush_model_brush_pose_angles_are_bare_longs() {
    let mut preset = brush_preset("Posed", computed_tip());
    preset.items.retain(|(k, _)| !k.matches(b"useBrushPose"));
    preset = preset
        .item(klong("useBrushPose"), BValue::Bool(true))
        .item(klong("overridePoseAngle"), BValue::Bool(false))
        .item(klong("overridePoseTiltX"), BValue::Bool(true))
        .item(klong("overridePoseTiltY"), BValue::Bool(true))
        .item(klong("overridePosePressure"), BValue::Bool(true))
        .item(klong("brushPosePressure"), BValue::UntF(*b"#Prc", 9.0))
        .item(klong("brushPoseTiltX"), BValue::Long(-54))
        .item(klong("brushPoseTiltY"), BValue::Long(23))
        .item(klong("brushPoseAngle"), BValue::Long(0));
    let bytes = AbrBuilder::new().brush(preset).build();
    let f = AbrFile::parse(&bytes).unwrap();
    let p = f.brushes[0].brush_pose.as_ref().unwrap();
    assert_eq!(p.tilt_x, Some(-54), "degrees, not a normalised −1..1");
    assert_eq!(p.tilt_y, Some(23));
    assert_eq!(p.angle, Some(0));
    assert_eq!(p.pressure, Some(0.09), "9% on the 0..100 scale");
    // The override booleans and the values are independent.
    assert_eq!(p.override_angle, Some(false));
}

#[test]
fn image_abr_brush_model_tool_options_class_id_names_the_tool_and_unknown_ids_fall_back() {
    let opts = |class: &str| {
        BDesc::new(k4(class))
            .item(klong("brushPreset"), BValue::Bool(true))
            .item(klong("flow"), BValue::Long(100))
            .item(k4("Opct"), BValue::Long(100))
            .item(k4("Md  "), BValue::Enum(k4("BlnM"), k4("Nrml")))
            .item(klong("smoothingValue"), BValue::Doub(0.0))
            .item(k4("Smoo"), BValue::Long(0))
    };
    let with = |class: &str| {
        brush_preset("Tooled", computed_tip()).item(klong("toolOptions"), BValue::Obj(opts(class)))
    };
    let bytes = AbrBuilder::new()
        .brush(with("PbTl"))
        .brush(with("MixB"))
        .brush(with("zzzz"))
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    let kind = |i: usize| f.brushes[i].tool_options.as_ref().unwrap().kind;
    assert_eq!(kind(0), image_psd::abr::ToolKind::PlainBrush, "PbTl");
    assert_eq!(kind(1), image_psd::abr::ToolKind::Mixer);
    assert_eq!(kind(2), image_psd::abr::ToolKind::PlainBrush, "fallback");
    let t = f.brushes[0].tool_options.as_ref().unwrap();
    assert_eq!(t.flow, 100.0, "a 0..100 plain number, not a #Prc float");
    assert_eq!(t.opacity, 100.0);
    assert_eq!(t.blend_mode, BlendMode::Normal);
    // The ~25 keys this view deliberately does not model are still there.
    assert_eq!(t.descriptor.number(b"smoothingValue"), Some(0.0));
    assert_eq!(t.descriptor.i32(b"Smoo"), Some(0));
}

// ── generic modelling (§7.1) ─────────────────────────────────────────

#[test]
fn image_abr_brush_model_unmodelled_keys_survive_on_the_descriptor() {
    // `Rpt` and `brushGroup` are on every brush in every real file and
    // were missing from every published key vocabulary. A fixed struct
    // would have dropped them.
    let bytes = AbrBuilder::new()
        .brush(
            brush_preset("Keys", computed_tip())
                .item(klong("interpretation"), BValue::Bool(true))
                .item(klong("aKeyFromTheFuture"), BValue::Long(7)),
        )
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    let d = &f.brushes[0].descriptor;
    assert_eq!(d.bool(b"Rpt"), Some(true));
    assert_eq!(
        d.descriptor(b"brushGroup")
            .and_then(|g| g.bool(b"useBrushGroup")),
        Some(false)
    );
    assert_eq!(d.bool(b"interpretation"), Some(true));
    assert_eq!(d.i32(b"aKeyFromTheFuture"), Some(7));
    assert_eq!(f.brushes[0].wet_edges, Some(false), "Wtdg is wet edges");
    assert_eq!(f.brushes[0].noise, Some(false), "Nose is noise");
}

// ── the folder hierarchy (§8.2) ──────────────────────────────────────

#[test]
fn image_abr_phry_is_a_flat_open_close_token_stream_bound_positionally() {
    let nodes = vec![
        BDesc::new(k4("Grup"))
            .item(k4("Nm  "), BValue::Text("Buildings".into()))
            .item(klong("zuid"), BValue::Text("uid-1".into())),
        BDesc::new(klong("preset")),
        BDesc::new(k4("Grup")).item(k4("Nm  "), BValue::Text("Small Towns".into())),
        BDesc::new(klong("preset")),
        BDesc::new(klong("groupEnd")),
        BDesc::new(klong("groupEnd")),
        BDesc::new(klong("preset")),
    ];
    let bytes = AbrBuilder::new()
        .brush(brush_preset("A", computed_tip()))
        .brush(brush_preset("B", computed_tip()))
        .brush(brush_preset("C", computed_tip()))
        .hierarchy(nodes)
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    use image_psd::abr::HierarchyNode as N;
    assert_eq!(f.hierarchy.len(), 7);
    match &f.hierarchy[0] {
        N::GroupOpen { name, uid } => {
            assert_eq!(name, "Buildings");
            assert_eq!(uid.as_deref(), Some("uid-1"));
        }
        other => panic!("{other:?}"),
    }
    let bound: Vec<_> = f
        .hierarchy
        .iter()
        .filter_map(|n| match n {
            N::Preset { brush_index } => Some(*brush_index),
            _ => None,
        })
        .collect();
    assert_eq!(bound, vec![Some(0), Some(1), Some(2)]);
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);
}

#[test]
fn image_abr_phry_count_mismatch_refuses_to_bind() {
    let bytes = AbrBuilder::new()
        .brush(brush_preset("A", computed_tip()))
        .brush(brush_preset("B", computed_tip()))
        .hierarchy(vec![BDesc::new(klong("preset"))])
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    assert!(f.warnings.iter().any(|w| matches!(
        w,
        AbrWarning::HierarchyCountMismatch {
            presets: 1,
            brushes: 2
        }
    )));
    assert_eq!(f.brushes.len(), 2, "the brushes are still all there");
}

#[test]
fn image_abr_phry_present_but_empty_does_not_hide_the_brushes() {
    let bytes = AbrBuilder::new()
        .brush(brush_preset("A", computed_tip()))
        .hierarchy(Vec::new())
        .build();
    let f = AbrFile::parse(&bytes).unwrap();
    assert!(f.hierarchy.is_empty());
    assert_eq!(f.brushes.len(), 1);
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);
}

// ── the brush-engine bridge ──────────────────────────────────────────

#[test]
fn image_abr_engine_bridge_sampled_tip_becomes_a_stampable_coverage_field() {
    // The whole point of the lane: a .abr sampled tip is the alpha
    // bitmap the engine paints with.
    let mut art = vec![0u8; 8 * 8];
    for y in 2..6 {
        for x in 2..6 {
            art[y * 8 + x] = 255;
        }
    }
    let bytes = AbrBuilder::new()
        .sample(
            SampleSpec::new(UUID_A, 8, 8, art.clone())
                .rle()
                .at_origin(400, 900),
        )
        .brush(brush_preset("Blob", sampled_tip("Blob", UUID_A, 16.0)))
        .build();

    let f = AbrFile::parse(&bytes).unwrap();
    let brush = &f.brushes[0];
    let sample = f.sample_for(brush).unwrap();
    assert_eq!(sample.coverage8(), art, "decoded through RLE, uninverted");

    let tip = brush.tip.as_sampled().unwrap();
    let engine_tip = SampledTip::new(sample.width, sample.height, sample.coverage8())
        .unwrap()
        // Dmtr is a TARGET diameter: 16 px from an 8 px bitmap ⇒ 2×.
        .with_diameter(tip.common.diameter as f32)
        .with_roundness(tip.roundness as f32)
        .with_flips(tip.common.flip_x, tip.common.flip_y);
    assert_eq!(engine_tip.diameter(), 16.0);

    let mut acc = StrokeAccumulator::new(64, 64);
    assert!(acc.stamp(&engine_tip, 32.0, 32.0, 1.0));
    // The painted square is 4 of 8 texels wide, doubled ⇒ 8 px across,
    // centred: solid at the centre, empty well outside.
    assert!(acc.value_at(32, 32) > 0.99, "centre painted");
    assert!(acc.value_at(34, 34) > 0.9, "inside the square");
    assert_eq!(acc.value_at(32, 45), 0.0, "outside the tip entirely");
    // The provenance origin did NOT displace the stamp.
    let b = acc.stroke_bounds().unwrap();
    assert!(b.x < 32 && b.right() > 32, "stamped at the cursor: {b:?}");
}
