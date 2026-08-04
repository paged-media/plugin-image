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

//! **LANE A** — the always-on `.abr` regression gate (spec §14.3.1).
//!
//! IMPLEMENTER-owned, ordinary CI, and it **needs no corpus byte**.
//!
//! The nine licensed `.abr` files that the behaviour spec was verified
//! against live inside `references/`, which the clean-room protocol
//! (repo `CLAUDE.md` §3.1) forbids an implementer from reading. Revision
//! 2 of the spec nonetheless asked the implementer to wire that corpus
//! into the suite; that instruction had no owner and was reported back as
//! errata item 8. The resolution splits the gate in two: an ANALYST reads
//! the mount and publishes *facts about behaviour* — counts, key/OSType/
//! unit tables, dimensions, one-way digests — and those facts drive a
//! lane built entirely from **synthesised** fixtures.
//!
//! So every expectation here was measured against 3,215 real presets and
//! 3,202 real tip records, while every byte fed to the reader comes from
//! [`image_conformance::abr_builder`], the INDEPENDENT emitter that
//! shares no code with the reader. Lane B — the one that opens the real
//! files — is `tests/abr_corpus.rs`, is `#[ignore]`d, and is the
//! analyst's.
//!
//! The seven gates, and what each actually asserts:
//!
//! | # | Gate |
//! |---|---|
//! | A1 | Every `key × OSType` and `key × unit` pair the corpus contains survives the reader with its type and unit intact — nothing is silently dropped. |
//! | A2 | Every observed class id is either dispatched to a typed model or degraded with a NAMED diagnostic; none costs the file. |
//! | A3 | Every observed ordinal parses; out-of-range degrades instead of indexing a table that exists nowhere in the file. |
//! | A4 | Every container shape the nine files contain between them parses — both section orders, empty sections, the `size ≡ 2 (mod 4)` case, and both last-section terminations. |
//! | A5 | Gate discipline, including the ABSENT gate: a missing gate reads as `false` and its group is not read. |
//! | A6 | The §2.2 self-check is measured against the DECLARED extent, for all four pad remainders — and it fires when a record genuinely does not reconcile. |
//! | A7 | The join is exact, case-sensitive string equality, many-to-one, and reports rather than guesses. |
//!
//! **Ownership.** `fixtures/abr/corpus-profile.json` and
//! `corpus-record-ledger.tsv` are ANALYST artifacts. A failure here is a
//! reader regression or a genuine change in the format understanding —
//! it is **never** fixed by editing an expectation file.

use std::collections::{BTreeMap, BTreeSet};

use image_conformance::abr_builder::*;
use image_conformance::abr_corpus::{ledger, profile};
use image_psd::abr::{AbrBrush, AbrFile, AbrTip, AbrWarning, ControlSource};

const UUID_A: &str = "0b0c4a97-2b53-4d3a-9b1f-6f2c9a1d5e70";
const UUID_B: &str = "5f31a0e6-9c44-4f0b-8a2d-1e7b3c9d0a55";

/// A tip whose rows all differ, so a transposed or inverted read shows.
fn ramp(w: usize, h: usize) -> Vec<u8> {
    (0..w * h).map(|i| (i * 7 % 251) as u8).collect()
}

/// A key in the dialect a fixture wants: `force_long` writes the
/// length-prefixed form, which is legal for ANY key including a
/// four-character one, and is what modern Photoshop increasingly does
/// (45,706 long-form item keys against 36,220 4-byte ones, in the same
/// files).
fn bkey(text: &str, force_long: bool) -> BKey {
    if force_long || text.len() != 4 {
        klong(text)
    } else {
        k4(text)
    }
}

fn cc(code: [u8; 4]) -> String {
    String::from_utf8_lossy(&code).into_owned()
}

/// A representative value of `ostype`, carrying `unit` when the OSType
/// is a unit float. Values are arbitrary; what is under test is that the
/// pair survives the round trip through the reader.
fn probe_value(ostype: [u8; 4], unit: Option<[u8; 4]>) -> BValue {
    match &ostype {
        b"Objc" => BValue::Obj(BDesc::new(k4("null"))),
        b"GlbO" => BValue::GlobalObj(BDesc::new(k4("null"))),
        b"VlLs" => BValue::List(vec![BValue::Bool(true), BValue::Long(2)]),
        b"doub" => BValue::Doub(1.25),
        b"UntF" => BValue::UntF(unit.unwrap_or(*b"#Prc"), 62.0),
        b"TEXT" => BValue::Text("probe".into()),
        b"enum" => BValue::Enum(k4("BlnM"), k4("Nrml")),
        b"long" => BValue::Long(-7),
        b"comp" => BValue::Comp(1 << 40),
        b"bool" => BValue::Bool(true),
        b"type" => BValue::Class("Class".into(), k4("Ptrn")),
        b"alis" => BValue::Alis(vec![1, 2, 3]),
        // Little-endian float32 1.0 — a `tdta` interior follows the
        // producing subsystem's convention, not PSD's big-endian rule.
        b"tdta" => BValue::Tdta(height_map_tdta(&[1.0])),
        other => panic!(
            "the profile carries OSType `{}`, which this fixture builder cannot emit — \
             extend abr_builder rather than narrowing the gate",
            cc(*other)
        ),
    }
}

/// A brush preset carrying exactly one item: the probe.
fn probe_brush(key: BKey, value: BValue) -> BDesc {
    BDesc::new(klong("brushPreset")).item(key, value)
}

fn parse_one(bytes: &[u8], ctx: &str) -> AbrFile {
    AbrFile::parse(bytes).unwrap_or_else(|e| panic!("{ctx}: the file must parse, got {e}"))
}

fn only_brush(f: &AbrFile, ctx: &str) -> AbrBrush {
    assert_eq!(f.brushes.len(), 1, "{ctx}: expected exactly one preset");
    f.brushes[0].clone()
}

/// Remove every occurrence of `key` from a descriptor.
fn without(mut d: BDesc, key: &[u8]) -> BDesc {
    d.items.retain(|(k, _)| !k.matches(key));
    d
}

/// Replace (or add) one item.
fn with(d: BDesc, key: BKey, value: BValue) -> BDesc {
    let text = key.text.clone();
    without(d, text.as_bytes()).item(key, value)
}

// ── an independent walk of the emitted container ─────────────────────
//
// The reader is what is under test, so the fixtures' own shape is
// checked with a second, deliberately naive walker: it re-reads the
// section list from the bytes rather than trusting the builder.

#[derive(Debug, PartialEq, Eq)]
struct Emitted {
    sections: Vec<(String, usize)>,
    /// `Some(true)` when the file ends ON the pad after its last
    /// section, `Some(false)` when it ends on the unpadded section end,
    /// and **`None` when the last section's size is already a multiple
    /// of 4** — the two terminations then produce identical bytes and
    /// the question is not answerable from the file.
    ends_on_pad: Option<bool>,
}

fn walk(bytes: &[u8]) -> Emitted {
    let mut i = 4usize; // the two version words
    let mut sections = Vec::new();
    let mut ends_on_pad = None;
    while i + 12 <= bytes.len() {
        assert_eq!(&bytes[i..i + 4], b"8BIM", "section signature at {i}");
        let kind = String::from_utf8_lossy(&bytes[i + 4..i + 8]).into_owned();
        let size = u32::from_be_bytes(bytes[i + 8..i + 12].try_into().unwrap()) as usize;
        sections.push((kind, size));
        i += 12 + size;
        let pad = (4 - (size % 4)) % 4;
        if pad == 0 {
            ends_on_pad = None;
            continue;
        }
        if i == bytes.len() {
            ends_on_pad = Some(false);
            break;
        }
        i += pad;
        ends_on_pad = Some(i == bytes.len());
    }
    Emitted {
        sections,
        ends_on_pad,
    }
}

/// The DECLARED (pre-rounding) length of every sampled-tip record in the
/// emitted `samp` section.
fn declared_record_lengths(bytes: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut i = 4usize;
    while i + 12 <= bytes.len() {
        let kind = &bytes[i + 4..i + 8];
        let size = u32::from_be_bytes(bytes[i + 8..i + 12].try_into().unwrap()) as usize;
        if kind == b"samp" {
            let (mut j, end) = (i + 12, i + 12 + size);
            while j + 4 <= end {
                let declared = u32::from_be_bytes(bytes[j..j + 4].try_into().unwrap()) as usize;
                out.push(declared);
                j += 4 + declared.div_ceil(4) * 4;
            }
            return out;
        }
        i += 12 + size + (4 - (size % 4)) % 4;
    }
    out
}

// ── A1 — vocabulary coverage ─────────────────────────────────────────

#[test]
fn image_abr_lane_a1_every_observed_key_ostype_pair_survives_the_reader() {
    let p = profile();
    let pairs = p.key_ostype_pairs();
    assert!(
        !pairs.is_empty(),
        "the profile's key × OSType table is empty"
    );

    let mut keys = BTreeSet::new();
    for (key, ostype, count) in &pairs {
        assert!(*count > 0, "{key}|{} has a zero count", cc(*ostype));
        keys.insert(key.clone());
        // Both dialects, because both occur in the same real files and a
        // reader that handles only one fails on every fixture.
        for force_long in [false, true] {
            let ctx = format!(
                "{key}|{} ({})",
                cc(*ostype),
                if force_long { "long-form" } else { "as stored" }
            );
            let brush = probe_brush(bkey(key, force_long), probe_value(*ostype, p.unit_for(key)));
            let bytes = AbrBuilder::new().brush(brush).build();
            let f = parse_one(&bytes, &ctx);
            let b = only_brush(&f, &ctx);
            let item = b
                .descriptor
                .items
                .iter()
                .find(|(k, _)| k.matches(key.as_bytes()))
                .unwrap_or_else(|| panic!("{ctx}: the reader DROPPED the key"));
            assert_eq!(
                item.1.ostype(),
                *ostype,
                "{ctx}: surfaced as {}",
                cc(item.1.ostype())
            );
            // The dialect is recorded and never acted on: the same key in
            // either spelling looks up identically.
            assert_eq!(
                item.0.is_long_form(),
                force_long || key.len() != 4,
                "{ctx}: the stored dialect is not remembered"
            );
            assert!(
                b.descriptor.get(key.as_bytes()).is_some(),
                "{ctx}: not reachable by key lookup"
            );
        }
    }

    assert_eq!(
        keys.len() as u64,
        p.total("distinct_keys"),
        "the key × OSType table must cover every distinct key the corpus has"
    );
    // One key carries two OSTypes — `Brsh`, which is a list at the top of
    // `desc` and a single shape descriptor inside a preset (spec §3.2
    // trap). A vocabulary keyed by name alone would collapse them.
    assert_eq!(
        pairs.len() - keys.len(),
        1,
        "exactly one key is polymorphic across OSTypes"
    );
}

#[test]
fn image_abr_lane_a1_every_observed_key_unit_pair_keeps_its_unit_code() {
    let p = profile();
    let pairs = p.key_unit_pairs();
    assert!(!pairs.is_empty());
    for (key, unit, _) in &pairs {
        let ctx = format!("{key}|{}", cc(*unit));
        for force_long in [false, true] {
            let brush = probe_brush(bkey(key, force_long), BValue::UntF(*unit, 200.0));
            let f = parse_one(&AbrBuilder::new().brush(brush).build(), &ctx);
            let b = only_brush(&f, &ctx);
            let (got_unit, raw) = b
                .descriptor
                .unit_float(key.as_bytes())
                .unwrap_or_else(|| panic!("{ctx}: not surfaced as a unit float"));
            assert_eq!(got_unit, *unit, "{ctx}: surfaced as {}", cc(got_unit));
            // Stored verbatim: 200.0 is a real observed `tiltScale`, and
            // percent-derived values are NOT clamped into 0..1.
            assert_eq!(raw, 200.0, "{ctx}: the stored value was altered");
        }
    }

    // A unit-float key without a unit row (or the reverse) would mean the
    // two tables disagree about what a key is.
    let untf: BTreeSet<String> = p
        .key_ostype_pairs()
        .into_iter()
        .filter(|(_, o, _)| o == b"UntF")
        .map(|(k, _, _)| k)
        .collect();
    let with_unit: BTreeSet<String> = pairs.iter().map(|(k, _, _)| k.clone()).collect();
    assert_eq!(
        untf, with_unit,
        "the key × OSType and key × unit tables disagree"
    );
    // Only three unit codes occur in the whole corpus.
    let codes: BTreeSet<String> = pairs.iter().map(|(_, u, _)| cc(*u)).collect();
    assert_eq!(
        codes,
        ["#Ang", "#Prc", "#Pxl"].map(str::to_string).into(),
        "the observed unit vocabulary changed"
    );
}

#[test]
fn image_abr_lane_a1_the_profile_tables_reconcile_with_their_totals() {
    // If the tables the gate is driven by do not add up, the gate is
    // measuring something other than the corpus.
    let p = profile();
    let values = p.total("descriptor_values");

    let by_ostype: BTreeMap<String, u64> = p.counts("ostype_counts").into_iter().collect();
    assert_eq!(by_ostype.values().sum::<u64>(), values);

    let mut rolled: BTreeMap<String, u64> = BTreeMap::new();
    for (_, ostype, n) in p.key_ostype_pairs() {
        *rolled.entry(cc(ostype)).or_default() += n;
    }
    assert_eq!(
        rolled, by_ostype,
        "key × OSType does not roll up to the OSType histogram"
    );

    assert_eq!(
        p.counts("key_form_counts")
            .iter()
            .map(|(_, n)| n)
            .sum::<u64>(),
        values,
        "every descriptor item is either long-form or a 4-byte code"
    );
    // Only nine OSTypes occur; the other five are implemented for safety.
    assert_eq!(by_ostype.len(), 9);
    assert!(!by_ostype.contains_key("obj "), "`obj ` was never observed");
}

// ── A2 — class-id dispatch coverage ──────────────────────────────────

/// The class ids that name a tip variant.
fn tip_classes() -> Vec<String> {
    profile().names("tip_class_counts")
}

#[test]
fn image_abr_lane_a2_every_observed_class_id_is_dispatched_or_named() {
    let p = profile();
    let ids = p.names("class_id_counts");
    assert_eq!(ids.len(), 14, "the observed class-id inventory changed");
    let tips = tip_classes();

    for id in &ids {
        // The tip position is where an unknown class id is most
        // dangerous: dispatch is by class id alone, so the reference
        // errors the WHOLE file there. This reader must not.
        let bytes = AbrBuilder::new()
            .brush(brush_preset("B", BDesc::new(bkey(id, false))))
            .build();
        let f = parse_one(&bytes, id);
        let b = only_brush(&f, id);
        if tips.contains(id) {
            assert!(
                !matches!(b.tip, AbrTip::Unsupported { .. } | AbrTip::Missing),
                "{id}: a known tip class must dispatch to a typed variant"
            );
            continue;
        }
        match &b.tip {
            AbrTip::Unsupported { class_id } => assert_eq!(class_id, id),
            other => panic!("{id}: expected a retained-but-unsupported tip, got {other:?}"),
        }
        assert!(
            f.warnings.iter().any(|w| matches!(
                w,
                AbrWarning::UnsupportedTipClass { class_id, .. } if class_id == id
            )),
            "{id}: degraded silently — the diagnostic must name the class id"
        );
        // Degraded, not lost: the preset is still there with its name.
        assert_eq!(b.name, "B");
    }
}

#[test]
fn image_abr_lane_a2_tip_class_ids_dispatch_to_their_typed_variants() {
    for (class, count) in profile().counts("tip_class_counts") {
        assert!(count > 0);
        let tip = match class.as_str() {
            "sampledBrush" => sampled_tip("T", UUID_A, 40.0),
            "computedBrush" => BDesc::new(klong("computedBrush"))
                .item(k4("Dmtr"), BValue::UntF(*b"#Pxl", 40.0))
                .item(k4("Rndn"), BValue::UntF(*b"#Prc", 100.0))
                .item(k4("Hrdn"), BValue::UntF(*b"#Prc", 0.0)),
            "dBrush" => BDesc::new(klong("dBrush"))
                .item(k4("Shp "), BValue::Long(6))
                .item(k4("Dnst"), BValue::UntF(*b"#Prc", 0.27)),
            "dTips" => BDesc::new(klong("dTips"))
                .item(k4("Shp "), BValue::Long(3))
                .item(klong("dtipsType"), BValue::Long(0)),
            other => panic!("unmapped tip class `{other}`"),
        };
        let bytes = AbrBuilder::new()
            .sample(SampleSpec::new(UUID_A, 4, 3, ramp(4, 3)))
            .brush(brush_preset("T", tip))
            .build();
        let f = parse_one(&bytes, &class);
        let b = only_brush(&f, &class);
        let ok = matches!(
            (&class[..], &b.tip),
            ("sampledBrush", AbrTip::Sampled(_))
                | ("computedBrush", AbrTip::Computed(_))
                | ("dBrush", AbrTip::Bristle(_))
                | ("dTips", AbrTip::Erodible(_))
        );
        assert!(ok, "{class}: dispatched to {:?}", b.tip);
    }
}

#[test]
fn image_abr_lane_a2_structural_class_ids_are_recognised_in_their_own_homes() {
    // Every class id that is NOT a tip variant, at the position it
    // actually occurs at, with the model member it is supposed to reach.
    let brush = brush_preset("Preset", sampled_tip("T", UUID_A, 40.0));
    // The helper writes every gate false; `with` REPLACES rather than
    // shadows, so the fixture carries one of each.
    let brush = with(brush, klong("useTipDynamics"), BValue::Bool(true));
    let brush = with(brush, klong("useTexture"), BValue::Bool(true));
    let brush = brush
        // `brVr` — the dynamics primitive; all 220 observed carry it.
        .item(klong("szVr"), BValue::Obj(dynamics(2, 25, 10.0, Some(0.0))))
        // `Ptrn` — the texture identity descriptor's class.
        .item(
            k4("Txtr"),
            BValue::Obj(
                BDesc::new(k4("Ptrn"))
                    .item(k4("Idnt"), BValue::Text("pattern-uuid".into()))
                    .item(k4("Nm  "), BValue::Text("Gray Granite".into())),
            ),
        )
        // `PbTl` — an observed tool class the tool table does not name;
        // the fallback is load-bearing, not defensive.
        .item(
            klong("toolOptions"),
            BValue::Obj(BDesc::new(k4("PbTl")).item(klong("flow"), BValue::Long(100))),
        );

    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp(4, 3)))
        .brush(brush)
        .hierarchy(vec![
            BDesc::new(k4("Grup"))
                .item(k4("Nm  "), BValue::Text("Buildings".into()))
                .item(klong("zuid"), BValue::Text("folder-uuid".into())),
            BDesc::new(klong("preset")),
            BDesc::new(klong("groupEnd")),
        ])
        .build();
    let f = parse_one(&bytes, "structural class ids");
    let b = only_brush(&f, "structural class ids");

    // `brushPreset` on the list element: dispatched, so no complaint.
    assert!(
        !f.warnings
            .iter()
            .any(|w| matches!(w, AbrWarning::UnexpectedClassId { .. })),
        "{:?}",
        f.warnings
    );
    // `null` — the class of both the `desc` and the `phry` roots.
    assert_eq!(f.brushes.len(), 1);
    assert_eq!(f.hierarchy.len(), 3, "the `phry` root parsed");
    // `brVr`, `Ptrn`, `PbTl`, `dualBrush`, `brushGroup`.
    assert!(b.shape_dynamics.as_ref().unwrap().size.is_some(), "brVr");
    let tx = b.texture.as_ref().expect("useTexture true");
    assert_eq!(tx.pattern_id.as_deref(), Some("pattern-uuid"), "Ptrn");
    assert_eq!(b.tool_options.as_ref().unwrap().class_id, "PbTl");
    assert!(b.dual_brush.is_some(), "dualBrush");
    assert!(b.descriptor.contains(b"brushGroup"), "brushGroup");
    // `Grup` / `preset` / `groupEnd` are the hierarchy's whole vocabulary.
    assert!(matches!(
        f.hierarchy[0],
        image_psd::abr::HierarchyNode::GroupOpen { .. }
    ));
    assert!(matches!(
        f.hierarchy[1],
        image_psd::abr::HierarchyNode::Preset {
            brush_index: Some(0)
        }
    ));
    assert!(matches!(
        f.hierarchy[2],
        image_psd::abr::HierarchyNode::GroupEnd
    ));
}

#[test]
fn image_abr_lane_a2_a_class_id_out_of_place_is_reported_and_the_file_survives() {
    // The two places the reader asserts a class id it expects.
    let bytes = AbrBuilder::new()
        .brush(
            // A list element that is not a `brushPreset`…
            BDesc::new(klong("notAPreset"))
                .item(klong("useTipDynamics"), BValue::Bool(true))
                // …carrying a dynamics descriptor that is not a `brVr`.
                .item(
                    klong("szVr"),
                    BValue::Obj(BDesc::new(klong("notBrVr")).item(k4("bVTy"), BValue::Long(2))),
                ),
        )
        .build();
    let f = parse_one(&bytes, "misplaced class ids");
    let reported: Vec<&String> = f
        .warnings
        .iter()
        .filter_map(|w| match w {
            AbrWarning::UnexpectedClassId { class_id, .. } => Some(class_id),
            _ => None,
        })
        .collect();
    assert_eq!(reported, vec!["notAPreset", "notBrVr"]);
    // Reported, not fatal, and the value still made it through.
    assert_eq!(
        f.brushes[0]
            .shape_dynamics
            .as_ref()
            .and_then(|s| s.size.as_ref())
            .map(|d| d.control_ordinal),
        Some(2)
    );
}

// ── A3 — ordinal acceptance and bounds ───────────────────────────────

/// A brush whose shape dynamics carry one `brVr` with the given ordinals.
fn brush_with_dynamics(control: i32, fade: i32) -> BDesc {
    with(
        with(
            brush_preset("D", sampled_tip("D", UUID_A, 40.0)),
            klong("useTipDynamics"),
            BValue::Bool(true),
        ),
        klong("szVr"),
        BValue::Obj(dynamics(control, fade, 10.0, None)),
    )
}

#[test]
fn image_abr_lane_a3_every_observed_ordinal_parses() {
    let p = profile();
    let tables: BTreeMap<String, BTreeMap<String, u64>> = p
        .nested_counts("ordinal_value_counts")
        .into_iter()
        .collect();
    assert_eq!(
        tables.keys().cloned().collect::<Vec<_>>(),
        ["Shp @dBrush", "Shp @dTips", "bVTy", "dtipsType", "fStp"],
        "the observed ordinal vocabulary changed"
    );

    for (table, values) in &tables {
        for raw in values.keys() {
            let v: i32 = raw.parse().expect("an ordinal is an integer");
            let ctx = format!("{table} = {v}");
            let (bytes, ()) = match table.as_str() {
                "bVTy" => (
                    AbrBuilder::new().brush(brush_with_dynamics(v, 25)).build(),
                    (),
                ),
                "fStp" => (
                    AbrBuilder::new().brush(brush_with_dynamics(0, v)).build(),
                    (),
                ),
                "Shp @dBrush" => (
                    AbrBuilder::new()
                        .brush(brush_preset(
                            "B",
                            BDesc::new(klong("dBrush")).item(k4("Shp "), BValue::Long(v)),
                        ))
                        .build(),
                    (),
                ),
                "Shp @dTips" => (
                    AbrBuilder::new()
                        .brush(brush_preset(
                            "B",
                            BDesc::new(klong("dTips")).item(k4("Shp "), BValue::Long(v)),
                        ))
                        .build(),
                    (),
                ),
                "dtipsType" => (
                    AbrBuilder::new()
                        .brush(brush_preset(
                            "B",
                            BDesc::new(klong("dTips")).item(klong("dtipsType"), BValue::Long(v)),
                        ))
                        .build(),
                    (),
                ),
                other => panic!("unmapped ordinal table `{other}`"),
            };
            let f = parse_one(&bytes, &ctx);
            let b = only_brush(&f, &ctx);
            match table.as_str() {
                "bVTy" => {
                    let d = b.shape_dynamics.unwrap().size.unwrap();
                    assert_eq!(d.control_ordinal, v, "{ctx}: the raw ordinal is retained");
                    assert_eq!(d.control, ControlSource::from_ordinal(v).unwrap(), "{ctx}");
                }
                "fStp" => {
                    assert_eq!(b.shape_dynamics.unwrap().size.unwrap().fade_steps, v);
                }
                "Shp @dBrush" => match b.tip {
                    AbrTip::Bristle(t) => {
                        assert_eq!(t.shape_ordinal, v);
                        assert!(t.shape.is_some(), "{ctx}: an observed shape must resolve");
                    }
                    other => panic!("{ctx}: {other:?}"),
                },
                "Shp @dTips" => match b.tip {
                    AbrTip::Erodible(t) => assert_eq!(t.shape_ordinal, v),
                    other => panic!("{ctx}: {other:?}"),
                },
                "dtipsType" => match b.tip {
                    AbrTip::Erodible(t) => assert_eq!(t.tips_type, Some(v)),
                    other => panic!("{ctx}: {other:?}"),
                },
                _ => unreachable!(),
            }
            assert!(
                !f.warnings
                    .iter()
                    .any(|w| matches!(w, AbrWarning::OrdinalOutOfRange { .. })),
                "{ctx}: an OBSERVED ordinal must not be reported out of range: {:?}",
                f.warnings
            );
        }
    }
}

#[test]
fn image_abr_lane_a3_an_out_of_range_ordinal_degrades_instead_of_indexing() {
    // The tables these index exist NOWHERE in the file, so the bounds are
    // this reader's own inference and a stray value must not become an
    // out-of-bounds index.
    for bad in [9i32, -1, i32::MAX, i32::MIN] {
        let f = parse_one(
            &AbrBuilder::new()
                .brush(brush_with_dynamics(bad, 25))
                .build(),
            "bVTy out of range",
        );
        let d = only_brush(&f, "bVTy").shape_dynamics.unwrap().size.unwrap();
        assert_eq!(d.control_ordinal, bad, "the raw ordinal is still retained");
        assert_eq!(d.control, ControlSource::Off, "degraded, not indexed");
        assert!(f.warnings.iter().any(|w| matches!(
            w,
            AbrWarning::OrdinalOutOfRange { key, value } if key == "bVTy" && *value == bad
        )));
    }
    for bad in [10i32, -1, i32::MAX] {
        let f = parse_one(
            &AbrBuilder::new()
                .brush(brush_preset(
                    "B",
                    BDesc::new(klong("dBrush")).item(k4("Shp "), BValue::Long(bad)),
                ))
                .build(),
            "bristle Shp out of range",
        );
        match only_brush(&f, "bristle").tip {
            AbrTip::Bristle(t) => {
                assert_eq!(t.shape_ordinal, bad);
                assert!(t.shape.is_none(), "no name is invented");
            }
            other => panic!("{other:?}"),
        }
        assert!(f.warnings.iter().any(|w| matches!(
            w,
            AbrWarning::OrdinalOutOfRange { key, value } if key == "Shp " && *value == bad
        )));
    }
}

#[test]
fn image_abr_lane_a3_the_erodible_ordinal_is_not_run_through_the_bristle_table() {
    // The corpus contradicts the third-party reference here: it indexes
    // the erodible `Shp ` into the BRISTLE table, and `Square Charcoal`
    // carries 3, which is *square* erodible and *round angle* bristle.
    // The erodible table's membership is unsettled, so the integer is
    // retained and NOTHING is resolved from it — which also means the
    // bristle bounds check must not fire on it.
    let f = parse_one(
        &AbrBuilder::new()
            .brush(brush_preset(
                "B",
                BDesc::new(klong("dTips"))
                    .item(k4("Shp "), BValue::Long(42))
                    .item(klong("dtipsType"), BValue::Long(7)),
            ))
            .build(),
        "erodible ordinals",
    );
    match only_brush(&f, "erodible").tip {
        AbrTip::Erodible(t) => {
            assert_eq!(t.shape_ordinal, 42);
            assert_eq!(t.tips_type, Some(7));
        }
        other => panic!("{other:?}"),
    }
    assert!(
        !f.warnings
            .iter()
            .any(|w| matches!(w, AbrWarning::OrdinalOutOfRange { .. })),
        "the bristle table's bounds were applied to an erodible ordinal: {:?}",
        f.warnings
    );
}

// ── A4 — container-shape coverage ────────────────────────────────────

#[test]
fn image_abr_lane_a4_every_observed_container_shape_parses() {
    let p = profile();
    let mut orders: BTreeSet<Vec<String>> = BTreeSet::new();
    let mut padded = 0usize;
    let mut unpadded = 0usize;

    for shape in p.files() {
        let kinds: Vec<String> = shape.kinds().into_iter().map(str::to_string).collect();
        orders.insert(kinds.clone());
        if shape.last_section_padded {
            padded += 1;
        } else {
            unpadded += 1;
        }

        // Synthesise the same shape: the same sections in the same order,
        // empty where the real file's section is empty, and the same
        // last-section termination. Volume is not the point; shape is.
        let brushes = shape.brushes.min(2);
        let tips = shape.sampled_tips.min(2);
        let ids = [UUID_A, UUID_B];
        let mut b = AbrBuilder::new()
            .version(shape.version, shape.minor_version)
            .descriptor_version(shape.descriptor_version)
            .pad_last_section(shape.last_section_padded);
        for kind in &kinds {
            match kind.as_str() {
                "samp" => {
                    if shape.section_size("samp") == Some(0) {
                        b = b.empty_samp();
                    } else {
                        for id in ids.iter().take(tips) {
                            b = b.sample(SampleSpec::new(id, 4, 3, ramp(4, 3)).rle());
                        }
                    }
                }
                "patt" => {
                    b = b.patt(if shape.section_size("patt") == Some(0) {
                        Vec::new()
                    } else {
                        vec![1, 2, 3, 4, 5]
                    });
                }
                "desc" => {
                    if brushes == 0 {
                        b = b.empty_desc();
                    }
                    for (i, id) in ids.iter().enumerate().take(brushes) {
                        let tip = if i < tips {
                            sampled_tip("T", id, 40.0)
                        } else {
                            BDesc::new(klong("computedBrush"))
                                .item(k4("Dmtr"), BValue::UntF(*b"#Pxl", 40.0))
                        };
                        b = b.brush(brush_preset("P", tip));
                    }
                }
                "phry" => {
                    b = b.hierarchy(
                        std::iter::repeat_with(|| BDesc::new(klong("preset")))
                            .take(brushes)
                            .collect(),
                    );
                }
                other => panic!("`{other}` is a section kind this builder cannot emit"),
            }
        }
        let bytes = b.build();
        let ctx = &shape.file;

        // The fixture really has the shape the profile describes.
        let emitted = walk(&bytes);
        assert_eq!(
            emitted
                .sections
                .iter()
                .map(|(k, _)| k.clone())
                .collect::<Vec<_>>(),
            kinds,
            "{ctx}: emitted section order"
        );
        if let Some(on_pad) = emitted.ends_on_pad {
            // Only meaningful when the last section actually needs a pad;
            // the both-terminations case gets its own fixture below.
            assert_eq!(
                on_pad, shape.last_section_padded,
                "{ctx}: last-section termination"
            );
        }

        let f = parse_one(&bytes, ctx);
        assert_eq!(
            (f.version, f.minor_version),
            (shape.version, shape.minor_version)
        );
        assert_eq!(f.brushes.len(), brushes, "{ctx}: brushes");
        assert_eq!(f.samples.len(), tips, "{ctx}: sampled tips");
        assert_eq!(
            f.patterns_raw.is_some(),
            shape.section_size("patt").is_some_and(|s| s > 0),
            "{ctx}: an EMPTY patt section is normal, not a pattern"
        );
        assert!(f.warnings.is_empty(), "{ctx}: {:?}", f.warnings);
    }

    // The nine files contain exactly two section orders between them, and
    // both last-section terminations — 4 files end on the pad, 5 do not,
    // and applying the pad to those 5 walks 1–3 bytes past EOF.
    assert_eq!(
        orders.len(),
        2,
        "the observed container orders changed: {orders:?}"
    );
    assert_eq!((padded, unpadded), (4, 5));
}

#[test]
fn image_abr_lane_a4_a_desc_of_size_two_mod_four_followed_by_phry_stays_aligned() {
    // One real fixture has a `desc` of 510,766 bytes (size % 4 == 2)
    // followed by a `phry`: the stream only stays aligned if the two pad
    // bytes are consumed. Unicode strings are the only odd-length payload
    // in a descriptor, so a name's length is the knob that moves the
    // section size across the remainder.
    let profile_has_it = profile().files().iter().any(|f| {
        f.section_size("desc").is_some_and(|s| s % 4 == 2) && f.kinds().last() == Some(&"phry")
    });
    assert!(profile_has_it, "the corpus no longer contains this shape");

    let mut built = None;
    for extra in 0..8 {
        let name = "n".repeat(extra);
        let bytes = AbrBuilder::new()
            .sample(SampleSpec::new(UUID_A, 4, 3, ramp(4, 3)))
            .brush(brush_preset(&name, sampled_tip(&name, UUID_A, 40.0)))
            .hierarchy(vec![BDesc::new(klong("preset"))])
            .build();
        let sections = walk(&bytes).sections;
        let desc = sections
            .iter()
            .find(|(k, _)| k == "desc")
            .expect("a desc section");
        if desc.1 % 4 == 2 {
            built = Some((bytes, desc.1));
            break;
        }
    }
    let (bytes, size) = built.expect("no brush-name length produced a desc of size ≡ 2 (mod 4)");
    assert_eq!(size % 4, 2);

    let f = parse_one(&bytes, "desc ≡ 2 (mod 4) then phry");
    assert_eq!(f.brushes.len(), 1);
    assert_eq!(f.hierarchy.len(), 1, "the phry after the pad was found");
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);
}

#[test]
fn image_abr_lane_a4_both_last_section_terminations_read_the_same_file() {
    // 4 of the 9 real files end ON the padded boundary and 5 end on the
    // unpadded section end, so a reader that applies the pad and then
    // reads a 12-byte header walks 1–3 bytes past EOF on those 5 and
    // reports a perfectly good file as corrupt. The two variants must be
    // byte-different (or the fixture proves nothing) and model-identical.
    let mut fixtures = Vec::new();
    for extra in 0..8 {
        let name = "n".repeat(extra);
        let build = |pad: bool| {
            AbrBuilder::new()
                .sample(SampleSpec::new(UUID_A, 4, 3, ramp(4, 3)))
                .brush(brush_preset(&name, sampled_tip(&name, UUID_A, 40.0)))
                .pad_last_section(pad)
                .build()
        };
        let (padded, unpadded) = (build(true), build(false));
        if padded.len() != unpadded.len() {
            fixtures = vec![(padded, unpadded)];
            break;
        }
    }
    let (padded, unpadded) = fixtures.pop().expect("no name length needed a pad");
    assert_eq!(
        walk(&padded).ends_on_pad,
        Some(true),
        "the padded variant ends on the pad"
    );
    assert_eq!(walk(&unpadded).ends_on_pad, Some(false));

    let a = parse_one(&padded, "padded");
    let b = parse_one(&unpadded, "unpadded");
    assert!(
        a.warnings.is_empty() && b.warnings.is_empty(),
        "{a:?} {b:?}"
    );
    assert_eq!(a.brushes, b.brushes, "the pad is not information");
    assert_eq!(a.samples, b.samples);
}

#[test]
fn image_abr_lane_a4_a_present_but_empty_hierarchy_does_not_hide_the_brushes() {
    // One fixture carries a `phry` while having a single brush; present
    // and empty means "no grouping", not an error, and must not be
    // confused with the count mismatch that refuses to bind.
    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp(4, 3)))
        .brush(brush_preset("P", sampled_tip("P", UUID_A, 40.0)))
        .hierarchy(Vec::new())
        .build();
    let f = parse_one(&bytes, "empty hierarchy");
    assert_eq!(f.brushes.len(), 1);
    assert!(f.hierarchy.is_empty());
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);
}

// ── A5 — gate discipline, including absence ──────────────────────────

/// One gated group: its gate, two keys it owns, and how to see it.
struct GateCase {
    gate: &'static str,
    keys: Vec<(BKey, BValue)>,
    present: fn(&AbrBrush) -> bool,
}

fn gate_cases() -> Vec<GateCase> {
    vec![
        GateCase {
            gate: "useTipDynamics",
            keys: vec![
                (klong("szVr"), BValue::Obj(dynamics(2, 25, 10.0, None))),
                (klong("tiltScale"), BValue::UntF(*b"#Prc", 200.0)),
            ],
            present: |b| b.shape_dynamics.is_some(),
        },
        GateCase {
            gate: "useScatter",
            keys: vec![
                (
                    klong("scatterDynamics"),
                    BValue::Obj(dynamics(0, 25, 368.0, None)),
                ),
                (k4("Cnt "), BValue::Doub(2.0)),
            ],
            present: |b| b.scatter.is_some(),
        },
        GateCase {
            gate: "useTexture",
            keys: vec![
                (klong("textureScale"), BValue::UntF(*b"#Prc", 62.0)),
                (klong("textureBrightness"), BValue::Long(-7)),
            ],
            present: |b| b.texture.is_some(),
        },
        GateCase {
            gate: "useColorDynamics",
            keys: vec![
                (klong("clVr"), BValue::Obj(dynamics(2, 25, 10.0, None))),
                (klong("purity"), BValue::UntF(*b"#Prc", 10.0)),
            ],
            present: |b| b.color_dynamics.is_some(),
        },
        GateCase {
            gate: "usePaintDynamics",
            keys: vec![
                (klong("prVr"), BValue::Obj(dynamics(2, 25, 0.0, Some(0.0)))),
                (klong("opVr"), BValue::Obj(dynamics(2, 25, 0.0, None))),
            ],
            present: |b| b.transfer.is_some(),
        },
        GateCase {
            gate: "useBrushPose",
            keys: vec![
                (klong("brushPoseAngle"), BValue::Long(23)),
                (klong("brushPosePressure"), BValue::UntF(*b"#Prc", 9.0)),
            ],
            present: |b| b.brush_pose.is_some(),
        },
    ]
}

/// A preset with the gate set (or removed) and the group keys optionally
/// present.
fn gated_brush(case: &GateCase, gate: Option<bool>, keys: bool) -> BDesc {
    let mut d = without(
        brush_preset("G", sampled_tip("G", UUID_A, 40.0)),
        case.gate.as_bytes(),
    );
    if let Some(v) = gate {
        d = d.item(klong(case.gate), BValue::Bool(v));
    }
    if keys {
        for (k, v) in &case.keys {
            d = d.item(k.clone(), v.clone());
        }
    }
    d
}

#[test]
fn image_abr_lane_a5_the_gate_decides_and_an_absent_gate_reads_as_false() {
    let p = profile();
    let gates: BTreeMap<String, BTreeMap<String, u64>> =
        p.nested_counts("gate_counts").into_iter().collect();
    let presets = p.total("brush_presets");

    // The discipline the reader leans on, restated from the profile:
    // every gate is present on every brush except `useBrushPose`, which
    // is absent on 36 — which is precisely why absence must read false.
    let absent: Vec<(&String, u64)> = gates
        .iter()
        .filter(|(_, c)| c["absent"] > 0)
        .map(|(g, c)| (g, c["absent"]))
        .collect();
    assert_eq!(absent, vec![(&"useBrushPose".to_string(), 36)]);
    for (g, c) in &gates {
        assert_eq!(c["present"] + c["absent"], presets, "{g}");
        assert!(c["true"] <= c["present"], "{g}");
    }
    assert_eq!(
        gates["useColorDynamics"]["true"], 0,
        "colour dynamics is the block no corpus file ever enabled"
    );

    for case in gate_cases() {
        assert!(gates.contains_key(case.gate), "{} is not a gate", case.gate);
        let ctx = case.gate;

        // true + its keys → the group is read.
        let f = parse_one(
            &AbrBuilder::new()
                .sample(SampleSpec::new(UUID_A, 4, 3, ramp(4, 3)))
                .brush(gated_brush(&case, Some(true), true))
                .build(),
            ctx,
        );
        assert!(
            (case.present)(&only_brush(&f, ctx)),
            "{ctx}: true must read"
        );

        // false, and — as every real file does — none of its keys.
        for gate in [Some(false), None] {
            let f = parse_one(
                &AbrBuilder::new()
                    .sample(SampleSpec::new(UUID_A, 4, 3, ramp(4, 3)))
                    .brush(gated_brush(&case, gate, false))
                    .build(),
                ctx,
            );
            assert!(
                !(case.present)(&only_brush(&f, ctx)),
                "{ctx}: {gate:?} must not read the group"
            );
            assert!(f.warnings.is_empty(), "{ctx}: {gate:?}: {:?}", f.warnings);
        }
    }
}

#[test]
fn image_abr_lane_a5_a_group_key_without_its_gate_is_reported_not_read() {
    // The combination the gate discipline says cannot happen: 3,215 of
    // 3,215 corpus brushes obey it. The gate still decides — a stray key
    // never causes the group to be read — but silence would hide either a
    // producer that breaks the discipline or a mistake about which keys
    // belong to which group.
    for case in gate_cases() {
        for gate in [Some(false), None] {
            let ctx = format!("{} {:?}", case.gate, gate);
            let f = parse_one(
                &AbrBuilder::new()
                    .sample(SampleSpec::new(UUID_A, 4, 3, ramp(4, 3)))
                    .brush(gated_brush(&case, gate, true))
                    .build(),
                &ctx,
            );
            let b = only_brush(&f, &ctx);
            assert!(!(case.present)(&b), "{ctx}: the gate decides, not the keys");
            let reported = f.warnings.iter().find_map(|w| match w {
                AbrWarning::GatedGroupKeysWithoutGate { gate, keys } if gate == case.gate => {
                    Some(keys.clone())
                }
                _ => None,
            });
            let mut keys = reported.unwrap_or_else(|| {
                panic!(
                    "{ctx}: the stray keys were dropped in silence: {:?}",
                    f.warnings
                )
            });
            keys.sort();
            let mut want: Vec<String> = case.keys.iter().map(|(k, _)| k.text.clone()).collect();
            want.sort();
            assert_eq!(
                keys, want,
                "{ctx}: the diagnostic must name the keys it found"
            );
            // Nothing is lost: the tree still carries them.
            for (k, _) in &case.keys {
                assert!(
                    b.descriptor.contains(k.text.as_bytes()),
                    "{ctx}: {}",
                    k.text
                );
            }
        }
    }
}

#[test]
fn image_abr_lane_a5_the_dual_brush_gate_lives_inside_its_own_descriptor() {
    // `useDualBrush` is the one gate that is NOT flattened onto the brush
    // (spec §6.5), and the `dualBrush` descriptor is present on
    // 3,215/3,215 brushes whether or not the feature is used — so its
    // presence tells you nothing and the inner boolean tells you
    // everything.
    let cases = [
        (Some(true), true),
        (Some(false), false),
        // Absent inside the sub-descriptor: false, like every other gate.
        (None, false),
    ];
    for (inner, expected) in cases {
        let mut db = BDesc::new(klong("dualBrush"));
        if let Some(v) = inner {
            db = db.item(klong("useDualBrush"), BValue::Bool(v));
        }
        let brush = with(
            brush_preset("D", sampled_tip("D", UUID_A, 40.0)),
            klong("dualBrush"),
            BValue::Obj(db),
        );
        let f = parse_one(
            &AbrBuilder::new()
                .sample(SampleSpec::new(UUID_A, 4, 3, ramp(4, 3)))
                .brush(brush)
                .build(),
            "dual brush gate",
        );
        let b = only_brush(&f, "dual brush gate");
        let dual = b
            .dual_brush
            .as_ref()
            .expect("the descriptor is always there");
        assert_eq!(dual.enabled, expected, "inner gate {inner:?}");
    }
}

// ── A6 — the §2.2 self-check identity ────────────────────────────────

#[test]
fn image_abr_lane_a6_the_ledgers_pad_arithmetic_is_the_declared_extent_identity() {
    let rows = ledger();
    let mut by_pad: BTreeMap<usize, usize> = BTreeMap::new();
    for r in rows {
        assert!(r.pad_len <= 3, "{}: pad {} out of range", r.id, r.pad_len);
        assert_eq!(
            r.pad_len,
            (4 - (r.declared_len % 4)) % 4,
            "{}: pad_len is not rounded − declared",
            r.id
        );
        *by_pad.entry(r.pad_len).or_default() += 1;
    }
    // All four remainders occur, and the NON-zero ones are the common
    // case: a self-check written against the ROUNDED extent would reject
    // 2,367 of 3,202 perfectly good records.
    assert_eq!(by_pad[&0], 835);
    assert_eq!(by_pad[&1], 808);
    assert_eq!(by_pad[&2], 766);
    assert_eq!(by_pad[&3], 793);
    assert_eq!(
        rows.len() - by_pad[&0],
        2367,
        "the padded-record majority changed"
    );
    assert_eq!(rows.len() as u64, profile().total("sampled_tip_records"));

    // The rest of the row's arithmetic, while we are here.
    for r in rows {
        assert_eq!(r.w, (r.right - r.left) as u32, "{}: w", r.id);
        assert_eq!(r.h, (r.bottom - r.top) as u32, "{}: h", r.id);
        assert!(r.w > 0 && r.h > 0, "{}: non-positive bounds", r.id);
        assert_eq!(r.depth, 8, "{}: every corpus tip is 8-bit", r.id);
        assert_eq!(
            r.decoded_bytes,
            (r.w * r.h) as usize,
            "{}: one byte per pixel, no row padding",
            r.id
        );
        assert_eq!(
            r.written_planes, 1,
            "{}: a tip is a single-channel mask",
            r.id
        );
        assert_eq!(r.array_count, 56, "{}: the serializer's slot ceiling", r.id);
        assert!(matches!(r.compression, 0 | 1), "{}", r.id);
    }
    // Raw is rare but real — not a branch to defer.
    assert_eq!(rows.iter().filter(|r| r.compression == 0).count(), 19);
    assert_eq!(rows.iter().filter(|r| r.compression == 1).count(), 3183);
}

#[test]
fn image_abr_lane_a6_a_record_of_every_pad_remainder_reads_clean() {
    // Emit records whose DECLARED length is ≡ 0, 1, 2 and 3 (mod 4) and
    // assert the structural parse lands exactly on the declared extent —
    // i.e. `leftover == rounded − declared` — in each. A reader that
    // measured the self-check against the rounded extent passes the first
    // and fails the other three.
    let mut covered: BTreeSet<usize> = BTreeSet::new();
    for w in 4..=7usize {
        let h = 3usize;
        let bytes = AbrBuilder::new()
            .sample(SampleSpec::new(UUID_A, w as i32, h as i32, ramp(w, h)))
            .brush(brush_preset("P", sampled_tip("P", UUID_A, w as f64)))
            .build();
        let declared = declared_record_lengths(&bytes);
        assert_eq!(declared.len(), 1);
        let remainder = declared[0] % 4;
        let pad = (4 - remainder) % 4;
        covered.insert(pad);

        let ctx = format!("declared {} ≡ {remainder} (mod 4)", declared[0]);
        let f = parse_one(&bytes, &ctx);
        assert!(
            !f.warnings
                .iter()
                .any(|w| matches!(w, AbrWarning::SampleRecordTrailingBytes { .. })),
            "{ctx}: the self-check fired on a well-formed record: {:?}",
            f.warnings
        );
        assert_eq!(f.samples.len(), 1, "{ctx}");
        assert_eq!(f.samples[0].coverage8(), ramp(w, h), "{ctx}: pixels");
        assert_eq!(
            (f.samples[0].width, f.samples[0].height),
            (w as u32, h as u32)
        );
    }
    assert_eq!(
        covered,
        BTreeSet::from([0, 1, 2, 3]),
        "the four pad remainders must all be exercised"
    );
}

#[test]
fn image_abr_lane_a6_the_identity_holds_when_the_layout_breaks_the_264_constant() {
    // The famous "264-byte skip" is arithmetic on `array_count == 56` with
    // the plane in slot 55 — 32 + 55×4 + 12 — and nothing more. A file
    // that declares a different slot count, or writes the plane somewhere
    // else, breaks the constant silently and reads its bounds out of the
    // middle of a slot table. Parsing the structure costs nothing and
    // keeps the self-check exact, which is how you know you parsed it
    // right.
    for (array_count, slot) in [(56u32, 55u32), (56, 3), (12, 0), (60, 59), (1, 2)] {
        let ctx = format!("array_count {array_count}, slot {slot}");
        let bytes = AbrBuilder::new()
            .sample(SampleSpec::new(UUID_A, 4, 3, ramp(4, 3)).with_layout(array_count, slot))
            .build();
        let f = parse_one(&bytes, &ctx);
        assert_eq!(
            f.samples.len(),
            1,
            "{ctx}: the structure is self-describing"
        );
        assert_eq!(f.samples[0].coverage8(), ramp(4, 3), "{ctx}: pixels");
        assert!(
            !f.warnings
                .iter()
                .any(|w| matches!(w, AbrWarning::SampleRecordTrailingBytes { .. })),
            "{ctx}: {:?}",
            f.warnings
        );
    }
}

#[test]
fn image_abr_lane_a6_a_genuine_leftover_is_reported_rather_than_absorbed() {
    // The identity is only worth asserting if its violation is visible.
    // Hand-patch four unmodelled bytes into the middle of a record — past
    // the array list, inside the DECLARED extent — and the self-check must
    // say so: `remaining` exceeds the expected pad by exactly those four.
    // (This is the one fixture the builder cannot express, because it only
    // emits records that reconcile.)
    let mut bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp(4, 3)))
        .build();
    assert_eq!(&bytes[4..8], b"8BIM");
    assert_eq!(&bytes[8..12], b"samp", "the samp section comes first");
    let size = u32::from_be_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let declared = u32::from_be_bytes(bytes[16..20].try_into().unwrap()) as usize;
    let pad = (4 - (declared % 4)) % 4;

    // Four extra bytes at the end of the record body: the declared length
    // and the section size both grow by 4, and the rounding is unchanged
    // because the remainder is.
    let at = 20 + declared;
    bytes.splice(at..at, [0xde, 0xad, 0xbe, 0xef]);
    bytes[12..16].copy_from_slice(&((size + 4) as u32).to_be_bytes());
    bytes[16..20].copy_from_slice(&((declared + 4) as u32).to_be_bytes());

    let f = parse_one(&bytes, "an unmodelled tail");
    let reported = f
        .warnings
        .iter()
        .find_map(|w| match w {
            AbrWarning::SampleRecordTrailingBytes {
                remaining,
                expected_pad,
                ..
            } => Some((*remaining, *expected_pad)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("the leftover was absorbed: {:?}", f.warnings));
    assert_eq!(reported, (pad + 4, pad), "remaining vs the expected pad");
    // Degraded, not fatal: the tip itself still decodes.
    assert_eq!(f.samples.len(), 1);
    assert_eq!(f.samples[0].coverage8(), ramp(4, 3));
}

// ── A7 — the join ────────────────────────────────────────────────────

#[test]
fn image_abr_lane_a7_the_join_is_exact_case_sensitive_string_equality() {
    let p = profile();
    // Every `sampledData` in the corpus is exactly 36 characters — a bare
    // UUID, no suffixes — and every one resolved by EXACT equality.
    let lengths = p.counts("sampled_data_length_counts");
    assert_eq!(
        lengths.len(),
        1,
        "more than one sampledData length: {lengths:?}"
    );
    assert_eq!(lengths[0].0, "36");
    let joins = p.counts("join_counts");
    assert_eq!(
        joins,
        vec![("resolved_exact".to_string(), lengths[0].1)],
        "the join table gained a bucket — a fallback ladder is being used"
    );

    // Exact resolves…
    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp(4, 3)))
        .brush(brush_preset("P", sampled_tip("P", UUID_A, 40.0)))
        .build();
    let f = parse_one(&bytes, "exact");
    assert_eq!(f.brushes[0].tip.as_sampled().unwrap().sample_index, Some(0));
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);

    // …and nothing else does. A case flip, a 36-character PREFIX of a
    // longer id, and a trailing-space variant are all misses, and a miss
    // is reported rather than papered over.
    let near_misses = [
        UUID_A.to_uppercase(),
        format!("{UUID_A}-variant"),
        format!("{UUID_A} "),
    ];
    for id in near_misses {
        let bytes = AbrBuilder::new()
            .sample(SampleSpec::new(UUID_A, 4, 3, ramp(4, 3)))
            .brush(brush_preset("P", sampled_tip("P", &id, 40.0)))
            .build();
        let f = parse_one(&bytes, &id);
        assert_eq!(
            f.brushes[0].tip.as_sampled().unwrap().sample_index,
            None,
            "`{id}` must NOT resolve against `{UUID_A}`"
        );
        assert!(
            f.warnings.iter().any(|w| matches!(
                w,
                AbrWarning::UnresolvedSampleReference { id: got, .. } if *got == id
            )),
            "`{id}`: an unresolvable reference must be reported"
        );
        // And emphatically not paired positionally: there is exactly one
        // sample and exactly one brush, which is the case a positional
        // fallback would silently "fix".
        assert!(f.warnings.iter().any(|w| matches!(
            w,
            AbrWarning::OrphanSample { id } if id == UUID_A
        )));
    }
}

#[test]
fn image_abr_lane_a7_samples_are_shared_many_to_one_across_the_whole_tree() {
    // One corpus file has 2,053 brushes against 2,052 samples precisely
    // because a record is referenced twice — in one case by a preset's
    // main tip AND another preset's dual-brush tip. So the join is a
    // LOOKUP, never a pairing, and it runs over every `sampledData` in
    // the tree.
    let dual = BDesc::new(klong("dualBrush"))
        .item(klong("useDualBrush"), BValue::Bool(true))
        .item(k4("Brsh"), BValue::Obj(sampled_tip("dual", UUID_A, 20.0)));
    let second = with(
        brush_preset("Second", sampled_tip("Second", UUID_B, 40.0)),
        klong("dualBrush"),
        BValue::Obj(dual),
    );
    let bytes = AbrBuilder::new()
        .sample(SampleSpec::new(UUID_A, 4, 3, ramp(4, 3)))
        .sample(SampleSpec::new(UUID_B, 5, 3, ramp(5, 3)))
        .brush(brush_preset("First", sampled_tip("First", UUID_A, 40.0)))
        .brush(second)
        .build();
    let f = parse_one(&bytes, "shared samples");
    assert_eq!(f.brushes.len(), 2);
    assert_eq!(f.samples.len(), 2);
    assert_eq!(f.brushes[0].tip.as_sampled().unwrap().sample_index, Some(0));
    assert_eq!(f.brushes[1].tip.as_sampled().unwrap().sample_index, Some(1));
    let dual_tip = f.brushes[1]
        .dual_brush
        .as_ref()
        .unwrap()
        .tip
        .as_ref()
        .unwrap();
    assert_eq!(
        dual_tip.as_sampled().unwrap().sample_index,
        Some(0),
        "the dual brush's tip participates in the same join"
    );
    // Consumed-as-you-go would have left the second reference orphaned.
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);
    assert_eq!(f.brushes[1].sampled_ids(), vec![UUID_B, UUID_A]);
}
