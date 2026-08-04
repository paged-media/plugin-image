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

//! The ANALYST-published `.abr` corpus artifacts, loaded.
//!
//! # What these are, and why an implementer may read them
//!
//! The `.abr` behaviour spec was verified against nine licensed files —
//! 3,215 brush presets, 3,202 sampled tips — plus a 238-image published
//! PNG oracle. That corpus lives at `references/abr-fixtures/`, inside
//! the clean-room mount: **an ANALYST may read it, an IMPLEMENTER may
//! not**, and CI has no copy at all (repo `CLAUDE.md` §3.1).
//!
//! What crosses that boundary is not bytes but *facts about behaviour*:
//! counts, key/OSType/unit tables, dimensions and one-way digests,
//! committed at `image-conformance/fixtures/abr/` with a provenance
//! header on each file. Consuming them is not reading `references/`
//! (spec §13.4, §14.3.1, and `fixtures/abr/README.md`).
//!
//! # The two lanes they make possible
//!
//! * **LANE A** — `tests/abr_lane_a.rs`. IMPLEMENTER-owned, always on,
//!   **needs no corpus**: it drives [`crate::abr_builder`] from these
//!   tables, so every fixture is synthesised and every expectation was
//!   measured against 3,215 real presets rather than invented.
//! * **LANE B** — `tests/abr_corpus.rs`. ANALYST-owned, `#[ignore]`d and
//!   opt-in, and it parses the real files. It skips loudly wherever the
//!   mount is absent, which is every machine that is not an analyst's.
//!
//! # Ownership — the rule that makes a Lane-B failure mean something
//!
//! **Only an ANALYST session with the mount regenerates
//! `corpus-profile.json` or `corpus-record-ledger.tsv`**, and a change to
//! either is a reviewable diff. So a Lane-B failure is unambiguous:
//! either the reader regressed, or the format understanding legitimately
//! changed and the analyst must re-publish. **Editing an expectation file
//! to make a lane pass destroys exactly the property the split was built
//! for**, and an implementer must never do it.

pub mod json;
pub mod sha256;

use std::collections::BTreeMap;
use std::sync::OnceLock;

use json::Json;

/// `corpus-profile.json`, verbatim. Compiled in rather than read from
/// disk so the gate does not depend on the process CWD.
pub const PROFILE_JSON: &str = include_str!("../../fixtures/abr/corpus-profile.json");

/// `corpus-record-ledger.tsv`, verbatim.
pub const LEDGER_TSV: &str = include_str!("../../fixtures/abr/corpus-record-ledger.tsv");

/// One row of `files[]`: the shape of one real container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileShape {
    pub file: String,
    pub bytes: u64,
    pub version: i16,
    pub minor_version: i16,
    /// Section kind and declared size, in file order.
    pub sections: Vec<(String, u64)>,
    /// Whether the file ends ON the 4-byte pad after its last section.
    /// Both variants occur (4 padded, 5 not) and one of them walks a
    /// naive reader past EOF.
    pub last_section_padded: bool,
    pub descriptor_version: u32,
    pub brushes: usize,
    pub sampled_tips: usize,
}

impl FileShape {
    /// The section kinds in order — the container's "shape" for the
    /// purposes of Lane A's A4.
    pub fn kinds(&self) -> Vec<&str> {
        self.sections.iter().map(|(k, _)| k.as_str()).collect()
    }

    pub fn section_size(&self, kind: &str) -> Option<u64> {
        self.sections
            .iter()
            .find(|(k, _)| k == kind)
            .map(|(_, s)| *s)
    }
}

/// The whole-corpus derived tables.
#[derive(Debug)]
pub struct CorpusProfile {
    root: Json,
}

impl CorpusProfile {
    fn parse() -> CorpusProfile {
        let root = Json::parse(PROFILE_JSON)
            .unwrap_or_else(|e| panic!("corpus-profile.json is not valid JSON: {e}"));
        assert!(
            root.get("_provenance")
                .and_then(Json::as_str)
                .is_some_and(|p| p.contains("ANALYST")),
            "the profile must carry its provenance header"
        );
        CorpusProfile { root }
    }

    /// A `totals.*` entry.
    pub fn total(&self, name: &str) -> u64 {
        self.root
            .get("totals")
            .and_then(|t| t.get(name))
            .and_then(Json::as_u64)
            .unwrap_or_else(|| panic!("totals.{name} missing from the profile"))
    }

    /// A flat `{ "name": count }` table, in file order.
    pub fn counts(&self, table: &str) -> Vec<(String, u64)> {
        self.root
            .get(table)
            .and_then(Json::as_object)
            .unwrap_or_else(|| panic!("`{table}` missing from the profile"))
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.as_u64()
                        .unwrap_or_else(|| panic!("{table}.{k} is not a count")),
                )
            })
            .collect()
    }

    /// The names of a flat count table.
    pub fn names(&self, table: &str) -> Vec<String> {
        self.counts(table).into_iter().map(|(k, _)| k).collect()
    }

    /// A `{ "name": { "sub": count } }` table (`gate_counts`,
    /// `ordinal_value_counts`).
    pub fn nested_counts(&self, table: &str) -> Vec<(String, BTreeMap<String, u64>)> {
        self.root
            .get(table)
            .and_then(Json::as_object)
            .unwrap_or_else(|| panic!("`{table}` missing from the profile"))
            .iter()
            .map(|(k, v)| {
                let inner = v
                    .as_object()
                    .unwrap_or_else(|| panic!("{table}.{k} is not an object"))
                    .iter()
                    .map(|(sk, sv)| {
                        (
                            sk.clone(),
                            sv.as_u64()
                                .unwrap_or_else(|| panic!("{table}.{k}.{sk} is not a count")),
                        )
                    })
                    .collect();
                (k.clone(), inner)
            })
            .collect()
    }

    /// `key_ostype_counts`, split into `(key, ostype, count)`.
    pub fn key_ostype_pairs(&self) -> Vec<(String, [u8; 4], u64)> {
        self.counts("key_ostype_counts")
            .into_iter()
            .map(|(pair, n)| {
                let (key, ostype) = split_pair(&pair);
                (key, four_cc(&ostype), n)
            })
            .collect()
    }

    /// `key_unit_counts`, split into `(key, unit, count)`.
    pub fn key_unit_pairs(&self) -> Vec<(String, [u8; 4], u64)> {
        self.counts("key_unit_counts")
            .into_iter()
            .map(|(pair, n)| {
                let (key, unit) = split_pair(&pair);
                (key, four_cc(&unit), n)
            })
            .collect()
    }

    /// The unit a key carries, when it is a unit float.
    pub fn unit_for(&self, key: &str) -> Option<[u8; 4]> {
        self.key_unit_pairs()
            .into_iter()
            .find(|(k, _, _)| k == key)
            .map(|(_, u, _)| u)
    }

    /// `enum_pair_counts`, split into `(key, type_key, value_key, count)`.
    pub fn enum_pairs(&self) -> Vec<(String, String, String, u64)> {
        self.counts("enum_pair_counts")
            .into_iter()
            .map(|(triple, n)| {
                let mut it = triple.splitn(3, '|');
                let key = it.next().unwrap_or_default().to_string();
                let type_key = it.next().unwrap_or_default().to_string();
                let value_key = it.next().unwrap_or_default().to_string();
                (key, type_key, value_key, n)
            })
            .collect()
    }

    pub fn files(&self) -> Vec<FileShape> {
        self.root
            .get("files")
            .and_then(Json::as_array)
            .expect("`files` missing from the profile")
            .iter()
            .map(|f| {
                let num = |name: &str| {
                    f.get(name)
                        .and_then(Json::as_i64)
                        .unwrap_or_else(|| panic!("files[].{name} missing"))
                };
                FileShape {
                    file: f
                        .get("file")
                        .and_then(Json::as_str)
                        .expect("files[].file")
                        .to_string(),
                    bytes: num("bytes") as u64,
                    version: num("version") as i16,
                    minor_version: num("minor_version") as i16,
                    sections: f
                        .get("sections")
                        .and_then(Json::as_array)
                        .expect("files[].sections")
                        .iter()
                        .map(|s| {
                            (
                                s.get("kind")
                                    .and_then(Json::as_str)
                                    .expect("sections[].kind")
                                    .to_string(),
                                s.get("size")
                                    .and_then(Json::as_u64)
                                    .expect("sections[].size"),
                            )
                        })
                        .collect(),
                    last_section_padded: f
                        .get("last_section_padded")
                        .and_then(Json::as_bool)
                        .expect("files[].last_section_padded"),
                    descriptor_version: num("descriptor_version") as u32,
                    brushes: num("brushes") as usize,
                    sampled_tips: num("sampled_tips") as usize,
                }
            })
            .collect()
    }
}

/// The parsed profile, loaded once.
pub fn profile() -> &'static CorpusProfile {
    static P: OnceLock<CorpusProfile> = OnceLock::new();
    P.get_or_init(CorpusProfile::parse)
}

/// Split a `left|right` table name. Neither half contains a `|`.
fn split_pair(pair: &str) -> (String, String) {
    match pair.rsplit_once('|') {
        Some((l, r)) => (l.to_string(), r.to_string()),
        None => panic!("`{pair}` is not a `left|right` pair"),
    }
}

fn four_cc(s: &str) -> [u8; 4] {
    let b = s.as_bytes();
    assert_eq!(b.len(), 4, "`{s}` is not a four-character code");
    [b[0], b[1], b[2], b[3]]
}

/// One row of `corpus-record-ledger.tsv` — one sampled-tip record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerRow {
    pub file: String,
    pub index: usize,
    pub id: String,
    /// The record's length field, BEFORE rounding up to a multiple of 4.
    pub declared_len: usize,
    /// `rounded_len − declared_len`, in `0..=3`.
    pub pad_len: usize,
    pub array_count: u32,
    pub written_planes: u32,
    pub top: i32,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
    pub w: u32,
    pub h: u32,
    pub depth: u16,
    /// 0 = raw, 1 = RLE.
    pub compression: u8,
    pub decoded_bytes: usize,
    /// SHA-256 of the decoded coverage mask — row-major, one byte per
    /// pixel, exactly `w * h` bytes, **not inverted**.
    pub sha256: String,
    /// For 238 rows, the independently published transparent PNG whose
    /// ALPHA channel hashes to the same value (spec §2.5).
    pub png_oracle: Option<String>,
}

/// The ledger, loaded once, in file order.
pub fn ledger() -> &'static [LedgerRow] {
    static L: OnceLock<Vec<LedgerRow>> = OnceLock::new();
    L.get_or_init(parse_ledger)
}

fn parse_ledger() -> Vec<LedgerRow> {
    let mut rows = Vec::new();
    let mut header_seen = false;
    let mut provenance_seen = false;
    for line in LEDGER_TSV.lines() {
        if line.starts_with('#') {
            provenance_seen |= line.contains("ANALYST");
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if !header_seen {
            assert_eq!(f[0], "file", "the first non-comment line is the header");
            assert_eq!(f.len(), 18, "the ledger has 18 columns");
            header_seen = true;
            continue;
        }
        assert_eq!(f.len(), 18, "ledger row has 18 columns: {line}");
        let num = |i: usize| -> i64 {
            f[i].parse()
                .unwrap_or_else(|e| panic!("column {i} of `{line}` is not a number: {e}"))
        };
        rows.push(LedgerRow {
            file: f[0].to_string(),
            index: num(1) as usize,
            id: f[2].to_string(),
            declared_len: num(3) as usize,
            pad_len: num(4) as usize,
            array_count: num(5) as u32,
            written_planes: num(6) as u32,
            top: num(7) as i32,
            left: num(8) as i32,
            bottom: num(9) as i32,
            right: num(10) as i32,
            w: num(11) as u32,
            h: num(12) as u32,
            depth: num(13) as u16,
            compression: num(14) as u8,
            decoded_bytes: num(15) as usize,
            sha256: f[16].to_string(),
            png_oracle: (!f[17].trim().is_empty()).then(|| f[17].trim().to_string()),
        });
    }
    assert!(
        provenance_seen,
        "the ledger must carry its provenance header"
    );
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_abr_corpus_profile_loads_with_its_headline_totals() {
        let p = profile();
        assert_eq!(p.total("files"), 9);
        assert_eq!(p.total("brush_presets"), 3215);
        assert_eq!(p.total("sampled_tip_records"), 3202);
        assert_eq!(p.total("descriptor_values"), 81926);
        assert_eq!(p.total("distinct_keys"), 102);
        assert_eq!(p.files().len(), 9);
    }

    #[test]
    fn image_abr_corpus_ledger_loads_one_row_per_record() {
        let l = ledger();
        assert_eq!(l.len(), profile().total("sampled_tip_records") as usize);
        assert_eq!(
            l.iter().filter(|r| r.png_oracle.is_some()).count(),
            238,
            "the published-PNG oracle covers 238 rows"
        );
        // Digests are lower-case hex, 32 bytes.
        for r in l {
            assert_eq!(r.sha256.len(), 64, "{}", r.id);
            assert!(
                r.sha256
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "{}",
                r.id
            );
        }
    }
}
