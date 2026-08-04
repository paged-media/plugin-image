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

//! The PSD **descriptor value tree** — the OSType-tagged, key-addressed
//! structure Photoshop uses to serialize almost every non-pixel setting
//! it has (`.abr` brush presets today; adjustment layers, layer effects,
//! smart-object placement and the text engine tomorrow).
//!
//! It lives here, in `image-psd`, and not in an ABR-only module, because
//! the ABR behaviour spec's own §4.3 finding is that `image-psd` had no
//! descriptor parser at all and that every future *modelled* (rather than
//! merely *preserved*) PSD block needs the same one.
//!
//! # Provenance
//!
//! Adobe Photoshop File Formats Specification, "Descriptor structure"
//! (the header, the `key-or-4cc` convention, the OSType table and the
//! unit codes) — `[PUB]`; the dual legacy/long-form dialect and the
//! localisable-string form are recorded in the behaviour spec
//! `thoughts/docs/paged/plugin-image/abr-brush-format-spec.md` §4.1, §6.4
//! and §9. `references/` is never read by implementers of this crate.
//!
//! # The preservation invariant is NOT weakened by this module
//!
//! Parsing a descriptor is a pure read. The opaque-bytes path stays
//! authoritative for re-emission (crate docs, strategy 2); a decoded
//! tree is a *derived view*. Nothing here writes.
//!
//! # Two dialects, forever
//!
//! Every key, class id and enum identifier arrives in one of two
//! spellings — a legacy 4-character code (`Nrml`) or a modern long-form
//! identifier (`normal`) — and **both occur inside the same file**
//! (spec §4.1 `[OBS]`: 45,706 long-form item keys against 36,220
//! 4-byte ones across the fixture corpus). [`Key`] normalises the
//! *framing* of the two (a `Key` compares by its bytes, so a `Nm  `
//! written either way looks up the same) and remembers which form was
//! on disk. Normalising the *vocabulary* — `Nrml` vs `normal` — is a
//! per-table job and lives with each table (see `abr::blend`).

use crate::reader::ByteReader;
use crate::{PsdError, Result};

/// Longest long-form key accepted. Real keys are short identifiers;
/// this is a hostile-input guard, not a format limit.
pub const MAX_KEY_LEN: usize = 1024;

/// Deepest descriptor nesting accepted. `.abr`'s deepest real nesting is
/// about five (`Brsh` list → preset → `dualBrush` → `Brsh` → dynamics);
/// the limit exists so a crafted file cannot drive the recursive reader
/// into a stack overflow.
pub const MAX_DESCRIPTOR_DEPTH: u32 = 64;

/// The descriptor-wrapper version PSD writes (`[PUB]`).
pub const DESCRIPTOR_VERSION: u32 = 16;

/// Unit codes carried by a [`DescriptorValue::UnitFloat`] (`[PUB]`).
/// Only `#Prc`, `#Pxl` and `#Ang` were observed in `.abr` (spec §4.1).
pub mod units {
    pub const ANGLE: [u8; 4] = *b"#Ang";
    pub const PERCENT: [u8; 4] = *b"#Prc";
    pub const PIXELS: [u8; 4] = *b"#Pxl";
    pub const DENSITY: [u8; 4] = *b"#Rsl";
    pub const DISTANCE: [u8; 4] = *b"#Rlt";
    pub const NONE: [u8; 4] = *b"#Nne";
    pub const POINTS: [u8; 4] = *b"#Pnt";
    pub const MILLIMETRES: [u8; 4] = *b"#Mlm";
}

/// A descriptor key or class id, as stored.
///
/// The wire convention is inverted-looking and is the single most common
/// way a first descriptor reader goes wrong (spec §4.1): a `u32` length
/// of **0** means "the next four bytes ARE the key", and any other value
/// means "the next `length` bytes are the key". Length 0 is *not* an
/// empty key.
///
/// Equality is over the key BYTES only, so the same key reaching us in
/// either dialect resolves identically. Trailing spaces are part of the
/// key and are never trimmed (`Nm  `, `Mnm `, `H   ` — spec §6.1).
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct Key {
    raw: Vec<u8>,
    long_form: bool,
}

impl Key {
    /// A legacy 4-byte key (`length == 0` on the wire).
    pub fn four(code: &[u8; 4]) -> Key {
        Key {
            raw: code.to_vec(),
            long_form: false,
        }
    }

    /// A modern long-form key (a non-zero length prefix on the wire).
    pub fn long(name: &str) -> Key {
        Key {
            raw: name.as_bytes().to_vec(),
            long_form: true,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.raw
    }

    /// `true` when the key arrived length-prefixed rather than as a bare
    /// four-character code. Recorded, never acted on.
    pub fn is_long_form(&self) -> bool {
        self.long_form
    }

    pub fn text_lossy(&self) -> String {
        String::from_utf8_lossy(&self.raw).into_owned()
    }

    pub fn matches(&self, key: &[u8]) -> bool {
        self.raw == key
    }
}

impl std::fmt::Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}{}",
            self.text_lossy(),
            if self.long_form { "" } else { "/4cc" }
        )
    }
}

/// One node of the descriptor value tree.
///
/// Every OSType the Adobe specification lists and that can be framed
/// from the bytes alone is modelled. The three that cannot —
/// `obj ` (reference), `ObAr` and `UnFl` — are rejected with a named
/// error rather than skipped, because a descriptor item is **not**
/// length-delimited: a reader that cannot decode a value also cannot
/// find the next one. Spec §4.1 `[OBS]`: none of the three occurred in
/// ~89,000 descriptor values, so this is a hedge, not a gap in practice.
#[derive(Debug, Clone, PartialEq)]
pub enum DescriptorValue {
    /// `Objc` — a nested descriptor.
    Descriptor(Descriptor),
    /// `GlbO` — a global object; the same payload as `Objc`.
    GlobalObject(Descriptor),
    /// `VlLs` — a list whose elements are INDIVIDUALLY typed (spec §4.1:
    /// a list is not homogeneous by construction).
    List(Vec<DescriptorValue>),
    /// `doub` — 8-byte IEEE double.
    Double(f64),
    /// `UntF` — a 4-byte unit code plus an 8-byte double.
    UnitFloat { unit: [u8; 4], value: f64 },
    /// `TEXT` — a unicode string. Stored RAW; see [`localized_display`]
    /// for the `$$$/…=Default` form.
    Text(String),
    /// `enum` — a type key plus a value key, each independently either
    /// dialect. There is no `.` in the bytes (spec §9 CORRECTION).
    Enum { type_key: Key, value: Key },
    /// `long` — 4-byte signed.
    Integer(i32),
    /// `comp` — 8-byte signed.
    LargeInteger(i64),
    /// `bool` — one byte.
    Bool(bool),
    /// `type` / `GlbC` — a class name plus a class id.
    Class { name: String, id: Key },
    /// `alis` — length-prefixed opaque bytes.
    Alias(Vec<u8>),
    /// `tdta` — length-prefixed opaque bytes. The interior obeys whatever
    /// convention the producing subsystem used and is NOT covered by the
    /// PSD big-endian rule (spec §5.5: the erodible height map is
    /// little-endian float32 inside a `tdta`).
    RawData(Vec<u8>),
}

impl DescriptorValue {
    /// The OSType this value was read from (or would be written as).
    pub fn ostype(&self) -> [u8; 4] {
        match self {
            DescriptorValue::Descriptor(_) => *b"Objc",
            DescriptorValue::GlobalObject(_) => *b"GlbO",
            DescriptorValue::List(_) => *b"VlLs",
            DescriptorValue::Double(_) => *b"doub",
            DescriptorValue::UnitFloat { .. } => *b"UntF",
            DescriptorValue::Text(_) => *b"TEXT",
            DescriptorValue::Enum { .. } => *b"enum",
            DescriptorValue::Integer(_) => *b"long",
            DescriptorValue::LargeInteger(_) => *b"comp",
            DescriptorValue::Bool(_) => *b"bool",
            DescriptorValue::Class { .. } => *b"type",
            DescriptorValue::Alias(_) => *b"alis",
            DescriptorValue::RawData(_) => *b"tdta",
        }
    }

    /// `Objc` or `GlbO` alike — both carry a descriptor.
    pub fn as_descriptor(&self) -> Option<&Descriptor> {
        match self {
            DescriptorValue::Descriptor(d) | DescriptorValue::GlobalObject(d) => Some(d),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[DescriptorValue]> {
        match self {
            DescriptorValue::List(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            DescriptorValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            DescriptorValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// The raw text, exactly as stored.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            DescriptorValue::Text(s) => Some(s),
            _ => None,
        }
    }

    /// The text with Adobe's localisable-string wrapper removed when it
    /// is present (spec §6.4). One rule, one place.
    pub fn text_display(&self) -> Option<&str> {
        self.as_text().map(localized_display)
    }

    /// A plain number, whatever numeric OSType carried it.
    ///
    /// Deliberately permissive across `doub`/`long`/`comp`: the format
    /// mixes widths for semantically identical quantities (spec §7.2:
    /// `Smoo` is a `long` while `smoothingValue` is a `doub`), and the
    /// single most destructive type error found in the corpus was
    /// reading `Cnt ` — a `doub` — as a `long` (spec §6.3). Reading is
    /// framed by the OSType on the wire, so accepting either here cannot
    /// desynchronise anything; it only stops a caller inventing a
    /// second accessor per width.
    pub fn as_number(&self) -> Option<f64> {
        match self {
            DescriptorValue::Double(d) => Some(*d),
            DescriptorValue::Integer(i) => Some(*i as f64),
            DescriptorValue::LargeInteger(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// A unit float, with its unit code. Callers must check the unit —
    /// spec §4.2's rules are per-unit and the strict variants reject.
    pub fn as_unit_float(&self) -> Option<([u8; 4], f64)> {
        match self {
            DescriptorValue::UnitFloat { unit, value } => Some((*unit, *value)),
            _ => None,
        }
    }

    /// The value of a unit float whose unit is exactly `unit`.
    pub fn as_unit(&self, unit: [u8; 4]) -> Option<f64> {
        match self {
            DescriptorValue::UnitFloat { unit: u, value } if *u == unit => Some(*value),
            _ => None,
        }
    }

    pub fn as_enum(&self) -> Option<(&Key, &Key)> {
        match self {
            DescriptorValue::Enum { type_key, value } => Some((type_key, value)),
            _ => None,
        }
    }

    pub fn as_raw_data(&self) -> Option<&[u8]> {
        match self {
            DescriptorValue::RawData(b) | DescriptorValue::Alias(b) => Some(b),
            _ => None,
        }
    }
}

/// A descriptor: an optional class name, a class id, and an ORDERED list
/// of key/value items.
///
/// Items are a `Vec`, not a map, on purpose (spec §7.1 trap 8): order is
/// information, duplicate keys are representable, and **nothing is
/// dropped**. A fixed struct silently loses keys the modeller did not
/// know about — the fixture experiment found two such keys (`Rpt`,
/// `brushGroup`) present on every brush in every file.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Descriptor {
    /// May be empty; frequently is.
    pub class_name: String,
    pub class_id: Key,
    pub items: Vec<(Key, DescriptorValue)>,
}

impl Descriptor {
    /// First value stored under `key` (byte-exact, both dialects).
    pub fn get(&self, key: &[u8]) -> Option<&DescriptorValue> {
        self.items
            .iter()
            .find(|(k, _)| k.matches(key))
            .map(|(_, v)| v)
    }

    pub fn contains(&self, key: &[u8]) -> bool {
        self.get(key).is_some()
    }

    pub fn descriptor(&self, key: &[u8]) -> Option<&Descriptor> {
        self.get(key).and_then(|v| v.as_descriptor())
    }

    pub fn list(&self, key: &[u8]) -> Option<&[DescriptorValue]> {
        self.get(key).and_then(|v| v.as_list())
    }

    pub fn bool(&self, key: &[u8]) -> Option<bool> {
        self.get(key).and_then(|v| v.as_bool())
    }

    pub fn i32(&self, key: &[u8]) -> Option<i32> {
        self.get(key).and_then(|v| v.as_i32())
    }

    pub fn number(&self, key: &[u8]) -> Option<f64> {
        self.get(key).and_then(|v| v.as_number())
    }

    pub fn text(&self, key: &[u8]) -> Option<&str> {
        self.get(key).and_then(|v| v.as_text())
    }

    pub fn text_display(&self, key: &[u8]) -> Option<&str> {
        self.get(key).and_then(|v| v.text_display())
    }

    pub fn unit_float(&self, key: &[u8]) -> Option<([u8; 4], f64)> {
        self.get(key).and_then(|v| v.as_unit_float())
    }

    pub fn unit(&self, key: &[u8], unit: [u8; 4]) -> Option<f64> {
        self.get(key).and_then(|v| v.as_unit(unit))
    }

    pub fn enum_value(&self, key: &[u8]) -> Option<(&Key, &Key)> {
        self.get(key).and_then(|v| v.as_enum())
    }

    pub fn raw_data(&self, key: &[u8]) -> Option<&[u8]> {
        self.get(key).and_then(|v| v.as_raw_data())
    }
}

/// Adobe's localisable-string form: `$$$/path/to/key=Default English`.
///
/// A UI that shows the raw value shows the whole path (spec §6.4 trap —
/// four of the five observed pattern names are in this form). Anything
/// not in the form is returned unchanged.
pub fn localized_display(s: &str) -> &str {
    match s.strip_prefix("$$$/") {
        Some(rest) => match rest.find('=') {
            Some(i) => &rest[i + 1..],
            None => s,
        },
        None => s,
    }
}

// ── reading ──────────────────────────────────────────────────────────

fn malformed(detail: String) -> PsdError {
    PsdError::Malformed {
        section: "descriptor",
        detail,
    }
}

/// Read a `key-or-4cc` (spec §4.1). Length 0 ⇒ the next four bytes ARE
/// the key; otherwise the next `length` bytes are.
pub fn read_key(r: &mut ByteReader) -> Result<Key> {
    let len = r.u32()? as usize;
    if len == 0 {
        return Ok(Key {
            raw: r.fourcc()?.to_vec(),
            long_form: false,
        });
    }
    if len > MAX_KEY_LEN {
        return Err(malformed(format!(
            "key length {len} exceeds the {MAX_KEY_LEN}-byte guard"
        )));
    }
    Ok(Key {
        raw: r.take(len)?.to_vec(),
        long_form: true,
    })
}

/// Read a descriptor unicode string: a `u32` count of UTF-16 code units
/// followed by that many big-endian units.
///
/// A single trailing NUL is trimmed — producers differ on whether it is
/// included in the count, exactly as they do for `luni` (see
/// `model::addl`). Decoding is LOSSY for unpaired surrogates: a name
/// that cannot be represented becomes U+FFFD rather than failing the
/// whole file.
pub fn read_unicode_string(r: &mut ByteReader) -> Result<String> {
    let count = r.u32()? as usize;
    if count > r.remaining() / 2 {
        return Err(malformed(format!(
            "unicode string of {count} code units exceeds the {} byte(s) available",
            r.remaining()
        )));
    }
    let mut units = Vec::with_capacity(count);
    for _ in 0..count {
        units.push(r.u16()?);
    }
    if matches!(units.last(), Some(0)) {
        units.pop();
    }
    Ok(String::from_utf16_lossy(&units))
}

/// Read the "version and descriptor" wrapper PSD uses wherever a
/// descriptor is embedded in a length-framed block (spec §3.1). Returns
/// the version word alongside the tree; the caller decides what to do
/// with a version other than [`DESCRIPTOR_VERSION`] (we do not reject —
/// the version has only ever been 16, and failing on a 17 would be
/// guessing).
pub fn read_versioned_descriptor(r: &mut ByteReader) -> Result<(u32, Descriptor)> {
    let version = r.u32()?;
    let d = read_descriptor(r, 0)?;
    Ok((version, d))
}

/// Read a descriptor: class name, class id, item count, items.
///
/// The class id must be read, never skipped — some PSD-family readers
/// treat it as optional and desynchronise here (spec §3.1).
pub fn read_descriptor(r: &mut ByteReader, depth: u32) -> Result<Descriptor> {
    if depth > MAX_DESCRIPTOR_DEPTH {
        return Err(malformed(format!(
            "descriptor nesting deeper than {MAX_DESCRIPTOR_DEPTH}"
        )));
    }
    let class_name = read_unicode_string(r)?;
    let class_id = read_key(r)?;
    let count = r.u32()? as usize;
    // Smallest possible item: a 4-byte length prefix of 0, a 4-byte key
    // and a 4-byte OSType. Anything claiming more items than the input
    // could hold is malformed, and bounding it stops a crafted count
    // from driving a huge speculative allocation.
    const MIN_ITEM_BYTES: usize = 12;
    if count.saturating_mul(MIN_ITEM_BYTES) > r.remaining() {
        return Err(malformed(format!(
            "descriptor claims {count} items, more than the {} byte(s) available",
            r.remaining()
        )));
    }
    let mut items = Vec::with_capacity(count);
    for _ in 0..count {
        let key = read_key(r)?;
        let value = read_value(r, depth + 1)?;
        items.push((key, value));
    }
    Ok(Descriptor {
        class_name,
        class_id,
        items,
    })
}

/// Read one OSType-tagged value.
pub fn read_value(r: &mut ByteReader, depth: u32) -> Result<DescriptorValue> {
    let ostype = r.fourcc()?;
    match &ostype {
        b"Objc" => Ok(DescriptorValue::Descriptor(read_descriptor(r, depth)?)),
        b"GlbO" => Ok(DescriptorValue::GlobalObject(read_descriptor(r, depth)?)),
        b"VlLs" => {
            let count = r.u32()? as usize;
            // Smallest element: a 4-byte OSType plus a 1-byte payload.
            const MIN_ELEM_BYTES: usize = 5;
            if count.saturating_mul(MIN_ELEM_BYTES) > r.remaining() {
                return Err(malformed(format!(
                    "list claims {count} elements, more than the {} byte(s) available",
                    r.remaining()
                )));
            }
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                // Each element carries its OWN OSType — a list is not
                // homogeneous by construction (spec §4.1 trap).
                out.push(read_value(r, depth + 1)?);
            }
            Ok(DescriptorValue::List(out))
        }
        b"doub" => Ok(DescriptorValue::Double(f64::from_bits(r.u64()?))),
        b"UntF" => {
            let unit = r.fourcc()?;
            Ok(DescriptorValue::UnitFloat {
                unit,
                value: f64::from_bits(r.u64()?),
            })
        }
        b"TEXT" => Ok(DescriptorValue::Text(read_unicode_string(r)?)),
        b"enum" => {
            let type_key = read_key(r)?;
            let value = read_key(r)?;
            Ok(DescriptorValue::Enum { type_key, value })
        }
        b"long" => Ok(DescriptorValue::Integer(r.i32()?)),
        b"comp" => Ok(DescriptorValue::LargeInteger(r.u64()? as i64)),
        b"bool" => Ok(DescriptorValue::Bool(r.u8()? != 0)),
        b"type" | b"GlbC" => {
            let name = read_unicode_string(r)?;
            let id = read_key(r)?;
            Ok(DescriptorValue::Class { name, id })
        }
        b"alis" => {
            let n = r.u32()? as usize;
            Ok(DescriptorValue::Alias(r.take(n)?.to_vec()))
        }
        b"tdta" => {
            let n = r.u32()? as usize;
            Ok(DescriptorValue::RawData(r.take(n)?.to_vec()))
        }
        // `obj ` (reference), `ObAr` and `UnFl`: their payload layouts
        // are not established by any source available to this crate, and
        // a descriptor item is not length-delimited — so an undecodable
        // value cannot be skipped to reach the next one. Refuse, by name,
        // rather than guess a layout or desynchronise silently. None of
        // the three occurred in ~89,000 observed `.abr` values (§4.1).
        _ => Err(PsdError::Unsupported(format!(
            "descriptor OSType {:?} at offset {} — layout not established; \
             a descriptor item is not length-delimited, so it cannot be skipped",
            String::from_utf8_lossy(&ostype),
            r.pos()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_psd_descriptor_key_accepts_both_dialects() {
        // length 0 ⇒ the next four bytes ARE the key.
        let four = [0, 0, 0, 0, b'N', b'm', b' ', b' '];
        let mut r = ByteReader::new(&four);
        let k = read_key(&mut r).unwrap();
        assert_eq!(k.as_bytes(), b"Nm  ");
        assert!(!k.is_long_form());

        // non-zero length ⇒ that many bytes of ASCII key.
        let long = [
            0, 0, 0, 11, b's', b'a', b'm', b'p', b'l', b'e', b'd', b'D', b'a', b't', b'a',
        ];
        let mut r = ByteReader::new(&long);
        let k = read_key(&mut r).unwrap();
        assert_eq!(k.as_bytes(), b"sampledData");
        assert!(k.is_long_form());
    }

    #[test]
    fn image_psd_descriptor_key_equality_ignores_dialect() {
        // The SAME key in the two spellings must resolve identically:
        // both forms occur in one file (spec §4.1 [OBS]). Lookup is by
        // BYTES, so `matches` cannot see the dialect.
        assert!(Key::long("Nm  ").matches(b"Nm  "));
        assert!(Key::four(b"Nm  ").matches(b"Nm  "));
        assert_eq!(Key::long("Nm  ").as_bytes(), Key::four(b"Nm  ").as_bytes());
        // …while the dialect itself is still recorded.
        assert!(Key::long("Nm  ").is_long_form());
        assert!(!Key::four(b"Nm  ").is_long_form());
    }

    #[test]
    fn image_psd_descriptor_trailing_spaces_are_significant() {
        assert!(!Key::four(b"Nm  ").matches(b"Nm"));
        assert!(!Key::four(b"Mnm ").matches(b"Mnm"));
        assert!(!Key::four(b"H   ").matches(b"H"));
    }

    #[test]
    fn image_psd_descriptor_key_length_guard() {
        let mut bytes = vec![0, 0, 0xFF, 0xFF];
        bytes.extend(std::iter::repeat_n(b'x', 16));
        let mut r = ByteReader::new(&bytes);
        assert!(read_key(&mut r).is_err());
    }

    #[test]
    fn image_psd_descriptor_localized_display_strips_the_token() {
        assert_eq!(
            localized_display("$$$/Presets/Patterns/Patterns_pat/GrayGranite=Gray Granite"),
            "Gray Granite"
        );
        assert_eq!(localized_display("Gray Granite"), "Gray Granite");
        // No `=` ⇒ not the form; leave it alone rather than truncating.
        assert_eq!(localized_display("$$$/no/equals"), "$$$/no/equals");
    }

    #[test]
    fn image_psd_descriptor_unsupported_ostype_is_named_not_skipped() {
        let bytes = *b"obj \x00\x00\x00\x00";
        let mut r = ByteReader::new(&bytes);
        let err = read_value(&mut r, 0).unwrap_err();
        match err {
            PsdError::Unsupported(msg) => assert!(msg.contains("obj "), "{msg}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn image_psd_descriptor_item_count_is_bounded() {
        // class name (empty) + class id (4cc) + an absurd item count.
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_be_bytes()); // class name: 0 units
        b.extend_from_slice(&0u32.to_be_bytes()); // key length 0
        b.extend_from_slice(b"null");
        b.extend_from_slice(&0xFFFF_FFFFu32.to_be_bytes()); // item count
        let mut r = ByteReader::new(&b);
        assert!(read_descriptor(&mut r, 0).is_err());
    }
}
