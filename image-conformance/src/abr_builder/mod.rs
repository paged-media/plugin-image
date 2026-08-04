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

//! The synthesized `.abr` fixture builder — an **INDEPENDENT** byte
//! emitter, the same discipline as [`crate::psd_builder`] and for the
//! same reason: if the emitter and the reader shared code, a shared bug
//! would hide. They share nothing but the on-disk format.
//!
//! Concretely it reuses `psd_builder`'s big-endian [`Emit`] sink and its
//! own greedy PackBits row encoder — neither of which `image-psd` uses —
//! and never touches `image_psd::descriptor`, `image_psd::vm_array` or
//! `image_psd::abr`.
//!
//! # What it can express, and why
//!
//! Everything the behaviour spec's trap index says a reader gets wrong.
//! In particular the builder can emit, per call:
//!
//! * a key or class id in **either dialect** ([`k4`] / [`klong`]), so
//!   one fixture can carry both forms of the same key;
//! * a virtual-memory-array-list with an arbitrary `array_count` and an
//!   arbitrary written slot, so the "264-byte skip" constant can be
//!   broken deliberately;
//! * a `Cnt ` as a `doub` **followed by more keys**, so a 4-byte read
//!   there shows up as garbage in the siblings rather than one bad
//!   field;
//! * a final section with or without its 4-byte pad, because both
//!   variants occur in real files and one of them walks a naive reader
//!   past EOF.
//!
//! All integers are big-endian. The one exception is the erodible
//! height map, which is little-endian float32 inside its opaque `tdta`
//! payload — [`height_map_tdta`] exists so a fixture states that
//! explicitly.

use crate::psd_builder::emit::{pack_bits_row, Emit};

/// Which spelling a key, class id or enum identifier is written in.
///
/// Modern Photoshop writes both, in the same file. A fixture that used
/// only one form would pass against a reader that handles only that one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// `u32 0` then the four bytes.
    FourCc,
    /// `u32 len` then `len` bytes.
    Long,
}

/// A key or class id plus the dialect to write it in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BKey {
    pub text: String,
    pub dialect: Dialect,
}

impl BKey {
    /// Byte equality on the key text, ignoring the dialect — the same
    /// rule the reader looks keys up by. Used by fixtures that replace a
    /// default key (e.g. flip a gate from false to true).
    pub fn matches(&self, key: &[u8]) -> bool {
        self.text.as_bytes() == key
    }
}

/// A four-character key (`length == 0` on the wire). Trailing spaces are
/// significant and must be written out: `k4("Nm  ")`.
pub fn k4(text: &str) -> BKey {
    assert_eq!(text.len(), 4, "a 4cc key is exactly four bytes: {text:?}");
    BKey {
        text: text.to_string(),
        dialect: Dialect::FourCc,
    }
}

/// A long-form key (a non-zero length prefix on the wire). Any key can
/// legitimately be written this way, including a four-character one.
pub fn klong(text: &str) -> BKey {
    BKey {
        text: text.to_string(),
        dialect: Dialect::Long,
    }
}

/// A descriptor value, by OSType.
#[derive(Debug, Clone, PartialEq)]
pub enum BValue {
    Obj(BDesc),
    GlobalObj(BDesc),
    List(Vec<BValue>),
    Doub(f64),
    UntF([u8; 4], f64),
    Text(String),
    /// A `TEXT` written with the trailing NUL some producers include in
    /// the code-unit count.
    TextNulTerminated(String),
    Enum(BKey, BKey),
    Long(i32),
    Comp(i64),
    Bool(bool),
    Class(String, BKey),
    Alis(Vec<u8>),
    Tdta(Vec<u8>),
    /// An OSType the reader is not expected to understand — for pinning
    /// the "named refusal, never a silent skip" behaviour.
    Unknown([u8; 4], Vec<u8>),
}

/// A descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct BDesc {
    pub class_name: String,
    pub class_id: BKey,
    pub items: Vec<(BKey, BValue)>,
}

impl BDesc {
    pub fn new(class_id: BKey) -> BDesc {
        BDesc {
            class_name: String::new(),
            class_id,
            items: Vec::new(),
        }
    }

    pub fn with_class_name(mut self, name: &str) -> BDesc {
        self.class_name = name.to_string();
        self
    }

    pub fn item(mut self, key: BKey, value: BValue) -> BDesc {
        self.items.push((key, value));
        self
    }
}

fn emit_key(e: &mut Emit, k: &BKey) {
    match k.dialect {
        Dialect::FourCc => {
            e.u32(0);
            e.raw(k.text.as_bytes());
        }
        Dialect::Long => {
            e.u32(k.text.len() as u32);
            e.raw(k.text.as_bytes());
        }
    }
}

fn emit_unicode(e: &mut Emit, s: &str, trailing_nul: bool) {
    let mut units: Vec<u16> = s.encode_utf16().collect();
    if trailing_nul {
        units.push(0);
    }
    e.u32(units.len() as u32);
    for u in units {
        e.u16(u);
    }
}

fn emit_value(e: &mut Emit, v: &BValue) {
    match v {
        BValue::Obj(d) => {
            e.raw(b"Objc");
            emit_descriptor(e, d);
        }
        BValue::GlobalObj(d) => {
            e.raw(b"GlbO");
            emit_descriptor(e, d);
        }
        BValue::List(items) => {
            e.raw(b"VlLs");
            e.u32(items.len() as u32);
            for it in items {
                emit_value(e, it);
            }
        }
        BValue::Doub(d) => {
            e.raw(b"doub");
            e.u64(d.to_bits());
        }
        BValue::UntF(unit, d) => {
            e.raw(b"UntF");
            e.raw(unit);
            e.u64(d.to_bits());
        }
        BValue::Text(s) => {
            e.raw(b"TEXT");
            emit_unicode(e, s, false);
        }
        BValue::TextNulTerminated(s) => {
            e.raw(b"TEXT");
            emit_unicode(e, s, true);
        }
        BValue::Enum(t, val) => {
            e.raw(b"enum");
            emit_key(e, t);
            emit_key(e, val);
        }
        BValue::Long(i) => {
            e.raw(b"long");
            e.u32(*i as u32);
        }
        BValue::Comp(i) => {
            e.raw(b"comp");
            e.u64(*i as u64);
        }
        BValue::Bool(b) => {
            e.raw(b"bool");
            e.u8(u8::from(*b));
        }
        BValue::Class(name, id) => {
            e.raw(b"type");
            emit_unicode(e, name, false);
            emit_key(e, id);
        }
        BValue::Alis(bytes) => {
            e.raw(b"alis");
            e.u32(bytes.len() as u32);
            e.raw(bytes);
        }
        BValue::Tdta(bytes) => {
            e.raw(b"tdta");
            e.u32(bytes.len() as u32);
            e.raw(bytes);
        }
        BValue::Unknown(ostype, bytes) => {
            e.raw(ostype);
            e.raw(bytes);
        }
    }
}

fn emit_descriptor(e: &mut Emit, d: &BDesc) {
    emit_unicode(e, &d.class_name, false);
    emit_key(e, &d.class_id);
    e.u32(d.items.len() as u32);
    for (k, v) in &d.items {
        emit_key(e, k);
        emit_value(e, v);
    }
}

/// Serialize a bare descriptor (no version word) — for unit-testing the
/// value tree without a container around it.
pub fn descriptor_bytes(d: &BDesc) -> Vec<u8> {
    let mut e = Emit::new();
    emit_descriptor(&mut e, d);
    e.into_bytes()
}

/// Serialize the "version and descriptor" wrapper.
pub fn versioned_descriptor_bytes(version: u32, d: &BDesc) -> Vec<u8> {
    let mut e = Emit::new();
    e.u32(version);
    emit_descriptor(&mut e, d);
    e.into_bytes()
}

/// A `gridSize²` little-endian float32 height map, as a `tdta` payload.
pub fn height_map_tdta(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// One sampled-tip record.
#[derive(Debug, Clone)]
pub struct SampleSpec {
    pub id: String,
    /// Stored y-first.
    pub top: i32,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
    pub depth: u16,
    /// 0 = raw, 1 = RLE.
    pub compression: u8,
    /// `w × h` coverage bytes, row-major.
    pub samples: Vec<u8>,
    /// How many array slots the container declares. Real files write 56;
    /// a fixture that writes something else proves the reader parses the
    /// structure instead of skipping a constant.
    pub array_count: u32,
    /// Which slot carries the plane. Real files write slot 55.
    pub slot: u32,
}

impl SampleSpec {
    /// A tip at the origin with the shape real files have (`array_count`
    /// 56, plane in slot 55 — the combination whose header measures the
    /// famous 264 bytes).
    pub fn new(id: &str, width: i32, height: i32, samples: Vec<u8>) -> SampleSpec {
        assert_eq!(samples.len(), (width * height) as usize);
        SampleSpec {
            id: id.to_string(),
            top: 0,
            left: 0,
            bottom: height,
            right: width,
            depth: 8,
            compression: 0,
            samples,
            array_count: 56,
            slot: 55,
        }
    }

    /// Place the tip's provenance rectangle away from the origin — real
    /// tips carry the bounding box of the region they were sampled from,
    /// which is NOT a canvas offset a painter should honour.
    pub fn at_origin(mut self, left: i32, top: i32) -> SampleSpec {
        let w = self.right - self.left;
        let h = self.bottom - self.top;
        self.left = left;
        self.top = top;
        self.right = left + w;
        self.bottom = top + h;
        self
    }

    pub fn rle(mut self) -> SampleSpec {
        self.compression = 1;
        self
    }

    pub fn with_layout(mut self, array_count: u32, slot: u32) -> SampleSpec {
        self.array_count = array_count;
        self.slot = slot;
        self
    }

    fn width(&self) -> i32 {
        self.right - self.left
    }

    fn height(&self) -> i32 {
        self.bottom - self.top
    }

    /// The plane payload as it sits in the array: raw bytes, or the
    /// up-front `h`-entry u16 row-length table followed by the packed
    /// rows back to back.
    fn payload(&self) -> Vec<u8> {
        if self.compression == 0 {
            return self.samples.clone();
        }
        let w = self.width() as usize;
        let mut e = Emit::new();
        let packed: Vec<Vec<u8>> = (0..self.height() as usize)
            .map(|y| pack_bits_row(&self.samples[y * w..(y + 1) * w]))
            .collect();
        for row in &packed {
            e.u16(row.len() as u16);
        }
        for row in &packed {
            e.raw(row);
        }
        e.into_bytes()
    }

    /// The record body: a pad-to-1 Pascal string id, then the virtual
    /// memory array list.
    fn body(&self) -> Vec<u8> {
        let mut e = Emit::new();
        e.u8(self.id.len() as u8);
        e.raw(self.id.as_bytes());

        let payload = self.payload();
        // Everything from `vm_length`'s successor to the end of the last
        // array — computed, not guessed, so a reader that reconciles the
        // field is actually being tested.
        let slots = self.array_count + 2;
        let vm_len = 16 + 4 + (slots as usize * 4) + (4 + 4 + 16 + 2 + 1 + payload.len());
        e.u32(0x0001_0000);
        e.u32(3);
        e.u32(vm_len as u32);
        e.i32(self.top);
        e.i32(self.left);
        e.i32(self.bottom);
        e.i32(self.right);
        e.u32(self.array_count);
        for i in 0..slots {
            if i != self.slot {
                e.u32(0);
                continue;
            }
            e.u32(1);
            e.u32((4 + 16 + 2 + 1 + payload.len()) as u32);
            e.u32(self.depth as u32);
            e.i32(self.top);
            e.i32(self.left);
            e.i32(self.bottom);
            e.i32(self.right);
            e.u16(self.depth);
            e.u8(self.compression);
            e.raw(&payload);
        }
        e.into_bytes()
    }
}

/// A whole `.abr` file.
#[derive(Debug, Clone)]
pub struct AbrBuilder {
    version: i16,
    minor_version: i16,
    samples: Vec<SampleSpec>,
    /// Emitted only when set — a file may have no `samp` section, or an
    /// empty one, and both are normal.
    emit_samp: bool,
    brushes: Vec<BDesc>,
    emit_desc: bool,
    hierarchy: Option<Vec<BDesc>>,
    patt: Option<Vec<u8>>,
    extra_sections: Vec<([u8; 4], Vec<u8>)>,
    descriptor_version: u32,
    pad_last_section: bool,
}

impl Default for AbrBuilder {
    fn default() -> Self {
        AbrBuilder::new()
    }
}

impl AbrBuilder {
    pub fn new() -> AbrBuilder {
        AbrBuilder {
            version: 6,
            minor_version: 2,
            samples: Vec::new(),
            emit_samp: false,
            brushes: Vec::new(),
            emit_desc: false,
            hierarchy: None,
            patt: None,
            extra_sections: Vec::new(),
            descriptor_version: 16,
            pad_last_section: true,
        }
    }

    pub fn version(mut self, major: i16, minor: i16) -> Self {
        self.version = major;
        self.minor_version = minor;
        self
    }

    pub fn descriptor_version(mut self, v: u32) -> Self {
        self.descriptor_version = v;
        self
    }

    /// In 5 of 9 real files the last section is NOT followed by its pad,
    /// so a reader that applies it unconditionally walks 1–3 bytes past
    /// EOF and reports a good file as corrupt.
    pub fn pad_last_section(mut self, pad: bool) -> Self {
        self.pad_last_section = pad;
        self
    }

    pub fn sample(mut self, s: SampleSpec) -> Self {
        self.samples.push(s);
        self.emit_samp = true;
        self
    }

    /// Emit a `samp` section with `size == 0` — normal, not an error.
    pub fn empty_samp(mut self) -> Self {
        self.emit_samp = true;
        self
    }

    pub fn brush(mut self, d: BDesc) -> Self {
        self.brushes.push(d);
        self.emit_desc = true;
        self
    }

    /// Emit a `desc` section whose `Brsh` list is empty.
    pub fn empty_desc(mut self) -> Self {
        self.emit_desc = true;
        self
    }

    pub fn hierarchy(mut self, nodes: Vec<BDesc>) -> Self {
        self.hierarchy = Some(nodes);
        self
    }

    pub fn patt(mut self, body: Vec<u8>) -> Self {
        self.patt = Some(body);
        self
    }

    /// An extra section of any type, emitted after the standard ones.
    pub fn extra_section(mut self, kind: &[u8; 4], body: Vec<u8>) -> Self {
        self.extra_sections.push((*kind, body));
        self
    }

    pub fn build(&self) -> Vec<u8> {
        let mut sections: Vec<([u8; 4], Vec<u8>)> = Vec::new();

        if self.emit_samp {
            let mut e = Emit::new();
            for s in &self.samples {
                let body = s.body();
                let rounded = body.len().div_ceil(4) * 4;
                // The DECLARED length excludes the pad; the record
                // occupies the rounded length. Real files declare
                // lengths that are frequently 1–3 short of a multiple
                // of 4, which is exactly what this produces.
                e.u32(body.len() as u32);
                e.raw(&body);
                e.raw(&vec![0u8; rounded - body.len()]);
            }
            sections.push((*b"samp", e.into_bytes()));
        }

        if let Some(body) = &self.patt {
            sections.push((*b"patt", body.clone()));
        }

        if self.emit_desc {
            let root = BDesc::new(k4("null")).item(
                k4("Brsh"),
                BValue::List(self.brushes.iter().cloned().map(BValue::Obj).collect()),
            );
            sections.push((
                *b"desc",
                versioned_descriptor_bytes(self.descriptor_version, &root),
            ));
        }

        if let Some(nodes) = &self.hierarchy {
            let root = BDesc::new(k4("null")).item(
                klong("hierarchy"),
                BValue::List(nodes.iter().cloned().map(BValue::Obj).collect()),
            );
            sections.push((
                *b"phry",
                versioned_descriptor_bytes(self.descriptor_version, &root),
            ));
        }

        for (kind, body) in &self.extra_sections {
            sections.push((*kind, body.clone()));
        }

        let mut e = Emit::new();
        e.i16(self.version);
        e.i16(self.minor_version);
        let last = sections.len().saturating_sub(1);
        for (i, (kind, body)) in sections.iter().enumerate() {
            e.raw(b"8BIM");
            e.raw(kind);
            e.u32(body.len() as u32);
            e.raw(body);
            let pad = (4 - (body.len() % 4)) % 4;
            if i != last || self.pad_last_section {
                e.raw(&vec![0u8; pad]);
            }
        }
        e.into_bytes()
    }
}

/// A minimal, well-formed `sampledBrush` tip descriptor carrying the six
/// shared keys plus the sample link.
pub fn sampled_tip(name: &str, sampled_id: &str, diameter: f64) -> BDesc {
    BDesc::new(klong("sampledBrush"))
        .item(k4("Dmtr"), BValue::UntF(*b"#Pxl", diameter))
        .item(k4("Angl"), BValue::UntF(*b"#Ang", 0.0))
        .item(k4("Spcn"), BValue::UntF(*b"#Prc", 25.0))
        .item(k4("Intr"), BValue::Bool(true))
        .item(klong("flipX"), BValue::Bool(false))
        .item(klong("flipY"), BValue::Bool(false))
        .item(k4("Rndn"), BValue::UntF(*b"#Prc", 100.0))
        .item(k4("Nm  "), BValue::Text(name.to_string()))
        .item(klong("sampledData"), BValue::Text(sampled_id.to_string()))
}

/// A brush preset wrapping `tip`, with the keys every real preset has.
pub fn brush_preset(name: &str, tip: BDesc) -> BDesc {
    BDesc::new(klong("brushPreset"))
        .item(k4("Nm  "), BValue::Text(name.to_string()))
        .item(k4("Brsh"), BValue::Obj(tip))
        .item(k4("Wtdg"), BValue::Bool(false))
        .item(k4("Nose"), BValue::Bool(false))
        // Two keys no published vocabulary had. They must survive.
        .item(klong("Rpt"), BValue::Bool(true))
        .item(
            klong("brushGroup"),
            BValue::Obj(
                BDesc::new(klong("brushGroup")).item(klong("useBrushGroup"), BValue::Bool(false)),
            ),
        )
        .item(klong("useTipDynamics"), BValue::Bool(false))
        .item(klong("useScatter"), BValue::Bool(false))
        .item(klong("useTexture"), BValue::Bool(false))
        .item(klong("useColorDynamics"), BValue::Bool(false))
        .item(klong("usePaintDynamics"), BValue::Bool(false))
        .item(klong("useBrushPose"), BValue::Bool(false))
        .item(
            klong("dualBrush"),
            BValue::Obj(
                BDesc::new(klong("dualBrush")).item(klong("useDualBrush"), BValue::Bool(false)),
            ),
        )
}

/// The dynamics primitive: class `brVr`, control/fade/jitter, plus an
/// optional floor.
pub fn dynamics(control: i32, fade: i32, jitter: f64, minimum: Option<f64>) -> BDesc {
    let mut d = BDesc::new(k4("brVr"))
        .item(k4("bVTy"), BValue::Long(control))
        .item(k4("fStp"), BValue::Long(fade))
        .item(klong("jitter"), BValue::UntF(*b"#Prc", jitter));
    if let Some(m) = minimum {
        d = d.item(k4("Mnm "), BValue::UntF(*b"#Prc", m));
    }
    d
}
