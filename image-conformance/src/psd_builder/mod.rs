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

//! The synthesized-PSD fixture builder — an INDEPENDENT byte emitter
//! (its own big-endian writer and its own PackBits encoder), a
//! deliberately separate code path from image-psd's production writer
//! so round-trip tests are never self-referential. The M0 corpus (the
//! 11 named fixtures) lives in `fixtures`.
//!
//! Lands with M0 fan-out unit U8 (emit/layers/channels/fixtures).
//!
//! # The byte layout (brief §1–§12)
//!
//! [`PsdBuilder`] assembles a whole file: header → color mode data →
//! image resources → layer & mask info → merged composite. It owns the
//! flat layer list (groups expand to folder + bounding-divider records,
//! brief §7) and emits with [`emit::Emit`] only — never image-psd's
//! writer. The companion [`FixtureManifest`] is the parser's answer key:
//! exactly what a faithful parse must recover from the bytes.

pub mod channels;
pub mod emit;
pub mod fixtures;
pub mod layers;

use image_psd::container::{Container, LenWidth};
use image_psd::model::header::SIGNATURE;
use image_psd::model::Compression;

pub use channels::Plane;
pub use fixtures::{ExpectedAddl, ExpectedLayer, ExpectedResource, FixtureManifest};
pub use layers::{AddlSpec, ChannelSpec, LayerSpec, MaskSpec, RealMask, GROUP_DIVIDER_NAME};

use channels::encode_composite;
use emit::Emit;
use layers::{addl_len_width, addl_parts, emit_layer_channels, emit_layer_record};

/// A fluent synthesizer for one RGB/8-bit PSD or PSB file (brief §1).
/// Resources and layers accumulate in declaration order; [`build`](Self::build)
/// frames everything per the brief and returns the file bytes. The flat
/// layer list is built bottom-up: PSD stores bottom-most first, so a
/// group's bounding divider is pushed, then members, then the folder
/// record (brief §7) — `group_open`/`group_close` manage that ordering.
#[derive(Debug, Clone)]
pub struct PsdBuilder {
    container: Container,
    width: u32,
    height: u32,
    channels: u16,
    /// `8BIM` image-resource blocks: (id, name, data) emitted verbatim
    /// by the section framer.
    resources: Vec<ResourceSpec>,
    /// The flat layer list in on-disk order (bottom-most first).
    layers: Vec<LayerSpec>,
    /// Open-group stack: each entry is the folder record to append once
    /// the group closes (LIFO so nesting reverses correctly).
    open_groups: Vec<PendingGroup>,
    /// Negative layer count flag (brief §4a): transparency in the merged
    /// result. Set when a composite includes an alpha plane.
    transparency_in_merged: bool,
    /// Document-level additional-layer-info blocks (brief §4c).
    doc_addl: Vec<AddlSpec>,
    composite: Option<(Compression, Vec<Plane>)>,
}

#[derive(Debug, Clone)]
struct ResourceSpec {
    id: u16,
    name: String,
    data: Vec<u8>,
}

/// A folder record deferred until its `group_close` (brief §7).
#[derive(Debug, Clone)]
struct PendingGroup {
    name: String,
    /// 1 = open folder, 2 = closed folder.
    kind: u32,
}

impl PsdBuilder {
    /// A new RGB/8-bit document. `channels` is the merged-composite
    /// channel count (3 = RGB, 4 = RGBA); per-layer channel counts are
    /// independent and declared per layer.
    pub fn new(container: Container, width: u32, height: u32, channels: u16) -> Self {
        PsdBuilder {
            container,
            width,
            height,
            channels,
            resources: Vec::new(),
            layers: Vec::new(),
            open_groups: Vec::new(),
            transparency_in_merged: false,
            doc_addl: Vec::new(),
            composite: None,
        }
    }

    /// An ICC profile resource (id 1039, brief §3).
    pub fn resource_icc(mut self, bytes: Vec<u8>) -> Self {
        self.resources.push(ResourceSpec {
            id: image_psd::model::resources::RES_ICC_PROFILE,
            name: String::new(),
            data: bytes,
        });
        self
    }

    /// A resolution resource (id 1005, brief §3) at `ppi` for both axes,
    /// units = ppi/inches. Encodes the 16-byte fixed-point record.
    pub fn resource_resolution(mut self, ppi: f64) -> Self {
        let fixed = (ppi * 65536.0).round() as u32;
        let mut e = Emit::new();
        e.u32(fixed).u16(1).u16(1); // h_res, h_unit=ppi, width_unit=inches
        e.u32(fixed).u16(1).u16(1); // v_res, v_unit=ppi, height_unit=inches
        self.resources.push(ResourceSpec {
            id: image_psd::model::resources::RES_RESOLUTION_INFO,
            name: String::new(),
            data: e.into_bytes(),
        });
        self
    }

    /// An unmodeled resource (any id, opaque payload) — the
    /// `unknown_resource` fixture's reason for being (brief §3).
    pub fn resource_opaque(mut self, id: u16, bytes: Vec<u8>) -> Self {
        self.resources.push(ResourceSpec {
            id,
            name: String::new(),
            data: bytes,
        });
        self
    }

    /// Append a fully-specified layer to the flat list (brief §5).
    pub fn layer(mut self, spec: LayerSpec) -> Self {
        self.layers.push(spec);
        self
    }

    /// Attach an opaque additional-layer-info block to the most recently
    /// added layer (brief §6) — the `unknown_addl` fixture's hook.
    pub fn layer_addl_opaque(mut self, sig: [u8; 4], key: [u8; 4], bytes: Vec<u8>) -> Self {
        let layer = self
            .layers
            .last_mut()
            .expect("layer_addl_opaque called before any layer");
        layer.extra_addl.push(AddlSpec::Opaque {
            sig,
            key,
            payload: bytes,
        });
        self
    }

    /// Open a group. Per brief §7 the bounding divider (kind 3, named
    /// `</Layer group>`) is pushed NOW — it sits below the members in the
    /// bottom-first list — and the folder record is deferred to
    /// [`group_close`](Self::group_close). `open` selects folder kind 1
    /// (open) vs 2 (closed).
    pub fn group_open(mut self, name: &str, open: bool) -> Self {
        // The bounding divider closes the group from below; it carries no
        // channels and a hidden marker name.
        self.layers.push(divider_layer());
        self.open_groups.push(PendingGroup {
            name: name.to_string(),
            kind: if open { 1 } else { 2 },
        });
        self
    }

    /// Close the most recently opened group: append its folder record
    /// (kind 1|2) ABOVE the members already pushed (brief §7).
    pub fn group_close(mut self) -> Self {
        let g = self
            .open_groups
            .pop()
            .expect("group_close without a matching group_open");
        self.layers.push(folder_layer(&g.name, g.kind));
        self
    }

    /// Set the merged composite (brief §11). A 4-channel composite flips
    /// the transparency-in-merged flag (the negative layer count).
    pub fn composite(mut self, comp: Compression, planes: Vec<Plane>) -> Self {
        if planes.len() >= 4 {
            self.transparency_in_merged = true;
        }
        self.composite = Some((comp, planes));
        self
    }

    /// A document-level additional-layer-info block (brief §4c).
    pub fn doc_addl(mut self, spec: AddlSpec) -> Self {
        self.doc_addl.push(spec);
        self
    }

    /// Assemble the full file (brief §1–§11).
    pub fn build(&self) -> Vec<u8> {
        assert!(
            self.open_groups.is_empty(),
            "build() with {} unclosed group(s)",
            self.open_groups.len()
        );
        let w = self.container.section_len_width();
        let mut e = Emit::new();

        // §1 File header (26 bytes).
        e.fourcc(SIGNATURE).u16(self.container.version());
        e.raw(&[0u8; 6]); // reserved
        e.u16(self.channels);
        e.u32(self.height).u32(self.width);
        e.u16(8); // depth — 8 in all M0 fixtures
        e.u16(3); // color mode — RGB

        // §2 Color mode data: empty for RGB.
        e.u32(0);

        // §3 Image resources section.
        self.emit_resources(&mut e);

        // §4 Layer & mask info section.
        self.emit_layer_mask(&mut e, w);

        // §11 Merged composite — last, unframed, runs to EOF.
        let (comp, planes) = self
            .composite
            .clone()
            .unwrap_or((Compression::Raw, self.default_composite_planes()));
        e.raw(&encode_composite(&planes, comp, w));

        e.into_bytes()
    }

    /// A flat composite of mid-gray planes when the caller set none —
    /// keeps `build()` total even for layer-only constructions.
    fn default_composite_planes(&self) -> Vec<Plane> {
        (0..self.channels)
            .map(|_| Plane::solid(self.width, self.height, 128))
            .collect()
    }

    /// §3: the resources section is u32-length-framed (PSD and PSB both —
    /// the section length is always u32 here, brief §3). Each block:
    /// `8BIM` + id + even-padded name + u32 data size + even-padded data.
    fn emit_resources(&self, e: &mut Emit) {
        let mut section = Emit::new();
        for r in &self.resources {
            section.fourcc(*b"8BIM").u16(r.id);
            section.pascal_string(r.name.as_bytes(), 2);
            section.u32(r.data.len() as u32);
            section.raw(&r.data);
            section.pad_to(2); // data padded to even; size excluded the pad
        }
        let bytes = section.into_bytes();
        e.u32(bytes.len() as u32);
        e.raw(&bytes);
    }

    /// §4: layer & mask info. Outer length (u32 PSD / u64 PSB) wraps:
    /// (a) layer info, (b) global layer mask info, (c) doc-level addl.
    fn emit_layer_mask(&self, e: &mut Emit, w: LenWidth) {
        let mut section = Emit::new();

        // (a) Layer info.
        self.emit_layer_info(&mut section, w);

        // (b) Global layer mask info: 0-length is common AND meaningful —
        // emit the 4-byte zero (brief §4b).
        section.u32(0);

        // (c) Document-level additional layer info, padded to a multiple
        // of 4 (brief §6 canonical padding 4c).
        for a in &self.doc_addl {
            emit_doc_addl(&mut section, self.container, a);
        }

        let bytes = section.into_bytes();
        e.len_field(w, bytes.len() as u64);
        e.raw(&bytes);
    }

    /// §4a: layer info — its own length field (content rounded to even),
    /// the signed layer count, all records, then all per-layer channels.
    fn emit_layer_info(&self, section: &mut Emit, w: LenWidth) {
        if self.layers.is_empty() {
            // No layers: a zero-length layer-info block (brief §4a).
            section.len_field(w, 0);
            return;
        }

        let mut info = Emit::new();
        let count = self.layers.len() as i16;
        // NEGATIVE count = transparency_in_merged flag (brief §4a).
        info.i16(if self.transparency_in_merged {
            -count
        } else {
            count
        });
        for layer in &self.layers {
            emit_layer_record(&mut info, self.container, layer);
        }
        for layer in &self.layers {
            emit_layer_channels(&mut info, self.container, layer);
        }
        // Content rounded up to even (brief §4a).
        info.pad_to(2);
        let bytes = info.into_bytes();
        section.len_field(w, bytes.len() as u64);
        section.raw(&bytes);
    }
}

/// A bounding-divider layer (brief §7): no channels, the hidden marker
/// name, an `lsct` block of kind 3.
fn divider_layer() -> LayerSpec {
    LayerSpec {
        top: 0,
        left: 0,
        bottom: 0,
        right: 0,
        name: GROUP_DIVIDER_NAME.to_string(),
        blend_key: *b"norm",
        opacity: 255,
        clipping: 0,
        flags: 0,
        mask: None,
        blend_ranges: Vec::new(),
        channels: Vec::new(),
        extra_addl: vec![AddlSpec::Lsct {
            kind: 3,
            blend_key: Some(*b"norm"),
            sub_kind: None,
        }],
    }
}

/// A folder record (brief §7): no channels, the group name, an `lsct`
/// block of the folder kind (1 open / 2 closed).
fn folder_layer(name: &str, kind: u32) -> LayerSpec {
    LayerSpec {
        top: 0,
        left: 0,
        bottom: 0,
        right: 0,
        name: name.to_string(),
        blend_key: *b"norm",
        opacity: 255,
        clipping: 0,
        flags: 0,
        mask: None,
        blend_ranges: Vec::new(),
        channels: Vec::new(),
        extra_addl: vec![AddlSpec::Lsct {
            kind,
            blend_key: Some(*b"norm"),
            sub_kind: None,
        }],
    }
}

/// Emit a document-level additional-layer-info block (brief §4c/§6).
/// Same payload as the layer-record level but padded to a multiple of 4
/// (the document-level canonical padding); the stored length excludes
/// the pad.
fn emit_doc_addl(e: &mut Emit, c: Container, spec: &AddlSpec) {
    let (sig, key, payload) = addl_parts(spec);
    let start = e.len();
    e.fourcc(sig).fourcc(key);
    e.len_field(addl_len_width(c, key), payload.len() as u64);
    e.raw(&payload);
    // Pad the whole block (from its signature) to a multiple of 4.
    let block_len = e.len() - start;
    let pad = (4 - (block_len % 4)) % 4;
    e.raw(&vec![0u8; pad]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psd_builder_header_signature_and_version() {
        let psd = PsdBuilder::new(Container::Psd, 4, 4, 3).build();
        assert_eq!(&psd[0..4], b"8BPS");
        assert_eq!(&psd[4..6], &[0, 1]); // version 1 = PSD

        let psb = PsdBuilder::new(Container::Psb, 4, 4, 3).build();
        assert_eq!(&psb[0..4], b"8BPS");
        assert_eq!(&psb[4..6], &[0, 2]); // version 2 = PSB
    }

    #[test]
    fn psd_builder_header_dims_and_mode() {
        let psd = PsdBuilder::new(Container::Psd, 7, 5, 3).build();
        // channels(2) height(4) width(4) depth(2) mode(2) at offset 12.
        assert_eq!(&psd[12..14], &[0, 3]); // channels
        assert_eq!(&psd[14..18], &[0, 0, 0, 5]); // height
        assert_eq!(&psd[18..22], &[0, 0, 0, 7]); // width
        assert_eq!(&psd[22..24], &[0, 8]); // depth
        assert_eq!(&psd[24..26], &[0, 3]); // color mode RGB
    }

    #[test]
    fn psd_builder_color_mode_section_empty_for_rgb() {
        let psd = PsdBuilder::new(Container::Psd, 4, 4, 3).build();
        // Color mode data length immediately after the 26-byte header.
        assert_eq!(&psd[26..30], &[0, 0, 0, 0]);
    }

    #[test]
    fn psd_builder_group_open_close_balances_and_orders() {
        let inner = LayerSpec {
            top: 0,
            left: 0,
            bottom: 1,
            right: 1,
            name: "inner".into(),
            blend_key: *b"norm",
            opacity: 255,
            clipping: 0,
            flags: 0,
            mask: None,
            blend_ranges: Vec::new(),
            channels: Vec::new(),
            extra_addl: Vec::new(),
        };
        let b = PsdBuilder::new(Container::Psd, 4, 4, 3)
            .group_open("G", false)
            .layer(inner)
            .group_close();
        // bottom-first: divider, member, folder.
        assert_eq!(b.layers.len(), 3);
        assert_eq!(b.layers[0].name, GROUP_DIVIDER_NAME);
        assert_eq!(b.layers[1].name, "inner");
        assert_eq!(b.layers[2].name, "G");
    }

    #[test]
    #[should_panic(expected = "unclosed group")]
    fn psd_builder_unclosed_group_panics_on_build() {
        let _ = PsdBuilder::new(Container::Psd, 4, 4, 3)
            .group_open("G", true)
            .build();
    }
}
