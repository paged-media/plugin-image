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

//! A layer stack as bytes, so a session can be stored in the document
//! and reopened: a compact MANIFEST (structure, properties, adjustment
//! chains) plus the raw BUFFERS it refers to by slot (each layer's
//! pixels at their own depth, its mask, a smart object's source). The
//! buffers are kept separate so a caller can store each once, by
//! content hash, and share it across revisions that did not change it.
//!
//! Manifest layout, little-endian: `"PGIL"`, u32 version (3), u32 width,
//! u32 height, u32 active, u32 next_id; u32 group count and per group
//! {u32 id, str name, u8 visible, f32 opacity, str blend, u8 pass_through,
//! i64 parent (-1 = none), u8 mask_enabled, i64 mask slot (-1 = none)}; u32 layer count and per layer {u32 id, str
//! name, u8 visible, u8 locked, f32 opacity, str blend, u8 clipped, i64
//! group, u8 mask_enabled, u8 kind (0 pixels, 1 adjustment, 2 smart), u8
//! depth (8 | 16), u32 pixel slot, then (version 3) i32 x, i32 y, u32 w,
//! u32 h — the rectangle the pixels cover, all four −1/0 sentinels
//! (x = −1) for the canvas (ADR 464), then i64 mask slot (-1 = none), then for an
//! adjustment u32 n + n f32 (`AdjustParams::to_wire`), for a smart object
//! u32 width, u32 height, f32 scale, u32 source slot}. A `str` is u32
//! length + UTF-8. Versions 1 (no group masks) and 2 (no layer
//! rectangles: every layer canvas-sized) are still read.
//!
//! The history is not persisted: a reopened stack starts with none.

use std::sync::Arc;

use image_core::SampleDepth;
use image_gpu::SelectionCoverage;
use image_graph::journal::TileJournal;

use super::{Layer, LayerGroup, LayerKind, LayerStack, SmartSource};
use crate::ingest::{AdjustParams, IngestError};

const MAGIC: &[u8; 4] = b"PGIL";
const VERSION: u32 = 3;

struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.0.extend_from_slice(s.as_bytes());
    }
}

struct R<'a> {
    b: &'a [u8],
    at: usize,
}

impl R<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], IngestError> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .ok_or_else(|| IngestError::Decode("layer manifest is truncated".into()))?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, IngestError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, IngestError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
    fn i32(&mut self) -> Result<i32, IngestError> {
        Ok(i32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
    fn i64(&mut self) -> Result<i64, IngestError> {
        Ok(i64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }
    fn f32(&mut self) -> Result<f32, IngestError> {
        Ok(f32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
    fn str(&mut self) -> Result<String, IngestError> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec())
            .map_err(|_| IngestError::Decode("layer manifest: a name is not UTF-8".into()))
    }
}

fn blend(name: &str) -> Result<&'static image_kernels::KernelDef, IngestError> {
    crate::stroke::blend_kernel(name)
        .ok_or_else(|| IngestError::Decode(format!("layer manifest: unknown blend \"{name}\"")))
}

impl LayerStack {
    /// The stack as a manifest plus the buffers it refers to by slot.
    pub fn export(&self) -> (Vec<u8>, Vec<Arc<[u8]>>) {
        let mut w = W(Vec::new());
        let mut buffers: Vec<Arc<[u8]>> = Vec::new();
        let mut slot = |b: Arc<[u8]>| -> u32 {
            buffers.push(b);
            (buffers.len() - 1) as u32
        };
        w.0.extend_from_slice(MAGIC);
        w.u32(VERSION);
        w.u32(self.width);
        w.u32(self.height);
        w.u32(self.active as u32);
        w.u32(self.next_id);
        w.u32(self.groups.len() as u32);
        for g in &self.groups {
            w.u32(g.id);
            w.str(&g.name);
            w.u8(g.visible as u8);
            w.f32(g.opacity);
            w.str(g.blend.id);
            w.u8(g.pass_through as u8);
            w.i64(g.parent.map_or(-1, i64::from));
            w.u8(g.mask_enabled as u8);
            let mask = g
                .mask
                .as_ref()
                .map(|m| slot(Arc::from(m.data().to_vec().into_boxed_slice())));
            w.i64(mask.map_or(-1, i64::from));
        }
        w.u32(self.layers.len() as u32);
        for l in &self.layers {
            w.u32(l.id);
            w.str(&l.name);
            w.u8(l.visible as u8);
            w.u8(l.locked as u8);
            w.f32(l.opacity);
            w.str(l.blend.id);
            w.u8(l.clipped as u8);
            w.i64(l.group.map_or(-1, i64::from));
            w.u8(l.mask_enabled as u8);
            w.u8(match l.kind {
                LayerKind::Pixels => 0,
                LayerKind::Adjustment(_) => 1,
                LayerKind::Smart(_) => 2,
            });
            w.u8(if l.rgba.is_16bit() { 16 } else { 8 });
            // A bounded layer is stored as it lives — expanding it here
            // would cost every untouched layer its canvas on each save.
            match l.rgba.bounded() {
                Some(b) => {
                    w.u32(slot(b.px.raw_arc()));
                    w.i32(b.rect.x);
                    w.i32(b.rect.y);
                    w.u32(b.rect.w);
                    w.u32(b.rect.h);
                }
                None => {
                    w.u32(slot(l.rgba.raw_arc()));
                    w.i32(-1);
                    w.i32(-1);
                    w.u32(0);
                    w.u32(0);
                }
            }
            let mask = l
                .mask
                .as_ref()
                .map(|m| slot(Arc::from(m.data().to_vec().into_boxed_slice())));
            w.i64(mask.map_or(-1, i64::from));
            match &l.kind {
                LayerKind::Pixels => {}
                LayerKind::Adjustment(p) => {
                    let block = p.to_wire();
                    w.u32(block.len() as u32);
                    for v in block {
                        w.f32(v);
                    }
                }
                LayerKind::Smart(src) => {
                    w.u32(src.width);
                    w.u32(src.height);
                    w.f32(src.scale);
                    let s = slot(Arc::clone(&src.rgba));
                    w.u32(s);
                }
            }
        }
        (w.0, buffers)
    }

    /// Rebuild a stack from [`export`](Self::export)'s output. Every size
    /// is checked against the manifest; the history starts empty.
    pub fn import(manifest: &[u8], buffers: &[Arc<[u8]>]) -> Result<LayerStack, IngestError> {
        let mut r = R { b: manifest, at: 0 };
        if r.take(4)? != MAGIC {
            return Err(IngestError::Decode("not a layer manifest".into()));
        }
        let version = r.u32()?;
        if !(1..=VERSION).contains(&version) {
            return Err(IngestError::Unsupported(format!(
                "layer manifest version {version} (this engine reads {VERSION})"
            )));
        }
        let (width, height) = (r.u32()?, r.u32()?);
        let active = r.u32()? as usize;
        let next_id = r.u32()?;
        let n_px = width as usize * height as usize;
        let buffer = |i: u32| -> Result<Arc<[u8]>, IngestError> {
            buffers
                .get(i as usize)
                .cloned()
                .ok_or_else(|| IngestError::Decode(format!("layer manifest: no buffer {i}")))
        };
        let mut groups = Vec::new();
        for _ in 0..r.u32()? {
            let (id, name, visible, opacity) = (r.u32()?, r.str()?, r.u8()? != 0, r.f32()?);
            let (blend, pass_through, parent) = (
                blend(&r.str()?)?,
                r.u8()? != 0,
                u32::try_from(r.i64()?).ok(),
            );
            let (mask_enabled, mask) = if version >= 2 {
                let enabled = r.u8()? != 0;
                let mask = match u32::try_from(r.i64()?).ok() {
                    Some(slot) => {
                        let data = buffer(slot)?;
                        Some(Arc::new(
                            SelectionCoverage::from_data(width, height, data.to_vec()).ok_or_else(
                                || {
                                    IngestError::Decode(format!(
                                        "layer manifest: group mask {slot} is {} bytes for \
                                         {width}x{height}",
                                        data.len()
                                    ))
                                },
                            )?,
                        ))
                    }
                    None => None,
                };
                (enabled, mask)
            } else {
                (true, None)
            };
            groups.push(LayerGroup {
                id,
                name,
                visible,
                opacity,
                blend,
                pass_through,
                parent,
                mask,
                mask_enabled,
            });
        }
        let mut layers = Vec::new();
        for _ in 0..r.u32()? {
            let id = r.u32()?;
            let name = r.str()?;
            let visible = r.u8()? != 0;
            let locked = r.u8()? != 0;
            let opacity = r.f32()?;
            let blend = blend(&r.str()?)?;
            let clipped = r.u8()? != 0;
            let group = u32::try_from(r.i64()?).ok();
            let mask_enabled = r.u8()? != 0;
            let kind_code = r.u8()?;
            let depth = match r.u8()? {
                8 => SampleDepth::U8,
                16 => SampleDepth::U16,
                d => return Err(IngestError::Decode(format!("layer manifest: depth {d}"))),
            };
            let px = buffer(r.u32()?)?;
            let bpp = if depth == SampleDepth::U16 { 8 } else { 4 };
            let rect = if version >= 3 {
                let (x, y, rw, rh) = (r.i32()?, r.i32()?, r.u32()?, r.u32()?);
                (x >= 0).then_some(image_core::Region::new(x, y, rw, rh))
            } else {
                None
            };
            let rgba: crate::pixels::LayerPixels = match rect {
                Some(rect) => crate::pixels::Bounded::new(
                    rect,
                    crate::pixels::Pixels::from_raw(px, depth),
                    width,
                    height,
                )
                .map(crate::pixels::LayerPixels::Bounded)
                .ok_or_else(|| {
                    IngestError::Decode(format!(
                        "layer \"{name}\": its rectangle {rect:?} does not fit the canvas \
                         or its pixels"
                    ))
                })?,
                None => {
                    if px.len() != n_px * bpp {
                        return Err(IngestError::Decode(format!(
                            "layer \"{name}\": {} pixel bytes for {width}×{height}",
                            px.len()
                        )));
                    }
                    crate::pixels::Pixels::from_raw(px, depth).into()
                }
            };
            let mask = match u32::try_from(r.i64()?) {
                Ok(slot) => {
                    let m = buffer(slot)?;
                    Some(Arc::new(
                        SelectionCoverage::from_data(width, height, m.to_vec()).ok_or_else(
                            || IngestError::Decode(format!("layer \"{name}\": mask is mis-sized")),
                        )?,
                    ))
                }
                Err(_) => None,
            };
            let kind = match kind_code {
                0 => LayerKind::Pixels,
                1 => {
                    let n = r.u32()? as usize;
                    let mut block = Vec::with_capacity(n);
                    for _ in 0..n {
                        block.push(r.f32()?);
                    }
                    LayerKind::Adjustment(Box::new(AdjustParams::from_wire(&block)?))
                }
                2 => {
                    let (sw, sh, scale) = (r.u32()?, r.u32()?, r.f32()?);
                    let src = buffer(r.u32()?)?;
                    if src.len() != sw as usize * sh as usize * 4 {
                        return Err(IngestError::Decode(format!(
                            "layer \"{name}\": smart source is mis-sized"
                        )));
                    }
                    LayerKind::Smart(Box::new(SmartSource {
                        rgba: src,
                        width: sw,
                        height: sh,
                        scale,
                    }))
                }
                k => {
                    return Err(IngestError::Decode(format!(
                        "layer manifest: layer kind {k}"
                    )))
                }
            };
            layers.push(Layer {
                kind,
                id,
                name,
                visible,
                locked,
                opacity,
                blend,
                rgba,
                mask,
                group,
                mask_enabled,
                clipped,
            });
        }
        if layers.is_empty() || active >= layers.len() {
            return Err(IngestError::Decode(
                "layer manifest: no layers, or active out of range".into(),
            ));
        }
        Ok(LayerStack {
            width,
            height,
            layers,
            groups,
            active,
            next_id,
            journal: TileJournal::new(),
            steps: Vec::new(),
            undone: Vec::new(),
            dropped_steps: 0,
            structure_generation: 0,
            edit_mask: false,
            fold: Default::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image_core::Region;

    fn px(w: u32, h: u32, v: u8) -> Arc<[u8]> {
        Arc::from(vec![v; (w * h * 4) as usize].into_boxed_slice())
    }

    /// Every feature a stack can carry: pixel, adjustment and smart
    /// layers, a 16-bit layer, a mask, clipping, a group, odd names.
    fn rich() -> LayerStack {
        let mut s = LayerStack::from_image(4, 3, px(4, 3, 10)).expect("stack");
        s.add("Paint “ünïcode”");
        s.edit_active("p", Region::new(0, 0, 4, 3), px(4, 3, 200).into())
            .expect("paint");
        s.set_opacity(1, 0.4).expect("opacity");
        s.set_blend(1, "multiply").expect("blend");
        let cov =
            SelectionCoverage::from_data(4, 3, (0..12).map(|i| i * 20).collect()).expect("cov");
        s.set_mask(1, Arc::new(cov)).expect("mask");
        s.set_mask_enabled(1, false).expect("mask off");
        s.add_adjustment(
            "Grade",
            AdjustParams {
                exposure_ev: 0.3,
                vibrance: 0.2,
                posterize: Some(5.0),
                curve_lut: Some(std::array::from_fn(|i| (255 - i) as u8)),
                ..AdjustParams::default()
            },
        );
        s.set_clipped(2, true).expect("clip");
        s.make_smart(0).expect("smart");
        s.group_range(1, 2, "G").expect("group");
        s.add("Deep");
        s.layers.last_mut().expect("layer").rgba =
            crate::pixels::Pixels::from_rgba16(&(0..48).map(|i| i * 1000).collect::<Vec<u16>>())
                .into();
        s
    }

    #[test]
    #[allow(non_snake_case)]
    fn a_stack_round_trips_byte_for_byte__feat__image_editor_layers() {
        let s = rich();
        let (m, bufs) = s.export();
        let back = LayerStack::import(&m, &bufs).expect("import");
        let (m2, bufs2) = back.export();
        assert_eq!(m, m2, "the manifest is identical after a round trip");
        assert_eq!(bufs.len(), bufs2.len());
        for (a, b) in bufs.iter().zip(&bufs2) {
            assert_eq!(a[..], b[..]);
        }
        assert_eq!(back.len(), s.len());
        assert!(
            !back.history().can_undo,
            "a reopened stack starts with no history"
        );
        // Adjustment parameters survive exactly (to_wire ∘ from_wire).
        match (&s.layers[2].kind, &back.layers[2].kind) {
            (LayerKind::Adjustment(a), LayerKind::Adjustment(b)) => {
                assert_eq!(a.to_wire(), b.to_wire());
                assert_eq!(b.posterize, Some(5.0));
            }
            _ => panic!("layer 2 is the adjustment"),
        }
        assert!(back.layers[3].rgba.is_16bit(), "16-bit stays 16-bit");
    }

    /// A GROUP's mask (version 2) comes back with its switch; a version-1
    /// manifest — no group mask fields — still opens, its groups unmasked.
    #[test]
    #[allow(non_snake_case)]
    fn a_group_mask_round_trips_and_version_1_still_reads__feat__image_editor_layers() {
        let mut s = rich();
        let cov = SelectionCoverage::from_data(4, 3, (0..12).map(|i| 255 - i * 10).collect())
            .expect("cov");
        s.groups[0].mask = Some(Arc::new(cov.clone()));
        s.groups[0].mask_enabled = false;
        let (m, bufs) = s.export();
        let back = LayerStack::import(&m, &bufs).expect("import");
        let g = &back.groups[0];
        assert_eq!(
            g.mask.as_deref().map(|c| c.data().to_vec()),
            Some(cov.data().to_vec())
        );
        assert!(!g.mask_enabled, "the switch survives");

        // Version 1: the same manifest without the group's two mask fields
        // and without the layers' rectangles (all canvas-sized here).
        let plain = rich();
        let (m2, bufs2) = plain.export();
        let mut v1 = m2.clone();
        v1[4..8].copy_from_slice(&1u32.to_le_bytes());
        let canvas_rect = [
            0xFFu8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        while let Some(at) = v1.windows(16).position(|w| w == canvas_rect) {
            v1.drain(at..at + 16);
        }
        // The group record ends with u8 mask_enabled + i64 mask slot (-1):
        // nine bytes, the first group's last ones before the layer count.
        let tail = [1u8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let at = v1
            .windows(9)
            .position(|w| w == tail)
            .expect("the group's mask fields");
        v1.drain(at..at + 9);
        let old = LayerStack::import(&v1, &bufs2).expect("a version-1 manifest opens");
        assert_eq!(old.groups.len(), 1);
        assert!(old.groups[0].mask.is_none() && old.groups[0].mask_enabled);
        assert_eq!(old.len(), plain.len());
    }

    /// A layer at its BOUNDS (ADR 464) is stored as it lives — its own
    /// rectangle and pixels, not an expansion — and comes back bounded.
    #[test]
    #[allow(non_snake_case)]
    fn a_bounded_layer_round_trips_at_its_bounds__feat__image_editor_layers() {
        let mut s = LayerStack::from_image(4, 3, px(4, 3, 10)).expect("stack");
        s.add("Dot");
        let mut dot = vec![0u8; 4 * 3 * 4];
        dot[(4 + 1) * 4..(4 + 3) * 4].copy_from_slice(&[7, 8, 9, 255, 7, 8, 9, 255]);
        s.edit_active(
            "p",
            Region::new(0, 0, 4, 3),
            crate::pixels::Pixels::from_rgba8(Arc::from(dot.clone())),
        )
        .expect("paint");
        assert_eq!(
            s.shrink_to_content(1).expect("shrink"),
            Some(Region::new(1, 1, 2, 1))
        );
        let (m, bufs) = s.export();
        assert_eq!(bufs[1].len(), 2 * 4, "the bounded pixels, not the canvas");
        let back = LayerStack::import(&m, &bufs).expect("import");
        let b = back.layers[1].rgba.bounded().expect("bounded");
        assert_eq!(b.rect, Region::new(1, 1, 2, 1));
        assert_eq!(&back.layers[1].rgba.to_rgba8()[..], &dot[..]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn a_damaged_manifest_is_refused_not_misread__feat__image_editor_layers() {
        let (m, bufs) = rich().export();
        assert!(
            LayerStack::import(&m[..m.len() - 3], &bufs).is_err(),
            "truncated"
        );
        assert!(
            LayerStack::import(b"NOPE", &bufs).is_err(),
            "not a manifest"
        );
        assert!(
            LayerStack::import(&m, &bufs[..1]).is_err(),
            "a buffer missing"
        );
        let mut wrong = bufs.clone();
        wrong[0] = Arc::from(vec![0u8; 3].into_boxed_slice());
        assert!(
            LayerStack::import(&m, &wrong).is_err(),
            "a buffer mis-sized"
        );
    }
}
