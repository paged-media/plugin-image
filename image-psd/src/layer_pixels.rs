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

//! LAYER PIXEL IMPORT — a parsed PSD's layer tree as canvas-extent
//! straight-RGBA8 plates, so the editor can open a PSD INTO a layer
//! stack instead of only flattening it.
//!
//! The parse has held all of this since M0 (bounds, per-channel
//! compressed payloads, blend key, opacity, visibility flag); what was
//! missing was a production path from those records to pixels — the
//! only decode that existed was the TEST-ONLY flatten oracle in
//! `image-conformance`. This is that path, and it is a pure READ: the
//! preservation model is untouched, so a file imported as layers still
//! re-emits byte-identically if nothing is edited (§10.4).
//!
//! # It declines rather than approximates
//!
//! Opening a PSD as layers replaces Photoshop's OWN composite (which the
//! merged-composite lane shows today) with ours. That is only an
//! improvement where ours is faithful, so [`PsdFile::layer_plates_rgba8`]
//! refuses every file whose structure it does not model, and the caller
//! keeps the flatten:
//!
//! * **not RGB at 8 or 16 bits** — 1/32-bit and CMYK/Lab are separate
//!   lanes;
//! * **a group with a mask of its own**, a **vector mask**, a mask with
//!   **density/feather parameters**, or a layer with both a user and a
//!   "real" mask (channel −3) — none of these is modelled;
//! * **layer effects** (`lfx2` / `lrFX`), **adjustment layers** (their
//!   pixels are not stored; the adjustment is) and **artboards** —
//!   importing them as plain pixel layers would draw something else;
//! * **fill opacity** below 100 % on one of Photoshop's eight special
//!   blend modes (where fill and layer opacity differ), or on a group;
//! * **a budget overrun** — plates are CANVAS-EXTENT (the layer model's
//!   deliberate simplification), so N layers of a big canvas is N × 4
//!   bytes per pixel. Past [`MAX_IMPORT_BYTES`] the import declines
//!   instead of exhausting the wasm heap.
//!
//! What it does import: pixel layers with opacity, blend and visibility
//! (FILL opacity, `iOpa`, folded into the opacity: without effects, and
//! outside the eight special modes, the two multiply);
//! CLIPPING (the record's clipping byte); GROUPS — the bounding divider
//! below a group's members and the folder record above them, with the
//! folder's name, blend (`pass` = pass-through), opacity and visibility,
//! nested; LAYER MASKS (channel −2: the mask rectangle, the default
//! colour outside it, the disabled and invert flags); and SMART OBJECTS
//! as their stored render ([`LayerPlate::smart`]).
//!
//! # Smart objects: the stored render, vouched for by the composite
//!
//! A smart object's layer pixels are Photoshop's render of its embedded
//! source, and a render can be stale: on corpus mock-ups the cache held
//! a replaced design while Photoshop, opening the file, showed the new
//! one. This module cannot tell a current cache from a stale one — it
//! does not render sources — so it imports the stored render and MARKS
//! it, and the consumer must check each marked plate against the file's
//! own merged composite before accepting the import (`image-js`'s
//! `smart_renders_agree`). A cache the composite agrees with is what
//! Photoshop drew when it saved the file; one it disagrees with declines
//! the whole import. Without real merged data (resource 0x0421) nothing
//! can vouch for a cache, so such a file declines here.
//!
//! Every refusal is a typed [`PsdError::Unsupported`] carrying the
//! reason, which the panel shows verbatim. "It flattened and did not say
//! why" is the failure mode this exists to avoid. The refusal names the
//! FIRST blocker; [`PsdFile::layer_import_blockers`] lists all of them,
//! which is what tells you how far a file is from importing.
//!
//! Provenance: Adobe Photoshop File Format specification — Layer
//! Records (bounds, blend-mode key, opacity, flags), Channel Image Data
//! (per-channel decode, ids 0/1/2 = R/G/B, −1 = transparency, −2 = user
//! mask), Layer Mask / Adjustment Layer Data (rectangle, default colour,
//! flags), Additional Layer Information (`lsct` section dividers, `luni`
//! names, `vmsk`/`vsms` vector masks).

use crate::composite::MergedData;
use crate::model::ColorMode;
use crate::model::{LayerRecord, PsdFile, SectionKind};
use crate::{PsdError, Result};

/// Ceiling on the total plate memory an import may allocate. Canvas
/// extent × 4 bytes × layer count; 384 MiB is ~8 layers of a
/// 4000×3000 canvas, or 30+ layers of a typical web-sized one.
pub const MAX_IMPORT_BYTES: usize = 384 * 1024 * 1024;

/// One pixel-bearing PSD layer, ready for the layer stack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPlate {
    /// The canonical name (`luni` when present, else the legacy Pascal).
    pub name: String,
    /// The blend-mode fourcc exactly as stored (`norm`, `mul `, …). The
    /// mapping to a `compose.*` kernel belongs to the consumer, which is
    /// where the kernel registry lives.
    pub blend_key: [u8; 4],
    /// 0–255 (the record's own scale).
    pub opacity: u8,
    /// Layer-record flags bit 1 (0x02).
    pub hidden: bool,
    /// Canvas-extent, tightly packed straight RGBA8. Pixels outside the
    /// layer's own rect are transparent black.
    pub rgba: Vec<u8>,
    /// The record's `clipping` byte: this layer is CLIPPED to the one
    /// below it. Carried across because the consumer models clipping —
    /// it did not when this importer was written, and the refusal that
    /// used to live here said so.
    pub clipped: bool,
    /// The innermost enclosing group, as an index into
    /// [`LayerImport::groups`]; `None` at the top level.
    pub group: Option<usize>,
    /// The layer's user mask, canvas-extent.
    pub mask: Option<MaskPlate>,
    /// The plate is a SMART OBJECT's stored render (`SoLd`/`PlLd`/`SoLE`),
    /// not pixels of its own. The import is only sound once the consumer
    /// has checked these plates against the merged composite (module
    /// docs).
    pub smart: bool,
}

/// One reason a file cannot be imported as layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportBlocker {
    /// A stable category (`colour-mode`, `vector-mask`, `effects`, …) —
    /// the ledger key.
    pub category: &'static str,
    /// The sentence the panel shows.
    pub message: String,
}

/// A layer mask as the stack takes it: one coverage byte per canvas
/// pixel (255 = shown), the record's default colour outside the mask
/// rectangle, inverted when the record says so.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskPlate {
    pub coverage: Vec<u8>,
    /// Mask flags bit 1 clear: a disabled mask is kept, not applied.
    pub enabled: bool,
}

/// A group (a Photoshop layer folder) as its folder record stores it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupPlate {
    pub name: String,
    /// The folder's blend key; `pass` is pass-through.
    pub blend_key: [u8; 4],
    pub opacity: u8,
    pub hidden: bool,
    /// The enclosing group (index into [`LayerImport::groups`]).
    pub parent: Option<usize>,
}

/// The whole importable layer tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerImport {
    pub width: u32,
    pub height: u32,
    /// The source was 16 bits per channel and every plate was REDUCED to
    /// 8. Reported for the same reason the composite reports it: a lossy
    /// step the user can see stated is a different thing from one they
    /// cannot.
    pub depth_reduced: bool,
    /// BOTTOM-first, the order PSD stores them in and the order the
    /// layer stack composites in.
    pub layers: Vec<LayerPlate>,
    /// Every group, outermost before its children; members name theirs by
    /// index.
    pub groups: Vec<GroupPlate>,
}

/// Additional-layer-info keys whose content the import does not model,
/// each with what it is. A record carrying one is refused.
const UNMODELLED: &[(&[u8; 4], &str)] = &[
    (b"lfx2", "layer effects"),
    (b"lrFX", "layer effects"),
    (b"artb", "an artboard"),
    (b"artd", "an artboard"),
    (b"abdd", "an artboard"),
    (b"levl", "a Levels adjustment layer"),
    (b"curv", "a Curves adjustment layer"),
    (b"hue2", "a Hue/Saturation adjustment layer"),
    (b"hue ", "a Hue/Saturation adjustment layer"),
    (b"brit", "a Brightness/Contrast adjustment layer"),
    (b"blnc", "a Color Balance adjustment layer"),
    (b"mixr", "a Channel Mixer adjustment layer"),
    (b"phfl", "a Photo Filter adjustment layer"),
    (b"post", "a Posterize adjustment layer"),
    (b"thrs", "a Threshold adjustment layer"),
    (b"nvrt", "an Invert adjustment layer"),
    (b"selc", "a Selective Color adjustment layer"),
    (b"vibA", "a Vibrance adjustment layer"),
    (b"blwh", "a Black & White adjustment layer"),
    (b"grdm", "a Gradient Map adjustment layer"),
    (b"expA", "an Exposure adjustment layer"),
    (b"clrL", "a Color Lookup adjustment layer"),
];

/// Photoshop's eight special blend modes, where fill opacity is applied
/// inside the blend rather than as a fade (so it is not layer opacity).
/// The keys that make a layer a smart object.
const SMART_KEYS: &[&[u8; 4]] = &[b"SoLd", b"PlLd", b"SoLE"];

/// The ledger category of an [`UNMODELLED`] entry.
fn unmodelled_category(what: &str) -> &'static str {
    if what.contains("effects") {
        "effects"
    } else if what.contains("artboard") {
        "artboards"
    } else {
        "adjustment-layers"
    }
}

/// The structural walk: the group tree, the pixel layers, and every
/// blocker found on the way.
struct Walk<'a> {
    pixel_layers: Vec<(&'a LayerRecord, Option<usize>)>,
    groups: Vec<GroupPlate>,
    masked: usize,
    blockers: Vec<ImportBlocker>,
}

const SPECIAL_FILL_MODES: &[&[u8; 4]] = &[
    b"idiv", b"lbrn", b"div ", b"lddg", b"vLit", b"lLit", b"hMix", b"diff",
];

/// A block's payload: after the signature, key and 4-byte length.
fn addl_payload(a: &crate::model::AdditionalLayerInfo) -> &[u8] {
    a.raw_block
        .as_deref()
        .and_then(|b| b.get(12..))
        .unwrap_or(&[])
}

/// Is the record a smart object (its pixels a stored render)?
fn is_smart(layer: &LayerRecord) -> bool {
    layer
        .addl
        .iter()
        .any(|a| SMART_KEYS.contains(&&a.key))
}

/// The record's fill opacity (`iOpa`), 255 when absent.
fn fill_opacity(layer: &LayerRecord) -> u8 {
    layer
        .addl
        .iter()
        .find(|a| &a.key == b"iOpa")
        .and_then(|a| addl_payload(a).first().copied())
        .unwrap_or(255)
}

/// The user-mask channel ids: −2 the user mask, −3 the "real" user mask
/// that appears beside a vector mask.
const USER_MASK: i16 = -2;
const REAL_USER_MASK: i16 = -3;

impl PsdFile {
    /// Decode every pixel-bearing layer into a canvas-extent straight
    /// RGBA8 plate, bottom-first. See the module docs for the exact set
    /// of files this declines (and why declining is the right answer).
    pub fn layer_plates_rgba8(&self) -> Result<LayerImport> {
        let h = &self.header;
        let Walk {
            pixel_layers,
            groups,
            masked,
            blockers,
        } = self.import_walk()?;
        if let Some(first) = blockers.into_iter().next() {
            return Err(PsdError::Unsupported(first.message));
        }
        let (cw, ch) = (h.width, h.height);
        let canvas_texels = (cw as usize)
            .checked_mul(ch as usize)
            .ok_or_else(|| PsdError::Unsupported("canvas extent overflows usize".into()))?;
        let plate_bytes = canvas_texels
            .checked_mul(4)
            .ok_or_else(|| PsdError::Unsupported("canvas extent overflows usize".into()))?;
        let total = plate_bytes
            .saturating_mul(pixel_layers.len())
            .saturating_add(canvas_texels.saturating_mul(masked));
        if total > MAX_IMPORT_BYTES {
            return Err(PsdError::Unsupported(format!(
                "layer import needs {} MiB ({} layers × {}×{} canvas-extent plates), \
                 over the {} MiB budget — the merged composite is kept instead",
                total / (1024 * 1024),
                pixel_layers.len(),
                cw,
                ch,
                MAX_IMPORT_BYTES / (1024 * 1024)
            )));
        }

        let mut layers = Vec::with_capacity(pixel_layers.len());
        for (layer, group) in pixel_layers {
            // Fill opacity folds into opacity (see the refusals above).
            let opacity =
                ((u32::from(layer.opacity) * u32::from(fill_opacity(layer)) + 127) / 255) as u8;
            layers.push(LayerPlate {
                name: layer.name(),
                blend_key: layer.blend_key,
                opacity,
                hidden: (layer.flags & 0x02) != 0,
                clipped: layer.clipping != 0,
                rgba: self.layer_canvas_rgba8(layer, cw, ch)?,
                group,
                mask: self.layer_mask_plate(layer, cw, ch)?,
                smart: is_smart(layer),
            });
        }
        Ok(LayerImport {
            width: cw,
            height: ch,
            depth_reduced: h.depth == 16,
            layers,
            groups,
        })
    }

    /// EVERY reason this file cannot be imported as layers, in file
    /// order (file-level ones first). Empty means the structure is
    /// importable; the budget is not checked here. A malformed layer
    /// tree is one blocker, `malformed`.
    pub fn layer_import_blockers(&self) -> Vec<ImportBlocker> {
        match self.import_walk() {
            Ok(w) => w.blockers,
            Err(e) => vec![ImportBlocker {
                category: "malformed",
                message: e.to_string(),
            }],
        }
    }

    /// Structural gate: refuse before decoding a single byte. Records
    /// run bottom-first, so a group's BOUNDING DIVIDER comes before its
    /// members and its FOLDER record after them.
    fn import_walk(&self) -> Result<Walk<'_>> {
        let h = &self.header;
        let mut blockers = Vec::new();
        let mut block = |category: &'static str, message: String| {
            blockers.push(ImportBlocker { category, message });
        };
        // 16-bit is ACCEPTED and reduced, as the merged composite already
        // is. The per-channel decode refuses 16-bit RLE on its own, with
        // its own evidence, so a 16-bit RLE file still declines — but a
        // 16-bit RAW one imports its LAYERS instead of the whole file
        // falling back to a flattened composite.
        if h.depth != 8 && h.depth != 16 {
            block(
                "depth",
                format!(
                    "layer import at depth {} (8- and 16-bit only; 1-bit and 32-bit \
                     float are separate lanes)",
                    h.depth
                ),
            );
        }
        if h.color_mode != ColorMode::Rgb {
            block(
                "colour-mode",
                format!(
                    "layer import for color mode {:?} (RGB only; CMYK/Lab are the M2 CMS lane)",
                    h.color_mode
                ),
            );
        }
        let mut pixel_layers = Vec::new();
        let mut groups: Vec<GroupPlate> = Vec::new();
        let mut open: Vec<usize> = Vec::new();
        let mut masked = 0usize;
        let mut smart = false;
        for layer in &self.layer_mask.layers {
            let kind = layer.addl.iter().find_map(|a| a.lsct()).map(|d| d.kind);
            if layer
                .addl
                .iter()
                .any(|a| &a.key == b"vmsk" || &a.key == b"vsms")
            {
                block(
                    "vector-mask",
                    format!(
                        "layer import of a PSD with a VECTOR MASK (\"{}\"): vector masks \
                         are not modelled, so the merged composite is kept instead",
                        layer.name()
                    ),
                );
            }
            for (_, what) in UNMODELLED
                .iter()
                .filter(|(key, _)| layer.addl.iter().any(|a| &a.key == *key))
            {
                block(
                    unmodelled_category(what),
                    format!(
                        "layer import of a PSD with {what} (\"{}\"): not modelled, so the \
                         merged composite is kept instead",
                        layer.name()
                    ),
                );
            }
            let fill = fill_opacity(layer);
            if fill < 255 && SPECIAL_FILL_MODES.contains(&&layer.blend_key) {
                block(
                    "fill-opacity",
                    format!(
                        "layer import of FILL OPACITY on a special blend mode (\"{}\"): there \
                         fill is not layer opacity, so the merged composite is kept instead",
                        layer.name()
                    ),
                );
            }
            let has_mask = layer.channels.iter().any(|c| c.id == USER_MASK);
            match kind {
                Some(SectionKind::BoundingDivider) => {
                    groups.push(GroupPlate {
                        name: String::new(),
                        blend_key: *b"pass",
                        opacity: 255,
                        hidden: false,
                        parent: open.last().copied(),
                    });
                    open.push(groups.len() - 1);
                    continue;
                }
                Some(SectionKind::OpenFolder) | Some(SectionKind::ClosedFolder) => {
                    let Some(g) = open.pop() else {
                        return Err(PsdError::Malformed {
                            section: "layer records",
                            detail: format!(
                                "group \"{}\" has no bounding divider below it",
                                layer.name()
                            ),
                        });
                    };
                    if fill < 255 {
                        block(
                            "fill-opacity",
                            format!(
                                "layer import of FILL OPACITY on a group (\"{}\"): not \
                                 modelled, so the merged composite is kept instead",
                                layer.name()
                            ),
                        );
                    }
                    if has_mask {
                        block(
                            "group-mask",
                            format!(
                                "layer import of a GROUP with a mask (\"{}\"): group masks \
                                 are not modelled, so the merged composite is kept instead",
                                layer.name()
                            ),
                        );
                    }
                    let lsct_blend = layer
                        .addl
                        .iter()
                        .find_map(|a| a.lsct())
                        .and_then(|d| d.blend_key);
                    groups[g] = GroupPlate {
                        name: layer.name(),
                        blend_key: lsct_blend.unwrap_or(layer.blend_key),
                        opacity: layer.opacity,
                        hidden: (layer.flags & 0x02) != 0,
                        parent: groups[g].parent,
                    };
                    continue;
                }
                _ => {}
            }
            if layer.channels.iter().any(|c| c.id == REAL_USER_MASK) {
                block(
                    "vector-mask",
                    format!(
                        "layer import of a layer with both a user and a vector-derived \
                         mask (\"{}\"): not modelled, so the merged composite is kept \
                         instead",
                        layer.name()
                    ),
                );
            }
            if has_mask {
                if layer.mask.as_ref().is_some_and(|m| m.flags & 0x10 != 0) {
                    block(
                        "mask-parameters",
                        format!(
                            "layer import of a mask with density/feather parameters \
                             (\"{}\"): not modelled, so the merged composite is kept instead",
                            layer.name()
                        ),
                    );
                }
                masked += 1;
            }
            smart |= is_smart(layer);
            pixel_layers.push((layer, open.last().copied()));
        }
        if !open.is_empty() {
            return Err(PsdError::Malformed {
                section: "layer records",
                detail: format!("{} group(s) never closed by a folder record", open.len()),
            });
        }
        if pixel_layers.is_empty() {
            block(
                "no-layers",
                "layer import of a PSD with no layer records (the merged composite is \
                 all there is)"
                    .into(),
            );
        }
        if smart && self.merged_data() != MergedData::Real {
            block(
                "smart-objects",
                format!(
                    "layer import of a PSD with smart objects but without real merged data \
                     ({}): a smart object's stored render can be stale, and only the \
                     file's own composite can vouch for it",
                    self.merged_data().as_str()
                ),
            );
        }
        // lfx2 and lrFX usually describe the SAME effects: one blocker
        // per layer and category is the honest count.
        blockers.dedup();
        Ok(Walk {
            pixel_layers,
            groups,
            masked,
            blockers,
        })
    }

    /// The layer's user mask (channel −2) at canvas extent: the record's
    /// default colour everywhere, the decoded mask inside its rectangle,
    /// inverted when flags bit 2 says so. `None` without a mask channel.
    fn layer_mask_plate(&self, layer: &LayerRecord, cw: u32, ch: u32) -> Result<Option<MaskPlate>> {
        let Some(ci) = layer.channels.iter().position(|c| c.id == USER_MASK) else {
            return Ok(None);
        };
        let Some(m) = layer.mask.as_ref() else {
            return Err(PsdError::Malformed {
                section: "layer mask data",
                detail: format!(
                    "layer \"{}\" has a mask channel but no mask data",
                    layer.name()
                ),
            });
        };
        let mut coverage = vec![m.default_color; (cw as usize) * (ch as usize)];
        let mw = (m.right - m.left).max(0) as u32;
        let mh = (m.bottom - m.top).max(0) as u32;
        if mw > 0 && mh > 0 {
            let data = layer
                .channel_data
                .get(ci)
                .ok_or_else(|| PsdError::Malformed {
                    section: "layer channel image data",
                    detail: format!("layer \"{}\" mask channel has no payload", layer.name()),
                })?;
            let plane = data.decode(self.container, mh, mw, self.header.depth)?;
            if plane.len() != (mw as usize) * (mh as usize) {
                return Err(PsdError::Malformed {
                    section: "layer channel image data",
                    detail: format!(
                        "layer \"{}\" mask decoded to {} bytes, expected {}",
                        layer.name(),
                        plane.len(),
                        (mw as usize) * (mh as usize)
                    ),
                });
            }
            for my in 0..mh as i64 {
                let dy = m.top as i64 + my;
                if dy < 0 || dy >= ch as i64 {
                    continue;
                }
                for mx in 0..mw as i64 {
                    let dx = m.left as i64 + mx;
                    if dx < 0 || dx >= cw as i64 {
                        continue;
                    }
                    coverage[(dy * cw as i64 + dx) as usize] =
                        plane[(my * mw as i64 + mx) as usize];
                }
            }
        }
        if m.flags & 0x04 != 0 {
            for v in &mut coverage {
                *v = 255 - *v;
            }
        }
        Ok(Some(MaskPlate {
            coverage,
            enabled: m.flags & 0x02 == 0,
        }))
    }

    /// One layer's canvas-extent straight RGBA8: decode its modeled
    /// channels (ids 0/1/2 = R/G/B, −1 = transparency) into planar
    /// buffers and place them at the layer rect, clipped to the canvas.
    /// A layer with no transparency channel is OPAQUE inside its rect
    /// (the PSD convention); everything outside stays transparent black.
    fn layer_canvas_rgba8(&self, layer: &LayerRecord, cw: u32, ch: u32) -> Result<Vec<u8>> {
        let mut canvas = vec![0u8; (cw as usize) * (ch as usize) * 4];
        let lw = (layer.right - layer.left).max(0) as u32;
        let lh = (layer.bottom - layer.top).max(0) as u32;
        let plane_len = (lw as usize) * (lh as usize);
        if plane_len == 0 {
            // A degenerate rect contributes nothing — an empty layer is
            // a legal, meaningful PSD layer.
            return Ok(canvas);
        }

        let mut r = vec![0u8; plane_len];
        let mut g = vec![0u8; plane_len];
        let mut b = vec![0u8; plane_len];
        let mut a = vec![255u8; plane_len];
        for (ci, info) in layer.channels.iter().enumerate() {
            let dst = match info.id {
                0 => &mut r,
                1 => &mut g,
                2 => &mut b,
                -1 => &mut a,
                // The mask is read by `layer_mask_plate`; anything else
                // here is a spot/extra channel with no composite meaning.
                _ => continue,
            };
            let data = layer
                .channel_data
                .get(ci)
                .ok_or_else(|| PsdError::Malformed {
                    section: "layer channel image data",
                    detail: format!(
                        "layer \"{}\" declares {} channels but holds {} payloads",
                        layer.name(),
                        layer.channels.len(),
                        layer.channel_data.len()
                    ),
                })?;
            let plane = data.decode(self.container, lh, lw, self.header.depth)?;
            if plane.len() != plane_len {
                return Err(PsdError::Malformed {
                    section: "layer channel image data",
                    detail: format!(
                        "layer \"{}\" channel {} decoded to {} bytes, expected {plane_len}",
                        layer.name(),
                        info.id,
                        plane.len()
                    ),
                });
            }
            dst.copy_from_slice(&plane);
        }

        for ly in 0..lh as i64 {
            let dy = layer.top as i64 + ly;
            if dy < 0 || dy >= ch as i64 {
                continue;
            }
            for lx in 0..lw as i64 {
                let dx = layer.left as i64 + lx;
                if dx < 0 || dx >= cw as i64 {
                    continue;
                }
                let si = (ly * lw as i64 + lx) as usize;
                let di = ((dy * cw as i64 + dx) as usize) * 4;
                canvas[di] = r[si];
                canvas[di + 1] = g[si];
                canvas[di + 2] = b[si];
                canvas[di + 3] = a[si];
            }
        }
        Ok(canvas)
    }
}
