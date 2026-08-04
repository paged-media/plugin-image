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
//! * **not 8-bit RGB** — 16/32-bit and CMYK/Lab are the M2 cast/CMS lane;
//! * **groups** (`lsct` section dividers) — a group has its own
//!   compositing rules (pass-through, isolation) that a flat stack does
//!   not reproduce;
//! * **clipping layers** (`clipping == 1`) — a clipped layer is masked by
//!   the one below it;
//! * **layer masks** (channel ids −2/−3) — the mask changes what the
//!   layer covers;
//! * **a budget overrun** — plates are CANVAS-EXTENT (the layer model's
//!   deliberate simplification), so N layers of a big canvas is N × 4
//!   bytes per pixel. Past [`MAX_IMPORT_BYTES`] the import declines
//!   instead of exhausting the wasm heap.
//!
//! Every refusal is a typed [`PsdError::Unsupported`] carrying the
//! reason, which the panel shows verbatim. "It flattened and did not say
//! why" is the failure mode this exists to avoid.
//!
//! Provenance: Adobe Photoshop File Format specification — Layer
//! Records (bounds, blend-mode key, opacity, flags), Channel Image Data
//! (per-channel decode, ids 0/1/2 = R/G/B and −1 = transparency),
//! Additional Layer Information (`lsct` section dividers, `luni` names).

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
}

/// The whole importable layer tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerImport {
    pub width: u32,
    pub height: u32,
    /// BOTTOM-first, the order PSD stores them in and the order the
    /// layer stack composites in.
    pub layers: Vec<LayerPlate>,
}

/// Is this record a group structural marker (divider/folder)? Such
/// records carry an `lsct` of kind 1/2/3 and hold no pixels.
fn is_group_marker(layer: &LayerRecord) -> bool {
    matches!(
        layer.addl.iter().find_map(|a| a.lsct()).map(|d| d.kind),
        Some(SectionKind::OpenFolder)
            | Some(SectionKind::ClosedFolder)
            | Some(SectionKind::BoundingDivider)
    )
}

impl PsdFile {
    /// Decode every pixel-bearing layer into a canvas-extent straight
    /// RGBA8 plate, bottom-first. See the module docs for the exact set
    /// of files this declines (and why declining is the right answer).
    pub fn layer_plates_rgba8(&self) -> Result<LayerImport> {
        let h = &self.header;
        if h.depth != 8 {
            return Err(PsdError::Unsupported(format!(
                "layer import at depth {} (8-bit only; 16/32-bit is the M2 cast lane)",
                h.depth
            )));
        }
        if h.color_mode != ColorMode::Rgb {
            return Err(PsdError::Unsupported(format!(
                "layer import for color mode {:?} (RGB only; CMYK/Lab are the M2 CMS lane)",
                h.color_mode
            )));
        }
        let (cw, ch) = (h.width, h.height);
        let canvas_texels = (cw as usize)
            .checked_mul(ch as usize)
            .ok_or_else(|| PsdError::Unsupported("canvas extent overflows usize".into()))?;
        let plate_bytes = canvas_texels
            .checked_mul(4)
            .ok_or_else(|| PsdError::Unsupported("canvas extent overflows usize".into()))?;

        // Structural gate FIRST: refuse before decoding a single byte.
        let mut pixel_layers = Vec::new();
        for layer in &self.layer_mask.layers {
            if is_group_marker(layer) {
                return Err(PsdError::Unsupported(
                    "layer import of a GROUPED PSD (lsct section dividers): group \
                     compositing (pass-through / isolation) is not modeled, so the \
                     merged composite is kept instead"
                        .into(),
                ));
            }
            if layer.clipping != 0 {
                return Err(PsdError::Unsupported(format!(
                    "layer import of a PSD with a CLIPPING layer (\"{}\"): a clipped \
                     layer is masked by the one below it, which is not modeled",
                    layer.name()
                )));
            }
            if layer.channels.iter().any(|c| c.id == -2 || c.id == -3) {
                return Err(PsdError::Unsupported(format!(
                    "layer import of a PSD with a LAYER MASK (\"{}\"): a mask changes \
                     what the layer covers, which is not modeled",
                    layer.name()
                )));
            }
            pixel_layers.push(layer);
        }
        if pixel_layers.is_empty() {
            return Err(PsdError::Unsupported(
                "layer import of a PSD with no layer records (the merged composite is \
                 all there is)"
                    .into(),
            ));
        }
        let total = plate_bytes.saturating_mul(pixel_layers.len());
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
        for layer in pixel_layers {
            layers.push(LayerPlate {
                name: layer.name(),
                blend_key: layer.blend_key,
                opacity: layer.opacity,
                hidden: (layer.flags & 0x02) != 0,
                rgba: self.layer_canvas_rgba8(layer, cw, ch)?,
            });
        }
        Ok(LayerImport {
            width: cw,
            height: ch,
            layers,
        })
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
                // The structural gate already refused masks; anything
                // else here is a spot/extra channel with no composite
                // meaning.
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
