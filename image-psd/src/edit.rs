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

//! Mutatable beginnings (spec §15 M2): edits re-encode through the
//! preservation writer. Each op mutates the typed model in place and
//! **dirties the touched subtree** — it sets that subtree's lazy-verbatim
//! guard (`Raw`) to `None` so the writer re-encodes exactly that part
//! canonically while everything else still re-emits its source bytes
//! verbatim (spec §10.4, "write semantics under editing").
//!
//! The dirtying rule for a layer-record edit is two-level:
//!
//!  * the **record-local** guard `LayerRecord::extra_raw` is nulled when
//!    the edit changes the record's extra-data span (mask / blend ranges /
//!    name / addl blocks), so the record re-frames its `u32` extra length;
//!  * the **section** guard `LayerAndMaskInfo::section_raw` is ALSO nulled,
//!    because the layer & mask info section length frame (and the layer
//!    info sub-frame) must re-compute once any record's size changes.
//!
//! Untouched records keep `extra_raw = Some(..)`, so the re-encode path
//! re-emits them verbatim; only the dirtied subtree is normalized.
//!
//! Provenance: spec §10.4 (write semantics under editing); Adobe
//! Photoshop File Format specification — Layer Records, Channel Image
//! Data, Additional Layer Information (`luni`, `lsct`).

use crate::model::addl::{AddlBody, SectionKind};
use crate::model::channel::{ChannelData, Compression};
use crate::model::resources::PascalString;
use crate::model::{AdditionalLayerInfo, PsdFile};
use crate::{PsdError, Result};

/// `luni` additional-layer-info key.
const LUNI_KEY: [u8; 4] = *b"luni";
/// `lsct` additional-layer-info key (section divider).
const LSCT_KEY: [u8; 4] = *b"lsct";

/// Set a layer's opacity. Dirties the record's extra-data guard AND the
/// section guard (opacity is a record scalar, but the section re-frames
/// on any record-size touch — keep the two guards consistent so the
/// edited record never re-emits a stale verbatim span). The reparsed
/// model reflects the new opacity.
pub fn set_layer_opacity(file: &mut PsdFile, layer_idx: usize, opacity: u8) -> Result<()> {
    let layer = layer_mut(file, layer_idx)?;
    layer.opacity = opacity;
    layer.extra_raw = None;
    file.layer_mask.section_raw = None;
    Ok(())
}

/// Rename a layer. Updates BOTH the legacy Pascal name (the in-record
/// field) AND the `luni` Unicode-name additional block (the canonical
/// name) — inserting a `luni` block if the record had none — then dirties
/// the record's extra-data guard and the section guard.
pub fn set_layer_name(file: &mut PsdFile, layer_idx: usize, name: &str) -> Result<()> {
    let layer = layer_mut(file, layer_idx)?;

    // Legacy Pascal name (lossy for non-ASCII; `luni` is canonical).
    layer.name_legacy = PascalString::new(name);

    // Canonical Unicode name in the `luni` block. Update in place if one
    // exists, else append a constructed block (`raw_block = None` ⇒
    // re-encoded canonically by the addl emitter).
    match layer
        .addl
        .iter_mut()
        .find(|a| a.key == LUNI_KEY && a.unicode_name().is_some())
    {
        Some(luni) => {
            luni.body = AddlBody::UnicodeName(name.to_string());
            luni.raw_block = None;
        }
        None => layer.addl.push(AdditionalLayerInfo {
            sig: *b"8BIM",
            key: LUNI_KEY,
            body: AddlBody::UnicodeName(name.to_string()),
            raw_block: None,
        }),
    }

    layer.extra_raw = None;
    file.layer_mask.section_raw = None;
    Ok(())
}

/// Replace one channel's pixels with a planar 8-bit buffer, re-encoded
/// under `compression` (RAW or RLE). Updates the parallel
/// `ChannelInfo::data_len` to the new on-disk size (the 2-byte
/// compression tag + payload) and dirties the record's extra-data guard
/// and the section guard.
///
/// `rows`/`cols` describe the channel plane and must satisfy
/// `planar.len() == rows * cols`. ZIP re-encoding is not a writer path
/// (the writer never produces a deflate stream); RAW and RLE are the
/// canonical mutatable forms.
pub fn replace_channel_pixels(
    file: &mut PsdFile,
    layer_idx: usize,
    channel_idx: usize,
    planar: &[u8],
    compression: Compression,
    rows: u32,
    cols: u32,
) -> Result<()> {
    let container = file.container;
    let layer = layer_mut(file, layer_idx)?;

    if channel_idx >= layer.channels.len() || channel_idx >= layer.channel_data.len() {
        return Err(PsdError::Malformed {
            section: "channel image data",
            detail: format!(
                "channel index {channel_idx} out of range ({} channels, {} data)",
                layer.channels.len(),
                layer.channel_data.len()
            ),
        });
    }

    let cd = match compression {
        Compression::Raw => ChannelData::encode_raw(planar),
        Compression::Rle => ChannelData::encode_rle(planar, container, rows, cols)?,
        Compression::Zip | Compression::ZipPrediction => {
            return Err(PsdError::Unsupported(
                "replace_channel_pixels re-encodes RAW or RLE only (ZIP is not a writer path)"
                    .into(),
            ));
        }
    };

    // On-disk length = 2-byte compression tag + payload.
    let data_len = 2 + cd.bytes.len() as u64;
    layer.channel_data[channel_idx] = cd;
    layer.channels[channel_idx].data_len = data_len;

    layer.extra_raw = None;
    file.layer_mask.section_raw = None;
    Ok(())
}

/// Remove a layer. Drops the record (and its parallel channel data, which
/// lives inside the record) and dirties the section guard so the layer
/// info count word and the section frame re-compute.
///
/// Simple `lsct` divider bookkeeping: a group is encoded as a pair — an
/// open/closed-folder record above its members and a bounding-divider
/// record (kind 3) below them. When the removed layer is itself a folder
/// record, the matching bounding divider (the nearest divider BELOW it
/// in the flat, bottom-first list) is removed too, so the group brackets
/// stay balanced. Removing a plain content layer touches nothing else.
pub fn remove_layer(file: &mut PsdFile, layer_idx: usize) -> Result<()> {
    let layers = &mut file.layer_mask.layers;
    if layer_idx >= layers.len() {
        return Err(PsdError::Malformed {
            section: "layer record",
            detail: format!(
                "layer index {layer_idx} out of range ({} layers)",
                layers.len()
            ),
        });
    }

    let folder_kind = section_kind(&layers[layer_idx]);
    let is_folder = matches!(
        folder_kind,
        Some(SectionKind::OpenFolder) | Some(SectionKind::ClosedFolder)
    );

    // If removing a folder, also drop its matching bounding divider — the
    // nearest kind-3 divider strictly below it (lower index, bottom-first).
    let mut divider_idx = None;
    if is_folder {
        for i in (0..layer_idx).rev() {
            if matches!(section_kind(&layers[i]), Some(SectionKind::BoundingDivider)) {
                divider_idx = Some(i);
                break;
            }
        }
    }

    layers.remove(layer_idx);
    if let Some(di) = divider_idx {
        // `di < layer_idx`, so the index is still valid after the first
        // removal shifted only the elements above `layer_idx`.
        layers.remove(di);
    }

    file.layer_mask.section_raw = None;
    Ok(())
}

// (intentionally no trailing re-exports — `Container` is reached via
// `file.container` and the channel encoders take it directly.)

/// The `lsct` section-divider kind of a record, if it carries one.
fn section_kind(layer: &crate::model::LayerRecord) -> Option<SectionKind> {
    layer
        .addl
        .iter()
        .filter(|a| a.key == LSCT_KEY)
        .find_map(|a| a.lsct())
        .map(|d| d.kind)
}

/// Mutable layer accessor with a typed out-of-range error.
fn layer_mut(file: &mut PsdFile, layer_idx: usize) -> Result<&mut crate::model::LayerRecord> {
    let n = file.layer_mask.layers.len();
    file.layer_mask
        .layers
        .get_mut(layer_idx)
        .ok_or_else(|| PsdError::Malformed {
            section: "layer record",
            detail: format!("layer index {layer_idx} out of range ({n} layers)"),
        })
}
