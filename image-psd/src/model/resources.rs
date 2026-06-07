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

//! Image resources section: `8BIM` blocks keyed by u16 id. Typed
//! bodies for the ids the engine consumes (ICC 1039, resolution 1005);
//! everything else opaque-verbatim. Every block additionally keeps its
//! complete source bytes (`raw_block`, signature through padding) for
//! the lazy-verbatim guard.
//!
//! Provenance: Adobe Photoshop File Format specification, "Image
//! Resources Section" + "Image Resource IDs".

/// Resource ids the model types (everything else is `Opaque`).
pub const RES_RESOLUTION_INFO: u16 = 1005;
pub const RES_ICC_PROFILE: u16 = 1039;

/// A Pascal string as stored: length byte + bytes, NO padding (padding
/// is contextual and applied by the emitters). Kept raw because PSD
/// names are MacRoman-ish legacy bytes, not guaranteed UTF-8.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PascalString {
    pub raw: Vec<u8>,
}

impl PascalString {
    /// Build from ASCII-safe text (lossy for anything else — the
    /// canonical name lives in `luni` anyway).
    pub fn new(s: &str) -> Self {
        let bytes: Vec<u8> = s.bytes().take(255).collect();
        let mut raw = Vec::with_capacity(bytes.len() + 1);
        raw.push(bytes.len() as u8);
        raw.extend_from_slice(&bytes);
        PascalString { raw }
    }

    pub fn text_lossy(&self) -> String {
        if self.raw.is_empty() {
            return String::new();
        }
        let n = self.raw[0] as usize;
        String::from_utf8_lossy(&self.raw[1..1 + n.min(self.raw.len() - 1)]).into_owned()
    }
}

/// Fixed-point resolution record (id 1005). Stored fields are the raw
/// 32-bit fixed-point values + display units — typed but lossless.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolutionInfo {
    /// Horizontal resolution, 16.16 fixed point, pixels per inch.
    pub h_res_fixed: u32,
    /// 1 = ppi, 2 = ppcm.
    pub h_res_unit: u16,
    /// 1 = inches, 2 = cm, 3 = points, 4 = picas, 5 = columns.
    pub width_unit: u16,
    pub v_res_fixed: u32,
    pub v_res_unit: u16,
    pub height_unit: u16,
}

impl ResolutionInfo {
    pub fn h_ppi(&self) -> f64 {
        self.h_res_fixed as f64 / 65536.0
    }

    pub fn v_ppi(&self) -> f64 {
        self.v_res_fixed as f64 / 65536.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResourceBody {
    IccProfile(Vec<u8>),
    ResolutionInfo(ResolutionInfo),
    Opaque,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageResourceBlock {
    pub id: u16,
    pub name: PascalString,
    /// Parsed view (`Opaque` for unmodeled ids; the bytes live in
    /// `raw_block`).
    pub body: ResourceBody,
    /// The complete block as read — signature, id, name (padded), size,
    /// data, padding. Verbatim re-emit unit (strategy 3); `None` only
    /// for constructed blocks, which re-encode canonically.
    pub raw_block: super::Raw,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ImageResources {
    pub blocks: Vec<ImageResourceBlock>,
}

impl ImageResources {
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.blocks.iter().find_map(|b| match &b.body {
            ResourceBody::IccProfile(p) => Some(p.as_slice()),
            _ => None,
        })
    }

    pub fn resolution(&self) -> Option<&ResolutionInfo> {
        self.blocks.iter().find_map(|b| match &b.body {
            ResourceBody::ResolutionInfo(r) => Some(r),
            _ => None,
        })
    }
}
