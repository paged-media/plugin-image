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

//! Additional layer information blocks (`8BIM`/`8B64` + fourcc key).
//! Typed in M0: `lsct` (section divider — the group structure), `luni`
//! (Unicode name), `lyid` (layer id). EVERYTHING else — `lfx2`, `SoCo`,
//! `TySh`, `SoLd`, `vmsk`, Adobe-private, undocumented — is preserved
//! opaquely (the constitutive §10.4 invariant). Each block keeps its
//! complete source bytes for verbatim re-emit; the parsed body is a
//! view.
//!
//! Provenance: Adobe Photoshop File Format specification, "Additional
//! Layer Information"; `luni` count/padding variance from black-box
//! corpus observation (synthesized fixtures pin both variants).

use super::Raw;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    /// 0 — any other type of layer.
    Other,
    /// 1 — open folder.
    OpenFolder,
    /// 2 — closed folder.
    ClosedFolder,
    /// 3 — bounding section divider (hidden marker closing a group).
    BoundingDivider,
}

impl SectionKind {
    pub const fn code(self) -> u32 {
        match self {
            SectionKind::Other => 0,
            SectionKind::OpenFolder => 1,
            SectionKind::ClosedFolder => 2,
            SectionKind::BoundingDivider => 3,
        }
    }

    pub fn from_code(c: u32) -> Option<Self> {
        Some(match c {
            0 => SectionKind::Other,
            1 => SectionKind::OpenFolder,
            2 => SectionKind::ClosedFolder,
            3 => SectionKind::BoundingDivider,
            _ => return None,
        })
    }
}

/// `lsct` payload: kind (+ optional blend key when length ≥ 12,
/// + optional sub-type when length ≥ 16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LsctData {
    pub kind: SectionKind,
    pub blend_key: Option<[u8; 4]>,
    pub sub_kind: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AddlBody {
    SectionDivider(LsctData),
    /// `luni` — UTF-16BE name, decoded.
    UnicodeName(String),
    /// `lyid`.
    LayerId(u32),
    /// Everything unmodeled. The bytes live in `raw_block`.
    Opaque,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdditionalLayerInfo {
    /// `8BIM` or `8B64`.
    pub sig: [u8; 4],
    pub key: [u8; 4],
    pub body: AddlBody,
    /// The complete block as read — signature through trailing padding.
    /// Verbatim re-emit unit; `None` only for constructed blocks.
    pub raw_block: Raw,
}

impl AdditionalLayerInfo {
    pub fn unicode_name(&self) -> Option<String> {
        match &self.body {
            AddlBody::UnicodeName(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn layer_id(&self) -> Option<u32> {
        match &self.body {
            AddlBody::LayerId(id) => Some(*id),
            _ => None,
        }
    }

    pub fn lsct(&self) -> Option<&LsctData> {
        match &self.body {
            AddlBody::SectionDivider(d) => Some(d),
            _ => None,
        }
    }
}
