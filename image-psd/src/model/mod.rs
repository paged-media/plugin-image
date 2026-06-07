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

//! The PSD-faithful data model (spec §10.4) — plain Rust data, no
//! engine dependencies. This model is the seed of the editor's
//! layer-tree document model; the companion editing-semantics spec
//! *lowers from* this structure.
//!
//! Frozen M0 phase 0. Per-module owners fill parse/emit in the M0
//! fan-out; the types here are the contract.

pub mod addl;
pub mod channel;
pub mod color_mode;
pub mod header;
pub mod layers;
pub mod resources;

pub use addl::{AdditionalLayerInfo, AddlBody, LsctData, SectionKind};
pub use channel::{ChannelData, Compression};
pub use color_mode::ColorModeData;
pub use header::{ColorMode, FileHeader};
pub use layers::{
    BlendRanges, ChannelInfo, GlobalLayerMask, LayerAndMaskInfo, LayerMaskData, LayerRecord,
};
pub use resources::{
    ImageResourceBlock, ImageResources, PascalString, ResolutionInfo, ResourceBody,
};

use crate::container::Container;

/// The lazy-verbatim guard (see crate docs): `Some(bytes)` = unmodified
/// since parse, the writer re-emits exactly these bytes; `None` =
/// constructed or edited, the writer re-encodes canonically from the
/// model.
pub type Raw = Option<Vec<u8>>;

/// The merged composite at the end of the file — every PSD ships its
/// own render oracle (spec §10.4). Kept opaque (compression tag + the
/// raw channel payload) until the M1 flatten pipeline maintains it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalImageData {
    pub compression: u16,
    pub raw: Vec<u8>,
}

impl GlobalImageData {
    /// The merged composite is the LAST section and is NOT length-framed —
    /// it runs to EOF (brief §11). A 2-byte compression tag + the rest.
    pub fn parse(r: &mut crate::reader::ByteReader) -> crate::Result<GlobalImageData> {
        let compression = r.u16()?;
        let raw = r.take(r.remaining())?.to_vec();
        Ok(GlobalImageData { compression, raw })
    }

    pub fn emit(&self, w: &mut crate::writer::ByteWriter) {
        w.u16(self.compression);
        w.bytes(&self.raw);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PsdFile {
    pub container: Container,
    pub header: FileHeader,
    pub color_mode: ColorModeData,
    pub resources: ImageResources,
    pub layer_mask: LayerAndMaskInfo,
    pub composite: GlobalImageData,
}
