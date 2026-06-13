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

//! `ImageSource` — streaming decode on CPU workers (spec §10.2,
//! frozen). Decoders advertise native downscale capabilities so the
//! shrink-on-load planner (§7.2, M1) can push work into the decoder.

use image_core::{PixelFormat, Region, TileSliceMut};

use crate::exif::Exif;
use crate::Result;

#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub width: u32,
    pub height: u32,
    /// The format `read_region` delivers (already mapped onto the
    /// spec-verbatim `ChannelLayout` set; a codec whose native layout
    /// differs converts at the slice boundary and records the native
    /// truth in `native_format`).
    pub format: PixelFormat,
    /// The container's own description, for provenance/diagnostics
    /// (e.g. "rgb8", "cmyk8+adobe-app14", "gray16").
    pub native_format: &'static str,
    /// Embedded ICC profile bytes, if any (interned by image-cms).
    pub icc: Option<Vec<u8>>,
    /// Raw EXIF payload, if any.
    pub exif: Option<Vec<u8>>,
    /// Container-provided mip levels (pyramid TIFF etc.); empty for
    /// single-resolution containers.
    pub native_mips: Vec<(u32, u32)>,
}

impl SourceInfo {
    /// Parse the raw EXIF payload (if any) into the advisory facts the
    /// ingest lane consumes: orientation, DPI, color-space tag. Returns
    /// `Exif::default()` (all `None`) when there is no EXIF or it is
    /// unreadable — EXIF is advisory and never fails a decode.
    pub fn exif_meta(&self) -> Exif {
        self.exif.as_deref().map(Exif::parse).unwrap_or_default()
    }
}

pub trait ImageSource {
    /// Parse headers; cheap. Must be called before `read_region`.
    fn probe(&mut self) -> Result<SourceInfo>;

    /// Native downscale factors the decoder can apply itself,
    /// ascending, always containing 1 (e.g. `[1, 2, 4, 8]` for JPEG
    /// DCT scaling; `[1]` for PNG).
    fn native_shrink(&self) -> &[u32];

    /// Decode `roi` (in post-shrink coordinates) at `shrink` (one of
    /// `native_shrink()`) into `out`. `out.format` must equal the
    /// probed `format` (no implicit conversions — spec §5.1).
    fn read_region(&mut self, roi: Region, shrink: u32, out: &mut TileSliceMut<'_>) -> Result<()>;
}
