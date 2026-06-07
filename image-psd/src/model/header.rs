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

//! File header (26 bytes, fixed): `8BPS` + version + 6 reserved zero
//! bytes + channels + height + width + depth + color mode. Fully typed
//! — re-encoding is byte-identical by construction (preservation
//! strategy 1).
//!
//! Provenance: Adobe Photoshop File Format specification, "File Header
//! Section".

pub const SIGNATURE: [u8; 4] = *b"8BPS";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    Bitmap,
    Grayscale,
    Indexed,
    Rgb,
    Cmyk,
    Multichannel,
    Duotone,
    Lab,
}

impl ColorMode {
    pub const fn code(self) -> u16 {
        match self {
            ColorMode::Bitmap => 0,
            ColorMode::Grayscale => 1,
            ColorMode::Indexed => 2,
            ColorMode::Rgb => 3,
            ColorMode::Cmyk => 4,
            ColorMode::Multichannel => 7,
            ColorMode::Duotone => 8,
            ColorMode::Lab => 9,
        }
    }

    pub fn from_code(c: u16) -> Option<Self> {
        Some(match c {
            0 => ColorMode::Bitmap,
            1 => ColorMode::Grayscale,
            2 => ColorMode::Indexed,
            3 => ColorMode::Rgb,
            4 => ColorMode::Cmyk,
            7 => ColorMode::Multichannel,
            8 => ColorMode::Duotone,
            9 => ColorMode::Lab,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileHeader {
    /// 1..=56 channels including alphas.
    pub channels: u16,
    pub height: u32,
    pub width: u32,
    /// Bits per channel: 1 | 8 | 16 | 32.
    pub depth: u16,
    pub color_mode: ColorMode,
}

use crate::container::Container;
use crate::reader::ByteReader;
use crate::writer::ByteWriter;
use crate::{PsdError, Result};

impl FileHeader {
    /// Parse the fixed 26-byte header and decide the container from the
    /// version word. The signature and the 6 reserved bytes are validated
    /// (not stored) — the writer re-emits the canonical zero-reserved form,
    /// which is byte-identical for any conformant producer.
    pub fn parse(r: &mut ByteReader) -> Result<(Container, FileHeader)> {
        let sig = r.fourcc()?;
        if sig != SIGNATURE {
            return Err(PsdError::BadSignature(
                String::from_utf8_lossy(&sig).into_owned(),
            ));
        }
        let version = r.u16()?;
        let container = Container::from_version(version).ok_or_else(|| PsdError::Malformed {
            section: "header",
            detail: format!("unknown version {version}"),
        })?;
        // 6 reserved bytes, spec-mandated zero. We reject non-zero rather
        // than silently normalize, so byte-identity is an honest claim.
        let reserved = r.take(6)?;
        if reserved != [0u8; 6] {
            return Err(PsdError::Malformed {
                section: "header",
                detail: "reserved bytes are not zero".into(),
            });
        }
        let channels = r.u16()?;
        let height = r.u32()?;
        let width = r.u32()?;
        let depth = r.u16()?;
        let color_code = r.u16()?;
        let color_mode = ColorMode::from_code(color_code).ok_or_else(|| PsdError::Malformed {
            section: "header",
            detail: format!("unknown color mode {color_code}"),
        })?;
        Ok((
            container,
            FileHeader {
                channels,
                height,
                width,
                depth,
                color_mode,
            },
        ))
    }

    /// Emit the canonical 26-byte header. Fully typed scalars: re-encode
    /// is byte-identical by construction (preservation strategy 1).
    pub fn emit(&self, w: &mut ByteWriter, container: Container) {
        w.fourcc(SIGNATURE);
        w.u16(container.version());
        w.bytes(&[0u8; 6]);
        w.u16(self.channels);
        w.u32(self.height);
        w.u32(self.width);
        w.u16(self.depth);
        w.u16(self.color_mode.code());
    }
}
