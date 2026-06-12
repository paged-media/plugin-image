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

//! Merged-composite decode (the M4 ingest slice of spec §10.4): turn the
//! [`GlobalImageData`] section — every PSD's embedded render oracle —
//! into interleaved RGBA8 pixels for the editor placement path
//! (C-5 bytes in → decode → adjust → C-1 scene-layer composite).
//!
//! Scope (honest M4 cut, mirroring the M1 flatten oracle's corpus):
//! 8-bit RGB[A] and Grayscale[+A], RAW (0) and RLE (1) compression, PSD
//! and PSB count-table widths. 16/32-bit, CMYK/Lab/Indexed/Bitmap/Duotone
//! and ZIP-compressed composites answer [`PsdError::Unsupported`] — the
//! M2 cast/CMS lane. The PRESERVATION model is untouched: this is a pure
//! READ over the already-parsed section (the verbatim re-emit guarantees
//! hold regardless of whether the composite decodes).
//!
//! Alpha: the merged composite's first extra channel is transparency
//! ONLY when the layer-count sign flag said so
//! (`LayerAndMaskInfo::transparency_in_merged`); otherwise extra
//! channels are alpha/spot channels and the merged result is opaque
//! (Adobe Photoshop File Format specification — Layer Info, "If it is a
//! negative number, its absolute value is the number of layers and the
//! first alpha channel contains the transparency data").

use crate::compression::packbits;
use crate::model::{ColorMode, PsdFile};
use crate::{Container, PsdError, Result};

/// A decoded merged composite: tightly packed, interleaved RGBA8.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeRgba8 {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes, row-major RGBA, straight alpha.
    pub rgba: Vec<u8>,
}

/// Composite compression codes (Image Data Section). The section is one
/// tag for ALL channels, unlike per-layer channels — RAW stores the
/// planes back to back; RLE prefixes ONE count table covering every
/// channel's scanlines (`channels * height` entries), then the packed
/// scanlines in channel-major order.
const COMPRESSION_RAW: u16 = 0;
const COMPRESSION_RLE: u16 = 1;

impl PsdFile {
    /// Decode the merged composite to straight RGBA8. See the module
    /// docs for the supported subset; everything outside it is a clean
    /// [`PsdError::Unsupported`] (never a wrong-looking image).
    pub fn composite_rgba8(&self) -> Result<CompositeRgba8> {
        let h = &self.header;
        if h.depth != 8 {
            return Err(PsdError::Unsupported(format!(
                "composite decode at depth {} (8-bit only in the M4 ingest slice)",
                h.depth
            )));
        }
        let color_channels: usize = match h.color_mode {
            ColorMode::Rgb => 3,
            ColorMode::Grayscale => 1,
            other => {
                return Err(PsdError::Unsupported(format!(
                    "composite decode for color mode {other:?} (RGB/Grayscale only in \
                     the M4 ingest slice; CMYK/Lab are the M2 cast/CMS lane)"
                )));
            }
        };
        if (h.channels as usize) < color_channels {
            return Err(PsdError::Malformed {
                section: "image data",
                detail: format!(
                    "{:?} needs {color_channels} channel(s), header says {}",
                    h.color_mode, h.channels
                ),
            });
        }

        let planes = decode_planes(self)?;

        // The first channel BEYOND the color channels is transparency
        // exactly when the layer-count sign flag declared it.
        let alpha = if self.layer_mask.transparency_in_merged {
            planes.get(color_channels)
        } else {
            None
        };

        let (w, hh) = (h.width as usize, h.height as usize);
        let n = w * hh;
        let mut rgba = vec![0u8; n * 4];
        for i in 0..n {
            let (r, g, b) = match h.color_mode {
                ColorMode::Rgb => (planes[0][i], planes[1][i], planes[2][i]),
                // Grayscale: replicate the single plane.
                _ => (planes[0][i], planes[0][i], planes[0][i]),
            };
            let a = alpha.map_or(255, |p| p[i]);
            let o = i * 4;
            rgba[o] = r;
            rgba[o + 1] = g;
            rgba[o + 2] = b;
            rgba[o + 3] = a;
        }
        Ok(CompositeRgba8 {
            width: h.width,
            height: h.height,
            rgba,
        })
    }
}

/// Decode every channel of the composite section into planar 8-bit
/// buffers (`width * height` each), channel-major as stored.
fn decode_planes(file: &PsdFile) -> Result<Vec<Vec<u8>>> {
    let h = &file.header;
    let channels = h.channels as usize;
    let rows = h.height as usize;
    let cols = h.width as usize;
    let plane_len = rows.checked_mul(cols).ok_or_else(|| PsdError::Malformed {
        section: "image data",
        detail: format!("plane size {rows}×{cols} overflows usize"),
    })?;
    let data = &file.composite.raw;

    match file.composite.compression {
        COMPRESSION_RAW => {
            let expect = plane_len
                .checked_mul(channels)
                .ok_or_else(|| malformed("RAW composite size overflows usize"))?;
            if data.len() != expect {
                return Err(malformed(format!(
                    "RAW composite is {} byte(s), expected {expect}",
                    data.len()
                )));
            }
            Ok((0..channels)
                .map(|c| data[c * plane_len..(c + 1) * plane_len].to_vec())
                .collect())
        }
        COMPRESSION_RLE => {
            // ONE count table for all channels' scanlines, then the
            // packed rows back to back, channel-major.
            let count_width = match file.container {
                Container::Psd => 2,
                Container::Psb => 4,
            };
            let table_entries = rows
                .checked_mul(channels)
                .ok_or_else(|| malformed("RLE count table size overflows usize"))?;
            let table_len = table_entries * count_width;
            if data.len() < table_len {
                return Err(malformed(format!(
                    "RLE count table needs {table_len} byte(s), section has {}",
                    data.len()
                )));
            }
            let mut counts = Vec::with_capacity(table_entries);
            for e in 0..table_entries {
                let off = e * count_width;
                let n = match count_width {
                    2 => u16::from_be_bytes([data[off], data[off + 1]]) as usize,
                    _ => {
                        u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
                            as usize
                    }
                };
                counts.push(n);
            }
            let packed_total: usize = counts.iter().sum();
            let packed = &data[table_len..];
            if packed.len() != packed_total {
                return Err(malformed(format!(
                    "RLE packed rows total {} byte(s), count table claims {packed_total}",
                    packed.len()
                )));
            }

            let mut planes = Vec::with_capacity(channels);
            let mut src_off = 0usize;
            for c in 0..channels {
                let mut plane = vec![0u8; plane_len];
                for r in 0..rows {
                    let n = counts[c * rows + r];
                    let row_src = &packed[src_off..src_off + n];
                    src_off += n;
                    packbits::decode(row_src, &mut plane[r * cols..r * cols + cols])?;
                }
                planes.push(plane);
            }
            Ok(planes)
        }
        other => Err(PsdError::Unsupported(format!(
            "composite compression {other} (RAW/RLE only in the M4 ingest slice; \
             ZIP composites are the M2 lane)"
        ))),
    }
}

fn malformed(detail: impl Into<String>) -> PsdError {
    PsdError::Malformed {
        section: "image data",
        detail: detail.into(),
    }
}
