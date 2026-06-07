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

//! Per-channel image data. Stored as the on-disk compressed payload —
//! decode-on-demand (preservation + the 500 MB-PSB streaming budget).
//! RAW (0) and RLE (1) decode↔encode in M0; ZIP (2/3) inflate landed in
//! M1 (the `rendered` tier's data feed). Decode is on-demand: the stored
//! `bytes` (the verbatim on-disk payload, minus the 2-byte compression
//! tag the parser strips) stay canonical and are never mutated by a
//! decode — preservation + zero-edit byte-identity hold regardless.
//!
//! # Decode model (spec §10.4)
//!
//! `ChannelData::decode` produces a *planar* 8-bit buffer of
//! `rows * cols` bytes (one channel plane, row-major):
//!
//! * **RAW (0)** — `bytes` already IS the plane; validate length, copy.
//! * **RLE (1)** — `bytes` = a per-row byte-count table (`u16` in PSD,
//!   `u32` in PSB, one entry per row) followed by the PackBits-packed
//!   rows. Each row decodes to exactly `cols` bytes.
//! * **ZIP (2)** — `bytes` = a zlib stream; inflate to the plane.
//! * **ZIP-with-prediction (3)** — inflate, then undo the per-row
//!   horizontal byte delta (`out[i] += out[i-1]` within each row). Only
//!   the 8-bit case is implemented here; 16-bit prediction (which deltas
//!   16-bit samples, big-endian) is deferred to M2 and reported as
//!   `Unsupported` (the `depth` param is taken now so the signature is
//!   stable). Method 2 just inflates raw bytes for any depth.
//!
//! Provenance: Adobe Photoshop File Format specification, "Channel
//! Image Data"; zlib/DEFLATE (RFC 1950/1951); TIFF horizontal
//! differencing predictor (the prediction model for method 3).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    Raw,
    Rle,
    Zip,
    ZipPrediction,
}

impl Compression {
    pub const fn code(self) -> u16 {
        match self {
            Compression::Raw => 0,
            Compression::Rle => 1,
            Compression::Zip => 2,
            Compression::ZipPrediction => 3,
        }
    }

    pub fn from_code(c: u16) -> Option<Self> {
        Some(match c {
            0 => Compression::Raw,
            1 => Compression::Rle,
            2 => Compression::Zip,
            3 => Compression::ZipPrediction,
            _ => return None,
        })
    }
}

/// One channel's image data: the compression tag + the payload exactly
/// as stored (for RLE this INCLUDES the per-row byte-count table that
/// precedes the packed rows).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelData {
    pub compression: Compression,
    pub bytes: Vec<u8>,
}

use crate::compression::packbits;
use crate::{Container, PsdError, Result};

impl ChannelData {
    /// Decode this channel's stored payload into a planar 8-bit buffer
    /// of exactly `rows * cols` bytes (decode-on-demand; `self.bytes`
    /// is not mutated and remains the canonical on-disk form).
    ///
    /// `container` selects the RLE count-table width (`u16`/`u32`).
    /// `depth` is the per-sample bit depth: it is consulted only by
    /// ZIP-with-prediction (method 3), where any depth other than 8 is
    /// `Unsupported` (M2). Method 2 inflates the raw bytes for any depth.
    pub fn decode(
        &self,
        container: Container,
        rows: u32,
        cols: u32,
        depth: u16,
    ) -> Result<Vec<u8>> {
        let plane_len = (rows as usize)
            .checked_mul(cols as usize)
            .ok_or_else(|| malformed(format!("plane size {rows}×{cols} overflows usize")))?;
        match self.compression {
            Compression::Raw => self.decode_raw(plane_len),
            Compression::Rle => self.decode_rle(container, rows, cols, plane_len),
            Compression::Zip => inflate(&self.bytes, plane_len),
            Compression::ZipPrediction => {
                if depth != 8 {
                    return Err(PsdError::Unsupported(format!(
                        "ZIP-with-prediction at depth {depth} (only 8-bit lands in M1; \
                         16/32-bit prediction is M2)"
                    )));
                }
                let mut plane = inflate(&self.bytes, plane_len)?;
                undo_prediction_8bit(&mut plane, rows as usize, cols as usize);
                Ok(plane)
            }
        }
    }

    fn decode_raw(&self, plane_len: usize) -> Result<Vec<u8>> {
        if self.bytes.len() != plane_len {
            return Err(malformed(format!(
                "RAW channel is {} byte(s), expected {plane_len}",
                self.bytes.len()
            )));
        }
        Ok(self.bytes.clone())
    }

    fn decode_rle(
        &self,
        container: Container,
        rows: u32,
        cols: u32,
        plane_len: usize,
    ) -> Result<Vec<u8>> {
        let count_width = count_width(container);
        let rows = rows as usize;
        let cols = cols as usize;
        let table_len = rows
            .checked_mul(count_width)
            .ok_or_else(|| malformed("RLE count table size overflows usize".into()))?;
        if self.bytes.len() < table_len {
            return Err(malformed(format!(
                "RLE count table needs {table_len} byte(s), payload has {}",
                self.bytes.len()
            )));
        }

        // Per-row packed-byte counts, then the packed rows back to back.
        let mut counts = Vec::with_capacity(rows);
        for r in 0..rows {
            let off = r * count_width;
            let n = match count_width {
                2 => u16::from_be_bytes([self.bytes[off], self.bytes[off + 1]]) as usize,
                4 => u32::from_be_bytes([
                    self.bytes[off],
                    self.bytes[off + 1],
                    self.bytes[off + 2],
                    self.bytes[off + 3],
                ]) as usize,
                _ => unreachable!("count_width is 2 or 4"),
            };
            counts.push(n);
        }
        let packed_total: usize = counts.iter().sum();
        let packed = &self.bytes[table_len..];
        if packed.len() != packed_total {
            return Err(malformed(format!(
                "RLE packed rows total {} byte(s), count table claims {packed_total}",
                packed.len()
            )));
        }

        let mut plane = vec![0u8; plane_len];
        let mut src_off = 0usize;
        for (r, &n) in counts.iter().enumerate() {
            let row_src = &packed[src_off..src_off + n];
            src_off += n;
            let dst = &mut plane[r * cols..r * cols + cols];
            packbits::decode(row_src, dst)?;
        }
        Ok(plane)
    }

    /// Build a RAW `ChannelData` from a planar 8-bit buffer (writer-side
    /// feed for M2 edits). The stored payload IS the plane verbatim.
    pub fn encode_raw(planar: &[u8]) -> ChannelData {
        ChannelData {
            compression: Compression::Raw,
            bytes: planar.to_vec(),
        }
    }

    /// Build an RLE `ChannelData` from a planar 8-bit buffer: the per-row
    /// count table (`u16`/`u32` per `container`) followed by the
    /// canonically PackBits-encoded rows. `planar.len()` must be
    /// `rows * cols`.
    pub fn encode_rle(
        planar: &[u8],
        container: Container,
        rows: u32,
        cols: u32,
    ) -> Result<ChannelData> {
        let rows = rows as usize;
        let cols = cols as usize;
        let plane_len = rows
            .checked_mul(cols)
            .ok_or_else(|| malformed(format!("plane size {rows}×{cols} overflows usize")))?;
        if planar.len() != plane_len {
            return Err(malformed(format!(
                "RLE encode: planar buffer is {} byte(s), expected {plane_len}",
                planar.len()
            )));
        }
        let count_width = count_width(container);

        // Pack each row, recording its packed length for the table.
        let mut packed_rows: Vec<Vec<u8>> = Vec::with_capacity(rows);
        for r in 0..rows {
            packed_rows.push(packbits::encode(&planar[r * cols..r * cols + cols]));
        }

        let mut bytes = Vec::new();
        for row in &packed_rows {
            match count_width {
                2 => {
                    let n = u16::try_from(row.len()).map_err(|_| {
                        malformed(format!(
                            "RLE row of {} byte(s) exceeds u16 count",
                            row.len()
                        ))
                    })?;
                    bytes.extend_from_slice(&n.to_be_bytes());
                }
                4 => bytes.extend_from_slice(&(row.len() as u32).to_be_bytes()),
                _ => unreachable!("count_width is 2 or 4"),
            }
        }
        for row in &packed_rows {
            bytes.extend_from_slice(row);
        }
        Ok(ChannelData {
            compression: Compression::Rle,
            bytes,
        })
    }
}

/// RLE per-row count-table entry width: PSD = 2, PSB = 4.
fn count_width(container: Container) -> usize {
    match container {
        Container::Psd => 2,
        Container::Psb => 4,
    }
}

/// Inflate a zlib stream into a buffer of exactly `expected` bytes.
fn inflate(zlib: &[u8], expected: usize) -> Result<Vec<u8>> {
    let plane = miniz_oxide::inflate::decompress_to_vec_zlib(zlib)
        .map_err(|e| malformed(format!("ZIP inflate failed: {e:?}")))?;
    if plane.len() != expected {
        return Err(malformed(format!(
            "ZIP inflated to {} byte(s), expected {expected}",
            plane.len()
        )));
    }
    Ok(plane)
}

/// Undo TIFF-style horizontal differencing for an 8-bit plane, per row:
/// each sample after the first in a row holds its delta from the
/// previous sample, so `out[i] += out[i-1]` (wrapping) reconstructs it.
fn undo_prediction_8bit(plane: &mut [u8], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut plane[r * cols..r * cols + cols];
        for i in 1..row.len() {
            row[i] = row[i].wrapping_add(row[i - 1]);
        }
    }
}

#[inline]
fn malformed(detail: String) -> PsdError {
    PsdError::Malformed {
        section: "channel image data",
        detail,
    }
}
