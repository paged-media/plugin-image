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

//! Residency manager (spec §9.1) — where each tile lives across the
//! three tiers:
//!
//! ```text
//! Tier 0  GPU texture pool   rgba16float TILE² slots (pool.rs)
//! Tier 1  wasm heap          Arc<[u8]> tile bytes (also the COW/undo tier)
//! Tier 2  OPFS scratch       evicted cold tiles (stub — BREAKAGE I-03)
//! ```
//!
//! This is the bookkeeping + the Tier-0↔Tier-1 transfer primitives:
//! `upload` writes heap bytes into a pool slot via `queue.write_texture`,
//! `download` reads a slot back to heap bytes. Tier 2 (OPFS) sits behind
//! the pending storage capability — its ops return a typed error naming
//! BREAKAGE I-03 so callers degrade rather than silently lose tiles.

use std::collections::HashMap;
use std::sync::Arc;

use image_core::{TextureSlot, TileCoord, TILE};

use crate::pool::TexturePool;
use crate::{GpuContext, GpuError};

const BYTES_PER_PIXEL: u32 = 8; // rgba16float
/// Tight (unpadded) heap-tile byte length — Tier 1 stores rows back to
/// back, like `image_core::TileData::Heap`.
pub const HEAP_TILE_BYTES: usize = (TILE * TILE * BYTES_PER_PIXEL) as usize;

/// Which tier currently holds a tile's pixels.
#[derive(Debug, Clone)]
pub enum Tier {
    /// Resident in the GPU texture pool (Tier 0).
    Tier0(TextureSlot),
    /// Resident in the wasm heap (Tier 1) — also the COW/undo tier.
    Tier1(Arc<[u8]>),
    /// Evicted to OPFS scratch (Tier 2) — stubbed on BREAKAGE I-03.
    Tier2,
}

/// Tracks the residency tier of every known tile and moves bytes between
/// Tier 0 and Tier 1. Holds no GPU resources itself — the pool owns the
/// textures; this layer is the map plus the transfer choreography.
#[derive(Default)]
pub struct ResidencyManager {
    tiers: HashMap<TileCoord, Tier>,
}

impl ResidencyManager {
    pub fn new() -> Self {
        ResidencyManager::default()
    }

    /// The tier a tile currently lives in, if known.
    pub fn tier(&self, coord: TileCoord) -> Option<&Tier> {
        self.tiers.get(&coord)
    }

    /// Record a tile as resident in a pool slot (Tier 0) without moving
    /// bytes — used after a kernel writes a slot directly.
    pub fn mark_resident(&mut self, coord: TileCoord, slot: TextureSlot) {
        self.tiers.insert(coord, Tier::Tier0(slot));
    }

    /// Upload a heap tile (Tier 1 bytes) into a pool slot (Tier 0). The
    /// slot must already be acquired from the pool; the tile becomes
    /// `Tier0(slot)`. `bytes` must be exactly one tightly-packed TILE²
    /// rgba16float tile.
    pub fn upload(
        &mut self,
        ctx: &GpuContext,
        pool: &mut TexturePool,
        coord: TileCoord,
        slot: TextureSlot,
        bytes: &[u8],
    ) -> Result<(), GpuError> {
        if bytes.len() != HEAP_TILE_BYTES {
            return Err(GpuError::Readback(format!(
                "upload expects {HEAP_TILE_BYTES} heap bytes, got {}",
                bytes.len()
            )));
        }
        let tex = pool
            .texture(slot)
            .ok_or_else(|| GpuError::Readback(format!("upload to unallocated slot {}", slot.0)))?;
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TILE * BYTES_PER_PIXEL),
                rows_per_image: Some(TILE),
            },
            wgpu::Extent3d {
                width: TILE,
                height: TILE,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit([]);
        pool.touch(slot);
        self.tiers.insert(coord, Tier::Tier0(slot));
        Ok(())
    }

    /// Download a pool slot (Tier 0) to tightly-packed heap bytes
    /// (Tier 1). Records the tile as `Tier1` and returns the bytes so the
    /// caller can journal them (COW/undo) or hand the freed slot back to
    /// the pool. Synchronous map (native path; the engines use async
    /// lanes), matching `execute_tile_once`.
    pub fn download(
        &mut self,
        ctx: &GpuContext,
        pool: &TexturePool,
        coord: TileCoord,
        slot: TextureSlot,
    ) -> Result<Arc<[u8]>, GpuError> {
        let tex = pool.texture(slot).ok_or_else(|| {
            GpuError::Readback(format!("download from unallocated slot {}", slot.0))
        })?;

        // Readback row stride must be aligned; we copy the padded buffer
        // out and repack to the tight heap layout.
        let row_bytes = TILE * BYTES_PER_PIXEL;
        let padded_row = row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("residency download"),
            size: (padded_row as u64) * (TILE as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("residency download"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(TILE),
                },
            },
            wgpu::Extent3d {
                width: TILE,
                height: TILE,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| GpuError::Readback("map callback dropped".into()))?
            .map_err(|e| GpuError::Readback(format!("map_async: {e:?}")))?;

        let mut heap = Vec::with_capacity(HEAP_TILE_BYTES);
        {
            let data = slice.get_mapped_range();
            for row in 0..TILE {
                let start = (row * padded_row) as usize;
                heap.extend_from_slice(&data[start..start + row_bytes as usize]);
            }
        }
        readback.unmap();

        let arc: Arc<[u8]> = Arc::from(heap.into_boxed_slice());
        self.tiers.insert(coord, Tier::Tier1(Arc::clone(&arc)));
        Ok(arc)
    }

    /// Tier-2 (OPFS scratch) spill — STUB. The storage capability is not
    /// yet expressed by the plugin SDK; until then a cold tile cannot be
    /// swapped out. See BREAKAGE I-03.
    pub fn swap_out(&mut self, _coord: TileCoord) -> Result<(), GpuError> {
        Err(GpuError::Tier2Unsupported)
    }

    /// Tier-2 (OPFS scratch) fault-in — STUB; same gate as `swap_out`.
    /// See BREAKAGE I-03.
    pub fn swap_in(&mut self, _coord: TileCoord) -> Result<Arc<[u8]>, GpuError> {
        Err(GpuError::Tier2Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coord(x: i32, y: i32) -> TileCoord {
        TileCoord { level: 0, x, y }
    }

    // Pure map bookkeeping + the Tier-2 stub contract: no device needed
    // (the upload/download GPU paths are covered by the conformance
    // dispatch test where an adapter exists).

    #[test]
    fn mark_and_query_tier() {
        let mut r = ResidencyManager::new();
        assert!(r.tier(coord(0, 0)).is_none());
        r.mark_resident(coord(0, 0), TextureSlot(7));
        assert!(matches!(
            r.tier(coord(0, 0)),
            Some(Tier::Tier0(TextureSlot(7)))
        ));
    }

    #[test]
    fn tier2_ops_report_the_breakage() {
        let mut r = ResidencyManager::new();
        let out = r.swap_out(coord(1, 1)).unwrap_err();
        let msg = out.to_string();
        assert!(
            msg.contains("I-03"),
            "Tier 2 error must name BREAKAGE I-03, got: {msg}"
        );
        assert!(r.swap_in(coord(1, 1)).is_err());
    }
}
