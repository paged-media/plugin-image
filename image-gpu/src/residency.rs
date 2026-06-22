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
//!
//! ## Residency hardening under over-subscription (spec §15 M3)
//!
//! The headline M3 item was an "OPFS swap tier hardened", but OPFS is the
//! Tier-2 storage capability the plugin SDK does not expose yet (BREAKAGE
//! I-03), so there is nothing to harden there. What *can* be hardened —
//! and what the engines actually hit first — is the **Tier-0↔Tier-1**
//! ladder when the working set is larger than the GPU pool: every kernel
//! over a big image touches more tiles than the pool has slots, so the
//! pool must spill its least-recently-used slot to the heap and fault it
//! back when re-requested, byte-identically.
//!
//! [`ResidencyManager::acquire_or_spill`] is that hardened path: it asks
//! the pool for a slot, and when the pool is full it evicts the LRU slot,
//! downloads its pixels to Tier 1 (so the bytes survive), and re-acquires
//! the freed index — never panicking, never leaking a slot, never losing
//! a tile. A re-requested evicted tile reloads from Tier 1 with
//! [`ResidencyManager::tier1_bytes`] and round-trips byte-identically
//! (`upload → evict → download == original`). Tier 2 stays a typed
//! `Tier2Unsupported` error.
//!
//! To make eviction self-describing the manager keeps a reverse
//! `slot → coord` index alongside the forward `coord → Tier` map, so when
//! the pool reports "I evicted slot N" the manager knows which tile that
//! was and where to journal its bytes — without the caller threading a
//! slot→coord table of its own.

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

/// What [`ResidencyManager::acquire_or_spill`] did to satisfy a request.
#[derive(Debug)]
pub struct Acquired {
    /// The pool slot now owned by the requested coord.
    pub slot: TextureSlot,
    /// `Some((coord, bytes))` when an LRU tile was evicted to Tier 1 to
    /// free this slot. The bytes are the journalled Tier-1 copy (also
    /// retained in the manager and reachable via [`tier1_bytes`]); the
    /// caller may use them to drive COW/undo. `None` when the pool had a
    /// free slot and nothing was spilled.
    ///
    /// [`tier1_bytes`]: ResidencyManager::tier1_bytes
    pub spilled: Option<(TileCoord, Arc<[u8]>)>,
}

/// Tracks the residency tier of every known tile and moves bytes between
/// Tier 0 and Tier 1. Holds no GPU resources itself — the pool owns the
/// textures; this layer is the map plus the transfer choreography.
///
/// Two indexes are kept in lockstep: `tiers` maps a coord to its current
/// tier, and `slot_owner` maps a live Tier-0 slot back to the coord that
/// occupies it. The reverse index lets [`acquire_or_spill`] turn the
/// pool's "evicted slot N" into "evicted tile C" so it can spill C's
/// bytes to Tier 1 on its own.
///
/// [`acquire_or_spill`]: ResidencyManager::acquire_or_spill
#[derive(Default)]
pub struct ResidencyManager {
    tiers: HashMap<TileCoord, Tier>,
    /// Reverse map: which coord currently occupies each Tier-0 slot.
    slot_owner: HashMap<u32, TileCoord>,
}

impl ResidencyManager {
    pub fn new() -> Self {
        ResidencyManager::default()
    }

    /// The tier a tile currently lives in, if known.
    pub fn tier(&self, coord: TileCoord) -> Option<&Tier> {
        self.tiers.get(&coord)
    }

    /// The coord currently occupying a Tier-0 slot, if any.
    pub fn slot_owner(&self, slot: TextureSlot) -> Option<TileCoord> {
        self.slot_owner.get(&slot.0).copied()
    }

    /// The journalled Tier-1 bytes for a tile, if it currently lives on
    /// the heap (e.g. after an eviction). The handle to fault an evicted
    /// tile back in: the caller acquires a slot and `upload`s these bytes.
    pub fn tier1_bytes(&self, coord: TileCoord) -> Option<Arc<[u8]>> {
        match self.tiers.get(&coord) {
            Some(Tier::Tier1(bytes)) => Some(Arc::clone(bytes)),
            _ => None,
        }
    }

    /// Number of tiles currently resident in a Tier-0 slot. Mirrors the
    /// pool's live count when the two are driven together; a divergence
    /// is a bookkeeping leak.
    pub fn tier0_count(&self) -> usize {
        self.slot_owner.len()
    }

    /// Internal: a tile now lives on the GPU in `slot`. Keeps both maps
    /// consistent — vacates whatever slot the coord held before, and
    /// evicts any stale owner of the target slot from the forward map.
    fn set_tier0(&mut self, coord: TileCoord, slot: TextureSlot) {
        // If this coord already sat in a different slot, free that slot's
        // reverse entry first.
        if let Some(Tier::Tier0(old)) = self.tiers.get(&coord) {
            if old.0 != slot.0 {
                self.slot_owner.remove(&old.0);
            }
        }
        self.slot_owner.insert(slot.0, coord);
        self.tiers.insert(coord, Tier::Tier0(slot));
    }

    /// Internal: a tile is no longer resident in any Tier-0 slot. Drops
    /// the reverse entry for the slot it held (if it held one).
    fn clear_tier0(&mut self, coord: TileCoord) {
        if let Some(Tier::Tier0(slot)) = self.tiers.get(&coord) {
            self.slot_owner.remove(&slot.0);
        }
    }

    /// Record a tile as resident in a pool slot (Tier 0) without moving
    /// bytes — used after a kernel writes a slot directly.
    pub fn mark_resident(&mut self, coord: TileCoord, slot: TextureSlot) {
        self.set_tier0(coord, slot);
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
        self.set_tier0(coord, slot);
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
        let arc = Self::read_texture_to_heap(ctx, tex)?;
        // The slot's bytes now live on the heap (Tier 1).
        self.clear_tier0(coord);
        self.tiers.insert(coord, Tier::Tier1(Arc::clone(&arc)));
        Ok(arc)
    }

    /// The raw Tier-0→heap readback: copy a `rgba16float` TILE² texture
    /// into tightly-packed heap bytes. Shared by [`download`] (slot still
    /// in the pool) and the spill path (the pool already detached the
    /// texture). Does no bookkeeping — pure transfer.
    ///
    /// [`download`]: ResidencyManager::download
    fn read_texture_to_heap(ctx: &GpuContext, tex: &wgpu::Texture) -> Result<Arc<[u8]>, GpuError> {
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

        Ok(Arc::from(heap.into_boxed_slice()))
    }

    /// Acquire a Tier-0 slot for `coord`, spilling the least-recently-used
    /// tile to Tier 1 (heap) when the pool is full — the hardened
    /// over-subscription path (spec §15 M3; see the module docs).
    ///
    /// Behaviour:
    /// - If the pool has a slot, takes it and records `coord` as Tier 0.
    /// - If the pool is full, evicts its LRU slot, downloads the victim's
    ///   pixels to the heap (so they survive), records the victim as
    ///   Tier 1, then re-acquires the freed index for `coord`.
    /// - Never panics and never leaks a slot: every evicted slot is either
    ///   reclaimed for `coord` or its bytes are journalled to Tier 1.
    ///
    /// Returns the slot now owned by `coord` plus, when a spill happened,
    /// the evicted `(coord, bytes)`. The caller still has to `upload`
    /// `coord`'s own pixels into the returned slot (this only frees and
    /// hands over the slot); for a fault-in of a previously-evicted tile,
    /// fetch its bytes with [`tier1_bytes`] and upload them.
    ///
    /// [`tier1_bytes`]: ResidencyManager::tier1_bytes
    pub fn acquire_or_spill(
        &mut self,
        ctx: &GpuContext,
        pool: &mut TexturePool,
        coord: TileCoord,
    ) -> Result<Acquired, GpuError> {
        if let Some(slot) = pool.acquire(ctx) {
            self.set_tier0(coord, slot);
            return Ok(Acquired {
                slot,
                spilled: None,
            });
        }

        // Pool full: evict the LRU slot and spill its bytes to Tier 1.
        let (evicted_slot, texture) = pool.evict_lru().ok_or_else(|| {
            // An empty pool that still cannot acquire is a capacity-0 pool;
            // surface it rather than spin.
            GpuError::Readback("pool exhausted with no slot to evict (capacity 0?)".into())
        })?;
        let victim = self.slot_owner.remove(&evicted_slot.0).ok_or_else(|| {
            GpuError::Readback(format!(
                "evicted slot {} had no recorded owner (residency/pool desync)",
                evicted_slot.0
            ))
        })?;
        let bytes = Self::read_texture_to_heap(ctx, &texture)?;
        // Texture is dropped here, after its bytes are safely on the heap.
        drop(texture);
        self.tiers.insert(victim, Tier::Tier1(Arc::clone(&bytes)));

        // The pool returned `evicted_slot`'s index to its free list, so a
        // second acquire reuses it (and re-allocates the texture).
        let slot = pool.acquire(ctx).ok_or_else(|| {
            GpuError::Readback("pool failed to re-acquire freed slot after eviction".into())
        })?;
        self.set_tier0(coord, slot);
        Ok(Acquired {
            slot,
            spilled: Some((victim, bytes)),
        })
    }

    /// The bookkeeping half of a Tier-0→Tier-1 spill, device-free: record
    /// that `coord`'s pixels now live on the heap as `bytes`, vacating
    /// whatever Tier-0 slot it held. Returns the slot it vacated (if it
    /// was resident), so the caller can hand that slot's index back to the
    /// pool. The GPU path ([`acquire_or_spill`]) does this after reading
    /// the texture; exposed separately so the spill *policy* (which tile,
    /// byte preservation, fault-in) is unit-testable with no adapter.
    ///
    /// [`acquire_or_spill`]: ResidencyManager::acquire_or_spill
    pub fn spill_to_tier1(&mut self, coord: TileCoord, bytes: Arc<[u8]>) -> Option<TextureSlot> {
        let vacated = match self.tiers.get(&coord) {
            Some(Tier::Tier0(slot)) => Some(*slot),
            _ => None,
        };
        if let Some(slot) = vacated {
            self.slot_owner.remove(&slot.0);
        }
        self.tiers.insert(coord, Tier::Tier1(bytes));
        vacated
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

    #[test]
    fn reverse_index_tracks_slot_owner() {
        let mut r = ResidencyManager::new();
        r.mark_resident(coord(2, 3), TextureSlot(5));
        assert_eq!(r.slot_owner(TextureSlot(5)), Some(coord(2, 3)));
        assert_eq!(r.tier0_count(), 1);
        // Re-homing the same coord to a new slot must vacate the old one.
        r.mark_resident(coord(2, 3), TextureSlot(9));
        assert_eq!(r.slot_owner(TextureSlot(5)), None);
        assert_eq!(r.slot_owner(TextureSlot(9)), Some(coord(2, 3)));
        assert_eq!(r.tier0_count(), 1);
    }

    #[test]
    fn spill_to_tier1_vacates_slot_and_preserves_bytes() {
        let mut r = ResidencyManager::new();
        r.mark_resident(coord(0, 0), TextureSlot(3));
        let bytes: Arc<[u8]> = Arc::from(vec![7u8; HEAP_TILE_BYTES].into_boxed_slice());
        let vacated = r.spill_to_tier1(coord(0, 0), Arc::clone(&bytes));
        assert_eq!(vacated, Some(TextureSlot(3)), "spill returns vacated slot");
        // The slot is now free (no owner), the tile is on the heap.
        assert_eq!(r.slot_owner(TextureSlot(3)), None);
        assert_eq!(r.tier0_count(), 0);
        let back = r.tier1_bytes(coord(0, 0)).expect("tile now on Tier 1");
        assert_eq!(&*back, &*bytes, "Tier-1 bytes survive the spill verbatim");
    }

    #[test]
    fn fault_in_round_trips_bytes_through_tier1() {
        // CPU model of evict→reload: the LRU victim's bytes go to Tier 1,
        // then a fault-in re-marks it Tier 0 — bytes must be byte-identical.
        let mut r = ResidencyManager::new();
        let original: Arc<[u8]> = {
            let mut v = vec![0u8; HEAP_TILE_BYTES];
            for (i, b) in v.iter_mut().enumerate() {
                *b = (i % 251) as u8;
            }
            Arc::from(v.into_boxed_slice())
        };
        r.mark_resident(coord(4, 4), TextureSlot(1));
        r.spill_to_tier1(coord(4, 4), Arc::clone(&original));
        // Re-request: read the journalled bytes (what the caller uploads).
        let reloaded = r.tier1_bytes(coord(4, 4)).expect("evicted tile reloads");
        assert_eq!(&*reloaded, &*original, "reload is byte-identical to upload");
        // After re-upload the caller re-marks Tier 0.
        r.mark_resident(coord(4, 4), TextureSlot(1));
        assert!(matches!(
            r.tier(coord(4, 4)),
            Some(Tier::Tier0(TextureSlot(1)))
        ));
        assert_eq!(r.tier1_bytes(coord(4, 4)), None, "no longer on the heap");
    }

    #[test]
    fn tier1_bytes_none_for_tier0_and_unknown() {
        let mut r = ResidencyManager::new();
        assert_eq!(r.tier1_bytes(coord(0, 0)), None, "unknown tile");
        r.mark_resident(coord(0, 0), TextureSlot(0));
        assert_eq!(
            r.tier1_bytes(coord(0, 0)),
            None,
            "Tier-0 tile has no heap copy"
        );
    }
}
