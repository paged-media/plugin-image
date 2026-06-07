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

//! Tier-0 texture pool (spec §9.1) — `rgba16float` TILE² slots with LRU
//! touch order. GPU texture memory lives OUTSIDE the wasm32 4 GB heap,
//! so this is deliberately the largest residency tier; eviction spills
//! the least-recently-used slot to Tier 1 (`residency`).
//!
//! The pool grows lazily up to a fixed capacity: `acquire` reuses a
//! freed slot or allocates a fresh texture until full, then the caller
//! must `evict_lru` to make room. The slot bookkeeping (free list + LRU
//! order) is a pure data structure (`SlotBook`) so it is unit-tested
//! with no device — the GPU path is exercised only where an adapter is
//! present (spec §6.3 device-skip discipline).

use image_core::{TextureSlot, TILE};

use crate::GpuContext;

/// Pure slot bookkeeping: which slots are live, the free list for reuse,
/// and the LRU touch order (front = least recently used). Split out from
/// the wgpu textures so it carries no device dependency and is fully
/// unit-testable.
#[derive(Debug)]
struct SlotBook {
    capacity: u32,
    /// Next never-yet-allocated index; grows until it hits `capacity`.
    high_water: u32,
    /// Released slots, ready to hand back before growing.
    free: Vec<u32>,
    /// Occupied slots, least-recently-touched first.
    lru: Vec<u32>,
}

impl SlotBook {
    fn new(capacity: u32) -> Self {
        SlotBook {
            capacity,
            high_water: 0,
            free: Vec::new(),
            lru: Vec::new(),
        }
    }

    /// Claim a slot, preferring a freed one, then growth. `None` when the
    /// pool is full and every slot is live (the caller must evict first).
    /// Returns `(index, fresh)` where `fresh` means a texture must be
    /// allocated for it.
    fn acquire(&mut self) -> Option<(u32, bool)> {
        if let Some(i) = self.free.pop() {
            self.lru.push(i);
            return Some((i, false));
        }
        if self.high_water < self.capacity {
            let i = self.high_water;
            self.high_water += 1;
            self.lru.push(i);
            return Some((i, true));
        }
        None
    }

    /// Return a slot to the free list. Idempotent against a slot already
    /// freed (it simply drops out of the LRU order).
    fn release(&mut self, index: u32) {
        if let Some(pos) = self.lru.iter().position(|&i| i == index) {
            self.lru.remove(pos);
            self.free.push(index);
        }
    }

    /// Move a live slot to the most-recently-used end. A no-op for a slot
    /// that is not currently occupied.
    fn touch(&mut self, index: u32) {
        if let Some(pos) = self.lru.iter().position(|&i| i == index) {
            self.lru.remove(pos);
            self.lru.push(index);
        }
    }

    /// Evict the least-recently-used live slot, returning its index. The
    /// slot does NOT re-enter the free list — eviction hands ownership of
    /// the texture to the spill path, which re-acquires it explicitly.
    fn evict_lru(&mut self) -> Option<u32> {
        if self.lru.is_empty() {
            None
        } else {
            Some(self.lru.remove(0))
        }
    }

    fn live(&self) -> usize {
        self.lru.len()
    }
}

/// A handle into the pool's Tier-0 textures. Mirrors `image_core::TextureSlot`
/// (the engine-agnostic newtype); the pool owns the backing wgpu texture.
pub use image_core::TextureSlot as PoolSlot;

/// LRU texture pool of `rgba16float` TILE² slots (Tier 0, spec §9.1).
pub struct TexturePool {
    book: SlotBook,
    /// Indexed by slot; `high_water` of these are allocated. `None` marks
    /// an evicted slot whose texture was reclaimed by the spill path.
    textures: Vec<Option<wgpu::Texture>>,
}

impl TexturePool {
    /// A pool over a device, growing up to `capacity` TILE² slots.
    pub fn new(capacity: u32) -> Self {
        TexturePool {
            book: SlotBook::new(capacity),
            textures: Vec::with_capacity(capacity as usize),
        }
    }

    /// One Tier-0 texture: `rgba16float`, TILE², usable as a kernel input
    /// (sampled via `textureLoad`), a storage output, and a copy
    /// endpoint for residency upload/download.
    fn make_slot_texture(ctx: &GpuContext, index: u32) -> wgpu::Texture {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("pool slot {index}")),
            size: wgpu::Extent3d {
                width: TILE,
                height: TILE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    /// Claim a slot, allocating a texture lazily on first use of a fresh
    /// index. `None` when the pool is full of live slots — the caller
    /// evicts (`evict_lru`) and retries.
    pub fn acquire(&mut self, ctx: &GpuContext) -> Option<TextureSlot> {
        let (index, fresh) = self.book.acquire()?;
        if fresh {
            debug_assert_eq!(index as usize, self.textures.len());
            self.textures
                .push(Some(Self::make_slot_texture(ctx, index)));
        } else if self.textures[index as usize].is_none() {
            // A reused index whose texture was reclaimed by a prior
            // eviction: re-allocate before handing it back.
            self.textures[index as usize] = Some(Self::make_slot_texture(ctx, index));
        }
        Some(TextureSlot(index))
    }

    /// Mark a slot as most-recently-used (call on every read/write).
    pub fn touch(&mut self, slot: TextureSlot) {
        self.book.touch(slot.0);
    }

    /// Return a slot to the pool for reuse.
    pub fn release(&mut self, slot: TextureSlot) {
        self.book.release(slot.0);
    }

    /// The texture backing a live slot.
    pub fn texture(&self, slot: TextureSlot) -> Option<&wgpu::Texture> {
        self.textures.get(slot.0 as usize).and_then(|t| t.as_ref())
    }

    /// Evict the least-recently-used live slot for a Tier-1 spill,
    /// returning the freed slot AND its texture so the residency manager
    /// can download the bytes before the texture is reclaimed. The slot's
    /// index re-enters the free list for a later `acquire`.
    pub fn evict_lru(&mut self) -> Option<(TextureSlot, wgpu::Texture)> {
        let index = self.book.evict_lru()?;
        let texture = self.textures[index as usize].take()?;
        // The index is now reusable; the texture left with the caller.
        self.book.free.push(index);
        Some((TextureSlot(index), texture))
    }

    /// Number of live (acquired, not released/evicted) slots.
    pub fn live(&self) -> usize {
        self.book.live()
    }

    /// The pool's slot capacity.
    pub fn capacity(&self) -> u32 {
        self.book.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure bookkeeping: no device needed (spec §6.3 device-skip rule —
    // these run everywhere, the GPU paths skip when no adapter exists).

    #[test]
    fn acquire_grows_then_blocks_at_capacity() {
        let mut b = SlotBook::new(3);
        assert_eq!(b.acquire(), Some((0, true)));
        assert_eq!(b.acquire(), Some((1, true)));
        assert_eq!(b.acquire(), Some((2, true)));
        // Full and all live: no slot until something is freed/evicted.
        assert_eq!(b.acquire(), None);
        assert_eq!(b.live(), 3);
    }

    #[test]
    fn release_then_reacquire_reuses_index_without_growth() {
        let mut b = SlotBook::new(2);
        let (a, _) = b.acquire().unwrap();
        let _ = b.acquire().unwrap();
        b.release(a);
        // The freed index comes back, NOT a fresh allocation.
        assert_eq!(b.acquire(), Some((a, false)));
        assert_eq!(b.high_water, 2);
    }

    #[test]
    fn touch_reorders_lru_eviction_victim() {
        let mut b = SlotBook::new(3);
        let (s0, _) = b.acquire().unwrap();
        let (s1, _) = b.acquire().unwrap();
        let (s2, _) = b.acquire().unwrap();
        // Touch s0 so the LRU victim becomes s1, not s0.
        b.touch(s0);
        assert_eq!(b.evict_lru(), Some(s1));
        // s2 is now the new LRU front (s0 was most-recently touched).
        assert_eq!(b.evict_lru(), Some(s2));
        assert_eq!(b.evict_lru(), Some(s0));
        assert_eq!(b.evict_lru(), None);
    }

    #[test]
    fn evicted_index_is_reusable() {
        let mut b = SlotBook::new(2);
        let (s0, _) = b.acquire().unwrap();
        let _ = b.acquire().unwrap();
        assert_eq!(b.evict_lru(), Some(s0));
        // The pool wrapper pushes the evicted index back onto `free`;
        // model that here so the book matches the pool's contract.
        b.free.push(s0);
        assert_eq!(b.acquire(), Some((s0, false)));
    }

    #[test]
    fn release_is_idempotent_against_unknown_slot() {
        let mut b = SlotBook::new(2);
        let (s0, _) = b.acquire().unwrap();
        b.release(s0);
        // Releasing again must not double-count into the free list.
        b.release(s0);
        assert_eq!(b.free.len(), 1);
    }
}
