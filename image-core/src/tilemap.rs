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

//! Sparse copy-on-write tile maps (spec §5.3) and the persistent
//! buffer (§8.1) — the Rust expression of the tiled-buffer idea:
//! sparse, tiled, mip-chained, swappable. Unallocated tiles read as a
//! constant, so sparse canvases cost nothing.

use std::collections::HashMap;
use std::sync::Arc;

use crate::format::PixelFormat;
use crate::tile::{Tile, TileCoord};

/// The value an unallocated tile reads as (premultiplied working-space
/// RGBA). Default: transparent black.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantPixel(pub [f32; 4]);

impl Default for ConstantPixel {
    fn default() -> Self {
        ConstantPixel([0.0; 4])
    }
}

#[derive(Debug, Clone)]
pub struct TileMap {
    pub format: PixelFormat,
    /// What unallocated coordinates read as.
    pub background: ConstantPixel,
    tiles: HashMap<TileCoord, Arc<Tile>>,
}

impl TileMap {
    pub fn new(format: PixelFormat) -> Self {
        TileMap {
            format,
            background: ConstantPixel::default(),
            tiles: HashMap::new(),
        }
    }

    pub fn get(&self, c: TileCoord) -> Option<&Arc<Tile>> {
        self.tiles.get(&c)
    }

    pub fn insert(&mut self, c: TileCoord, tile: Arc<Tile>) {
        self.tiles.insert(c, tile);
    }

    pub fn remove(&mut self, c: TileCoord) -> Option<Arc<Tile>> {
        self.tiles.remove(&c)
    }

    /// COW write access (§5.3): clones the tile only if shared
    /// (`Arc::make_mut`). The undo journal holds the old `Arc` — undo
    /// is O(changed tiles), never O(canvas) (§8.5).
    pub fn make_mut(&mut self, c: TileCoord) -> Option<&mut Tile> {
        self.tiles.get_mut(&c).map(Arc::make_mut)
    }

    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&TileCoord, &Arc<Tile>)> {
        self.tiles.iter()
    }
}

/// Residency bookkeeping carried by a persistent buffer. Skeleton in
/// M0 (Tier 0/1 only — Tier 2 is BREAKAGE I-03); the residency manager
/// in image-gpu owns the actual movement.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResidencyMeta {
    pub gpu_resident_tiles: u32,
    pub heap_resident_tiles: u32,
    pub swapped_tiles: u32,
}

/// `PersistentBuffer` (§8.1): TileMap + format + residency metadata.
/// Engine B's source-node state; defined in core now so the §5.3 freeze
/// covers it (Engine B itself is M2).
#[derive(Debug, Clone)]
pub struct PersistentBuffer {
    pub tiles: TileMap,
    pub format: PixelFormat,
    pub residency: ResidencyMeta,
}

impl PersistentBuffer {
    pub fn new(format: PixelFormat) -> Self {
        PersistentBuffer {
            tiles: TileMap::new(format),
            format,
            residency: ResidencyMeta::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::TileData;

    fn heap_tile(gen: u64) -> Arc<Tile> {
        Arc::new(Tile {
            format: PixelFormat::GPU_WORKING,
            data: TileData::Heap(Arc::from(vec![0u8; 16].into_boxed_slice())),
            generation: gen,
        })
    }

    #[test]
    fn cow_clones_only_when_shared() {
        let mut map = TileMap::new(PixelFormat::GPU_WORKING);
        let c = TileCoord {
            level: 0,
            x: 0,
            y: 0,
        };
        let tile = heap_tile(1);
        map.insert(c, Arc::clone(&tile)); // shared: map + local

        let t = map.make_mut(c).unwrap();
        t.generation = 2;

        // The journal's Arc still sees the old generation (COW).
        assert_eq!(tile.generation, 1);
        assert_eq!(map.get(c).unwrap().generation, 2);
    }

    #[test]
    fn sparse_is_empty_until_written() {
        let map = TileMap::new(PixelFormat::GPU_WORKING);
        assert!(map.is_empty());
        assert!(map
            .get(TileCoord {
                level: 0,
                x: 9,
                y: 9
            })
            .is_none());
        assert_eq!(map.background, ConstantPixel::default());
    }
}
