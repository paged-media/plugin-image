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

//! Per-node sparse output cache (§8.1 "output cache"). An entry records
//! the inputs it was derived from — `params_hash` + the generation of
//! each input tile — so staleness is a pure equality check (§8.2), no
//! dirty flags to get wrong.

use std::collections::HashMap;
use std::sync::Arc;

use image_core::{ParamsHash, TileCoord};

/// A computed output tile: rgba16float bytes + its own generation
/// (bumped on every recompute, so downstream caches see the change).
#[derive(Debug, Clone)]
pub struct CachedTile {
    pub bytes: Arc<[u8]>,
    pub generation: u64,
}

/// What a cache entry was derived from — the validity key (§8.2).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Provenance {
    params_hash: ParamsHash,
    /// Generations of the input tiles consumed, in input order. For a
    /// windowed kernel this is the generations of every source tile in
    /// the gathered window.
    input_generations: Vec<u64>,
}

#[derive(Debug, Default)]
pub struct NodeCache {
    tiles: HashMap<TileCoord, (CachedTile, Provenance)>,
}

impl NodeCache {
    pub fn new() -> Self {
        NodeCache::default()
    }

    /// The cached tile IFF its provenance still matches.
    pub fn get(
        &self,
        coord: TileCoord,
        params_hash: ParamsHash,
        input_generations: &[u64],
    ) -> Option<&CachedTile> {
        self.tiles.get(&coord).and_then(|(tile, prov)| {
            if prov.params_hash == params_hash && prov.input_generations == input_generations {
                Some(tile)
            } else {
                None
            }
        })
    }

    pub fn put(
        &mut self,
        coord: TileCoord,
        tile: CachedTile,
        params_hash: ParamsHash,
        input_generations: Vec<u64>,
    ) {
        self.tiles.insert(
            coord,
            (
                tile,
                Provenance {
                    params_hash,
                    input_generations,
                },
            ),
        );
    }

    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Drop entries intersecting a damaged tile set (the eager sweep;
    /// the lazy provenance check below catches the rest, but pruning
    /// keeps the map from growing without bound under heavy editing).
    pub fn invalidate(&mut self, coords: &[TileCoord]) {
        for c in coords {
            self.tiles.remove(c);
        }
    }
}
