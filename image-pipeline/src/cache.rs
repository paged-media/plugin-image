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

//! The operation cache (spec §7.1): memoizes a node's materialized
//! `TileMap` keyed on `(op id, ParamsHash, input ContentHash)`. Aligned
//! with salsa semantics so pipeline results participate in document
//! dependency tracking ("asset pyramid depends on source bytes + ICC +
//! recipe").
//!
//! M0 is the honest cut: the cache is unbounded (the LRU bound is M1
//! once the residency manager owns eviction — BREAKAGE I-03) and stores
//! the WHOLE per-node `TileMap` rather than per-region slices. The hit
//! counter is exposed so tests can assert a re-pull is served from the
//! cache rather than recomputed.

use std::collections::HashMap;

use image_core::{ContentHash, ParamsHash, TileMap};

/// Cache identity for one materialized node (§7.1). `input` is the
/// content hash of the upstream node's result (a leaf source has no
/// upstream, so its `input` is the hash of its own decoded identity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpKey {
    pub op_id: u64,
    pub params: ParamsHash,
    pub input: ContentHash,
}

#[derive(Default)]
pub struct OperationCache {
    entries: HashMap<OpKey, TileMap>,
    hits: u64,
    misses: u64,
}

impl OperationCache {
    pub fn new() -> Self {
        OperationCache::default()
    }

    /// Look up a materialized node. Bumps the hit counter on a hit, the
    /// miss counter otherwise — the caller computes and `insert`s on a
    /// miss.
    pub fn get(&mut self, key: OpKey) -> Option<&TileMap> {
        if self.entries.contains_key(&key) {
            self.hits += 1;
            self.entries.get(&key)
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, key: OpKey, map: TileMap) {
        self.entries.insert(key, map);
    }

    /// Total cache hits since construction — the test hook proving a
    /// re-pull is memoized, not recomputed.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image_core::PixelFormat;

    fn key(input: u64) -> OpKey {
        OpKey {
            op_id: 1,
            params: ParamsHash(7),
            input: ContentHash(input),
        }
    }

    #[test]
    fn miss_then_hit_counts() {
        let mut c = OperationCache::new();
        let k = key(42);
        assert!(c.get(k).is_none());
        assert_eq!(c.misses(), 1);
        assert_eq!(c.hits(), 0);

        c.insert(k, TileMap::new(PixelFormat::GPU_WORKING));
        assert!(c.get(k).is_some());
        assert_eq!(c.hits(), 1);

        // A different input hash is a different row: a miss.
        assert!(c.get(key(43)).is_none());
        assert_eq!(c.misses(), 2);
    }
}
