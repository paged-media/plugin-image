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

//! Tiles (spec §5.3). Mip-aware fixed-size tiles; residency expressed
//! by `TileData` over the three tiers (§9.1): GPU texture pool, wasm
//! heap, OPFS scratch.

use std::sync::Arc;

use crate::format::PixelFormat;

/// Tile edge length in pixels (D-1: 256², revisit with M2 benchmarks).
pub const TILE: u32 = 256;

/// Mip level + grid coordinate. `level > 0` stores downsampled content
/// (each level halves resolution); Engine B evaluates at the viewport's
/// mip level (§8.3), Engine A uses levels for shrink-on-load (§7.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileCoord {
    pub level: u8,
    pub x: i32,
    pub y: i32,
}

/// Index into image-gpu's texture pool (Tier 0). A plain newtype so
/// this crate stays engine-agnostic — the pool owns the wgpu texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureSlot(pub u32);

/// Key into the OPFS scratch tier (Tier 2). The store lives behind the
/// (pending) storage capability — BREAKAGE I-03; typed now so the
/// residency ladder is complete from M0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpfsKey(pub u64);

#[derive(Debug, Clone)]
pub enum TileData {
    /// Resident in the GPU texture pool (Tier 0).
    Gpu(TextureSlot),
    /// Resident in the wasm heap (Tier 1) — also the COW/undo tier.
    /// Interleaved rows, `PixelFormat::bytes_per_pixel() * TILE` stride.
    Heap(Arc<[u8]>),
    /// Evicted to OPFS scratch (Tier 2).
    Swapped(OpfsKey),
}

#[derive(Debug, Clone)]
pub struct Tile {
    pub format: PixelFormat,
    pub data: TileData,
    /// Monotone; drives cache invalidation. A committed Operation bumps
    /// generations of touched tiles; downstream caches key on
    /// (node id, params hash, input tile generations) (§5.3).
    pub generation: u64,
}
