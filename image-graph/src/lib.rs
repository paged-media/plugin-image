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

//! Engine B — the persistent buffer graph (spec §8): persistent mutable
//! state, incremental re-evaluation, sub-frame interactive updates.
//!
//! # M2 interface freeze
//!
//! `BufferGraph` holds source/op/sink nodes. A `SinkNode` (the viewport)
//! requests exactly the `(level, coord)` tiles it needs; only invalid
//! tiles recompute, upstream-first. Invalidation is generation-driven
//! (§5.3): every op-node output tile is tagged with the input tile
//! generations + the params hash it was computed from; a param change or
//! buffer write bumps generations, and a cached tile whose recorded
//! input generations no longer match is stale (§8.2).
//!
//! ## What M2 implements
//!
//! - Point and windowed unary/binary op nodes, evaluated at a requested
//!   mip level (mip-aware eval, §8.3 — the graph evaluates AT the
//!   viewing level, requesting source tiles at that level).
//! - Per-node sparse output cache keyed on `TileCoord`; entry carries
//!   `(params_hash, input_generations)`; a miss/staleness recomputes via
//!   the GPU (`image_gpu::execute_tile_once` / `execute_windowed_once`).
//! - `set_params` (the committed-Operation path, §8.5) and `gesture`
//!   (the ephemeral override) both route through the same param update;
//!   damage propagates downstream, `Windowed` kernels inflating it by
//!   radius (the `region_prop` rule in the push direction).
//! - The incremental-correctness invariant (§12.4): `request` after any
//!   damage sequence equals `evaluate_from_scratch` — the non-negotiable
//!   Engine-B gate (its failure mode is stale tiles, not wrong math).
//!
//! ## Deferred (documented)
//!
//! - `WriteBuffer` undo journaling (COW `Arc<Tile>` snapshots, §8.5) —
//!   the types exist (`PersistentBuffer`'s COW `TileMap`); the op-log
//!   integration rides the SDK mutation surface.
//! - The viewport sink's presentation is degraded until the GPU-surface
//!   RFC (BREAKAGE I-01); M2 returns tiles as heap bytes.
//! - Batched multi-tile dispatch per node (image-gpu `DispatchBatch`
//!   exists) — M2 evaluates tile-by-tile for clarity; batching is a
//!   perf follow-up, not a correctness one.

mod cache;
mod eval;
mod graph;

pub use cache::{CachedTile, NodeCache};
pub use eval::EvaluatedTile;
pub use graph::{BufferGraph, GraphError, NodeId, SourceData};

use image_core::Region;

/// Damage region (§8.2). A param change/buffer write produces one;
/// `Windowed` kernels inflate it by radius as it propagates downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Damage {
    pub region: Region,
}
