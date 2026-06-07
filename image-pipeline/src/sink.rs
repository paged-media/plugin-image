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

//! Sinks (spec §7.3). M0 ships `to_buffer`: materialize a region into a
//! `TileMap` (feeds Engine B and the SDK texture-provider contract).
//! `to_pyramid` and `to_encoder` are M1.
//!
//! The sink is the demand origin: it partitions the requested ROI into
//! work units (`Region::tiles_at`) and pulls the graph for them. Here
//! the partition is implicit — `materialize_node` tiles the ROI itself
//! — so `to_buffer` is a thin entry point that delegates to the bridge
//! and returns the assembled per-node `TileMap`.

use image_core::{Region, TileMap};
use image_gpu::GpuContext;

use crate::cache::OperationCache;
use crate::node::OpNode;
use crate::schedule::materialize_node;
use crate::{NodeId, PipelineError};

/// Materialize `node` over `roi` into a `TileMap` of rgba16float heap
/// tiles. Empty ROIs yield an empty map (no work, no tiles).
pub(crate) fn to_buffer(
    nodes: &[OpNode],
    cache: &mut OperationCache,
    node: NodeId,
    roi: Region,
    ctx: &GpuContext,
) -> Result<TileMap, PipelineError> {
    if roi.is_empty() {
        return Ok(TileMap::new(image_core::PixelFormat::GPU_WORKING));
    }
    let (map, _hash) = materialize_node(nodes, cache, node, roi, ctx)?;
    Ok(map)
}
