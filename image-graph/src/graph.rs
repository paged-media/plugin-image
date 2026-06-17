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

//! The graph store + the mutation surface (`set_params`/`gesture`).
//! Evaluation lives in `eval.rs`.

use std::collections::HashMap;
use std::sync::Arc;

use image_core::{ParamsHash, TileCoord, TILE};
use image_kernels::KernelDef;

use crate::cache::NodeCache;

pub type NodeId = usize;

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("gpu: {0}")]
    Gpu(#[from] image_gpu::GpuError),
    #[error("graph: {0}")]
    Graph(String),
}

/// A source node's pixel data: per-level sparse tiles (rgba16float
/// bytes). Mip-aware — the graph evaluates at a requested level and
/// pulls source tiles at that level (§8.3). Sources start at level 0;
/// higher levels are caller-provided (the pyramid; M2 tests build the
/// levels they request).
#[derive(Debug, Default, Clone)]
pub struct SourceData {
    /// (level, coord) → tile bytes + generation.
    tiles: HashMap<TileCoord, (Arc<[u8]>, u64)>,
}

impl SourceData {
    pub fn new() -> Self {
        SourceData::default()
    }

    pub fn set_tile(&mut self, coord: TileCoord, bytes: impl Into<Arc<[u8]>>, generation: u64) {
        self.tiles.insert(coord, (bytes.into(), generation));
    }

    pub fn tile(&self, coord: TileCoord) -> Option<(&Arc<[u8]>, u64)> {
        self.tiles.get(&coord).map(|(b, g)| (b, *g))
    }
}

pub(crate) enum Node {
    Source {
        data: SourceData,
    },
    Op {
        kernel: &'static KernelDef,
        params: Arc<[u8]>,
        params_hash: ParamsHash,
        inputs: Vec<NodeId>,
        cache: NodeCache,
    },
}

pub struct BufferGraph {
    pub(crate) nodes: Vec<Node>,
}

impl Default for BufferGraph {
    fn default() -> Self {
        BufferGraph::new()
    }
}

impl BufferGraph {
    pub fn new() -> Self {
        BufferGraph { nodes: Vec::new() }
    }

    pub fn add_source(&mut self, data: SourceData) -> NodeId {
        self.push(Node::Source { data })
    }

    /// An op node. `params` is the kernel's `#[repr(C)]` block bytes
    /// (identity = bytes, §6.1); `inputs.len()` must match
    /// `kernel.inputs`.
    pub fn add_op(
        &mut self,
        kernel: &'static KernelDef,
        params: impl Into<Arc<[u8]>>,
        inputs: Vec<NodeId>,
    ) -> NodeId {
        let params: Arc<[u8]> = params.into();
        let params_hash = ParamsHash::of(&params);
        self.push(Node::Op {
            kernel,
            params,
            params_hash,
            inputs,
            cache: NodeCache::new(),
        })
    }

    /// Mutate an op node's params — the committed-Operation path (§8.5)
    /// AND the ephemeral-gesture path (the cache makes a re-set of the
    /// same bytes a no-op; a different value invalidates by provenance).
    /// Returns false if `node` is not an op.
    pub fn set_params(&mut self, node: NodeId, params: impl Into<Arc<[u8]>>) -> bool {
        match self.nodes.get_mut(node) {
            Some(Node::Op {
                params: p,
                params_hash,
                cache,
                ..
            }) => {
                let new: Arc<[u8]> = params.into();
                let new_hash = ParamsHash::of(&new);
                if new_hash != *params_hash {
                    // Drop this node's cached tiles: their params_hash no
                    // longer matches. Downstream nodes invalidate lazily
                    // via the recomputed tiles' bumped generations.
                    *cache = NodeCache::new();
                }
                *p = new;
                *params_hash = new_hash;
                true
            }
            _ => false,
        }
    }

    /// Ephemeral gesture override (§8.5) — identical mechanics to
    /// `set_params` in M2 (no separate scratch tier yet); named for the
    /// call-site intent and the future divergence.
    pub fn gesture(&mut self, node: NodeId, params: impl Into<Arc<[u8]>>) -> bool {
        self.set_params(node, params)
    }

    /// Overwrite a source tile (the `WriteBuffer` Operation, §8.5),
    /// bumping its generation so downstream caches see the change.
    /// Returns false if `node` is not a source.
    pub fn write_source_tile(
        &mut self,
        node: NodeId,
        coord: TileCoord,
        bytes: impl Into<Arc<[u8]>>,
        generation: u64,
    ) -> bool {
        match self.nodes.get_mut(node) {
            Some(Node::Source { data }) => {
                data.set_tile(coord, bytes, generation);
                true
            }
            _ => false,
        }
    }

    /// Read a single SOURCE tile `(level, coord)` without a GPU context —
    /// the passthrough mip-window path (a source read does no kernel
    /// dispatch, so it needs no device). Returns the tile's rgba16float
    /// bytes (a freshly-allocated transparent-black tile for an
    /// unallocated coord, per the sparse-canvas rule §5.3), or `None` if
    /// `node` is not a source. The op-bearing evaluation stays on
    /// [`Self::request`], which takes the GPU context the kernels need.
    pub fn read_source_tile(&self, node: NodeId, coord: TileCoord) -> Option<Arc<[u8]>> {
        match self.nodes.get(node)? {
            Node::Source { data } => Some(match data.tile(coord) {
                Some((b, _g)) => Arc::clone(b),
                None => {
                    const TILE_BYTES: usize = (TILE * TILE * 8) as usize; // rgba16float
                    Arc::from(vec![0u8; TILE_BYTES].into_boxed_slice())
                }
            }),
            _ => None,
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of cached output tiles at an op node (0 for sources) —
    /// the test hook the incremental-correctness suite uses to assert
    /// the cache is actually populated/pruned (§8.2).
    pub fn cache_len(&self, node: NodeId) -> usize {
        match &self.nodes[node] {
            Node::Op { cache, .. } => cache.len(),
            _ => 0,
        }
    }

    fn push(&mut self, n: Node) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(n);
        id
    }
}
