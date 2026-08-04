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

//! The graph store + the mutation surface (`set_params`/`gesture`).
//! Evaluation lives in `eval.rs`.

use std::collections::HashMap;
use std::sync::Arc;

use image_core::{ParamsHash, Region, TileCoord, TILE};
use image_kernels::KernelDef;

use crate::cache::NodeCache;
use crate::journal::{RecordOutcome, TileJournal, TileSource, TileStore};

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
    /// The generation a journal RESTORE stamps on the tile it puts back.
    /// A restore must look like a NEW write to every downstream cache
    /// (the pixels changed), so it takes a fresh generation rather than
    /// the one it had before — otherwise undo would serve stale
    /// derived tiles whose provenance still "matched" (§8.2).
    next_generation: u64,
}

impl SourceData {
    pub fn new() -> Self {
        SourceData::default()
    }

    pub fn set_tile(&mut self, coord: TileCoord, bytes: impl Into<Arc<[u8]>>, generation: u64) {
        self.next_generation = self.next_generation.max(generation + 1);
        self.tiles.insert(coord, (bytes.into(), generation));
    }

    pub fn tile(&self, coord: TileCoord) -> Option<(&Arc<[u8]>, u64)> {
        self.tiles.get(&coord).map(|(b, g)| (b, *g))
    }
}

/// The §8.5 `WriteBuffer` COW seam: a source node's sparse tile map IS
/// a journalable store — snapshotting a tile is an `Arc` clone.
impl TileSource for SourceData {
    fn read_tile(&self, coord: TileCoord) -> Option<Arc<[u8]>> {
        self.tiles.get(&coord).map(|(b, _)| Arc::clone(b))
    }
}

impl TileStore for SourceData {
    fn write_tile(&mut self, coord: TileCoord, bytes: Option<Arc<[u8]>>) {
        match bytes {
            Some(b) => {
                let g = self.next_generation;
                self.next_generation += 1;
                self.tiles.insert(coord, (b, g));
            }
            // The tile was a HOLE before the edit; restoring the hole is
            // what makes undo exact on a sparse canvas.
            None => {
                self.tiles.remove(&coord);
            }
        }
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

    /// The JOURNALED `WriteBuffer` Operation (§8.5): snapshot the tiles
    /// covering `damage` into `journal` (COW — an `Arc` clone per tile),
    /// then hand the source's tile map to `write` so it can put the new
    /// pixels down. The snapshot happens FIRST and unconditionally, so a
    /// write that touches fewer tiles than it claimed still undoes
    /// exactly; a write that touches MORE is a caller bug the journal
    /// cannot see, which is why `damage` is the caller's honest damage
    /// region and not a guess.
    ///
    /// Answers `None` when `node` is not a source; otherwise the
    /// journal's verdict (including [`RecordOutcome::TooLarge`], where
    /// the write still happens but the history is gone — see the journal
    /// module docs).
    pub fn write_source_journaled(
        &mut self,
        node: NodeId,
        journal: &mut TileJournal,
        label: impl Into<String>,
        damage: Region,
        write: impl FnOnce(&mut SourceData),
    ) -> Option<RecordOutcome> {
        let Some(Node::Source { data }) = self.nodes.get_mut(node) else {
            return None;
        };
        // The node id IS the scope: one journal can serve several
        // sources, and undo must restore into the one that was written.
        let outcome = journal.record(label, node as u64, &*data, damage);
        write(data);
        Some(outcome)
    }

    /// Undo the newest journaled write on `node`, restoring its tiles
    /// (and bumping their generations, so every downstream cache sees
    /// the change). Returns the reverted edit's label.
    pub fn undo_source(&mut self, node: NodeId, journal: &mut TileJournal) -> Option<String> {
        match self.nodes.get_mut(node) {
            Some(Node::Source { data }) => journal.undo(data),
            _ => None,
        }
    }

    /// Replay the newest undone write on `node`.
    pub fn redo_source(&mut self, node: NodeId, journal: &mut TileJournal) -> Option<String> {
        match self.nodes.get_mut(node) {
            Some(Node::Source { data }) => journal.redo(data),
            _ => None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use image_core::TILE;

    const TILE_BYTES: usize = (TILE * TILE * 8) as usize; // rgba16float

    fn tile(fill: u8) -> Vec<u8> {
        vec![fill; TILE_BYTES]
    }

    #[test]
    fn image_editor_undo_a_journaled_source_write_restores_the_tile_and_bumps_its_generation() {
        let mut g = BufferGraph::new();
        let mut data = SourceData::new();
        let coord = TileCoord {
            level: 0,
            x: 0,
            y: 0,
        };
        data.set_tile(coord, tile(1), 7);
        let src = g.add_source(data);

        let mut journal = TileJournal::new();
        let out = g.write_source_journaled(
            src,
            &mut journal,
            "paint",
            Region::new(0, 0, TILE, TILE),
            |d| d.set_tile(coord, tile(2), 8),
        );
        assert!(out.expect("a source node").is_recorded());
        assert_eq!(g.read_source_tile(src, coord).unwrap()[0], 2);

        assert_eq!(g.undo_source(src, &mut journal).as_deref(), Some("paint"));
        let restored = g.read_source_tile(src, coord).unwrap();
        assert_eq!(restored[0], 1, "the pre-edit tile is back");
        // …and it reads as a NEW generation, so no downstream cache can
        // serve a tile derived from the version we just reverted.
        let (_, gen) = match &g.nodes[src] {
            Node::Source { data } => data.tile(coord).expect("tile"),
            _ => unreachable!(),
        };
        assert!(
            gen > 8,
            "a restore is a write: generation {gen} must be new"
        );

        assert_eq!(g.redo_source(src, &mut journal).as_deref(), Some("paint"));
        assert_eq!(g.read_source_tile(src, coord).unwrap()[0], 2);
    }

    #[test]
    fn image_editor_undo_a_journaled_write_into_a_hole_restores_the_hole() {
        // A sparse canvas: the tile did not exist before the write, so
        // undo must REMOVE it again (not leave transparent-black bytes
        // that would defeat the sparse-canvas rule).
        let mut g = BufferGraph::new();
        let src = g.add_source(SourceData::new());
        let coord = TileCoord {
            level: 0,
            x: 3,
            y: 2,
        };
        let mut journal = TileJournal::new();
        let damage = Region::new(3 * TILE as i32, 2 * TILE as i32, TILE, TILE);
        let out = g
            .write_source_journaled(src, &mut journal, "paint", damage, |d| {
                d.set_tile(coord, tile(9), 1)
            })
            .expect("a source node");
        assert!(out.is_recorded(), "a hole is still a state worth recording");

        g.undo_source(src, &mut journal);
        let held = match &g.nodes[src] {
            Node::Source { data } => data.tile(coord).is_some(),
            _ => unreachable!(),
        };
        assert!(!held, "the hole is a hole again");
        // The passthrough read still answers transparent black (§5.3).
        assert_eq!(g.read_source_tile(src, coord).unwrap().len(), TILE_BYTES);
    }

    #[test]
    fn image_editor_undo_journaling_an_op_node_is_refused() {
        let mut g = BufferGraph::new();
        let src = g.add_source(SourceData::new());
        let op = g.add_op(
            &image_kernels::families::linear::MATH_INVERT,
            image_kernels::families::linear::MathInvertParams::new()
                .as_bytes()
                .to_vec(),
            vec![src],
        );
        let mut journal = TileJournal::new();
        assert!(g
            .write_source_journaled(op, &mut journal, "x", Region::new(0, 0, 1, 1), |_| {})
            .is_none());
        assert!(g.undo_source(op, &mut journal).is_none());
    }
}
