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

//! Engine A — the demand-driven streaming pipeline (spec §7): a lazy
//! DAG of op nodes; nothing executes until a sink pulls; ROIs propagate
//! upstream expanded per `KernelClass`; leaves answer from codec decode
//! streams; every kernel stage is a GPU dispatch.
//!
//! M0 ships the skeleton: `to_buffer` sink, region propagation, the
//! operation cache, and a SINGLE-THREADED decode→upload→dispatch→
//! readback bridge. The CPU-worker pool (wasm-bindgen-rayon over SAB)
//! and the decode/dispatch overlap scheduler are deliberately M1 —
//! gated on the worker-capability RFC (BREAKAGE I-02). Shrink-on-load
//! planning (§7.2) and the `to_pyramid` / `to_encoder` sinks are M1.
//!
//! Module skeletons land with the M0 fan-out: `node` (OpNode DAG),
//! `region_prop` (ROI propagation), `cache` (op cache keyed on
//! (op id, ParamsHash, input ContentHash)), `schedule` (the bridge),
//! `sink` (`to_buffer`).

pub mod cache;
pub mod node;
pub mod region_prop;
mod schedule;
mod sink;

use std::sync::Arc;

use image_codecs::ImageSource;
use image_core::{ParamsHash, Region, TileMap};
use image_gpu::GpuContext;
use image_kernels::KernelDef;

use crate::cache::OperationCache;
use crate::node::{ApplyNode, OpNode, SourceNode};

pub use crate::node::NodeId;

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("codec: {0}")]
    Codec(#[from] image_codecs::CodecError),
    #[error("gpu: {0}")]
    Gpu(#[from] image_gpu::GpuError),
    #[error("graph: {0}")]
    Graph(String),
}

/// Engine A — the lazy op-node DAG plus its operation cache (spec §7.1).
/// Nodes are appended by `source`/`apply` and never executed until a
/// sink (`to_buffer`) pulls. The cache memoizes per-node materialized
/// `TileMap`s so a re-pull of an unchanged subtree is served without
/// recompute.
#[derive(Default)]
pub struct Pipeline {
    nodes: Vec<OpNode>,
    cache: OperationCache,
    /// Monotone op-id source for leaves — keeps two `RawSource`s of
    /// identical bytes from colliding in the cache (their decoded
    /// identity is the leaf id, not the pixels). Mutation-driven content
    /// hashing arrives with Engine B.
    next_source_id: u64,
}

impl Pipeline {
    pub fn new() -> Self {
        Pipeline::default()
    }

    /// Add a decode leaf. The source is owned by the pipeline (it
    /// carries parser state mutated during a pull).
    pub fn source(&mut self, source: Box<dyn ImageSource + Send>) -> NodeId {
        let op_id = self.next_source_id;
        self.next_source_id += 1;
        self.push(OpNode::Source(SourceNode {
            source: std::sync::Mutex::new(source),
            op_id,
        }))
    }

    /// Apply a kernel to `input`. `params` is the kernel's `#[repr(C)]`
    /// param block bytes (the uniform upload AND the cache key — §6.1);
    /// it must match `def.params.size` (checked at dispatch).
    pub fn apply(
        &mut self,
        input: NodeId,
        def: &'static KernelDef,
        params: impl Into<Arc<[u8]>>,
    ) -> NodeId {
        let params: Arc<[u8]> = params.into();
        let params_hash = ParamsHash::of(&params);
        self.push(OpNode::Apply(ApplyNode {
            input,
            def,
            params,
            params_hash,
        }))
    }

    /// Materialize `node` over `roi` into a `TileMap` (spec §7.3). The
    /// single sink M0 ships; `to_pyramid`/`to_encoder` are M1.
    pub fn to_buffer(
        &mut self,
        node: NodeId,
        roi: Region,
        ctx: &GpuContext,
    ) -> Result<TileMap, PipelineError> {
        sink::to_buffer(&self.nodes, &mut self.cache, node, roi, ctx)
    }

    /// Cache hits since construction — the test hook proving a re-pull is
    /// memoized, not recomputed.
    pub fn cache_hits(&self) -> u64 {
        self.cache.hits()
    }

    pub fn cache_misses(&self) -> u64 {
        self.cache.misses()
    }

    fn push(&mut self, node: OpNode) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }
}
