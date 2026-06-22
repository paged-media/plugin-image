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
pub mod shrink;
mod sink;

use std::sync::Arc;

use image_codecs::{ImageSource, ImageTarget};
use image_core::{ParamsHash, PixelFormat, Region, TileMap};
use image_gpu::GpuContext;
use image_kernels::KernelDef;

use crate::cache::OperationCache;
use crate::node::{ApplyInputs, ApplyNode, OpNode, SourceNode};

pub use crate::node::NodeId;
pub use crate::shrink::plan_shrink;
pub use image_codecs::EncodedStats;

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

    /// Add a decode leaf at full resolution (decode shrink 1). The source
    /// is owned by the pipeline (it carries parser state mutated during a
    /// pull).
    pub fn source(&mut self, source: Box<dyn ImageSource + Send>) -> NodeId {
        self.push_source(source, 1)
    }

    /// Add a decode leaf with shrink-on-load planning (spec §7.2). Probes
    /// the source's `native_shrink()` factors, runs [`plan_shrink`]
    /// against `requested_scale` (a downscale fraction in `(0, 1]`), and
    /// wires the chosen native shrink into the decode pull — the decoder
    /// does as much of the downscale as it can natively, leaving the
    /// residual fraction (the second element of the plan) to a downstream
    /// resample kernel. Returns the leaf node plus the planned
    /// `(shrink, residual)` so the caller can size/parameterize that
    /// resample stage.
    ///
    /// With PNG/JPEG advertising only `[1]` today the chosen shrink is
    /// always 1 (the decode pull is full-res and the residual carries the
    /// whole scale); the planner algebra itself is unit-tested over
    /// synthetic native ladders.
    pub fn source_scaled(
        &mut self,
        source: Box<dyn ImageSource + Send>,
        requested_scale: f32,
    ) -> Result<(NodeId, u32, f32), PipelineError> {
        let mut src = source;
        // probe() before native_shrink() — JPEG learns its DCT ladder
        // from the header (PNG/raw answer [1] unconditionally).
        let _ = src.probe()?;
        let (shrink, residual) = plan_shrink(src.native_shrink(), requested_scale);
        let node = self.push_source(src, shrink);
        Ok((node, shrink, residual))
    }

    fn push_source(&mut self, source: Box<dyn ImageSource + Send>, decode_shrink: u32) -> NodeId {
        let op_id = self.next_source_id;
        self.next_source_id += 1;
        self.push(OpNode::Source(SourceNode {
            source: std::sync::Mutex::new(source),
            op_id,
            decode_shrink,
        }))
    }

    /// Apply a unary kernel to `input`. `params` is the kernel's
    /// `#[repr(C)]` param block bytes (the uniform upload AND the cache
    /// key — §6.1); it must match `def.params.size` (checked at
    /// dispatch).
    pub fn apply(
        &mut self,
        input: NodeId,
        def: &'static KernelDef,
        params: impl Into<Arc<[u8]>>,
    ) -> NodeId {
        let params: Arc<[u8]> = params.into();
        let params_hash = ParamsHash::of(&params);
        self.push(OpNode::Apply(ApplyNode {
            inputs: ApplyInputs::Unary(input),
            def,
            params,
            params_hash,
        }))
    }

    /// Apply a BINARY kernel to two upstream nodes — the compose lane
    /// (`def.inputs == 2`, e.g. `math.add`). Both inputs are materialized
    /// over the propagated ROI before dispatch and handed to the kernel
    /// as `TileInput`s in `(a, b)` order; the cache key folds BOTH inputs'
    /// content hashes, so a re-pull is a hit only when neither subtree
    /// changed. `params` is the param block bytes (matches
    /// `def.params.size` at dispatch).
    pub fn apply2(
        &mut self,
        a: NodeId,
        b: NodeId,
        def: &'static KernelDef,
        params: impl Into<Arc<[u8]>>,
    ) -> NodeId {
        let params: Arc<[u8]> = params.into();
        let params_hash = ParamsHash::of(&params);
        self.push(OpNode::Apply(ApplyNode {
            inputs: ApplyInputs::Binary(a, b),
            def,
            params,
            params_hash,
        }))
    }

    /// Materialize `node` over `roi` into a `TileMap` (spec §7.3). The
    /// M0 sink; the structured-readback sink is [`to_encoder`].
    pub fn to_buffer(
        &mut self,
        node: NodeId,
        roi: Region,
        ctx: &GpuContext,
    ) -> Result<TileMap, PipelineError> {
        sink::to_buffer(&self.nodes, &mut self.cache, node, roi, ctx)
    }

    /// Stream `node`'s output over `roi` into an [`ImageTarget`],
    /// strip-by-strip (spec §7.3) — the ONLY structured GPU-readback path
    /// in the system. Each strip is a tile-row band: the full ROI width ×
    /// up to one tile (256px) high. Tiles are pulled through the normal
    /// demand path (op cache included), the rgba16float working pixels are
    /// converted to the requested 8-bit `fmt`, and the strip is handed to
    /// `target.write_strip` in top-to-bottom order. `begin`/`finish`
    /// bracket the run; the `TargetInfo` is built from the ROI dimensions
    /// and `fmt`.
    ///
    /// `fmt` must be a `SampleDepth::U8` format. Alpha handling: formats
    /// WITHOUT an alpha channel (`Gray`, `Rgb`-less — i.e. `ChannelLayout`
    /// whose count drops the working alpha) drop the working alpha
    /// channel; `Rgba`/`GrayA` carry it straight. The U8 conversion is the
    /// straight `clamp(x, 0, 1) * 255` round (no transfer/CMS — that is
    /// the M1 cast lane, BREAKAGE I-02; this mirrors the decode bridge's
    /// verbatim policy on the way back out).
    pub fn to_encoder(
        &mut self,
        node: NodeId,
        roi: Region,
        ctx: &GpuContext,
        target: &mut dyn ImageTarget,
        fmt: PixelFormat,
    ) -> Result<EncodedStats, PipelineError> {
        sink::to_encoder(&self.nodes, &mut self.cache, node, roi, ctx, target, fmt)
    }

    /// [`Self::to_buffer`]'s ASYNC twin — the wasm32/WebGPU lane, where a
    /// blocking readback poll cannot pump the map callback (the browser
    /// event loop delivers it; the await suspends until then). Natively
    /// it runs under pollster and produces byte-for-byte the sync result
    /// (the conformance async-parity test). Same demand path, same cache.
    pub async fn to_buffer_async(
        &mut self,
        node: NodeId,
        roi: Region,
        ctx: &GpuContext,
    ) -> Result<TileMap, PipelineError> {
        sink::to_buffer_async(&self.nodes, &mut self.cache, node, roi, ctx).await
    }

    /// [`Self::to_encoder`]'s ASYNC twin (see [`Self::to_buffer_async`]
    /// for the lane rationale) — the image-js M4 ingest slice's readback:
    /// decode → adjust → RGBA8 out, awaited in the bundle realm.
    pub async fn to_encoder_async(
        &mut self,
        node: NodeId,
        roi: Region,
        ctx: &GpuContext,
        target: &mut dyn ImageTarget,
        fmt: PixelFormat,
    ) -> Result<EncodedStats, PipelineError> {
        sink::to_encoder_async(&self.nodes, &mut self.cache, node, roi, ctx, target, fmt).await
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
