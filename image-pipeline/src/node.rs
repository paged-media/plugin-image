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

//! The lazy DAG (spec §7.1): a flat arena of `OpNode`s built by the
//! `Pipeline` fluent API. Nothing here executes — a node records only
//! what to compute (a leaf source, or a kernel applied to an upstream
//! node) and how to key it. `schedule`/`sink` pull on this structure.

use std::sync::Arc;

use image_codecs::ImageSource;
use image_core::{ContentHash, ParamsHash};
use image_kernels::KernelDef;

/// Stable handle into the `Pipeline`'s node arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// A decode leaf: an `ImageSource` plus its op identity. The source is
/// boxed behind a `Mutex` because `read_region` takes `&mut self`
/// (decoders carry parser state) while the node arena is shared `&self`
/// during a pull — single-threaded in M0, so contention never occurs.
pub struct SourceNode {
    pub source: std::sync::Mutex<Box<dyn ImageSource + Send>>,
    /// Stable op id for cache keying — derived once at insert time so a
    /// re-pull of the same leaf hits the same cache row.
    pub op_id: u64,
    /// Native decode shrink to pull at (one of the source's
    /// `native_shrink()` factors). `1` for a plain `source`; set by the
    /// shrink-on-load planner (`source_scaled`, §7.2). The decode ROI is
    /// expressed in POST-shrink coordinates, so the scheduler scales the
    /// requested region down by this factor before `read_region`.
    pub decode_shrink: u32,
}

/// A kernel applied to upstream node(s). M1 supports unary (`apply`) and
/// binary (`apply2`, the compose lane — `def.inputs == 2`); generator
/// arity arrives with T2. Carries the frozen `KernelDef` and the param
/// block bytes (the uniform upload AND the cache key, per the §6.1 "one
/// definition, three uses" rule).
pub struct ApplyNode {
    /// The first (or only) input. `inputs[1]` is `Some` exactly for
    /// binary kernels (`def.inputs == 2`); the scheduler folds BOTH
    /// materialized inputs' content hashes into the cache key.
    pub inputs: ApplyInputs,
    pub def: &'static KernelDef,
    pub params: Arc<[u8]>,
    /// `ParamsHash::of(params)` — byte identity is param identity (§5.3).
    pub params_hash: ParamsHash,
}

/// Upstream wiring of an apply node: one input (unary) or two (binary
/// compose lane). Generator (zero-input) arity is T2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyInputs {
    Unary(NodeId),
    Binary(NodeId, NodeId),
}

impl ApplyInputs {
    /// `(a, Some(b))` for binary, `(a, None)` for unary — the dispatch
    /// order in which the materialized input maps are handed to the
    /// kernel as `TileInput`s.
    pub fn as_pair(&self) -> (NodeId, Option<NodeId>) {
        match self {
            ApplyInputs::Unary(a) => (*a, None),
            ApplyInputs::Binary(a, b) => (*a, Some(*b)),
        }
    }
}

/// One node of the lazy graph.
pub enum OpNode {
    Source(SourceNode),
    Apply(ApplyNode),
}

impl OpNode {
    /// The cache op-id component: the source's stable id, or the
    /// kernel's registry id hashed into the same `u64` space.
    pub fn op_key(&self) -> u64 {
        match self {
            OpNode::Source(s) => s.op_id,
            // ContentHash is just FNV-1a over the bytes — reused here so
            // a kernel's identity lands in the same `u64` space as a
            // source's op id without a second hash primitive.
            OpNode::Apply(a) => ContentHash::of(a.def.id.as_bytes()).0,
        }
    }
}
