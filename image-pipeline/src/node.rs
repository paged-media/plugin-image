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
}

/// A kernel applied to one upstream node (M0 is unary — the binary and
/// generator arities arrive with the T1/T2 kernels). Carries the frozen
/// `KernelDef` and the param block bytes (the uniform upload AND the
/// cache key, per the §6.1 "one definition, three uses" rule).
pub struct ApplyNode {
    pub input: NodeId,
    pub def: &'static KernelDef,
    pub params: Arc<[u8]>,
    /// `ParamsHash::of(params)` — byte identity is param identity (§5.3).
    pub params_hash: ParamsHash,
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
