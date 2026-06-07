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

//! Engine B — the persistent buffer graph (spec §8): incremental
//! re-evaluation over `PersistentBuffer`s, damage propagation,
//! mip-aware evaluation, the viewport sink.
//!
//! **M2 milestone.** M0 carries the node-shape types only, so the §8
//! vocabulary is frozen alongside the §5 core types it builds on
//! (`PersistentBuffer`, tile generations, `Region` damage). The
//! Gesture/Operation lowering (§8.5) arrives with the SDK mutation
//! integration.

use image_core::{PersistentBuffer, Region};
use image_kernels::KernelDef;

pub type NodeId = u32;

/// §8.1 node kinds — type skeleton (the engine lands in M2).
pub enum GraphNode {
    Source(PersistentBuffer),
    Op {
        kernel: &'static KernelDef,
        /// Param block bytes (`Params::as_bytes()`); identity = bytes.
        params: Vec<u8>,
        inputs: Vec<NodeId>,
    },
    /// Viewport sink: owns the visible tile set and requests exactly
    /// those (§8.1). The GPU-surface contract it presents through is
    /// BREAKAGE I-01.
    Sink {
        input: NodeId,
    },
}

/// Damage computation vocabulary (§8.2): a param change or buffer
/// write produces a damage region; `Windowed` kernels inflate damage
/// by their radius downstream (the same `region_prop` rule Engine A
/// uses, applied in the push direction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Damage {
    pub region: Region,
}
