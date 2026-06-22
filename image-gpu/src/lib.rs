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

//! GPU execution layer (spec §9) — the ONLY execution layer. WGSL
//! compute through wgpu: WebGPU in the browser (via the bundle realm,
//! BREAKAGE I-07), Vulkan/Metal/DX12 natively, software adapter in CI.
//! No CPU kernel path exists anywhere in production.
//!
//! M0 fan-out adds `pool` (rgba16float tile-slot arrays, LRU),
//! `residency` (the three tiers; Tier 2 stubbed on I-03), and batched
//! `dispatch`. Phase 0 lands the device, the kernel pipeline (the four
//! frozen bind groups, §9.2) and the single-tile execute path the
//! conformance harness drives.

mod device;
mod dispatch;
mod execute;
mod pipeline;
mod pool;
mod residency;

// T2 reductions (spec §11): histogram + statistics. NOT KernelDefs —
// they collapse a tile to a table/scalars, so they have no registry
// kernel row and no per-texel WGSL ABI dispatch (see the module docs).
pub mod reduce;

// Selection-mask plumbing (spec §6.1 / §15 M3): the typed `SelectionMask`
// builder that produces the r16float bytes `execute_tile_once`'s `mask`
// argument consumes — the surface the editor's selections lower to.
pub mod selection;

// T3 breadth op (spec §11): a CPU two-pass chamfer/Euclidean-approx
// distance transform over a binary mask tile. Sequential/iterative (the
// GPU jump-flood version is the M3 follow-up — see the module docs), so
// like `reduce` it runs on the CPU over the working tile bytes and owns
// its own state-registry row (`image.kernel.distance-transform`), not a
// per-texel WGSL ABI kernel row.
pub mod distance;

pub use device::GpuContext;
pub use dispatch::{BatchTile, DispatchBatch};
pub use distance::{distance_transform, DistanceParams, MaskChannel};
pub use execute::{execute_tile_once, execute_tile_once_async, execute_windowed_once, TileInput};
pub use pipeline::KernelPipeline;
pub use pool::{PoolSlot, TexturePool};
pub use reduce::{
    auto_enhance, histogram, histogram_rgba8, statistics, AutoEnhance, Histogram,
    RgbaLumaHistogram, Stats, AUTO_LEVELS_CLIP_HIGH, AUTO_LEVELS_CLIP_LOW,
};
pub use residency::{Acquired, ResidencyManager, Tier, HEAP_TILE_BYTES};
pub use selection::SelectionMask;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no compatible GPU adapter (backends tried: {0})")]
    NoAdapter(String),
    #[error("device request failed: {0}")]
    Device(String),
    #[error("readback failed: {0}")]
    Readback(String),
    #[error("kernel `{kernel}` rejected: {detail}")]
    Kernel {
        kernel: &'static str,
        detail: String,
    },
    /// Tier 2 (OPFS scratch) needs the storage capability the plugin SDK
    /// does not yet expose — see BREAKAGE I-03.
    #[error("Tier 2 (OPFS scratch) unavailable: storage capability pending (BREAKAGE I-03)")]
    Tier2Unsupported,
}
