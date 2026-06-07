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

pub use device::GpuContext;
pub use dispatch::{BatchTile, DispatchBatch};
pub use execute::{execute_tile_once, execute_windowed_once, TileInput};
pub use pipeline::KernelPipeline;
pub use pool::{PoolSlot, TexturePool};
pub use residency::{ResidencyManager, Tier, HEAP_TILE_BYTES};

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
