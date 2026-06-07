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

pub mod region_prop;

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("codec: {0}")]
    Codec(#[from] image_codecs::CodecError),
    #[error("gpu: {0}")]
    Gpu(#[from] image_gpu::GpuError),
    #[error("graph: {0}")]
    Graph(String),
}
