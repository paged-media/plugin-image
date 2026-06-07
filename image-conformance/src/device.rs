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

//! The shared test GPU device. Local runs hit the machine adapter
//! (Metal on macOS); CI selects the pinned software adapter via
//! `WGPU_BACKEND` / `WGPU_FALLBACK` (spec §9.3). Tests that need a
//! device call [`test_device`] and SKIP (not fail) when no adapter
//! exists — the merge gate's GPU lane runs where one is guaranteed.

use std::sync::OnceLock;

use image_gpu::GpuContext;

static DEVICE: OnceLock<Option<GpuContext>> = OnceLock::new();

/// The process-wide test device, or `None` when the environment has no
/// usable adapter.
pub fn test_device() -> Option<&'static GpuContext> {
    DEVICE
        .get_or_init(|| match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => {
                eprintln!(
                    "conformance GPU: {} ({:?})",
                    ctx.adapter_info.name, ctx.adapter_info.backend
                );
                Some(ctx)
            }
            Err(e) => {
                eprintln!("conformance GPU unavailable: {e} — GPU parity tests will skip");
                None
            }
        })
        .as_ref()
}
