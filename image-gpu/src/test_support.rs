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

//! The ONE test device every crate's GPU tests share (feature
//! `test-support`, enabled only through dev-dependencies, so it never
//! reaches the shipped wasm).
//!
//! A machine without an adapter SKIPS the device tests, which is right
//! for a laptop without a GPU and wrong for a lane that exists to run
//! them: there, a skip is a pass that proved nothing. `REQUIRE_GPU=1`
//! turns the skip into a panic, so a lane that sets it cannot go green
//! without executing every GPU test. There used to be four copies of
//! this helper (conformance, layers, stroke, the ingest tests), each
//! skipping on its own, which is why none of them could be made strict.

use std::sync::OnceLock;

use crate::GpuContext;

static DEVICE: OnceLock<Option<GpuContext>> = OnceLock::new();

/// True when the environment demands a device (`REQUIRE_GPU=1`).
pub fn gpu_required() -> bool {
    std::env::var("REQUIRE_GPU").is_ok_and(|v| v == "1")
}

/// The process-wide test device, or `None` when the machine has no
/// usable adapter. `who` names the caller in the one log line.
///
/// # Panics
///
/// When no adapter is available and `REQUIRE_GPU=1` is set.
pub fn device_or_skip(who: &str) -> Option<&'static GpuContext> {
    let ctx = DEVICE
        .get_or_init(|| match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => {
                eprintln!(
                    "test GPU: {} ({:?})",
                    ctx.adapter_info.name, ctx.adapter_info.backend
                );
                Some(ctx)
            }
            Err(e) => {
                eprintln!("test GPU unavailable: {e}");
                None
            }
        })
        .as_ref();
    if ctx.is_none() {
        assert!(
            !gpu_required(),
            "{who}: REQUIRE_GPU=1 but no GPU adapter is available — this lane exists to \
             run the device tests, so it fails instead of skipping them"
        );
        eprintln!("{who}: no GPU adapter — device test skipped");
    }
    ctx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GPU lanes run this with `--nocapture` first, so their log
    /// names the adapter every later device test ran on. (The
    /// `__feat__` suffix is the feature registry's join convention.)
    #[test]
    #[allow(non_snake_case)]
    fn the_test_device_names_its_adapter__feat__image_conformance_harness() {
        if let Some(ctx) = device_or_skip("adapter") {
            let info = &ctx.adapter_info;
            println!(
                "test GPU adapter: {} | backend {:?} | type {:?} | driver {} {}",
                info.name, info.backend, info.device_type, info.driver, info.driver_info
            );
        }
    }
}
