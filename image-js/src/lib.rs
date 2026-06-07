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

//! The wasm-bindgen surface consumed by `glue/` — the bundle's compute
//! artifact (manifest `capabilities.wasm[0]`).
//!
//! ARCHITECTURE NOTE (BREAKAGE I-07): a module loaded via
//! `loadBundleWasm` has no ambient authority — no `navigator.gpu`. So
//! this crate is loaded through its wasm-bindgen JS glue in the bundle
//! realm (the `@paged-media/sdk` pattern), where WebGPU IS reachable;
//! the engines' GPU device lives behind this surface. Pixels never
//! cross into plugin JS (spec §2.1.3); the surface speaks handles,
//! regions, and encoded bytes.
//!
//! M0 keeps the surface to identity/version probes — the engine
//! methods land milestone by milestone behind it. The release-build
//! guarantee proven by CI: NO reference code (image-conformance /
//! `image-kernels` feature `reference`) is reachable from this crate
//! (cargo-tree guard, spec §4 dep rule 2).

// Native builds of this crate exist so `cargo test --workspace` covers
// it; the wasm-bindgen exports are wasm32-only.

/// The frozen kernel ABI version this artifact was built against.
pub fn abi_version() -> u32 {
    image_kernels::ABI_VERSION
}

/// Registered kernel count (dispatch-table probe).
pub fn kernel_count() -> usize {
    image_kernels::registry().len()
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(start)]
    pub fn init() {
        console_error_panic_hook::set_once();
    }

    #[wasm_bindgen]
    pub fn abi_version() -> u32 {
        super::abi_version()
    }

    #[wasm_bindgen]
    pub fn kernel_count() -> usize {
        super::kernel_count()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn abi_and_registry_reachable() {
        assert_eq!(super::abi_version(), 1);
        assert!(super::kernel_count() >= 2);
    }
}
