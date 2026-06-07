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

//! Every defined kernel's assembled WGSL module parses and validates
//! under naga — the generated-shader analog of core's build.rs shader
//! validation (handwritten T1 shaders will ALSO get the build.rs
//! treatment when they arrive).

use image_kernels::{abi, all_defined};

#[test]
fn every_kernel_assembles_to_valid_wgsl() {
    let defs = all_defined();
    assert!(!defs.is_empty());
    for def in defs {
        let src = abi::assemble(def);
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("{}: WGSL parse failed: {e}\n{src}", def.id));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}: WGSL validation failed: {e:?}\n{src}", def.id));
    }
}
