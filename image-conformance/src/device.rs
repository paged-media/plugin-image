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

//! The shared test GPU device. Local runs hit the machine adapter
//! (Metal on macOS); CI selects the software adapter via
//! `WGPU_BACKEND` / `WGPU_FALLBACK=1` (spec §9.3). Tests that need a
//! device call [`test_device`] and SKIP when no adapter exists — unless
//! `REQUIRE_GPU=1`, which the GPU lanes set, and which turns the skip
//! into a failure (`image_gpu::test_support`).

use image_gpu::GpuContext;

/// The process-wide test device, or `None` when the environment has no
/// usable adapter and does not require one.
pub fn test_device() -> Option<&'static GpuContext> {
    image_gpu::test_support::device_or_skip("conformance")
}
