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

//! Demand-driven ROI propagation (spec §7.1): given a requested output
//! region at a node, the required input region per `KernelClass`.
//! Frozen with the phase-0 interfaces because both engines consume it.

use image_core::Region;
use image_kernels::KernelClass;

/// The input ROI a node must materialize to produce `out`.
pub fn required_input_roi(class: KernelClass, out: Region) -> Region {
    match class {
        KernelClass::Point => out,
        KernelClass::Windowed { radius: (rx, ry) } => out.expand_by(rx, ry),
        // Resample footprints depend on the scale factor carried in the
        // node's params; the planner computes them (M1, with the T1
        // resample kernels). Until then resample nodes are unreachable
        // (not in the registry).
        KernelClass::Resample { .. } => out,
        // Reductions consume their full input by definition; the engine
        // handles them as whole-region pulls (T2).
        KernelClass::Reduction(_) => out,
        KernelClass::Generator => Region::new(0, 0, 0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_is_identity() {
        let r = Region::new(10, 10, 64, 64);
        assert_eq!(required_input_roi(KernelClass::Point, r), r);
    }

    #[test]
    fn windowed_inflates_by_radius() {
        let r = Region::new(0, 0, 64, 64);
        assert_eq!(
            required_input_roi(KernelClass::Windowed { radius: (3, 1) }, r),
            Region::new(-3, -1, 70, 66)
        );
    }
}
