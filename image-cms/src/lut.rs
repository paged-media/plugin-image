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

//! GPU LUT baking — the apply-on-GPU half of §10.1. The `cms.apply`
//! kernel (T1/M1) samples the 3D lattice with trilinear interpolation;
//! LUT-vs-exact precision is conformance-tested per profile class with
//! declared ΔE tolerance.

use crate::ExactTransform;

/// A baked transform: `dim`³ RGBA8 lattice (alpha unused, kept for
/// texel alignment). Shaper curves (1D pre/post LUTs) join in M1 with
/// the moxcms lane — qcms-baked lattices fold the curves in.
#[derive(Debug, Clone)]
pub struct GpuLut {
    pub dim: u32,
    /// dim³ × 4 bytes, x-major (r fastest).
    pub lattice: Vec<u8>,
}

pub(crate) fn bake_from_exact(t: &dyn ExactTransform, dim: u32) -> GpuLut {
    assert!(dim >= 2, "LUT lattice needs at least 2 points per axis");
    let n = dim as usize;
    let mut lattice = Vec::with_capacity(n * n * n * 4);
    // Sample the exact transform over the lattice; one row at a time
    // keeps the working set tiny.
    let mut row: Vec<u8> = vec![0; n * 4];
    for b in 0..n {
        for g in 0..n {
            for (r, px) in row.chunks_exact_mut(4).enumerate() {
                px[0] = lattice_coord(r, n);
                px[1] = lattice_coord(g, n);
                px[2] = lattice_coord(b, n);
                px[3] = 255;
            }
            t.apply_rgba8(&mut row);
            lattice.extend_from_slice(&row);
        }
    }
    GpuLut { dim, lattice }
}

fn lattice_coord(i: usize, n: usize) -> u8 {
    ((i * 255) / (n - 1)) as u8
}
