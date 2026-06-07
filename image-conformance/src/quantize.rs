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

//! f32 → f16 quantization — applied to the REFERENCE as the final step
//! before diffing against GPU output (spec §5.2/§6.3), plus the f16
//! ULP distance the `ChannelEpsF16` tolerance speaks.

use half::f16;

pub fn f32_to_f16_bits(x: f32) -> u16 {
    f16::from_f32(x).to_bits()
}

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

/// Distance in f16 representation steps (ULPs), order-preserving over
/// the full range including negatives. NaNs compare at maximum
/// distance — a NaN-vs-number divergence must never pass a tolerance.
pub fn f16_ulp_distance(a_bits: u16, b_bits: u16) -> u32 {
    let a = f16::from_bits(a_bits);
    let b = f16::from_bits(b_bits);
    match (a.is_nan(), b.is_nan()) {
        (true, true) => 0,
        (true, false) | (false, true) => u32::MAX,
        (false, false) => {
            // Map the sign-magnitude f16 bit pattern onto a monotone
            // integer line, then take the absolute difference.
            let m = |bits: u16| -> i32 {
                if bits & 0x8000 != 0 {
                    -((bits & 0x7FFF) as i32)
                } else {
                    (bits & 0x7FFF) as i32
                }
            };
            (m(a_bits) - m(b_bits)).unsigned_abs()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_zero_for_equal() {
        let one = f32_to_f16_bits(1.0);
        assert_eq!(f16_ulp_distance(one, one), 0);
    }

    #[test]
    fn ulp_one_for_adjacent() {
        let one = f32_to_f16_bits(1.0);
        assert_eq!(f16_ulp_distance(one, one + 1), 1);
    }

    #[test]
    fn ulp_across_zero() {
        // +0 and -0 are 0 apart; smallest pos vs smallest neg = 2.
        let pz = f32_to_f16_bits(0.0);
        let nz = f32_to_f16_bits(-0.0);
        assert_eq!(f16_ulp_distance(pz, nz), 0);
        assert_eq!(f16_ulp_distance(1, 0x8001), 2);
    }

    #[test]
    fn nan_never_close() {
        let nan = f32_to_f16_bits(f32::NAN);
        let one = f32_to_f16_bits(1.0);
        assert_eq!(f16_ulp_distance(nan, one), u32::MAX);
        assert_eq!(f16_ulp_distance(nan, nan), 0);
    }
}
