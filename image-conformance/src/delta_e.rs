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

//! Perceptual distance for the oracle comparisons: CIEDE2000 between
//! sRGB-encoded colours (the pixel model holds encoded values, ADR 455).
//!
//! Sources: IEC 61966-2-1 (the sRGB transfer function and the D65
//! RGB→XYZ matrix), CIE 15 (XYZ→CIELAB), and G. Sharma, W. Wu,
//! E. N. Dalal, "The CIEDE2000 color-difference formula: implementation
//! notes, supplementary test data, and mathematical observations",
//! Color Research & Application 30(1), 2005 — whose test pairs the unit
//! tests below reproduce to four decimals.
//!
//! A per-channel byte difference says how far apart two renders are in
//! code values; ΔE00 says how far apart they LOOK, which is what a
//! comparison against Photoshop's own composite needs (a 3-code
//! difference in a dark blue and in a light yellow are not the same
//! mistake).

/// A CIELAB colour (D65 reference white).
pub type Lab = [f64; 3];

fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB-encoded `[0,1]` RGB → CIELAB (D65).
pub fn srgb_to_lab(rgb: [f64; 3]) -> Lab {
    let [r, g, b] = rgb.map(|c| srgb_to_linear(c.clamp(0.0, 1.0)));
    let x = 0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175_0 * b;
    let z = 0.019_333_9 * r + 0.119_192_0 * g + 0.950_304_1 * b;
    // D65 white, Y = 1.
    let (xn, yn, zn) = (0.950_47, 1.0, 1.088_83);
    let f = |t: f64| {
        const E: f64 = 216.0 / 24389.0;
        const K: f64 = 24389.0 / 27.0;
        if t > E {
            t.cbrt()
        } else {
            (K * t + 16.0) / 116.0
        }
    };
    let (fx, fy, fz) = (f(x / xn), f(y / yn), f(z / zn));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// 8-bit sRGB → CIELAB (D65).
pub fn srgb8_to_lab(rgb: [u8; 3]) -> Lab {
    srgb_to_lab(rgb.map(|c| c as f64 / 255.0))
}

/// CIEDE2000 colour difference (kL = kC = kH = 1).
pub fn ciede2000(a: Lab, b: Lab) -> f64 {
    use std::f64::consts::PI;
    let deg = |r: f64| r * 180.0 / PI;
    let rad = |d: f64| d * PI / 180.0;

    let (l1, a1, b1) = (a[0], a[1], a[2]);
    let (l2, a2, b2) = (b[0], b[1], b[2]);
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_bar7 = ((c1 + c2) / 2.0).powi(7);
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + 25f64.powi(7))).sqrt());
    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;
    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();
    let hue = |bb: f64, ap: f64| {
        if bb == 0.0 && ap == 0.0 {
            0.0
        } else {
            let h = deg(bb.atan2(ap));
            if h < 0.0 {
                h + 360.0
            } else {
                h
            }
        }
    };
    let h1p = hue(b1, a1p);
    let h2p = hue(b2, a2p);

    let dl = l2 - l1;
    let dc = c2p - c1p;
    let dh = if c1p * c2p == 0.0 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p - h1p > 180.0 {
        h2p - h1p - 360.0
    } else {
        h2p - h1p + 360.0
    };
    let dhh = 2.0 * (c1p * c2p).sqrt() * rad(dh / 2.0).sin();

    let l_bar = (l1 + l2) / 2.0;
    let c_bar_p = (c1p + c2p) / 2.0;
    let h_bar_p = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };
    let t = 1.0 - 0.17 * rad(h_bar_p - 30.0).cos()
        + 0.24 * rad(2.0 * h_bar_p).cos()
        + 0.32 * rad(3.0 * h_bar_p + 6.0).cos()
        - 0.20 * rad(4.0 * h_bar_p - 63.0).cos();
    let d_theta = 30.0 * (-((h_bar_p - 275.0) / 25.0).powi(2)).exp();
    let c_bar_p7 = c_bar_p.powi(7);
    let rc = 2.0 * (c_bar_p7 / (c_bar_p7 + 25f64.powi(7))).sqrt();
    let l50 = (l_bar - 50.0).powi(2);
    let sl = 1.0 + 0.015 * l50 / (20.0 + l50).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;
    let rt = -rad(2.0 * d_theta).sin() * rc;

    let (tl, tc, th) = (dl / sl, dc / sc, dhh / sh);
    (tl * tl + tc * tc + th * th + rt * tc * th).sqrt()
}

/// Summary of ΔE00 over two images.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeltaEStats {
    pub mean: f64,
    pub p95: f64,
    pub max: f64,
    /// Pixels compared.
    pub count: usize,
}

/// ΔE00 between two straight-alpha RGBA8 buffers of the same size, each
/// composited over `background` first (so a transparent pixel compares
/// as the background it shows, the way a flattened file stores it).
///
/// # Panics
///
/// When the buffers differ in length or are not whole RGBA8 pixels.
pub fn delta_e_rgba8(a: &[u8], b: &[u8], background: [u8; 3]) -> DeltaEStats {
    assert_eq!(a.len(), b.len(), "ΔE needs two images of the same size");
    assert_eq!(a.len() % 4, 0, "RGBA8 buffers");
    let over = |p: &[u8]| -> [u8; 3] {
        let al = p[3] as u32;
        std::array::from_fn(|c| {
            ((p[c] as u32 * al + background[c] as u32 * (255 - al) + 127) / 255) as u8
        })
    };
    let mut d: Vec<f64> = a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .map(|(pa, pb)| ciede2000(srgb8_to_lab(over(pa)), srgb8_to_lab(over(pb))))
        .collect();
    let count = d.len();
    if count == 0 {
        return DeltaEStats {
            mean: 0.0,
            p95: 0.0,
            max: 0.0,
            count: 0,
        };
    }
    let mean = d.iter().sum::<f64>() / count as f64;
    d.sort_by(|x, y| x.total_cmp(y));
    let p95 = d[((count as f64 * 0.95).ceil() as usize).clamp(1, count) - 1];
    DeltaEStats {
        mean,
        p95,
        max: d[count - 1],
        count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sharma, Wu & Dalal (2005), Table 1: all 34 pairs, ΔE00 to 4 dp.
    const SHARMA: [(Lab, Lab, f64); 34] = [
        ([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485], 2.0425),
        ([50.0, 3.1571, -77.2803], [50.0, 0.0, -82.7485], 2.8615),
        ([50.0, 2.8361, -74.0200], [50.0, 0.0, -82.7485], 3.4412),
        ([50.0, -1.3802, -84.2814], [50.0, 0.0, -82.7485], 1.0000),
        ([50.0, -1.1848, -84.8006], [50.0, 0.0, -82.7485], 1.0000),
        ([50.0, -0.9009, -85.5211], [50.0, 0.0, -82.7485], 1.0000),
        ([50.0, 0.0, 0.0], [50.0, -1.0, 2.0], 2.3669),
        ([50.0, -1.0, 2.0], [50.0, 0.0, 0.0], 2.3669),
        ([50.0, 2.4900, -0.0010], [50.0, -2.4900, 0.0009], 7.1792),
        ([50.0, 2.4900, -0.0010], [50.0, -2.4900, 0.0010], 7.1792),
        ([50.0, 2.4900, -0.0010], [50.0, -2.4900, 0.0011], 7.2195),
        ([50.0, 2.4900, -0.0010], [50.0, -2.4900, 0.0012], 7.2195),
        ([50.0, -0.0010, 2.4900], [50.0, 0.0009, -2.4900], 4.8045),
        ([50.0, -0.0010, 2.4900], [50.0, 0.0010, -2.4900], 4.8045),
        ([50.0, -0.0010, 2.4900], [50.0, 0.0011, -2.4900], 4.7461),
        ([50.0, 2.5, 0.0], [50.0, 0.0, -2.5], 4.3065),
        ([50.0, 2.5, 0.0], [73.0, 25.0, -18.0], 27.1492),
        ([50.0, 2.5, 0.0], [61.0, -5.0, 29.0], 22.8977),
        ([50.0, 2.5, 0.0], [56.0, -27.0, -3.0], 31.9030),
        ([50.0, 2.5, 0.0], [58.0, 24.0, 15.0], 19.4535),
        ([50.0, 2.5, 0.0], [50.0, 3.1736, 0.5854], 1.0000),
        ([50.0, 2.5, 0.0], [50.0, 3.2972, 0.0], 1.0000),
        ([50.0, 2.5, 0.0], [50.0, 1.8634, 0.5757], 1.0000),
        ([50.0, 2.5, 0.0], [50.0, 3.2592, 0.3350], 1.0000),
        (
            [60.2574, -34.0099, 36.2677],
            [60.4626, -34.1751, 39.4387],
            1.2644,
        ),
        (
            [63.0109, -31.0961, -5.8663],
            [62.8187, -29.7946, -4.0864],
            1.2630,
        ),
        (
            [61.2901, 3.7196, -5.3901],
            [61.4292, 2.2480, -4.9620],
            1.8731,
        ),
        (
            [35.0831, -44.1164, 3.7933],
            [35.0232, -40.0716, 1.5901],
            1.8645,
        ),
        (
            [22.7233, 20.0904, -46.6940],
            [23.0331, 14.9730, -42.5619],
            2.0373,
        ),
        (
            [36.4612, 47.8580, 18.3852],
            [36.2715, 50.5065, 21.2231],
            1.4146,
        ),
        (
            [90.8027, -2.0831, 1.4410],
            [91.1528, -1.6435, 0.0447],
            1.4441,
        ),
        (
            [90.9257, -0.5406, -0.9208],
            [88.6381, -0.8985, -0.7239],
            1.5381,
        ),
        (
            [6.7747, -0.2908, -2.4247],
            [5.8714, -0.0985, -2.2286],
            0.6377,
        ),
        (
            [2.0776, 0.0795, -1.1350],
            [0.9033, -0.0636, -0.5514],
            0.9082,
        ),
    ];

    #[test]
    #[allow(non_snake_case)]
    fn ciede2000_reproduces_the_sharma_test_pairs__feat__image_conformance_harness() {
        for (i, (a, b, want)) in SHARMA.iter().enumerate() {
            let got = ciede2000(*a, *b);
            assert!(
                (got - want).abs() < 5e-5,
                "pair {}: got {got:.4}, published {want:.4}",
                i + 1
            );
            // The formula is symmetric.
            assert!(
                (ciede2000(*b, *a) - got).abs() < 1e-9,
                "pair {} asymmetric",
                i + 1
            );
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn srgb_white_black_and_primaries_land_on_known_lab__feat__image_conformance_harness() {
        let near = |got: Lab, want: Lab| {
            got.iter()
                .zip(want.iter())
                .all(|(g, w)| (g - w).abs() < 0.05)
        };
        assert!(near(srgb8_to_lab([255, 255, 255]), [100.0, 0.0, 0.0]));
        assert!(near(srgb8_to_lab([0, 0, 0]), [0.0, 0.0, 0.0]));
        // sRGB red, D65: L 53.24, a 80.09, b 67.20.
        assert!(near(srgb8_to_lab([255, 0, 0]), [53.24, 80.09, 67.20]));
    }

    #[test]
    #[allow(non_snake_case)]
    fn rgba_stats_compare_what_each_pixel_shows__feat__image_conformance_harness() {
        // Identical images: zero.
        let a = [10u8, 20, 30, 255, 200, 100, 50, 255];
        assert_eq!(delta_e_rgba8(&a, &a, [255; 3]).max, 0.0);
        // A transparent pixel shows the background, so it equals an
        // opaque pixel OF the background.
        let t = [0u8, 0, 0, 0];
        let w = [255u8, 255, 255, 255];
        assert!(delta_e_rgba8(&t, &w, [255; 3]).max < 1e-9);
        // One differing pixel of two: max > 0, mean is half of it.
        let b = [10u8, 20, 30, 255, 0, 0, 0, 255];
        let s = delta_e_rgba8(&a, &b, [255; 3]);
        assert_eq!(s.count, 2);
        assert!(s.max > 10.0);
        assert!((s.mean - s.max / 2.0).abs() < 1e-9);
        assert_eq!(s.p95, s.max);
    }
}
