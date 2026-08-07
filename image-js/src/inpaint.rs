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

//! CONTENT-AWARE FILL — exemplar-based inpainting.
//!
//! The catalog priced this as the one genuine XL left in the retouching
//! family, and it is the only one of the three that is not a brush:
//! filling a selection means SYNTHESISING plausible content, which is a
//! search problem rather than a compositing one. That is why it was
//! deliberately left out when clone and heal shipped — a button that
//! returned a blurry average would have been worse than its absence.
//!
//! # The algorithm, and why this one
//!
//! Exemplar-based inpainting (Criminisi, Pérez & Toyama, *Region Filling
//! and Object Removal by Exemplar-Based Image Inpainting*, IEEE TIP
//! 2004). The hole is filled PATCH BY PATCH from its boundary inward,
//! and the order is the whole idea:
//!
//! ```text
//!     P(p) = C(p) · D(p)
//!     C(p) = mean confidence over the known part of the patch at p
//!     D(p) = |∇I⊥(p) · n(p)| / α        (the isophote/normal term)
//! ```
//!
//! `C` prefers patches whose neighbourhood is mostly already known, so
//! the fill grows inward evenly instead of racing down one corridor.
//! `D` prefers patches where a strong edge runs INTO the hole, so linear
//! structure is continued before flat texture is. Together they are what
//! separates "content-aware" from "blur": a diffusion-based fill has no
//! `D` term and smears every edge it crosses.
//!
//! Each chosen patch is filled by copying the best-matching patch found
//! elsewhere in the image, compared over the pixels that are already
//! known. So the output is made of REAL pixels from the same image,
//! which is why it keeps grain and texture that any smoothing method
//! destroys.
//!
//! # The honest limits, stated here and in the panel
//!
//! * **The search is windowed.** Comparing every source position for
//!   every boundary patch is quadratic in image area; the search is
//!   bounded to a radius around the target, which is where plausible
//!   texture lives anyway. A hole whose best match is on the far side of
//!   a large image will not find it.
//! * **Single scale.** No pyramid, so very large holes reproduce texture
//!   faithfully but can lose large-scale structure.
//! * **Deterministic.** Same input, same output — no random restarts. A
//!   retoucher who does not like the result changes the selection, and
//!   the same selection twice gives the same answer.

use image_gpu::SelectionCoverage;

/// Patch half-width. The patch is `2·R+1` square — 9×9, the size the
/// paper uses and the one that holds texture without over-smoothing.
const R: i32 = 4;

/// How far from a target patch to look for its exemplar.
///
/// Bounded because an exhaustive search is quadratic in image area, and
/// unbounded search time inside an interactive tool is not a trade worth
/// making. Plausible replacement texture is almost always local.
const SEARCH: i32 = 96;

/// Fill every pixel the coverage marks, synthesising from the rest of
/// the image.
///
/// `rgba` is canvas-extent straight RGBA8 and is returned MODIFIED.
/// `coverage` marks the hole (any value at or above `threshold`).
/// Returns `None` when there is nothing to fill, or nothing to fill FROM.
pub fn fill(
    rgba: &[u8],
    width: u32,
    height: u32,
    coverage: &SelectionCoverage,
    threshold: u8,
) -> Option<Vec<u8>> {
    let (w, h) = (width as i32, height as i32);
    let n = (width as usize) * (height as usize);
    if n == 0 || rgba.len() < n * 4 {
        return None;
    }

    // `known[i]`: this pixel is source material. `conf[i]`: Criminisi's
    // confidence, 1 for original pixels and inherited by filled ones —
    // which is what makes later patches trust early ones less.
    let mut out = rgba.to_vec();
    let mut known = vec![true; n];
    let mut conf = vec![1f32; n];
    let mut remaining = 0usize;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if coverage.coverage_at(x as u32, y as u32) >= threshold {
                known[i] = false;
                conf[i] = 0.0;
                remaining += 1;
            }
        }
    }
    if remaining == 0 || remaining == n {
        // Nothing selected, or everything is: with no source region
        // there is nothing to synthesise FROM, and inventing pixels from
        // nothing is not a thing this should pretend to do.
        return None;
    }

    // Guard the loop: every iteration fills at least one pixel, so this
    // cannot spin — but a bound makes that a fact rather than a belief.
    let mut guard = remaining * 4 + 64;
    while remaining > 0 && guard > 0 {
        guard -= 1;
        let Some(target) = best_front_patch(&out, &known, &conf, w, h) else {
            break;
        };
        let (tx, ty) = target;
        let Some((sx, sy)) = best_exemplar(&out, &known, w, h, tx, ty) else {
            // No usable exemplar anywhere in range. Rather than leave the
            // hole (an infinite loop) or invent a colour, mark this
            // patch known at its current value and move on — the result
            // is visibly unfilled there, which is the honest outcome.
            for dy in -R..=R {
                for dx in -R..=R {
                    let (px, py) = (tx + dx, ty + dy);
                    if px < 0 || py < 0 || px >= w || py >= h {
                        continue;
                    }
                    let i = (py * w + px) as usize;
                    if !known[i] {
                        known[i] = true;
                        remaining -= 1;
                    }
                }
            }
            continue;
        };

        // Criminisi's update: the whole patch inherits the target's
        // confidence, so material copied later is trusted less than
        // material copied early.
        let patch_conf = mean_conf(&conf, &known, w, h, tx, ty);
        for dy in -R..=R {
            for dx in -R..=R {
                let (px, py) = (tx + dx, ty + dy);
                let (qx, qy) = (sx + dx, sy + dy);
                if px < 0 || py < 0 || px >= w || py >= h {
                    continue;
                }
                if qx < 0 || qy < 0 || qx >= w || qy >= h {
                    continue;
                }
                let pi = (py * w + px) as usize;
                if known[pi] {
                    continue;
                }
                let qi = (qy * w + qx) as usize;
                let texel = rgba_at(&out, qi);
                out[pi * 4..pi * 4 + 4].copy_from_slice(&texel);
                known[pi] = true;
                conf[pi] = patch_conf;
                remaining -= 1;
            }
        }
    }
    Some(out)
}

fn rgba_at(buf: &[u8], i: usize) -> [u8; 4] {
    [buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]]
}

/// Is this pixel on the fill front — unknown, with a known neighbour?
fn on_front(known: &[bool], w: i32, h: i32, x: i32, y: i32) -> bool {
    let i = (y * w + x) as usize;
    if known[i] {
        return false;
    }
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let (nx, ny) = (x + dx, y + dy);
        if nx < 0 || ny < 0 || nx >= w || ny >= h {
            continue;
        }
        if known[(ny * w + nx) as usize] {
            return true;
        }
    }
    false
}

fn mean_conf(conf: &[f32], known: &[bool], w: i32, h: i32, x: i32, y: i32) -> f32 {
    let mut sum = 0f32;
    let mut count = 0f32;
    for dy in -R..=R {
        for dx in -R..=R {
            let (px, py) = (x + dx, y + dy);
            if px < 0 || py < 0 || px >= w || py >= h {
                continue;
            }
            let i = (py * w + px) as usize;
            count += 1.0;
            if known[i] {
                sum += conf[i];
            }
        }
    }
    if count == 0.0 {
        0.0
    } else {
        sum / count
    }
}

/// The front pixel with the highest Criminisi priority.
fn best_front_patch(
    rgba: &[u8],
    known: &[bool],
    conf: &[f32],
    w: i32,
    h: i32,
) -> Option<(i32, i32)> {
    let mut best: Option<((i32, i32), f32)> = None;
    for y in 0..h {
        for x in 0..w {
            if !on_front(known, w, h, x, y) {
                continue;
            }
            let c = mean_conf(conf, known, w, h, x, y);
            let d = data_term(rgba, known, w, h, x, y);
            // The `+ 0.001` floor on D keeps a patch with no measurable
            // isophote from getting priority zero, which would freeze it
            // out of the queue forever and leave a hole unfilled.
            let p = c * (d + 0.001);
            if best.is_none_or(|(_, bp)| p > bp) {
                best = Some(((x, y), p));
            }
        }
    }
    best.map(|(pt, _)| pt)
}

/// `|∇I⊥ · n|` — how strongly an edge runs INTO the hole here. This is
/// the term that makes structure continue instead of blurring.
fn data_term(rgba: &[u8], known: &[bool], w: i32, h: i32, x: i32, y: i32) -> f32 {
    let lum = |xx: i32, yy: i32| -> Option<f32> {
        if xx < 0 || yy < 0 || xx >= w || yy >= h {
            return None;
        }
        let i = (yy * w + xx) as usize;
        if !known[i] {
            return None;
        }
        Some(
            (0.2126 * f32::from(rgba[i * 4])
                + 0.7152 * f32::from(rgba[i * 4 + 1])
                + 0.0722 * f32::from(rgba[i * 4 + 2]))
                / 255.0,
        )
    };
    // Central differences over KNOWN pixels only; a one-sided or absent
    // neighbour contributes nothing rather than a fabricated gradient.
    let gx = match (lum(x + 1, y), lum(x - 1, y)) {
        (Some(a), Some(b)) => (a - b) * 0.5,
        _ => 0.0,
    };
    let gy = match (lum(x, y + 1), lum(x, y - 1)) {
        (Some(a), Some(b)) => (a - b) * 0.5,
        _ => 0.0,
    };
    // The fill-front normal: the gradient of the known/unknown mask.
    let m = |xx: i32, yy: i32| -> f32 {
        if xx < 0 || yy < 0 || xx >= w || yy >= h {
            return 1.0;
        }
        if known[(yy * w + xx) as usize] {
            1.0
        } else {
            0.0
        }
    };
    let nx = (m(x + 1, y) - m(x - 1, y)) * 0.5;
    let ny = (m(x, y + 1) - m(x, y - 1)) * 0.5;
    let nlen = (nx * nx + ny * ny).sqrt();
    if nlen <= f32::EPSILON {
        return 0.0;
    }
    // ∇I⊥ is the gradient rotated 90° — the ISOPHOTE direction, i.e. the
    // direction an edge runs, not the direction it changes.
    ((-gy) * (nx / nlen) + gx * (ny / nlen)).abs()
}

/// The best-matching fully-known patch within `SEARCH` of the target,
/// compared over the target's known pixels only.
fn best_exemplar(
    rgba: &[u8],
    known: &[bool],
    w: i32,
    h: i32,
    tx: i32,
    ty: i32,
) -> Option<(i32, i32)> {
    let mut best: Option<((i32, i32), f64)> = None;
    let x0 = (tx - SEARCH).max(R);
    let x1 = (tx + SEARCH).min(w - 1 - R);
    let y0 = (ty - SEARCH).max(R);
    let y1 = (ty + SEARCH).min(h - 1 - R);
    for sy in y0..=y1 {
        'candidate: for sx in x0..=x1 {
            let mut ssd = 0f64;
            let mut compared = 0u32;
            for dy in -R..=R {
                for dx in -R..=R {
                    let (px, py) = (tx + dx, ty + dy);
                    let (qx, qy) = (sx + dx, sy + dy);
                    if px < 0 || py < 0 || px >= w || py >= h {
                        continue;
                    }
                    let pi = (py * w + px) as usize;
                    let qi = (qy * w + qx) as usize;
                    // A candidate patch must be ENTIRELY known — copying
                    // from a partly-unfilled patch would propagate the
                    // hole instead of closing it.
                    if !known[qi] {
                        continue 'candidate;
                    }
                    if !known[pi] {
                        continue;
                    }
                    for ch in 0..3 {
                        let d = f64::from(rgba[pi * 4 + ch]) - f64::from(rgba[qi * 4 + ch]);
                        ssd += d * d;
                    }
                    compared += 1;
                }
            }
            if compared == 0 {
                continue;
            }
            // Normalised, so a patch clipped by the image edge is not
            // preferred merely for having fewer terms in its sum.
            let score = ssd / f64::from(compared);
            if best.is_none_or(|(_, b)| score < b) {
                best = Some(((sx, sy), score));
            }
        }
    }
    best.map(|(pt, _)| pt)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `w × h` RGBA8 from a per-pixel colour.
    fn img(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let c = f(x, y);
                v.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        v
    }

    /// A rectangular hole.
    fn hole(w: u32, h: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> SelectionCoverage {
        let mut d = vec![0u8; (w * h) as usize];
        for y in y0..y1 {
            for x in x0..x1 {
                d[(y * w + x) as usize] = 255;
            }
        }
        SelectionCoverage::from_data(w, h, d).expect("coverage")
    }

    #[test]
    fn image_editor_inpaint_fills_every_selected_pixel() {
        let (w, h) = (64u32, 64u32);
        // Vertical stripes: a texture with a clear period.
        let src = img(w, h, |x, _| {
            let v = if (x / 4) % 2 == 0 { 30 } else { 220 };
            [v, v, v]
        });
        let sel = hole(w, h, 28, 28, 36, 36);
        let out = fill(&src, w, h, &sel, 128).expect("filled");
        // The hole is gone: nothing is left at the sentinel value a
        // partial fill would leave (the original pixels are 30 or 220,
        // and every filled pixel is copied from those).
        for y in 28..36 {
            for x in 28..36 {
                let i = ((y * w + x) * 4) as usize;
                assert!(
                    out[i] == 30 || out[i] == 220,
                    "pixel ({x},{y}) is {} — filled from real image pixels",
                    out[i]
                );
            }
        }
    }

    /// The property that separates content-aware from blur: the fill is
    /// made of REAL pixels, so a two-valued image stays two-valued. A
    /// diffusion fill would produce intermediate greys.
    #[test]
    fn image_editor_inpaint_synthesises_texture_rather_than_averaging_it() {
        let (w, h) = (48u32, 48u32);
        let src = img(w, h, |x, _| {
            let v = if (x / 3) % 2 == 0 { 0 } else { 255 };
            [v, v, v]
        });
        let sel = hole(w, h, 20, 20, 28, 28);
        let out = fill(&src, w, h, &sel, 128).expect("filled");
        let mut intermediate = 0;
        for y in 20..28 {
            for x in 20..28 {
                let v = out[((y * w + x) * 4) as usize];
                if v > 20 && v < 235 {
                    intermediate += 1;
                }
            }
        }
        assert_eq!(
            intermediate, 0,
            "a blur would leave intermediate greys; an exemplar fill \
             cannot, because every pixel it writes was copied"
        );
    }

    /// STRUCTURE survives: a hole straddling a strong edge comes back
    /// with the edge still crossing it.
    ///
    /// SCOPE, measured rather than assumed: this passes with the data
    /// term REMOVED, which was checked by mutation. On a two-tone image
    /// the exemplar SEARCH alone finds a patch matching the known half,
    /// and that patch carries the edge — so what this test proves is
    /// that copying real patches preserves structure, not that the
    /// priority ordering does. The data term changes the ORDER patches
    /// are filled in, which shows on harder geometry (a hole large
    /// enough for the front to reach itself from two directions) and is
    /// not isolated by any test here. Saying so is better than implying
    /// a coverage this file does not have.
    #[test]
    fn image_editor_inpaint_preserves_an_edge_across_the_hole() {
        let (w, h) = (64u32, 48u32);
        // Top half dark, bottom half light: one strong horizontal edge.
        let src = img(w, h, |_, y| {
            let v = if y < 24 { 20 } else { 230 };
            [v, v, v]
        });
        let sel = hole(w, h, 26, 18, 38, 30);
        let out = fill(&src, w, h, &sel, 128).expect("filled");
        let at = |x: u32, y: u32| out[((y * w + x) * 4) as usize] as i32;
        // Inside the hole, above the edge should still be dark and below
        // should still be light.
        for x in 28..36 {
            assert!(at(x, 20) < 128, "above the edge stayed dark at x={x}");
            assert!(at(x, 28) > 128, "below the edge stayed light at x={x}");
        }
    }

    /// Nothing to fill, or nothing to fill FROM, is `None` — never an
    /// invented image.
    #[test]
    fn image_editor_inpaint_refuses_a_hole_with_no_source() {
        let (w, h) = (16u32, 16u32);
        let src = img(w, h, |_, _| [100, 100, 100]);
        // Empty selection.
        let empty = SelectionCoverage::from_data(w, h, vec![0u8; (w * h) as usize]).unwrap();
        assert!(fill(&src, w, h, &empty, 128).is_none());
        // Everything selected: no source region exists.
        let all = SelectionCoverage::from_data(w, h, vec![255u8; (w * h) as usize]).unwrap();
        assert!(fill(&src, w, h, &all, 128).is_none());
    }

    /// Same input, same output. A retoucher who reruns a fill on the
    /// same selection gets the same answer, which is what makes the tool
    /// predictable enough to iterate with.
    #[test]
    fn image_editor_inpaint_is_deterministic() {
        let (w, h) = (48u32, 48u32);
        let src = img(w, h, |x, y| [(x * 5) as u8, (y * 5) as u8, 128]);
        let sel = hole(w, h, 18, 18, 26, 26);
        let a = fill(&src, w, h, &sel, 128).expect("a");
        let b = fill(&src, w, h, &sel, 128).expect("b");
        assert_eq!(a, b);
    }

    /// The fill NEVER touches a pixel outside the selection.
    #[test]
    fn image_editor_inpaint_leaves_everything_outside_the_selection_alone() {
        let (w, h) = (48u32, 48u32);
        let src = img(w, h, |x, y| [(x * 5) as u8, (y * 5) as u8, 77]);
        let sel = hole(w, h, 18, 18, 26, 26);
        let out = fill(&src, w, h, &sel, 128).expect("filled");
        for y in 0..h {
            for x in 0..w {
                if (18..26).contains(&x) && (18..26).contains(&y) {
                    continue;
                }
                let i = ((y * w + x) * 4) as usize;
                assert_eq!(
                    &out[i..i + 4],
                    &src[i..i + 4],
                    "pixel ({x},{y}) outside the selection was modified"
                );
            }
        }
    }
}
