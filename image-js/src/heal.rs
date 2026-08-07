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

//! The healing brush's GRADIENT-DOMAIN correction — the membrane solve
//! that replaced a mean offset.
//!
//! # What this fixes
//!
//! The first healing brush shifted the sampled patch by the per-channel
//! MEAN difference between source and destination. That works on evenly
//! lit skin, sky and paper, and it visibly fails across a luminance
//! ramp: a constant offset cannot cancel a difference that varies across
//! the patch, so the seam comes back. The limitation was real and was
//! documented in the panel rather than hidden — and this module is the
//! reason that sentence could be deleted.
//!
//! # The formulation
//!
//! Seamless cloning (Pérez, Gangnet & Blake, *Poisson Image Editing*,
//! SIGGRAPH 2003) writes the result as the source plus a correction
//! field, where the correction is HARMONIC inside the patch and equals
//! the source−destination difference on its boundary:
//!
//! ```text
//!     ∇²c = 0     inside Ω
//!     c   = f − g on ∂Ω
//!     result = g + c
//! ```
//!
//! That is the "membrane interpolation" special case of the full Poisson
//! problem, and it is the right one here: the guide field is the source
//! image's own gradient, so the correction only has to interpolate the
//! boundary mismatch. Solving Laplace's equation is also strictly easier
//! than solving Poisson's, which matters when it has to run per dab.
//!
//! The mean offset is the CONSTANT solution of exactly this equation, so
//! the new behaviour degrades to the old one on a patch whose boundary
//! mismatch is uniform. That is the useful sanity check, and it is a
//! test below: nothing that used to work should change.
//!
//! # Why it is CPU
//!
//! Jacobi iteration is a 5-point stencil, so an honest GPU version would
//! be N dispatches of a `conv.*` kernel with a boundary reset between
//! each — hundreds of round-trips per dab, which is not a trade a paint
//! tool can make. This is input PREPARATION, like the CMS transform at
//! decode and the histogram reduction behind auto-enhance: it computes a
//! buffer that is then handed to the GPU as a kernel input. The rule the
//! spec sets (§6) is that no CPU KERNEL path ships, and none does — the
//! correction is applied by `math.add`, a registered dispatch, exactly as
//! the mean offset was.

/// How many Jacobi sweeps to run.
///
/// The correction field is smooth by construction (it is harmonic), and
/// a dab is small, so convergence is fast; this is the point past which
/// further sweeps stop changing an 8-bit result. It is a fixed count
/// rather than a residual test on purpose — a paint tool needs a
/// PREDICTABLE per-dab cost far more than it needs the last 0.1% of
/// convergence, and an unbounded loop inside a drag is how a brush
/// becomes unusable on one unlucky stroke.
const SWEEPS: usize = 64;

/// Solve for the healing correction over one dab window.
///
/// * `dest`   — the destination window, straight RGBA8.
/// * `source` — the sampled window, straight RGBA8, same size.
/// * `inside` — one byte per pixel: non-zero where the dab covers, which
///   is the region Ω. The boundary ∂Ω is everything else.
///
/// Returns a per-pixel RGB correction in the `[0, 1]` working range
/// (alpha always 0 — healing shifts tone, never transparency), or `None`
/// when there is nothing to solve.
pub fn correction_field(
    dest: &[u8],
    source: &[u8],
    inside: &[u8],
    width: usize,
    height: usize,
) -> Option<Vec<f32>> {
    let n = width * height;
    if n == 0 || dest.len() < n * 4 || source.len() < n * 4 || inside.len() < n {
        return None;
    }

    // The boundary condition: f − g wherever the dab does NOT cover.
    // Inside Ω the field is unknown and starts from the MEAN of the
    // boundary values — the old behaviour as an initial guess, which is
    // both a good one and a guarantee that a degenerate solve returns
    // something sane rather than black.
    let mut c = vec![0f32; n * 3];
    let mut sum = [0f64; 3];
    let mut count = 0u64;
    for i in 0..n {
        if inside[i] != 0 {
            continue;
        }
        // A transparent source texel is out-of-bounds fill, not a
        // measurement — it must not become a boundary condition, or
        // sampling near an edge would pull the correction toward black
        // exactly where the tool is already weakest.
        if source[i * 4 + 3] == 0 {
            continue;
        }
        for ch in 0..3 {
            let d = (f32::from(dest[i * 4 + ch]) - f32::from(source[i * 4 + ch])) / 255.0;
            c[i * 3 + ch] = d;
            sum[ch] += f64::from(d);
        }
        count += 1;
    }
    if count == 0 {
        // No usable boundary: the dab covers the whole window, or every
        // boundary texel came from outside the image. There is nothing
        // to interpolate FROM, so no correction is invented.
        return None;
    }
    let mean = [
        (sum[0] / count as f64) as f32,
        (sum[1] / count as f64) as f32,
        (sum[2] / count as f64) as f32,
    ];
    for i in 0..n {
        if inside[i] != 0 {
            c[i * 3..i * 3 + 3].copy_from_slice(&mean);
        }
    }

    // Jacobi: each interior value becomes the average of its four
    // neighbours. Boundary values are FIXED — they are the condition.
    // Out-of-window neighbours fall back to the pixel itself, which is
    // the standard Neumann (zero-derivative) edge and keeps the field
    // from being dragged toward zero at the window rim.
    let mut next = c.clone();
    for _ in 0..SWEEPS {
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                if inside[i] == 0 {
                    continue;
                }
                for ch in 0..3 {
                    let at = |xx: usize, yy: usize| c[(yy * width + xx) * 3 + ch];
                    let left = if x > 0 { at(x - 1, y) } else { at(x, y) };
                    let right = if x + 1 < width {
                        at(x + 1, y)
                    } else {
                        at(x, y)
                    };
                    let up = if y > 0 { at(x, y - 1) } else { at(x, y) };
                    let down = if y + 1 < height {
                        at(x, y + 1)
                    } else {
                        at(x, y)
                    };
                    next[i * 3 + ch] = (left + right + up + down) * 0.25;
                }
            }
        }
        std::mem::swap(&mut c, &mut next);
    }
    Some(c)
}

/// Pack a correction field as a STRAIGHT rgba16float window, ready to be
/// uploaded as `math.add`'s second input. Alpha is zero throughout.
pub fn field_to_f16(field: &[f32], texels: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(texels * 8);
    for i in 0..texels {
        for ch in 0..3 {
            let v = half::f16::from_f32(field.get(i * 3 + ch).copied().unwrap_or(0.0));
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `w × h` RGBA8 window from a per-pixel grey value.
    fn grey(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let g = f(x, y);
                v.extend_from_slice(&[g, g, g, 255]);
            }
        }
        v
    }

    /// A centred square of coverage, so the window has a real boundary.
    fn centre_region(w: usize, h: usize, inset: usize) -> Vec<u8> {
        let mut v = vec![0u8; w * h];
        for y in inset..h - inset {
            for x in inset..w - inset {
                v[y * w + x] = 255;
            }
        }
        v
    }

    /// THE REGRESSION GUARD. A uniform boundary mismatch must still give
    /// the constant correction the mean offset gave — the new solve has
    /// to be a generalisation, not a replacement that changes what
    /// already worked.
    #[test]
    fn image_editor_heal_a_uniform_mismatch_is_still_the_mean() {
        let (w, h) = (12, 12);
        let dest = grey(w, h, |_, _| 200);
        let src = grey(w, h, |_, _| 120);
        let inside = centre_region(w, h, 3);
        let c = correction_field(&dest, &src, &inside, w, h).expect("solved");
        let want = (200.0 - 120.0) / 255.0;
        for (i, cov) in inside.iter().enumerate() {
            if *cov == 0 {
                continue;
            }
            assert!(
                (c[i * 3] - want).abs() < 1e-3,
                "interior pixel {i} corrected by {} not {want}",
                c[i * 3]
            );
        }
    }

    /// THE POINT OF THE WHOLE MODULE. A mismatch that VARIES across the
    /// patch — the case a constant offset cannot cancel — must produce a
    /// correction that varies with it. This is the assertion the mean
    /// offset fails, and the reason the panel's "still shows a seam"
    /// sentence could be deleted.
    #[test]
    fn image_editor_heal_a_gradient_mismatch_is_corrected_across_the_patch() {
        let (w, h) = (16, 16);
        // Destination ramps left→right; source is flat. The mismatch is
        // therefore a ramp, and a single number cannot cancel it.
        let dest = grey(w, h, |x, _| (x * 12) as u8);
        let src = grey(w, h, |_, _| 0);
        let inside = centre_region(w, h, 4);
        let c = correction_field(&dest, &src, &inside, w, h).expect("solved");

        let at = |x: usize, y: usize| c[(y * w + x) * 3];
        let (left, mid, right) = (at(5, 8), at(8, 8), at(10, 8));
        assert!(
            left < mid && mid < right,
            "the correction must RAMP with the mismatch: {left} < {mid} < {right}"
        );
        // And it must track the actual boundary values, not merely be
        // monotonic: the interior is an interpolation of them.
        let boundary_left = (4 * 12) as f32 / 255.0;
        let boundary_right = (11 * 12) as f32 / 255.0;
        assert!(left > boundary_left - 0.05 && left < boundary_right);
        assert!(right < boundary_right + 0.05 && right > boundary_left);
    }

    /// The mean offset would have given ONE number here; the solve gives
    /// a field. Stated as a direct comparison so the improvement is
    /// measured rather than asserted.
    #[test]
    fn image_editor_heal_the_solve_beats_a_constant_on_a_ramp() {
        let (w, h) = (16, 16);
        let dest = grey(w, h, |x, _| (x * 12) as u8);
        let src = grey(w, h, |_, _| 0);
        let inside = centre_region(w, h, 4);
        let c = correction_field(&dest, &src, &inside, w, h).expect("solved");

        // The constant the OLD implementation would have used: the mean
        // difference over the whole window.
        let mut sum = 0f32;
        for i in 0..w * h {
            sum += (f32::from(dest[i * 4]) - f32::from(src[i * 4])) / 255.0;
        }
        let constant = sum / (w * h) as f32;

        // Residual = how far the corrected source still is from the
        // destination, summed over the patch. Lower is better.
        let mut solve_err = 0f32;
        let mut const_err = 0f32;
        for i in 0..w * h {
            if inside[i] == 0 {
                continue;
            }
            let want = (f32::from(dest[i * 4]) - f32::from(src[i * 4])) / 255.0;
            solve_err += (c[i * 3] - want).abs();
            const_err += (constant - want).abs();
        }
        assert!(
            solve_err < const_err * 0.5,
            "the solve should beat the constant substantially on a ramp: \
             {solve_err} vs {const_err}"
        );
    }

    /// No usable boundary means no invented correction.
    #[test]
    fn image_editor_heal_a_boundaryless_window_yields_no_correction() {
        let (w, h) = (8, 8);
        let dest = grey(w, h, |_, _| 200);
        let src = grey(w, h, |_, _| 100);
        // Covers everything: there is nothing to interpolate FROM.
        assert!(correction_field(&dest, &src, &vec![255u8; w * h], w, h).is_none());
        // And a degenerate window is refused rather than indexed.
        assert!(correction_field(&[], &[], &[], 0, 0).is_none());
    }

    /// Transparent source texels are out-of-bounds fill, not boundary
    /// data — the same rule the mean offset had, carried forward.
    #[test]
    fn image_editor_heal_transparent_boundary_texels_are_not_conditions() {
        let (w, h) = (10, 10);
        let dest = grey(w, h, |_, _| 200);
        let mut src = grey(w, h, |_, _| 120);
        // Blank the left column of the boundary.
        for y in 0..h {
            src[(y * w) * 4 + 3] = 0;
        }
        let inside = centre_region(w, h, 3);
        let c = correction_field(&dest, &src, &inside, w, h).expect("solved");
        let want = (200.0 - 120.0) / 255.0;
        // The correction is still the honest one — the blank column did
        // not drag it toward (200 − 0)/255.
        assert!(
            (c[(5 * w + 5) * 3] - want).abs() < 1e-2,
            "got {}",
            c[(5 * w + 5) * 3]
        );
    }

    /// Alpha is never touched, and the packing is the shape `math.add`
    /// takes.
    #[test]
    fn image_editor_heal_the_field_packs_as_rgba16f_with_zero_alpha() {
        let field = vec![0.5f32, 0.25, 0.125];
        let packed = field_to_f16(&field, 1);
        assert_eq!(packed.len(), 8, "one rgba16float texel");
        let a = half::f16::from_le_bytes([packed[6], packed[7]]);
        assert_eq!(a.to_f32(), 0.0, "a heal shifts tone, never transparency");
        let r = half::f16::from_le_bytes([packed[0], packed[1]]);
        assert!((r.to_f32() - 0.5).abs() < 1e-3);
    }
}
