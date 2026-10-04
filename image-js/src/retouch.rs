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

//! The PATCH tool — replace the selection with another region of the
//! image, healed so it blends.
//!
//! `patch` is the healing brush applied to a SELECTION instead of a
//! stroke: the source is the image shifted by the drag offset, the
//! membrane correction (`crate::heal::correction_field`) is solved over
//! the selected region Ω with the source−destination mismatch on its
//! boundary, and the corrected source lands through the selection's own
//! coverage as the mask — the same `PaintMode::Sample` composite the
//! healing brush uses, on the GPU. So the patch keeps the source's
//! texture and takes on the destination's tone, and a feathered
//! selection blends at its edge.
//!
//! The solve runs on a window of the selection's bounds plus a margin of
//! [`MARGIN`] px, which is where its boundary lives. Pixels the selection
//! does not cover keep their exact bytes (the splice is gated on the
//! coverage, as the stroke's is). Out-of-canvas source reads are
//! transparent, exactly as the clone stamp's are — a patch dragged half
//! off the canvas copies nothing from where there is nothing.

use image_core::Region;
use image_gpu::{composite_stroke_window, GpuContext, PaintMode, SelectionCoverage};
use image_kernels::families::compose::COMPOSE_NORMAL;

use crate::fill::{f16_to_rgba8, rgba8_to_f16};
use crate::ingest::IngestError;

/// The boundary ring the membrane solve reads around the selection.
pub const MARGIN: i32 = 6;

/// PATCH: replace the region `coverage` selects with the region at
/// `(dx, dy)` from it, tone-matched to its surroundings.
///
/// `rgba` is canvas-extent straight RGBA8. Returns the patched canvas, or
/// `None` when the selection is empty or the offset is zero (a patch
/// from itself is the identity and not worth an undo step).
pub async fn patch_rgba8(
    ctx: &GpuContext,
    rgba: &[u8],
    width: u32,
    height: u32,
    coverage: &SelectionCoverage,
    dx: i32,
    dy: i32,
) -> Result<Option<Vec<u8>>, IngestError> {
    if (dx, dy) == (0, 0) {
        return Ok(None);
    }
    let Some(bounds) = coverage.bounds() else {
        return Ok(None);
    };
    let Some(win) = Region::new(
        bounds.x - MARGIN,
        bounds.y - MARGIN,
        bounds.w + 2 * MARGIN as u32,
        bounds.h + 2 * MARGIN as u32,
    )
    .intersect(Region::new(0, 0, width, height)) else {
        return Ok(None);
    };
    let (ww, wh) = (win.w as usize, win.h as usize);
    let dest = window(rgba, width, win, 0, 0);
    let source = window(rgba, width, win, dx, dy);
    let inside: Vec<u8> = (0..ww * wh)
        .map(|i| {
            let (x, y) = (
                win.x as u32 + (i % ww) as u32,
                win.y as u32 + (i / ww) as u32,
            );
            if coverage.coverage_at(x, y) > 0 {
                255
            } else {
                0
            }
        })
        .collect();
    let field = crate::heal::correction_field(&dest, &source, &inside, ww, wh)
        .map(|f| crate::heal::field_to_f16(&f, ww * wh));
    let source_f16 = rgba8_to_f16(&source);
    let mode = PaintMode::Sample {
        blend: &COMPOSE_NORMAL,
        source_f16: &source_f16,
        correction_f16: field.as_deref(),
    };
    let out_f16 = composite_stroke_window(
        ctx,
        &mode,
        &rgba8_to_f16(&dest),
        &coverage.mask_window_f16(win),
        win.w,
        win.h,
    )
    .await
    .map_err(|e| IngestError::Pipeline(e.to_string()))?;
    let out = f16_to_rgba8(&out_f16);
    let mut patched = rgba.to_vec();
    for (i, &m) in inside.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let (x, y) = (win.x as usize + i % ww, win.y as usize + i / ww);
        let d = (y * width as usize + x) * 4;
        patched[d..d + 4].copy_from_slice(&out[i * 4..i * 4 + 4]);
    }
    Ok(Some(patched))
}

/// `region` of `rgba`, read SHIFTED by `(dx, dy)`; out-of-canvas texels
/// are transparent black.
fn window(rgba: &[u8], width: u32, region: Region, dx: i32, dy: i32) -> Vec<u8> {
    let height = (rgba.len() / 4 / width as usize) as i64;
    let mut out = vec![0u8; (region.w as usize) * (region.h as usize) * 4];
    for y in 0..region.h as i64 {
        let sy = region.y as i64 + y + dy as i64;
        if sy < 0 || sy >= height {
            continue;
        }
        for x in 0..region.w as i64 {
            let sx = region.x as i64 + x + dx as i64;
            if sx < 0 || sx >= width as i64 {
                continue;
            }
            let s = ((sy as usize) * width as usize + sx as usize) * 4;
            let d = ((y as usize) * region.w as usize + x as usize) * 4;
            out[d..d + 4].copy_from_slice(&rgba[s..s + 4]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Option<&'static GpuContext> {
        image_gpu::test_support::device_or_skip("retouch")
    }

    /// A horizontal luminance ramp with fine vertical stripes, and a dark
    /// square "stain" at 28..36 × 12..20.
    fn stained(w: u32, h: u32) -> Vec<u8> {
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let stain = (28..36).contains(&x) && (12..20).contains(&y);
                let v = if stain {
                    15u8
                } else {
                    (80 + 2 * x + if x % 4 < 2 { 6 } else { 0 }) as u8
                };
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
        }
        rgba
    }

    fn at(px: &[u8], w: u32, x: u32, y: u32) -> i32 {
        px[((y * w + x) * 4) as usize] as i32
    }

    #[test]
    #[allow(non_snake_case)]
    fn the_shifted_window_reads_the_offset_region_and_transparency_off_canvas__feat__image_editor_patch(
    ) {
        let rgba = stained(64, 32);
        let w = window(&rgba, 64, Region::new(0, 0, 4, 2), 30, 12);
        assert_eq!(w[0], 15, "the stain, read from its offset");
        let off = window(&rgba, 64, Region::new(0, 0, 4, 2), -10, 0);
        assert!(off.iter().all(|&b| b == 0), "off canvas copies nothing");
    }

    #[test]
    #[allow(non_snake_case)]
    fn a_patch_replaces_the_selection_with_healed_source_and_nothing_else__feat__image_editor_patch(
    ) {
        let Some(ctx) = device() else { return };
        let (w, h) = (96u32, 32u32);
        let rgba = stained(w, h);
        // Select the stain (with a pixel of slack); patch it from 30 px
        // to the right — a BRIGHTER part of the ramp (+60 levels), so a
        // plain copy would leave a visibly light square.
        let sel = SelectionCoverage::rasterize_rect(w, h, 27.0, 11.0, 10.0, 10.0);
        let out = pollster::block_on(patch_rgba8(ctx, &rgba, w, h, &sel, 30, 0))
            .expect("patch")
            .expect("something patched");
        // Inside: the stain is gone and the tone follows the destination
        // ramp, not the source's (which sits ~60 levels higher).
        let want = |x: u32| 80 + 2 * x as i32 + 3;
        for x in [30u32, 32, 34] {
            let got = at(&out, w, x, 16);
            assert!(
                (got - want(x)).abs() < 16,
                "({x},16) healed to {got}, the ramp there is ~{} (a plain copy is ~{})",
                want(x),
                want(x + 30)
            );
        }
        // Outside the selection, every byte is the original's.
        for y in 0..h {
            for x in 0..w {
                if sel.coverage_at(x, y) == 0 {
                    let i = ((y * w + x) * 4) as usize;
                    assert_eq!(out[i..i + 4], rgba[i..i + 4], "({x},{y}) outside");
                }
            }
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn a_patch_from_nowhere_is_none__feat__image_editor_patch() {
        let Some(ctx) = device() else { return };
        let rgba = stained(32, 32);
        let empty = SelectionCoverage::empty(32, 32);
        assert!(
            pollster::block_on(patch_rgba8(ctx, &rgba, 32, 32, &empty, 5, 0))
                .unwrap()
                .is_none()
        );
        let sel = SelectionCoverage::rasterize_rect(32, 32, 4.0, 4.0, 4.0, 4.0);
        assert!(
            pollster::block_on(patch_rgba8(ctx, &rgba, 32, 32, &sel, 0, 0))
                .unwrap()
                .is_none()
        );
    }
}
