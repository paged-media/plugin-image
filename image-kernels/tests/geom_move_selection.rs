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

//! `geom.move_selection` — the two-mask move.
//!
//! These are MODEL tests: they restate the kernel's compositing algebra
//! in scalar Rust and prove the properties the WGSL claims. They argue
//! the DESIGN; agreement with a real adapter is the GPU lane's job, and
//! nothing here should be read as evidence of that.

use image_kernels::families::geom::{MoveSelectionParams, VacateMode, GEOM_MOVE_SELECTION};
use image_kernels::{KernelClass, Tolerance};

/// The kernel's composite, in scalar form: what one pixel becomes.
fn compose(a: f32, moved: f32, m_src: f32, m_dst: f32, copy: bool) -> f32 {
    // Additive: the unselected part stays, the selected part
    // translates. `leaving` is the only difference between the modes.
    let leaving = if copy { m_src } else { m_dst };
    a * (1.0 - leaving) + moved * m_src
}

#[test]
fn identity_at_zero_offset_for_every_mask_value_and_mode() {
    // At d = 0 the source and destination masks are the same value, and
    // the vacate cancels against the landing. This must hold for SOFT
    // masks too, which is the case a hard-edged implementation gets
    // wrong.
    for i in 0..=20 {
        let m = i as f32 / 20.0;
        for a in [0.0f32, 0.25, 0.5, 1.0] {
            for copy in [false, true] {
                // moved == a because a zero offset samples the pixel itself.
                let out = compose(a, a, m, m, copy);
                assert!(
                    (out - a).abs() < 1e-6,
                    "identity broke at m={m} a={a} copy={copy}: {out}"
                );
            }
        }
    }
}

#[test]
fn a_fully_selected_pixel_that_moves_away_is_vacated() {
    // Destination selected (it is leaving), nothing arriving.
    let out = compose(0.8, 0.3, 0.0, 1.0, false);
    assert_eq!(
        out, 0.0,
        "the source region must be emptied, not left behind"
    );
}

#[test]
fn copy_mode_leaves_the_source_intact() {
    // The SAME configuration under Copy keeps the original — this is
    // alt-drag, and it is the only difference between the two modes.
    let out = compose(0.8, 0.3, 0.0, 1.0, true);
    assert!((out - 0.8).abs() < 1e-6, "copy mode must not vacate: {out}");
}

#[test]
fn arriving_pixels_land_over_the_cleared_region() {
    // Both fire on the same pixel: it is leaving AND something is
    // arriving. This is the overlap case a single-mask formulation
    // cannot express at all, which is the whole reason this kernel
    // exists separately from geom.offset.
    let out = compose(0.8, 0.3, 1.0, 1.0, false);
    assert!(
        (out - 0.3).abs() < 1e-6,
        "the arriving pixel must win over the vacated one: {out}"
    );
}

#[test]
fn an_untouched_pixel_is_bit_exact() {
    // Neither leaving nor receiving: the kernel must not perturb it.
    for a in [0.0f32, 0.137, 0.5, 1.0] {
        assert_eq!(compose(a, 0.9, 0.0, 0.0, false), a);
        assert_eq!(compose(a, 0.9, 0.0, 0.0, true), a);
    }
}

#[test]
fn a_soft_mask_edge_survives_the_move() {
    // A feathered selection must arrive feathered: half coverage
    // arriving where nothing is leaving contributes half the moved
    // value ON TOP of the untouched destination. If the coverage were
    // thresholded this would be either 0.8 or 1.0.
    let out = compose(0.8, 0.4, 0.5, 0.0, false);
    assert!(
        (out - (0.8 + 0.4 * 0.5)).abs() < 1e-6,
        "soft coverage must blend, not threshold: {out}"
    );
}

#[test]
fn vacate_mode_decodes_strictly() {
    assert_eq!(VacateMode::from_u32(0), Some(VacateMode::Transparent));
    assert_eq!(VacateMode::from_u32(1), Some(VacateMode::Copy));
    // A newer producer's mode must be REJECTED at the boundary rather
    // than silently becoming Transparent — deleting someone's pixels is
    // the worst available answer to an unknown enum.
    assert_eq!(VacateMode::from_u32(2), None);
    assert_eq!(VacateMode::from_u32(u32::MAX), None);
}

#[test]
fn params_encode_to_the_frozen_wire_shape() {
    let p = MoveSelectionParams::new(3.0, -4.0, VacateMode::Copy);
    let b = p.as_bytes();
    assert_eq!(b.len(), 16, "params must stay 16 bytes");
    assert_eq!(&b[0..4], &3.0f32.to_le_bytes());
    assert_eq!(&b[4..8], &(-4.0f32).to_le_bytes());
    assert_eq!(&b[8..12], &1u32.to_le_bytes());
    assert_eq!(&b[12..16], &0u32.to_le_bytes(), "the ABI pad must be zero");
}

#[test]
fn kernel_metadata_matches_what_the_module_needs() {
    assert_eq!(GEOM_MOVE_SELECTION.id, "geom.move_selection");
    assert_eq!(GEOM_MOVE_SELECTION.inputs, 1);
    assert!(GEOM_MOVE_SELECTION.module);
    assert!(matches!(
        GEOM_MOVE_SELECTION.class,
        KernelClass::Resample { .. }
    ));
    assert!(matches!(
        GEOM_MOVE_SELECTION.gpu_tolerance,
        Tolerance::ChannelEpsF16(_)
    ));
    assert_eq!(
        GEOM_MOVE_SELECTION.params.size,
        core::mem::size_of::<MoveSelectionParams>()
    );
}

#[test]
fn the_module_reads_the_mask_at_both_positions() {
    let w = GEOM_MOVE_SELECTION.wgsl;
    // The defining property of this kernel. If a future edit collapses
    // it to a single mask read it becomes geom.offset with extra steps,
    // and every behavioural test above would still pass because they
    // test the scalar model rather than the shader.
    assert!(w.contains("m_src"), "must sample the mask at the SOURCE");
    assert!(
        w.contains("a * (1.0 - leaving) + moved * m_src"),
        "must use the ADDITIVE composite — a sequential clear-then-land \
         double-counts at fractional mask values"
    );
    assert!(
        w.contains("m_dst"),
        "must sample the mask at the DESTINATION"
    );
    assert!(
        w.contains("fn maskAt"),
        "must have the out-of-bounds-is-unselected mask reader"
    );
    // It must NOT end with the pointwise epilogue — it consumes the
    // mask itself, so re-applying it would scope the result to the
    // vacated region and undo the landing.
    assert!(
        !w.contains("mix(a, result, vec4<f32>(m))"),
        "this kernel writes result directly; the pointwise epilogue would undo the move"
    );
    assert!(w.contains("@group(2) @binding(0) var mask"));
    assert!(w.contains("textureStore(outp, xy, result)"));
}

#[test]
fn it_is_registered_exactly_once() {
    let n = image_kernels::families::geom::FAMILY
        .iter()
        .filter(|k| k.id == "geom.move_selection")
        .count();
    assert_eq!(n, 1, "registered {n} times in the geom family");
    let all = image_kernels::all_defined();
    assert_eq!(
        all.iter().filter(|k| k.id == "geom.move_selection").count(),
        1
    );
}
