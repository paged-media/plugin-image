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

//! Defects found by the composite oracles, each pinned by a test stating
//! the behaviour Photoshop shows. A pin is `#[ignore]`d with a `DEFECT`
//! reason until its fix lands, and the fix removes the `#[ignore]`; every
//! pin in this file is fixed today.

use std::sync::Arc;

use image_js::layers::LayerStack;
use image_psd::{LayerImport, LayerPlate};

const W: u32 = 4;
const H: u32 = 4;

fn plate(name: &str, rgb: [u8; 3], opacity: u8, key: &[u8; 4], clipped: bool) -> LayerPlate {
    LayerPlate {
        name: name.into(),
        blend_key: *key,
        opacity,
        hidden: false,
        rgba: (0..W * H)
            .flat_map(|_| [rgb[0], rgb[1], rgb[2], 255])
            .collect(),
        rect: None,
        clipped,
        group: None,
        mask: None,
        vector_mask: None,
        mask_params: None,
        smart: false,
        text: false,
        adjustment: None,
        color_overlay: None,
    }
}

fn flatten(layers: Vec<LayerPlate>) -> Option<Arc<[u8]>> {
    let ctx = image_conformance::device::test_device()?;
    let stack = LayerStack::from_psd_plates(&LayerImport {
        groups: Vec::new(),
        width: W,
        height: H,
        depth_reduced: false,
        converted_from_cmyk: false,
        layers,
    })
    .expect("stack");
    Some(pollster::block_on(stack.composite(Some(ctx), None)).expect("composite"))
}

/// CLIPPING GROUP OPACITY. In Photoshop the clip BASE's opacity (and
/// blend mode) apply to the whole clipping group: the clipped layers
/// blend onto the base at full strength, and THAT result is faded onto
/// what lies below. The fold applies the base's opacity to the base
/// alone and then blended the clipped layer over the already-faded
/// result; it now folds a clip base and its clipped layers as a group
/// (`layers/fold.rs`, `ClipOpen` / `ClipClose`).
///
/// Found by the corpus composite oracle (a base at 90 % opacity under a
/// clipped `divide` layer: 4,135 pixels off by up to 9 levels; the
/// per-pixel arithmetic matches the group semantics exactly). With a 50 %
/// base and a clipped multiply the error is ~50 levels:
///
/// ```text
/// L0 (200,200,200) normal 100 %
/// L1 ( 40, 60, 80) normal  50 %          ← clip base
/// L2 (128,128,128) multiply 100 %, clipped
/// Photoshop: L0 ⊕ 50 %·(L1 × L2)  = (110, 115, 120)
/// ours:      (L0 ⊕ 50 %·L1) × L2  = ( 60,  65,  70)
/// ```
#[test]
#[allow(non_snake_case)]
fn clip_base_opacity_fades_the_whole_clipping_group__feat__image_layers_clipping() {
    let Some(out) = flatten(vec![
        plate("L0", [200, 200, 200], 255, b"norm", false),
        plate("L1", [40, 60, 80], 128, b"norm", false),
        plate("L2", [128, 128, 128], 255, b"mul ", true),
    ]) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let a = 128.0 / 255.0;
    let want: Vec<u8> = [40.0f32, 60.0, 80.0]
        .iter()
        .map(|c| {
            let group = c * 128.0 / 255.0;
            (group * a + 200.0 * (1.0 - a)).round() as u8
        })
        .collect();
    let got = &out[..3];
    assert!(
        got.iter().zip(&want).all(|(g, w)| g.abs_diff(*w) <= 2),
        "clipping group: got {got:?}, Photoshop gives {want:?}"
    );
}

/// The full-opacity case agrees — the defect is the opacity, not the
/// clip itself. Kept always-on so a fix for the pin above cannot break
/// the case that already works.
#[test]
#[allow(non_snake_case)]
fn clip_at_full_base_opacity_matches_the_group_semantics__feat__image_layers_clipping() {
    let Some(out) = flatten(vec![
        plate("L0", [200, 200, 200], 255, b"norm", false),
        plate("L1", [40, 60, 80], 255, b"norm", false),
        plate("L2", [128, 128, 128], 255, b"mul ", true),
    ]) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let want: Vec<u8> = [40.0f32, 60.0, 80.0]
        .iter()
        .map(|c| (c * 128.0 / 255.0).round() as u8)
        .collect();
    let got = &out[..3];
    assert!(
        got.iter().zip(&want).all(|(g, w)| g.abs_diff(*w) <= 1),
        "got {got:?}, want {want:?}"
    );
}
