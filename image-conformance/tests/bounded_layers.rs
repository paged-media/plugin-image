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

//! LAYERS AT THEIR BOUNDS (ADR 464) — pixels and masks — composite exactly
//! like the same layers at canvas size. Every Photoshop-written layered fixture is imported
//! twice — with the plates the import produces (each at its record's
//! rectangle) and with every plate expanded to the canvas — and the two
//! stacks must fold to the SAME bytes: the windowed blend of a bounded
//! plate is the canvas blend restricted to where the plate can change
//! anything.

use std::path::PathBuf;

use image_js::layers::LayerStack;
use image_psd::PsdFile;

fn fixtures() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/photoshop");
    let mut out = Vec::new();
    for dir in [
        "layer-stacks",
        "adjustment-layers",
        "layer-effects",
        "group-masks",
        "vector-masks",
    ] {
        let Ok(rd) = std::fs::read_dir(root.join(dir)) else {
            continue;
        };
        for e in rd.flatten() {
            if e.path().extension().is_some_and(|x| x == "psd") {
                out.push(e.path());
            }
        }
    }
    out.sort();
    out
}

#[test]
#[allow(non_snake_case)]
fn bounded_layers_fold_to_the_same_bytes_as_canvas_layers__feat__image_psd_layer_import() {
    let Some(ctx) = image_conformance::device::test_device() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let mut checked = 0;
    let mut bounded_plates = 0;
    for path in fixtures() {
        let file = PsdFile::parse(&std::fs::read(&path).expect("read")).expect("parse");
        let Ok(import) = file.layer_plates_rgba8() else {
            continue; // a refused fixture has no layers to compare
        };
        let mut expanded = import.clone();
        for p in &mut expanded.layers {
            if p.rect.is_some() && p.adjustment.is_none() {
                bounded_plates += usize::from(
                    p.rect != Some(image_core::Region::new(0, 0, import.width, import.height)),
                );
                p.rgba = p.canvas_rgba8(import.width, import.height);
                p.rect = None;
                if let Some(m) = &mut p.mask {
                    m.coverage = m.canvas_coverage(import.width, import.height, 0);
                    m.rect = None;
                }
            }
        }
        let a = LayerStack::from_psd_plates(&import).expect("bounded stack");
        let b = LayerStack::from_psd_plates(&expanded).expect("canvas stack");
        let fa = pollster::block_on(a.composite(Some(ctx), None)).expect("fold a");
        let fb = pollster::block_on(b.composite(Some(ctx), None)).expect("fold b");
        assert!(
            fa[..] == fb[..],
            "{}: bounded and canvas-sized layers fold differently",
            path.display()
        );
        checked += 1;
    }
    assert!(checked >= 20, "only {checked} fixtures compared");
    assert!(
        bounded_plates >= 10,
        "only {bounded_plates} plates were bounded"
    );
    println!("{checked} fixtures, {bounded_plates} bounded plates: identical folds");
}
