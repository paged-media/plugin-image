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

//! Wall-clock trend for the operations `tests/perf_budgets.rs` budgets in
//! counts. REPORTED, never gated: a timing depends on the machine, a
//! count does not. Skips without a GPU adapter.
//!
//! `cargo bench -p image-js --bench ops`

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
use image_core::Region;
use image_js::ingest::{adjust_rgba8, AdjustParams, DecodedImage};
use image_js::layers::LayerStack;
use image_js::pixels::Pixels;

fn ramp(w: u32, h: u32, seed: u8) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| [(i % w) as u8 ^ seed, (i / w) as u8, seed, 200])
        .collect()
}

fn stack(w: u32, h: u32, n: usize) -> LayerStack {
    let mut s = LayerStack::from_image(w, h, Arc::from(ramp(w, h, 1))).expect("stack");
    for i in 1..n {
        s.add(&format!("L{i}"));
        s.edit_active(
            "fill",
            Region::new(0, 0, w, h),
            Pixels::from_rgba8(Arc::from(ramp(w, h, (i * 40) as u8))),
        )
        .expect("edit");
    }
    s
}

fn ops(c: &mut Criterion) {
    let Some(ctx) = image_gpu::test_support::device_or_skip("bench") else {
        return;
    };
    let s = stack(1024, 1024, 3);
    c.bench_function("composite 3 layers 1024²", |b| {
        b.iter(|| pollster::block_on(s.composite(Some(ctx), None)).expect("composite"))
    });
    let img = DecodedImage::from_rgba8(1024, 1024, ramp(1024, 1024, 3)).expect("img");
    let p = AdjustParams {
        exposure_ev: 0.3,
        contrast: 0.2,
        saturation: 0.1,
        hue_degrees: 10.0,
        blur_sigma: 1.5,
        sharpen_amount: 0.5,
        ..AdjustParams::default()
    };
    c.bench_function("apply 6 stages 1024²", |b| {
        b.iter(|| pollster::block_on(adjust_rgba8(ctx, &img, &p, None)).expect("apply"))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = ops
}
criterion_main!(benches);
