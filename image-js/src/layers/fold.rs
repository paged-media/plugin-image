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

//! The GPU-RESIDENT layer fold.
//!
//! The fold used to read the whole accumulator back after every layer
//! and upload it again for the next one, and to upload and premultiply
//! every layer's plate on every composite. Here the accumulator stays
//! on the device: every blend of one composite is recorded into one
//! batch, one submit, ONE readback of the finished canvas.
//!
//! Three caches live between composites, all keyed by IDENTITY (the
//! `Arc` of a layer's pixels, of its mask, of its clip base) plus the
//! layer's properties — so they need no invalidation hook: an edit
//! replaces the `Arc`, and the old key simply stops matching. Each
//! cache holds clones of the `Arc`s it keys on, so an address cannot be
//! reused under a stale entry.
//!
//! * PLATES: each layer's premultiplied rgba16float plate, uploaded once
//!   per pixel generation.
//! * A PREFIX CHECKPOINT: the fold state just before the active layer.
//!   An edit at or above it (paint, opacity, blend, mask) re-folds from
//!   there instead of from the bottom.
//! * THE LAST COMPOSITE: an unchanged stack hands back the last result.
//!
//! Every recorded dispatch is the dispatch the CPU round-trip fold made
//! (same kernels, params, masks, extents), and rgba16float textures hold
//! exactly the bytes that round trip re-uploaded, so the result is byte
//! for byte the same (`image-conformance/tests/prop_layer_cache.rs`
//! compares the two). The one rewrite that is not a pure relocation:
//! the old fold skipped its final unpremultiply when the whole canvas
//! was opaque (it could look at the bytes); the resident fold cannot
//! without a second readback, so it always dispatches it — which over
//! an opaque texel divides by one and is the identity (asserted
//! exhaustively over f16 by the same test file).

use std::collections::HashMap;
use std::sync::Arc;

use image_gpu::coverage::SelectionCoverage;
use image_gpu::selection::SelectionMask;
use image_gpu::stroke::window_is_opaque;
use image_gpu::{GpuBatch, GpuContext, Resident, TexFormat};
use image_kernels::families::cast::{
    CastPremultiplyParams, CastUnpremultiplyParams, CAST_PREMULTIPLY, CAST_UNPREMULTIPLY,
};
use image_kernels::families::compose::ComposeParams;
use image_kernels::KernelDef;

use super::{alpha_of, effective_coverage, LayerStack};
use crate::fill::{f16_to_rgba8, rgba8_to_f16};
use crate::ingest::{AdjustParams, IngestError};

/// Device bytes of plates the cache keeps between composites. Past it a
/// plate is still uploaded for the composite that needs it, just not
/// kept — the cache trades memory for uploads, and an unbounded trade
/// would put a large many-layer document's every plate on the device.
const PLATE_CACHE_BYTES: u64 = 512 << 20;

fn gpu_err(e: image_gpu::GpuError) -> IngestError {
    IngestError::Pipeline(e.to_string())
}

fn opt_ptr_eq<T: ?Sized>(a: &Option<Arc<T>>, b: &Option<Arc<T>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}

/// What a step's coverage is built from: the layer's own live mask and
/// the clip base's pixels (whose alpha is the clip).
#[derive(Clone, Default)]
pub(super) struct CovKey {
    pub own: Option<Arc<SelectionCoverage>>,
    pub clip: Option<Arc<[u8]>>,
}

impl CovKey {
    fn same(&self, o: &CovKey) -> bool {
        opt_ptr_eq(&self.own, &o.own) && opt_ptr_eq(&self.clip, &o.clip)
    }

    fn is_none(&self) -> bool {
        self.own.is_none() && self.clip.is_none()
    }

    /// The combined coverage, exactly as the fold has always built it.
    pub(super) fn coverage(&self, w: u32, h: u32) -> Option<Arc<SelectionCoverage>> {
        let clip = self.clip.as_ref().map(|px| alpha_of(px));
        effective_coverage(self.own.as_ref(), clip.as_deref(), w, h)
    }

    /// The r16float mask bytes, exactly as the fold has always lowered them.
    fn mask_bytes(&self, w: u32, h: u32) -> Option<Vec<u8>> {
        self.coverage(w, h).map(|cov| {
            SelectionMask::from_fn(w, h, |x, y| f32::from(cov.coverage_at(x, y)) / 255.0)
                .into_bytes()
        })
    }
}

/// One step of the fold, in the order the fold runs them.
#[derive(Clone)]
pub(super) enum Step {
    /// Blend a pixel plate into the accumulator.
    Pixel {
        layer: u32,
        px: Arc<[u8]>,
        blend: &'static KernelDef,
        opacity: f32,
        cov: CovKey,
    },
    /// Run the adjust chain over the accumulator.
    Adjust {
        params: Box<AdjustParams>,
        cov: CovKey,
    },
    /// Park the accumulator and start an isolated group's own.
    Open { group: u32 },
    /// Blend the group's accumulator into the parked one (`None`: the
    /// group vanished, and the parked accumulator comes back unchanged).
    Close {
        group: u32,
        blend: Option<(&'static KernelDef, f32)>,
    },
}

impl Step {
    fn same(&self, o: &Step) -> bool {
        match (self, o) {
            (
                Step::Pixel {
                    layer,
                    px,
                    blend,
                    opacity,
                    cov,
                },
                Step::Pixel {
                    layer: l2,
                    px: p2,
                    blend: b2,
                    opacity: o2,
                    cov: c2,
                },
            ) => {
                layer == l2
                    && Arc::ptr_eq(px, p2)
                    && std::ptr::eq(*blend, *b2)
                    && opacity.to_bits() == o2.to_bits()
                    && cov.same(c2)
            }
            (
                Step::Adjust { params, cov },
                Step::Adjust {
                    params: p2,
                    cov: c2,
                },
            ) => params == p2 && cov.same(c2),
            (Step::Open { group }, Step::Open { group: g2 }) => group == g2,
            (
                Step::Close { group, blend },
                Step::Close {
                    group: g2,
                    blend: b2,
                },
            ) => {
                group == g2
                    && match (blend, b2) {
                        (None, None) => true,
                        (Some((d, o)), Some((d2, o2))) => {
                            std::ptr::eq(*d, *d2) && o.to_bits() == o2.to_bits()
                        }
                        _ => false,
                    }
            }
            _ => false,
        }
    }

    fn is_layer(&self) -> bool {
        matches!(self, Step::Pixel { .. } | Step::Adjust { .. })
    }
}

pub(super) fn same_steps(a: &[Step], b: &[Step]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.same(y))
}

/// The accumulator and the parked accumulators of the open isolated
/// groups. `None` is transparent black (never materialized).
#[derive(Clone, Default)]
pub(super) struct FoldState {
    acc: Option<Resident>,
    parked: Vec<Option<Resident>>,
}

/// Everything the fold keeps on the device between composites.
#[derive(Default)]
pub(super) struct FoldCache {
    /// The device these textures live on (its address).
    device: usize,
    /// Layer id → (the pixels the plate was made from, the plate).
    plates: HashMap<u32, (Arc<[u8]>, Resident)>,
    /// Layer id → (what the mask was made from, the r16float mask).
    masks: HashMap<u32, (CovKey, Resident)>,
    /// The steps before the checkpoint, and the state after them.
    checkpoint: Option<(Vec<Step>, FoldState)>,
    /// The last composite's steps and result.
    last: Option<(Vec<Step>, Arc<[u8]>)>,
}

impl FoldCache {
    fn bind(&mut self, ctx: &GpuContext) {
        let id = ctx as *const GpuContext as usize;
        if self.device != id {
            *self = FoldCache {
                device: id,
                ..FoldCache::default()
            };
        }
    }

    /// The premultiplied plate for `layer`'s pixels `px`: the cached one
    /// when the pixels are the same allocation, else uploaded (and
    /// premultiplied unless opaque — the fold's long-standing skip,
    /// which is decided on the CPU bytes, exactly as before).
    fn plate(
        &mut self,
        batch: &mut GpuBatch<'_>,
        layer: u32,
        px: &Arc<[u8]>,
        w: u32,
        h: u32,
    ) -> Result<Resident, IngestError> {
        if let Some((cached, tex)) = self.plates.get(&layer) {
            if Arc::ptr_eq(cached, px) {
                return Ok(tex.clone());
            }
        }
        // At least a canvas of RGBA8. (A 16-bit plate is longer; the
        // fold has always read its bytes as RGBA8 and the upload takes
        // the first canvas' worth — kept byte-equal here, and recorded
        // as a defect of the 16-bit stack, not changed in passing.)
        if px.len() < (w as usize) * (h as usize) * 4 {
            return Err(IngestError::Decode(format!(
                "layer plate is {} bytes for {w}×{h}",
                px.len()
            )));
        }
        let straight = rgba8_to_f16(px);
        let tex = if window_is_opaque(&straight) {
            batch.upload(w, h, TexFormat::Rgba16Float, &straight)
        } else {
            let src = batch.upload(w, h, TexFormat::Rgba16Float, &straight);
            let out = batch.target(w, h);
            batch
                .dispatch(
                    &CAST_PREMULTIPLY,
                    &[&src],
                    CastPremultiplyParams::new().as_bytes(),
                    None,
                    &out,
                )
                .map_err(gpu_err)?;
            out
        };
        let held: u64 = self
            .plates
            .iter()
            .filter(|(id, _)| **id != layer)
            .map(|(_, (_, t))| t.bytes())
            .sum();
        if held + tex.bytes() <= PLATE_CACHE_BYTES {
            self.plates.insert(layer, (Arc::clone(px), tex.clone()));
        } else {
            self.plates.remove(&layer);
        }
        Ok(tex)
    }

    /// The r16float mask for a step, cached per layer by what built it.
    fn mask(
        &mut self,
        batch: &mut GpuBatch<'_>,
        layer: u32,
        cov: &CovKey,
        w: u32,
        h: u32,
    ) -> Option<Resident> {
        if cov.is_none() {
            return None;
        }
        if let Some((key, tex)) = self.masks.get(&layer) {
            if key.same(cov) {
                return Some(tex.clone());
            }
        }
        let bytes = cov.mask_bytes(w, h)?;
        let tex = batch.upload(w, h, TexFormat::R16Float, &bytes);
        self.masks.insert(layer, (cov.clone(), tex.clone()));
        Some(tex)
    }

    /// Forget plates and masks of layers the latest plan no longer has.
    fn retain(&mut self, steps: &[Step]) {
        let live: Vec<u32> = steps
            .iter()
            .filter_map(|s| match s {
                Step::Pixel { layer, .. } => Some(*layer),
                _ => None,
            })
            .collect();
        self.plates.retain(|id, _| live.contains(id));
        self.masks.retain(|id, _| live.contains(id));
    }
}

impl LayerStack {
    /// Plan the fold: the steps it runs, bottom-up, and where the
    /// checkpoint goes (before the first contributing layer at or above
    /// the active one). The SAME walk the fold has always made — group
    /// opening and closing, the hidden-group skip, the clip base — just
    /// recorded instead of executed.
    pub(super) fn plan_fold(&self, plates: &[(usize, Arc<[u8]>)]) -> (Vec<Step>, usize) {
        let mut steps = Vec::new();
        let mut open: Vec<u32> = Vec::new();
        let mut clip_base: Option<Arc<[u8]>> = None;
        let mut checkpoint = None;
        for (index, px) in plates {
            let layer = &self.layers[*index];
            if *index >= self.active && checkpoint.is_none() {
                checkpoint = Some(steps.len());
            }
            let want = self.chain_of(layer.group);
            if want.iter().any(|g| !self.group_enabled(*g)) {
                continue;
            }
            while let Some(gid) = open.last() {
                if want.contains(gid) {
                    break;
                }
                let gid = open.pop().expect("non-empty");
                steps.push(self.close_step(gid));
                clip_base = None;
            }
            for gid in want.iter().copied() {
                if open.contains(&gid) {
                    continue;
                }
                let isolates = self
                    .groups
                    .iter()
                    .find(|g| g.id == gid)
                    .is_some_and(|g| g.isolates());
                if isolates {
                    open.push(gid);
                    steps.push(Step::Open { group: gid });
                    clip_base = None;
                }
            }
            if layer.clipped {
                if clip_base.is_none() {
                    continue;
                }
            } else if layer.is_pixels() {
                clip_base = Some(Arc::clone(px));
            }
            let cov = CovKey {
                own: layer.live_mask().cloned(),
                clip: if layer.clipped {
                    clip_base.clone()
                } else {
                    None
                },
            };
            if let Some(params) = layer.adjust_params() {
                steps.push(Step::Adjust {
                    params: Box::new(params.clone()),
                    cov,
                });
                continue;
            }
            steps.push(Step::Pixel {
                layer: layer.id,
                px: Arc::clone(px),
                blend: layer.blend,
                opacity: layer.opacity,
                cov,
            });
        }
        let checkpoint = checkpoint.unwrap_or(steps.len());
        while let Some(gid) = open.pop() {
            steps.push(self.close_step(gid));
        }
        (steps, checkpoint)
    }

    fn close_step(&self, gid: u32) -> Step {
        Step::Close {
            group: gid,
            blend: self
                .groups
                .iter()
                .find(|g| g.id == gid)
                .map(|g| (g.blend, g.opacity)),
        }
    }

    /// Run `steps[from..]` from `state`, recording into `batch` and
    /// flushing it only where an adjustment layer needs the accumulator
    /// on the CPU. Returns the final state and, when the checkpoint index
    /// was passed on the way, the state there.
    #[allow(clippy::too_many_arguments)]
    async fn run_steps<'c>(
        &self,
        ctx: &'c GpuContext,
        cache: &mut FoldCache,
        batch: &mut GpuBatch<'c>,
        steps: &[Step],
        from: usize,
        mut st: FoldState,
        checkpoint: usize,
    ) -> Result<(FoldState, Option<FoldState>), IngestError> {
        let (w, h) = (self.width, self.height);
        let mut snap = None;
        let mut folded = 0u64;
        for (i, step) in steps.iter().enumerate().skip(from) {
            if i == checkpoint {
                snap = Some(st.clone());
            }
            if step.is_layer() {
                folded += 1;
            }
            match step {
                Step::Open { .. } => {
                    let acc = st.acc.take();
                    st.parked.push(acc);
                }
                Step::Close { blend, .. } => {
                    let inner = st.acc.take();
                    let parked = st.parked.pop().flatten();
                    st.acc = match blend {
                        None => parked,
                        Some((def, opacity)) => {
                            let a = parked.unwrap_or_else(|| ctx.zeros(w, h));
                            let b = inner.unwrap_or_else(|| ctx.zeros(w, h));
                            let out = batch.target(w, h);
                            batch
                                .dispatch(
                                    def,
                                    &[&a, &b],
                                    ComposeParams::new(*opacity).as_bytes(),
                                    None,
                                    &out,
                                )
                                .map_err(gpu_err)?;
                            Some(out)
                        }
                    };
                }
                Step::Pixel {
                    layer,
                    px,
                    blend,
                    opacity,
                    cov,
                } => {
                    let plate = cache.plate(batch, *layer, px, w, h)?;
                    let mask = cache.mask(batch, *layer, cov, w, h);
                    let acc = st.acc.take().unwrap_or_else(|| ctx.zeros(w, h));
                    let out = batch.target(w, h);
                    batch
                        .dispatch(
                            blend,
                            &[&acc, &plate],
                            ComposeParams::new(*opacity).as_bytes(),
                            mask.as_ref(),
                            &out,
                        )
                        .map_err(gpu_err)?;
                    st.acc = Some(out);
                }
                Step::Adjust { params, cov } => {
                    // The adjust chain runs through the pipeline on CPU
                    // bytes, so the accumulator comes back here — exactly
                    // as it did before, and only for this step.
                    let ticket = st.acc.as_ref().map(|a| batch.read(a));
                    let done = std::mem::replace(batch, GpuBatch::new(ctx));
                    let mut reads = done.finish_async().await.map_err(gpu_err)?;
                    let acc = match ticket {
                        Some(t) => std::mem::take(&mut reads[t.0]),
                        None => vec![0u8; (w as usize) * (h as usize) * 8],
                    };
                    let straight = super::unpremultiply(ctx, &acc, w, h).await?;
                    let adjusted =
                        crate::ingest::adjust_f16(ctx, w, h, &straight, params, cov.coverage(w, h))
                            .await?;
                    let premul = super::premultiply(ctx, &adjusted, w, h).await?;
                    st.acc = Some(batch.upload(w, h, TexFormat::Rgba16Float, &premul));
                }
            }
        }
        if checkpoint == steps.len() && checkpoint >= from {
            snap = Some(st.clone());
        }
        crate::counters::bump(|c| c.layers_folded += folded);
        Ok((st, snap))
    }

    /// The resident composite (the caller has already handled the
    /// trivial stacks).
    pub(super) async fn composite_resident(
        &self,
        ctx: &GpuContext,
        plates: &[(usize, Arc<[u8]>)],
    ) -> Result<Arc<[u8]>, IngestError> {
        let (w, h) = (self.width, self.height);
        let (steps, checkpoint) = self.plan_fold(plates);
        // Taken out for the awaits (no lock is held across a suspension
        // point) and put back after; a failed composite leaves it empty,
        // which only costs the next composite its uploads.
        let mut cache = std::mem::take(&mut *self.fold.lock().expect("fold cache lock"));
        cache.bind(ctx);

        // Nothing changed since the last composite: its result stands.
        if let Some((last_steps, out)) = &cache.last {
            if same_steps(last_steps, &steps) {
                let out = Arc::clone(out);
                *self.fold.lock().expect("fold cache lock") = cache;
                return Ok(out);
            }
        }

        // Resume from the checkpoint when everything below it is unchanged.
        let (from, state) = match &cache.checkpoint {
            Some((prefix, state))
                if prefix.len() <= steps.len() && same_steps(prefix, &steps[..prefix.len()]) =>
            {
                (prefix.len(), state.clone())
            }
            _ => (0, FoldState::default()),
        };

        let mut batch = GpuBatch::new(ctx);
        let (st, snap) = self
            .run_steps(ctx, &mut cache, &mut batch, &steps, from, state, checkpoint)
            .await?;
        if let Some(snap) = snap {
            cache.checkpoint = Some((steps[..checkpoint].to_vec(), snap));
        }

        // Out of premultiplied space on the device, then the one readback.
        let acc = st.acc.unwrap_or_else(|| ctx.zeros(w, h));
        let out = batch.target(w, h);
        batch
            .dispatch(
                &CAST_UNPREMULTIPLY,
                &[&acc],
                CastUnpremultiplyParams::new().as_bytes(),
                None,
                &out,
            )
            .map_err(gpu_err)?;
        let ticket = batch.read(&out);
        let reads = batch.finish_async().await.map_err(gpu_err)?;
        let rgba: Arc<[u8]> = Arc::from(f16_to_rgba8(&reads[ticket.0]).into_boxed_slice());

        cache.retain(&steps);
        cache.last = Some((steps, Arc::clone(&rgba)));
        *self.fold.lock().expect("fold cache lock") = cache;
        Ok(rgba)
    }
}
