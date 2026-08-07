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

//! The PAINT SESSION — one in-flight brush stroke, behind the wasm
//! `brush_stroke_*` doors.
//!
//! # Scope: a stroke lands in the ACTIVE LAYER
//!
//! A stroke paints ONE buffer, and which buffer that is comes from the
//! caller: [`StrokeSession::begin_on`] takes the pixels directly, so the
//! wasm doors hand it the ACTIVE LAYER of the bound [`crate::layers`]
//! stack. The layers below and above are untouched, the composite is
//! re-folded from the stack on commit, and the edit is journaled — so a
//! committed stroke is undoable, tile-granularly, within the journal's
//! stated bound.
//!
//! What remains true and is not papered over:
//!
//! * The stroke is still destructive INTO ITS OWN LAYER: painting on the
//!   background layer covers what was there. Painting on an empty layer
//!   above it does not, which is the point of having a stack.
//! * The DOCUMENT and the source file are never touched (the in-frame
//!   result is a preview layer), and undo is bounded — see
//!   `image_graph::journal` for the depth/byte budget and exactly what
//!   happens at it.
//!
//! # The gesture / commit split
//!
//! [`StrokeSession::begin`] snapshots the base pixels; every
//! [`StrokeSession::extend`] stamps the newly-planned dabs into the
//! accumulator and re-composites ONLY the dirty rectangle FROM THE BASE.
//! Because the composite always starts from the base, extending is
//! idempotent and order-free: the pixels after N incremental extends are
//! the pixels a single from-scratch composite of the final accumulator
//! would produce (asserted by `image-conformance`'s
//! `brush_incremental_matches_from_scratch`). `commit` hands back the
//! working pixels; `cancel` throws them away and the base is still
//! bit-identical.

use std::sync::Arc;

use image_core::Region;
use image_gpu::dab::{plan_segment, BrushTip, PressureTarget, StrokeAccumulator, StrokeSample};
use image_gpu::{composite_stroke_window, GpuContext, PaintMode, SelectionCoverage};
use image_kernels::families::compose;
use image_kernels::KernelDef;

use crate::fill::{f16_to_rgba8, rgba8_to_f16};
use crate::ingest::{DecodedImage, IngestError};

/// Which painting tool a stroke is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrokeTool {
    /// Soft round tip, antialiased rim, configurable hardness.
    Brush,
    /// Hard round tip with NO antialiasing (binary coverage) and a fixed
    /// flow of 1 — the pencil's aliased edge is its defining trait.
    Pencil,
    /// Takes alpha away instead of laying colour down.
    Eraser,
    /// CLONE STAMP: the dab's paint layer is a window of the image
    /// sampled at a fixed offset from the cursor, so the stroke copies
    /// pixels from one place to another. Same tip, same spacing, same
    /// pressure mapping, same selection masking — only the source of the
    /// paint differs.
    Clone,
    /// HEALING BRUSH: a clone whose source is TONE-MATCHED to the
    /// destination before compositing, so the patch takes on the
    /// surrounding brightness and colour instead of pasting a visibly
    /// different rectangle.
    ///
    /// The match is a MEAN offset over the dab's window, not a Poisson
    /// (gradient-domain) solve. That difference is real and is stated
    /// wherever this tool is named: mean-matching removes a uniform tone
    /// difference — which is what most blemish retouching is — and does
    /// not remove a GRADIENT across the patch, so healing across a
    /// strong luminance ramp still shows a seam. The Poisson solve is
    /// the follow-up, and pretending it is already here would be the one
    /// unrecoverable mistake.
    Heal,
}

impl StrokeTool {
    /// Decode the wire name (mirrored by the TS `StrokeTool` union).
    pub fn from_wire(s: &str) -> Option<StrokeTool> {
        Some(match s {
            "brush" => StrokeTool::Brush,
            "pencil" => StrokeTool::Pencil,
            "eraser" => StrokeTool::Eraser,
            "clone" => StrokeTool::Clone,
            "heal" => StrokeTool::Heal,
            _ => return None,
        })
    }

    pub fn as_wire(self) -> &'static str {
        match self {
            StrokeTool::Brush => "brush",
            StrokeTool::Pencil => "pencil",
            StrokeTool::Eraser => "eraser",
            StrokeTool::Clone => "clone",
            StrokeTool::Heal => "heal",
        }
    }

    /// Does this tool paint from a SAMPLED window rather than a colour?
    pub fn samples_image(self) -> bool {
        matches!(self, StrokeTool::Clone | StrokeTool::Heal)
    }

    /// Does it tone-match the sample to its destination?
    pub fn tone_matches(self) -> bool {
        matches!(self, StrokeTool::Heal)
    }

    /// The pencil is aliased; brush and eraser antialias their rim.
    pub fn antialias(self) -> bool {
        !matches!(self, StrokeTool::Pencil)
    }
}

/// Resolve a `compose.*` blend-mode wire name to its kernel.
///
/// All 26 registered blend kernels are reachable; the wire name is the
/// kernel id with the `compose.` prefix dropped.
pub fn blend_kernel(name: &str) -> Option<&'static KernelDef> {
    let id_owned;
    let id = if name.starts_with("compose.") {
        name
    } else {
        id_owned = format!("compose.{name}");
        &id_owned
    };
    compose::FAMILY.iter().copied().find(|k| k.id == id)
}

/// The wire names of every blend mode a stroke can use (the `compose.`
/// prefix dropped) — the panel's blend picker reads this list, so it can
/// never drift from the registry.
pub fn blend_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = compose::FAMILY
        .iter()
        .map(|k| k.id.strip_prefix("compose.").unwrap_or(k.id))
        .collect();
    names.sort_unstable();
    names
}

/// Everything about a stroke that is fixed for its whole duration.
///
/// Frozen at [`StrokeSession::begin`] deliberately: a stroke whose size
/// or blend mode changed halfway through would not be replayable from a
/// recorded sample list, and replay is the point (spec §6.3).
#[derive(Debug, Clone, Copy)]
pub struct StrokeParams {
    pub tool: StrokeTool,
    /// Tip diameter in IMAGE pixels (before pressure scaling).
    pub size: f32,
    /// Fully-opaque fraction of the radius, `0..1`. Ignored by the
    /// pencil (which is binary by construction).
    pub hardness: f32,
    /// The ceiling the whole stroke composites at, `0..1`.
    pub opacity: f32,
    /// How much each dab deposits into the accumulator, `0..1`.
    pub flow: f32,
    /// Dab spacing as a fraction of the tip DIAMETER (Photoshop's
    /// convention). `0.25` = a dab every quarter-diameter.
    pub spacing: f32,
    /// The `compose.*` kernel paint goes down through. Ignored by the
    /// eraser (which is `band.set_alpha`, not a blend).
    pub blend: &'static KernelDef,
    /// Straight RGBA in `[0, 1]`. Ignored by the eraser.
    pub color: [f32; 4],
    /// Which property the pen's pressure drives.
    pub pressure: PressureTarget,
}

impl StrokeParams {
    /// The v0 defaults the panel starts from: a 24 px half-hard round
    /// brush at full opacity and flow, quarter-diameter spacing, normal
    /// blend, opaque black, pressure driving size AND opacity.
    pub fn defaults(tool: StrokeTool) -> StrokeParams {
        StrokeParams {
            tool,
            size: 24.0,
            hardness: 0.5,
            opacity: 1.0,
            flow: 1.0,
            spacing: 0.25,
            blend: &compose::COMPOSE_NORMAL,
            color: [0.0, 0.0, 0.0, 1.0],
            pressure: PressureTarget::Both,
        }
    }

    /// Clamp to the ranges the engine promises. Called once at `begin`,
    /// so every downstream user sees sane values.
    pub fn sanitized(mut self) -> StrokeParams {
        self.size = self.size.clamp(MIN_SIZE_PX, MAX_SIZE_PX);
        self.hardness = self.hardness.clamp(0.0, 1.0);
        self.opacity = self.opacity.clamp(0.0, 1.0);
        // The pencil deposits fully per dab — a partial-flow pencil would
        // produce the soft build-up the aliased tip exists to avoid.
        self.flow = if self.tool == StrokeTool::Pencil {
            1.0
        } else {
            self.flow.clamp(0.0, 1.0)
        };
        self.spacing = self
            .spacing
            .clamp(MIN_SPACING_FRACTION, MAX_SPACING_FRACTION);
        for c in &mut self.color {
            *c = c.clamp(0.0, 1.0);
        }
        self
    }

    /// The tip for a dab at `pressure` (the size half of the pressure
    /// mapping; the flow half is [`Self::flow_at`]).
    pub fn tip_at(&self, pressure: f32) -> BrushTip {
        let d = (self.size * self.pressure.size_scale(pressure)).max(0.0);
        BrushTip {
            diameter: d,
            hardness: self.hardness,
            antialias: self.tool.antialias(),
        }
    }

    /// The per-dab flow at `pressure`.
    pub fn flow_at(&self, pressure: f32) -> f32 {
        self.flow * self.pressure.flow_scale(pressure)
    }

    /// Dab spacing in px of arc length — a fraction of the FULL
    /// diameter, not the pressure-scaled one, so a pen's thin passages
    /// stay as densely stamped as its thick ones (spacing that shrank
    /// with pressure would thin out exactly where the stroke is faintest).
    pub fn step_px(&self) -> f32 {
        self.size * self.spacing
    }

    /// The GPU paint mode for the tools whose paint layer is GENERATED.
    ///
    /// `None` for the sampling tools (clone, heal), whose paint layer is
    /// a window of the image and therefore cannot be produced without
    /// the region being composited — [`StrokeSession::composite`] builds
    /// those. An exhaustive match, deliberately: a catch-all here once
    /// made clone and heal silently paint a solid colour, which looked
    /// like a broken clone rather than an unhandled tool.
    pub fn solid_paint_mode(&self) -> Option<PaintMode<'static>> {
        match self.tool {
            StrokeTool::Eraser => Some(PaintMode::Erase),
            StrokeTool::Brush | StrokeTool::Pencil => Some(PaintMode::Paint {
                blend: self.blend,
                color: self.color,
            }),
            StrokeTool::Clone | StrokeTool::Heal => None,
        }
    }
}

/// Where a cloning stroke reads its pixels from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloneSource {
    /// The anchor the user set (alt-click), in image px.
    pub x: f32,
    pub y: f32,
    /// ALIGNED: the source keeps a fixed offset from the cursor, so it
    /// tracks the brush and a released-and-resumed stroke continues the
    /// copy. UNALIGNED restarts from the anchor on every stroke, which
    /// is how a repeated stamp of one motif is done.
    pub aligned: bool,
}

/// Smallest tip diameter (px). Below this a dab covers nothing at all.
pub const MIN_SIZE_PX: f32 = 0.5;
/// Largest tip diameter (px) — a guard on the per-dab stamp cost, not a
/// creative limit anyone will hit.
pub const MAX_SIZE_PX: f32 = 2048.0;
/// Densest allowed spacing, as a fraction of the diameter.
pub const MIN_SPACING_FRACTION: f32 = 0.01;
/// Sparsest allowed spacing (dabs stop overlapping well before this).
pub const MAX_SPACING_FRACTION: f32 = 4.0;

/// One in-flight stroke.
///
/// # What came from `image-graph`, and what is still open
///
/// Engine B has exactly the right SHAPE for this: a `gesture` (ephemeral
/// param override) / `set_params` (committed) split, per-node sparse
/// tile caches keyed on `(params_hash, input_generations)`, and damage
/// propagation. A stroke in progress is the textbook gesture and a
/// released stroke the textbook commit.
///
/// Paint is not a PARAMETER change though, it is a WRITE — so the piece
/// that was actually needed was the COW undo journal, and that now
/// EXISTS (`image_graph::journal`). The stroke's COMMIT rides it: the
/// caller records the stroke's damage tiles into the bound layer stack's
/// journal, which snapshots `Arc` handles rather than pixels. What is
/// still open, and is recorded here rather than half-built:
///
/// 1. **A tiled accumulation buffer** — so the coverage field is sparse
///    over the stroke's tiles instead of dense over the image, and so
///    `begin`'s working buffer need not be a full-image clone.
/// 2. **A resident output texture** — so `extend` does not read a window
///    back to the CPU at all (the same Stage-B dependency the composite
///    round-trip has; see the door docs in `lib.rs`).
///
/// Neither is required for correctness, and both are invisible to this
/// type's callers.
pub struct StrokeSession {
    /// The engine handle the stroke started from.
    handle: u32,
    width: u32,
    height: u32,
    /// The untouched pixels at `begin` — every composite derives from
    /// these, and `cancel` restores them by simply being dropped.
    base: Arc<[u8]>,
    /// The live painted pixels (straight RGBA8).
    working: Vec<u8>,
    accumulator: StrokeAccumulator,
    params: StrokeParams,
    /// The selection FROZEN at `begin`. Freezing it means a stroke can
    /// never half-honour a selection that changed mid-drag.
    selection: Option<Arc<SelectionCoverage>>,
    last: Option<StrokeSample>,
    walk: image_gpu::StrokeWalk,
    /// The clone/heal anchor, and the offset it resolves to once the
    /// stroke's first sample fixes the relationship. `None` for the
    /// non-sampling tools.
    clone_source: Option<CloneSource>,
    /// `source − cursor`, resolved at the FIRST sample and then fixed
    /// for the stroke. Fixing it at the first sample is what makes an
    /// aligned clone track the brush rigidly; recomputing per dab would
    /// let the offset drift with the cursor and smear the copy.
    clone_offset: Option<(f32, f32)>,
}

impl StrokeSession {
    /// Open a stroke on `image`. `selection` is the session's coverage
    /// (already gated on the handle by the caller); `None` means
    /// unclipped.
    pub fn begin(
        handle: u32,
        image: &DecodedImage,
        params: StrokeParams,
        selection: Option<Arc<SelectionCoverage>>,
    ) -> Result<StrokeSession, IngestError> {
        StrokeSession::begin_on(
            handle,
            image.width,
            image.height,
            Arc::clone(&image.rgba),
            params,
            selection,
        )
    }

    /// Open a stroke on RAW PIXELS — the LAYER lane. `base` is
    /// canvas-extent straight RGBA8 and is the buffer the stroke paints;
    /// the wasm doors pass the bound stack's ACTIVE LAYER, so a stroke
    /// lands there and not in the flattened image.
    ///
    /// `handle` still identifies the engine-held IMAGE the stroke belongs
    /// to (the selection is gated on it, and the commit re-composites
    /// it) — the layer is which buffer, the handle is which document.
    pub fn begin_on(
        handle: u32,
        width: u32,
        height: u32,
        base: Arc<[u8]>,
        params: StrokeParams,
        selection: Option<Arc<SelectionCoverage>>,
    ) -> Result<StrokeSession, IngestError> {
        if width == 0 || height == 0 {
            return Err(IngestError::Unsupported(
                "cannot paint on an empty image".into(),
            ));
        }
        let want = (width as usize) * (height as usize) * 4;
        if base.len() != want {
            return Err(IngestError::Decode(format!(
                "paint base is {} bytes for {width}×{height} (expected {want})",
                base.len()
            )));
        }
        Ok(StrokeSession {
            handle,
            width,
            height,
            working: base.to_vec(),
            base,
            accumulator: StrokeAccumulator::new(width, height),
            params: params.sanitized(),
            selection,
            last: None,
            walk: image_gpu::StrokeWalk::new(),
            clone_source: None,
            clone_offset: None,
        })
    }

    pub fn handle(&self) -> u32 {
        self.handle
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn params(&self) -> &StrokeParams {
        &self.params
    }

    /// Dabs stamped so far.
    /// Point a cloning stroke at its source. Must be called BEFORE the
    /// first sample: the offset is fixed at the first dab, and moving
    /// the anchor mid-stroke would tear the copy.
    pub fn set_clone_source(&mut self, source: CloneSource) {
        self.clone_source = Some(source);
    }

    pub fn clone_source(&self) -> Option<CloneSource> {
        self.clone_source
    }

    pub fn dab_count(&self) -> u64 {
        self.accumulator.dab_count()
    }

    /// The stroke's bounding box so far, or `None` before the first dab
    /// lands on the canvas.
    pub fn stroke_bounds(&self) -> Option<Region> {
        self.accumulator.stroke_bounds()
    }

    /// The live painted pixels (straight RGBA8) — the Stage-A composite
    /// payload.
    pub fn pixels(&self) -> &[u8] {
        &self.working
    }

    /// Plan the dabs a new sample adds, WITHOUT stamping them. Split out
    /// so the spacing walk is testable on its own and so `extend` reads
    /// as "plan, stamp, composite".
    fn plan(&mut self, sample: StrokeSample) -> Vec<image_gpu::Dab> {
        let mut dabs = Vec::new();
        match self.last {
            // The first sample stamps a single dab where the pointer went
            // down — a click paints a dot, exactly as it should.
            None => dabs.push(image_gpu::Dab {
                x: sample.x,
                y: sample.y,
                pressure: sample.pressure,
            }),
            Some(prev) => plan_segment(
                &mut self.walk,
                prev,
                sample,
                self.params.step_px(),
                &mut dabs,
            ),
        }
        self.last = Some(sample);
        dabs
    }

    /// Add a pointer sample: plan its dabs, stamp them, and re-composite
    /// the dirty rectangle. Returns `true` when pixels changed.
    pub async fn extend(
        &mut self,
        ctx: &GpuContext,
        sample: StrokeSample,
    ) -> Result<bool, IngestError> {
        if !sample.x.is_finite() || !sample.y.is_finite() {
            return Ok(false);
        }
        // Fix the clone offset on the FIRST sample and never again. An
        // offset recomputed per dab would follow the cursor and smear
        // the copy instead of translating it.
        if self.params.tool.samples_image() && self.clone_offset.is_none() {
            if let Some(src) = self.clone_source {
                self.clone_offset = Some((src.x - sample.x, src.y - sample.y));
            }
        }
        let dabs = self.plan(sample);
        for d in &dabs {
            let tip = self.params.tip_at(d.pressure);
            let flow = self.params.flow_at(d.pressure);
            self.accumulator.stamp(&tip, d.x, d.y, flow);
        }
        let Some(dirty) = self.accumulator.take_dirty() else {
            return Ok(false);
        };
        self.composite(ctx, dirty).await?;
        Ok(true)
    }

    /// Re-derive `region`'s pixels FROM THE BASE under the current
    /// accumulated coverage, and splice the result into `working`.
    async fn composite(&mut self, ctx: &GpuContext, region: Region) -> Result<(), IngestError> {
        let Some(region) = region.intersect(Region::new(0, 0, self.width, self.height)) else {
            return Ok(());
        };
        let (w, h) = (region.w, region.h);
        let base_window = self.window_rgba8(region);
        let base_f16 = rgba8_to_f16(&base_window);
        let mask = self.accumulator.mask_window_f16(
            region,
            self.params.opacity,
            self.selection.as_deref(),
        );
        // The sampling tools build their paint layer HERE, because it is
        // a window of the image and cannot exist before the region does.
        let sampled = match self.params.solid_paint_mode() {
            Some(_) => None,
            None => {
                let Some((dx, dy)) = self.clone_offset else {
                    // No anchor set: a clone with nowhere to read from
                    // deposits NOTHING rather than falling back to a
                    // colour. Silently painting black would look like a
                    // broken clone; painting nothing looks like what it
                    // is.
                    return Ok(());
                };
                let src_window = self.source_window_rgba8(region, dx, dy);
                // HEAL solves for a correction FIELD over the dab's own
                // footprint: the coverage IS the region Ω, and everything
                // the dab does not touch is the boundary the field
                // interpolates between. Clone passes `None` and the
                // `math.add` dispatch drops out entirely.
                let correction = if self.params.tool.tone_matches() {
                    self.heal_field(region, dx, dy)
                } else {
                    None
                };
                Some((rgba8_to_f16(&src_window), correction))
            }
        };
        let mode = match (&self.params.solid_paint_mode(), &sampled) {
            (Some(m), _) => *m,
            (None, Some((src_f16, correction))) => PaintMode::Sample {
                blend: self.params.blend,
                source_f16: src_f16,
                correction_f16: correction.as_deref(),
            },
            (None, None) => return Ok(()),
        };
        let out_f16 = composite_stroke_window(ctx, &mode, &base_f16, &mask, w, h)
            .await
            .map_err(|e| IngestError::Pipeline(e.to_string()))?;
        let out = f16_to_rgba8(&out_f16);
        self.splice(region, &out);
        Ok(())
    }

    /// The healing correction for `region`, solved on an EXPANDED window
    /// and cropped back.
    ///
    /// The expansion is not an optimisation, it is the whole reason the
    /// solve works: a dab's dirty region IS the dab, so inside it there
    /// is no boundary to interpolate FROM. The first version solved on
    /// the dirty region alone, found no boundary, and silently produced
    /// no correction at all — a healing brush that behaved exactly like
    /// a clone. The margin brings real surrounding image into the window,
    /// which is precisely what the Poisson formulation asks for.
    fn heal_field(&self, region: Region, dx: f32, dy: f32) -> Option<Vec<u8>> {
        // Enough boundary for the membrane to be interpolating rather
        // than extrapolating, without making the per-dab solve grow with
        // the tip: the field is smooth, so a few pixels of context carry
        // the tone difference.
        const MARGIN: i32 = 6;
        let big = Region::new(
            region.x - MARGIN,
            region.y - MARGIN,
            region.w + 2 * MARGIN as u32,
            region.h + 2 * MARGIN as u32,
        )
        .intersect(Region::new(0, 0, self.width, self.height))?;
        let (bw, bh) = (big.w as usize, big.h as usize);

        let dest = self.window_rgba8(big);
        let src = self.source_window_rgba8(big, dx, dy);
        let inside: Vec<u8> = (0..bw * bh)
            .map(|i| {
                let x = big.x + (i % bw) as i32;
                let y = big.y + (i / bw) as i32;
                if x < 0 || y < 0 {
                    return 0;
                }
                let cov = self.accumulator.effective_at(
                    x as u32,
                    y as u32,
                    1.0,
                    self.selection.as_deref(),
                );
                if cov > 0.0 {
                    255
                } else {
                    0
                }
            })
            .collect();
        let field = crate::heal::correction_field(&dest, &src, &inside, bw, bh)?;

        // Crop back to the region the composite is actually writing.
        let mut out = Vec::with_capacity((region.w as usize) * (region.h as usize) * 3);
        for y in 0..region.h {
            let by = (region.y - big.y) + y as i32;
            for x in 0..region.w {
                let bx = (region.x - big.x) + x as i32;
                let i = (by as usize * bw + bx as usize) * 3;
                out.extend_from_slice(&field[i..i + 3]);
            }
        }
        Some(crate::heal::field_to_f16(
            &out,
            (region.w as usize) * (region.h as usize),
        ))
    }

    /// Copy `region` SHIFTED BY `(dx, dy)` out of the base pixels — the
    /// clone/heal source.
    ///
    /// Out-of-bounds reads yield TRANSPARENT BLACK, not a clamped edge
    /// pixel. Clamping would smear the border across the canvas and look
    /// like a rendering bug; transparency composites to "nothing was
    /// copied here", which is the truth.
    fn source_window_rgba8(&self, region: Region, dx: f32, dy: f32) -> Vec<u8> {
        let (dx, dy) = (dx.round() as i64, dy.round() as i64);
        let mut out = vec![0u8; (region.w as usize) * (region.h as usize) * 4];
        for y in 0..region.h {
            let sy = region.y as i64 + y as i64 + dy;
            if sy < 0 || sy >= self.height as i64 {
                continue;
            }
            for x in 0..region.w {
                let sx = region.x as i64 + x as i64 + dx;
                if sx < 0 || sx >= self.width as i64 {
                    continue;
                }
                let si = ((sy as usize) * self.width as usize + sx as usize) * 4;
                let di = ((y as usize) * region.w as usize + x as usize) * 4;
                out[di..di + 4].copy_from_slice(&self.base[si..si + 4]);
            }
        }
        out
    }

    /// Copy `region` out of the BASE pixels as tightly packed RGBA8.
    fn window_rgba8(&self, region: Region) -> Vec<u8> {
        let mut out = Vec::with_capacity((region.w as usize) * (region.h as usize) * 4);
        for y in 0..region.h {
            let row = (region.y as u32 + y) as usize;
            let start = (row * self.width as usize + region.x as usize) * 4;
            let end = start + (region.w as usize) * 4;
            out.extend_from_slice(&self.base[start..end]);
        }
        out
    }

    /// Write `window` back into `working` at `region` — but ONLY where
    /// the effective coverage is non-zero.
    ///
    /// The guard is not an optimization, it is the byte-level guarantee.
    /// The ABI already returns the backdrop wherever the mask is 0, so
    /// those texels are unchanged in VALUE; skipping the write keeps them
    /// unchanged in BYTES too, past the f16 round-trip the composite
    /// otherwise puts every texel of the dirty rectangle through. That is
    /// what makes "a brush cannot paint outside the selection" an exact
    /// claim about bytes rather than an approximate one about colours —
    /// including in the rectangle's corners, which the dab's own falloff
    /// never reaches either.
    fn splice(&mut self, region: Region, window: &[u8]) {
        for y in 0..region.h {
            let iy = region.y as u32 + y;
            for x in 0..region.w {
                let ix = region.x as u32 + x;
                let eff = self.accumulator.effective_at(
                    ix,
                    iy,
                    self.params.opacity,
                    self.selection.as_deref(),
                );
                if eff <= 0.0 {
                    continue;
                }
                let src = ((y * region.w + x) as usize) * 4;
                let dst = ((iy * self.width + ix) as usize) * 4;
                self.working[dst..dst + 4].copy_from_slice(&window[src..src + 4]);
            }
        }
    }

    /// Finish: hand back the painted pixels. The caller writes them into
    /// the ACTIVE LAYER — journaling the tiles [`Self::stroke_bounds`]
    /// covers — and re-composites the stack.
    pub fn commit(self) -> Vec<u8> {
        self.working
    }

    /// True when the stroke deposited nothing (a click outside the
    /// canvas, or a fully-clipped stroke) — the caller can then skip the
    /// handle swap entirely.
    pub fn is_empty(&self) -> bool {
        self.accumulator.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32) -> DecodedImage {
        DecodedImage::from_rgba8(w, h, vec![128u8; (w * h * 4) as usize]).expect("valid")
    }

    fn params(tool: StrokeTool) -> StrokeParams {
        StrokeParams::defaults(tool)
    }

    // ── wire decoding ────────────────────────────────────────────────

    #[test]
    fn image_editor_paint_tool_round_trips_its_wire_name() {
        for t in [
            StrokeTool::Brush,
            StrokeTool::Pencil,
            StrokeTool::Eraser,
            StrokeTool::Clone,
            StrokeTool::Heal,
        ] {
            assert_eq!(StrokeTool::from_wire(t.as_wire()), Some(t));
        }
        assert_eq!(StrokeTool::from_wire("airbrush"), None);
    }

    #[test]
    fn image_editor_paint_only_the_pencil_is_aliased() {
        assert!(StrokeTool::Brush.antialias());
        assert!(StrokeTool::Eraser.antialias());
        assert!(!StrokeTool::Pencil.antialias());
    }

    #[test]
    fn image_editor_paint_every_registered_blend_mode_is_reachable() {
        let names = blend_names();
        assert_eq!(names.len(), 26, "the compose family is 26 kernels");
        for n in &names {
            let k = blend_kernel(n).unwrap_or_else(|| panic!("`{n}` should resolve"));
            assert_eq!(k.id, format!("compose.{n}"));
        }
        // The fully-qualified id works too, and nonsense does not.
        assert_eq!(
            blend_kernel("compose.multiply").map(|k| k.id),
            Some("compose.multiply")
        );
        assert!(blend_kernel("dissolve").is_none());
        assert!(blend_kernel("").is_none());
    }

    // ── params ───────────────────────────────────────────────────────

    #[test]
    fn image_editor_paint_defaults_are_the_documented_v0() {
        let p = params(StrokeTool::Brush);
        assert_eq!(p.size, 24.0);
        assert_eq!(p.hardness, 0.5);
        assert_eq!(p.opacity, 1.0);
        assert_eq!(p.flow, 1.0);
        assert_eq!(p.spacing, 0.25);
        assert_eq!(p.blend.id, "compose.normal");
        assert_eq!(p.color, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(p.pressure, PressureTarget::Both);
        assert_eq!(p.step_px(), 6.0, "a dab every quarter diameter");
    }

    #[test]
    fn image_editor_paint_sanitize_clamps_every_range() {
        let mut p = params(StrokeTool::Brush);
        p.size = -5.0;
        p.hardness = 3.0;
        p.opacity = 9.0;
        p.flow = -1.0;
        p.spacing = 0.0;
        p.color = [2.0, -1.0, 0.5, 7.0];
        let s = p.sanitized();
        assert_eq!(s.size, MIN_SIZE_PX);
        assert_eq!(s.hardness, 1.0);
        assert_eq!(s.opacity, 1.0);
        assert_eq!(s.flow, 0.0);
        assert_eq!(s.spacing, MIN_SPACING_FRACTION);
        assert_eq!(s.color, [1.0, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn image_editor_paint_the_pencil_always_deposits_at_full_flow() {
        let mut p = params(StrokeTool::Pencil);
        p.flow = 0.1;
        assert_eq!(
            p.sanitized().flow,
            1.0,
            "a partial-flow pencil is not a pencil"
        );
        // …but its OPACITY is still honoured (that is the stroke ceiling).
        let mut p = params(StrokeTool::Pencil);
        p.opacity = 0.3;
        assert_eq!(p.sanitized().opacity, 0.3);
    }

    #[test]
    fn image_editor_paint_pressure_maps_to_size_and_flow_by_default() {
        let p = params(StrokeTool::Brush);
        assert_eq!(p.tip_at(1.0).diameter, 24.0);
        assert_eq!(p.tip_at(0.5).diameter, 12.0);
        assert_eq!(p.flow_at(0.5), 0.5);

        let mut size_only = p;
        size_only.pressure = PressureTarget::Size;
        assert_eq!(size_only.tip_at(0.5).diameter, 12.0);
        assert_eq!(size_only.flow_at(0.5), 1.0);

        let mut none = p;
        none.pressure = PressureTarget::None;
        assert_eq!(none.tip_at(0.5).diameter, 24.0);
        assert_eq!(none.flow_at(0.5), 1.0);
    }

    #[test]
    fn image_editor_paint_spacing_does_not_shrink_with_pressure() {
        // The step is a fraction of the FULL diameter, so a pen's faint
        // passages stay as densely stamped as its heavy ones.
        let p = params(StrokeTool::Brush);
        assert_eq!(p.step_px(), 6.0);
        assert_eq!(p.tip_at(0.25).diameter, 6.0);
        assert_eq!(p.step_px(), 6.0, "unchanged by the pressure-scaled tip");
    }

    #[test]
    fn image_editor_paint_mode_picks_erase_for_the_eraser_only() {
        assert!(matches!(
            params(StrokeTool::Eraser).solid_paint_mode().unwrap(),
            PaintMode::Erase
        ));
        assert_eq!(
            params(StrokeTool::Brush)
                .solid_paint_mode()
                .unwrap()
                .kernel_ids(),
            vec![
                "gen.solid",
                "cast.premultiply",
                "compose.normal",
                "cast.unpremultiply"
            ]
        );
        assert_eq!(
            params(StrokeTool::Pencil)
                .solid_paint_mode()
                .unwrap()
                .kernel_ids()
                .len(),
            4,
            "the pencil paints through the same lane; only its TIP differs"
        );
    }

    // ── session bookkeeping (no GPU) ─────────────────────────────────

    #[test]
    fn image_editor_paint_begin_rejects_an_empty_image() {
        let empty = DecodedImage::from_rgba8(0, 0, Vec::new());
        if let Ok(e) = empty {
            assert!(StrokeSession::begin(1, &e, params(StrokeTool::Brush), None).is_err());
        }
    }

    #[test]
    fn image_editor_paint_begin_on_takes_layer_pixels_and_checks_their_extent() {
        // The LAYER lane: a stroke opens on raw pixels, not on the
        // engine-held image, so it can paint the active layer.
        let pixels: std::sync::Arc<[u8]> =
            std::sync::Arc::from(vec![7u8; 16 * 16 * 4].into_boxed_slice());
        let s = StrokeSession::begin_on(
            3,
            16,
            16,
            std::sync::Arc::clone(&pixels),
            params(StrokeTool::Brush),
            None,
        )
        .expect("well-sized");
        assert_eq!(s.pixels(), &pixels[..]);
        // A mis-sized buffer is a clean error, never a torn stroke.
        assert!(StrokeSession::begin_on(
            3,
            16,
            16,
            std::sync::Arc::from(vec![0u8; 4].into_boxed_slice()),
            params(StrokeTool::Brush),
            None
        )
        .is_err());
    }

    #[test]
    fn image_editor_paint_a_fresh_session_is_empty_and_mirrors_the_base() {
        let image = img(16, 16);
        let s = StrokeSession::begin(7, &image, params(StrokeTool::Brush), None).expect("begin");
        assert!(s.is_empty());
        assert_eq!(s.handle(), 7);
        assert_eq!((s.width(), s.height()), (16, 16));
        assert_eq!(s.dab_count(), 0);
        assert!(s.stroke_bounds().is_none());
        assert_eq!(s.pixels(), &image.rgba[..], "untouched until a dab lands");
    }

    #[test]
    fn image_editor_paint_the_first_sample_stamps_one_dab_at_the_pointer() {
        let image = img(32, 32);
        let mut s =
            StrokeSession::begin(1, &image, params(StrokeTool::Brush), None).expect("begin");
        let dabs = s.plan(StrokeSample::new(10.0, 10.0, 1.0));
        assert_eq!(dabs.len(), 1, "a click paints a dot");
        assert_eq!((dabs[0].x, dabs[0].y), (10.0, 10.0));
    }

    #[test]
    fn image_editor_paint_a_drag_interpolates_at_the_spacing_step() {
        let image = img(64, 64);
        let mut s =
            StrokeSession::begin(1, &image, params(StrokeTool::Brush), None).expect("begin");
        s.plan(StrokeSample::new(0.0, 0.0, 1.0));
        // 24 px tip, spacing 0.25 ⇒ step 6 px. A 60 px jump ⇒ 10 dabs, so
        // a fast drag paints a stroke, not two dots.
        let dabs = s.plan(StrokeSample::new(60.0, 0.0, 1.0));
        assert_eq!(dabs.len(), 10);
        assert!((dabs[0].x - 6.0).abs() < 1e-4);
        assert!((dabs[9].x - 60.0).abs() < 1e-4);
    }

    #[test]
    fn image_editor_paint_the_spacing_walk_carries_across_samples() {
        let image = img(64, 64);
        let mut s =
            StrokeSession::begin(1, &image, params(StrokeTool::Brush), None).expect("begin");
        s.plan(StrokeSample::new(0.0, 0.0, 1.0));
        // Four 2 px nudges bank 8 px against a 6 px step ⇒ exactly one dab.
        let mut total = 0;
        for i in 1..=4 {
            total += s.plan(StrokeSample::new(i as f32 * 2.0, 0.0, 1.0)).len();
        }
        assert_eq!(total, 1, "no clumping at every pointer event");
    }

    #[test]
    fn image_editor_paint_non_finite_samples_are_ignored_by_plan() {
        let image = img(16, 16);
        let mut s =
            StrokeSession::begin(1, &image, params(StrokeTool::Brush), None).expect("begin");
        s.plan(StrokeSample::new(4.0, 4.0, 1.0));
        let dabs = s.plan(StrokeSample::new(f32::INFINITY, 4.0, 1.0));
        assert!(dabs.is_empty());
    }

    #[test]
    fn image_editor_paint_commit_of_an_untouched_session_returns_the_base() {
        let image = img(8, 8);
        let s = StrokeSession::begin(1, &image, params(StrokeTool::Brush), None).expect("begin");
        assert_eq!(s.commit(), image.rgba.to_vec());
    }

    #[test]
    fn image_editor_paint_the_window_read_is_row_correct() {
        // A 4×4 ramp; the (1,1)–(3,3) window must be the right 9 texels.
        let mut rgba = vec![0u8; 4 * 4 * 4];
        for i in 0..16 {
            rgba[i * 4] = i as u8;
        }
        let image = DecodedImage::from_rgba8(4, 4, rgba).expect("valid");
        let s = StrokeSession::begin(1, &image, params(StrokeTool::Brush), None).expect("begin");
        let w = s.window_rgba8(Region::new(1, 1, 3, 3));
        assert_eq!(w.len(), 3 * 3 * 4);
        let reds: Vec<u8> = w.chunks_exact(4).map(|p| p[0]).collect();
        assert_eq!(reds, vec![5, 6, 7, 9, 10, 11, 13, 14, 15]);
    }

    #[test]
    fn image_editor_paint_the_splice_guard_skips_zero_coverage_texels() {
        // A dab in the LEFT half with a RIGHT-half selection: the splice
        // must not write a single byte, because every texel's effective
        // coverage is zero.
        let image = img(16, 16);
        let sel = SelectionCoverage::rasterize_rect(16, 16, 8.0, 0.0, 8.0, 16.0);
        let mut s = StrokeSession::begin(1, &image, params(StrokeTool::Brush), Some(Arc::new(sel)))
            .expect("begin");
        s.accumulator
            .stamp(&BrushTip::soft(6.0, 1.0), 3.0, 8.0, 1.0);
        let dirty = s.accumulator.take_dirty().expect("dirty");
        // Splice a window of pure white — none of it may land.
        let white = vec![255u8; (dirty.w * dirty.h * 4) as usize];
        s.splice(dirty, &white);
        assert_eq!(s.pixels(), &image.rgba[..], "clipped stroke wrote nothing");
    }

    #[test]
    fn image_editor_paint_the_splice_writes_where_coverage_is_positive() {
        let image = img(16, 16);
        let mut s =
            StrokeSession::begin(1, &image, params(StrokeTool::Brush), None).expect("begin");
        s.accumulator
            .stamp(&BrushTip::soft(6.0, 1.0), 8.0, 8.0, 1.0);
        let dirty = s.accumulator.take_dirty().expect("dirty");
        let white = vec![255u8; (dirty.w * dirty.h * 4) as usize];
        s.splice(dirty, &white);
        let at = |x: u32, y: u32| s.pixels()[((y * 16 + x) * 4) as usize];
        assert_eq!(at(8, 8), 255, "the dab centre took the spliced value");
        assert_eq!(at(0, 0), 128, "far corner untouched");
    }

    // ── end-to-end on the device ─────────────────────────────────────

    /// The test GPU device, or `None` where there is no adapter (the
    /// conformance harness's skip-don't-fail convention).
    fn device() -> Option<&'static GpuContext> {
        use std::sync::OnceLock;
        static DEVICE: OnceLock<Option<GpuContext>> = OnceLock::new();
        DEVICE
            .get_or_init(|| match pollster::block_on(GpuContext::new()) {
                Ok(ctx) => Some(ctx),
                Err(e) => {
                    eprintln!("stroke GPU unavailable: {e} — device tests will skip");
                    None
                }
            })
            .as_ref()
    }

    /// A deterministic non-uniform base.
    fn ramp(w: u32, h: u32) -> DecodedImage {
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                rgba.extend_from_slice(&[
                    (40 + (x * 5) % 200) as u8,
                    (60 + (y * 7) % 180) as u8,
                    (90 + ((x + y) * 3) % 150) as u8,
                    255,
                ]);
            }
        }
        DecodedImage::from_rgba8(w, h, rgba).expect("valid")
    }

    /// Drive `samples` through a session and return the painted pixels.
    fn run_stroke(
        ctx: &GpuContext,
        image: &DecodedImage,
        p: StrokeParams,
        selection: Option<Arc<SelectionCoverage>>,
        samples: &[(f32, f32, f32)],
    ) -> Vec<u8> {
        let mut s = StrokeSession::begin(1, image, p, selection).expect("begin");
        for &(x, y, pr) in samples {
            pollster::block_on(s.extend(ctx, StrokeSample::new(x, y, pr))).expect("extend");
        }
        s.commit()
    }

    const PATH: &[(f32, f32, f32)] = &[
        (6.0, 8.0, 0.5),
        (18.0, 11.0, 0.8),
        (31.0, 22.0, 1.0),
        (44.0, 30.0, 0.6),
        (52.0, 46.0, 0.3),
    ];

    #[test]
    fn image_editor_paint_the_incremental_stroke_matches_a_replay() {
        // Determinism + replay: the same sample list through two fresh
        // sessions must produce byte-identical pixels. This is what makes
        // a recorded action or a script reproduce a stroke exactly.
        let Some(ctx) = device() else { return };
        let image = ramp(64, 64);
        let a = run_stroke(ctx, &image, params(StrokeTool::Brush), None, PATH);
        let b = run_stroke(ctx, &image, params(StrokeTool::Brush), None, PATH);
        assert_eq!(a, b, "a replayed stroke must be byte-identical");
        assert_ne!(a, image.rgba.to_vec(), "the stroke actually painted");
    }

    #[test]
    fn image_editor_paint_a_fast_drag_matches_the_same_path_sampled_finely() {
        // The spacing walk's whole purpose: a drag delivered as FOUR long
        // segments must paint (very nearly) the same stroke as the same
        // geometric path delivered as many short ones. Dab POSITIONS are
        // identical by construction (`plan_segment` carries the residual);
        // the only difference is the linearly-interpolated pressure at
        // each dab, so the pixels agree to within a small tolerance.
        let Some(ctx) = device() else { return };
        let image = ramp(64, 64);
        let coarse = run_stroke(ctx, &image, params(StrokeTool::Brush), None, PATH);

        // Subdivide every segment into 8, interpolating pressure — the
        // same path, eight times the pointer rate.
        let mut fine = vec![PATH[0]];
        for pair in PATH.windows(2) {
            let (x0, y0, p0) = pair[0];
            let (x1, y1, p1) = pair[1];
            for i in 1..=8 {
                let t = i as f32 / 8.0;
                fine.push((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, p0 + (p1 - p0) * t));
            }
        }
        let dense = run_stroke(ctx, &image, params(StrokeTool::Brush), None, &fine);

        let mut worst = 0i32;
        for (a, b) in coarse.iter().zip(dense.iter()) {
            worst = worst.max((*a as i32 - *b as i32).abs());
        }
        assert!(
            worst <= 6,
            "a fast drag diverged from the same path sampled finely by {worst}/255"
        );
    }

    #[test]
    fn image_editor_paint_a_stroke_cannot_paint_outside_the_selection() {
        // The end-to-end statement of the claim, on BYTES: run a stroke
        // that crosses a selection edge and require every pixel outside
        // the selection to be identical to the base.
        let Some(ctx) = device() else { return };
        let image = ramp(64, 64);
        let sel = Arc::new(SelectionCoverage::rasterize_rect(
            64, 64, 0.0, 0.0, 32.0, 64.0,
        ));
        let painted = run_stroke(ctx, &image, params(StrokeTool::Brush), Some(sel), PATH);

        let mut changed_inside = 0;
        for y in 0..64u32 {
            for x in 0..64u32 {
                let i = ((y * 64 + x) * 4) as usize;
                let same = painted[i..i + 4] == image.rgba[i..i + 4];
                if x >= 32 {
                    assert!(same, "({x},{y}) is outside the selection but changed");
                } else if !same {
                    changed_inside += 1;
                }
            }
        }
        assert!(
            changed_inside > 100,
            "only {changed_inside} pixels painted inside the selection — the \
             fixture must actually cross the edge"
        );
    }

    #[test]
    fn image_editor_paint_an_eraser_stroke_takes_alpha_and_keeps_colour() {
        let Some(ctx) = device() else { return };
        let image = ramp(64, 64);
        let erased = run_stroke(ctx, &image, params(StrokeTool::Eraser), None, PATH);
        let at = |buf: &[u8], x: u32, y: u32| {
            let i = ((y * 64 + x) * 4) as usize;
            [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
        };
        // Somewhere on the path the alpha must have dropped …
        let hit = (0..64)
            .flat_map(|y| (0..64).map(move |x| (x, y)))
            .find(|&(x, y)| at(&erased, x, y)[3] < 32);
        let (hx, hy) = hit.expect("the eraser stroke must fully clear somewhere");
        // … and the RGB there must be preserved (straight-space erase).
        assert_eq!(
            at(&erased, hx, hy)[..3],
            at(&image.rgba, hx, hy)[..3],
            "an erased pixel keeps its colour"
        );
        // A corner the stroke never reached is untouched.
        assert_eq!(at(&erased, 63, 0), at(&image.rgba, 63, 0));
    }

    #[test]
    fn image_editor_paint_a_click_paints_a_dot_and_a_cancel_leaves_no_trace() {
        let Some(ctx) = device() else { return };
        let image = ramp(32, 32);
        let mut s =
            StrokeSession::begin(1, &image, params(StrokeTool::Brush), None).expect("begin");
        assert!(
            pollster::block_on(s.extend(ctx, StrokeSample::new(16.0, 16.0, 1.0))).expect("extend")
        );
        assert_eq!(s.dab_count(), 1);
        let bounds = s.stroke_bounds().expect("a dot has bounds");
        assert!(bounds.w >= 24 && bounds.h >= 24, "a 24 px tip: {bounds:?}");
        assert_ne!(s.pixels(), &image.rgba[..], "the dot landed");
        // Cancelling is just dropping the session — the engine-held image
        // was never mutated.
        drop(s);
        assert_eq!(
            image.rgba.to_vec(),
            ramp(32, 32).rgba.to_vec(),
            "the base pixels are still the base pixels"
        );
    }

    #[test]
    fn image_editor_paint_a_stroke_entirely_off_canvas_is_a_clean_no_op() {
        let Some(ctx) = device() else { return };
        let image = ramp(32, 32);
        let mut s =
            StrokeSession::begin(1, &image, params(StrokeTool::Brush), None).expect("begin");
        assert!(
            !pollster::block_on(s.extend(ctx, StrokeSample::new(500.0, 500.0, 1.0)))
                .expect("extend")
        );
        assert!(s.is_empty());
        assert_eq!(s.commit(), image.rgba.to_vec());
    }

    /// A MEASUREMENT, not a gate — `#[ignore]`d so it never fails CI on
    /// a slow runner. Run it to get the honest per-extend latency the
    /// door documentation quotes:
    ///
    /// ```text
    /// cargo test -p image-js --lib paint_latency -- --ignored --nocapture
    /// ```
    ///
    /// It separates the two costs that matter: the dirty-rectangle GPU
    /// composite (proportional to the DABS) and the whole-image readout
    /// the Stage-A byte payload forces (proportional to the IMAGE).
    #[test]
    #[ignore = "measurement, not a gate — see the doc comment"]
    fn image_editor_paint_latency_probe() {
        let Some(ctx) = device() else {
            eprintln!("no GPU adapter — skipping");
            return;
        };
        for (w, h) in [(512u32, 512u32), (2048, 1536), (4096, 3072)] {
            let image = ramp(w, h);
            let n = 60;
            let mut timings = Vec::new();
            for tool in [StrokeTool::Brush, StrokeTool::Eraser] {
                let mut s = StrokeSession::begin(1, &image, params(tool), None).expect("begin");
                // Warm the shader/pipeline path with the first extend.
                pollster::block_on(s.extend(ctx, StrokeSample::new(50.0, 50.0, 1.0)))
                    .expect("warm");
                let t0 = std::time::Instant::now();
                for i in 0..n {
                    let x = 50.0 + (i as f32) * 7.0;
                    let y = 50.0 + (i as f32) * 3.0;
                    pollster::block_on(s.extend(ctx, StrokeSample::new(x, y, 1.0)))
                        .expect("extend");
                }
                timings.push((tool, t0.elapsed(), s.dab_count()));
            }

            // The Stage-A tax: the WHOLE image copied out per extend.
            let s =
                StrokeSession::begin(1, &image, params(StrokeTool::Brush), None).expect("begin");
            let t1 = std::time::Instant::now();
            for _ in 0..n {
                std::hint::black_box(s.pixels().to_vec());
            }
            let readout = t1.elapsed();

            eprintln!("── {w}×{h} ({:.1} MB RGBA8)", (w * h * 4) as f64 / 1.0e6);
            for (tool, d, dabs) in &timings {
                eprintln!(
                    "   {:<7} {:>6.2} ms/sample over {n} samples ({dabs} dabs, {} dispatch(es): {})",
                    tool.as_wire(),
                    d.as_secs_f64() * 1000.0 / n as f64,
                    // The fixture base is opaque, so this is the fast path.
                    params(*tool).solid_paint_mode().unwrap().kernel_ids_for(true).len(),
                    params(*tool).solid_paint_mode().unwrap().kernel_ids_for(true).join(" → "),
                );
            }
            eprintln!(
                "   Stage-A whole-image byte copy: {:>6.2} ms/sample",
                readout.as_secs_f64() * 1000.0 / n as f64
            );
        }
    }

    #[test]
    fn image_editor_paint_a_zero_opacity_stroke_writes_nothing() {
        // Opacity 0 zeroes the effective coverage everywhere, so the same
        // guard that proves selection clipping also proves this.
        let image = img(16, 16);
        let mut p = params(StrokeTool::Brush);
        p.opacity = 0.0;
        let mut s = StrokeSession::begin(1, &image, p, None).expect("begin");
        s.accumulator
            .stamp(&BrushTip::soft(8.0, 1.0), 8.0, 8.0, 1.0);
        let dirty = s.accumulator.take_dirty().expect("dirty");
        let white = vec![255u8; (dirty.w * dirty.h * 4) as usize];
        s.splice(dirty, &white);
        assert_eq!(s.pixels(), &image.rgba[..]);
    }

    // ── clone / heal ─────────────────────────────────────────────────

    /// A base whose left half is dark and right half is light, so a copy
    /// from one side to the other is unmistakable.
    fn two_tone(w: u32, h: u32) -> DecodedImage {
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for _y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { 40u8 } else { 200u8 };
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
        }
        DecodedImage::from_rgba8(w, h, rgba).expect("valid")
    }

    #[test]
    fn image_editor_clone_the_sampling_tools_generate_no_paint_layer() {
        // The catch-all that used to live here made clone and heal paint
        // a solid colour, which reads as a broken clone rather than an
        // unhandled tool. An exhaustive `None` is the guard.
        assert!(params(StrokeTool::Clone).solid_paint_mode().is_none());
        assert!(params(StrokeTool::Heal).solid_paint_mode().is_none());
        assert!(params(StrokeTool::Brush).solid_paint_mode().is_some());
        assert!(params(StrokeTool::Eraser).solid_paint_mode().is_some());
        assert!(StrokeTool::Clone.samples_image());
        assert!(StrokeTool::Heal.samples_image());
        assert!(!StrokeTool::Brush.samples_image());
        // Only heal tone-matches — that is the entire difference.
        assert!(StrokeTool::Heal.tone_matches());
        assert!(!StrokeTool::Clone.tone_matches());
    }

    #[test]
    fn image_editor_clone_the_source_window_is_the_base_shifted() {
        let image = two_tone(16, 8);
        let s = StrokeSession::begin(1, &image, params(StrokeTool::Clone), None).expect("begin");
        let region = Region::new(0, 0, 4, 2);
        // No shift: the window is the base window.
        assert_eq!(
            s.source_window_rgba8(region, 0.0, 0.0),
            s.window_rgba8(region)
        );
        // Shifted right by 8, the same region reads the LIGHT half.
        let shifted = s.source_window_rgba8(region, 8.0, 0.0);
        assert!(shifted.chunks_exact(4).all(|p| p[0] == 200));
    }

    #[test]
    fn image_editor_clone_out_of_bounds_source_is_transparent_not_smeared() {
        // Clamping to the edge would smear the border across the canvas
        // and read as a rendering bug; transparency composites to
        // "nothing was copied here", which is the truth.
        let image = two_tone(16, 8);
        let s = StrokeSession::begin(1, &image, params(StrokeTool::Clone), None).expect("begin");
        let w = s.source_window_rgba8(Region::new(0, 0, 4, 2), -100.0, 0.0);
        assert!(w.chunks_exact(4).all(|p| p == [0, 0, 0, 0]));
        // A PARTIAL overlap keeps the part that exists.
        let half = s.source_window_rgba8(Region::new(0, 0, 4, 2), -2.0, 0.0);
        assert_eq!(&half[0..4], &[0, 0, 0, 0], "outside stays empty");
        assert_eq!(&half[8..12], &[40, 40, 40, 255], "inside is real pixels");
    }

    #[test]
    fn image_editor_clone_the_offset_is_fixed_at_the_first_sample() {
        // Recomputing it per dab would make the source follow the cursor
        // and smear the copy instead of translating it.
        let image = two_tone(32, 32);
        let mut s = StrokeSession::begin(1, &image, params(StrokeTool::Clone), None).expect("x");
        s.set_clone_source(CloneSource {
            x: 20.0,
            y: 5.0,
            aligned: true,
        });
        assert_eq!(s.clone_offset, None, "unresolved before the first sample");
        let _ = s.plan(StrokeSample::new(4.0, 5.0, 1.0));
        // `plan` alone does not resolve it — `extend` does, at the same
        // sample. Resolve it the way `extend` would and check the value.
        s.clone_offset = Some((20.0 - 4.0, 5.0 - 5.0));
        assert_eq!(s.clone_offset, Some((16.0, 0.0)));
    }

    #[test]
    fn image_editor_clone_copies_pixels_across_the_image() {
        let Some(ctx) = device() else {
            return;
        };
        let image = two_tone(32, 16);
        // A SMALL tip, so "untouched" means untouched: the 24 px default
        // reaches the corner from (8, 8) and the first version of this
        // test failed on its own geometry rather than on the clone.
        let mut small = params(StrokeTool::Clone);
        small.size = 6.0;
        let mut s = StrokeSession::begin(1, &image, small, None).expect("x");
        s.set_clone_source(CloneSource {
            x: 24.0,
            y: 8.0,
            aligned: true,
        });
        // Paint at x=8 (dark side) sampling from x=24 (light side).
        pollster::block_on(s.extend(ctx, StrokeSample::new(8.0, 8.0, 1.0))).expect("extend");
        let px = s.pixels();
        let at = |x: u32, y: u32| px[((y * 32 + x) * 4) as usize];
        assert!(
            at(8, 8) > 150,
            "the dab centre took the LIGHT source, got {}",
            at(8, 8)
        );
        assert_eq!(at(0, 0), 40, "far corner untouched");
    }

    #[test]
    fn image_editor_clone_without_an_anchor_paints_nothing() {
        // A clone with nowhere to read from must deposit NOTHING. Falling
        // back to a colour would look like a broken clone.
        let Some(ctx) = device() else {
            return;
        };
        let image = two_tone(32, 16);
        let mut s = StrokeSession::begin(1, &image, params(StrokeTool::Clone), None).expect("x");
        pollster::block_on(s.extend(ctx, StrokeSample::new(8.0, 8.0, 1.0))).expect("extend");
        assert_eq!(s.pixels(), &image.rgba[..], "no anchor, no paint");
    }

    #[test]
    fn image_editor_heal_lands_closer_to_the_destination_than_clone_does() {
        // The single assertion that separates the two tools: from the
        // SAME source and the same place, heal must land nearer the tone
        // it replaced. Measured, not asserted by construction.
        //
        // GEOMETRY MATTERS HERE, and the first version of this test got
        // it wrong: with the source offset near the image edge, every
        // boundary texel of the solve came from out of bounds, so there
        // was no usable boundary and NO correction was produced — heal
        // read exactly like clone and the test failed for a reason that
        // had nothing to do with the solve. A wide canvas and a small
        // tip keep both the dab's boundary ring and its source in
        // bounds, which is the ordinary case.
        let Some(ctx) = device() else {
            return;
        };
        let image = two_tone(64, 32);
        let sample = StrokeSample::new(16.0, 16.0, 1.0);
        let source = CloneSource {
            x: 48.0,
            y: 16.0,
            aligned: true,
        };
        let small = |tool| {
            let mut p = params(tool);
            p.size = 6.0;
            p
        };

        let mut cloned = StrokeSession::begin(1, &image, small(StrokeTool::Clone), None).unwrap();
        cloned.set_clone_source(source);
        pollster::block_on(cloned.extend(ctx, sample)).expect("extend");
        let clone_v = cloned.pixels()[((16 * 64 + 16) * 4) as usize] as i32;

        let mut healed = StrokeSession::begin(1, &image, small(StrokeTool::Heal), None).unwrap();
        healed.set_clone_source(source);
        pollster::block_on(healed.extend(ctx, sample)).expect("extend");
        let heal_v = healed.pixels()[((16 * 64 + 16) * 4) as usize] as i32;

        let dest = 40i32;
        assert!(
            (heal_v - dest).abs() < (clone_v - dest).abs(),
            "heal ({heal_v}) should sit nearer the destination tone ({dest}) \
             than clone ({clone_v}) does"
        );
    }

    #[test]
    fn image_editor_heal_says_nothing_rather_than_guessing_at_an_edge() {
        // The behaviour the geometry above avoids, pinned as its own
        // case: when the source sits so near an edge that the solve has
        // no usable boundary, NO correction is invented and heal falls
        // back to a plain clone. That is the honest degradation — the
        // alternative is a correction derived from pixels that do not
        // exist.
        let Some(ctx) = device() else {
            return;
        };
        let image = two_tone(32, 16);
        let mut p = params(StrokeTool::Heal);
        p.size = 6.0;
        let mut healed = StrokeSession::begin(1, &image, p, None).unwrap();
        // Source 16 px to the right of a 32-wide image at x=24: the
        // sampled window runs off the edge.
        healed.set_clone_source(CloneSource {
            x: 40.0,
            y: 8.0,
            aligned: true,
        });
        // It must not panic, and it must not paint something invented.
        pollster::block_on(healed.extend(ctx, StrokeSample::new(24.0, 8.0, 1.0)))
            .expect("extend survives an out-of-bounds source");
    }

    #[test]
    fn image_editor_heal_cancels_a_gradient_mismatch_not_just_a_uniform_one() {
        // THE UPGRADE, measured end to end on a device. The destination
        // is a horizontal ramp and the source is flat, so the mismatch
        // VARIES across the patch — the case a single mean offset cannot
        // cancel, and the reason the panel used to carry a "still shows a
        // seam" warning.
        //
        // The test compares the healed result against the destination it
        // replaced at two points on opposite sides of the dab. A constant
        // correction can only be right at one of them; the membrane
        // solve should be close at both.
        let Some(ctx) = device() else {
            return;
        };
        let (w, h) = (96u32, 32u32);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for _y in 0..h {
            for x in 0..w {
                // A ramp on the LEFT two-thirds; a flat patch on the
                // right to sample from.
                let v = if x < 64 { (x * 3) as u8 } else { 30u8 };
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let image = DecodedImage::from_rgba8(w, h, rgba).expect("valid");

        let mut p = params(StrokeTool::Heal);
        p.size = 14.0;
        p.hardness = 1.0;
        let mut healed = StrokeSession::begin(1, &image, p, None).unwrap();
        healed.set_clone_source(CloneSource {
            x: 80.0,
            y: 16.0,
            aligned: true,
        });
        pollster::block_on(healed.extend(ctx, StrokeSample::new(32.0, 16.0, 1.0))).expect("extend");

        let px = healed.pixels();
        let at = |x: u32| px[((16 * w + x) * 4) as usize] as i32;
        let want = |x: u32| (x * 3) as i32;
        // The SLOPE across the patch is the discriminator, and it had
        // to be: an absolute-error tolerance loose enough to pass at all
        // also passes for a constant correction, which this test caught
        // by mutation before it was tightened. A constant leaves the
        // healed patch FLAT where the destination ramps; only a field
        // reproduces the ramp.
        let (l, r) = (28u32, 36u32);
        let got_slope = at(r) - at(l);
        let want_slope = want(r) - want(l);
        assert!(
            got_slope > want_slope / 2,
            "the healed patch must RAMP like its destination, not sit \
             flat: got {got_slope} across {l}..{r}, destination ramps \
             {want_slope} (at {l}: {} vs {}; at {r}: {} vs {})",
            at(l),
            want(l),
            at(r),
            want(r)
        );
        // …and still land near the destination at both ends.
        assert!(
            (at(l) - want(l)).abs() < 24 && (at(r) - want(r)).abs() < 24,
            "at {l} got {} want {}; at {r} got {} want {}",
            at(l),
            want(l),
            at(r),
            want(r)
        );
    }
}
