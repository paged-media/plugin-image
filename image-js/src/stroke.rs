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
//! # Honest scope: there is no layer graph
//!
//! paged.image holds ONE image per handle. A stroke therefore paints
//! into that single engine-held image — not into a paint layer, not into
//! a layer above the photo. There is nothing to hide behind the word
//! "layer" here and the UI says so in as many words. The consequences
//! are real and are not papered over:
//!
//! * A committed stroke is DESTRUCTIVE into the engine's working pixels
//!   (the crop / resize / fill commit pattern: it registers a NEW
//!   engine-held image and the caller swaps handles). The DOCUMENT and
//!   the source file are still untouched — re-ingesting the frame
//!   restores the original pixels, and that is the only "undo" this
//!   plugin owns.
//! * The `image-graph` (Engine B) COW/undo journaling that would make a
//!   stroke a first-class undoable operation is a recorded deferral in
//!   that crate; see the note on [`StrokeSession`] for exactly what was
//!   wanted from it and what exists today.
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
}

impl StrokeTool {
    /// Decode the wire name (mirrored by the TS `StrokeTool` union).
    pub fn from_wire(s: &str) -> Option<StrokeTool> {
        Some(match s {
            "brush" => StrokeTool::Brush,
            "pencil" => StrokeTool::Pencil,
            "eraser" => StrokeTool::Eraser,
            _ => return None,
        })
    }

    pub fn as_wire(self) -> &'static str {
        match self {
            StrokeTool::Brush => "brush",
            StrokeTool::Pencil => "pencil",
            StrokeTool::Eraser => "eraser",
        }
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

    /// The GPU paint mode this stroke composites through.
    pub fn paint_mode(&self) -> PaintMode {
        match self.tool {
            StrokeTool::Eraser => PaintMode::Erase,
            _ => PaintMode::Paint {
                blend: self.blend,
                color: self.color,
            },
        }
    }
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
/// # What was wanted from `image-graph`, and what exists
///
/// Engine B has exactly the right SHAPE for this: a `gesture` (ephemeral
/// param override) / `set_params` (committed) split, per-node sparse
/// tile caches keyed on `(params_hash, input_generations)`, and damage
/// propagation. A stroke in progress is the textbook gesture and a
/// released stroke the textbook commit.
///
/// What it does not have is the thing a stroke actually needs: paint is
/// not a PARAMETER change, it is a WRITE, and `BufferGraph`'s write path
/// (`write_source_tile`) has no undo journal — the COW `Arc<Tile>`
/// snapshot log is a documented deferral in `image-graph/src/lib.rs`.
/// Routing a stroke through Engine B today would therefore buy tile
/// caching while still leaving this session to own the base snapshot and
/// the dirty-rectangle bookkeeping — the two things it exists for. So it
/// does NOT ride Engine B; it keeps a flat base snapshot and its own
/// dirty rect, and what it needs from Engine B is recorded here rather
/// than half-built:
///
/// 1. **`WriteBuffer` COW journaling** — so `begin`'s snapshot becomes a
///    generation marker instead of a full-image `Vec<u8>` clone.
/// 2. **A tiled accumulation buffer** — so the coverage field is sparse
///    over the stroke's tiles instead of dense over the image.
/// 3. **A resident output texture** — so `extend` does not read a window
///    back to the CPU at all (the same Stage-B dependency the composite
///    round-trip has; see the door docs in `lib.rs`).
///
/// None of the three is required for correctness, and all three are
/// invisible to this type's callers.
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
        if image.width == 0 || image.height == 0 {
            return Err(IngestError::Unsupported(
                "cannot paint on an empty image".into(),
            ));
        }
        Ok(StrokeSession {
            handle,
            width: image.width,
            height: image.height,
            base: Arc::clone(&image.rgba),
            working: image.rgba.to_vec(),
            accumulator: StrokeAccumulator::new(image.width, image.height),
            params: params.sanitized(),
            selection,
            last: None,
            walk: image_gpu::StrokeWalk::new(),
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
        let out_f16 =
            composite_stroke_window(ctx, &self.params.paint_mode(), &base_f16, &mask, w, h)
                .await
                .map_err(|e| IngestError::Pipeline(e.to_string()))?;
        let out = f16_to_rgba8(&out_f16);
        self.splice(region, &out);
        Ok(())
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

    /// Finish: hand back the painted pixels. The caller registers them as
    /// a new engine-held image (the destructive-commit pattern).
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
        for t in [StrokeTool::Brush, StrokeTool::Pencil, StrokeTool::Eraser] {
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
            params(StrokeTool::Eraser).paint_mode(),
            PaintMode::Erase
        ));
        assert_eq!(
            params(StrokeTool::Brush).paint_mode().kernel_ids(),
            vec![
                "gen.solid",
                "cast.premultiply",
                "compose.normal",
                "cast.unpremultiply"
            ]
        );
        assert_eq!(
            params(StrokeTool::Pencil).paint_mode().kernel_ids().len(),
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
                    params(*tool).paint_mode().kernel_ids_for(true).len(),
                    params(*tool).paint_mode().kernel_ids_for(true).join(" → "),
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
}
