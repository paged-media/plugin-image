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

//! THE LAYER GRAPH — an ordered stack of pixel layers composited
//! bottom-up through the `compose.*` kernels, plus the COW undo journal
//! that makes an edit to one of them reversible.
//!
//! This is what turns paged.image from an adjustment pipeline into an
//! editor: a stroke lands in the ACTIVE LAYER instead of overwriting the
//! one image the plugin used to hold, and it can be taken back.
//!
//! # The model
//!
//! A [`LayerStack`] is `width × height` fixed (the canvas) and holds
//! [`Layer`]s bottom-first — index 0 is the bottom, `len() - 1` the top,
//! the same order PSD stores them in. Every layer is CANVAS-EXTENT
//! straight RGBA8. That is a deliberate simplification over per-layer
//! bounds: it makes the composite a pure fold with no offset arithmetic
//! and lets the brush paint anywhere without growing anything, at the
//! cost of 4 bytes per pixel per layer. The cost is paid honestly — the
//! PSD import budget (`image_psd::layer_pixels`) is bounded because of it.
//!
//! Each layer carries `name`, `visible`, `locked`, `opacity` (0–1) and a
//! `blend` — one of the 26 registered `compose.*` kernels, resolved
//! through [`crate::stroke::blend_kernel`] so the set can never drift
//! from the kernels that exist.
//!
//! # The composite, and the premultiplied invariant
//!
//! This repo has been bitten twice by a straight-vs-premultiplied seam
//! (in `stroke.rs` and in `fill.rs`), so the rule here is stated once
//! and holds everywhere:
//!
//! * **Layer pixels are STRAIGHT** RGBA8 (the engine's working
//!   convention — the decode bridge maps `u8/255` with no alpha
//!   association).
//! * **The fold accumulator is PREMULTIPLIED** rgba16float, starting at
//!   transparent black (all zeros), which is what the `compose.*`
//!   family's contract requires on BOTH inputs.
//! * A layer therefore enters through `cast.premultiply` and the FINAL
//!   accumulator leaves through `cast.unpremultiply` — once each, never
//!   per pair.
//!
//! Both casts are skipped exactly where they are PROVABLY the identity,
//! on the same test the stroke compositor uses
//! ([`image_gpu::stroke::window_is_opaque`]): premultiplying a
//! fully-opaque window is `rgb·1`, and unpremultiplying a fully-opaque
//! accumulator divides by one. That is an exact statement about bytes,
//! not an approximation.
//!
//! The per-layer step is one dispatch:
//!
//! ```text
//! acc ← compose.<blend>(acc, premul(layer), opacity = layer.opacity)
//! ```
//!
//! and the compose family computes `over(a, b·α)`, i.e. the layer's
//! opacity IS the `α` its param block already carries. **No new kernel
//! is needed for any of this**, which is the point: a layer composite is
//! a fold over kernels that shipped with the blend-mode work.
//!
//! # The fast path (why a one-layer document costs nothing)
//!
//! A stack of ONE visible layer at opacity 1 with `compose.normal` folds
//! to `unpremultiply(over(transparent, premultiply(L)))` ≡ `L`. So that
//! case returns the layer's pixels VERBATIM — the same `Arc`, no f16
//! round-trip, no dispatch, **no GPU required**. Every document starts
//! that way, so opening a layer stack over an ingested image costs one
//! `Arc` clone and compositing it costs nothing at all. The lanes that
//! never wanted a device (identity adjust, tiles, histogram, save-back)
//! keep working exactly as before.
//!
//! # Undo
//!
//! [`LayerStack`] owns an `image_graph::TileJournal`. A pixel edit
//! ([`LayerStack::edit_active`]) snapshots only the tiles its damage
//! region covers before writing, so a small stroke on a big canvas
//! journals a tile or two. The bound and its behaviour at the limit are
//! the journal's, documented there and surfaced through
//! [`LayerStack::history`].
//!
//! Each entry is SCOPED to the layer it edited (by the layer's stable
//! id), so undoing after switching layers restores the layer that was
//! painted — not whichever one is selected — and makes it active again.
//!
//! **The journal is a PIXEL log.** Layer STRUCTURE — add, remove,
//! reorder, rename, opacity, blend, visibility — is not journaled. That
//! is a stated limit, not an oversight: those operations are cheap to
//! reverse by hand, and journaling them would mean holding a removed
//! layer's whole canvas. Two consequences follow and both are enforced
//! here rather than discovered: removing the LAST layer is refused (a
//! document can never become pixel-less by one click), and removing any
//! other CLEARS the journal, because its entries are keyed to pixels
//! that no longer exist and an entry that can never be applied is worse
//! than no entry at all.

use std::sync::Arc;

use image_core::Region;
use image_gpu::coverage::SelectionCoverage;
use image_gpu::selection::SelectionMask;
use image_gpu::stroke::window_is_opaque;
use image_gpu::{GpuContext, TileInput};
use image_graph::journal::{FlatImage, RecordOutcome, TileJournal};
use image_kernels::families::cast::{
    CastPremultiplyParams, CastUnpremultiplyParams, CAST_PREMULTIPLY, CAST_UNPREMULTIPLY,
};
use image_kernels::families::compose::{ComposeParams, COMPOSE_NORMAL};
use image_kernels::KernelDef;

use crate::fill::{f16_to_rgba8, rgba8_to_f16};
use crate::ingest::{adjust_rgba8, AdjustParams, DecodedImage, IngestError};
use crate::stroke::blend_kernel;

/// The default name of the layer an ingested image becomes.
pub const BACKGROUND_LAYER_NAME: &str = "Background";

/// What a layer CONTRIBUTES to the fold.
///
/// A pixel layer contributes its own pixels; an adjustment layer
/// contributes a TRANSFORMATION OF EVERYTHING BENEATH IT. That is the
/// whole non-destructive idea, and it is why this is a kind on the layer
/// rather than a second stack: order, opacity, blend, visibility, lock
/// and the MASK all mean the same thing for both, so they must not be
/// re-implemented per kind.
#[derive(Debug, Clone)]
pub enum LayerKind {
    /// Canvas-extent pixels of its own.
    Pixels,
    /// No pixels — the adjust chain, run over the backdrop beneath.
    /// Boxed because `AdjustParams` is much larger than a discriminant
    /// and every pixel layer would otherwise pay for it.
    Adjustment(Box<AdjustParams>),
    /// A SMART OBJECT: the layer's pixels are a cached RENDER of
    /// preserved source bytes at a scale, not the source itself.
    ///
    /// The distinction is the entire point. A pixel layer scaled to 25%
    /// and back to 100% has lost three quarters of its information for
    /// good; a smart object re-renders from `source` at the new scale,
    /// so the round trip is lossless and the original survives every
    /// edit. §32 decision 5 warns to add this model "before many
    /// destructive features, or later migration becomes expensive" —
    /// which is why it is a layer KIND rather than a wrapper: order,
    /// opacity, blend, visibility and the mask keep meaning exactly what
    /// they mean for every other layer.
    Smart(Box<SmartSource>),
}

/// The preserved original behind a smart object, plus the scale its
/// cached pixels were rendered at.
#[derive(Debug, Clone)]
pub struct SmartSource {
    /// The ORIGINAL, at its own resolution. Never resampled in place —
    /// every re-render reads this.
    pub rgba: Arc<[u8]>,
    pub width: u32,
    pub height: u32,
    /// The scale the layer's current `rgba` was rendered at (1.0 = the
    /// source's own size). Kept so a re-render knows what changed and a
    /// UI can show it.
    pub scale: f32,
}

/// One pixel layer: canvas-extent straight RGBA8 plus the four
/// properties the composite reads and the one (`locked`) it refuses on.
#[derive(Debug, Clone)]
pub struct Layer {
    /// Pixels of its own, or an adjustment over what is below.
    pub kind: LayerKind,
    /// Stable across reorders — the id the UI keys rows by.
    pub id: u32,
    pub name: String,
    pub visible: bool,
    /// A locked layer refuses PIXEL edits (paint, fill, bake). Its
    /// properties are still editable — that is what "lock the pixels"
    /// means, and pretending otherwise would just be a different lie.
    pub locked: bool,
    /// 0–1.
    pub opacity: f32,
    pub blend: &'static KernelDef,
    /// Canvas-extent, tightly packed straight RGBA8. `Arc` so a layer
    /// can share the ingest's allocation and a snapshot is a pointer.
    pub rgba: Arc<[u8]>,
    /// The layer MASK — a canvas-extent grayscale coverage field, or
    /// `None` for "fully opaque everywhere" (the overwhelming default,
    /// and cheaper than materializing a constant-one field per layer).
    ///
    /// This is deliberately the SAME `SelectionCoverage` the selection
    /// tools already author: a layer mask and a selection are the same
    /// object with a different owner, so masking a layer needs no new
    /// authoring surface — "make selection into mask" is a move, not a
    /// conversion. It lowers to the ABI's `@group(2)` r16float mask that
    /// every dispatch already takes.
    pub mask: Option<Arc<SelectionCoverage>>,
    /// A DISABLED mask is retained, not discarded — Photoshop's
    /// shift-click. Toggling it off must not lose the painted coverage,
    /// which is the whole reason it is a separate flag rather than
    /// setting `mask` to `None`.
    pub mask_enabled: bool,
    /// CLIPPED to the layer beneath: this layer contributes only where
    /// its clip BASE is opaque, so an adjustment can be confined to one
    /// object without painting a mask around it.
    ///
    /// It is expressed as an extra MASK factor rather than as a new
    /// compositing path, which is the point — the base's alpha and a
    /// painted mask are the same kind of thing, so clipping needed no
    /// new kernel and no second fold. This is also what "smart filters"
    /// wanted: an adjustment layer clipped to a smart object IS a smart
    /// filter.
    pub clipped: bool,
}

impl Layer {
    /// The blend's wire name (the `compose.` prefix dropped) — what the
    /// panel's picker and the JSON readout use.
    pub fn blend_name(&self) -> &'static str {
        self.blend
            .id
            .strip_prefix("compose.")
            .unwrap_or(self.blend.id)
    }

    /// Are this layer's PROPERTIES ones that let it contribute? A hidden
    /// layer or one at zero opacity contributes EXACTLY nothing — the
    /// compose spine sets `alpha_s = b.a · opacity`, and at `alpha_s = 0`
    /// its output reduces to the backdrop for all 26 blend modes — so
    /// skipping it is exact, not an approximation.
    fn enabled(&self) -> bool {
        self.visible && self.opacity > 0.0
    }

    /// Is this layer a plain, unmodified pass-through — the shape that
    /// makes a one-layer composite the identity?
    fn is_plain(&self) -> bool {
        self.is_pixels()
            && self.opacity >= 1.0
            && std::ptr::eq(self.blend, &COMPOSE_NORMAL)
            && self.live_mask().is_none()
    }

    /// Does this layer carry pixels of its own?
    pub fn is_pixels(&self) -> bool {
        // A smart object's CACHED RENDER is pixels as far as the fold is
        // concerned; what makes it smart is where those pixels came from
        // and that they can be regenerated, not how they composite.
        matches!(self.kind, LayerKind::Pixels | LayerKind::Smart(_))
    }

    /// The preserved source behind a smart object.
    pub fn smart_source(&self) -> Option<&SmartSource> {
        match &self.kind {
            LayerKind::Smart(s) => Some(s),
            _ => None,
        }
    }

    /// The adjust parameters when this is an adjustment layer.
    pub fn adjust_params(&self) -> Option<&AdjustParams> {
        match &self.kind {
            LayerKind::Adjustment(p) => Some(p),
            // A smart object contributes PIXELS (its cached render), not
            // a transform of the backdrop — so it has no adjust params.
            LayerKind::Pixels | LayerKind::Smart(_) => None,
        }
    }

    /// The mask that actually applies: `None` when there is none or it is
    /// disabled, and `None` too when it is all-one (which is the identity
    /// — materializing it would cost an upload to change nothing).
    pub fn live_mask(&self) -> Option<&Arc<SelectionCoverage>> {
        if !self.mask_enabled {
            return None;
        }
        match self.mask.as_ref() {
            Some(m) if !m.is_all_one() => Some(m),
            _ => None,
        }
    }
}

/// Is every texel of a straight RGBA8 buffer fully TRANSPARENT?
///
/// Such a layer is exactly the identity in the fold — `alpha_s = 0` in
/// the compose spine leaves the backdrop untouched for every blend mode
/// — so it is skipped. That is not a micro-optimization: "add a layer"
/// is the first thing anyone does, and skipping the empty one keeps the
/// A plate's alpha channel, one byte per pixel — the clip base.
///
/// A clipping base IS its alpha: "show this layer only where the one
/// below is opaque" is exactly what a coverage field says, which is why
/// clipping folds into the existing mask path rather than needing a
/// second compositing mode.
fn alpha_of(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4).map(|p| p[3]).collect()
}

/// A layer's own coverage AND its clip base, multiplied.
///
/// Two coverages MULTIPLY — they do not override one another — so a
/// layer that is both masked and clipped is confined by both. Returning
/// `None` when neither exists keeps the constant-one fast path, so an
/// ordinary layer still pays nothing for a feature it does not use.
fn effective_coverage(
    own: Option<&Arc<SelectionCoverage>>,
    clip: Option<&[u8]>,
    w: u32,
    h: u32,
) -> Option<Arc<SelectionCoverage>> {
    match (own, clip) {
        (None, None) => None,
        (Some(cov), None) => Some(Arc::clone(cov)),
        (None, Some(base)) => SelectionCoverage::from_data(w, h, base.to_vec()).map(Arc::new),
        (Some(cov), Some(base)) => {
            let data: Vec<u8> = (0..(w as usize) * (h as usize))
                .map(|i| {
                    let x = (i % w as usize) as u32;
                    let y = (i / w as usize) as u32;
                    let a = u32::from(cov.coverage_at(x, y));
                    let b = u32::from(base[i]);
                    // Round-half-up on the /255, so full × full stays
                    // full — a clipped, fully-masked layer must not lose
                    // a level to integer truncation on every composite.
                    ((a * b + 127) / 255) as u8
                })
                .collect();
            SelectionCoverage::from_data(w, h, data).map(Arc::new)
        }
    }
}

/// composite trivial (and therefore GPU-free) until something is
/// actually painted into it.
fn is_fully_transparent(rgba: &[u8]) -> bool {
    rgba.chunks_exact(4).all(|t| t[3] == 0)
}

/// The undo/redo readout the panel shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryStats {
    pub can_undo: bool,
    pub can_redo: bool,
    pub depth: usize,
    pub redo_depth: usize,
    pub bytes: usize,
    pub max_bytes: usize,
    pub max_entries: usize,
    /// Entries evicted by the bound so far (see the journal docs) —
    /// surfaced so "history is a window" is said, never discovered.
    pub dropped: u64,
    pub generation: u64,
}

/// An ordered stack of pixel layers over one canvas, with the COW undo
/// journal for its pixel edits.
pub struct LayerStack {
    width: u32,
    height: u32,
    /// Bottom-first (index 0 is the bottom-most layer).
    layers: Vec<Layer>,
    active: usize,
    next_id: u32,
    journal: TileJournal,
}

impl LayerStack {
    /// Open a stack over `rgba` — one full-canvas [`BACKGROUND_LAYER_NAME`]
    /// layer. The pixels are SHARED (an `Arc` clone), so this is O(1) and
    /// costs no extra memory over the image it was opened on.
    pub fn from_image(width: u32, height: u32, rgba: Arc<[u8]>) -> Result<LayerStack, IngestError> {
        let want = (width as usize) * (height as usize) * 4;
        if width == 0 || height == 0 || rgba.len() != want {
            return Err(IngestError::Decode(format!(
                "layer stack: {} bytes for {width}×{height} (expected {want})",
                rgba.len()
            )));
        }
        Ok(LayerStack {
            width,
            height,
            layers: vec![Layer {
                id: 1,
                name: BACKGROUND_LAYER_NAME.to_string(),
                visible: true,
                locked: false,
                opacity: 1.0,
                blend: &COMPOSE_NORMAL,
                rgba,
                kind: LayerKind::Pixels,
                // A new layer is unmasked; the mask is authored later.
                mask: None,
                mask_enabled: true,
                clipped: false,
            }],
            active: 0,
            next_id: 2,
            journal: TileJournal::new(),
        })
    }

    /// Open a stack from a PSD's imported layer plates
    /// ([`image_psd::LayerImport`], bottom-first) — the layered PSD lane.
    /// Blend keys resolve through [`psd_blend_kernel`]; an unmodeled key
    /// falls back to `normal` (the file's own bytes are preserved
    /// regardless, and the panel names the layer so the user can see it).
    pub fn from_psd_plates(import: &image_psd::LayerImport) -> Result<LayerStack, IngestError> {
        let (width, height) = (import.width, import.height);
        if import.layers.is_empty() {
            return Err(IngestError::Unsupported(
                "PSD layer import produced no layers".into(),
            ));
        }
        let want = (width as usize) * (height as usize) * 4;
        let mut layers = Vec::with_capacity(import.layers.len());
        for (i, plate) in import.layers.iter().enumerate() {
            if plate.rgba.len() != want {
                return Err(IngestError::Decode(format!(
                    "PSD layer \"{}\" is {} bytes for {width}×{height} (expected {want})",
                    plate.name,
                    plate.rgba.len()
                )));
            }
            layers.push(Layer {
                id: (i as u32) + 1,
                name: if plate.name.is_empty() {
                    format!("Layer {}", i + 1)
                } else {
                    plate.name.clone()
                },
                visible: !plate.hidden,
                locked: false,
                opacity: plate.opacity as f32 / 255.0,
                blend: psd_blend_kernel(&plate.blend_key),
                rgba: Arc::from(plate.rgba.clone().into_boxed_slice()),
                kind: LayerKind::Pixels,
                // A new layer is unmasked; the mask is authored later.
                mask: None,
                mask_enabled: true,
                clipped: false,
            });
        }
        let next_id = layers.len() as u32 + 1;
        Ok(LayerStack {
            width,
            height,
            active: layers.len() - 1,
            layers,
            next_id,
            journal: TileJournal::new(),
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    pub fn active_index(&self) -> usize {
        self.active
    }

    pub fn active(&self) -> &Layer {
        &self.layers[self.active]
    }

    /// A transparent canvas-extent buffer (a new empty layer's pixels).
    fn transparent(&self) -> Arc<[u8]> {
        Arc::from(vec![0u8; (self.width as usize) * (self.height as usize) * 4].into_boxed_slice())
    }

    fn fresh_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Add an empty transparent layer directly ABOVE the active one and
    /// make it active. Returns its index.
    /// Insert an ADJUSTMENT layer above the active one. It carries no
    /// pixels; it transforms everything beneath it at composite time, so
    /// the pixels it affects are never modified and deleting it restores
    /// the original exactly.
    ///
    /// This is what makes the 15 reachable §14.1 adjustments
    /// non-destructive: the same `AdjustParams` the panel already builds,
    /// evaluated in the fold instead of written into a layer.
    pub fn add_adjustment(&mut self, name: &str, params: AdjustParams) -> usize {
        let id = self.fresh_id();
        let at = self.active + 1;
        let pixels = self.transparent();
        self.layers.insert(
            at,
            Layer {
                kind: LayerKind::Adjustment(Box::new(params)),
                id,
                name: if name.is_empty() {
                    format!("Adjustment {id}")
                } else {
                    name.to_string()
                },
                visible: true,
                locked: false,
                opacity: 1.0,
                blend: &COMPOSE_NORMAL,
                rgba: pixels,
                mask: None,
                mask_enabled: true,
                clipped: false,
            },
        );
        self.active = at;
        at
    }

    /// CONVERT a pixel layer into a smart object, preserving its current
    /// pixels as the source. From here a rescale is lossless: the render
    /// comes from `source`, never from the previous render.
    ///
    /// Converting is one-way by design. Going back would mean discarding
    /// the source, and a "convert to pixels" that silently threw away
    /// the original is exactly the destructive move this rung exists to
    /// prevent — rasterize by baking into a NEW pixel layer instead.
    pub fn make_smart(&mut self, index: usize) -> Result<(), IngestError> {
        let (w, h) = (self.width, self.height);
        let layer = self.layer_mut(index)?;
        if matches!(layer.kind, LayerKind::Adjustment(_)) {
            return Err(IngestError::Unsupported(format!(
                "layer {index} is an adjustment layer, which has no pixels to preserve"
            )));
        }
        if layer.smart_source().is_some() {
            return Ok(()); // already smart — idempotent, not an error
        }
        layer.kind = LayerKind::Smart(Box::new(SmartSource {
            rgba: Arc::clone(&layer.rgba),
            width: w,
            height: h,
            scale: 1.0,
        }));
        Ok(())
    }

    /// Record a re-render of a smart object at `scale`.
    ///
    /// The CALLER does the resampling (it is a GPU kernel dispatch and
    /// this module holds no device), but the invariant lives here: the
    /// source is never replaced, only the cached render is. That is what
    /// makes scaling down and back up lossless, and it is asserted
    /// directly in the tests.
    pub fn set_smart_render(
        &mut self,
        index: usize,
        rendered: Arc<[u8]>,
        scale: f32,
    ) -> Result<(), IngestError> {
        let want = (self.width as usize) * (self.height as usize) * 4;
        if rendered.len() != want {
            return Err(IngestError::Unsupported(format!(
                "smart render is {} bytes but the canvas needs {want}",
                rendered.len()
            )));
        }
        let layer = self.layer_mut(index)?;
        match &mut layer.kind {
            LayerKind::Smart(src) => {
                src.scale = scale;
                layer.rgba = rendered;
                Ok(())
            }
            _ => Err(IngestError::Unsupported(format!(
                "layer {index} is not a smart object"
            ))),
        }
    }

    /// Whether `index` is a smart object.
    pub fn is_smart(&self, index: usize) -> bool {
        self.layers
            .get(index)
            .is_some_and(|l| l.smart_source().is_some())
    }

    /// Retune an existing adjustment layer. Errors on a pixel layer
    /// rather than silently converting it — a conversion would discard
    /// pixels, which is the one thing this whole feature exists to avoid.
    pub fn set_adjustment(
        &mut self,
        index: usize,
        params: AdjustParams,
    ) -> Result<(), IngestError> {
        let layer = self.layer_mut(index)?;
        match &mut layer.kind {
            LayerKind::Adjustment(p) => {
                **p = params;
                Ok(())
            }
            LayerKind::Pixels | LayerKind::Smart(_) => Err(IngestError::Unsupported(format!(
                "layer {index} holds pixels, not an adjustment"
            ))),
        }
    }

    /// Whether `index` is an adjustment layer.
    pub fn is_adjustment(&self, index: usize) -> bool {
        self.layers
            .get(index)
            .is_some_and(|l| l.adjust_params().is_some())
    }

    pub fn add(&mut self, name: &str) -> usize {
        let id = self.fresh_id();
        let at = self.active + 1;
        let pixels = self.transparent();
        self.layers.insert(
            at,
            Layer {
                id,
                name: if name.is_empty() {
                    format!("Layer {id}")
                } else {
                    name.to_string()
                },
                visible: true,
                locked: false,
                opacity: 1.0,
                blend: &COMPOSE_NORMAL,
                rgba: pixels,
                kind: LayerKind::Pixels,
                // A new layer is unmasked; the mask is authored later.
                mask: None,
                mask_enabled: true,
                clipped: false,
            },
        );
        self.active = at;
        at
    }

    /// Duplicate `index` directly above itself (pixels shared behind the
    /// `Arc` until one of the two is edited) and make the copy active.
    pub fn duplicate(&mut self, index: usize) -> Option<usize> {
        let src = self.layers.get(index)?.clone();
        let id = self.fresh_id();
        let at = index + 1;
        self.layers.insert(
            at,
            Layer {
                id,
                name: format!("{} copy", src.name),
                // `..src` carries the mask too: duplicating a masked
                // layer must duplicate its mask, or the copy would
                // silently reveal what the original hides.
                ..src
            },
        );
        self.active = at;
        Some(at)
    }

    /// Remove `index`. Refused for the LAST layer — a document with no
    /// pixels at all is not a state this offers by one click.
    pub fn remove(&mut self, index: usize) -> Result<(), IngestError> {
        if index >= self.layers.len() {
            return Err(IngestError::Unsupported(format!("no layer {index}")));
        }
        if self.layers.len() == 1 {
            return Err(IngestError::Unsupported(
                "cannot remove the only layer (a document keeps at least one)".into(),
            ));
        }
        self.layers.remove(index);
        if self.active >= self.layers.len() {
            self.active = self.layers.len() - 1;
        }
        // The journal's entries are keyed by LAYER ID, and this one's
        // pixels are gone — an entry that can never be applied is worse
        // than no entry, so the whole history goes. That is the price of
        // a linear journal and it is stated in the panel rather than
        // discovered when Undo does nothing.
        self.journal.clear();
        Ok(())
    }

    /// Move `from` to `to` (both in stack order, 0 = bottom), carrying
    /// the active selection with the moved layer.
    pub fn reorder(&mut self, from: usize, to: usize) -> Result<(), IngestError> {
        if from >= self.layers.len() || to >= self.layers.len() {
            return Err(IngestError::Unsupported(format!(
                "reorder {from}→{to} outside 0..{}",
                self.layers.len()
            )));
        }
        if from == to {
            return Ok(());
        }
        let was_active = self.layers[self.active].id;
        let l = self.layers.remove(from);
        self.layers.insert(to, l);
        self.active = self
            .layers
            .iter()
            .position(|l| l.id == was_active)
            .unwrap_or(to);
        Ok(())
    }

    pub fn set_active(&mut self, index: usize) -> Result<(), IngestError> {
        if index >= self.layers.len() {
            return Err(IngestError::Unsupported(format!("no layer {index}")));
        }
        self.active = index;
        Ok(())
    }

    fn layer_mut(&mut self, index: usize) -> Result<&mut Layer, IngestError> {
        self.layers
            .get_mut(index)
            .ok_or_else(|| IngestError::Unsupported(format!("no layer {index}")))
    }

    pub fn set_visible(&mut self, index: usize, visible: bool) -> Result<(), IngestError> {
        self.layer_mut(index)?.visible = visible;
        Ok(())
    }

    pub fn set_locked(&mut self, index: usize, locked: bool) -> Result<(), IngestError> {
        self.layer_mut(index)?.locked = locked;
        Ok(())
    }

    pub fn set_opacity(&mut self, index: usize, opacity: f32) -> Result<(), IngestError> {
        self.layer_mut(index)?.opacity = opacity.clamp(0.0, 1.0);
        Ok(())
    }

    // ------------------------------------------------- layer masks
    //
    // A mask is the same `SelectionCoverage` the marquee / lasso / wand
    // already produce, so the authoring surface needed no new engine:
    // "make the selection a layer mask" is a MOVE. What is new is that
    // the compose fold finally passes something into the `@group(2)`
    // argument it always took.

    /// Attach `coverage` as `index`'s mask. Rejects a size mismatch
    /// rather than resampling: a mask that silently stretched would hide
    /// a caller bug behind plausible-looking pixels.
    pub fn set_mask(
        &mut self,
        index: usize,
        coverage: Arc<SelectionCoverage>,
    ) -> Result<(), IngestError> {
        let (w, h) = (self.width, self.height);
        if coverage.width() != w || coverage.height() != h {
            return Err(IngestError::Unsupported(format!(
                "layer mask is {}×{} but the canvas is {w}×{h} (no implicit resample)",
                coverage.width(),
                coverage.height()
            )));
        }
        let layer = self.layer_mut(index)?;
        layer.mask = Some(coverage);
        layer.mask_enabled = true;
        Ok(())
    }

    /// DELETE the mask — the coverage is gone. Distinct from disabling,
    /// which keeps it; both exist because Photoshop's users rely on the
    /// difference and losing painted coverage to a toggle is a real loss.
    pub fn clear_mask(&mut self, index: usize) -> Result<(), IngestError> {
        let layer = self.layer_mut(index)?;
        layer.mask = None;
        layer.mask_enabled = true;
        Ok(())
    }

    /// Toggle whether an attached mask applies, RETAINING it either way.
    pub fn set_mask_enabled(&mut self, index: usize, enabled: bool) -> Result<(), IngestError> {
        self.layer_mut(index)?.mask_enabled = enabled;
        Ok(())
    }

    /// Clip `index` to the layer beneath it (or release it).
    pub fn set_clipped(&mut self, index: usize, clipped: bool) -> Result<(), IngestError> {
        self.layer_mut(index)?.clipped = clipped;
        Ok(())
    }

    /// Whether `index` has a mask attached at all (enabled or not).
    pub fn has_mask(&self, index: usize) -> bool {
        self.layers.get(index).is_some_and(|l| l.mask.is_some())
    }

    pub fn set_name(&mut self, index: usize, name: &str) -> Result<(), IngestError> {
        self.layer_mut(index)?.name = name.to_string();
        Ok(())
    }

    /// Set the blend by wire name (`"multiply"` or `"compose.multiply"`)
    /// — resolved through the kernel registry, so an unknown name is a
    /// clean error rather than a silent fall back to normal.
    pub fn set_blend(&mut self, index: usize, name: &str) -> Result<(), IngestError> {
        let k = blend_kernel(name).ok_or_else(|| {
            IngestError::Unsupported(format!(
                "unknown blend mode \"{name}\" (a compose.* kernel name)"
            ))
        })?;
        self.layer_mut(index)?.blend = k;
        Ok(())
    }

    // ───────────────────────── pixel edits ──────────────────────────

    /// Replace the ACTIVE layer's pixels, journaling the tiles `damage`
    /// covers first. `pixels` must be canvas-extent.
    ///
    /// `damage` is the caller's honest damage region — the stroke's
    /// bounds, the fill's rect, the whole canvas for a bake. Undo
    /// restores exactly those tiles, so a damage region that under-claims
    /// makes undo incomplete; every caller here passes the region the
    /// engine itself computed.
    pub fn edit_active(
        &mut self,
        label: &str,
        damage: Region,
        pixels: Arc<[u8]>,
    ) -> Result<RecordOutcome, IngestError> {
        let want = (self.width as usize) * (self.height as usize) * 4;
        if pixels.len() != want {
            return Err(IngestError::Decode(format!(
                "layer edit: {} bytes for {}×{} (expected {want})",
                pixels.len(),
                self.width,
                self.height
            )));
        }
        let active = &self.layers[self.active];
        if active.locked {
            return Err(IngestError::Unsupported(format!(
                "layer \"{}\" is locked",
                active.name
            )));
        }
        let clipped = damage
            .intersect(Region::new(0, 0, self.width, self.height))
            .unwrap_or(Region::new(0, 0, 0, 0));
        let outcome = {
            let view = FlatImage::new(self.width, self.height, 4, &*active.rgba)
                .ok_or_else(|| IngestError::Decode("layer pixels are mis-sized".into()))?;
            // The layer's stable ID is the entry's SCOPE, so undo lands
            // in the layer that was painted and not in whichever one
            // happens to be selected when the user reaches for it.
            self.journal.record(label, active.id as u64, &view, clipped)
        };
        self.layers[self.active].rgba = pixels;
        Ok(outcome)
    }

    /// Is the active layer editable (present, unlocked)? The doors check
    /// this BEFORE doing GPU work so a refusal is instant and honest.
    pub fn active_is_editable(&self) -> Result<(), IngestError> {
        let a = self.active();
        if a.locked {
            return Err(IngestError::Unsupported(format!(
                "layer \"{}\" is locked — unlock it to paint on it",
                a.name
            )));
        }
        Ok(())
    }

    // ──────────────────────────── undo ──────────────────────────────

    /// Revert the newest journaled pixel edit. Returns its label, or
    /// `None` when there is nothing to undo.
    pub fn undo(&mut self) -> Option<String> {
        self.apply_history(true)
    }

    /// Replay the newest undone pixel edit.
    pub fn redo(&mut self) -> Option<String> {
        self.apply_history(false)
    }

    /// Undo/redo share everything but direction.
    ///
    /// The entry's SCOPE says which layer it belongs to — the newest
    /// edit is not necessarily on the layer that happens to be active —
    /// so the scope is resolved to a layer id FIRST, the restore lands
    /// there, and that layer becomes active so the change is visibly
    /// where it happened. The layer's shared pixels are materialized
    /// once (the journal splices into a mutable buffer) and re-shared.
    fn apply_history(&mut self, undo: bool) -> Option<String> {
        let scope = if undo {
            self.journal.undo_scope()
        } else {
            self.journal.redo_scope()
        }?;
        // A scope with no layer cannot happen — `remove` clears the
        // journal precisely so a removed layer leaves no orphan entries
        // — but if it ever did, doing nothing is the only safe answer.
        let idx = self.layers.iter().position(|l| l.id as u64 == scope)?;
        let (w, h) = (self.width, self.height);
        let mut buf: Vec<u8> = self.layers[idx].rgba.to_vec();
        let label = {
            let mut view = FlatImage::new(w, h, 4, buf.as_mut_slice())?;
            if undo {
                self.journal.undo(&mut view)
            } else {
                self.journal.redo(&mut view)
            }
        }?;
        self.layers[idx].rgba = Arc::from(buf.into_boxed_slice());
        self.active = idx;
        Some(label)
    }

    pub fn history(&self) -> HistoryStats {
        let b = self.journal.budget();
        HistoryStats {
            can_undo: self.journal.can_undo(),
            can_redo: self.journal.can_redo(),
            depth: self.journal.depth(),
            redo_depth: self.journal.redo_depth(),
            bytes: self.journal.bytes(),
            max_bytes: b.max_bytes,
            max_entries: b.max_entries,
            dropped: self.journal.dropped(),
            generation: self.journal.generation(),
        }
    }

    pub fn undo_label(&self) -> Option<&str> {
        self.journal.undo_label()
    }

    pub fn redo_label(&self) -> Option<&str> {
        self.journal.redo_label()
    }

    /// Every retained undo step, oldest first — what a History panel
    /// lists. See `TileJournal::undo_labels` on why this is the RETAINED
    /// history and not the whole session.
    pub fn undo_labels(&self) -> Vec<&str> {
        self.journal.undo_labels()
    }

    /// Every redo step, next-to-replay first.
    pub fn redo_labels(&self) -> Vec<&str> {
        self.journal.redo_labels()
    }

    /// Drop the history (a resolution change makes every tile snapshot
    /// meaningless — better to say "no history" than to restore garbage).
    pub fn clear_history(&mut self) {
        self.journal.clear();
    }

    // ───────────────────────── the composite ────────────────────────

    /// Can the composite run without a GPU? True exactly when the fast
    /// path applies (see the module docs) — the doors use it so a
    /// GPU-less realm still gets its one-layer document.
    pub fn composite_is_trivial(&self) -> bool {
        let mut plates = self.plates(None).into_iter();
        match (plates.next(), plates.next()) {
            (None, _) => true,
            (Some((l, _)), None) => l.is_plain(),
            _ => false,
        }
    }

    /// The layers that actually contribute, bottom-first, paired with
    /// the pixels to composite (the active layer's may be overridden by
    /// an in-flight stroke). Hidden, zero-opacity and fully transparent
    /// layers drop out here — each is exactly the identity in the fold.
    fn plates<'a>(
        &'a self,
        override_active: Option<&'a Arc<[u8]>>,
    ) -> Vec<(&'a Layer, &'a Arc<[u8]>)> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if !l.enabled() {
                    return None;
                }
                let px = match override_active {
                    Some(o) if i == self.active => o,
                    _ => &l.rgba,
                };
                // An ADJUSTMENT layer has no pixels of its own — its
                // `rgba` is a transparent placeholder — so the
                // transparency skip would drop exactly the layers whose
                // whole job is to change what is beneath them.
                if l.is_pixels() && is_fully_transparent(px) {
                    return None;
                }
                Some((l, px))
            })
            .collect()
    }

    /// Fold the stack bottom-up into one straight-RGBA8 canvas.
    ///
    /// `override_active` replaces the ACTIVE layer's pixels for this
    /// composite only — how a stroke in flight previews through the rest
    /// of the stack without being committed to it.
    ///
    /// GPU-only whenever there is anything to blend (every step is a
    /// registered `compose.*`/`cast.*` dispatch, spec §6); the trivial
    /// stack short-circuits before touching the device.
    pub async fn composite(
        &self,
        ctx: Option<&GpuContext>,
        override_active: Option<&Arc<[u8]>>,
    ) -> Result<Arc<[u8]>, IngestError> {
        let (w, h) = (self.width, self.height);
        let plates = self.plates(override_active);

        match plates.as_slice() {
            // Nothing contributes: an honest transparent canvas.
            [] => return Ok(self.transparent()),
            // The identity fold — the pixels ARE the composite. Handed
            // back as the very same allocation (an `Arc` clone), so a
            // one-layer document costs nothing to composite.
            [(l, px)] if l.is_plain() => return Ok(Arc::clone(px)),
            _ => {}
        }

        let ctx = ctx.ok_or_else(|| {
            IngestError::Unsupported(
                "compositing layers is GPU-only — call init_gpu first (the blend \
                 is a registered WGSL kernel dispatch; no CPU blend path ships)"
                    .into(),
            )
        })?;

        // The accumulator is PREMULTIPLIED rgba16float, starting at
        // transparent black — the compose family's `in0` contract.
        let mut acc = vec![0u8; (w as usize) * (h as usize) * 8];
        // CLIPPING: the alpha of the most recent UNCLIPPED plate, which
        // is the clip base for every clipped layer stacked directly on
        // top of it. Kept as one byte per pixel, because that is already
        // the coverage representation everything else here speaks.
        let mut clip_base: Option<Vec<u8>> = None;
        for (layer, px) in plates {
            if layer.clipped {
                // A clipped layer with NOTHING to clip to contributes
                // nothing. Compositing it unclipped would be the one
                // behaviour a designer cannot recover from — the whole
                // reason they clipped it was to confine it.
                if clip_base.is_none() {
                    continue;
                }
            } else if layer.is_pixels() {
                clip_base = Some(alpha_of(px));
            }
            // AN ADJUSTMENT LAYER transforms the backdrop instead of
            // blending over it. The accumulator is the backdrop, so the
            // chain runs on THAT and replaces it — which is precisely
            // what "non-destructive" means: no layer's own pixels are
            // touched, and removing this layer restores the result
            // exactly.
            //
            // The layer MASK becomes the chain's `selection`. The adjust
            // chain has always taken one, so a masked adjustment layer
            // needed no new plumbing — the mask a designer paints and the
            // selection a marquee makes are the same object again.
            // The clip base multiplies into the layer's own coverage,
            // so a clipped-AND-masked layer is confined by both. Two
            // coverages multiply; they do not override each other.
            let clip = if layer.clipped {
                clip_base.as_deref()
            } else {
                None
            };
            if let Some(params) = layer.adjust_params() {
                let straight8 = f16_to_rgba8(&unpremultiply(ctx, &acc, w, h).await?);
                let image = DecodedImage {
                    width: w,
                    height: h,
                    rgba: Arc::from(straight8.into_boxed_slice()),
                    // Post-ingest pixels: the display transform already ran.
                    display: crate::display::DisplayTreatment::AssumedSrgb,
                };
                let adjusted = adjust_rgba8(
                    ctx,
                    &image,
                    params,
                    effective_coverage(layer.live_mask(), clip, w, h),
                )
                .await?;
                acc = premultiply(ctx, &rgba8_to_f16(&adjusted), w, h).await?;
                continue;
            }

            let straight = rgba8_to_f16(px);
            // `premultiply` over a fully-opaque window is `rgb·1` — the
            // identity, provably; skip the round-trip there.
            let premul = if window_is_opaque(&straight) {
                straight
            } else {
                dispatch_unary(
                    ctx,
                    &CAST_PREMULTIPLY,
                    CastPremultiplyParams::new().as_bytes(),
                    &straight,
                    w,
                    h,
                )
                .await?
            };
            // Lower the layer's coverage to the ABI mask. `None` keeps
            // the constant-1 fast path, so an unmasked layer pays nothing.
            let mask_bytes: Option<Vec<u8>> = effective_coverage(layer.live_mask(), clip, w, h)
                .map(|cov| {
                    SelectionMask::from_fn(w, h, |x, y| f32::from(cov.coverage_at(x, y)) / 255.0)
                        .bytes()
                        .to_vec()
                });
            acc = image_gpu::execute_tile_once_async(
                ctx,
                layer.blend,
                &[
                    TileInput { f16_bytes: &acc },
                    TileInput { f16_bytes: &premul },
                ],
                // The layer's opacity IS the compose family's α: the
                // spine computes `over(a, b·α)`.
                ComposeParams::new(layer.opacity).as_bytes(),
                // THE LAYER MASK. The compose spine already took a mask
                // here and every caller passed `None`; a masked layer is
                // that argument finally carrying something. Lowered to
                // r16float per dispatch, exactly like a selection.
                mask_bytes.as_deref(),
                w,
                h,
            )
            .await
            .map_err(|e| IngestError::Pipeline(e.to_string()))?;
        }

        // …and back out of premultiplied space, once — skipped where the
        // result is fully opaque and the division is by one.
        let out = if window_is_opaque(&acc) {
            acc
        } else {
            dispatch_unary(
                ctx,
                &CAST_UNPREMULTIPLY,
                CastUnpremultiplyParams::new().as_bytes(),
                &acc,
                w,
                h,
            )
            .await?
        };
        Ok(Arc::from(f16_to_rgba8(&out).into_boxed_slice()))
    }
}

/// The `compose.*` kernel for a PSD blend-mode fourcc.
///
/// All 26 registered blend modes have a PSD key and all 26 are mapped;
/// keys outside that set (`diss` dissolve, `pass` group pass-through,
/// and Photoshop's own additions) fall back to `normal` — the honest
/// approximation, and one the panel makes visible because the layer's
/// blend then READS as "normal" rather than silently claiming otherwise.
///
/// Provenance: Adobe Photoshop File Format specification — Layer
/// Records, "Blend mode key"; the operator semantics are W3C
/// Compositing and Blending Level 1, which is what the `compose.*`
/// kernels implement.
pub fn psd_blend_kernel(key: &[u8; 4]) -> &'static KernelDef {
    let name = match key {
        b"norm" => "normal",
        b"mul " => "multiply",
        b"scrn" => "screen",
        b"over" => "overlay",
        b"dark" => "darken",
        b"lite" => "lighten",
        b"div " => "color_dodge",
        b"idiv" => "color_burn",
        b"hLit" => "hard_light",
        b"sLit" => "soft_light",
        b"diff" => "difference",
        b"smud" => "exclusion",
        b"hue " => "hue",
        b"sat " => "saturation",
        b"colr" => "color",
        b"lum " => "luminosity",
        b"lbrn" => "linear_burn",
        b"lddg" => "linear_dodge",
        b"dkCl" => "darker_color",
        b"lgCl" => "lighter_color",
        b"vLit" => "vivid_light",
        b"lLit" => "linear_light",
        b"pLit" => "pin_light",
        b"hMix" => "hard_mix",
        b"fsub" => "subtract",
        b"fdiv" => "divide",
        _ => return &COMPOSE_NORMAL,
    };
    blend_kernel(name).unwrap_or(&COMPOSE_NORMAL)
}

/// One whole-image UNARY dispatch (the `cast.*` bracket steps) — the
/// same shape `crate::fill` uses.
async fn dispatch_unary(
    ctx: &GpuContext,
    def: &'static KernelDef,
    params: &[u8],
    src_f16: &[u8],
    w: u32,
    h: u32,
) -> Result<Vec<u8>, IngestError> {
    image_gpu::execute_tile_once_async(
        ctx,
        def,
        &[TileInput { f16_bytes: src_f16 }],
        params,
        None,
        w,
        h,
    )
    .await
    .map_err(|e| IngestError::Pipeline(e.to_string()))
}

/// Premultiply a straight f16 RGBA window (the compose family's `in0`
/// contract). Identity over a fully-opaque window, so that is skipped.
async fn premultiply(
    ctx: &GpuContext,
    straight: &[u8],
    w: u32,
    h: u32,
) -> Result<Vec<u8>, IngestError> {
    if window_is_opaque(straight) {
        return Ok(straight.to_vec());
    }
    dispatch_unary(
        ctx,
        &CAST_PREMULTIPLY,
        CastPremultiplyParams::new().as_bytes(),
        straight,
        w,
        h,
    )
    .await
}

/// The inverse — out of premultiplied space, skipped where the division
/// is by one.
async fn unpremultiply(
    ctx: &GpuContext,
    premul: &[u8],
    w: u32,
    h: u32,
) -> Result<Vec<u8>, IngestError> {
    if window_is_opaque(premul) {
        return Ok(premul.to_vec());
    }
    dispatch_unary(
        ctx,
        &CAST_UNPREMULTIPLY,
        CastUnpremultiplyParams::new().as_bytes(),
        premul,
        w,
        h,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn px(w: u32, h: u32, v: u8) -> Arc<[u8]> {
        Arc::from(vec![v; (w * h * 4) as usize].into_boxed_slice())
    }

    fn stack(w: u32, h: u32) -> LayerStack {
        LayerStack::from_image(w, h, px(w, h, 128)).expect("valid")
    }

    fn device() -> Option<&'static GpuContext> {
        use std::sync::OnceLock;
        static DEVICE: OnceLock<Option<GpuContext>> = OnceLock::new();
        DEVICE
            .get_or_init(|| match pollster::block_on(GpuContext::new()) {
                Ok(ctx) => Some(ctx),
                Err(e) => {
                    eprintln!("layers GPU unavailable: {e} — device tests will skip");
                    None
                }
            })
            .as_ref()
    }

    // ── smart objects ────────────────────────────────────────────────

    #[test]
    fn image_editor_layers_converting_to_smart_preserves_the_pixels_as_source() {
        let mut s = stack(4, 4);
        let before = s.active().rgba.to_vec();
        s.make_smart(0).expect("convert");
        assert!(s.is_smart(0));
        let src = s.layers[0].smart_source().expect("source");
        assert_eq!(src.rgba.to_vec(), before, "the source IS the pixels it had");
        assert_eq!(src.scale, 1.0);
        assert!(
            s.layers[0].is_pixels(),
            "and it still contributes pixels to the fold"
        );
    }

    #[test]
    fn image_editor_layers_making_smart_twice_is_idempotent() {
        // Not an error, and — critically — it must not re-capture the
        // CURRENT render as the new source, which would quietly bake in
        // whatever scaling had happened.
        let mut s = stack(4, 4);
        s.make_smart(0).expect("first");
        s.set_smart_render(0, px(4, 4, 200), 0.25).expect("render");
        s.make_smart(0).expect("second is a no-op");
        let src = s.layers[0].smart_source().expect("source");
        assert_eq!(src.scale, 0.25, "the recorded scale survived");
        assert!(
            src.rgba.iter().all(|&b| b == 128),
            "and the source is still the ORIGINAL, not the 0.25 render"
        );
    }

    #[test]
    fn image_editor_layers_an_adjustment_layer_cannot_become_smart() {
        // It has no pixels to preserve, so the conversion would invent a
        // source out of a transparent placeholder.
        let mut s = stack(4, 4);
        let at = s.add_adjustment("Brighten", bright(0.2));
        let err = s.make_smart(at).expect_err("refused");
        assert!(err.to_string().contains("no pixels to preserve"));
    }

    #[test]
    fn image_editor_layers_rescaling_a_smart_object_is_lossless() {
        // THE property, stated as a test. Scale down hard, then back up:
        // a pixel layer would have lost the information for good, but the
        // smart object re-renders FROM SOURCE, so what the caller renders
        // at 1.0 is derived from the original bytes — not from the 0.1
        // render. The stack's job is to never replace the source, and
        // that is what this asserts.
        let mut s = stack(8, 8);
        let original = s.active().rgba.to_vec();
        s.make_smart(0).expect("convert");

        // A brutal round trip through the cache.
        s.set_smart_render(0, px(8, 8, 3), 0.1).expect("down");
        s.set_smart_render(0, px(8, 8, 250), 1.0).expect("back up");

        let src = s.layers[0].smart_source().expect("source");
        assert_eq!(
            src.rgba.to_vec(),
            original,
            "the source survived both renders untouched — this is what makes \
             the round trip lossless, since the next render reads it and not \
             the 0.1 cache"
        );
    }

    #[test]
    fn image_editor_layers_a_smart_render_must_match_the_canvas() {
        let mut s = stack(4, 4);
        s.make_smart(0).expect("convert");
        let err = s
            .set_smart_render(0, px(2, 2, 0), 0.5)
            .expect_err("size mismatch refused");
        assert!(err.to_string().contains("canvas needs"));
    }

    #[test]
    fn image_editor_layers_only_a_smart_object_takes_a_smart_render() {
        let mut s = stack(4, 4);
        let err = s
            .set_smart_render(0, px(4, 4, 1), 0.5)
            .expect_err("a plain pixel layer refuses");
        assert!(err.to_string().contains("not a smart object"));
    }

    #[test]
    fn image_editor_layers_a_smart_object_composites_like_any_other() {
        // Being smart changes where its pixels COME FROM, not how they
        // blend — so the fold must treat it exactly like a pixel layer.
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 16);
        let plain = pollster::block_on(s.composite(Some(ctx), None)).expect("plain");
        s.make_smart(0).expect("convert");
        let smart = pollster::block_on(s.composite(Some(ctx), None)).expect("smart");
        assert_eq!(
            smart.to_vec(),
            plain.to_vec(),
            "conversion alone changes no pixel"
        );
    }

    // ── adjustment layers ────────────────────────────────────────────

    fn bright(v: f32) -> AdjustParams {
        AdjustParams {
            brightness: v,
            ..Default::default()
        }
    }

    #[test]
    fn image_editor_layers_an_adjustment_layer_carries_no_pixels() {
        let mut s = stack(4, 4);
        let at = s.add_adjustment("Brighten", bright(0.25));
        assert!(s.is_adjustment(at));
        assert!(!s.layers[at].is_pixels(), "it holds params, not pixels");
        assert!(
            !s.layers[at].is_plain(),
            "and it is never a pass-through — the fold must visit it"
        );
    }

    #[test]
    fn image_editor_layers_retuning_a_pixel_layer_as_an_adjustment_is_refused() {
        // Converting would discard pixels, which is the single thing this
        // feature exists to avoid.
        let mut s = stack(4, 4);
        let err = s
            .set_adjustment(0, bright(0.5))
            .expect_err("a pixel layer is not retunable");
        assert!(err.to_string().contains("holds pixels"));
    }

    #[test]
    fn image_editor_layers_an_adjustment_layer_survives_the_transparency_skip() {
        // Its `rgba` IS transparent — the skip that drops empty pixel
        // layers must not drop it, or a brightness layer would silently
        // do nothing.
        let mut s = stack(4, 4);
        s.add_adjustment("Brighten", bright(0.25));
        assert!(
            !s.composite_is_trivial(),
            "the stack is no longer a trivial one-layer fold"
        );
    }

    #[test]
    fn image_editor_layers_an_adjustment_layer_changes_the_composite() {
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 16);
        let base = pollster::block_on(s.composite(Some(ctx), None)).expect("base");
        s.add_adjustment("Brighten", bright(0.25));
        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("adjusted");
        assert_ne!(
            out.to_vec(),
            base.to_vec(),
            "the adjustment layer transformed the backdrop beneath it"
        );
        assert!(
            out[0] > base[0],
            "and brightened it: {} should exceed {}",
            out[0],
            base[0]
        );
    }

    #[test]
    fn image_editor_layers_removing_an_adjustment_layer_restores_exactly() {
        // THE non-destructive claim, stated as a test: the pixels beneath
        // were never written, so deleting the adjustment returns the
        // original byte-for-byte — not approximately.
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 16);
        let base = pollster::block_on(s.composite(Some(ctx), None)).expect("base");
        let at = s.add_adjustment("Brighten", bright(0.4));
        let _ = pollster::block_on(s.composite(Some(ctx), None)).expect("adjusted");
        s.remove(at).expect("remove");
        let after = pollster::block_on(s.composite(Some(ctx), None)).expect("restored");
        assert_eq!(
            after.to_vec(),
            base.to_vec(),
            "removing an adjustment layer restores the original exactly"
        );
    }

    #[test]
    fn image_editor_layers_hiding_an_adjustment_layer_is_the_identity() {
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 16);
        let base = pollster::block_on(s.composite(Some(ctx), None)).expect("base");
        let at = s.add_adjustment("Brighten", bright(0.4));
        s.set_visible(at, false).expect("hide");
        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("hidden");
        assert_eq!(
            out.to_vec(),
            base.to_vec(),
            "a hidden adjustment does nothing"
        );
    }

    #[test]
    fn image_editor_layers_a_masked_adjustment_applies_only_inside_the_mask() {
        // The two rungs meeting: the layer mask becomes the adjust
        // chain's selection, so the right half must be untouched.
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 16);
        let base = pollster::block_on(s.composite(Some(ctx), None)).expect("base");
        let at = s.add_adjustment("Brighten", bright(0.5));
        let half = SelectionCoverage::rasterize_rect(16, 16, 0.0, 0.0, 8.0, 16.0);
        s.set_mask(at, Arc::new(half)).expect("mask it");

        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("masked");
        let at_px = |buf: &[u8], x: usize, y: usize| buf[(y * 16 + x) * 4];
        assert!(
            at_px(&out, 2, 8) > at_px(&base, 2, 8),
            "inside the mask the adjustment applied"
        );
        assert_eq!(
            at_px(&out, 13, 8),
            at_px(&base, 13, 8),
            "outside it the pixels are untouched"
        );
    }

    // ── layer masks ──────────────────────────────────────────────────

    #[test]
    fn image_editor_layers_a_new_layer_has_no_mask() {
        let s = stack(4, 4);
        assert!(!s.has_mask(0), "a fresh layer is unmasked");
        assert!(
            s.layers[0].live_mask().is_none(),
            "and nothing is lowered for it"
        );
    }

    #[test]
    fn image_editor_layers_a_mask_must_match_the_canvas() {
        let mut s = stack(4, 4);
        let wrong = Arc::new(SelectionCoverage::full(2, 2));
        let err = s.set_mask(0, wrong).expect_err("size mismatch is rejected");
        // The message must name both extents — a resample would be the
        // silent alternative and is exactly what this refuses.
        let msg = err.to_string();
        assert!(msg.contains("2×2"), "names the mask extent: {msg}");
        assert!(msg.contains("4×4"), "names the canvas extent: {msg}");
        assert!(!s.has_mask(0), "and nothing was attached");
    }

    #[test]
    fn image_editor_layers_disabling_a_mask_retains_it() {
        let mut s = stack(4, 4);
        s.set_mask(0, Arc::new(SelectionCoverage::empty(4, 4)))
            .expect("attach");
        assert!(s.has_mask(0));
        assert!(s.layers[0].live_mask().is_some(), "an empty mask applies");

        s.set_mask_enabled(0, false).expect("disable");
        assert!(
            s.has_mask(0),
            "DISABLED is not DELETED — the coverage stays"
        );
        assert!(
            s.layers[0].live_mask().is_none(),
            "but it does not apply while disabled"
        );

        s.set_mask_enabled(0, true).expect("re-enable");
        assert!(
            s.layers[0].live_mask().is_some(),
            "and re-enabling restores the same coverage"
        );
    }

    #[test]
    fn image_editor_layers_clearing_a_mask_deletes_it() {
        let mut s = stack(4, 4);
        s.set_mask(0, Arc::new(SelectionCoverage::empty(4, 4)))
            .expect("attach");
        s.clear_mask(0).expect("clear");
        assert!(!s.has_mask(0), "cleared means gone, not disabled");
    }

    #[test]
    fn image_editor_layers_an_all_one_mask_is_the_identity() {
        // Materializing a constant-one mask would cost an upload to
        // change nothing, so it must not count as "masked" — and the
        // layer must stay eligible for the plain-fold fast path.
        let mut s = stack(4, 4);
        s.set_mask(0, Arc::new(SelectionCoverage::full(4, 4)))
            .expect("attach");
        assert!(s.has_mask(0), "it IS attached");
        assert!(
            s.layers[0].live_mask().is_none(),
            "but an all-one mask lowers to nothing"
        );
        assert!(
            s.layers[0].is_plain(),
            "so the identity short-circuit still applies"
        );
    }

    #[test]
    fn image_editor_layers_a_real_mask_defeats_the_plain_fast_path() {
        // The inverse of the test above, and the one that matters: a
        // layer with a live mask must NOT be treated as plain, or the
        // fold would hand back its pixels verbatim and drop the mask.
        let mut s = stack(4, 4);
        assert!(s.layers[0].is_plain(), "unmasked and default: plain");
        s.set_mask(0, Arc::new(SelectionCoverage::empty(4, 4)))
            .expect("attach");
        assert!(
            !s.layers[0].is_plain(),
            "a masked layer is never plain, whatever its opacity and blend"
        );
        assert!(
            !s.composite_is_trivial(),
            "and the whole composite stops being trivial"
        );
    }

    #[test]
    fn image_editor_layers_duplicating_carries_the_mask() {
        // A duplicate that lost its mask would reveal what the original
        // hides — a silent content leak, not a cosmetic difference.
        let mut s = stack(4, 4);
        s.set_mask(0, Arc::new(SelectionCoverage::empty(4, 4)))
            .expect("attach");
        s.duplicate(0).expect("duplicate");
        assert_eq!(s.len(), 2);
        assert!(s.has_mask(0) && s.has_mask(1), "both carry the mask");
    }

    // ── the model ────────────────────────────────────────────────────

    #[test]
    fn image_editor_layers_an_ingested_image_opens_as_one_background_layer() {
        let s = stack(8, 8);
        assert_eq!(s.len(), 1);
        assert_eq!(s.active_index(), 0);
        let l = s.active();
        assert_eq!(l.name, BACKGROUND_LAYER_NAME);
        assert!(l.visible && !l.locked);
        assert_eq!(l.opacity, 1.0);
        assert_eq!(l.blend_name(), "normal");
    }

    #[test]
    fn image_editor_layers_opening_a_stack_shares_the_pixels() {
        // O(1): the background layer must not clone the ingest buffer.
        let pixels = px(64, 64, 3);
        let s = LayerStack::from_image(64, 64, Arc::clone(&pixels)).expect("valid");
        assert!(Arc::ptr_eq(&s.active().rgba, &pixels));
    }

    #[test]
    fn image_editor_layers_a_mis_sized_buffer_is_a_clean_error() {
        assert!(LayerStack::from_image(4, 4, px(2, 2, 0)).is_err());
        assert!(LayerStack::from_image(0, 0, px(0, 0, 0)).is_err());
    }

    #[test]
    fn image_editor_layers_add_puts_a_transparent_layer_above_the_active_one() {
        let mut s = stack(4, 4);
        let at = s.add("Paint");
        assert_eq!(at, 1, "above the background, not below it");
        assert_eq!(s.len(), 2);
        assert_eq!(s.active_index(), 1, "and becomes active");
        assert_eq!(s.active().name, "Paint");
        assert!(s.active().rgba.iter().all(|&b| b == 0), "transparent");
        // Ids are stable and unique.
        assert_ne!(s.layers()[0].id, s.layers()[1].id);
    }

    #[test]
    fn image_editor_layers_reorder_carries_the_active_selection() {
        let mut s = stack(4, 4);
        s.add("A");
        s.add("B"); // [bg, A, B], active = B (index 2)
        assert_eq!(s.active().name, "B");
        s.reorder(2, 0).expect("in range");
        assert_eq!(
            s.layers()
                .iter()
                .map(|l| l.name.as_str())
                .collect::<Vec<_>>(),
            vec!["B", "Background", "A"]
        );
        assert_eq!(s.active().name, "B", "the moved layer stays active");
        assert!(s.reorder(0, 9).is_err());
    }

    #[test]
    fn image_editor_layers_the_last_layer_cannot_be_removed() {
        let mut s = stack(4, 4);
        assert!(s.remove(0).is_err(), "a document keeps at least one layer");
        s.add("A");
        assert!(s.remove(1).is_ok());
        assert_eq!(s.len(), 1);
        assert_eq!(s.active_index(), 0, "the active index follows the removal");
    }

    #[test]
    fn image_editor_layers_blend_is_resolved_through_the_kernel_registry() {
        let mut s = stack(4, 4);
        s.set_blend(0, "multiply").expect("registered");
        assert_eq!(s.layers()[0].blend.id, "compose.multiply");
        s.set_blend(0, "compose.screen")
            .expect("qualified id works");
        assert_eq!(s.layers()[0].blend_name(), "screen");
        assert!(
            s.set_blend(0, "dissolve").is_err(),
            "an unregistered mode is a clean error, never a silent normal"
        );
    }

    #[test]
    fn image_editor_layers_opacity_is_clamped() {
        let mut s = stack(4, 4);
        s.set_opacity(0, 5.0).expect("in range");
        assert_eq!(s.layers()[0].opacity, 1.0);
        s.set_opacity(0, -1.0).expect("in range");
        assert_eq!(s.layers()[0].opacity, 0.0);
    }

    #[test]
    fn image_editor_layers_a_locked_layer_refuses_pixel_edits_but_not_properties() {
        let mut s = stack(4, 4);
        s.set_locked(0, true).expect("in range");
        assert!(s.active_is_editable().is_err());
        assert!(s
            .edit_active("paint", Region::new(0, 0, 4, 4), px(4, 4, 9))
            .is_err());
        // …properties still move (that is what "lock the PIXELS" means).
        assert!(s.set_opacity(0, 0.5).is_ok());
        assert!(s.set_name(0, "Locked").is_ok());
    }

    // ── the PSD lane ─────────────────────────────────────────────────

    #[test]
    fn image_editor_layers_every_psd_blend_key_maps_to_a_registered_kernel() {
        // All 26 compose kernels have a PSD key and all 26 are reachable
        // — the mapping cannot silently collapse modes into normal.
        const KEYS: [&[u8; 4]; 26] = [
            b"norm", b"mul ", b"scrn", b"over", b"dark", b"lite", b"div ", b"idiv", b"hLit",
            b"sLit", b"diff", b"smud", b"hue ", b"sat ", b"colr", b"lum ", b"lbrn", b"lddg",
            b"dkCl", b"lgCl", b"vLit", b"lLit", b"pLit", b"hMix", b"fsub", b"fdiv",
        ];
        let mut seen: Vec<&str> = KEYS.iter().map(|k| psd_blend_kernel(k).id).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 26, "every key maps to a DISTINCT kernel");
        // An unmodeled key (dissolve, group pass-through) falls back to
        // normal — the honest approximation, and a visible one.
        assert_eq!(psd_blend_kernel(b"diss").id, "compose.normal");
        assert_eq!(psd_blend_kernel(b"pass").id, "compose.normal");
    }

    #[test]
    fn image_editor_layers_a_psd_import_becomes_a_stack_bottom_first() {
        let plate = |name: &str, key: &[u8; 4], opacity: u8, hidden: bool| image_psd::LayerPlate {
            name: name.to_string(),
            blend_key: *key,
            opacity,
            hidden,
            rgba: vec![0u8; 4 * 4 * 4],
        };
        let import = image_psd::LayerImport {
            width: 4,
            height: 4,
            layers: vec![
                plate("base", b"norm", 255, false),
                plate("mult", b"mul ", 128, true),
            ],
        };
        let s = LayerStack::from_psd_plates(&import).expect("well-formed");
        assert_eq!(s.len(), 2);
        assert_eq!(s.layers()[0].name, "base");
        assert_eq!(s.layers()[1].name, "mult");
        assert_eq!(s.layers()[1].blend_name(), "multiply");
        assert!((s.layers()[1].opacity - 128.0 / 255.0).abs() < 1e-6);
        assert!(!s.layers()[1].visible, "the PSD hidden flag carries");
        assert_eq!(s.active_index(), 1, "the TOP layer starts active");
    }

    #[test]
    fn image_editor_layers_a_psd_import_with_a_mis_sized_plate_is_a_clean_error() {
        let import = image_psd::LayerImport {
            width: 4,
            height: 4,
            layers: vec![image_psd::LayerPlate {
                name: "short".into(),
                blend_key: *b"norm",
                opacity: 255,
                hidden: false,
                rgba: vec![0u8; 8],
            }],
        };
        assert!(LayerStack::from_psd_plates(&import).is_err());
    }

    // ── the composite ────────────────────────────────────────────────

    #[test]
    fn image_editor_layers_a_plain_single_layer_composites_to_itself_without_a_gpu() {
        let s = stack(8, 8);
        let out = pollster::block_on(s.composite(None, None)).expect("no device needed");
        assert!(
            Arc::ptr_eq(&out, &s.active().rgba),
            "the identity fold returns the very same buffer"
        );
        assert!(s.composite_is_trivial());
    }

    #[test]
    fn image_editor_layers_a_hidden_only_layer_composites_to_transparent() {
        let mut s = stack(4, 4);
        s.set_visible(0, false).expect("in range");
        assert!(s.composite_is_trivial());
        let out = pollster::block_on(s.composite(None, None)).expect("no device needed");
        assert!(out.iter().all(|&b| b == 0));
    }

    #[test]
    fn image_editor_layers_an_empty_layer_keeps_the_composite_trivial() {
        // "Add layer" is the first thing anyone does. A fully
        // TRANSPARENT layer is exactly the identity in the fold
        // (`alpha_s = 0` leaves the backdrop for every blend mode), so
        // it is skipped — the composite stays GPU-free until something
        // is actually painted into it.
        let mut s = stack(4, 4);
        s.add("Paint");
        assert!(s.composite_is_trivial());
        let out = pollster::block_on(s.composite(None, None)).expect("no device needed");
        assert!(Arc::ptr_eq(&out, &s.layers()[0].rgba));
    }

    #[test]
    fn image_editor_layers_a_second_painted_layer_makes_the_composite_gpu_only() {
        let mut s = stack(4, 4);
        s.add("Paint");
        s.edit_active("fill", Region::new(0, 0, 4, 4), px(4, 4, 200))
            .expect("unlocked");
        assert!(!s.composite_is_trivial());
        // …and says so rather than inventing a CPU blend.
        let err = pollster::block_on(s.composite(None, None)).expect_err("GPU-only");
        assert!(format!("{err}").contains("GPU-only"));
    }

    #[test]
    fn image_editor_layers_a_transparent_layer_on_top_leaves_the_backdrop_alone() {
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 16);
        let base = s.active().rgba.to_vec();
        s.add("Empty");
        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("composite");
        assert_eq!(
            out.to_vec(),
            base,
            "source-over with a fully transparent source is the identity"
        );
    }

    #[test]
    fn image_editor_layers_a_zero_mask_hides_the_layer_it_masks() {
        // The load-bearing claim: an opaque white layer that WOULD cover
        // the backdrop is fully suppressed by an all-zero mask. Same
        // stack as the test below, one call different, opposite result.
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 16);
        let base = s.active().rgba.to_vec();
        s.add("Cover");
        let white: Arc<[u8]> = px(16, 16, 255);
        s.edit_active("fill", Region::new(0, 0, 16, 16), Arc::clone(&white))
            .expect("unlocked");
        s.set_mask(1, Arc::new(SelectionCoverage::empty(16, 16)))
            .expect("attach a zero mask");

        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("composite");
        assert_eq!(
            out.to_vec(),
            base,
            "a zero-coverage mask makes the layer contribute nothing"
        );
    }

    #[test]
    fn image_editor_layers_a_disabled_mask_stops_hiding_it() {
        // Proves the enable flag reaches the GPU, not just the model: the
        // same zero mask, disabled, lets the cover layer through again.
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 16);
        s.add("Cover");
        let white: Arc<[u8]> = px(16, 16, 255);
        s.edit_active("fill", Region::new(0, 0, 16, 16), Arc::clone(&white))
            .expect("unlocked");
        s.set_mask(1, Arc::new(SelectionCoverage::empty(16, 16)))
            .expect("attach");
        s.set_mask_enabled(1, false).expect("disable");

        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("composite");
        assert!(
            out.iter().all(|&b| b == 255),
            "with the mask disabled the opaque layer covers again"
        );
    }

    #[test]
    fn image_editor_layers_a_half_mask_covers_half_the_canvas() {
        // A partial mask is the real case, and the one a boolean
        // "masked/unmasked" implementation would pass the two tests above
        // while failing: the left half must be covered and the right half
        // must be the backdrop.
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 16);
        let base = s.active().rgba.to_vec();
        s.add("Cover");
        let white: Arc<[u8]> = px(16, 16, 255);
        s.edit_active("fill", Region::new(0, 0, 16, 16), Arc::clone(&white))
            .expect("unlocked");
        // Left half selected, right half not.
        let half = SelectionCoverage::rasterize_rect(16, 16, 0.0, 0.0, 8.0, 16.0);
        s.set_mask(1, Arc::new(half)).expect("attach");

        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("composite");
        let at = |x: usize, y: usize| out[(y * 16 + x) * 4];
        assert_eq!(at(2, 8), 255, "inside the mask the cover layer shows");
        assert_eq!(
            at(13, 8),
            base[(8 * 16 + 13) * 4],
            "outside it the backdrop survives untouched"
        );
    }

    #[test]
    fn image_editor_layers_an_opaque_layer_on_top_hides_the_one_below() {
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 16);
        s.add("Cover");
        let white: Arc<[u8]> = px(16, 16, 255);
        s.edit_active("fill", Region::new(0, 0, 16, 16), Arc::clone(&white))
            .expect("unlocked");
        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("composite");
        assert!(out.iter().all(|&b| b == 255));
    }

    #[test]
    fn image_editor_layers_opacity_rides_the_compose_param_block() {
        let Some(ctx) = device() else { return };
        // Black backdrop, white layer at 50%: the result is mid-grey.
        let mut s = LayerStack::from_image(8, 8, px(8, 8, 0)).expect("valid");
        // …but the backdrop's ALPHA must be opaque, or "over" is not
        // what we are measuring. px(8,8,0) is transparent black, so set
        // an opaque black background explicitly.
        let mut opaque_black = vec![0u8; 8 * 8 * 4];
        for p in opaque_black.chunks_exact_mut(4) {
            p[3] = 255;
        }
        s.edit_active(
            "base",
            Region::new(0, 0, 8, 8),
            Arc::from(opaque_black.into_boxed_slice()),
        )
        .expect("unlocked");
        s.add("White");
        s.edit_active("fill", Region::new(0, 0, 8, 8), px(8, 8, 255))
            .expect("unlocked");
        s.set_opacity(1, 0.5).expect("in range");
        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("composite");
        for t in out.chunks_exact(4) {
            assert!(
                (t[0] as i32 - 128).abs() <= 2,
                "50% white over black should be ~128, got {}",
                t[0]
            );
            assert_eq!(t[3], 255, "…and stay opaque");
        }
    }

    #[test]
    fn image_editor_layers_multiply_darkens_through_the_registered_kernel() {
        let Some(ctx) = device() else { return };
        let mut s = LayerStack::from_image(8, 8, px(8, 8, 255)).expect("valid");
        s.add("Half");
        let mut half = vec![128u8; 8 * 8 * 4];
        for p in half.chunks_exact_mut(4) {
            p[3] = 255;
        }
        s.edit_active(
            "fill",
            Region::new(0, 0, 8, 8),
            Arc::from(half.into_boxed_slice()),
        )
        .expect("unlocked");
        s.set_blend(1, "multiply").expect("registered");
        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("composite");
        for t in out.chunks_exact(4) {
            assert!(
                (t[0] as i32 - 128).abs() <= 2,
                "white × 50% grey is 50% grey, got {}",
                t[0]
            );
        }
    }

    #[test]
    fn image_editor_layers_the_stroke_override_previews_through_the_stack() {
        let Some(ctx) = device() else { return };
        let mut s = stack(8, 8);
        s.add("Paint");
        // The override stands in for the active layer WITHOUT committing.
        let white: Arc<[u8]> = px(8, 8, 255);
        let out = pollster::block_on(s.composite(Some(ctx), Some(&white))).expect("composite");
        assert!(out.iter().all(|&b| b == 255), "the override composited");
        assert!(
            s.active().rgba.iter().all(|&b| b == 0),
            "…and the layer itself was never touched"
        );
    }

    // ── the deliverable, end to end ──────────────────────────────────

    #[test]
    fn image_editor_layers_a_stroke_lands_in_the_active_layer_and_undoes() {
        // THE caveat this whole thing exists to retire, proven on the
        // device: paint on a layer ABOVE the photo, and (1) the photo's
        // own pixels are untouched, (2) the composite shows the paint,
        // (3) Undo takes it back exactly. Same lane the wasm door drives
        // (`StrokeSession::begin_on` over the active layer's pixels).
        let Some(ctx) = device() else { return };
        use crate::stroke::{StrokeParams, StrokeSession, StrokeTool};
        use image_gpu::dab::StrokeSample;

        let (w, h) = (64u32, 64u32);
        // An OPAQUE photo layer, so the composite is well defined.
        let mut photo = vec![0u8; (w * h * 4) as usize];
        for p in photo.chunks_exact_mut(4) {
            p[0] = 40;
            p[1] = 90;
            p[2] = 160;
            p[3] = 255;
        }
        let photo: Arc<[u8]> = Arc::from(photo.into_boxed_slice());
        let mut s = LayerStack::from_image(w, h, Arc::clone(&photo)).expect("valid");
        s.add("Paint");
        assert_eq!(s.active_index(), 1);

        // Paint a red dot into the ACTIVE (empty, top) layer.
        let mut params = StrokeParams::defaults(StrokeTool::Brush);
        params.color = [1.0, 0.0, 0.0, 1.0];
        params.hardness = 1.0;
        let mut stroke =
            StrokeSession::begin_on(1, w, h, Arc::clone(&s.active().rgba), params, None)
                .expect("begin on the layer");
        pollster::block_on(stroke.extend(ctx, StrokeSample::new(32.0, 32.0, 1.0))).expect("extend");
        let damage = stroke.stroke_bounds().expect("a dot has bounds");
        let painted: Arc<[u8]> = Arc::from(stroke.commit().into_boxed_slice());

        let composite_before = pollster::block_on(s.composite(Some(ctx), None)).expect("fold");
        s.edit_active("Paint", damage, painted).expect("unlocked");

        // (1) the photo layer never moved.
        assert!(
            Arc::ptr_eq(&s.layers()[0].rgba, &photo),
            "the layer below is untouched — not merely equal, the same buffer"
        );
        // (2) the composite shows the paint at the dab centre and the
        //     photo in the corner the dab never reached.
        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("fold");
        let at = |buf: &[u8], x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
        };
        let centre = at(&out, 32, 32);
        assert!(
            centre[0] > 200 && centre[1] < 60 && centre[2] < 60,
            "the dab should read red over the photo, got {centre:?}"
        );
        assert_eq!(at(&out, 0, 0), at(&photo, 0, 0), "the corner is the photo");

        // (3) undo restores the layer and therefore the composite.
        assert_eq!(s.undo().as_deref(), Some("Paint"));
        let after_undo = pollster::block_on(s.composite(Some(ctx), None)).expect("fold");
        assert_eq!(
            after_undo.to_vec(),
            composite_before.to_vec(),
            "undo returns the composite byte for byte"
        );
        // …and redo puts it back.
        assert_eq!(s.redo().as_deref(), Some("Paint"));
        let after_redo = pollster::block_on(s.composite(Some(ctx), None)).expect("fold");
        assert_eq!(after_redo.to_vec(), out.to_vec());
    }

    // ── the journal, through the stack ───────────────────────────────

    #[test]
    fn image_editor_undo_a_layer_edit_is_reversible_byte_for_byte() {
        let mut s = stack(300, 200);
        let before = s.active().rgba.to_vec();
        let mut painted = before.clone();
        for p in painted.chunks_exact_mut(4) {
            p[0] = 200;
        }
        s.edit_active(
            "Paint",
            Region::new(0, 0, 300, 200),
            Arc::from(painted.clone().into_boxed_slice()),
        )
        .expect("unlocked");
        assert_eq!(s.active().rgba.to_vec(), painted);

        assert_eq!(s.undo().as_deref(), Some("Paint"));
        assert_eq!(s.active().rgba.to_vec(), before, "byte-for-byte");
        assert_eq!(s.redo().as_deref(), Some("Paint"));
        assert_eq!(s.active().rgba.to_vec(), painted);
        assert!(s.undo().is_some());
        assert!(s.undo().is_none(), "nothing left to undo");
    }

    #[test]
    fn image_editor_undo_a_small_edit_journals_only_the_tiles_it_covered() {
        // 1024×1024 = 16 tiles; a stroke in one corner journals ONE.
        let mut s = stack(1024, 1024);
        let painted = s.active().rgba.to_vec();
        s.edit_active(
            "Paint",
            Region::new(10, 10, 40, 40),
            Arc::from(painted.into_boxed_slice()),
        )
        .expect("unlocked");
        let h = s.history();
        assert!(h.can_undo);
        assert_eq!(h.bytes, 256 * 256 * 4, "one tile, not the 4 MB canvas");
        assert!(h.bytes < 1024 * 1024 * 4 / 4);
    }

    #[test]
    fn image_editor_undo_the_history_readout_states_its_bound() {
        let s = stack(8, 8);
        let h = s.history();
        assert!(!h.can_undo && !h.can_redo);
        assert_eq!(h.depth, 0);
        assert_eq!(h.dropped, 0);
        assert_eq!(h.max_bytes, image_graph::DEFAULT_MAX_BYTES);
        assert_eq!(h.max_entries, image_graph::DEFAULT_MAX_ENTRIES);
    }

    #[test]
    fn image_editor_undo_lands_in_the_layer_that_was_painted_not_the_active_one() {
        // The bug this exists to prevent: paint on layer B, select layer
        // A, hit Undo — and get B's pre-edit tiles written into A.
        let mut s = stack(64, 64);
        s.add("B");
        let b_before = s.active().rgba.to_vec();
        s.edit_active("Paint B", Region::new(0, 0, 64, 64), px(64, 64, 200))
            .expect("unlocked");
        let a_before = s.layers()[0].rgba.to_vec();

        s.set_active(0).expect("in range");
        assert_eq!(s.undo().as_deref(), Some("Paint B"));
        assert_eq!(s.layers()[0].rgba.to_vec(), a_before, "A is untouched");
        assert_eq!(s.layers()[1].rgba.to_vec(), b_before, "B is restored");
        // …and the layer the undo landed in becomes active, so the
        // change is visibly where it happened.
        assert_eq!(s.active_index(), 1);
    }

    #[test]
    fn image_editor_undo_removing_a_layer_clears_the_history() {
        // The journal is keyed by layer id; a removed layer's entries
        // could never be applied, so the whole (linear) history goes —
        // stated, not discovered when Undo silently does nothing.
        let mut s = stack(32, 32);
        s.add("B");
        s.edit_active("Paint", Region::new(0, 0, 32, 32), px(32, 32, 9))
            .expect("unlocked");
        assert!(s.history().can_undo);
        s.remove(1).expect("not the last layer");
        assert!(!s.history().can_undo);
        assert_eq!(s.history().bytes, 0);
    }

    #[test]
    fn image_editor_undo_a_damage_region_outside_the_canvas_records_nothing() {
        let mut s = stack(16, 16);
        let px16 = s.active().rgba.clone();
        let out = s
            .edit_active("Paint", Region::new(500, 500, 10, 10), px16)
            .expect("unlocked");
        assert_eq!(out, RecordOutcome::NoChange);
        assert!(!s.history().can_undo);
    }

    // ── clipping ─────────────────────────────────────────────────────

    #[test]
    fn image_editor_layers_clipping_defaults_off_and_toggles() {
        let mut s = stack(8, 8);
        let at = s.add_adjustment("Brighten", bright(0.3));
        assert!(!s.layers()[at].clipped, "clipping is opt-in");
        s.set_clipped(at, true).expect("clip");
        assert!(s.layers()[at].clipped);
        s.set_clipped(at, false).expect("release");
        assert!(!s.layers()[at].clipped);
        assert!(s.set_clipped(99, true).is_err(), "out of range is an error");
    }

    #[test]
    fn image_editor_layers_two_coverages_multiply_rather_than_override() {
        // A layer that is BOTH masked and clipped must be confined by
        // both. Letting one win would silently widen or narrow the
        // effect, which is the failure a designer cannot see coming.
        let half = vec![255u8, 128, 0, 255];
        let base = vec![255u8, 255, 255, 0];
        let cov = Arc::new(SelectionCoverage::from_data(4, 1, half).expect("cov"));
        let out = effective_coverage(Some(&cov), Some(&base), 4, 1).expect("combined");
        assert_eq!(out.coverage_at(0, 0), 255, "full × full stays full");
        assert_eq!(out.coverage_at(1, 0), 128, "half × full is half");
        assert_eq!(out.coverage_at(2, 0), 0);
        assert_eq!(out.coverage_at(3, 0), 0, "full × none is none");
    }

    #[test]
    fn image_editor_layers_neither_mask_nor_clip_keeps_the_fast_path() {
        // `None` is the constant-one fast path, so an ordinary layer must
        // pay nothing for a feature it does not use.
        assert!(effective_coverage(None, None, 4, 4).is_none());
    }

    /// Bottom: fully opaque grey. Middle: opaque on the LEFT half only —
    /// the clip base. The backdrop is therefore opaque EVERYWHERE, which
    /// is what makes the confinement visible: without an opaque backdrop
    /// beneath, both the clipped and the unclipped adjustment read back
    /// as zero on the transparent side and the test proves nothing. (It
    /// did exactly that on the first attempt.)
    fn clip_fixture(w: u32, h: u32) -> (LayerStack, usize) {
        let mut s = stack(w, h);
        let base = s.add("Base");
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for _y in 0..h {
            for x in 0..w {
                let a = if x < w / 2 { 255u8 } else { 0u8 };
                rgba.extend_from_slice(&[200, 200, 200, a]);
            }
        }
        s.layer_mut(base).expect("base").rgba = Arc::from(rgba.into_boxed_slice());
        (s, base)
    }

    #[test]
    fn image_editor_layers_a_clipped_adjustment_is_confined_to_its_base() {
        let Some(ctx) = device() else { return };

        let (mut unclipped, _) = clip_fixture(16, 8);
        unclipped.add_adjustment("Brighten", bright(0.5));
        let free = pollster::block_on(unclipped.composite(Some(ctx), None)).expect("free");

        let (mut s, _) = clip_fixture(16, 8);
        let at = s.add_adjustment("Brighten", bright(0.5));
        s.set_clipped(at, true).expect("clip");
        let clipped = pollster::block_on(s.composite(Some(ctx), None)).expect("clipped");

        let px = |v: &[u8], x: usize| v[x * 4] as i32;
        // LEFT — inside the base: both brighten, so the clip changes
        // nothing there.
        assert!(
            (px(&clipped, 2) - px(&free, 2)).abs() <= 2,
            "inside the base: {} vs {}",
            px(&clipped, 2),
            px(&free, 2)
        );
        // RIGHT — outside the base, over an OPAQUE backdrop: the
        // unclipped adjustment brightens it and the clipped one must
        // not.
        assert!(
            px(&free, 12) > px(&clipped, 12) + 5,
            "outside the base the clip must confine the adjustment: \
             clipped {} should stay below free {}",
            px(&clipped, 12),
            px(&free, 12)
        );
    }

    #[test]
    fn image_editor_layers_a_clipped_layer_with_no_base_contributes_nothing() {
        // Compositing it unclipped instead would be the one behaviour a
        // designer cannot recover from — confining it was the point.
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 8);
        // Clip the BOTTOM layer, which has nothing beneath it.
        s.set_clipped(0, true).expect("clip");
        let at = s.add_adjustment("Brighten", bright(0.5));
        s.set_clipped(at, true).expect("clip the adjustment too");
        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("composite");
        assert!(
            out.chunks_exact(4).all(|p| p[3] == 0),
            "everything was clipped to nothing, so nothing shows"
        );
    }

    #[test]
    fn image_editor_layers_the_clip_flag_reaches_the_json_readout() {
        let mut s = stack(8, 8);
        let at = s.add_adjustment("Brighten", bright(0.3));
        s.set_clipped(at, true).expect("clip");
        // The panel keys its toggle off this; a missing field would make
        // the row render as released while the engine clips.
        assert!(s.layers()[at].clipped);
    }
}
