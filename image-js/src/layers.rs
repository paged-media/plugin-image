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
#[cfg(any(test, feature = "reference-fold"))]
use image_gpu::selection::SelectionMask;
use image_gpu::stroke::window_is_opaque;
use image_gpu::{GpuContext, TileInput};
use image_graph::journal::{FlatImage, RecordOutcome, TileJournal};

// The stack as bytes (persistence); a child module so it reads the fields.
#[path = "layers_persist.rs"]
mod persist;
use image_kernels::families::cast::{
    CastPremultiplyParams, CastUnpremultiplyParams, CAST_PREMULTIPLY, CAST_UNPREMULTIPLY,
};
#[cfg(any(test, feature = "reference-fold"))]
use image_kernels::families::compose::ComposeParams;
use image_kernels::families::compose::COMPOSE_NORMAL;
use image_kernels::KernelDef;

#[cfg(any(test, feature = "reference-fold"))]
use crate::fill::{f16_to_rgba8, rgba8_to_f16};
use crate::ingest::{AdjustParams, IngestError};
use crate::stroke::blend_kernel;

/// How deeply groups may nest. Each open level parks a full-canvas
/// accumulator while it composites, so this is a memory bound rather
/// than a modelling one — and an explicit bound beats discovering it as
/// an allocation failure mid-fold.
const MAX_GROUP_DEPTH: usize = 8;

// The GPU-resident fold (the composite's implementation).
mod fold;
pub mod mask;
pub use mask::{BoundedMask, LayerMask};

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
/// A whole-canvas geometric change: every layer, every mask and every
/// smart object's source moves together, so the stack keeps its layers.
/// All of these are exact pixel moves (no resampling), the same class of
/// CPU windowing as a crop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanvasOp {
    RotateCw,
    RotateCcw,
    Rotate180,
    FlipHorizontal,
    FlipVertical,
    /// Image ▸ Canvas Size: a new extent, the old canvas placed by
    /// `anchor` (0 = left/top, 1 = centre, 2 = right/bottom on each
    /// axis). Added area is transparent; area cut off is gone.
    Resize {
        width: u32,
        height: u32,
        anchor_x: u8,
        anchor_y: u8,
    },
}

impl CanvasOp {
    /// The canvas extent after the op.
    pub fn output_size(&self, w: u32, h: u32) -> (u32, u32) {
        match *self {
            CanvasOp::RotateCw | CanvasOp::RotateCcw => (h, w),
            CanvasOp::Rotate180 | CanvasOp::FlipHorizontal | CanvasOp::FlipVertical => (w, h),
            CanvasOp::Resize { width, height, .. } => (width, height),
        }
    }

    /// For an output pixel, the input pixel it comes from (None = added
    /// area). `(w, h)` is the INPUT extent.
    fn source(&self, x: u32, y: u32, w: u32, h: u32) -> Option<(u32, u32)> {
        match *self {
            CanvasOp::RotateCw => Some((y, h - 1 - x)),
            CanvasOp::RotateCcw => Some((w - 1 - y, x)),
            CanvasOp::Rotate180 => Some((w - 1 - x, h - 1 - y)),
            CanvasOp::FlipHorizontal => Some((w - 1 - x, y)),
            CanvasOp::FlipVertical => Some((x, h - 1 - y)),
            CanvasOp::Resize {
                width,
                height,
                anchor_x,
                anchor_y,
            } => {
                let off = |new: u32, old: u32, a: u8| -> i64 {
                    (i64::from(new) - i64::from(old)) * i64::from(a.min(2)) / 2
                };
                let sx = i64::from(x) - off(width, w, anchor_x);
                let sy = i64::from(y) - off(height, h, anchor_y);
                (sx >= 0 && sy >= 0 && sx < i64::from(w) && sy < i64::from(h))
                    .then_some((sx as u32, sy as u32))
            }
        }
    }

    /// Remap a `bpp`-bytes-per-pixel buffer of extent `(w, h)`; added
    /// area takes `fill`.
    fn remap(&self, src: &[u8], w: u32, h: u32, bpp: usize, fill: u8) -> Vec<u8> {
        let (ow, oh) = self.output_size(w, h);
        let mut out = vec![fill; ow as usize * oh as usize * bpp];
        for y in 0..oh {
            for x in 0..ow {
                if let Some((sx, sy)) = self.source(x, y, w, h) {
                    let d = (y as usize * ow as usize + x as usize) * bpp;
                    let s = (sy as usize * w as usize + sx as usize) * bpp;
                    out[d..d + bpp].copy_from_slice(&src[s..s + bpp]);
                }
            }
        }
        out
    }
}

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
    /// Canvas-extent, tightly packed straight RGBA — at the depth the
    /// buffer itself declares. `Pixels` carries that depth so a call
    /// site cannot assume four bytes per pixel; the `Arc` inside means
    /// a layer still shares the ingest's allocation and a snapshot is
    /// still a pointer copy.
    pub rgba: crate::pixels::LayerPixels,
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
    pub mask: Option<LayerMask>,
    /// The GROUP this layer belongs to, if any. Membership is stored on
    /// the layer rather than as a member list on the group, because the
    /// fold walks layers and this is the lookup it actually needs.
    pub group: Option<u32>,
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
    /// Blend this layer (in Normal mode) in a GAMMA space — how Photoshop
    /// composites TEXT layers ("Blend Text Colors Using Gamma"); `None`
    /// is the ordinary blend. Set by the layered PSD import on text
    /// layers (`compose.normal_gamma`).
    pub blend_gamma: Option<f32>,
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
    pub fn live_mask(&self) -> Option<&LayerMask> {
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

/// The journal-scope bit that marks an entry as a MASK edit. Layer ids
/// are `u32`, so the bit above them can never collide with a pixel
/// entry's scope.
pub const MASK_SCOPE_BIT: u64 = 1 << 32;

/// The journal scope of layer `id`'s mask.
pub fn mask_scope(id: u32) -> u64 {
    u64::from(id) | MASK_SCOPE_BIT
}

/// A mask as an opaque grey RGBA8 plate (`v, v, v, 255`).
pub fn mask_to_grey(mask: &SelectionCoverage) -> Arc<[u8]> {
    let mut out = Vec::with_capacity(mask.data().len() * 4);
    for &v in mask.data() {
        out.extend_from_slice(&[v, v, v, 255]);
    }
    Arc::from(out.into_boxed_slice())
}

/// Read a painted grey plate back as a mask: the RED channel. The plate
/// stays grey through a stroke — every paint colour on it is a grey —
/// so red, green and blue agree and any one of them is the value.
pub fn mask_from_grey(width: u32, height: u32, rgba: &[u8]) -> Option<SelectionCoverage> {
    SelectionCoverage::from_data(width, height, rgba.chunks_exact(4).map(|p| p[0]).collect())
}

/// A layer's own coverage AND its clip base, multiplied.
///
/// Two coverages MULTIPLY — they do not override one another — so a
/// layer that is both masked and clipped is confined by both. Returning
/// `None` when neither exists keeps the constant-one fast path, so an
/// ordinary layer still pays nothing for a feature it does not use.
/// Two coverages multiplied (`None` is full coverage). One side alone is
/// returned as is, so its `Arc` — and the fold's cache of it — survives.
pub(crate) fn multiply_coverage(a: Option<LayerMask>, b: Option<LayerMask>) -> Option<LayerMask> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(a), Some(b)) => {
            let (ca, cb) = (a.to_canvas(), b.to_canvas());
            let data = ca
                .data()
                .iter()
                .zip(cb.data())
                .map(|(&x, &y)| ((u32::from(x) * u32::from(y) + 127) / 255) as u8)
                .collect();
            SelectionCoverage::from_data(ca.width(), ca.height(), data)
                .map(|c| LayerMask::Canvas(Arc::new(c)))
        }
    }
}

fn effective_coverage(
    own: Option<&LayerMask>,
    clip: Option<&[u8]>,
    w: u32,
    h: u32,
) -> Option<Arc<SelectionCoverage>> {
    match (own, clip) {
        (None, None) => None,
        (Some(cov), None) => Some(cov.to_canvas()),
        (None, Some(base)) => SelectionCoverage::from_data(w, h, base.to_vec()).map(Arc::new),
        (Some(cov), Some(base)) => {
            let data: Vec<u8> = (0..(w as usize) * (h as usize))
                .map(|i| {
                    let x = (i % w as usize) as u32;
                    let y = (i / w as usize) as u32;
                    let a = u32::from(cov.at(x, y));
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
/// A LAYER GROUP: a named, contiguous run of layers.
///
/// An ISOLATED group composites its members into their own accumulator
/// before blending that result into the stack, which is what makes a
/// group more than a folder — a group at 50% fades the COMPOSITE of its
/// members, a different picture from fading each member individually.
///
/// A PASS-THROUGH group does not isolate: its members composite directly
/// into whatever is beneath the group, so an adjustment layer inside one
/// reaches the whole stack below. This is Photoshop's DEFAULT, and it is
/// the mode most designers actually mean by "folder".
///
/// CONTIGUOUS, and enforced. The stack is a Vec and the fold walks it
/// once; a group whose members were scattered would composite as several
/// runs, and none of its properties would then mean what the panel
/// showed. NESTING works by the same rule at one level up: a nested
/// One contributing layer's pixels as the fold takes them: canvas-sized
/// (`rect` is `None`), or a bounded layer's rectangle and the pixels
/// inside it.
#[derive(Clone)]
pub(crate) struct Plate {
    pub px: Arc<[u8]>,
    pub rect: Option<Region>,
}

/// group's run must lie inside its parent's.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerGroup {
    pub id: u32,
    pub name: String,
    pub visible: bool,
    /// 0–1. On an ISOLATED group this fades the composite. On a
    /// pass-through group anything below 1 FORCES isolation — that is
    /// Photoshop's own rule and it is not a shortcut: "fade the group"
    /// has no meaning until there is a group-shaped thing to fade.
    pub opacity: f32,
    /// How the group's composite blends into what is beneath it.
    /// Meaningful only when isolated.
    pub blend: &'static KernelDef,
    /// PASS THROUGH (the Photoshop default): members composite straight
    /// into the stack below instead of into an isolated buffer.
    pub pass_through: bool,
    /// The enclosing group, for nesting. `None` is a top-level group.
    pub parent: Option<u32>,
    /// The GROUP's mask (a PSD folder's user mask): canvas-extent
    /// coverage the group's composite blends through.
    pub mask: Option<Arc<SelectionCoverage>>,
    /// A disabled mask is kept, not applied.
    pub mask_enabled: bool,
}

impl LayerGroup {
    /// Does this group composite in isolation?
    ///
    /// Pass-through is the declared mode, but an opacity below 1 forces
    /// isolation regardless — Photoshop's rule, and the honest one:
    /// fading a group that has no composite of its own is not defined.
    ///
    /// A MASK does not: Photoshop keeps a masked pass-through group
    /// pass-through — its members still blend straight into what is below,
    /// each through the group's mask (measured: isolating it put a
    /// multiply member 25 levels off). See [`LayerStack::pass_through_mask`].
    pub fn isolates(&self) -> bool {
        !self.pass_through || self.opacity < 1.0
    }

    /// The mask the group's composite blends through, when there is one
    /// and it is switched on.
    pub fn live_mask(&self) -> Option<&Arc<SelectionCoverage>> {
        self.mask.as_ref().filter(|_| self.mask_enabled)
    }

    pub fn blend_name(&self) -> &'static str {
        self.blend
            .id
            .strip_prefix("compose.")
            .unwrap_or(self.blend.id)
    }
}

pub struct LayerStack {
    width: u32,
    height: u32,
    /// Bottom-first (index 0 is the bottom-most layer).
    layers: Vec<Layer>,
    /// Groups, in no particular order — membership is on the layers.
    groups: Vec<LayerGroup>,
    active: usize,
    next_id: u32,
    journal: TileJournal,
    /// The UNDO list, oldest first: pixel steps (the journal holds their
    /// tiles, in the same order) and structure steps (a snapshot).
    steps: Vec<Step>,
    /// The REDO list, next-to-replay last.
    undone: Vec<Step>,
    /// Steps dropped from the front by the bounds, all kinds.
    dropped_steps: u64,
    /// Bumped by every recorded or replayed structure step.
    structure_generation: u64,
    /// The EDIT TARGET of the active layer: `true` when pixel tools write
    /// its MASK rather than its pixels. Session state, not document state
    /// (it is not in a [`Snapshot`] and is not persisted); it only means
    /// something while the active layer HAS a mask, so it is read
    /// through [`Self::edit_target_is_mask`] and reset whenever the
    /// active layer changes.
    edit_mask: bool,
    /// What the GPU-resident fold keeps between composites (plates, the
    /// checkpoint below the active layer, the last result). Keyed by
    /// identity, so it needs no invalidation — see `fold`.
    fold: std::sync::Mutex<fold::FoldCache>,
}

/// Most structure steps kept (the journal bounds the pixel ones).
pub const MAX_STRUCTURE_STEPS: usize = 200;

/// Everything but pixels-in-the-journal: what a structure step restores.
/// Cheap — layer pixels are shared `Arc`s, so a snapshot copies pointers.
#[derive(Debug, Clone)]
struct Snapshot {
    width: u32,
    height: u32,
    layers: Vec<Layer>,
    groups: Vec<LayerGroup>,
    active: usize,
    next_id: u32,
}

/// One undo step.
///
/// Why one list works: undo is LIFO, so a pixel step is only ever replayed
/// against the stack it was recorded on — every structure change made
/// after it (a removed layer, a rotated canvas) has been undone first.
/// That is what lets removing a layer or rotating the canvas stay in the
/// history instead of wiping it.
#[derive(Debug, Clone)]
enum Step {
    /// One journal entry (its label lives in the journal).
    Pixels,
    /// A structure change; `other` is the state on the far side of it
    /// (before, on the undo list; after, on the redo list).
    Structure {
        label: String,
        /// Consecutive steps with the same key merge into one (a slider
        /// drag is one undo step, not twenty).
        merge_key: Option<String>,
        other: Box<Snapshot>,
    },
}

impl LayerStack {
    /// Open a stack over `rgba` — one full-canvas [`BACKGROUND_LAYER_NAME`]
    /// layer. The pixels are SHARED (an `Arc` clone), so this is O(1) and
    /// costs no extra memory over the image it was opened on.
    /// [`from_image`](Self::from_image) for a buffer that already knows
    /// its depth — the door a 16-bit ingest comes through.
    pub fn from_image_px(
        width: u32,
        height: u32,
        rgba: crate::pixels::Pixels,
    ) -> Result<LayerStack, IngestError> {
        let want = (width as usize) * (height as usize) * rgba.bytes_per_pixel();
        if width == 0 || height == 0 || rgba.len() != want {
            return Err(IngestError::Decode(format!(
                "layer stack: {} bytes for {width}x{height} (expected {want})",
                rgba.len()
            )));
        }
        // Build through the 8-bit door with a correctly-sized dummy so
        // all the stack's other invariants are established exactly
        // once, then swap the real buffer in.
        let dummy = vec![0u8; (width as usize) * (height as usize) * 4];
        let mut st = Self::from_image(width, height, Arc::from(dummy))?;
        st.layers[0].rgba = rgba.into();
        Ok(st)
    }

    pub fn from_image(width: u32, height: u32, rgba: Arc<[u8]>) -> Result<LayerStack, IngestError> {
        let want = (width as usize) * (height as usize) * 4;
        if width == 0 || height == 0 || rgba.len() != want {
            return Err(IngestError::Decode(format!(
                "layer stack: {} bytes for {width}×{height} (expected {want})",
                rgba.len()
            )));
        }
        Ok(LayerStack {
            groups: Vec::new(),
            width,
            height,
            layers: vec![Layer {
                id: 1,
                name: BACKGROUND_LAYER_NAME.to_string(),
                visible: true,
                locked: false,
                opacity: 1.0,
                blend: &COMPOSE_NORMAL,
                rgba: crate::pixels::Pixels::from_rgba8(rgba).into(),
                kind: LayerKind::Pixels,
                // A new layer is unmasked; the mask is authored later.
                mask: None,
                mask_enabled: true,
                group: None,
                clipped: false,
                blend_gamma: None,
            }],
            active: 0,
            next_id: 2,
            journal: TileJournal::new(),
            steps: Vec::new(),
            undone: Vec::new(),
            dropped_steps: 0,
            structure_generation: 0,
            edit_mask: false,
            fold: Default::default(),
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
        // A Color Overlay becomes a layer of its own (below), so the stack
        // holds one layer per plate plus one per overlay.
        // …and an ARTBOARD's background is a layer at the bottom of its
        // group.
        let n_layers = import.layers.len()
            + import
                .layers
                .iter()
                .filter(|p| p.color_overlay.is_some())
                .count()
            + import
                .groups
                .iter()
                .filter(|g| g.artboard.is_some_and(|a| a.background.is_some()))
                .count();
        let mut backgrounds_drawn = vec![false; import.groups.len()];
        // Ids: layers 1..=N, then the groups (one id space, as `fresh_id`).
        let group_id = |g: usize| (n_layers + g) as u32 + 1;
        let groups: Vec<LayerGroup> = import
            .groups
            .iter()
            .enumerate()
            .map(|(g, plate)| LayerGroup {
                id: group_id(g),
                name: if plate.name.is_empty() {
                    format!("Group {}", g + 1)
                } else {
                    plate.name.clone()
                },
                visible: !plate.hidden,
                opacity: plate.opacity as f32 / 255.0,
                pass_through: &plate.blend_key == b"pass",
                blend: psd_blend_kernel(&plate.blend_key),
                parent: plate.parent.map(group_id),
                mask: plate.mask.as_ref().map(|m| {
                    Arc::new(
                        SelectionCoverage::from_data(width, height, m.coverage.clone())
                            .expect("a canvas-extent mask"),
                    )
                }),
                mask_enabled: plate.mask.as_ref().is_none_or(|m| m.enabled),
            })
            .collect();
        let mut layers: Vec<Layer> = Vec::with_capacity(n_layers);
        // A base's Color Overlay waits until its clipped layers are in.
        let mut pending_overlay: Option<Layer> = None;
        for (i, plate) in import.layers.iter().enumerate() {
            if !plate.clipped {
                if let Some(mut o) = pending_overlay.take() {
                    o.id = layers.len() as u32 + 1;
                    layers.push(o);
                }
            }
            // ARTBOARD backgrounds of the groups this plate enters, the
            // outermost first: each a solid layer at the artboard's
            // rectangle, at the bottom of its group.
            let mut chain = Vec::new();
            let mut cur = plate.group;
            while let Some(g) = cur {
                chain.push(g);
                cur = import.groups[g].parent;
            }
            for &g in chain.iter().rev() {
                let Some(a) = import.groups[g].artboard else {
                    continue;
                };
                let Some(bg) = a.background else { continue };
                if backgrounds_drawn[g] || a.rect.w == 0 || a.rect.h == 0 {
                    continue;
                }
                backgrounds_drawn[g] = true;
                let solid: Vec<u8> = (0..a.rect.w as usize * a.rect.h as usize)
                    .flat_map(|_| [bg[0], bg[1], bg[2], 255])
                    .collect();
                layers.push(Layer {
                    id: layers.len() as u32 + 1,
                    name: format!("{} · Artboard", import.groups[g].name),
                    visible: true,
                    locked: false,
                    opacity: 1.0,
                    blend: &COMPOSE_NORMAL,
                    rgba: bounded_pixels(a.rect, solid, width, height)?,
                    kind: LayerKind::Pixels,
                    mask: None,
                    mask_enabled: true,
                    group: Some(group_id(g)),
                    clipped: false,
                    blend_gamma: None,
                });
            }
            let adjustment = plate.adjustment.as_ref().map(psd_adjust_params);
            // The user and vector masks fold into the layer's one mask
            // (`psd_vector_mask::plate_mask`).
            let mask = crate::psd_vector_mask::plate_mask(plate, width, height);
            let plate_rect = plate.rect.unwrap_or(Region::new(0, 0, width, height));
            let plate_want = plate_rect.w as usize * plate_rect.h as usize * 4;
            if adjustment.is_none() && plate.rgba.len() != plate_want {
                return Err(IngestError::Decode(format!(
                    "PSD layer \"{}\" is {} bytes for its {}×{} rectangle (expected {plate_want})",
                    plate.name,
                    plate.rgba.len(),
                    plate_rect.w,
                    plate_rect.h
                )));
            }
            let name = if plate.name.is_empty() {
                format!("Layer {}", i + 1)
            } else {
                plate.name.clone()
            };
            layers.push(Layer {
                id: layers.len() as u32 + 1,
                name: name.clone(),
                visible: !plate.hidden,
                locked: false,
                opacity: plate.opacity as f32 / 255.0,
                blend: psd_blend_kernel(&plate.blend_key),
                // At its BOUNDS (ADR 464): an adjustment keeps an empty
                // rectangle, a pixel layer its record's.
                rgba: bounded_pixels(
                    if adjustment.is_some() {
                        Region::new(0, 0, 0, 0)
                    } else {
                        plate_rect
                    },
                    if adjustment.is_some() {
                        Vec::new()
                    } else {
                        plate.rgba.clone()
                    },
                    width,
                    height,
                )?,
                // A smart object's stored render arrives as pixels: the
                // source is not rendered here (`smart_renders_agree`).
                kind: match adjustment {
                    Some(params) => LayerKind::Adjustment(Box::new(params)),
                    None => LayerKind::Pixels,
                },
                mask_enabled: mask.as_ref().is_none_or(|m| m.enabled),
                // At the plate's rectangle when the import cut it there
                // (ADR 464); outside it the layer is transparent, so the
                // value there is immaterial.
                mask: match mask {
                    Some(m) => Some(psd_layer_mask(m, width, height)?),
                    None => None,
                },
                group: plate.group.map(group_id),
                clipped: plate.clipped,
                // Text blends in a gamma space — in an RGB document. A
                // CMYK document blends in its inks; its measured
                // composites got worse with the gamma (3 corpus files).
                blend_gamma: (plate.text
                    && &plate.blend_key == b"norm"
                    && !import.converted_from_cmyk)
                    .then_some(TEXT_BLEND_GAMMA),
            });
            // COLOR OVERLAY: Photoshop draws it over the layer's content —
            // and over the layers CLIPPED to it, which it covers (measured:
            // a normal 100 % overlay hides a clipped multiply layer
            // completely) — inside the layer's shape, before the clipping
            // group blends. That is a solid layer clipped to the base,
            // placed after the base's own clipped layers.
            if let Some(o) = &plate.color_overlay {
                // Drawn only inside its base, so it lives at the base's
                // rectangle.
                let solid: Vec<u8> = (0..plate_want / 4)
                    .flat_map(|_| [o.rgb[0], o.rgb[1], o.rgb[2], 255])
                    .collect();
                pending_overlay = Some(Layer {
                    id: 0,
                    name: format!("{name} · Color Overlay"),
                    visible: !plate.hidden,
                    locked: false,
                    opacity: f32::from(o.opacity) / 255.0,
                    blend: psd_blend_kernel(&o.blend_key),
                    rgba: bounded_pixels(plate_rect, solid, width, height)?,
                    kind: LayerKind::Pixels,
                    mask: None,
                    mask_enabled: true,
                    group: plate.group.map(group_id),
                    clipped: true,
                    blend_gamma: None,
                });
            }
        }
        if let Some(mut o) = pending_overlay.take() {
            o.id = layers.len() as u32 + 1;
            layers.push(o);
        }
        let next_id = (layers.len() + groups.len()) as u32 + 1;
        let stack = LayerStack {
            groups,
            width,
            height,
            active: layers.len() - 1,
            layers,
            next_id,
            journal: TileJournal::new(),
            steps: Vec::new(),
            undone: Vec::new(),
            dropped_steps: 0,
            structure_generation: 0,
            edit_mask: false,
            fold: Default::default(),
        };
        if let Some(deep) = stack
            .groups
            .iter()
            .find(|g| stack.depth_of(Some(g.id)) > MAX_GROUP_DEPTH)
        {
            return Err(IngestError::Unsupported(format!(
                "PSD group \"{}\" is nested deeper than {MAX_GROUP_DEPTH}: each level parks a \
                 full-canvas buffer while it composites",
                deep.name
            )));
        }
        Ok(stack)
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
                rgba: crate::pixels::Pixels::from_rgba8(pixels).into(),
                mask: None,
                mask_enabled: true,
                group: None,
                clipped: false,
                blend_gamma: None,
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
    /// Rotate, flip or re-size the whole canvas: every layer's pixels,
    /// every mask and every smart object's source move together, so the
    /// stack keeps its layers (a crop, by contrast, flattens).
    ///
    /// One undo step; the history before it stays (undoing the op first
    /// restores the extent older pixel steps were recorded against).
    /// Canvas Size is refused while a smart object is in the stack (its
    /// render would no longer line up with its source).
    pub fn transform_canvas(&mut self, op: CanvasOp) -> Result<(), IngestError> {
        let label = match op {
            CanvasOp::RotateCw => "Rotate 90° clockwise",
            CanvasOp::RotateCcw => "Rotate 90° counter-clockwise",
            CanvasOp::Rotate180 => "Rotate 180°",
            CanvasOp::FlipHorizontal => "Flip horizontal",
            CanvasOp::FlipVertical => "Flip vertical",
            CanvasOp::Resize { .. } => "Canvas size",
        };
        // Always recorded, for the reason `remove` is: older pixel steps
        // address the old extent and must not be reachable across it.
        self.recorded(label, None, |s| s.transform_canvas_unrecorded(op))
    }

    fn transform_canvas_unrecorded(&mut self, op: CanvasOp) -> Result<(), IngestError> {
        let (w, h) = (self.width, self.height);
        let (ow, oh) = op.output_size(w, h);
        if ow == 0 || oh == 0 {
            return Err(IngestError::Unsupported(
                "canvas size must be at least 1×1".into(),
            ));
        }
        let resize = matches!(op, CanvasOp::Resize { .. });
        if resize
            && self
                .layers
                .iter()
                .any(|l| matches!(l.kind, LayerKind::Smart(_)))
        {
            return Err(IngestError::Unsupported(
                "Canvas Size with a smart object in the stack is not supported \
                 (its render would no longer line up with its source)"
                    .into(),
            ));
        }
        for layer in &mut self.layers {
            let canvas = layer.rgba.canvas();
            let (bpp, depth) = (canvas.bytes_per_pixel(), canvas.depth());
            let px = op.remap(canvas.raw(), w, h, bpp, 0);
            layer.rgba =
                crate::pixels::Pixels::from_raw(Arc::from(px.into_boxed_slice()), depth).into();
            if let Some(mask) = &layer.mask {
                // Added area is revealed, as a reveal-all mask extends.
                let m = op.remap(mask.canvas().data(), w, h, 1, 255);
                layer.mask = SelectionCoverage::from_data(ow, oh, m).map(|c| Arc::new(c).into());
            }
            if let LayerKind::Smart(src) = &mut layer.kind {
                let s = op.remap(&src.rgba, src.width, src.height, 4, 0);
                let (sw, sh) = op.output_size(src.width, src.height);
                src.rgba = Arc::from(s.into_boxed_slice());
                src.width = sw;
                src.height = sh;
            }
        }
        self.width = ow;
        self.height = oh;
        Ok(())
    }

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
            rgba: layer.rgba.raw_arc(),
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
                layer.rgba = crate::pixels::Pixels::from_rgba8(rendered).into();
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
        if layer.locked {
            return Err(IngestError::Unsupported(format!(
                "layer \"{}\" is locked",
                layer.name
            )));
        }
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
                rgba: crate::pixels::Pixels::from_rgba8(pixels).into(),
                kind: LayerKind::Pixels,
                // A new layer is unmasked; the mask is authored later.
                mask: None,
                mask_enabled: true,
                group: None,
                clipped: false,
                blend_gamma: None,
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
    /// Remove a layer, as ONE undo step (always recorded: the journal's
    /// entries for the removed layer stay valid only because undo has to
    /// bring the layer back before it can reach them).
    pub fn remove(&mut self, index: usize) -> Result<(), IngestError> {
        self.recorded("Delete layer", None, |s| s.remove_unrecorded(index))
    }

    fn remove_unrecorded(&mut self, index: usize) -> Result<(), IngestError> {
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
        // The journal's entries for this layer stay: undo is LIFO, so they
        // can only be replayed after the removal itself has been undone
        // (when recorded through `recorded`), which brings the layer back.
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
        if index != self.active {
            // A new active layer starts on its PIXELS, as in Photoshop:
            // the mask target belongs to the layer it was chosen on.
            self.edit_mask = false;
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
        layer.mask = Some(coverage.into());
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

    /// ADD LAYER MASK in its two Photoshop forms: REVEAL ALL (all-one,
    /// the layer looks unchanged until the mask is painted) or HIDE ALL
    /// (all-zero, the layer disappears until it is painted back in).
    /// Refused when the layer already has one — adding would silently
    /// discard painted coverage; delete the old mask first.
    pub fn add_mask(&mut self, index: usize, reveal_all: bool) -> Result<(), IngestError> {
        if self.has_mask(index) {
            return Err(IngestError::Unsupported(format!(
                "layer {index} already has a mask — delete it before adding another"
            )));
        }
        let (w, h) = (self.width, self.height);
        let cov = if reveal_all {
            SelectionCoverage::full(w, h)
        } else {
            SelectionCoverage::empty(w, h)
        };
        self.set_mask(index, Arc::new(cov))
    }

    /// Make `index` the active layer and choose what pixel tools write on
    /// it: its pixels (`mask == false`) or its MASK. Choosing the mask of
    /// a layer that has none is an error rather than a silent fall back
    /// to pixels — a stroke the user meant for a mask landing in the
    /// image would be the worst outcome.
    pub fn set_edit_target(&mut self, index: usize, mask: bool) -> Result<(), IngestError> {
        if mask && !self.has_mask(index) {
            return Err(IngestError::Unsupported(format!(
                "layer {index} has no mask to edit — add a layer mask first"
            )));
        }
        self.set_active(index)?;
        self.edit_mask = mask;
        Ok(())
    }

    /// Do pixel tools currently write the ACTIVE layer's mask? True only
    /// while the mask target was chosen AND the layer still has a mask
    /// (an undo or a delete can take the mask away under the target).
    pub fn edit_target_is_mask(&self) -> bool {
        self.edit_mask && self.layers[self.active].mask.is_some()
    }

    /// The active layer's mask as an OPAQUE GREY RGBA8 plate (`v, v, v,
    /// 255` per texel) — the buffer a stroke paints when the edit target
    /// is the mask. Painting a mask is then painting a grey image through
    /// the very same stroke compositor; [`mask_from_grey`] reads it back.
    pub fn active_mask_as_grey(&self) -> Option<Arc<[u8]>> {
        let m = self.layers[self.active].mask.as_ref()?;
        Some(mask_to_grey(&m.to_canvas()))
    }

    /// Replace the ACTIVE layer's MASK, journaling the tiles `damage`
    /// covers first — the mask twin of [`Self::edit_active`].
    ///
    /// The journal entry's scope is `layer id | MASK_SCOPE_BIT`, so the
    /// one undo list replays it into the MASK of the right layer, even
    /// when another layer (or that layer's pixels) is the target by then.
    /// The journal is depth-agnostic, so a one-byte-per-texel mask needs
    /// nothing from `image-graph` beyond `FlatImage`'s texel size.
    pub fn edit_active_mask(
        &mut self,
        label: &str,
        damage: Region,
        mask: SelectionCoverage,
    ) -> Result<RecordOutcome, IngestError> {
        let (w, h) = (self.width, self.height);
        if mask.width() != w || mask.height() != h {
            return Err(IngestError::Decode(format!(
                "mask edit is {}×{} for a {w}×{h} canvas",
                mask.width(),
                mask.height()
            )));
        }
        let active = &self.layers[self.active];
        if active.locked {
            return Err(IngestError::Unsupported(format!(
                "layer \"{}\" is locked",
                active.name
            )));
        }
        let Some(old) = active.mask.as_ref() else {
            return Err(IngestError::Unsupported(format!(
                "layer \"{}\" has no mask to edit",
                active.name
            )));
        };
        let clipped = damage
            .intersect(Region::new(0, 0, w, h))
            .unwrap_or(Region::new(0, 0, 0, 0));
        let outcome = {
            // A bounded mask is edited at canvas size (ADR 464).
            let old = old.canvas();
            let view = FlatImage::new(w, h, 1, old.data())
                .ok_or_else(|| IngestError::Decode("layer mask is mis-sized".into()))?;
            self.journal
                .record(label, mask_scope(active.id), &view, clipped)
        };
        self.layers[self.active].mask = Some(Arc::new(mask).into());
        if matches!(outcome, RecordOutcome::Recorded { .. }) {
            self.steps.push(Step::Pixels);
            self.undone.clear();
        }
        self.sync_with_journal();
        Ok(outcome)
    }

    /// Fold the stack with the ACTIVE layer's mask replaced by `mask` for
    /// this composite only — how a mask stroke in flight previews. The
    /// stack is restored before returning, whatever the outcome; nothing
    /// about the fold itself changes (the mask is an ordinary mask).
    pub async fn composite_with_active_mask(
        &mut self,
        ctx: Option<&GpuContext>,
        mask: Arc<SelectionCoverage>,
    ) -> Result<Arc<[u8]>, IngestError> {
        let i = self.active;
        let saved = self.layers[i].mask.replace(mask.into());
        let out = self.composite(ctx, None).await;
        self.layers[i].mask = saved;
        out
    }

    // ── groups ───────────────────────────────────────────────────────

    pub fn groups(&self) -> &[LayerGroup] {
        &self.groups
    }

    pub fn group_of(&self, index: usize) -> Option<&LayerGroup> {
        let gid = self.layers.get(index)?.group?;
        self.groups.iter().find(|g| g.id == gid)
    }

    /// Group the CONTIGUOUS run `from..=to` under `name`.
    ///
    /// Contiguity is a requirement rather than something to work around:
    /// the fold walks the stack once, so a scattered group would
    /// composite as several runs and its opacity, blend and visibility
    /// would each mean something different from what the panel showed.
    ///
    /// NESTING is supported, and it is the same rule one level up: every
    /// layer in the range must share the SAME enclosing group (possibly
    /// none), and the new group becomes its child. A range straddling
    /// two different parents is refused, because the result would not be
    /// a tree — it would be a group that is inside two things at once.
    pub fn group_range(&mut self, from: usize, to: usize, name: &str) -> Result<u32, IngestError> {
        let (lo, hi) = (from.min(to), from.max(to));
        if hi >= self.layers.len() {
            return Err(IngestError::Unsupported(format!(
                "cannot group {lo}..={hi}: the stack has {} layers",
                self.layers.len()
            )));
        }
        let parent = self.layers[lo].group;
        if self.layers[lo..=hi].iter().any(|l| l.group != parent) {
            return Err(IngestError::Unsupported(
                "cannot group a range that straddles two different groups: the \
                 result would be inside both, which is not a tree"
                    .into(),
            ));
        }
        // Depth guard. Nesting is genuinely supported, but a runaway
        // chain would make the fold park an unbounded number of
        // full-canvas accumulators — a memory cliff, not a feature.
        if self.depth_of(parent) >= MAX_GROUP_DEPTH {
            return Err(IngestError::Unsupported(format!(
                "group nesting deeper than {MAX_GROUP_DEPTH}: each level parks a \
                 full-canvas buffer while it composites"
            )));
        }
        let id = self.fresh_id();
        self.groups.push(LayerGroup {
            id,
            name: name.to_string(),
            visible: true,
            opacity: 1.0,
            blend: &COMPOSE_NORMAL,
            // Photoshop's default, and the one most people mean by
            // "folder": members reach the stack below.
            pass_through: true,
            parent,
            mask: None,
            mask_enabled: true,
        });
        for l in &mut self.layers[lo..=hi] {
            l.group = Some(id);
        }
        Ok(id)
    }

    /// How many groups enclose `gid` (itself included).
    fn depth_of(&self, gid: Option<u32>) -> usize {
        let mut n = 0;
        let mut cur = gid;
        while let Some(id) = cur {
            n += 1;
            if n > MAX_GROUP_DEPTH * 2 {
                break; // a cycle cannot be built through the API, but do not hang on one
            }
            cur = self
                .groups
                .iter()
                .find(|g| g.id == id)
                .and_then(|g| g.parent);
        }
        n
    }

    /// [`chain_of`](Self::chain_of) for the PSD writer.
    pub(crate) fn group_chain(&self, gid: Option<u32>) -> Vec<u32> {
        self.chain_of(gid)
    }

    /// The enclosing chain of `gid`, OUTERMOST first — what the fold
    /// compares against to decide which groups to open and close.
    /// Store layer `index` at the bounding box of its visible pixels
    /// (ADR 464): nothing it draws changes — outside the box every pixel
    /// is transparent — but it then occupies the box, not the canvas.
    /// Returns the box, or `None` when the layer is not an 8-bit pixel
    /// layer or already as small as it gets. Not an undo step: it does
    /// not change the image.
    pub fn shrink_to_content(&mut self, index: usize) -> Result<Option<Region>, IngestError> {
        let (w, h) = (self.width, self.height);
        let layer = self.layer_mut(index)?;
        if !layer.is_pixels() || layer.rgba.is_16bit() {
            return Ok(None);
        }
        let px = layer.rgba.canvas().raw_arc();
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
        for y in 0..h {
            for x in 0..w {
                if px[((y * w + x) * 4 + 3) as usize] != 0 {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x + 1);
                    y1 = y1.max(y + 1);
                }
            }
        }
        let rect = if x1 <= x0 {
            Region::new(0, 0, 0, 0)
        } else {
            Region::new(x0 as i32, y0 as i32, x1 - x0, y1 - y0)
        };
        if let Some(b) = layer.rgba.bounded() {
            if b.rect == rect {
                return Ok(None);
            }
        } else if rect == Region::new(0, 0, w, h) {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(rect.w as usize * rect.h as usize * 4);
        for y in rect.y as u32..rect.y as u32 + rect.h {
            let s = ((y * w + rect.x as u32) * 4) as usize;
            out.extend_from_slice(&px[s..s + rect.w as usize * 4]);
        }
        layer.rgba = bounded_pixels(rect, out, w, h)?;
        // Its mask matters only where it draws, so it shrinks with it.
        if let Some(m) = layer.mask.take() {
            let c = m.to_canvas();
            let mut data = Vec::with_capacity(rect.w as usize * rect.h as usize);
            for y in rect.y as u32..rect.y as u32 + rect.h {
                let s = (y * w + rect.x as u32) as usize;
                data.extend_from_slice(&c.data()[s..s + rect.w as usize]);
            }
            layer.mask = Some(match BoundedMask::new(rect, 0, data, w, h) {
                Some(b) if rect != Region::new(0, 0, w, h) => LayerMask::Bounded(Arc::new(b)),
                _ => m,
            });
        }
        Ok(Some(rect))
    }

    /// Set (or clear) a GROUP's mask and its switch. Not an undo step:
    /// group masks arrive with a PSD import, the panel does not edit them
    /// yet.
    pub fn set_group_mask(
        &mut self,
        id: u32,
        mask: Option<Arc<SelectionCoverage>>,
        enabled: bool,
    ) -> Result<(), IngestError> {
        let (w, h) = (self.width, self.height);
        if let Some(m) = &mask {
            if m.width() != w || m.height() != h {
                return Err(IngestError::Unsupported(format!(
                    "group mask is {}×{}, the canvas {w}×{h}",
                    m.width(),
                    m.height()
                )));
            }
        }
        let g = self
            .groups
            .iter_mut()
            .find(|g| g.id == id)
            .ok_or_else(|| IngestError::Unsupported(format!("no group {id}")))?;
        g.mask = mask;
        g.mask_enabled = enabled;
        self.structure_generation += 1;
        Ok(())
    }

    /// The masks of the PASS-THROUGH groups around `gid` (innermost
    /// first, up to the nearest isolated one, whose own mask applies when
    /// it closes), multiplied: what a member of `gid` composites through.
    /// A single mask comes back as its own `Arc`.
    pub(crate) fn pass_through_mask(&self, gid: Option<u32>) -> Option<LayerMask> {
        let mut out = None;
        let mut cur = gid;
        let mut depth = 0;
        while let Some(id) = cur {
            let Some(g) = self.groups.iter().find(|g| g.id == id) else {
                break;
            };
            if g.isolates() || depth > MAX_GROUP_DEPTH * 2 {
                break;
            }
            out = multiply_coverage(out, g.live_mask().cloned().map(LayerMask::Canvas));
            cur = g.parent;
            depth += 1;
        }
        out
    }

    fn chain_of(&self, gid: Option<u32>) -> Vec<u32> {
        let mut out = Vec::new();
        let mut cur = gid;
        while let Some(id) = cur {
            out.push(id);
            if out.len() > MAX_GROUP_DEPTH * 2 {
                break;
            }
            cur = self
                .groups
                .iter()
                .find(|g| g.id == id)
                .and_then(|g| g.parent);
        }
        out.reverse();
        out
    }

    /// Dissolve a group. Its layers STAY, in place and unchanged — the
    /// group was a compositing wrapper, not a container, so removing it
    /// cannot lose pixels.
    ///
    /// Children are RE-PARENTED to the dissolved group's parent rather
    /// than orphaned or deleted, for the same reason: dissolving one
    /// level of wrapping must not take a level of content with it.
    pub fn ungroup(&mut self, id: u32) -> Result<(), IngestError> {
        let Some(g) = self.groups.iter().find(|g| g.id == id) else {
            return Err(IngestError::Unsupported(format!("unknown group {id}")));
        };
        let parent = g.parent;
        for l in &mut self.layers {
            if l.group == Some(id) {
                l.group = parent;
            }
        }
        for child in &mut self.groups {
            if child.parent == Some(id) {
                child.parent = parent;
            }
        }
        self.groups.retain(|g| g.id != id);
        Ok(())
    }

    /// Switch a group between pass-through and isolated.
    pub fn set_group_pass_through(
        &mut self,
        id: u32,
        pass_through: bool,
    ) -> Result<(), IngestError> {
        self.group_mut(id)?.pass_through = pass_through;
        Ok(())
    }

    fn group_mut(&mut self, id: u32) -> Result<&mut LayerGroup, IngestError> {
        self.groups
            .iter_mut()
            .find(|g| g.id == id)
            .ok_or_else(|| IngestError::Unsupported(format!("unknown group {id}")))
    }

    pub fn set_group_visible(&mut self, id: u32, visible: bool) -> Result<(), IngestError> {
        self.group_mut(id)?.visible = visible;
        Ok(())
    }

    pub fn set_group_opacity(&mut self, id: u32, opacity: f32) -> Result<(), IngestError> {
        self.group_mut(id)?.opacity = opacity.clamp(0.0, 1.0);
        Ok(())
    }

    pub fn set_group_name(&mut self, id: u32, name: &str) -> Result<(), IngestError> {
        self.group_mut(id)?.name = name.to_string();
        Ok(())
    }

    pub fn set_group_blend(&mut self, id: u32, blend: &str) -> Result<(), IngestError> {
        let k = crate::stroke::blend_kernel(blend)
            .ok_or_else(|| IngestError::Unsupported(format!("unknown blend mode {blend:?}")))?;
        self.group_mut(id)?.blend = k;
        Ok(())
    }

    /// Does this group contribute at all? Hidden or at zero opacity it
    /// contributes exactly nothing, by the same argument the per-layer
    /// check makes: `over(a, b·0)` reduces to the backdrop for all 26
    /// blend modes, so skipping is exact and not an approximation.
    fn group_enabled(&self, id: u32) -> bool {
        self.groups
            .iter()
            .find(|g| g.id == id)
            .is_some_and(|g| g.visible && g.opacity > 0.0)
    }

    /// Blend a group's finished composite into the outer stack.
    #[cfg(any(test, feature = "reference-fold"))]
    async fn blend_group(
        &self,
        ctx: &GpuContext,
        id: u32,
        stack: Vec<u8>,
        inner: Vec<u8>,
        w: u32,
        h: u32,
    ) -> Result<Vec<u8>, IngestError> {
        let Some(g) = self.groups.iter().find(|g| g.id == id) else {
            return Ok(stack);
        };
        // One dispatch, the same one a layer takes — a group is a plate
        // like any other once its members have been folded, its mask
        // included.
        let mask = multiply_coverage(
            g.live_mask().cloned().map(LayerMask::Canvas),
            self.pass_through_mask(g.parent),
        );
        let mask = mask.map(|cov| {
            SelectionMask::from_fn(w, h, |x, y| f32::from(cov.at(x, y)) / 255.0).into_bytes()
        });
        image_gpu::execute_tile_once_async(
            ctx,
            g.blend,
            &[
                TileInput { f16_bytes: &stack },
                TileInput { f16_bytes: &inner },
            ],
            ComposeParams::new(g.opacity).as_bytes(),
            mask.as_deref(),
            w,
            h,
        )
        .await
        .map_err(|e| IngestError::Pipeline(e.to_string()))
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
        pixels: crate::pixels::Pixels,
    ) -> Result<RecordOutcome, IngestError> {
        // Sized from the INCOMING buffer's own depth. The journal is
        // depth-agnostic — it stores opaque tile byte handles and
        // `FlatImage` already took bytes-per-pixel as a parameter — so
        // a 16-bit edit journals and undoes at 16 bits with no change
        // to `image-graph` at all (docs/design/16-bit-stack.md step 4).
        let want = (self.width as usize) * (self.height as usize) * pixels.bytes_per_pixel();
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
        // A layer and the edit landing on it must agree on depth, or
        // the journal would snapshot tiles in one geometry and restore
        // them in another. Refuse rather than silently narrow.
        if active.rgba.depth() != pixels.depth() {
            return Err(IngestError::Unsupported(format!(
                "layer edit at {:?} onto a {:?} layer — depths must match",
                pixels.depth(),
                active.rgba.depth()
            )));
        }
        let outcome = {
            let bpp = active.rgba.canvas().bytes_per_pixel();
            let raw = active.rgba.canvas().raw();
            let view = FlatImage::new(self.width, self.height, bpp, raw)
                .ok_or_else(|| IngestError::Decode("layer pixels are mis-sized".into()))?;
            // The layer's stable ID is the entry's SCOPE, so undo lands
            // in the layer that was painted and not in whichever one
            // happens to be selected when the user reaches for it.
            self.journal.record(label, active.id as u64, &view, clipped)
        };
        self.layers[self.active].rgba = pixels.into();
        if matches!(outcome, RecordOutcome::Recorded { .. }) {
            self.steps.push(Step::Pixels);
            self.undone.clear();
        }
        self.sync_with_journal();
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

    /// Run a STRUCTURE change (anything but painting pixels) as one undo
    /// step labelled `label`. Consecutive steps with the same `merge_key`
    /// merge — a slider drag is one step. Nothing is recorded when `f`
    /// fails.
    pub fn recorded<T>(
        &mut self,
        label: &str,
        merge_key: Option<&str>,
        f: impl FnOnce(&mut Self) -> Result<T, IngestError>,
    ) -> Result<T, IngestError> {
        let before = self.snapshot();
        let out = f(self)?;
        let merges = matches!(
            (self.steps.last(), merge_key),
            (Some(Step::Structure { merge_key: Some(k), .. }), Some(m)) if k == m
        );
        if !merges {
            self.steps.push(Step::Structure {
                label: label.to_string(),
                merge_key: merge_key.map(str::to_string),
                other: Box::new(before),
            });
        }
        self.undone.clear();
        self.structure_generation += 1;
        self.enforce_step_bound();
        Ok(out)
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            width: self.width,
            height: self.height,
            layers: self.layers.clone(),
            groups: self.groups.clone(),
            active: self.active,
            next_id: self.next_id,
        }
    }

    fn restore(&mut self, s: Snapshot) {
        self.width = s.width;
        self.height = s.height;
        self.layers = s.layers;
        self.groups = s.groups;
        self.active = s.active;
        self.next_id = s.next_id;
    }

    /// The journal bounds its own entries (and clears itself when one
    /// edit is too large). Every pixel step it no longer holds — and every
    /// step older than that, which could only be reached through it — is
    /// dropped here, so the two lists never disagree.
    fn sync_with_journal(&mut self) {
        let held = self.journal.depth();
        let mut pixel_steps = self
            .steps
            .iter()
            .filter(|s| matches!(s, Step::Pixels))
            .count();
        while pixel_steps > held {
            match self.steps.remove(0) {
                // The journal already counted its own eviction.
                Step::Pixels => pixel_steps -= 1,
                Step::Structure { .. } => self.dropped_steps += 1,
            }
        }
    }

    /// At most [`MAX_STRUCTURE_STEPS`] structure steps; the oldest steps
    /// go first (pixel ones included, which the journal then still holds
    /// unreachably until its own bound evicts them).
    fn enforce_step_bound(&mut self) {
        while self
            .steps
            .iter()
            .filter(|s| matches!(s, Step::Structure { .. }))
            .count()
            > MAX_STRUCTURE_STEPS
        {
            self.steps.remove(0);
            self.dropped_steps += 1;
        }
    }

    /// Undo the newest step, pixels or structure. Returns its label, or
    /// `None` when there is nothing to undo.
    pub fn undo(&mut self) -> Option<String> {
        let step = self.steps.pop()?;
        match step {
            Step::Pixels => match self.apply_pixels(true) {
                Some(label) => {
                    self.undone.push(Step::Pixels);
                    Some(label)
                }
                None => {
                    // Cannot happen while the lists are synced; if it did,
                    // the step is unreplayable and is dropped.
                    self.sync_with_journal();
                    None
                }
            },
            Step::Structure {
                label,
                merge_key,
                other,
            } => {
                let now = self.snapshot();
                self.restore(*other);
                self.structure_generation += 1;
                self.undone.push(Step::Structure {
                    label: label.clone(),
                    merge_key,
                    other: Box::new(now),
                });
                Some(label)
            }
        }
    }

    /// Replay the newest undone step.
    pub fn redo(&mut self) -> Option<String> {
        let step = self.undone.pop()?;
        match step {
            Step::Pixels => {
                let label = self.apply_pixels(false)?;
                self.steps.push(Step::Pixels);
                Some(label)
            }
            Step::Structure {
                label,
                merge_key,
                other,
            } => {
                let now = self.snapshot();
                self.restore(*other);
                self.structure_generation += 1;
                self.steps.push(Step::Structure {
                    label: label.clone(),
                    merge_key,
                    other: Box::new(now),
                });
                Some(label)
            }
        }
    }

    /// Replay one journal entry, either way.
    ///
    /// The entry's SCOPE says which layer it belongs to — the newest
    /// edit is not necessarily on the layer that happens to be active —
    /// so the scope is resolved to a layer id FIRST, the restore lands
    /// there, and that layer becomes active so the change is visibly
    /// where it happened. The layer's shared pixels are materialized
    /// once (the journal splices into a mutable buffer) and re-shared.
    fn apply_pixels(&mut self, undo: bool) -> Option<String> {
        let scope = if undo {
            self.journal.undo_scope()
        } else {
            self.journal.redo_scope()
        }?;
        if scope & MASK_SCOPE_BIT != 0 {
            return self.apply_mask_entry(scope & !MASK_SCOPE_BIT, undo);
        }
        let idx = self.layers.iter().position(|l| l.id as u64 == scope)?;
        let (w, h) = (self.width, self.height);
        // The layer's OWN bytes, at its own depth — narrowing here
        // would make an undo lose precision the edit had kept.
        let depth = self.layers[idx].rgba.depth();
        let bpp = self.layers[idx].rgba.canvas().bytes_per_pixel();
        let mut buf: Vec<u8> = self.layers[idx].rgba.canvas().raw().to_vec();
        let label = {
            let mut view = FlatImage::new(w, h, bpp, buf.as_mut_slice())?;
            if undo {
                self.journal.undo(&mut view)
            } else {
                self.journal.redo(&mut view)
            }
        }?;
        self.layers[idx].rgba =
            crate::pixels::Pixels::from_raw(Arc::from(buf.into_boxed_slice()), depth).into();
        if idx != self.active {
            self.edit_mask = false;
        }
        self.active = idx;
        Some(label)
    }

    /// Replay one MASK journal entry into layer `id`'s mask. The layer
    /// becomes active with its MASK as the edit target, so the change is
    /// visibly where it happened.
    fn apply_mask_entry(&mut self, id: u64, undo: bool) -> Option<String> {
        let idx = self.layers.iter().position(|l| u64::from(l.id) == id)?;
        let (w, h) = (self.width, self.height);
        let mut buf: Vec<u8> = self.layers[idx].mask.as_ref()?.canvas().data().to_vec();
        let label = {
            let mut view = FlatImage::new(w, h, 1, buf.as_mut_slice())?;
            if undo {
                self.journal.undo(&mut view)
            } else {
                self.journal.redo(&mut view)
            }
        }?;
        self.layers[idx].mask = Some(Arc::new(SelectionCoverage::from_data(w, h, buf)?).into());
        self.active = idx;
        self.edit_mask = true;
        Some(label)
    }

    pub fn history(&self) -> HistoryStats {
        let b = self.journal.budget();
        HistoryStats {
            can_undo: !self.steps.is_empty(),
            can_redo: !self.undone.is_empty(),
            depth: self.steps.len(),
            redo_depth: self.undone.len(),
            bytes: self.journal.bytes(),
            max_bytes: b.max_bytes,
            max_entries: b.max_entries,
            dropped: self.journal.dropped() + self.dropped_steps,
            generation: self.journal.generation() + self.structure_generation,
        }
    }

    pub fn undo_label(&self) -> Option<&str> {
        self.undo_labels().pop()
    }

    pub fn redo_label(&self) -> Option<&str> {
        self.redo_labels().first().copied()
    }

    /// Every retained undo step, oldest first — what a History panel
    /// lists (pixel labels come from the journal, in the same order).
    pub fn undo_labels(&self) -> Vec<&str> {
        let mut pixel = self.journal.undo_labels().into_iter();
        self.steps
            .iter()
            .map(|s| match s {
                Step::Pixels => pixel.next().unwrap_or("Edit"),
                Step::Structure { label, .. } => label.as_str(),
            })
            .collect()
    }

    /// Every redo step, next-to-replay first.
    pub fn redo_labels(&self) -> Vec<&str> {
        let mut pixel = self.journal.redo_labels().into_iter();
        self.undone
            .iter()
            .rev()
            .map(|s| match s {
                Step::Pixels => pixel.next().unwrap_or("Edit"),
                Step::Structure { label, .. } => label.as_str(),
            })
            .collect()
    }

    /// Drop the history.
    pub fn clear_history(&mut self) {
        self.journal.clear();
        self.dropped_steps += self.steps.len() as u64;
        self.steps.clear();
        self.undone.clear();
    }

    // ───────────────────────── the composite ────────────────────────

    /// Can the composite run without a GPU? True exactly when the fast
    /// path applies (see the module docs) — the doors use it so a
    /// GPU-less realm still gets its one-layer document.
    pub fn composite_is_trivial(&self) -> bool {
        let mut plates = self.plates_indexed(None).into_iter();
        match (plates.next(), plates.next()) {
            (None, _) => true,
            (Some((i, _)), None) => self.layers[i].is_plain(),
            _ => false,
        }
    }

    /// The layers that actually contribute, bottom-first, paired with
    /// the pixels to composite (the active layer's may be overridden by
    /// an in-flight stroke). Hidden, zero-opacity and fully transparent
    /// layers drop out here — each is exactly the identity in the fold.
    /// Canvas-sized: a bounded layer is expanded here (the reference fold
    /// and other whole-canvas readers); the resident fold uses
    /// [`Self::plates_indexed`], which keeps the bounds.
    #[cfg_attr(not(any(test, feature = "reference-fold")), allow(dead_code))]
    fn plates<'a>(&'a self, override_active: Option<&'a Arc<[u8]>>) -> Vec<(&'a Layer, Arc<[u8]>)> {
        self.plates_indexed(override_active)
            .into_iter()
            .map(|(i, p)| {
                let px = match p.rect {
                    Some(_) => self.layers[i].rgba.raw_arc(),
                    None => p.px,
                };
                (&self.layers[i], px)
            })
            .collect()
    }

    /// The contributing layers by index, each with its plate: canvas-sized,
    /// or a bounded layer's own rectangle and pixels (ADR 464).
    fn plates_indexed(&self, override_active: Option<&Arc<[u8]>>) -> Vec<(usize, Plate)> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if !l.enabled() {
                    return None;
                }
                // An OWNED `Arc` rather than a borrow: `Pixels` hands
                // out a clone of its handle, not a reference into
                // itself. The clone is a refcount bump, not a copy.
                let plate = match (override_active, l.rgba.bounded()) {
                    (Some(o), _) if i == self.active => Plate {
                        px: Arc::clone(o),
                        rect: None,
                    },
                    (_, Some(b)) => Plate {
                        px: b.px.raw_arc(),
                        rect: Some(b.rect),
                    },
                    _ => Plate {
                        px: l.rgba.raw_arc(),
                        rect: None,
                    },
                };
                // An ADJUSTMENT layer has no pixels of its own — its
                // `rgba` is a transparent placeholder — so the
                // transparency skip would drop exactly the layers whose
                // whole job is to change what is beneath them.
                if l.is_pixels() && is_fully_transparent(&plate.px) {
                    return None;
                }
                Some((i, plate))
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
        let plates = self.plates_indexed(override_active);
        crate::counters::bump(|c| c.composites += 1);

        match plates.as_slice() {
            // Nothing contributes: an honest transparent canvas.
            [] => return Ok(self.transparent()),
            // The identity fold — the pixels ARE the composite. Handed
            // back as the very same allocation (an `Arc` clone), so a
            // one-layer document costs nothing to composite.
            [(i, p)] if self.layers[*i].is_plain() => {
                return Ok(match p.rect {
                    Some(_) => self.layers[*i].rgba.raw_arc(),
                    None => Arc::clone(&p.px),
                })
            }
            _ => {}
        }
        let ctx = ctx.ok_or_else(|| {
            IngestError::Unsupported(
                "compositing layers is GPU-only — call init_gpu first (the blend \
                 is a registered WGSL kernel dispatch; no CPU blend path ships)"
                    .into(),
            )
        })?;
        self.composite_resident(ctx, &plates).await
    }

    /// The fold as it ran before it became resident: every blend its own
    /// upload, dispatch and readback. Kept, test-only, as the reference
    /// the resident fold is proven byte-equal against.
    #[cfg(any(test, feature = "reference-fold"))]
    #[doc(hidden)]
    pub async fn composite_reference(
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

        crate::counters::bump(|c| c.layers_folded += plates.len() as u64);
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
        // GROUPS. An ISOLATED group composites its members into their own
        // accumulator, which is then blended in as one plate — that is
        // the difference between a group and a folder. A PASS-THROUGH
        // group (Photoshop's default) does not isolate at all; its
        // members reach the stack below directly, which is why an
        // adjustment inside one affects everything beneath the group.
        //
        // NESTING needs no tree of layers, only a STACK of parked
        // accumulators: at each layer, close the open groups that no
        // longer enclose it and open the ones that newly do. The
        // ancestor chain is what those comparisons run on.
        let mut open: Vec<(u32, Vec<u8>)> = Vec::new();
        // CLIPPING GROUPS (the resident fold's ClipOpen/ClipClose, op for
        // op): the last unclipped pixel layer as it was blended, which a
        // clipped layer above it reopens as a clipping group.
        let mut last_base: Option<RefClipBase> = None;
        let mut clip_group: Option<RefClipGroup> = None;
        for (layer, px) in plates {
            let want = self.chain_of(layer.group);
            // A layer inside ANY hidden group contributes nothing —
            // exact, not an approximation, by the same `over(a, b·0)`
            // argument the per-layer check makes.
            if want.iter().any(|g| !self.group_enabled(*g)) {
                continue;
            }
            if !layer.clipped {
                if let Some(g) = clip_group.take() {
                    acc = ref_clip_close(ctx, &acc, g, w, h).await?;
                }
            }
            // Close what no longer encloses this layer, innermost first.
            while let Some((gid, _)) = open.last() {
                let still = want.contains(gid);
                if still {
                    break;
                }
                if let Some(g) = clip_group.take() {
                    acc = ref_clip_close(ctx, &acc, g, w, h).await?;
                }
                last_base = None;
                let (gid, parked) = open.pop().expect("non-empty");
                let inner = std::mem::replace(&mut acc, parked);
                acc = self.blend_group(ctx, gid, acc, inner, w, h).await?;
                clip_base = None;
            }
            // Open what newly does, outermost first. Only an ISOLATED
            // group parks a buffer; a pass-through one is bookkeeping.
            for gid in want.iter().copied() {
                if open.iter().any(|(o, _)| *o == gid) {
                    continue;
                }
                let isolates = self
                    .groups
                    .iter()
                    .find(|g| g.id == gid)
                    .is_some_and(|g| g.isolates());
                if isolates {
                    if let Some(g) = clip_group.take() {
                        acc = ref_clip_close(ctx, &acc, g, w, h).await?;
                    }
                    last_base = None;
                    let parked =
                        std::mem::replace(&mut acc, vec![0u8; (w as usize) * (h as usize) * 8]);
                    open.push((gid, parked));
                    clip_base = None;
                }
            }
            // THE CLIP BASE: the alpha of the most recent unclipped
            // pixel plate, which is what every clipped layer stacked on
            // top of it is confined to. Tracked here, after the group
            // bookkeeping, because opening or closing a group resets it
            // — a clip cannot reach across a group boundary any more
            // than it can reach across the bottom of the stack.
            if layer.clipped {
                if clip_base.is_none() {
                    // Nothing to clip to: contributing unclipped would be
                    // the one behaviour a designer cannot recover from.
                    continue;
                }
                if clip_group.is_none() {
                    if let Some(base) = last_base.take() {
                        let (g, opaque) = ref_clip_open(ctx, base, w, h).await?;
                        clip_group = Some(g);
                        acc = opaque;
                    }
                }
            } else if layer.is_pixels() {
                clip_base = Some(alpha_of(&px));
            }

            // The clip base multiplies into the layer's own coverage,
            // so a clipped-AND-masked layer is confined by both. Two
            // coverages multiply; they do not override each other.
            // Inside a clipping group the base's alpha is applied once,
            // at its close, instead.
            let clip = if layer.clipped && clip_group.is_none() {
                clip_base.as_deref()
            } else {
                None
            };
            if let Some(params) = layer.adjust_params() {
                // STAYS IN f16. This used to unpremultiply, drop to
                // 8-bit, adjust, and lift back — quantizing the backdrop
                // TWICE per adjustment layer. Three stacked adjustments
                // meant three round trips through 256 levels, which is
                // where banding in a gradient actually came from; the
                // kernels were always f16 and only this bridge was not.
                let straight = unpremultiply(ctx, &acc, w, h).await?;
                let adjusted = crate::ingest::adjust_f16(
                    ctx,
                    w,
                    h,
                    &straight,
                    params,
                    with_opacity(
                        effective_coverage(
                            multiply_coverage(
                                layer.live_mask().cloned(),
                                self.pass_through_mask(layer.group),
                            )
                            .as_ref(),
                            clip,
                            w,
                            h,
                        ),
                        layer.opacity,
                        w,
                        h,
                    ),
                )
                .await?;
                acc = premultiply(ctx, &adjusted, w, h).await?;
                last_base = None;
                continue;
            }

            let straight = rgba8_to_f16(&px);
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
            let own = multiply_coverage(
                layer.live_mask().cloned(),
                self.pass_through_mask(layer.group),
            );
            let mask_bytes: Option<Vec<u8>> =
                effective_coverage(own.as_ref(), clip, w, h).map(|cov| {
                    SelectionMask::from_fn(w, h, |x, y| f32::from(cov.coverage_at(x, y)) / 255.0)
                        .bytes()
                        .to_vec()
                });
            let below = (!layer.clipped).then(|| acc.clone());
            // A text layer in Normal mode blends in a gamma space.
            let gamma = layer
                .blend_gamma
                .filter(|_| std::ptr::eq(layer.blend, &COMPOSE_NORMAL));
            let (kernel, params): (&'static KernelDef, Vec<u8>) = match gamma {
                Some(g) => (
                    &image_kernels::families::compose::COMPOSE_NORMAL_GAMMA,
                    image_kernels::families::compose::ComposeGammaParams::new(layer.opacity, g)
                        .as_bytes()
                        .to_vec(),
                ),
                None => (
                    layer.blend,
                    ComposeParams::new(layer.opacity).as_bytes().to_vec(),
                ),
            };
            acc = image_gpu::execute_tile_once_async(
                ctx,
                kernel,
                &[
                    TileInput { f16_bytes: &acc },
                    TileInput { f16_bytes: &premul },
                ],
                // The layer's opacity IS the compose family's α: the
                // spine computes `over(a, b·α)`.
                &params,
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
            last_base = below.map(|below| RefClipBase {
                below,
                premul,
                mask: mask_bytes,
                blend: layer.blend,
                opacity: layer.opacity,
            });
        }

        if let Some(g) = clip_group.take() {
            acc = ref_clip_close(ctx, &acc, g, w, h).await?;
        }
        // Groups still open at the top of the stack close here,
        // innermost first.
        while let Some((gid, parked)) = open.pop() {
            let inner = std::mem::replace(&mut acc, parked);
            acc = self.blend_group(ctx, gid, acc, inner, w, h).await?;
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

/// The reference fold's record of the last unclipped pixel layer.
#[cfg(any(test, feature = "reference-fold"))]
struct RefClipBase {
    below: Vec<u8>,
    premul: Vec<u8>,
    mask: Option<Vec<u8>>,
    blend: &'static KernelDef,
    opacity: f32,
}

/// The reference fold's open clipping group.
#[cfg(any(test, feature = "reference-fold"))]
struct RefClipGroup {
    parked: Vec<u8>,
    base: Vec<u8>,
    blend: &'static KernelDef,
    opacity: f32,
}

/// `fold::clip_base` on bytes: the base alone, and its colour opaque.
#[cfg(any(test, feature = "reference-fold"))]
async fn ref_clip_open(
    ctx: &GpuContext,
    b: RefClipBase,
    w: u32,
    h: u32,
) -> Result<(RefClipGroup, Vec<u8>), IngestError> {
    use image_kernels::families::band::{BandSetAlphaParams, BAND_SET_ALPHA};
    let zeros = vec![0u8; (w as usize) * (h as usize) * 8];
    let base = image_gpu::execute_tile_once_async(
        ctx,
        &COMPOSE_NORMAL,
        &[
            TileInput { f16_bytes: &zeros },
            TileInput {
                f16_bytes: &b.premul,
            },
        ],
        ComposeParams::new(1.0).as_bytes(),
        b.mask.as_deref(),
        w,
        h,
    )
    .await
    .map_err(|e| IngestError::Pipeline(e.to_string()))?;
    let straight = dispatch_unary(
        ctx,
        &CAST_UNPREMULTIPLY,
        CastUnpremultiplyParams::new().as_bytes(),
        &base,
        w,
        h,
    )
    .await?;
    let opaque = dispatch_unary(
        ctx,
        &BAND_SET_ALPHA,
        BandSetAlphaParams::new(1.0).as_bytes(),
        &straight,
        w,
        h,
    )
    .await?;
    Ok((
        RefClipGroup {
            parked: b.below,
            base,
            blend: b.blend,
            opacity: b.opacity,
        },
        opaque,
    ))
}

/// `fold::clip_close` on bytes.
#[cfg(any(test, feature = "reference-fold"))]
async fn ref_clip_close(
    ctx: &GpuContext,
    inner: &[u8],
    g: RefClipGroup,
    w: u32,
    h: u32,
) -> Result<Vec<u8>, IngestError> {
    use image_kernels::families::arithmetic::{MathMulParams, MATH_MUL};
    use image_kernels::families::band::{BandBroadcastAlphaParams, BAND_BROADCAST_ALPHA};
    let pipe = |e: image_gpu::GpuError| IngestError::Pipeline(e.to_string());
    let alpha = dispatch_unary(
        ctx,
        &BAND_BROADCAST_ALPHA,
        BandBroadcastAlphaParams::new().as_bytes(),
        &g.base,
        w,
        h,
    )
    .await?;
    let group = image_gpu::execute_tile_once_async(
        ctx,
        &MATH_MUL,
        &[
            TileInput { f16_bytes: inner },
            TileInput { f16_bytes: &alpha },
        ],
        MathMulParams::new().as_bytes(),
        None,
        w,
        h,
    )
    .await
    .map_err(pipe)?;
    image_gpu::execute_tile_once_async(
        ctx,
        g.blend,
        &[
            TileInput {
                f16_bytes: &g.parked,
            },
            TileInput { f16_bytes: &group },
        ],
        ComposeParams::new(g.opacity).as_bytes(),
        None,
        w,
        h,
    )
    .await
    .map_err(pipe)
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

/// An adjustment layer's coverage with its OPACITY folded in: the chain
/// mixes `mix(backdrop, adjusted, coverage)`, so opacity is a uniform
/// scale of the coverage (an unmasked layer's coverage being 1). That is
/// Photoshop's rule — an adjustment at 60 % moves each pixel 60 % of the
/// way, and the backdrop's alpha is untouched. The fold used to ignore an
/// adjustment layer's opacity: a Curves layer at 60 % came out at full
/// strength, 13 levels from Photoshop's composite.
pub(crate) fn with_opacity(
    cov: Option<Arc<SelectionCoverage>>,
    opacity: f32,
    w: u32,
    h: u32,
) -> Option<Arc<SelectionCoverage>> {
    if opacity >= 1.0 {
        return cov;
    }
    let o = opacity.clamp(0.0, 1.0);
    let scale = |v: u8| (f32::from(v) * o).round() as u8;
    let data = match &cov {
        Some(c) => c.data().iter().map(|&v| scale(v)).collect(),
        None => vec![scale(255); (w as usize) * (h as usize)],
    };
    SelectionCoverage::from_data(w, h, data).map(Arc::new)
}

/// The gamma a text layer blends in. Photoshop composites TEXT layers in
/// a gamma space ("Blend Text Colors Using Gamma"; its colour settings
/// say 1.45). Fitted to text layers Photoshop 27.10 wrote: the best power
/// on the stored pixels is 1.52–1.6 depending on the colours, so this is
/// an approximation — 19–40 levels on anti-aliased edges become 2–7.
pub const TEXT_BLEND_GAMMA: f32 = 1.55;

/// One Levels record as a 256-entry table: the input range normalised,
/// raised to 1/gamma, mapped onto the output range.
fn levels_lut(r: &image_psd::adjustment::LevelsRecord) -> [u8; 256] {
    let (ib, iw) = (f64::from(r.in_black), f64::from(r.in_white));
    let (ob, ow) = (f64::from(r.out_black), f64::from(r.out_white));
    let g = (f64::from(r.gamma_x100) / 100.0).max(0.01);
    let mut lut = [0u8; 256];
    for (x, slot) in lut.iter_mut().enumerate() {
        let t = ((x as f64 - ib) / (iw - ib).max(1.0)).clamp(0.0, 1.0);
        *slot = (ob + t.powf(1.0 / g) * (ow - ob)).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

/// Photoshop's Exposure as a table: the code decoded to light with a 2.2
/// power, scaled by 2^exposure, offset, raised to 1/gamma, re-encoded.
/// Measured on a Photoshop-written layer (exposure 0.4, offset −0.03,
/// gamma 1.2): within 1 level.
fn exposure_lut(exposure: f32, offset: f32, gamma: f32) -> [u8; 256] {
    let (e, o, g) = (
        f64::from(exposure),
        f64::from(offset),
        f64::from(gamma).max(0.01),
    );
    let mut lut = [0u8; 256];
    for (x, slot) in lut.iter_mut().enumerate() {
        let light = (x as f64 / 255.0).powf(2.2) * e.exp2() + o;
        let v = light.max(0.0).powf(1.0 / g).min(1.0).powf(1.0 / 2.2);
        *slot = (v * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

/// A PSD mask plate as a layer mask: bounded at its rectangle, or
/// canvas-sized.
fn psd_layer_mask(
    m: image_psd::MaskPlate,
    width: u32,
    height: u32,
) -> Result<LayerMask, IngestError> {
    let bad = || IngestError::Decode("a PSD layer mask does not fit the canvas".into());
    match m.rect {
        Some(r) if r != Region::new(0, 0, width, height) => {
            BoundedMask::new(r, 0, m.coverage, width, height)
                .map(|b| LayerMask::Bounded(Arc::new(b)))
                .ok_or_else(bad)
        }
        _ => SelectionCoverage::from_data(width, height, m.coverage)
            .map(|c| LayerMask::Canvas(Arc::new(c)))
            .ok_or_else(bad),
    }
}

/// A layer's pixels at `rect` (ADR 464); the whole canvas stays the
/// plain canvas shape.
fn bounded_pixels(
    rect: Region,
    rgba: Vec<u8>,
    width: u32,
    height: u32,
) -> Result<crate::pixels::LayerPixels, IngestError> {
    let px = crate::pixels::Pixels::from_rgba8(Arc::from(rgba.into_boxed_slice()));
    if rect == Region::new(0, 0, width, height) {
        return Ok(px.into());
    }
    crate::pixels::Bounded::new(rect, px, width, height)
        .map(crate::pixels::LayerPixels::Bounded)
        .ok_or_else(|| {
            IngestError::Decode(format!(
                "a layer rectangle {rect:?} does not fit the {width}×{height} canvas"
            ))
        })
}

/// A PSD adjustment layer as the adjust chain's parameters. Only what
/// `image_psd::adjustment::Adjustment::unmodelled` lets through reaches
/// here, so every field it reads has a counterpart.
pub fn psd_adjust_params(adj: &image_psd::adjustment::Adjustment) -> AdjustParams {
    use image_psd::adjustment::Adjustment;
    let mut p = AdjustParams::default();
    let lut = |pts: &[(u8, u8)]| {
        let pts: Vec<(f32, f32)> = pts
            .iter()
            .map(|&(i, o)| (f32::from(i) / 255.0, f32::from(o) / 255.0))
            .collect();
        // Photoshop's Curves is a natural cubic spline (`curve_lut_natural`).
        image_core::curve_lut_natural(&pts)
    };
    match adj {
        Adjustment::Curves { curves } => {
            p.curve_lut = curves[0].as_deref().map(lut);
            if curves[1..].iter().any(Option::is_some) {
                let id = lut(&[(0, 0), (255, 255)]);
                p.curve_rgb = Some(Box::new(
                    [1, 2, 3].map(|c| curves[c].as_deref().map_or(id, lut)),
                ));
            }
        }
        // Levels as TABLES, the way Photoshop applies them: each channel's
        // record first, then the composite's — so a per-channel OUTPUT
        // range, which the levels stages have no field for, is exact too.
        // Measured on Photoshop-written layers: within 2 levels (channel
        // then composite; the other order is 13 off).
        Adjustment::Levels { records, .. } => {
            p.curve_lut = Some(levels_lut(&records[0]));
            if records[1..].iter().any(|r| !r.is_identity()) {
                p.curve_rgb = Some(Box::new([1, 2, 3].map(|c| levels_lut(&records[c]))));
            }
        }
        // Plain exposure is the exposure stage (it agrees with Photoshop
        // within 2 levels); an offset or gamma becomes a table.
        Adjustment::Exposure {
            exposure,
            offset,
            gamma,
        } => {
            if *offset == 0.0 && *gamma == 1.0 {
                p.exposure_ev = *exposure;
            } else {
                p.curve_lut = Some(exposure_lut(*exposure, *offset, *gamma));
            }
        }
        Adjustment::Invert => p.invert = true,
        Adjustment::HueSaturation {
            colorized,
            colorize,
            master,
            ranges,
            ..
        } => {
            let hsl = |t: &[i16; 3]| {
                [
                    f32::from(t[0]),
                    f32::from(t[1]) / 100.0,
                    f32::from(t[2]) / 100.0,
                    0.0,
                ]
            };
            p.hue_sat.master = hsl(master);
            for (k, r) in ranges.iter().enumerate() {
                p.hue_sat.ranges[k] = hsl(r);
            }
            if *colorized {
                p.hue_sat.colorize = [
                    1.0,
                    f32::from(colorize[0]).rem_euclid(360.0),
                    f32::from(colorize[1]) / 100.0,
                    f32::from(colorize[2]) / 100.0,
                ];
            }
        }
    }
    p
}

/// Levels (over white) beyond which a pixel of a smart object's
/// footprint counts as DISAGREEING with the merged composite.
pub const SMART_RENDER_LEVELS: u8 = 8;

/// The share of a smart object's visible footprint, in percent, that may
/// disagree before its stored render is judged stale.
pub const SMART_RENDER_MAX_PCT: f64 = 1.0;

/// How one smart object's stored render compares with the composite.
#[derive(Debug, Clone, PartialEq)]
pub struct SmartRenderAgreement {
    pub name: String,
    /// Pixels where the plate has coverage.
    pub footprint: usize,
    /// Percent of the footprint more than [`SMART_RENDER_LEVELS`] off.
    pub pct_off: f64,
    /// Mean absolute difference over the footprint, in levels.
    pub mean: f64,
}

/// Measure every smart-object plate of `import` against the file's
/// merged composite: `ours` is the stack's flatten of the same import,
/// `theirs` Photoshop's composite (both straight RGBA8, compared over
/// white). Only the plate's own footprint is read — where its pixels and
/// its enabled mask have coverage — so a disagreement elsewhere in the
/// file is not blamed on it.
///
/// What it can and cannot see: a stale render that SHOWS differs from
/// the composite and is caught; a smart object that is hidden or wholly
/// covered by the layers above it agrees vacuously, which is also what
/// Photoshop shows until the user reveals it.
pub fn smart_render_agreement(
    import: &image_psd::LayerImport,
    ours: &[u8],
    theirs: &[u8],
) -> Vec<SmartRenderAgreement> {
    let over_white = |p: &[u8], c: usize| -> i32 {
        let a = u32::from(p[3]);
        ((u32::from(p[c]) * a + 255 * (255 - a) + 127) / 255) as i32
    };
    let mut out = Vec::new();
    for plate in import.layers.iter().filter(|p| p.smart) {
        let mask = crate::psd_vector_mask::plate_mask(plate, import.width, import.height);
        let mut footprint = 0usize;
        let mut off = 0usize;
        let mut sum = 0u64;
        // The plate covers its rectangle (or the canvas); `i` is the
        // canvas index of each of its pixels.
        let r = plate
            .rect
            .unwrap_or(Region::new(0, 0, import.width, import.height));
        for (k, px) in plate.rgba.chunks_exact(4).enumerate() {
            if px[3] == 0 || r.w == 0 {
                continue;
            }
            let (x, y) = (
                r.x as usize + k % r.w as usize,
                r.y as usize + k / r.w as usize,
            );
            let i = y * import.width as usize + x;
            if let Some(m) = mask.as_ref().filter(|m| m.enabled) {
                // The mask covers the plate's rectangle, or the canvas.
                let mi = match m.rect {
                    Some(mr) => (y - mr.y as usize) * mr.w as usize + (x - mr.x as usize),
                    None => i,
                };
                if m.coverage[mi] == 0 {
                    continue;
                }
            }
            let (Some(a), Some(b)) = (ours.get(i * 4..i * 4 + 4), theirs.get(i * 4..i * 4 + 4))
            else {
                continue;
            };
            footprint += 1;
            let d = (0..3)
                .map(|c| (over_white(a, c) - over_white(b, c)).unsigned_abs())
                .max()
                .unwrap_or(0);
            sum += u64::from(d);
            if d > u32::from(SMART_RENDER_LEVELS) {
                off += 1;
            }
        }
        let n = footprint.max(1) as f64;
        out.push(SmartRenderAgreement {
            name: plate.name.clone(),
            footprint,
            pct_off: 100.0 * off as f64 / n,
            mean: sum as f64 / n,
        });
    }
    out
}

/// The verdict: `Err` names the first smart object whose stored render
/// the composite does not vouch for.
pub fn smart_renders_agree(
    import: &image_psd::LayerImport,
    ours: &[u8],
    theirs: &[u8],
) -> Result<usize, IngestError> {
    let checked = smart_render_agreement(import, ours, theirs);
    if let Some(bad) = checked.iter().find(|a| a.pct_off > SMART_RENDER_MAX_PCT) {
        return Err(IngestError::Unsupported(format!(
            "layer import of a PSD whose smart object \"{}\" disagrees with the file's own \
             composite ({:.1}% of its pixels more than {SMART_RENDER_LEVELS} levels off): its \
             stored render is stale, and rendering the embedded source is not modelled, so \
             the merged composite is kept instead",
            bad.name, bad.pct_off
        )));
    }
    Ok(checked.len())
}

/// A CMYK document's layered flatten may differ from the file's own
/// composite (both converted by the same transform) by more than this
/// many levels on at most [`CMYK_FLATTEN_MAX_PCT`] of its pixels. The
/// same bar a smart object's stored render must clear, for the same
/// reason: it is the line between "anti-aliased edges mix a little
/// differently" and "this is a different-looking image".
pub const CMYK_FLATTEN_LEVELS: u8 = SMART_RENDER_LEVELS;

/// The share of the canvas, in percent, that may disagree.
pub const CMYK_FLATTEN_MAX_PCT: f64 = SMART_RENDER_MAX_PCT;

/// The share of the canvas, in percent, that may be more than 2 levels
/// off. The first bar catches a different-looking image; this one a
/// broad TINT: Multiply blended in RGB over a CMYK document's pale stock
/// stayed under 8 levels everywhere yet shifted 19–42 % of four corpus
/// pages by 3–8, while the CMYK files whose layers mix only at their
/// edges stay at or below 1.5 %.
pub const CMYK_FLATTEN_MAX_PCT_SHIFT: f64 = 5.0;

/// How a CMYK document's RGB flatten compares with its converted merged
/// composite, over the whole canvas (over white).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CmykFlattenAgreement {
    /// Percent of pixels more than [`CMYK_FLATTEN_LEVELS`] off.
    pub pct_off: f64,
    /// Percent of pixels more than 2 levels off.
    pub pct_shift: f64,
    /// Mean absolute difference (worst channel per pixel), in levels.
    pub mean: f64,
    /// Worst difference, in levels.
    pub max: u8,
}

/// Measure `ours` (the stack's RGB flatten of a CMYK import) against
/// `theirs` (the file's merged CMYK composite, converted by the SAME ink
/// transform — so the CMM is not on trial here, only the blend space).
pub fn cmyk_flatten_agreement(ours: &[u8], theirs: &[u8]) -> CmykFlattenAgreement {
    let over_white = |p: &[u8], c: usize| -> i32 {
        let a = u32::from(p[3]);
        ((u32::from(p[c]) * a + 255 * (255 - a) + 127) / 255) as i32
    };
    let (mut n, mut off, mut shift, mut sum, mut max) = (0usize, 0usize, 0usize, 0u64, 0u32);
    for (a, b) in ours.chunks_exact(4).zip(theirs.chunks_exact(4)) {
        n += 1;
        let d = (0..3)
            .map(|c| (over_white(a, c) - over_white(b, c)).unsigned_abs())
            .max()
            .unwrap_or(0)
            .max(u32::from(a[3].abs_diff(b[3])));
        sum += u64::from(d);
        max = max.max(d);
        if d > u32::from(CMYK_FLATTEN_LEVELS) {
            off += 1;
        }
        if d > 2 {
            shift += 1;
        }
    }
    let n = n.max(1) as f64;
    CmykFlattenAgreement {
        pct_off: 100.0 * off as f64 / n,
        pct_shift: 100.0 * shift as f64 / n,
        mean: sum as f64 / n,
        max: max.min(255) as u8,
    }
}

/// Accept a CMYK document's layered import only when its RGB flatten
/// agrees with the file's own composite (see [`CMYK_FLATTEN_LEVELS`]).
/// Photoshop blends a CMYK document in CMYK; the stack blends the
/// converted plates in RGB. Opaque pixels agree exactly; mixed ones
/// (soft edges, partial opacity) by an amount that depends on the
/// colours, which only this measurement can tell.
pub fn cmyk_flatten_agrees(
    ours: &[u8],
    theirs: &[u8],
) -> Result<CmykFlattenAgreement, IngestError> {
    let a = cmyk_flatten_agreement(ours, theirs);
    if a.pct_shift > CMYK_FLATTEN_MAX_PCT_SHIFT {
        return Err(IngestError::Unsupported(format!(
            "layer import of a CMYK document whose layers, converted to RGB and blended \
             there, shift {:.1}% of the pixels by more than 2 levels from Photoshop's CMYK \
             composite (a tint, not an edge): Photoshop mixes the inks, the RGB stack mixes \
             the converted colours, so the merged composite is kept instead",
            a.pct_shift
        )));
    }
    if a.pct_off > CMYK_FLATTEN_MAX_PCT {
        return Err(IngestError::Unsupported(format!(
            "layer import of a CMYK document whose layers, converted to RGB and blended \
             there, disagree with Photoshop's CMYK composite ({:.1}% of the pixels more \
             than {CMYK_FLATTEN_LEVELS} levels off, worst {}): Photoshop mixes the inks, \
             the RGB stack mixes the converted colours, so the merged composite is kept \
             instead",
            a.pct_off, a.max
        )));
    }
    Ok(a)
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
        image_gpu::test_support::device_or_skip("layers")
    }

    #[test]
    #[allow(non_snake_case)]
    fn an_adjustment_layer_is_editable_and_only_an_adjustment_layer__feat__image_editor_layers() {
        let mut s = stack(2, 2);
        let a = s.add_adjustment(
            "brighten",
            AdjustParams {
                exposure_ev: 0.5,
                ..AdjustParams::default()
            },
        );
        let p = AdjustParams {
            exposure_ev: -1.0,
            ..AdjustParams::default()
        };
        s.set_adjustment(a, p.clone()).expect("edit");
        match &s.layers[a].kind {
            LayerKind::Adjustment(q) => assert_eq!(q.exposure_ev, -1.0),
            _ => panic!("still an adjustment layer"),
        }
        assert!(
            s.set_adjustment(0, p.clone()).is_err(),
            "a pixel layer is not one"
        );
        s.set_locked(a, true).expect("lock");
        assert!(s.set_adjustment(a, p).is_err(), "locked");
    }

    #[test]
    #[allow(non_snake_case)]
    fn structure_and_pixel_steps_share_one_list_and_drags_merge__feat__image_editor_undo_journal() {
        let mut s = stack(4, 4);
        s.recorded("New layer", None, |st| Ok(st.add("B")))
            .expect("add");
        s.edit_active("Paint", Region::new(0, 0, 4, 4), px(4, 4, 7).into())
            .expect("paint");
        // A twenty-step opacity drag is ONE step.
        for k in 0..20 {
            s.recorded("Layer opacity", Some("opacity:1"), |st| {
                st.set_opacity(1, 1.0 - k as f32 / 40.0)
            })
            .expect("opacity");
        }
        assert_eq!(s.undo_labels(), vec!["New layer", "Paint", "Layer opacity"]);
        assert!((s.layers[1].opacity - (1.0 - 19.0 / 40.0)).abs() < 1e-6);
        // Undo walks back through both kinds, in order.
        assert_eq!(s.undo().as_deref(), Some("Layer opacity"));
        assert_eq!(s.layers[1].opacity, 1.0, "the drag's START value");
        assert_eq!(s.undo().as_deref(), Some("Paint"));
        assert_eq!(s.undo().as_deref(), Some("New layer"));
        assert_eq!(s.len(), 1);
        assert_eq!(s.undo(), None);
        assert_eq!(s.redo_labels(), vec!["New layer", "Paint", "Layer opacity"]);
        // A new step clears the redo list.
        s.redo().expect("redo");
        s.recorded("Rename layer", None, |st| st.set_name(1, "C"))
            .expect("rename");
        assert_eq!(s.redo(), None);
        assert!(s.history().generation > 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn a_failing_structure_edit_records_nothing__feat__image_editor_undo_journal() {
        let mut s = stack(2, 2);
        assert!(s
            .recorded("Opacity", None, |st| st.set_opacity(9, 0.5))
            .is_err());
        assert!(!s.history().can_undo);
    }

    // ── canvas operations ────────────────────────────────────────────

    /// A 3×2 canvas whose pixel at (x, y) has red = 10·y + x.
    fn numbered() -> LayerStack {
        let mut px = Vec::new();
        for y in 0..2u8 {
            for x in 0..3u8 {
                px.extend_from_slice(&[10 * y + x, 0, 0, 255]);
            }
        }
        LayerStack::from_image(3, 2, Arc::from(px.into_boxed_slice())).expect("valid")
    }

    fn reds(s: &LayerStack) -> Vec<u8> {
        s.active()
            .rgba
            .to_rgba8()
            .chunks_exact(4)
            .map(|p| p[0])
            .collect()
    }

    #[test]
    #[allow(non_snake_case)]
    fn canvas_rotate_and_flip_move_every_pixel_exactly__feat__image_editor_crop() {
        let mut s = numbered();
        s.transform_canvas(CanvasOp::RotateCw).expect("cw");
        assert_eq!((s.width, s.height), (2, 3));
        // Clockwise: the old bottom-left (10) is the new top-left.
        assert_eq!(reds(&s), vec![10, 0, 11, 1, 12, 2]);
        s.transform_canvas(CanvasOp::RotateCcw).expect("ccw");
        assert_eq!(reds(&s), reds(&numbered()), "cw then ccw is the identity");
        s.transform_canvas(CanvasOp::FlipHorizontal).expect("flip");
        assert_eq!(reds(&s), vec![2, 1, 0, 12, 11, 10]);
        s.transform_canvas(CanvasOp::FlipHorizontal)
            .expect("flip back");
        s.transform_canvas(CanvasOp::Rotate180).expect("180");
        assert_eq!(reds(&s), vec![12, 11, 10, 2, 1, 0]);
        s.transform_canvas(CanvasOp::FlipVertical).expect("v");
        assert_eq!(reds(&s), vec![2, 1, 0, 12, 11, 10]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn canvas_size_places_the_old_canvas_by_its_anchor__feat__image_editor_crop() {
        let mut s = numbered();
        s.transform_canvas(CanvasOp::Resize {
            width: 5,
            height: 2,
            anchor_x: 2,
            anchor_y: 0,
        })
        .expect("grow");
        // Anchored right: two transparent columns on the left.
        let px = s.active().rgba.to_rgba8().into_owned();
        let alpha: Vec<u8> = px.chunks_exact(4).map(|p| p[3]).collect();
        assert_eq!(alpha, vec![0, 0, 255, 255, 255, 0, 0, 255, 255, 255]);
        assert_eq!(reds(&s)[2..5], [0, 1, 2]);
        // Shrink centred: the middle column survives.
        let mut s = numbered();
        s.transform_canvas(CanvasOp::Resize {
            width: 1,
            height: 2,
            anchor_x: 1,
            anchor_y: 1,
        })
        .expect("shrink");
        assert_eq!(reds(&s), vec![1, 11]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn canvas_ops_move_masks_and_every_layer__feat__image_editor_layers() {
        let mut s = numbered();
        s.add("top");
        let cov = SelectionCoverage::from_data(3, 2, vec![255, 0, 0, 0, 0, 0]).expect("cov");
        s.layers[1].mask = Some(Arc::new(cov).into());
        s.transform_canvas(CanvasOp::FlipHorizontal).expect("flip");
        assert_eq!(
            s.layers[1].mask.as_ref().expect("mask").canvas().data(),
            &[0, 0, 255, 0, 0, 0]
        );
        assert_eq!(
            reds(&s),
            vec![0; 6],
            "the active (empty) layer moved too, harmlessly"
        );
        s.set_active(0).expect("active");
        assert_eq!(reds(&s), vec![2, 1, 0, 12, 11, 10]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn canvas_ops_are_undoable_and_refuse_resize_over_a_smart_object__feat__image_editor_layers() {
        let mut s = numbered();
        s.edit_active("paint", Region::new(0, 0, 3, 2), px(3, 2, 9).into())
            .expect("edit");
        s.transform_canvas(CanvasOp::RotateCw).expect("rotate");
        assert_eq!((s.width, s.height), (2, 3));
        assert_eq!(s.undo_labels(), vec!["paint", "Rotate 90° clockwise"]);
        // Undo the rotation, then the paint — each on the extent it knew.
        assert_eq!(s.undo().as_deref(), Some("Rotate 90° clockwise"));
        assert_eq!((s.width, s.height), (3, 2));
        assert_eq!(reds(&s), vec![9; 6]);
        assert_eq!(s.undo().as_deref(), Some("paint"));
        assert_eq!(reds(&s), reds(&numbered()));
        assert_eq!(s.redo().as_deref(), Some("paint"));
        assert_eq!(s.redo().as_deref(), Some("Rotate 90° clockwise"));
        assert_eq!((s.width, s.height), (2, 3));
        s.make_smart(0).expect("smart");
        let err = s
            .transform_canvas(CanvasOp::Resize {
                width: 4,
                height: 4,
                anchor_x: 0,
                anchor_y: 0,
            })
            .expect_err("refused");
        assert!(err.to_string().contains("smart object"));
        assert_eq!(
            s.undo_labels().last().copied(),
            Some("Rotate 90° clockwise"),
            "a refused op records nothing"
        );
        // Rotation is fine: the source rotates with the render.
        s.transform_canvas(CanvasOp::Rotate180)
            .expect("rotate smart");
    }

    // ── smart objects ────────────────────────────────────────────────

    #[test]
    fn image_editor_layers_converting_to_smart_preserves_the_pixels_as_source() {
        let mut s = stack(4, 4);
        let before = s.active().rgba.to_rgba8().into_owned();
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
        let original = s.active().rgba.to_rgba8().into_owned();
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
        assert!(Arc::ptr_eq(&s.active().rgba.raw_arc(), &pixels));
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
        assert!(
            s.active().rgba.to_rgba8().iter().all(|&b| b == 0),
            "transparent"
        );
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
            .edit_active("paint", Region::new(0, 0, 4, 4), px(4, 4, 9).into())
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
            clipped: false,
            group: None,
            mask: None,
            vector_mask: None,
            mask_params: None,
            smart: false,
            text: false,
            adjustment: None,
            color_overlay: None,
            name: name.to_string(),
            blend_key: *key,
            opacity,
            hidden,
            rgba: vec![0u8; 4 * 4 * 4],
            rect: None,
        };
        let import = image_psd::LayerImport {
            groups: Vec::new(),
            depth_reduced: false,
            converted_from_cmyk: false,
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
            groups: Vec::new(),
            depth_reduced: false,
            converted_from_cmyk: false,
            width: 4,
            height: 4,
            layers: vec![image_psd::LayerPlate {
                clipped: false,
                group: None,
                mask: None,
                vector_mask: None,
                mask_params: None,
                smart: false,
                text: false,
                adjustment: None,
                color_overlay: None,
                name: "short".into(),
                blend_key: *b"norm",
                opacity: 255,
                hidden: false,
                rgba: vec![0u8; 8],
                rect: None,
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
            Arc::ptr_eq(&out, &s.active().rgba.raw_arc()),
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
        assert!(Arc::ptr_eq(&out, &s.layers()[0].rgba.raw_arc()));
    }

    #[test]
    fn image_editor_layers_a_second_painted_layer_makes_the_composite_gpu_only() {
        let mut s = stack(4, 4);
        s.add("Paint");
        s.edit_active("fill", Region::new(0, 0, 4, 4), px(4, 4, 200).into())
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
        let base = s.active().rgba.to_rgba8().into_owned();
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
        let base = s.active().rgba.to_rgba8().into_owned();
        s.add("Cover");
        let white: Arc<[u8]> = px(16, 16, 255);
        s.edit_active("fill", Region::new(0, 0, 16, 16), Arc::clone(&white).into())
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
        s.edit_active("fill", Region::new(0, 0, 16, 16), Arc::clone(&white).into())
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
        let base = s.active().rgba.to_rgba8().into_owned();
        s.add("Cover");
        let white: Arc<[u8]> = px(16, 16, 255);
        s.edit_active("fill", Region::new(0, 0, 16, 16), Arc::clone(&white).into())
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
        s.edit_active("fill", Region::new(0, 0, 16, 16), Arc::clone(&white).into())
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
            crate::pixels::Pixels::from_rgba8(Arc::from(opaque_black.into_boxed_slice())),
        )
        .expect("unlocked");
        s.add("White");
        s.edit_active("fill", Region::new(0, 0, 8, 8), px(8, 8, 255).into())
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
            crate::pixels::Pixels::from_rgba8(Arc::from(half.into_boxed_slice())),
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
            s.active().rgba.to_rgba8().iter().all(|&b| b == 0),
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
        let mut stroke = StrokeSession::begin_on(1, w, h, s.active().rgba.raw_arc(), params, None)
            .expect("begin on the layer");
        pollster::block_on(stroke.extend(ctx, StrokeSample::new(32.0, 32.0, 1.0))).expect("extend");
        let damage = stroke.stroke_bounds().expect("a dot has bounds");
        let painted: Arc<[u8]> = Arc::from(stroke.commit().into_boxed_slice());

        let composite_before = pollster::block_on(s.composite(Some(ctx), None)).expect("fold");
        s.edit_active("Paint", damage, painted.into())
            .expect("unlocked");

        // (1) the photo layer never moved.
        assert!(
            Arc::ptr_eq(&s.layers()[0].rgba.raw_arc(), &photo),
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
        let before = s.active().rgba.to_rgba8().into_owned();
        let mut painted = before.clone();
        for p in painted.chunks_exact_mut(4) {
            p[0] = 200;
        }
        s.edit_active(
            "Paint",
            Region::new(0, 0, 300, 200),
            crate::pixels::Pixels::from_rgba8(Arc::from(painted.clone().into_boxed_slice())),
        )
        .expect("unlocked");
        assert_eq!(s.active().rgba.to_rgba8().into_owned(), painted);

        assert_eq!(s.undo().as_deref(), Some("Paint"));
        assert_eq!(
            s.active().rgba.to_rgba8().into_owned(),
            before,
            "byte-for-byte"
        );
        assert_eq!(s.redo().as_deref(), Some("Paint"));
        assert_eq!(s.active().rgba.to_rgba8().into_owned(), painted);
        assert!(s.undo().is_some());
        assert!(s.undo().is_none(), "nothing left to undo");
    }

    #[test]
    fn image_editor_undo_a_small_edit_journals_only_the_tiles_it_covered() {
        // 1024×1024 = 16 tiles; a stroke in one corner journals ONE.
        let mut s = stack(1024, 1024);
        let painted = s.active().rgba.to_rgba8().into_owned();
        s.edit_active(
            "Paint",
            Region::new(10, 10, 40, 40),
            crate::pixels::Pixels::from_rgba8(Arc::from(painted.into_boxed_slice())),
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
        let b_before = s.active().rgba.to_rgba8().into_owned();
        s.edit_active("Paint B", Region::new(0, 0, 64, 64), px(64, 64, 200).into())
            .expect("unlocked");
        let a_before = s.layers()[0].rgba.to_rgba8().into_owned();

        s.set_active(0).expect("in range");
        assert_eq!(s.undo().as_deref(), Some("Paint B"));
        assert_eq!(
            s.layers()[0].rgba.to_rgba8().into_owned(),
            a_before,
            "A is untouched"
        );
        assert_eq!(
            s.layers()[1].rgba.to_rgba8().into_owned(),
            b_before,
            "B is restored"
        );
        // …and the layer the undo landed in becomes active, so the
        // change is visibly where it happened.
        assert_eq!(s.active_index(), 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn image_editor_undo_removing_a_layer_is_one_step_and_its_paint_stays_reachable__feat__image_editor_undo_journal(
    ) {
        // Undo is LIFO: the paint on the removed layer can only be undone
        // after the removal is, which brings the layer back first.
        let mut s = stack(32, 32);
        s.add("B");
        s.edit_active("Paint", Region::new(0, 0, 32, 32), px(32, 32, 9).into())
            .expect("unlocked");
        s.remove(1).expect("not the last layer");
        assert_eq!(s.len(), 1);
        assert_eq!(s.undo_labels(), vec!["Paint", "Delete layer"]);
        assert_eq!(s.undo().as_deref(), Some("Delete layer"));
        assert_eq!(s.len(), 2);
        assert_eq!(
            s.layers[1].rgba.to_rgba8()[0],
            9,
            "the layer comes back painted"
        );
        assert_eq!(s.undo().as_deref(), Some("Paint"));
        assert_eq!(
            s.layers[1].rgba.to_rgba8()[3],
            0,
            "and the paint undoes on it"
        );
        assert_eq!(s.redo().as_deref(), Some("Paint"));
        assert_eq!(s.redo().as_deref(), Some("Delete layer"));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn image_editor_undo_a_damage_region_outside_the_canvas_records_nothing() {
        let mut s = stack(16, 16);
        let px16 = s.active().rgba.raw_arc();
        let out = s
            .edit_active("Paint", Region::new(500, 500, 10, 10), px16.into())
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
        let out =
            effective_coverage(Some(&LayerMask::Canvas(cov)), Some(&base), 4, 1).expect("combined");
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
        s.layer_mut(base).expect("base").rgba =
            crate::pixels::Pixels::from_rgba8(Arc::from(rgba.into_boxed_slice())).into();
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

    // ── groups ───────────────────────────────────────────────────────

    #[test]
    fn image_editor_layers_grouping_requires_a_contiguous_run() {
        let mut s = stack(8, 8);
        s.add("A");
        s.add("B");
        let id = s.group_range(1, 2, "Set").expect("grouped");
        assert_eq!(s.layers()[1].group, Some(id));
        assert_eq!(s.layers()[2].group, Some(id));
        assert_eq!(s.layers()[0].group, None, "the background stayed out");
        // Out of range is an error, not a clamp.
        assert!(s.group_range(1, 99, "Nope").is_err());
    }

    #[test]
    fn image_editor_layers_groups_nest() {
        // Nesting is supported: the inner group becomes a CHILD of the
        // one already enclosing its members. (An earlier version refused
        // this, honestly, because the fold parked one accumulator; it
        // parks a stack now and the limitation went with it.)
        let mut s = stack(8, 8);
        s.add("A");
        s.add("B");
        let outer = s.group_range(1, 2, "Outer").expect("grouped");
        let inner = s.group_range(1, 2, "Inner").expect("nests");
        assert_ne!(inner, outer);
        let g = s.groups().iter().find(|g| g.id == inner).expect("inner");
        assert_eq!(g.parent, Some(outer), "the inner group knows its parent");
        assert_eq!(s.layers()[1].group, Some(inner), "members moved inward");
    }

    #[test]
    fn image_editor_layers_a_group_cannot_straddle_two_parents() {
        // The one thing nesting still cannot be: a group inside two
        // things at once is not a tree, so the range is refused rather
        // than silently re-parented.
        let mut s = stack(8, 8);
        s.add("A");
        s.add("B");
        s.add("C");
        s.group_range(1, 1, "First").expect("grouped");
        s.group_range(2, 2, "Second").expect("grouped");
        let err = s.group_range(1, 2, "Across").expect_err("refused");
        assert!(
            format!("{err}").contains("straddles"),
            "the refusal must SAY why: {err}"
        );
    }

    #[test]
    fn image_editor_layers_nesting_is_bounded() {
        // Each open level parks a full-canvas buffer, so the depth bound
        // is a memory guard — and an explicit refusal beats discovering
        // it as an allocation failure mid-fold.
        let mut s = stack(8, 8);
        s.add("A");
        for _ in 0..MAX_GROUP_DEPTH {
            s.group_range(1, 1, "G").expect("nests");
        }
        let err = s.group_range(1, 1, "TooDeep").expect_err("bounded");
        assert!(format!("{err}").contains("nesting deeper"), "{err}");
    }

    #[test]
    fn image_editor_layers_ungrouping_reparents_its_children() {
        // Dissolving one level of wrapping must not take a level of
        // content with it.
        let mut s = stack(8, 8);
        s.add("A");
        let outer = s.group_range(1, 1, "Outer").expect("grouped");
        let inner = s.group_range(1, 1, "Inner").expect("nests");
        s.ungroup(outer).expect("dissolved");
        let g = s
            .groups()
            .iter()
            .find(|g| g.id == inner)
            .expect("inner survives");
        assert_eq!(g.parent, None, "the child rose to the outer's parent");
        assert_eq!(s.layers()[1].group, Some(inner), "and kept its members");
    }

    #[test]
    fn image_editor_layers_ungrouping_keeps_every_layer() {
        // A group is a compositing wrapper, not a container — dissolving
        // it cannot lose pixels.
        let mut s = stack(8, 8);
        s.add("A");
        s.add("B");
        let id = s.group_range(1, 2, "Set").expect("grouped");
        let before = s.layers().len();
        s.ungroup(id).expect("ungrouped");
        assert_eq!(s.layers().len(), before);
        assert!(s.layers().iter().all(|l| l.group.is_none()));
        assert!(s.groups().is_empty());
        assert!(s.ungroup(id).is_err(), "twice is an error, not a no-op");
    }

    #[test]
    fn image_editor_layers_group_properties_clamp_and_report() {
        let mut s = stack(8, 8);
        s.add("A");
        let id = s.group_range(1, 1, "Set").expect("grouped");
        s.set_group_opacity(id, 5.0).expect("set");
        assert_eq!(s.groups()[0].opacity, 1.0, "opacity is clamped");
        s.set_group_opacity(id, -1.0).expect("set");
        assert_eq!(s.groups()[0].opacity, 0.0);
        s.set_group_blend(id, "multiply").expect("set");
        assert_eq!(s.groups()[0].blend_name(), "multiply");
        assert!(s.set_group_blend(id, "nonsense").is_err());
        s.set_group_name(id, "Renamed").expect("set");
        assert_eq!(s.groups()[0].name, "Renamed");
        assert!(s.set_group_visible(999, false).is_err());
    }

    /// GROUPS ISOLATE, and this is the test that says so.
    ///
    /// An adjustment layer inside a group affects its group-MATES and
    /// nothing beneath them. The first version of this test grouped an
    /// adjustment on its own and expected the composite to change; it
    /// did not, and the code was right — an isolated group starts from a
    /// transparent accumulator, so an adjustment alone in one has
    /// nothing to adjust. Photoshop calls the non-isolating alternative
    /// "Pass Through" and makes it the default; this stack does not
    /// offer it, and says so rather than pretending Normal behaves like
    /// it.
    #[test]
    fn image_editor_layers_a_group_isolates_what_it_contains() {
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 16);
        let base = pollster::block_on(s.composite(Some(ctx), None)).expect("base");

        // A real pixel layer, then an adjustment ON it, both in a group.
        let px = s.add("Inner");
        s.layer_mut(px).expect("inner").rgba = crate::pixels::Pixels::from_rgba8(Arc::from(
            vec![80u8; 16 * 16 * 4].into_boxed_slice(),
        ))
        .into();
        let adj = s.add_adjustment("Brighten", bright(0.5));
        let id = s.group_range(px, adj, "Set").expect("grouped");
        // ISOLATED explicitly. Pass-through is the default now, and
        // under it this test's assertions hold for the wrong reason —
        // the adjustment would brighten everything below, not just its
        // group-mate, and the name would be overclaiming.
        s.set_group_pass_through(id, false).expect("isolate");

        let visible = pollster::block_on(s.composite(Some(ctx), None)).expect("visible");
        assert_ne!(
            visible.to_vec(),
            base.to_vec(),
            "the group's contents reached the composite"
        );
        assert!(
            visible[0] > 80,
            "the adjustment brightened its group-mate: got {}",
            visible[0]
        );

        s.set_group_visible(id, false).expect("hide");
        let hidden = pollster::block_on(s.composite(Some(ctx), None)).expect("hidden");
        assert_eq!(
            hidden.to_vec(),
            base.to_vec(),
            "a hidden group is exactly the identity — contents and all"
        );
    }

    /// THE PAIR THAT DEFINES THE TWO MODES, in one test because the
    /// difference is only meaningful as a comparison.
    ///
    /// An adjustment alone in a group reaches the stack below when the
    /// group is PASS-THROUGH (Photoshop's default, and this stack's) and
    /// cannot when it is ISOLATED — because isolation starts the group
    /// from a transparent accumulator, leaving the adjustment nothing to
    /// transform.
    #[test]
    fn image_editor_layers_pass_through_reaches_below_and_isolation_does_not() {
        let Some(ctx) = device() else { return };

        let mut s = stack(16, 16);
        let base = pollster::block_on(s.composite(Some(ctx), None)).expect("base");
        let at = s.add_adjustment("Brighten", bright(0.5));
        let id = s.group_range(at, at, "Set").expect("grouped");

        // DEFAULT: pass through. The adjustment brightens the background
        // beneath the group.
        assert!(
            s.groups()[0].pass_through,
            "pass-through is the default, as in Photoshop"
        );
        let through = pollster::block_on(s.composite(Some(ctx), None)).expect("through");
        assert_ne!(
            through.to_vec(),
            base.to_vec(),
            "a pass-through group's adjustment reaches the stack below"
        );

        // ISOLATED: it cannot.
        s.set_group_pass_through(id, false).expect("isolate");
        let isolated = pollster::block_on(s.composite(Some(ctx), None)).expect("isolated");
        assert_eq!(
            isolated.to_vec(),
            base.to_vec(),
            "an isolated group's adjustment has nothing beneath it to transform"
        );
    }

    #[test]
    fn image_editor_layers_opacity_below_one_forces_isolation() {
        // Photoshop's own rule, and the honest one: "fade the group" has
        // no meaning until there is a group-shaped thing to fade, so a
        // pass-through group with opacity < 1 isolates.
        let mut s = stack(8, 8);
        s.add("A");
        let id = s.group_range(1, 1, "Set").expect("grouped");
        assert!(!s.groups()[0].isolates(), "pass-through at full opacity");
        s.set_group_opacity(id, 0.5).expect("fade");
        assert!(s.groups()[0].isolates(), "fading forces isolation");
        s.set_group_opacity(id, 1.0).expect("restore");
        assert!(!s.groups()[0].isolates());
        s.set_group_pass_through(id, false).expect("isolate");
        assert!(s.groups()[0].isolates(), "and declaring it works too");
    }

    /// THE POINT OF GROUPS, and the assertion that separates one from a
    /// folder: a group at 50% fades the COMPOSITE of its members, which
    /// is a different picture from fading each member individually.
    #[test]
    fn image_editor_layers_group_opacity_fades_the_composite_not_each_member() {
        let Some(ctx) = device() else { return };

        // Two stacked opaque-ish layers, faded two different ways.
        let build = || {
            let mut s = stack(16, 16);
            let a = s.add("A");
            s.layer_mut(a).expect("a").rgba = crate::pixels::Pixels::from_rgba8(Arc::from(
                vec![255u8; 16 * 16 * 4].into_boxed_slice(),
            ))
            .into();
            let b = s.add("B");
            s.layer_mut(b).expect("b").rgba = crate::pixels::Pixels::from_rgba8(Arc::from(
                vec![0u8; 16 * 16 * 4].into_boxed_slice(),
            ))
            .into();
            // B is transparent black, so it must not hide A; give it real
            // alpha in its left half only.
            let mut px = vec![0u8; 16 * 16 * 4];
            for y in 0..16 {
                for x in 0..8 {
                    let i = (y * 16 + x) * 4;
                    px[i..i + 4].copy_from_slice(&[0, 0, 0, 255]);
                }
            }
            s.layer_mut(b).expect("b").rgba =
                crate::pixels::Pixels::from_rgba8(Arc::from(px.into_boxed_slice())).into();
            (s, a, b)
        };

        // PER-MEMBER: each layer at 50%.
        let (mut per_member, a, b) = build();
        per_member.set_opacity(a, 0.5).expect("set");
        per_member.set_opacity(b, 0.5).expect("set");
        let m = pollster::block_on(per_member.composite(Some(ctx), None)).expect("m");

        // AS A GROUP: both at full, the group at 50%.
        let (mut grouped, a2, b2) = build();
        let id = grouped.group_range(a2, b2, "Set").expect("grouped");
        grouped.set_group_opacity(id, 0.5).expect("set");
        let g = pollster::block_on(grouped.composite(Some(ctx), None)).expect("g");

        // Where B covers A, the two differ: fading members lets the
        // faded A show through the faded B, while fading the group fades
        // an already-opaque composite.
        let idx = ((8 * 16 + 4) * 4) as usize;
        assert_ne!(
            (g[idx], g[idx + 3]),
            (m[idx], m[idx + 3]),
            "group opacity must fade the COMPOSITE, not each member"
        );
    }

    /// NESTING, end to end on a device. The inner group's adjustment
    /// must reach its own members and stop at the inner boundary — which
    /// is the thing a stack of parked accumulators buys and a single
    /// parked one could not.
    #[test]
    fn image_editor_layers_a_nested_group_stops_at_its_own_boundary() {
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 16);

        // Outer group: a mid-grey plate. Inner group (above it): a
        // darker plate plus an adjustment, both isolated.
        let outer_px = s.add("OuterPlate");
        s.layer_mut(outer_px).expect("o").rgba = crate::pixels::Pixels::from_rgba8(Arc::from(
            vec![120u8; 16 * 16 * 4].into_boxed_slice(),
        ))
        .into();
        let inner_px = s.add("InnerPlate");
        s.layer_mut(inner_px).expect("i").rgba = crate::pixels::Pixels::from_rgba8(Arc::from(
            vec![60u8; 16 * 16 * 4].into_boxed_slice(),
        ))
        .into();
        let adj = s.add_adjustment("Brighten", bright(0.4));

        let outer = s.group_range(outer_px, adj, "Outer").expect("outer");
        s.set_group_pass_through(outer, false)
            .expect("isolate outer");
        let inner = s.group_range(inner_px, adj, "Inner").expect("inner");
        s.set_group_pass_through(inner, false)
            .expect("isolate inner");
        assert_eq!(
            s.groups().iter().find(|g| g.id == inner).unwrap().parent,
            Some(outer),
            "the inner group nests inside the outer one"
        );

        // It composites without panicking and produces something between
        // the two plates' tones — the inner plate brightened, over the
        // outer one. The structural assertion above is the point; this
        // is the proof the fold actually ran the nesting.
        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("nested composite");
        assert!(
            out[0] > 60,
            "the inner adjustment brightened its own plate: {}",
            out[0]
        );
    }

    #[test]
    fn image_editor_layers_a_hidden_outer_group_hides_its_nested_contents() {
        // A layer inside ANY hidden ancestor contributes nothing —
        // checking only its immediate group would leak nested content
        // out of a folder the designer closed.
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 16);
        let base = pollster::block_on(s.composite(Some(ctx), None)).expect("base");
        let px = s.add("Inner");
        s.layer_mut(px).expect("i").rgba = crate::pixels::Pixels::from_rgba8(Arc::from(
            vec![250u8; 16 * 16 * 4].into_boxed_slice(),
        ))
        .into();
        let outer = s.group_range(px, px, "Outer").expect("outer");
        s.group_range(px, px, "Inner").expect("inner");
        s.set_group_visible(outer, false)
            .expect("hide the OUTER one");
        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("out");
        assert_eq!(
            out.to_vec(),
            base.to_vec(),
            "hiding an ancestor hides everything nested inside it"
        );
    }

    // ── precision: the fold stays in f16 ─────────────────────────────

    /// THE POINT, measured. A stack of adjustment layers used to quantize
    /// to 8 bits between each one, so a smooth ramp came back with fewer
    /// distinct levels than it went in with. Banding, literally counted.
    ///
    /// Four stacked adjustments over a 256-level ramp: the f16 fold must
    /// preserve more distinct output levels than an 8-bit round trip per
    /// layer could. The assertion is a LEVEL COUNT rather than a
    /// tolerance, because that is what banding actually is.
    #[test]
    fn image_editor_layers_stacked_adjustments_do_not_band_the_backdrop() {
        let Some(ctx) = device() else { return };
        let (w, h) = (256u32, 4u32);
        // A horizontal ramp: every one of the 256 levels appears.
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for _y in 0..h {
            for x in 0..w {
                let v = x as u8;
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let mut s =
            LayerStack::from_image(w, h, Arc::from(rgba.into_boxed_slice())).expect("valid");

        // COMPRESS then EXPAND. This is the shape that exposes
        // quantization: squeezing the ramp into a fraction of the range
        // and pulling it back is lossless in f16 and irreversibly
        // destructive through 8 bits, because the compressed state has
        // fewer than 256 distinct values to round to.
        //
        // Four gentle nudges did NOT expose it — a mutation check
        // (reinstating the 8-bit round trip) still passed, because a
        // small shift on already-8-bit values collapses almost nothing.
        // The test had to measure the mechanism, not merely exercise it.
        s.add_adjustment(
            "Compress",
            AdjustParams {
                contrast: 0.1,
                ..Default::default()
            },
        );
        s.add_adjustment(
            "Expand",
            AdjustParams {
                contrast: 10.0,
                ..Default::default()
            },
        );
        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("composite");

        let mut seen = [false; 256];
        for px in out.chunks_exact(4) {
            seen[px[0] as usize] = true;
        }
        let levels = seen.iter().filter(|b| **b).count();
        // An 8-bit round trip per layer collapses neighbouring levels on
        // every pass; staying in f16 keeps them apart. The bar is set
        // well below the ideal so it measures the mechanism, not the
        // exact rounding of one GPU.
        assert!(
            levels > 120,
            "a compress/expand pair kept only {levels} distinct levels of 256 — \
             the fold is quantizing between layers"
        );
    }

    #[test]
    fn image_editor_layers_an_identity_adjustment_is_still_exactly_the_identity() {
        // The f16 path must not perturb pixels it was asked to leave
        // alone. This is the guard against the new sink or source bridge
        // quietly changing the working semantics — a transfer or
        // premultiply mistake there would show up here as drift, not as
        // a crash.
        let Some(ctx) = device() else { return };
        let mut s = stack(16, 16);
        let base = pollster::block_on(s.composite(Some(ctx), None)).expect("base");
        s.add_adjustment("Nothing", AdjustParams::default());
        let out = pollster::block_on(s.composite(Some(ctx), None)).expect("identity");
        assert_eq!(
            out.to_vec(),
            base.to_vec(),
            "an identity adjustment through the f16 path changes nothing"
        );
    }

    // ── mask painting: the edit target ───────────────────────────────

    /// A two-layer stack: grey 128 background, an opaque WHITE layer on
    /// top (active).
    fn two_layers(w: u32, h: u32) -> LayerStack {
        let mut s = stack(w, h);
        let top = s.add("Top");
        s.layers[top].rgba = crate::pixels::Pixels::from_rgba8(px(w, h, 255)).into();
        s
    }

    #[test]
    #[allow(non_snake_case)]
    fn add_mask_comes_in_reveal_all_and_hide_all_and_refuses_a_second__feat__image_editor_mask_painting(
    ) {
        let mut s = two_layers(8, 8);
        s.add_mask(1, true).expect("reveal all");
        assert!(s.layers[1].mask.as_ref().unwrap().is_all_one());
        assert!(
            s.layers[1].live_mask().is_none(),
            "a reveal-all mask is the identity, so the fold pays nothing for it"
        );
        let err = s.add_mask(1, false).expect_err("a second mask is refused");
        assert!(err.to_string().contains("already has a mask"), "{err}");
        s.clear_mask(1).expect("clear");
        s.add_mask(1, false).expect("hide all");
        assert!(s.layers[1].mask.as_ref().unwrap().canvas().is_all_zero());
    }

    #[test]
    #[allow(non_snake_case)]
    fn the_edit_target_needs_a_mask_and_follows_the_active_layer__feat__image_editor_mask_painting()
    {
        let mut s = two_layers(8, 8);
        assert!(!s.edit_target_is_mask(), "a stack starts on pixels");
        assert!(
            s.set_edit_target(1, true).is_err(),
            "no mask, no mask target — never a silent fall back to pixels"
        );
        s.add_mask(1, true).expect("mask");
        s.set_edit_target(1, true).expect("target the mask");
        assert!(s.edit_target_is_mask());
        // Re-selecting the same layer keeps the target …
        s.set_active(1).expect("same layer");
        assert!(s.edit_target_is_mask());
        // … another layer starts on its pixels.
        s.set_active(0).expect("other layer");
        assert!(!s.edit_target_is_mask());
        s.set_edit_target(1, true).expect("back to the mask");
        // Deleting the mask takes the target with it.
        s.clear_mask(1).expect("clear");
        assert!(!s.edit_target_is_mask());
    }

    #[test]
    #[allow(non_snake_case)]
    fn the_grey_plate_round_trips_a_mask_exactly__feat__image_editor_mask_painting() {
        let data: Vec<u8> = (0..64u32).map(|i| (i * 4) as u8).collect();
        let m = SelectionCoverage::from_data(8, 8, data.clone()).expect("mask");
        let grey = mask_to_grey(&m);
        assert!(grey
            .chunks_exact(4)
            .all(|p| p[0] == p[1] && p[1] == p[2] && p[3] == 255));
        assert_eq!(mask_from_grey(8, 8, &grey).expect("back").data(), &data[..]);
        assert_ne!(mask_scope(7), 7, "a mask scope never equals a pixel scope");
        assert_eq!(mask_scope(7) & !MASK_SCOPE_BIT, 7);
    }

    #[test]
    #[allow(non_snake_case)]
    fn a_mask_edit_is_one_step_on_the_one_undo_list_and_replays_into_the_mask__feat__image_editor_mask_painting(
    ) {
        let mut s = two_layers(16, 16);
        s.recorded("Add mask", None, |st| st.add_mask(1, false))
            .expect("mask");
        s.set_edit_target(1, true).expect("target");
        let pixels_before = s.layers[1].rgba.raw_arc();
        let mut painted = vec![0u8; 256];
        for y in 4..8 {
            for x in 4..8 {
                painted[y * 16 + x] = 255;
            }
        }
        let new = SelectionCoverage::from_data(16, 16, painted.clone()).unwrap();
        let out = s
            .edit_active_mask("Paint mask", Region::new(4, 4, 4, 4), new)
            .expect("edit");
        assert!(matches!(out, RecordOutcome::Recorded { .. }));
        assert_eq!(
            s.layers[1].mask.as_ref().unwrap().canvas().data(),
            &painted[..]
        );
        assert_eq!(
            s.layers[1].rgba.raw_arc(),
            pixels_before,
            "a mask edit leaves the layer's pixels alone"
        );
        assert_eq!(s.undo_labels(), vec!["Add mask", "Paint mask"]);

        // Move away first: undo must find the MASK of layer 1 regardless.
        s.set_active(0).expect("elsewhere");
        assert_eq!(s.undo().as_deref(), Some("Paint mask"));
        assert!(
            s.layers[1].mask.as_ref().unwrap().canvas().is_all_zero(),
            "undone"
        );
        assert_eq!(s.active_index(), 1, "undo lands where the edit was");
        assert!(s.edit_target_is_mask(), "… on the mask");
        assert_eq!(s.redo().as_deref(), Some("Paint mask"));
        assert_eq!(
            s.layers[1].mask.as_ref().unwrap().canvas().data(),
            &painted[..]
        );
        // Undo past it removes the mask itself (the structure step).
        s.undo();
        assert_eq!(s.undo().as_deref(), Some("Add mask"));
        assert!(!s.has_mask(1));
        assert!(!s.edit_target_is_mask());
    }

    #[test]
    #[allow(non_snake_case)]
    fn a_mask_edit_needs_a_mask_and_honours_the_lock__feat__image_editor_mask_painting() {
        let mut s = two_layers(4, 4);
        let m = || SelectionCoverage::full(4, 4);
        assert!(s
            .edit_active_mask("x", Region::new(0, 0, 4, 4), m())
            .is_err());
        s.add_mask(1, false).expect("mask");
        s.set_locked(1, true).expect("lock");
        assert!(s
            .edit_active_mask("x", Region::new(0, 0, 4, 4), m())
            .is_err());
        s.set_locked(1, false).expect("unlock");
        let wrong = SelectionCoverage::full(2, 2);
        assert!(s
            .edit_active_mask("x", Region::new(0, 0, 4, 4), wrong)
            .is_err());
        assert!(s.undo_labels().is_empty(), "no refusal spent an undo step");
    }

    #[test]
    #[allow(non_snake_case)]
    fn painting_a_hide_all_mask_reveals_the_layer_under_the_stroke__feat__image_editor_mask_painting(
    ) {
        // End to end on the device: hide-all mask on a white layer over
        // grey; an ERASER stroke on the mask target paints WHITE into the
        // mask, so the white layer shows through exactly under the
        // stroke and nowhere else. The preview composite (mask override)
        // equals the committed composite byte for byte.
        let Some(ctx) = device() else { return };
        let (w, h) = (48u32, 32u32);
        let mut s = two_layers(w, h);
        s.add_mask(1, false).expect("hide all");
        s.set_edit_target(1, true).expect("target");
        let hidden = pollster::block_on(s.composite(Some(ctx), None)).expect("composite");
        assert!(hidden.chunks_exact(4).all(|p| p[0] == 128), "fully hidden");

        let mut p = crate::stroke::StrokeParams::defaults(crate::stroke::StrokeTool::Eraser);
        p.size = 8.0;
        p.hardness = 1.0;
        let p = p.for_mask_target();
        let grey = s.active_mask_as_grey().expect("grey plate");
        let mut stroke =
            crate::stroke::StrokeSession::begin_on(1, w, h, grey, p, None).expect("begin");
        for x in [12.0f32, 18.0, 24.0] {
            pollster::block_on(stroke.extend(ctx, image_gpu::dab::StrokeSample::new(x, 16.0, 1.0)))
                .expect("extend");
        }
        let bounds = stroke.stroke_bounds().expect("painted");
        let painted = stroke.commit();
        let mask = mask_from_grey(w, h, &painted).expect("mask");
        let preview =
            pollster::block_on(s.composite_with_active_mask(Some(ctx), Arc::new(mask.clone())))
                .expect("preview");
        assert!(
            s.layers[1].mask.as_ref().unwrap().canvas().is_all_zero(),
            "the preview restored the real mask"
        );
        s.edit_active_mask("Paint mask", bounds, mask)
            .expect("commit");
        let after = pollster::block_on(s.composite(Some(ctx), None)).expect("composite");
        assert_eq!(preview.to_vec(), after.to_vec(), "preview == commit");
        let at = |x: u32, y: u32| after[((y * w + x) * 4) as usize];
        assert_eq!(at(18, 16), 255, "the stroke revealed the white layer");
        assert_eq!(at(2, 2), 128, "away from the stroke the layer stays hidden");
        assert_eq!(at(40, 28), 128);
        // One undo puts the mask — and so the composite — back.
        assert_eq!(s.undo().as_deref(), Some("Paint mask"));
        let undone = pollster::block_on(s.composite(Some(ctx), None)).expect("composite");
        assert_eq!(undone.to_vec(), hidden.to_vec());
    }

    #[test]
    #[allow(non_snake_case)]
    fn a_mask_stroke_paints_the_foregrounds_grey_and_the_eraser_paints_white__feat__image_editor_mask_painting(
    ) {
        use crate::stroke::{StrokeParams, StrokeTool};
        let mut brush = StrokeParams::defaults(StrokeTool::Brush);
        brush.color = [1.0, 0.0, 0.0, 0.5];
        let m = brush.for_mask_target();
        assert_eq!(m.tool, StrokeTool::Brush);
        assert!(
            (m.color[0] - 0.299).abs() < 1e-6,
            "red's luma: {:?}",
            m.color
        );
        assert_eq!(m.color[0], m.color[1]);
        assert_eq!(m.color[3], 1.0, "a mask paints opaque grey");
        let e = StrokeParams::defaults(StrokeTool::Eraser).for_mask_target();
        assert_eq!(e.tool, StrokeTool::Brush, "erasing a mask PAINTS white");
        assert_eq!(e.color, [1.0, 1.0, 1.0, 1.0]);
        assert!(e.tool.antialias(), "with the eraser's antialiased tip");
        let c = StrokeParams::defaults(StrokeTool::Clone).for_mask_target();
        assert_eq!(c.tool, StrokeTool::Clone, "sampling tools are unchanged");
    }
}
