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

//! The COW TILE JOURNAL (§8.5's `WriteBuffer` undo log) — the piece
//! Engine B carried as a documented deferral until now.
//!
//! # What it is
//!
//! A pixel WRITE is not a parameter change: `set_params`/`gesture` +
//! the provenance-keyed [`crate::NodeCache`] make a *recomputable* node
//! cheap to invalidate, but nothing recomputes a stroke. So a write has
//! to be journaled, and the journal has to be cheap enough that a
//! 4000×3000 canvas does not pay a 48 MB clone per edit.
//!
//! The unit is therefore the TILE ([`image_core::TILE`]², the same grid
//! the buffer graph and the node caches use) and the storage is
//! copy-on-write: an entry holds `Arc<[u8]>` handles to the tiles that
//! were there BEFORE the edit, snapshotted only over the DAMAGED
//! region. A stroke in one corner of a big canvas journals the two or
//! three tiles it actually touched, not the canvas.
//!
//! # Generations
//!
//! [`TileJournal::generation`] is a monotone marker of "which edit the
//! buffer is at". Recording an edit bumps it; [`TileJournal::undo`]
//! returns it to the value the undone entry recorded (its PRE-edit
//! generation), and [`TileJournal::redo`] moves it forward again. So
//! "restore by generation" is exact: the generation identifies the
//! buffer state, and a cache keyed on it (the graph's provenance rule,
//! §8.2) invalidates correctly across undo.
//!
//! # The bound, and what happens at it (this is not optional)
//!
//! Silent unbounded growth on a big canvas is not acceptable, so the
//! journal has TWO bounds — a depth ([`DEFAULT_MAX_ENTRIES`]) and a
//! byte budget ([`DEFAULT_MAX_BYTES`]) — and its behaviour at each is
//! stated rather than emergent:
//!
//! * **Over depth or over budget** → the OLDEST undo entries are
//!   evicted, oldest-first, until both bounds hold again. History is a
//!   sliding window: the edits that fell off the back are permanent,
//!   and [`TileJournal::dropped`] counts them so the UI can say so
//!   instead of silently offering a shorter undo than the user expects.
//! * **A single edit larger than the whole budget** → it is NOT
//!   recorded and the journal is CLEARED ([`RecordOutcome::TooLarge`]).
//!   Recording it would evict everything and then evict itself, so the
//!   honest answer is "this edit is not undoable, and neither is
//!   anything before it" — said once, loudly, rather than discovered
//!   later.
//! * **An edit that changed nothing** → [`RecordOutcome::NoChange`];
//!   no entry, no redo-stack clear.
//!
//! A recorded edit CLEARS the redo stack (the standard linear-history
//! rule: branching is not modeled).
//!
//! # Two stores, one journal
//!
//! [`TileStore`] is the seam. `image-graph`'s own [`crate::SourceData`]
//! implements it (so `write_source_tile` can be journaled — the §8.5
//! `WriteBuffer` Operation), and so does [`FlatImage`], a view over a
//! tightly-packed interleaved buffer, which is what the wasm surface's
//! layer pixels actually are. The journal does not care which.

use std::collections::VecDeque;
use std::sync::Arc;

use image_core::{Region, TileCoord, TILE};

/// Default undo depth. Deep enough that a working session is not
/// truncated by counting, shallow enough that the byte budget is the
/// bound that actually bites.
pub const DEFAULT_MAX_ENTRIES: usize = 32;

/// Default byte budget for the whole journal (undo + redo). 256 MiB is
/// ~5 whole-canvas edits of a 4000×3000 RGBA8 image, or hundreds of
/// ordinary strokes.
pub const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

/// The two bounds. Both are enforced on every record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JournalBudget {
    pub max_entries: usize,
    pub max_bytes: usize,
}

impl Default for JournalBudget {
    fn default() -> Self {
        JournalBudget {
            max_entries: DEFAULT_MAX_ENTRIES,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

/// A buffer the journal can snapshot tiles OUT of. Tiles are addressed
/// at LEVEL 0 (the journal does not journal mip levels — they are
/// derived). Recording needs only this half, which is why it is split
/// off: a layer's pixels are shared behind an `Arc` and are read, never
/// written, at record time.
pub trait TileSource {
    /// The tile's bytes, or `None` for a tile the store does not hold
    /// (the sparse-canvas case; the journal then records a hole and
    /// restores one).
    fn read_tile(&self, coord: TileCoord) -> Option<Arc<[u8]>>;
}

/// A [`TileSource`] the journal can also write tiles back INTO —
/// what undo/redo needs.
pub trait TileStore: TileSource {
    /// Put `bytes` back at `coord`. `None` restores the hole.
    fn write_tile(&mut self, coord: TileCoord, bytes: Option<Arc<[u8]>>);
}

/// One tile's pre-edit state.
#[derive(Debug, Clone)]
struct Snapshot {
    coord: TileCoord,
    /// `None` = the store held no tile there.
    bytes: Option<Arc<[u8]>>,
}

/// One journaled edit.
#[derive(Debug, Clone)]
struct Entry {
    label: String,
    /// WHICH buffer this edit belongs to — an opaque caller id (a layer
    /// id, a node id). One journal can serve several buffers, and undo
    /// MUST restore into the one the edit came from, not into whichever
    /// is selected when the user reaches for it.
    scope: u64,
    /// The generation the buffer returns to when this entry is undone.
    generation: u64,
    tiles: Vec<Snapshot>,
    bytes: usize,
}

/// What [`TileJournal::record`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordOutcome {
    /// Journaled: `tiles` tiles, `bytes` bytes.
    Recorded { tiles: usize, bytes: usize },
    /// The damaged region held no tiles at all — nothing to journal.
    NoChange,
    /// The snapshot alone exceeds the byte budget: nothing was
    /// recorded and the journal was CLEARED (see the module docs).
    TooLarge { bytes: usize, budget: usize },
}

impl RecordOutcome {
    pub fn is_recorded(self) -> bool {
        matches!(self, RecordOutcome::Recorded { .. })
    }
}

/// A bounded, COW, tile-granular undo/redo log over a [`TileStore`].
#[derive(Debug)]
pub struct TileJournal {
    budget: JournalBudget,
    undo: VecDeque<Entry>,
    redo: Vec<Entry>,
    bytes: usize,
    generation: u64,
    dropped: u64,
}

impl Default for TileJournal {
    fn default() -> Self {
        TileJournal::new()
    }
}

impl TileJournal {
    pub fn new() -> Self {
        TileJournal::with_budget(JournalBudget::default())
    }

    pub fn with_budget(budget: JournalBudget) -> Self {
        TileJournal {
            budget,
            undo: VecDeque::new(),
            redo: Vec::new(),
            bytes: 0,
            generation: 0,
            dropped: 0,
        }
    }

    pub fn budget(&self) -> JournalBudget {
        self.budget
    }

    /// The buffer's current edit marker (see the module docs).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn depth(&self) -> usize {
        self.undo.len()
    }

    pub fn redo_depth(&self) -> usize {
        self.redo.len()
    }

    /// Bytes held by undo + redo together.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Entries evicted by the bounds so far — the count the UI needs to
    /// say "history is a window", never a silent shortening.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// The label of the edit `undo` would revert, if any.
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.back().map(|e| e.label.as_str())
    }

    /// The label of the edit `redo` would replay, if any.
    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(|e| e.label.as_str())
    }

    /// WHICH buffer the next `undo` belongs to — read this BEFORE
    /// calling `undo`, and hand it the matching store. Applying an
    /// entry to the wrong buffer would write one layer's history into
    /// another's pixels, so the scope is not advisory.
    pub fn undo_scope(&self) -> Option<u64> {
        self.undo.back().map(|e| e.scope)
    }

    /// WHICH buffer the next `redo` belongs to.
    pub fn redo_scope(&self) -> Option<u64> {
        self.redo.last().map(|e| e.scope)
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.bytes = 0;
    }

    /// Snapshot the tiles covering `damage` out of `store` — call this
    /// BEFORE the edit writes. The snapshot is COW: each tile is an
    /// `Arc` clone, so journaling costs a pointer per tile, not a copy.
    pub fn record<S: TileSource + ?Sized>(
        &mut self,
        label: impl Into<String>,
        scope: u64,
        store: &S,
        damage: Region,
    ) -> RecordOutcome {
        let mut tiles = Vec::new();
        let mut bytes = 0usize;
        for coord in damage.tiles_at(0) {
            let b = store.read_tile(coord);
            bytes += b.as_ref().map_or(0, |x| x.len());
            tiles.push(Snapshot { coord, bytes: b });
        }
        if tiles.is_empty() {
            return RecordOutcome::NoChange;
        }
        if bytes > self.budget.max_bytes {
            // Recording would evict everything and then itself — say so
            // once, and leave no history that pretends otherwise.
            self.clear();
            return RecordOutcome::TooLarge {
                bytes,
                budget: self.budget.max_bytes,
            };
        }
        let n = tiles.len();
        // A new edit ends the redo branch (linear history).
        self.bytes -= self.redo.iter().map(|e| e.bytes).sum::<usize>();
        self.redo.clear();

        let entry = Entry {
            label: label.into(),
            scope,
            generation: self.generation,
            tiles,
            bytes,
        };
        self.generation += 1;
        self.bytes += bytes;
        self.undo.push_back(entry);
        self.enforce_bounds();
        RecordOutcome::Recorded { tiles: n, bytes }
    }

    /// Revert the newest recorded edit, writing its pre-edit tiles back
    /// into `store` and pushing the tiles it replaced onto the redo
    /// stack. Returns the reverted edit's label.
    pub fn undo<S: TileStore>(&mut self, store: &mut S) -> Option<String> {
        let entry = self.undo.pop_back()?;
        self.bytes -= entry.bytes;
        // The inverse must carry the generation a REDO lands on — the
        // one we are leaving — while we drop back to the entry's own
        // pre-edit marker.
        let inverse = self.apply(store, &entry, self.generation);
        let label = entry.label.clone();
        self.generation = entry.generation;
        self.bytes += inverse.bytes;
        self.redo.push(inverse);
        // The inverse can be bigger than what it replaced (a hole
        // restores as real bytes), so the budget is re-checked here too.
        self.enforce_bounds();
        Some(label)
    }

    /// Replay the newest undone edit.
    pub fn redo<S: TileStore>(&mut self, store: &mut S) -> Option<String> {
        let entry = self.redo.pop()?;
        self.bytes -= entry.bytes;
        let target = entry.generation;
        let inverse = self.apply(store, &entry, self.generation);
        let label = entry.label.clone();
        self.generation = target;
        self.bytes += inverse.bytes;
        self.undo.push_back(inverse);
        self.enforce_bounds();
        Some(label)
    }

    /// Write `entry`'s tiles into `store`, returning the entry that
    /// restores what was there (the inverse).
    fn apply<S: TileStore>(&self, store: &mut S, entry: &Entry, generation: u64) -> Entry {
        let mut tiles = Vec::with_capacity(entry.tiles.len());
        let mut bytes = 0usize;
        for snap in &entry.tiles {
            let current = store.read_tile(snap.coord);
            bytes += current.as_ref().map_or(0, |b| b.len());
            store.write_tile(snap.coord, snap.bytes.clone());
            tiles.push(Snapshot {
                coord: snap.coord,
                bytes: current,
            });
        }
        Entry {
            label: entry.label.clone(),
            scope: entry.scope,
            generation,
            tiles,
            bytes,
        }
    }

    /// Evict oldest-first until both bounds hold.
    fn enforce_bounds(&mut self) {
        while self.undo.len() > self.budget.max_entries
            || (self.bytes > self.budget.max_bytes && !self.undo.is_empty())
        {
            let Some(old) = self.undo.pop_front() else {
                break;
            };
            self.bytes -= old.bytes;
            self.dropped += 1;
        }
        // If the redo stack alone still busts the budget, it goes: a
        // redo the user cannot reach is not worth the memory.
        if self.bytes > self.budget.max_bytes {
            self.bytes -= self.redo.iter().map(|e| e.bytes).sum::<usize>();
            self.redo.clear();
        }
    }
}

/// A [`TileStore`] view over a tightly packed, interleaved image buffer
/// (`width · height · bytes_per_texel`, row-major) — what the wasm
/// surface's layer pixels are.
///
/// Tiles are the level-0 [`TILE`] grid clipped to the image extent, so
/// an edge tile's snapshot is exactly its clipped rect (no padding is
/// stored and none is needed: the same clip is applied on write).
/// `B` is the byte container: `&[u8]` gives a read-only view (enough to
/// RECORD), `&mut [u8]` a writable one (what undo/redo needs).
pub struct FlatImage<B> {
    width: u32,
    height: u32,
    bytes_per_texel: usize,
    bytes: B,
}

impl<B: AsRef<[u8]>> FlatImage<B> {
    /// `bytes` must be exactly `width · height · bytes_per_texel`;
    /// anything else is a programmer error and answers `None`.
    pub fn new(width: u32, height: u32, bytes_per_texel: usize, bytes: B) -> Option<FlatImage<B>> {
        let want = (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(bytes_per_texel)?;
        if bytes.as_ref().len() != want {
            return None;
        }
        Some(FlatImage {
            width,
            height,
            bytes_per_texel,
            bytes,
        })
    }

    /// The tile's clipped rect in image space, or `None` when the tile
    /// lies entirely outside.
    fn rect(&self, coord: TileCoord) -> Option<Region> {
        if coord.level != 0 {
            return None;
        }
        let t = TILE as i64;
        let x = coord.x as i64 * t;
        let y = coord.y as i64 * t;
        Region::new(i32::try_from(x).ok()?, i32::try_from(y).ok()?, TILE, TILE)
            .intersect(Region::new(0, 0, self.width, self.height))
    }
}

impl<B: AsRef<[u8]>> TileSource for FlatImage<B> {
    fn read_tile(&self, coord: TileCoord) -> Option<Arc<[u8]>> {
        let r = self.rect(coord)?;
        let bpt = self.bytes_per_texel;
        let src = self.bytes.as_ref();
        let mut out = Vec::with_capacity(r.w as usize * r.h as usize * bpt);
        for row in 0..r.h {
            let y = r.y as usize + row as usize;
            let start = (y * self.width as usize + r.x as usize) * bpt;
            out.extend_from_slice(&src[start..start + r.w as usize * bpt]);
        }
        Some(Arc::from(out.into_boxed_slice()))
    }
}

impl<B: AsRef<[u8]> + AsMut<[u8]>> TileStore for FlatImage<B> {
    fn write_tile(&mut self, coord: TileCoord, bytes: Option<Arc<[u8]>>) {
        let Some(r) = self.rect(coord) else { return };
        let Some(src) = bytes else { return };
        let bpt = self.bytes_per_texel;
        let width = self.width as usize;
        let row_bytes = r.w as usize * bpt;
        if src.len() != row_bytes * r.h as usize {
            return;
        }
        let dst = self.bytes.as_mut();
        for row in 0..r.h {
            let y = r.y as usize + row as usize;
            let start = (y * width + r.x as usize) * bpt;
            let s = row as usize * row_bytes;
            dst[start..start + row_bytes].copy_from_slice(&src[s..s + row_bytes]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BPT: usize = 4;

    fn image(w: u32, h: u32, fill: u8) -> Vec<u8> {
        vec![fill; (w as usize) * (h as usize) * BPT]
    }

    fn store(w: u32, h: u32, bytes: &mut [u8]) -> FlatImage<&mut [u8]> {
        FlatImage::new(w, h, BPT, bytes).expect("well-sized")
    }

    /// Paint `value` into `damage` of a flat buffer.
    fn paint(w: u32, bytes: &mut [u8], damage: Region, value: u8) {
        for y in 0..damage.h {
            for x in 0..damage.w {
                let i = (((damage.y as u32 + y) * w + damage.x as u32 + x) as usize) * BPT;
                bytes[i..i + BPT].fill(value);
            }
        }
    }

    #[test]
    fn image_editor_undo_a_small_edit_journals_only_the_tiles_it_touched() {
        // 1024×512 = 4×2 tiles; a 10×10 edit in the top-left touches ONE.
        let (w, h) = (1024u32, 512u32);
        let mut px = image(w, h, 10);
        let mut j = TileJournal::new();
        let damage = Region::new(4, 4, 10, 10);
        let out = j.record("Paint", 0, &store(w, h, &mut px), damage);
        assert_eq!(
            out,
            RecordOutcome::Recorded {
                tiles: 1,
                bytes: (TILE * TILE) as usize * BPT
            },
            "one 256² tile, not the 2 MB canvas"
        );
        assert!(j.bytes() < (w * h) as usize * BPT / 2);
    }

    #[test]
    fn image_editor_undo_restores_the_pixels_exactly_and_redo_replays_them() {
        let (w, h) = (300u32, 200u32);
        let before = image(w, h, 10);
        let mut px = before.clone();
        let mut j = TileJournal::new();
        let damage = Region::new(0, 0, 64, 64);
        assert!(j
            .record("Paint", 0, &store(w, h, &mut px), damage)
            .is_recorded());
        paint(w, &mut px, damage, 200);
        let after = px.clone();
        assert_ne!(after, before);

        assert_eq!(j.undo(&mut store(w, h, &mut px)).as_deref(), Some("Paint"));
        assert_eq!(px, before, "undo restores byte-for-byte");
        assert_eq!(j.redo(&mut store(w, h, &mut px)).as_deref(), Some("Paint"));
        assert_eq!(px, after, "redo replays byte-for-byte");
    }

    #[test]
    fn image_editor_undo_walks_a_stack_and_the_generation_tracks_it() {
        let (w, h) = (64u32, 64u32);
        let mut px = image(w, h, 0);
        let mut j = TileJournal::new();
        assert_eq!(j.generation(), 0);
        let steps = [(Region::new(0, 0, 8, 8), 1u8), (Region::new(8, 0, 8, 8), 2)];
        let mut snapshots = vec![px.clone()];
        for (damage, v) in steps {
            j.record("edit", 0, &store(w, h, &mut px), damage);
            paint(w, &mut px, damage, v);
            snapshots.push(px.clone());
        }
        assert_eq!(j.generation(), 2);
        assert_eq!(j.depth(), 2);

        j.undo(&mut store(w, h, &mut px));
        assert_eq!(j.generation(), 1);
        assert_eq!(px, snapshots[1]);
        j.undo(&mut store(w, h, &mut px));
        assert_eq!(j.generation(), 0);
        assert_eq!(px, snapshots[0]);
        assert!(!j.can_undo());
        assert!(j.undo(&mut store(w, h, &mut px)).is_none());
        // …and back up again.
        j.redo(&mut store(w, h, &mut px));
        j.redo(&mut store(w, h, &mut px));
        assert_eq!(px, snapshots[2]);
        assert_eq!(j.generation(), 2);
    }

    #[test]
    fn image_editor_undo_a_new_edit_ends_the_redo_branch() {
        let (w, h) = (64u32, 64u32);
        let mut px = image(w, h, 0);
        let mut j = TileJournal::new();
        j.record("a", 0, &store(w, h, &mut px), Region::new(0, 0, 8, 8));
        paint(w, &mut px, Region::new(0, 0, 8, 8), 1);
        j.undo(&mut store(w, h, &mut px));
        assert!(j.can_redo());
        j.record("b", 0, &store(w, h, &mut px), Region::new(0, 0, 8, 8));
        assert!(!j.can_redo(), "linear history: branching is not modeled");
        // The 64×64 canvas is ONE clipped tile, so the bytes held are
        // exactly one clipped tile per entry (no padding is stored).
        assert_eq!(j.bytes(), j.depth() * (w * h) as usize * BPT);
    }

    #[test]
    fn image_editor_undo_evicts_the_oldest_entries_at_the_depth_bound() {
        let (w, h) = (64u32, 64u32);
        let mut px = image(w, h, 0);
        let mut j = TileJournal::with_budget(JournalBudget {
            max_entries: 3,
            max_bytes: DEFAULT_MAX_BYTES,
        });
        for i in 0..5u8 {
            let d = Region::new(0, 0, 8, 8);
            j.record(format!("edit {i}"), 0, &store(w, h, &mut px), d);
            paint(w, &mut px, d, i + 1);
        }
        assert_eq!(j.depth(), 3, "a sliding window, not unbounded growth");
        assert_eq!(j.dropped(), 2, "and it SAYS how many fell off the back");
        assert_eq!(j.undo_label(), Some("edit 4"));
    }

    #[test]
    fn image_editor_undo_evicts_the_oldest_entries_at_the_byte_bound() {
        let (w, h) = (512u32, 512u32); // 2×2 tiles = 1 MiB per whole-canvas edit
        let tile_bytes = (TILE * TILE) as usize * BPT;
        let mut px = image(w, h, 0);
        let mut j = TileJournal::with_budget(JournalBudget {
            max_entries: 100,
            max_bytes: tile_bytes * 9, // ~2 whole-canvas edits (4 tiles each)
        });
        for i in 0..5u8 {
            let d = Region::new(0, 0, w, h);
            j.record(format!("edit {i}"), 0, &store(w, h, &mut px), d);
            paint(w, &mut px, d, i + 1);
        }
        assert!(j.bytes() <= tile_bytes * 9);
        assert_eq!(j.depth(), 2);
        assert_eq!(j.dropped(), 3);
    }

    #[test]
    fn image_editor_undo_an_edit_over_the_whole_budget_clears_and_says_so() {
        let (w, h) = (512u32, 512u32);
        let mut px = image(w, h, 0);
        let mut j = TileJournal::with_budget(JournalBudget {
            max_entries: 8,
            max_bytes: 1024, // smaller than a single tile
        });
        j.record("small", 0, &store(w, h, &mut px), Region::new(0, 0, 8, 8));
        assert!(!j.can_undo(), "even the first edit busts this budget");
        let out = j.record("whole", 0, &store(w, h, &mut px), Region::new(0, 0, w, h));
        assert!(matches!(out, RecordOutcome::TooLarge { budget: 1024, .. }));
        assert_eq!(j.depth(), 0);
        assert_eq!(j.bytes(), 0, "the journal is CLEARED, never half-true");
    }

    #[test]
    fn image_editor_undo_an_empty_damage_region_records_nothing() {
        let (w, h) = (64u32, 64u32);
        let mut px = image(w, h, 0);
        let mut j = TileJournal::new();
        assert_eq!(
            j.record("nothing", 0, &store(w, h, &mut px), Region::new(0, 0, 0, 0)),
            RecordOutcome::NoChange
        );
        assert!(!j.can_undo());
    }

    #[test]
    fn image_editor_undo_edge_tiles_snapshot_their_clipped_rect_only() {
        // 300×200 with TILE 256: the right/bottom tiles are clipped, so
        // their snapshots are 44 and 200 wide, not 256 (no padding is
        // stored, and the restore uses the same clip).
        let (w, h) = (300u32, 200u32);
        let mut px = image(w, h, 7);
        let s = store(w, h, &mut px);
        let full = s.read_tile(TileCoord {
            level: 0,
            x: 0,
            y: 0,
        });
        assert_eq!(full.unwrap().len(), 256 * 200 * BPT);
        let edge = s.read_tile(TileCoord {
            level: 0,
            x: 1,
            y: 0,
        });
        assert_eq!(edge.unwrap().len(), 44 * 200 * BPT);
        assert!(s
            .read_tile(TileCoord {
                level: 0,
                x: 2,
                y: 0
            })
            .is_none());
    }

    #[test]
    fn image_editor_undo_an_entry_names_the_buffer_it_belongs_to() {
        // One journal, TWO buffers. Undo must restore into the buffer
        // the edit came from, not into whichever the caller happens to
        // hold — so the scope is readable BEFORE the apply.
        let (w, h) = (32u32, 32u32);
        let mut a = image(w, h, 1);
        let mut b = image(w, h, 2);
        let mut j = TileJournal::new();
        let d = Region::new(0, 0, 8, 8);
        j.record("edit A", 7, &store(w, h, &mut a), d);
        paint(w, &mut a, d, 100);
        j.record("edit B", 9, &store(w, h, &mut b), d);
        paint(w, &mut b, d, 200);

        assert_eq!(j.undo_scope(), Some(9), "the newest edit was B's");
        j.undo(&mut store(w, h, &mut b));
        assert_eq!(b, image(w, h, 2), "B is restored…");
        assert_ne!(a, image(w, h, 1), "…and A is untouched");
        assert_eq!(j.redo_scope(), Some(9));
        assert_eq!(j.undo_scope(), Some(7), "next comes A's");
        j.undo(&mut store(w, h, &mut a));
        assert_eq!(a, image(w, h, 1));
    }

    #[test]
    fn image_editor_undo_a_flat_view_rejects_a_mis_sized_buffer() {
        let mut px = vec![0u8; 10];
        assert!(FlatImage::new(4, 4, BPT, px.as_mut_slice()).is_none());
    }
}
