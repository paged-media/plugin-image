# ADR 454 — Two evaluation engines over one kernel set

- **Status:** Accepted. Recorded retroactively on 2026-10-02 from the code at `f7d21e5`.
- **Scope:** `image-pipeline` (Engine A), `image-graph` (Engine B), the layer stack in `image-js/src/layers.rs`, and the tile types in `image-core`

## Context

The README describes the engine as two shapes: a demand-driven streaming pipeline
(Engine A) and a persistent tiled buffer graph (Engine B) (`README.md:3-7`,
`CLAUDE.md:9-11`). Both appear in the first two days of the commit history, on the kernel
definitions of [ADR 451](451-one-kernel-definition.md) and the dispatch of `image-gpu`.
It gives no argument for building both. The repository does not record why.

Commit `cc67861` (2026-08-04) added a layer stack in `image-js`, so that a stroke lands in
a layer and can be undone. Painting, fills and filters write into it today.

## Decision

Two evaluation models share the `image-core` types, the kernel set and the GPU dispatch.

- **Engine A** (`Pipeline`): a list of source and apply nodes. Nothing runs until a sink
  (`to_buffer`, `to_encoder` and their async twins) pulls a region. Each node covers the
  region with 256² tiles, asks its input for the region the kernel class requires,
  dispatches tile by tile, and stores its result in a cache keyed by operation, parameter
  hash and input content hash.
- **Engine B** (`BufferGraph`): persistent source and op nodes over tiles addressed by
  `(level, x, y)`. Each op node caches output tiles with the parameter hash and input tile
  generations they were computed from. `set_params` and `gesture` change parameters;
  `request` recomputes only tiles whose record no longer matches. `TileJournal` snapshots
  the tiles a write will damage, for undo.
- **The layer stack** is a third structure, outside both graphs: an ordered list of
  canvas-sized layers (pixels, adjustment, smart object) with masks, clipping and groups.
  The composite folds bottom-up into a premultiplied `rgba16float` accumulator, one
  `compose.*` dispatch per pixel layer over the whole canvas. One plain layer is returned
  as it is, without a device. The result is written back into the image handle the stack
  is bound to. Undo is a `TileJournal` scoped per layer, bounded at 32 entries and 256 MiB.

On the wasm surface Engine A runs the adjustment chain, for the panel and for adjustment
layers. Of Engine B the surface uses the journal, and a source-only mip pyramid behind an
export the bundle does not call. The single-kernel doors, fills, strokes and the layer
fold call the `image-gpu` dispatch functions directly, through neither engine.

## Evidence

- `image-pipeline/src/lib.rs:33-36`, `image-pipeline/src/schedule.rs:41-49`, `:270-321` —
  Engine A: the lazy pull, tiling, the cache key, the per-tile dispatch
- `image-graph/src/lib.rs:38-69`, `image-graph/src/eval.rs:168-283` — Engine B: generations,
  the parameter paths, the journal; window gathering and dispatch
- `image-core/src/tile.rs:41-52`, `image-graph/src/journal.rs:98-106` — tiles; journal bounds
- `image-js/src/ingest.rs:876`, `:1121` — the two places production code builds a `Pipeline`
- `image-js/src/layers.rs:41-124`, `:1389-1586` — the layer model and the fold
- `image-js/src/lib.rs:2463-2473`, `:2501-2519` — the composite replaces the bound image
- `image-js/src/mip.rs:57-73`, `glue/test/wasm-surface.spec.ts:50-63` — the one
  `BufferGraph` in the surface, and its export without a caller

## Alternatives considered

No single-engine alternative is recorded. For the layer model,
`image-js/src/layers.rs:45-50` names per-layer bounds and declines them: canvas-sized
layers make the composite a fold with no offset arithmetic and let the brush paint
anywhere, at the cost of memory per layer. Journalling layer structure is declined at
`image-js/src/layers.rs:115-119`: those operations are cheap to reverse by hand, and
logging them would mean holding a removed layer's whole canvas.
`image-js/src/stroke.rs:318-329` weighs Engine B's parameter split for strokes and
concludes that paint is a write, so only the journal was needed.

## Consequences

Both engines read a kernel's class; neither reads `mip_exact`. Memory grows with layers
times canvas; the PSD layer import is capped at 384 MiB for that reason
(`image-psd/src/layer_pixels.rs:61-64`, `:79-82`). Layer structure is not undoable, and
removing a layer clears the journal (`image-js/src/layers.rs:819-824`).

Engine B's graph evaluation is not reached by the plugin: `add_op` and `request` have no caller outside `image-graph` and the tests. Engine
A is built anew for each adjust call, so its cache does not persist between calls. Engine A's dispatch does not branch on the kernel class:
for a `Windowed` kernel it requests the expanded region upstream (`image-pipeline/src/schedule.rs:100`), then passes the upstream tile for
the same coordinate to `execute_tile_once` with the output tile's width and height (`image-pipeline/src/schedule.rs:270-321`). Engine B
gathers an expanded window and uses `execute_windowed_once` (`image-graph/src/eval.rs:178-181`, `:258-263`). The texture pool, the
residency manager and batched dispatch in `image-gpu` are used only by tests.

Comments are stale in three places. `image-graph/Cargo.toml:8` describes types and stubs.
`image-pipeline/src/lib.rs:38-43` lists the encoder sink as future. The header of
`image-js/src/layers.rs:45-46` says layers are RGBA8; the field is a `Pixels` value of 8 or
16 bits ([ADR 455](455-pixel-model.md)).

## Related

- [ADR 450](450-gpu-only-kernels.md), [ADR 452](452-kernel-abi.md) — what both engines dispatch, and through which interface
- [ADR 455](455-pixel-model.md) — tile and layer pixel formats
- [ADR 459](459-scene-layer-image-and-tiles.md), [ADR 460](460-document-is-not-the-store.md) — how the composite reaches the page; undo as the plugin's own journal
