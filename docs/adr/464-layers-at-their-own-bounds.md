# ADR 464 — Layers live at their own bounds

- **Status:** Proposed, 2026-10-05.
- **Scope:** `image-js/src/pixels.rs`, `image-js/src/layers.rs` and `layers/fold.rs` (the stack and
  its fold), `image-js/src/layers_persist.rs`, `image-psd/src/layer_pixels.rs` (the import plates)

## Context

Every layer is canvas-sized: its pixels, its mask, and the GPU plate the fold caches for it. A
layered PSD import therefore costs `layers × width × height × 4` bytes of wasm memory before
anything is drawn, and twice that per layer on the GPU (the fold's plates are RGBA16F). The
import refuses past 384 MiB rather than exhaust the heap.

Mock-ups are exactly the files that hit it. On the private corpus, 35 PSDs have no structural
blocker left and are refused only by the budget: 11–26 layers at 3500×2300 to 4500×3000 need
429–1338 MiB canvas-sized. The same layers at their own bounds need 122–272 MiB (plus 5–74 MiB
of masks). Photoshop itself stores each layer at its bounds.

## Decision

A layer's pixels are stored at their BOUNDS — a rectangle of the canvas plus the pixels inside
it; everything outside is transparent. A canvas-sized layer is the special case whose bounds are
the canvas.

- The import keeps each plate at the layer record's rectangle (clipped to the canvas) instead of
  expanding it.
- The fold composites a bounded plate through a WINDOW: the accumulator's rectangle is copied
  out, the plate (uploaded at its own size) blended into it, and the result copied back — the
  same copy and dispatch primitives the stroke preview's windowed fold already uses, so no
  kernel or ABI changes. Steps that need a canvas-sized operand (a clip base's alpha) expand it
  for that step only.
- Any EDIT of a layer's pixels (paint, fill, filter, transform) expands that layer to the canvas
  first, once, so every editing path keeps its canvas-sized contract. Memory grows by the layers
  a user actually edits.
- Persistence stores the bounds with the pixels (layer manifest version 3), so saving does not
  expand every layer.
- The type makes the two shapes explicit, so no code can read bounded bytes as a canvas by
  accident: the compiler lists every place that touched layer pixels.

## Consequences

The import budget counts bounded bytes, which lets the 35 mock-ups open as layers (and later the
CMYK mock-ups behind vector masks). GPU memory per untouched layer drops to its bounds × 8 bytes.
The windowed blend adds two texture copies per bounded layer per full composite; the fold's
checkpoint cache keeps the steady-state cost (one layer edited) where it is.
