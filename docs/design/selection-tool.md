# The selection tool (over the shipped §6.1 mask)

2026-08-03. Design note: problem, design, scope. Taken from a longer internal design document that covered several plugins; only this section is reproduced. "§6.1" is section 6.1 of [concept.md](../concept.md).

*Status note (2026-10-02): this note predates the implementation. The tool is built, in a different shape: separate tools (`marqueeRect`, `marqueeEllipse`, `lasso`, `polygonal-lasso`, `magicWand`, `quickSelect` in `glue/manifest.json`) instead of one `select` tool; add, subtract and intersect on modifier keys; the feather is a Gaussian on the coverage mask on the CPU (`image-gpu/src/coverage.rs`), not the `conv.gaussian_*` kernels; and the magic wand, listed below as out of scope, exists. See `../status.md`, and [ADR 452](../adr/452-kernel-abi.md) for the mask binding.*

**Problem.** Selection-mask plumbing ships end-to-end (every pointwise dispatch
is `mix(a, result, mask)`-aware at `@group(2)` r16float) with no way to author
a mask.

**Design.** A `media.paged.image.tool.select` gesture tool + a session mask
lane:

- v1 authoring: RECTANGULAR marquee + ELLIPSE (modifier), with add (shift) /
  subtract (alt) composition and a feather radius (the existing
  `conv.gaussian_*` kernels blur the mask — no new kernel). The mask
  rasterizes CPU-side at source resolution (u8 → r16float upload), stored on
  the session per source handle.
- The adjust chain passes the session mask into every stage (the `mask`
  parameter every dispatch already takes; today it is `None`). The panel shows
  a "selection active — adjustments apply inside it" banner + a clear button;
  the composite preview renders the mask boundary as an overlay polyline
  (`setToolPreview` vocabulary suffices for rect/ellipse outlines).
- v2: polygon lasso (the pencil machine's sample+RDP lane reused verbatim on
  mask coordinates); brush = stamped gaussian discs.

**Not in scope:** color-range/magic-wand selection (needs a statistics design),
mask channels in PSD save-back (a separate PSD-door rung).
