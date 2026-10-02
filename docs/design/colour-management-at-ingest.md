# CMS ingest wiring

2026-08-03. Design note: problem, design, scope. Taken from a longer internal design document that covered several plugins; only this section is reproduced.

*Status note (2026-10-02): this note predates the implementation. Rung 1 is built for PNG and JPEG: `image-js/src/display.rs` transforms RGB pixels with an embedded profile into working sRGB through qcms at decode, the panel states which treatment was applied, and a CMYK JPEG is cast on ingest instead of rejected. A PSD's profile is not read yet and is reported as "sRGB assumed". There is no session flag; the transform runs whenever an embedded profile compiles. Rung 2 (print and soft-proof) is not built. See [ADR 457](../adr/457-colour-management.md).*

**Problem.** The CMS lanes are built (qcms display, moxcms print, CMYK cast on
main) but `decode_rgba8` maps U8 verbatim (straight encoded
values).

**Design.** Wire at DECODE, keep the kernels untouched (their math was always
specified over the post-CMS working space):

- `decode_rgba8` grows an optional `DisplayTransform` (source profile from the
  codec metadata — EXIF/ICC already read — else sRGB assumed) → qcms into the
  working sRGB. Behind a session flag defaulting ON, with the panel stating
  which profile was honored ("sRGB assumed" is an honest state, not a silent
  one).
- CMYK JPEG: the moxcms cast (already reachable) becomes the default ingest
  path rather than a rejection.
- Print/proof (moxcms per-intent + soft-proof UI) is rung 2 — it needs the
  host's proof-setup state (`setProofSetup` exists on the wire) piped to the
  plugin, a request to the plugin platform when rung 1 lands.
- Conformance: decode fixtures with embedded ICC vs the qcms oracle; the
  straight-values path stays available for the kernels' existing tolerance
  ladders (tests pin the transform OFF).
