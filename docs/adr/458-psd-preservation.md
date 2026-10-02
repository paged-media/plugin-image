# ADR 458 — PSD is read and written by an own parser under a preservation rule

- **Status:** Accepted. Recorded retroactively on 2026-10-02 from the code at `f7d21e5`.
- **Scope:** `image-psd`; the PSD exports of `image-js` and `image-js/src/saveback.rs`

## Context

The plugin opens PSD and PSB files, models only part of what they contain, and writes them
back. The README names "PSD/PSB round-trip as a constitutive property" (`README.md:6-7`),
and the README and the crate header put the rule in one sentence: "Paged never destroys a
PSD." `CLAUDE.md:71-75` states it as a hard rule of the repo: every block that is not
modelled is retained as opaque bytes and re-emitted verbatim, and a round trip without
edits is byte-identical.

The crate header states the purpose of each part of the mechanism. Typed nodes keep their
source bytes "so zero-edit round-trips stay byte-identical even when a producer used a
non-canonical encoding our re-encoder would normalize" (`image-psd/src/lib.rs:59-61`).
Channel pixel data stays in its compressed on-disk form until edited, for preservation and
for a streaming budget for large PSB files (`image-psd/src/lib.rs:65-67`).

`image-psd` depends on `image-core`, `thiserror` and `miniz_oxide` (for ZIP-compressed
channels) and on no PSD library. The repository does not record why.

## Decision

`image-psd` parses PSD and PSB into a typed model that keeps source bytes beside the typed
fields, and writes from that model. A file that was parsed and not edited is written back
byte for byte.

- Three storage strategies apply per node: fixed-width scalar structures are typed and
  re-encoded; every block that is not modelled is kept as opaque bytes and re-emitted
  verbatim; typed blocks also carry `Raw = Option<Vec<u8>>`, and a node whose `Raw` is
  `Some` re-emits those bytes.
- An edit sets the guard to `None` on what it touches: the layer record's `extra_raw` and
  the section's `section_raw`. The writer re-encodes that part canonically; untouched
  records are still written verbatim.
- PSD and PSB are one model; a `Container` value, set once from the header version, is passed to the parser and writer of the layer and mask
  section, where the two containers differ in the width of length fields (`image-psd/src/container.rs:72-106`, `image-psd/src/parse.rs:50-53`).
- Decoding pixels (the merged composite, per-layer plates) reads the model and does not
  change it. Structure the layer import does not model is refused with a typed error: "It
  declines rather than approximates" (`image-psd/src/layer_pixels.rs:45`).

`image-js` retains the parsed file behind a handle (`psd_open`) and exports three record
edits, the pixel save-back and `psd_save`. The bundle's PSD exporter calls `psd_save`
without the save-back when no adjustment is set.

## Evidence

- `image-psd/src/lib.rs:45-67` — the three strategies and the guard
- `image-psd/src/model/mod.rs:61-65` — `Raw`: `Some` is re-emitted, `None` is re-encoded
- `image-psd/src/edit.rs:33-50`, `:72-78` — the dirtying rule, and one edit that applies it
- `image-psd/tests/roundtrip.rs:297-317`, `:381-455` — the byte-identity tests, and the
  tests that clear every guard, write, parse again and compare
- `image-conformance/src/psd_builder/mod.rs:33-37` — fixtures come from a separate emitter
- `image-js/src/lib.rs:3643-3663`, `:3720-3734` — the retained parse and `psd_save`
- `image-js/src/saveback.rs:518-531`, `:567-575` — the save-back: in place, or flattened
- `glue/src/session.ts:1736-1749` — the exporter skips the save-back for an unadjusted file

## Alternatives considered

Writing a new file from decoded pixels is what the save-back does when it cannot attribute
adjusted pixels to one layer; it is a declared fallback (`image-js/src/saveback.rs:51-58`).
Approximating structure the import does not model was declined, as quoted above. No other
parser or library is recorded in the repository.

## Consequences

`registry/psd-blocks.yaml` tracks each block family on a ladder of five rungs (`parsed`, `preserved`, `rendered`, `mutatable`, `round-trips`); of its 20 rows, 14 are at
`preserved`, 3 at `rendered`, 2 at `parsed`, 1 at `mutatable` and none at `round-trips`. Blocks with no row "ride the opaque-verbatim path" (`registry/psd-blocks.yaml:5-6`).

Editing is narrow. `image-psd/src/edit.rs` has four operations: opacity, name, channel pixel
replacement and layer removal. Replacement writes RAW or RLE only (`image-psd/src/edit.rs:148-157`).
Saving adjusted pixels into a PSD is 8-bit RGB only (`image-js/src/saveback.rs:470-482`). It
replaces channels in place only when the file has exactly one content layer, covering the
canvas, with channels it can address; any other file becomes a new single-layer PSD and
the original layer structure is lost, which the returned description says (`:101-104`).
The merged composite is rewritten in both cases, and document-level blocks are kept.

The byte-identity tests run on files the repo builds itself. The test over real Photoshop
files is opt-in and asserts that they parse with sane headers, not that they round-trip
(`image-conformance/tests/real_psd_corpus.rs:127-129`). The check that an outside reader
(psd-tools) opens what the writer emits is also opt-in (`image-conformance/tests/psd_ecosystem.rs:62-65`).
One comment is behind the code: `image-psd/src/layer_pixels.rs:53-58` lists 16-bit files and
clipping layers as refused; the depth gate admits 16-bit (`:145-151`) and clipping is carried (`:213`).

## Related

- [ADR 461](461-clean-room-protocol.md) — the sources this crate may be derived from
- [ADR 454](454-two-evaluation-engines.md) — the layer stack a PSD's layers are imported into
- [ADR 460](460-document-is-not-the-store.md) — how saved bytes leave the plugin
