# ADR 462 — Raster sessions persist in container parts and bake into the frame

- **Status:** Accepted, 2026-10-04. Supersedes [ADR 460](460-document-is-not-the-store.md).
- **Scope:** `glue/src/session.ts`, `glue/src/session-store.ts`, `image-js/src/layers_persist.rs`, `image-js/src/saveback.rs`

## Context

ADR 460 kept every edit in the session: layers, masks and adjustment layers lived in the
wasm module's memory and left only through save-back to a file. Closing the document lost
them, and the page kept showing the original placed image unless a preview layer was up.

Three host doors made a different decision possible: the per-plugin `.paged` container
parts door (`host.parts`), the `replaceImageBytes` mutation (one undoable step that replaces
a frame's placed image with inline bytes) and plugin metadata on the frame.

## Decision

A session is COMMITTED to the document in two steps, in this order:

1. The layer stack is written into the plugin's parts: a manifest and the buffers it refers
   to, each stored once under its SHA-256 (`px/<sha>.bin`), and one record per revision
   (`f/<frame>/r<n>.json`) naming the manifest, the buffers and the hash of the baked image.
2. One batch mutation replaces the frame's placed image with a PNG of the stack's composite
   and sets the frame's marker to `{v: 2, data: {owns: "pixels", rev, record, baked}}`.

Reopening a frame restores the recorded layers when, and only when, the frame's placed bytes
hash to the marker's `baked`; otherwise the image was changed outside the plugin and opens as
one layer, with the reason stated.

## Evidence

- `glue/src/session.ts:2363-2438` — `commitToDocument`: parts first, then the batch
- `glue/src/session.ts:2409` — the frame's image is replaced in the same batch as the marker
- `glue/src/session.ts:982-1013` — reopen: the hash check, then `layersImport`
- `glue/src/session-store.ts:56-85` — content-addressed buffers and the revision record
- `glue/src/session-store.ts:90-119` — every buffer verified against its hash on read
- `image-js/src/layers_persist.rs:134-200` — the stack as a manifest plus buffers
- `image-js/src/layers_persist.rs:204-337` — the inverse, refusing damaged input
- `glue/src/session.ts` (`MAX_BAKED_BYTES`) — the bake is capped at 8 MB only on the JSON lane
- `glue/src/host66.ts` — the binary mutation lane, part deletion and the will-save hook, each probed

## Alternatives considered

Keeping ADR 460 and offering only file save-back: rejected because a document that loses
its layers on close is not an editor. Storing the layers as a layered PSD part: the PSD
writer refuses adjustment layers (`image-js/src/saveback.rs:985`), and the plugin's own
manifest round-trips every layer kind byte for byte.

## Consequences

The frame's placed image is a normal image after a commit: the page renders, prints and
exports without this plugin, and an IDML export carries the baked pixels, not the layers.
Parts are never overwritten, so the document's undo can only land on a marker whose revision
exists. Unchanged layers are stored once across revisions.

A commit happens on the "Commit image edits to the document" command, when the image's
edit context is left or committed, when a frame with uncommitted edits is deselected, and,
on a host that announces saves (`document.onWillSave@1`), before the document is saved.

Where the host offers the protocol-66 doors (`glue/src/host66.ts`, each feature-detected):
the composite PNG crosses as bytes (`document.mutateBinary@1`), so it has no size cap, and
after each commit the frame's revisions older than the last eight, and the layer data only
they named, are deleted (`storage.parts@2`; `glue/src/session-store.ts`, `collectGarbage`).
The collection is awaited inside the commit, so it never overlaps the next commit's writes,
which store buffers before the record that names them. On an older host the bake is still
capped at 8 MB and nothing is deleted.

## Related

- [ADR 460](460-document-is-not-the-store.md) — superseded
- [ADR 459](459-scene-layer-image-and-tiles.md) — the in-frame preview layer, unchanged
- [ADR 463](463-one-undo-list.md) — what undo covers before and after a commit
- [ADR 458](458-psd-preservation.md) — the layered PSD save sits beside the preservation writer
