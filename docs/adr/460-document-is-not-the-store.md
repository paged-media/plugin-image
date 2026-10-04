# ADR 460 — The document is not the store: edits live in the session and leave through save-back

- **Status:** Accepted. Recorded retroactively on 2026-10-02 from the code at `f7d21e5`.
- **Scope:** `glue/src/session.ts`, `glue/src/activate.ts`, the journal in `image-graph`

## Context

A frame in the document holds a placed image: the original file bytes, which the host
serves to a plugin through `assets.images@1`. The plugin edits pixels. The session header
states where that leaves the document: "the DOCUMENT is never mutated (the original placed
bytes stay the truth — adjusted-pixel save-back is a later milestone, stated in the panel)"
(`glue/src/session.ts:28-30`). Save-back has since been built.

The host contract also has a door for per-plugin bytes stored inside the document file
(`plugin-sdk: packages/plugin-api/src/host.ts:1265-1280`). This bundle does not call it.
The repository does not record why edits are not stored in the document.

## Decision

Pixels, adjustment parameters, the layer stack, the selection and the undo history are held
by the session, in the bundle and in wasm memory, and are not written into the document.
Edited pixels leave only as file bytes.

- **In.** A selected frame's bytes are read with `host.assets.getPlacedImage` and decoded
  into the engine. The importer for `.psd .psb .png .jpg .jpeg` decodes into the session as
  well: "it does NOT replace the document". The session holds one source at a time.
- **Out.** `applyToFile` computes the result at full resolution and stages file bytes in the
  source's own format: PSD for a PSD, JPEG for a JPEG, otherwise PNG. `saveToFile` hands the
  staged bytes to `host.shell.saveFile` when `shell.saveFile@1` is supported. Three
  exporters (PSD, PNG, JPEG) deliver through the host's exporter registry.
- **Document writes** are two. An ownership marker on the frame, `{ v: 1, data: { owns:
  "pixels" } }`, lets the plugin's edit context claim the frame on double-click. `insertPath`
  mutations convert a selection into paths.
- **Undo** of pixel edits is the plugin's own tile journal (32 entries, 256 MiB), driven
  by the plugin's `undo` and `redo` commands, not by the host's history.

## Evidence

- `glue/src/session.ts:1480`, `glue/src/activate.ts:597-610` — placed bytes; the importer
- `glue/src/session.ts:1393-1394`, `:1110-1132` — a new decode frees the previous source,
  its layer stack and its PSD parse
- `glue/src/session.ts:1635-1707`, `:1709-1734`, `glue/src/activate.ts:628-650` —
  `applyToFile`, `saveToFile`, the three exporters
- `glue/src/session.ts:1359-1381`, `glue/src/activate.ts:669-679` — the marker; the edit
  context matches on the plugin's own metadata
- `glue/src/session.ts:2305-2313` — `insertPath`, the only `host.document.mutate` call
- `glue/src/activate.ts:400-415`, `image-graph/src/journal.rs:98-106` — the undo commands
  and the journal bounds
- `glue/manifest.json:8-11` — `document: { read: "broad", write: "scoped" }`

## Alternatives considered

- **The exporter registry as the only way out.** That was the earlier state. `saveToFile`
  was added when the contract gained a save-file door; the exporters "remain the delivery
  lane for a host with no saver" (`glue/src/activate.ts:427-431`).
- **Claiming frames by element kind instead of a marker.** Rejected in a comment: "matching
  on kind would claim every rectangle in the document and steal a gesture that belongs to
  the host" (`glue/src/activate.ts:676-677`).
- **A marker that carries state.** Declined: the marker "says who owns the frame's pixels
  and nothing else, so it can never disagree with the engine about them"
  (`glue/src/session.ts:1362-1364`).

## Consequences

A saved document contains nothing from this plugin except the marker and any inserted paths. The session keeps pixels, layers and
history in memory only. `glue/src` calls no storage door, and each decode calls `freeSource()`, which closes the layer stack, frees
the previous image handle and closes its PSD parse (`glue/src/session.ts:1393-1394`, `:1110-1132`). No code in `glue/src` restores
session state, and none links the frame to a saved file: save-back produces bytes for the host's saver or export UI and stops there.

There are two undo stacks. The host's undo does not cover pixel edits; the plugin's
journal does not cover layer structure (`image-js/src/layers.rs:115-124`). `glue/src/menu.ts:17-21`
keeps the plugin's undo out of the menu bar because it works on "a DIFFERENT stack (the
image session's, not the document's)". The marker write is itself a document mutation, so
it is skipped when a marker is already present (`glue/src/session.ts:1373-1376`).

The PSD exporter and `applyToFile` differ. The exporter calls `applyToFile` only when the adjustment parameters are
not the identity; at identity it returns `psd_save` of the retained parse (`glue/src/session.ts:1736-1749`,
`:1622-1625`). Pixels reach the retained parse only through `psd_apply_adjusted` (`image-js/src/lib.rs:3616-3631`),
and `applyToFile` is its only caller (`glue/src/session.ts:1657`). `applyToFile` runs that save-back whether or not
the parameters are the identity; when they are not, it needs a GPU device (`glue/src/session.ts:1641-1646`).

Two comments are behind the code: `glue/src/session.ts:29-30` calls save-back a later
milestone, and `glue/src/activate.ts:760-768` logs that the contract has no save-file door
while `glue/src/session.ts:1721` calls `host.shell.saveFile`.

## Related

- [ADR 459](459-scene-layer-image-and-tiles.md) — the in-frame preview; [ADR 458](458-psd-preservation.md), [ADR 456](456-codecs.md) — what the saved bytes contain
- [ADR 017](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/017-importer-exporter-door-shape.md) — the importer and exporter doors
- [ADR 311](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/311-plugin-state-under-own-id.md), [ADR 316](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/316-native-content-and-baking.md) — where the contract expects plugin state and content to live

## Amendment — 2026-10-04

Superseded by [ADR 462](462-sessions-persist-in-parts.md). A session is now committed to the
document: the layers are stored in the plugin's container parts and their composite replaces
the frame's placed image in one undoable step (`glue/src/session.ts:2363-2438`); reopening the
frame restores the layers (`glue/src/session.ts:982-1013`). The decision above described the
code before that commit path existed.
