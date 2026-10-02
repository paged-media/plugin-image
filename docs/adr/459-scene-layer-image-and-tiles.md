# ADR 459 — Results reach the page as a retained scene-layer image plus a tile provider

- **Status:** Accepted. Recorded retroactively on 2026-10-02 from the code at `f7d21e5`.
- **Scope:** `glue/src/session.ts`, `glue/src/tile-provider.ts`, the tile and adjust exports
  of `image-js`

## Context

The plugin computes on its own WebGPU device inside the bundle (`glue/manifest.json:27-29`
declares `gpu.realm: "bundle"`). The page is rendered by the host. This bundle puts pixels
into a frame through two host doors, and both take bytes: an in-frame scene layer
(`rendering.sceneLayer@1`) and an image resource whose tiles the renderer pulls
(`rendering.resourceProvider@1`).

Handing a GPU texture to the host is not available. A note in the bundle's tests calls it
"deferred by ADR-018 pending a core external-texture path"
(`glue/test/wasm-surface.spec.ts:57-59`). The session header draws the consequence: the
image lane is "static quality by design", and an interactive path is named as later work
(`glue/src/session.ts:25-27`).

## Decision

The session shows its result inside the placed frame by submitting one image item of RGBA8
bytes to the host's scene layer for that frame. Separately, on a command, it claims the
frame's image resource and serves level-0 tiles of 256 pixels.

- `submitLayer` is the only composite path. It submits one item of kind `"image"`: the
  whole image as an array of numbers with its pixel width and height, aspect-fitted and
  centred in the frame's content box, which is read through
  `host.document.elementGeometry`. Clipping and the frame's transform are left to the host.
- The payload is bytes: the `adjust_image` exports return RGBA8, read back from the
  plugin's device whenever the adjustment chain is not the identity.
- The layer is submitted by `apply()`: from the panel's Apply button and after a commit
  that changes the pixels (a crop, the end of a brush stroke, a layer edit). During a brush
  stroke each processed sample also submits the painted image, without the adjustment
  chain. The layer is cleared when the frame leaves the selection, and on Reset.
- `claimImageTiles` calls `host.images.claimImageResource` with `levels: 1` and
  `tileSize: 256`. Its `source` callback returns `null` above level 0 and otherwise cuts a
  window from the image behind the engine handle (`image_tile_rgba8`). `bump()` raises the
  claim's revision when pixels change in place under the same handle.
- Both doors are probed with `host.supports` before use.

## Evidence

- `glue/src/session.ts:1312-1349`, `:841-845` — `submitLayer`; the probe
- `glue/src/session.ts:3162-3217`, `:2727-2734` — `apply()`; the per-sample stroke preview
- `glue/src/session.ts:1432-1442`, `:3273-3282` — cleared on deselect and on Reset
- `image-js/src/lib.rs:42-48`, `:506-525` — what `adjust_image` returns, and from where
- `glue/src/tile-provider.ts:75-127`, `:26-37` — the claim; why only level 0 is served
- `image-js/src/lib.rs:744-769` — the level-0 tile cut: windowing of the held buffer
- `glue/src/activate.ts:120-133`, `glue/src/menu.ts:22-23` — the claim is a command, kept
  out of the menu
- `glue/test/wasm-surface.spec.ts:50-63` — the mip-level tile export has no caller

## Alternatives considered

- **A shared GPU device and a texture composited by the host.** Not available on the host
  side (see ADR 018 below). The session header names it as the interactive path.
- **Tiles at mip levels above 0.** The engine has the export (`image_tile_rgba8_level`,
  `image-js/src/lib.rs:771-786`); the bundle does not call it, because "the tile provider
  fetches the whole image at its natural extent, so no LOD is ever requested"
  (`glue/test/wasm-surface.spec.ts:56-57`).

## Consequences

Each submission moves the whole image: the engine returns it as bytes (a GPU readback when
a kernel ran), and `Array.from` copies it into a JavaScript array of numbers
(`glue/src/session.ts:1337`). This happens on every Apply and every processed brush sample.

The in-frame image is a preview tied to the selection. The session's own status text says
"document unchanged — preview layer only" (`glue/src/session.ts:3207`); deselecting the
frame clears it. How edits become permanent is [ADR 460](460-document-is-not-the-store.md).

The tile lane does not show the adjustment chain. `image_tile_rgba8` reads the image held
behind the handle, and the chain's output is returned to the caller without being stored
there (`image-js/src/lib.rs:506-525`). The tile claim is reachable only through the command
`media.paged.image.command.claimTiles`; nothing else in `glue/src` calls `claimTiles`.

Two comments are behind the code. `glue/src/session.ts:25-26` says resubmission is "never
per-drag"; brush strokes submit per sample. `glue/src/tile-provider.ts:29-31` says the wasm
surface publishes only decode, adjust, free and the level-0 cut; it exports far more,
including `image_tile_rgba8_level`.

## Related

- [ADR 013](https://github.com/paged-media/core/blob/main/docs/adr/013-in-frame-scenelayer.md) — the host's in-frame scene layer
- [ADR 018](https://github.com/paged-media/core/blob/main/docs/adr/018-stage-b-gpu-texture-defer-record-only.md) — the deferral of a shared device and texture hand-off
- [ADR 108](https://github.com/paged-media/core/blob/main/docs/adr/108-plugin-raster-tiles.md) — how plugin tiles enter the renderer
- [ADR 305](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/305-doors-always-present.md) — why doors are probed with `supports()`
- [ADR 454](454-two-evaluation-engines.md) — the engines, and the layer stack behind the handle
