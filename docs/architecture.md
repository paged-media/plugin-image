# Architecture

How the `paged.image` plugin is built: a raster image engine written in Rust, compiled to
one WebAssembly module that runs its pixel kernels on WebGPU, and a TypeScript bundle that
connects it to the editor. This page describes what the code does at commit `f7d21e5`. The
reason behind each choice is in an ADR under [`adr/`](adr/README.md), linked where it
applies. Paths are relative to the repository root.

## Layout and crates

Two workspaces sit side by side at the root; there is no `crates/` or `packages/` folder.
The Cargo workspace has ten crates (toolchain 1.93.0); the pnpm workspace has two packages,
`glue/` and `manifest/`. `registry/` holds five YAML files that record sources, tolerances
and tests per kernel, PSD block, codec, colour lane and brush-file block;
`registry/kernels.yaml` is also build input. `references/` is gitignored (third-party
source trees, see [ADR 461](adr/461-clean-room-protocol.md)).

| Crate | Owns | Workspace dependencies |
|---|---|---|
| `image-core` | `PixelFormat`, `Region`, `Tile` and `TileMap` (256-pixel tiles), generation and content hashes; crop geometry and the curve-to-table builder | nothing |
| `image-kernels` | `KernelDef`, the shader binding interface (`src/abi.rs`), the `kernel_family!` macro, the kernel definitions (`src/families/`) | core |
| `image-gpu` | the wgpu device, shader assembly, dispatch and readback; selection coverage, brush dab planning, the stroke compositor, histogram and statistics, a distance transform; a texture pool and a tile residency manager | core, kernels |
| `image-pipeline` | "Engine A": a lazy graph of sources and kernel applications, evaluated when a sink pulls a region | core, kernels, gpu, codecs |
| `image-graph` | "Engine B": a persistent tiled buffer graph with per-tile invalidation, and the tile undo journal | core, kernels, gpu, pipeline |
| `image-codecs` | `ImageSource` and `ImageTarget` over a `ByteSource`; PNG, JPEG and raw adapters; an EXIF reader | core |
| `image-cms` | the `CmsEngine` trait and its two backends | core |
| `image-psd` | the PSD/PSB parser, model, edit operations and writer; a reader for `.abr` brush files | core |
| `image-js` | the wasm surface and the editor-facing logic behind it: ingest, the layer stack, selection, strokes, fills, healing, raster type, save-back | the eight crates above |
| `image-conformance` | tests only: the scalar reference harness, fixture builders, property tests, one benchmark | core, kernels with feature `reference`, gpu, pipeline, graph, codecs, psd |

`image-gpu` is the only crate that depends on `wgpu`; `image-js` is the only `cdylib` and
the only crate that depends on `wasm-bindgen`. `deny.toml` denies unknown registries and
git sources. `.github/workflows/ci.yml` adds two `cargo tree` checks: `image-kernels` must
not reach the pipeline, graph, gpu, cms or js crates, and the wasm32 tree of `image-js` must
not contain `image-conformance` or `proptest`.

## The bundle and the two manifests

`glue/` is the published package `@paged-media/image`. `glue/src/index.ts` calls
`defineBundle({ manifest, activate })`; `activate.ts` registers the panel, commands, tools,
importer, exporters and edit context; `session.ts` holds the session state and makes nearly
every call to the host and to the engine; `tile-provider.ts`, which it calls, also reaches both;
`engine.ts` is a typed facade over the wasm exports. The interaction state machines
(`crop-machine.ts`, `selection-machine.ts`, `brush-machine.ts`), the image-to-page transform
(`frame-fit.ts`) and the growth rule of the quick-selection tool (`quick-select.ts`) are
TypeScript. `scripts/check-contract-imports.mjs` rejects any static import or re-export whose
specifier is not `@paged-media/plugin-api`, `@paged-media/plugin-sdk`, one of this repository's
own packages, `react` or a relative path: the plugin takes no code from the engine or the editor
([plugin-sdk ADR 315](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/315-isolation-contract.md)).

There are two manifest files with the same content. `glue/manifest.json` is the one the
bundle uses: `index.ts` and `activate.ts` import it, and `glue/package.json` lists it in
`files`, so it ships in the npm package. `manifest/manifest.json` is a second tracked copy,
in the private package `@paged-media/image-manifest`: `pnpm validate:manifest` and the
"validate manifest" step of CI check this file, and 14 of the 19 spec files in `glue/test/`
import it. No script copies one to the other; the test "is byte-identical to the validated
manifest" in `glue/test/activate.spec.ts` fails when they differ. The wasm artefact is
doubled the same way: `scripts/build-wasm.sh` writes it to `glue/wasm/` (committed; the
bundle loads it from there) and copies it to `manifest/wasm/` (gitignored), where the
manifest's path `wasm/image_js_bg.wasm` resolves from `manifest/`.

## The wasm boundary

The engine is one module, `glue/wasm/image_js_bg.wasm` (2,743,131 bytes), built from
`image-js` with `wasm-bindgen --target web`. `glue/src/engine.ts` imports the generated
JavaScript and instantiates the module itself, in the bundle's own realm, because the
engine creates its WebGPU device from `navigator.gpu` (`image-js/src/lib.rs:36-40`). The
manifest declares `gpu: { realm: "bundle" }` and one `wasm` entry
([plugin-sdk ADR 308](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/308-plugin-wasm.md)).

The surface is 125 exported functions and one small class (`glue/wasm/image_js.d.ts`),
taking numbers, strings, typed arrays and packed `f32` blocks. State stays in thread-locals
inside the module: the GPU context, decoded images by `u32` handle, mip pyramids, one
selection, one layer stack, one stroke in flight, and parsed PSD files by handle.
`mod wasm` in `image-js/src/lib.rs` is compiled with `#[deny(dead_code)]`, and
`glue/test/wasm-surface.spec.ts` compares the hand-written TypeScript interface with the
generated `.d.ts`. A second package entry, `glue/src/decode-worker.ts`, loads its own
instance of the module in a worker and decodes.

## One edit, end to end

```
placed frame selected            or   file opened / dropped
   | host.assets.getPlacedImage        | importer (.psd .psb .png .jpg .jpeg)
   v                                   v
original file bytes  -------------------------------------------- glue/src/session.ts
   | decode: worker pool if the host grants one, else main thread
   |   PNG / JPEG adapters or the PSD merged composite            image-js/src/ingest.rs
   |   EXIF orientation, colour transform to sRGB (CPU)           display.rs, cmyk.rs
   v
engine-held image (u32 handle)  +  PSD parse kept for save-back   image-js/src/lib.rs
   | layer stack opened over the handle; selection bound to it    layers.rs, selection.rs
   v
"Apply": adjust chain through Engine A, GPU kernels  -> RGBA8     ingest.rs
   v
surface.submit(frame, { kind: "image", rgba, ... })               session.ts (submitLayer)
   -> the host draws the image inside the frame; the document is unchanged
```

1. **Ingest and decode.** "Adjust image" reads the original bytes of the one selected
   frame. The importer decodes a file into the session without replacing the document, and
   binds the image to the selected frame when exactly one is selected. With a worker pool
   (`glue/src/decode-pool.ts`) a worker decodes and the main realm registers the RGBA8
   result; otherwise `decode_image` runs on the main thread. A PSD is also parsed
   structurally and, when its layers can be reproduced, opened as a layer stack; otherwise
   its merged composite is the one "Background" layer and the status line says why.
2. **Adjust.** `adjust_image_ext` builds an Engine A pipeline: a raw source over the held
   pixels, one node per stage that is not at identity, and a sink that pulls the image in
   strips of 256-pixel tiles and returns straight RGBA8. At identity the held pixels are
   returned without a dispatch. A curve is a 256-entry table applied on the CPU afterwards.
3. **Composite.** `submitLayer` fits the image into the frame's box and submits one image
   item to the host's scene layer. It runs on Apply; at the end of a layer operation, a
   filter, an undo or redo, a crop and a committed stroke, when the image is bound to a
   frame; and once per processed brush sample during a stroke. A fill or a resize waits
   for the next Apply. The layer is cleared when the frame leaves the selection and on
   Reset ([ADR 459](adr/459-scene-layer-image-and-tiles.md)).
4. **Save-back.** "Apply to file" runs the same chain and encodes the result into the
   retained PSD parse, or as PNG, or as JPEG when the source was a JPEG; a session with more
   than one layer exports as a layered PSD (`image-js/src/saveback.rs`, `psd_write_stack`).
   The bytes leave through `host.shell.saveFile` or an exporter.
5. **Commit.** "Commit image edits to the document" writes the layer stack into the
   plugin's container parts and replaces the frame's placed image with the composite, in
   one batch mutation; reopening the frame restores the layers
   ([ADR 462](adr/462-sessions-persist-in-parts.md)).

## Kernels and how they run

A kernel is one `KernelDef` (`image-kernels/src/lib.rs`): an id, a class, an input count, a
`#[repr(C)]` parameter block, WGSL source and a tolerance. 128 are registered, under 14 id
prefixes (`adjust`, `band`, `bool`, `cast`, `compose`, `conv`, `gallery`, `gen`, `geom`,
`math`, `morph`, `rank`, `rel`, `resample`). `image-kernels/build.rs` generates the lookup
table from the rows of `registry/kernels.yaml`, so a definition without a row cannot be
looked up, and a unit test asserts that registry and code define the same set
([plugin-sdk ADR 317](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/317-registry-driven-dispatch.md)).
Every kernel uses one binding interface: inputs in group 0, parameters in group 1, a
selection mask in group 2, an `rgba16float` output in group 3
([ADR 452](adr/452-kernel-abi.md)). Kernels that fit a small expression language are
written once and emitted as WGSL and, under the cargo feature `reference`, as scalar Rust;
the others are handwritten WGSL modules ([ADR 451](adr/451-one-kernel-definition.md)). Only
`image-conformance` enables `reference`; the shipped module has no CPU kernel
([ADR 450](adr/450-gpu-only-kernels.md)).

`image-js` runs kernels in two ways. The adjustment chain goes through Engine A. Filters, fills, resize
and the layer fold dispatch over the whole image as one texture (`image_gpu::execute_tile_once_async`,
`execute_windowed_once_async`), outside Engine A. A brush stroke composites only the rectangle its new
dabs touched (`image_gpu::composite_stroke_window`). The rest is CPU work: decoding, colour transforms,
coverage rasterisation, histograms, the curve table, the content-aware fill search (`inpaint.rs`), the
healing solve (`heal.rs`), glyph shaping and rasterisation (`text.rs`) and the mip downsample
(`mip.rs`). From Engine B the wasm surface uses the tile journal and a source-only mip pyramid; the
graph's op nodes and incremental evaluation, and the texture pool, residency manager and batched
dispatch in `image-gpu`, are called only from tests ([ADR 454](adr/454-two-evaluation-engines.md)).

## Layers, selection, colour and depth

The layer stack (`image-js/src/layers.rs`) is an ordered list of canvas-sized layers:
pixel, adjustment and smart-object layers, each with visibility, lock, opacity, a blend
mode that is one of the 26 `compose.*` kernels, an optional mask, an optional group and a
clip flag. The composite is a bottom-up fold in premultiplied `rgba16float`: one blend
dispatch per pixel layer, and the adjustment chain over the accumulator for an adjustment
layer. A stack of one plain layer returns that layer's pixels without a dispatch. The
result is written back into the image handle the stack is bound to, so every other lane
keeps addressing one image. Pixel edits to the active layer are recorded in the tile
journal, at most 32 entries and 256 MiB (`image-graph/src/journal.rs:101-106`). Undo and
redo are the plugin's own commands and act on this journal, not on the host's history.

The active layer has an edit target, its pixels or its mask (`LayerStack::set_edit_target`).
With the mask as target a stroke paints the mask as an opaque grey plate through the same
stroke compositor (`StrokeParams::for_mask_target`: the paint colour becomes the
foreground's grey, the eraser paints white), previews by folding the stack with that plate
standing in for the mask, and commits through `LayerStack::edit_active_mask`. The mask edit
is journaled one byte per texel under the scope `layer id | MASK_SCOPE_BIT`, so it is one
step on the same undo list and replays into the mask of the right layer. The fold itself is
unchanged: a painted mask is an ordinary mask.

The selection is an 8-bit coverage field at image resolution (`image-gpu/src/coverage.rs`),
rasterised on the CPU and passed to the kernels as the group-2 mask. A brush stroke keeps a
base snapshot and a coverage accumulator in the engine (`image-js/src/stroke.rs`).

Colour transforms run on the CPU at decode: an embedded RGB profile is converted to sRGB with
qcms (`image-js/src/display.rs`), an embedded CMYK profile with moxcms (`image-js/src/cmyk.rs`)
([ADR 457](adr/457-colour-management.md)). Held pixels are straight (not premultiplied) RGBA at
8 or 16 bits per sample in the `Pixels` type (`image-js/src/pixels.rs`), and are uploaded as
encoded values: ingest applies no premultiply and no transfer to linear
(`image-js/src/ingest.rs:43-47`; [ADR 455](adr/455-pixel-model.md)). `image-psd` re-encodes
fixed-width scalar structures, keeps the source bytes of every other block it parses, and
writes unmodified blocks back byte for byte ([ADR 458](adr/458-psd-preservation.md)).

## Where data is stored

During a session, images, layers, the selection, the undo list and the PSD parse live in
the wasm module's memory, and the panel's parameters in the session object. A COMMIT
(ADR 462) writes the layer stack into the plugin's container parts (`px/<sha256>.bin`
buffers, shared across revisions, and `f/<frame>/r<n>.json` records;
`glue/src/session-store.ts`), replaces the frame's placed image with the composite, and sets
the frame's marker to `{ v: 2, data: { owns: "pixels", rev, record, baked } }`. Before a
commit the marker is `{ v: 1, data: { owns: "pixels" } }`, written at ingest so the
`rasterImage` edit context matches. "Selection to path" inserts paths. The undo history is
not stored.

## Host doors

| Door | What the plugin uses it for |
|---|---|
| `contributePanel`, `host.contribute.command`, `contributeTool`, `host.contribute.menu` | one panel ("Image"), 37 commands, 15 tools, 34 menu entries |
| `host.contribute.importer`, `host.contribute.exporter` | one importer for `.psd .psb .png .jpg .jpeg`; exporters for PSD, PNG and JPEG |
| `host.contribute.editContext`, `host.contribute.bindingProvider` | the `rasterImage` context, entered by double-click; two providers that make the host's Layers and Character panels show the raster stack and the raster type settings |
| `host.assets.getPlacedImage`, `host.assets.getFontFace` | original bytes of a placed image; font bytes for raster type |
| `host.contribute.sceneLayer()` | submit and clear the image item inside a frame |
| `host.images.claimImageResource` | serve 256-pixel tiles of level 0 to the renderer, on the "claim tiles" command |
| `host.overlay.setToolPreview` | the crop frame, the selection outline, the brush tip ring |
| `host.document.elementGeometry`, `host.selection.get` / `onDidChange` | the frame's box; the target frame; clearing the layer on deselect |
| `host.document.getMetadata` / `setMetadata` | the ownership marker |
| `host.document.mutate` (`insertPath`), `host.document.pathAnchors` | selection to path and path to selection |
| `host.shell.saveFile`, `host.shell.pickFile`, `host.shell.openPanel` | deliver save-back bytes; open an `.abr` file; raise the panel |
| `host.workers.concurrency` / `spawn` | the decode worker pool |
| `host.supports`, `host.log` | probing optional doors; logging |

The manifest declares `document` (read `broad`, write `scoped`), `rendering`
(`sceneLayer`, `hitTest`, `resourceProvider`, `overlay`), `assets` (`images`, `fonts`),
`network: false`, `workers` (at most 4, `sharedMemory: true`), `gpu` and one `wasm` module
of at most 8 MiB. No code in `glue/src` calls a hit-test door or uses shared memory.

## Build and test

- `bash scripts/build-wasm.sh` builds `image-js` for wasm32, checks that the installed
  `wasm-bindgen` CLI matches `Cargo.lock`, runs it with `--target web`, runs `wasm-opt -Oz`
  if installed, and fails above 100 MB. `.github/workflows/publish.yml` runs it and
  publishes `@paged-media/image` under the `canary` tag when the version is new.
- Rust: `cargo test --workspace`. Tests that need a GPU device skip when no adapter is
  found (`image-conformance/src/device.rs`). CI runs on `ubuntu-latest`, which per the
  comment in `ci.yml` has none, so GPU output is compared with the scalar reference only
  on a machine with an adapter ([ADR 453](adr/453-tolerance-not-golden-bytes.md)).
- TypeScript: `pnpm test` runs the import lint, then vitest in `glue/` against the SDK's
  `createBundleHost` and a fake editor. `glue/test/wasm-exec.spec.ts` loads the committed
  module in Node and drives the CPU lanes; CI sets `REQUIRE_REAL_WASM=1`, which turns a
  missing module from a skip into a failure. The manifest validator is the `plugin-cli` of
  a sibling `plugin-sdk` checkout, called by the scripts of `manifest/`.
