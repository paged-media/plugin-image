# Documentation

What this folder holds.

- [`concept.md`](concept.md): why the plugin exists, what it is for, and what it will never do.
- [`architecture.md`](architecture.md): how it is built. The ten crates and the bundle, the
  two manifests, the wasm boundary, the path of one edit, how kernels run, where data is
  stored, the host doors used, and how it is built and tested.
- [`status.md`](status.md): what ships today, the limits of what ships, and what is not built.
- [`adr/`](adr/README.md): the decision records of this repository, 450–461.
- [`design/16-bit-stack.md`](design/16-bit-stack.md): the design note for carrying 16 bits
  per sample through the layer store.
- [`design/selection-tool.md`](design/selection-tool.md): the design note for the selection
  tools over the kernel mask.
- [`design/colour-management-at-ingest.md`](design/colour-management-at-ingest.md): the
  design note for applying colour transforms at decode.

## Decisions in other repositories that bind this one

These records live in other public paged-media repositories. The code here rests on each of
them. The last column says what the decision means for this plugin.

| ADR | Repository | Decision | What it means here |
|---|---|---|---|
| [010](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/010-raw-mutate-gate-capability-enforcement.md) | plugin-sdk | The raw-mutate gate and the capability enforcement line | The manifest asks for `document.write: "scoped"`. The plugin's document writes are raw `insertPath` mutations and its own metadata marker (`glue/src/session.ts`). A contribution the manifest does not declare fails at activation with a `PluginCapabilityError` (comment in `glue/test/activate.spec.ts`), which is why a test keeps the two manifest copies identical. |
| [017](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/017-importer-exporter-door-shape.md) | plugin-sdk | Importer and exporter door shape | `glue/src/activate.ts` registers one importer for `.psd .psb .png .jpg .jpeg` and three exporters (PSD, PNG, JPEG). The importer receives a file's bytes and decodes them into the session; it does not replace the document. |
| [305](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/305-doors-always-present.md) | plugin-sdk | Every door is always present; `supports()` reports a missing backend | Each optional door is probed with `host.supports(...)` and has a fallback: a status message for placed-image bytes, the scene layer, the tile resource, fonts and the file picker; the exporters when there is no save door; main-thread decode when there are no workers; no registration for the importer, exporters, edit context and binding providers. |
| [307](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/307-contract-as-peer-dependency.md) | plugin-sdk | Bundles take the contract packages as peer dependencies | `@paged-media/plugin-api`, `@paged-media/plugin-sdk` and `react` are peer dependencies of `@paged-media/image` (`glue/package.json`). |
| [308](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/308-plugin-wasm.md) | plugin-sdk | Plugin wasm is a declared capability, loaded by the bundle, under one size budget | The manifest declares one wasm module, `wasm/image_js_bg.wasm`, with an 8 MiB ceiling, and `gpu: { realm: "bundle" }`. `glue/src/engine.ts` loads the module in the bundle's realm on first use. `scripts/build-wasm.sh` stops at 100 MB. |
| [310](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/310-one-write-door.md) | plugin-sdk | One write door: `document.mutate`, engine-owned history, failures as outcomes | Path inserts go through `host.document.mutate`, and the code tests `outcome.applied`. Pixel edits are not document mutations: their undo is the plugin's own tile journal and its own commands, outside the host's history. |
| [311](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/311-plugin-state-under-own-id.md) | plugin-sdk | Plugin state lives only under the plugin's own id | The only state the plugin puts in the document is its marker, written with `host.document.setMetadata` as the plugin's own metadata on a frame. The edit context matches on that metadata, never on the element kind. |
| [314](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/314-plugin-shape.md) | plugin-sdk | The plugin shape: semantics in Rust behind one wasm module, a logic-free shim, one published package | One wasm module (`image-js`) and one published package (`@paged-media/image`). The TypeScript side is not logic-free: the interaction state machines and the growth rule of the quick-selection tool are in `glue/src/`. |
| [315](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/315-isolation-contract.md) | plugin-sdk | The isolation contract: a plugin depends only on the published contract | `scripts/check-contract-imports.mjs` runs before the tests. `deny.toml` denies unknown registries and git sources, and CI checks two dependency trees. The plugin brings its own codec, colour and text-shaping crates. |
| [317](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/317-registry-driven-dispatch.md) | plugin-sdk | Function and kernel dispatch is generated from a registry, with a coverage gate | `image-kernels/build.rs` generates the kernel lookup table from `registry/kernels.yaml`; a unit test in `image-kernels/src/lib.rs` asserts that the registry and the code define the same set. |
| [318](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/318-host-spawned-workers.md) | plugin-sdk | Workers are spawned by the host on a declared capability | The manifest declares `workers` with `max: 4`. `glue/src/decode-pool.ts` asks `host.workers.concurrency()` and spawns up to that many workers with the module path `workers/decode.js`; the worker entry is `glue/src/decode-worker.ts`. |
| [003](https://github.com/paged-media/core/blob/main/docs/adr/003-lcms2-color.md) | core | Colour CMM: lcms2 native, qcms on wasm | The display lane pins qcms 0.3 with the features `cmyk` and `iccv4-enabled` (`Cargo.toml`), the same declaration as `core: crates/paged-color/Cargo.toml:24`, so an embedded RGB profile is converted by the same library the canvas uses. `registry/cms.yaml` states this as the reason. |
| [013](https://github.com/paged-media/core/blob/main/docs/adr/013-in-frame-scenelayer.md) | core | In-frame plugin rendering via `SceneLayer` | Every result is shown as one `image` scene-layer item submitted for the frame (`submitLayer` in `glue/src/session.ts`). The plugin computes the fit inside the frame's box; the host clips and transforms. |
| [018](https://github.com/paged-media/core/blob/main/docs/adr/018-stage-b-gpu-texture-defer-record-only.md) | core | Shared GPU device and plugin texture: deferred | No GPU texture is shared with the host. Results are read back from the plugin's own device and cross as bytes, and the mip-level tile export has no TypeScript caller. |
| [108](https://github.com/paged-media/core/blob/main/docs/adr/108-plugin-raster-tiles.md) | core | Plugin raster content enters as tiles through the ordinary image lane | `glue/src/tile-provider.ts` claims a frame's image resource with `host.images.claimImageResource` and serves 256-pixel tiles of level 0. |
| [023](https://github.com/paged-media/editor/blob/main/docs/adr/023-shared-panels-binding-providers.md) | editor | Shared panels: the host owns the panel, plugins provide the values | Two binding providers are registered for the `rasterImage` context (`glue/src/binding-provider/`): one serves the raster layer stack to the host's Layers panel, one serves the raster type settings to the host's Character panel. |
| [024](https://github.com/paged-media/editor/blob/main/docs/adr/024-context-sensitivity-is-a-core-concept.md) | editor | Context-sensitivity is a core concept | The `rasterImage` edit context is entered by double-click on a frame that carries the plugin's marker. It names the raster tools and the Image panel as its surface (`glue/src/activate.ts`). |
