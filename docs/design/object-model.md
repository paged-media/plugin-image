# The object model (layers, masks, adjustments, smart objects as typed objects)

2026-10-06. Design note for paged.image's contribution to the plugin
object-model contract (ADR 323, `contribute.objectModel` /
`host.objects`). Code: `glue/src/object-model/`. Tests:
`glue/test/object-model.spec.ts`, `glue/test/object-model-manifest.spec.ts`,
`image-js/tests/object_schema.rs`.

**Problem.** An image frame's layer stack lived in the engine's single
layer slot and the image panel. A data binding, a script, the Node CLI or
another panel could not name a layer, read its opacity or replace its
pixels. Every command took an untyped payload.

## Kinds and addresses

Addresses are `plugin:media.paged.image/<kind>/<id>`.

| Kind | Id | Rows | Source of the rows |
|---|---|---|---|
| `image` | `<frame>` | 5 (all read-only: width, height, layerCount, revision, pending) | hand |
| `layer` | `<frame>/<layerId>` | 12 + `content` (name, visible, opacity %, blendMode, order, locked, clipped, kind, id, hasMask, bounds, pixelSource) | hand; `blendMode` members generated from the `compose.*` registry |
| `mask` | `<frame>/<layerId>` | 2 + `content` (enabled, bounds) | hand |
| `adjustment` | `<frame>/<layerId>` | 89 | **generated** (see below) |
| `smartObject` | `<frame>/<layerId>` | 5 (source, sourceWidth, sourceHeight, scale, renderPending) | hand |

`<frame>` is the frame's `Self` id and `<layerId>` the engine's stable
layer id (it survives reorders). Both are **session handles**: InDesign
renumbers `Self` ids and drops the container parts. A stored reference
selects by name, for example
`rectangle[label.x-paged:oid=…] > plugin:media.paged.image/layer[name="Badge"]`.

### The generated adjustment schema

`image-js/tests/object_schema.rs` writes
`glue/src/object-model/generated/image-schema.json` from two sources:

- the kernels' `ParamsLayout`, through `image_kernels::lookup`;
- the stored block, `AdjustParams::to_wire`, which gives each value's slot
  and default.

The only hand-written part is the binding of a wire slot to a kernel field.
The test checks it in both directions:

- every bound kernel field exists with the component count of its WGSL type;
- every field of a bound kernel is either bound or named internal;
- every wire slot is covered exactly once;
- every slot is live: perturbing it round-trips through `from_wire` and
  `to_wire`.

The liveness check found the two gated values: `posterize.levels` and
`threshold.threshold` are only kept while their gate is on. Their rows say
so (`gatedBy`).

The file is a test output. Nothing enters the wasm, so the committed
artifact's source hash does not change.

## Reading and writing without the engine's layer slot or a GPU

The stack is read and written as the stored manifest. `pgil.ts` is a
TypeScript codec of the engine's `PGIL` format (`layers_persist.rs`). It
re-encodes a real engine export byte for byte, which the spec proves
against the committed wasm. The engine is used only for its CPU codec:
decoding the placed image or a replacement, and encoding the bake.

**Writes** use the session store's label-hash pattern (ADR 462). The
content-addressed parts come first: the manifest, the buffers and a record
at `f/<frame>/r<rev>-<hash>.json`. Then ONE engine batch moves the frame's
`x-paged:media.paged.image` marker. Undo reverts the marker, so a write is
one undo step, and redo still lands on a revision whose parts exist.

- The marker is the plugin's ONE label on the frame: exactly
  `x-paged:media.paged.image`, whose value is the envelope
  `{"v": 2, "data": {…marker}}` that core and the SDK registry require.
- The planner is the PLUGIN-level `batch` of the contribution (SDK
  0.2.43). Layers, masks, adjustments and smart objects share one stack,
  so every op of a batch, whatever its kind, is applied to one copy of each
  frame's stack in batch order: one planner call, one write. (Planned per
  kind, the second kind's plan would start from the stack before the
  first's and drop its change; the registry now refuses that conflict.)
- `hostOf` answers the canonical item address (`rectangle:<id>`), read from
  the scene tree, so it is async; the same address is the write's label
  host.
- A page or doc scope (`page:#2 > plugin:media.paged.image/layer`) is
  resolved by the registry through `hostOf`; the plugin's `list` sees only
  item and plugin scopes.
- `content` (the pixels) and the `asset` rows `layer.pixelSource` and
  `smartObject.source` are `heavy`: `getAll` reads them only with
  `{ content: true }`.

**The bake.** Compositing is a GPU kernel fold, and the constitution allows
no CPU blend path.

- A stack whose composite is the identity is baked in the same batch with
  `replaceImageBytes`. That covers nothing visible and a single plain
  visible 8-bit layer.
- Any other stack is stored with `pending: true`, and the frame keeps its
  previous image.
- A smart object whose source was replaced is stored with
  `pendingRender`.
- When a GPU realm opens the frame, the interactive session finishes both:
  it re-renders, re-composites, and commits on exit.

## Typed commands

All 39 commands have a typed twin with the same id, so the untyped
baseline for `media.paged.image` drops from 39 to 0.

- 35 twins keep their behaviour, acting on the image in the edit context.
  They take no arguments.
- `addLayer {image, name}`, `addLayerMask {layer}`,
  `addLayerMaskHideAll {layer}` and `convertLayerToSmart {layer}` name
  their target and run headless through `host.objects`.
- `replaceLayerPixels {layer, source}` is new.

## Data sets (Photoshop variables)

A Photoshop data set is one apply per record. Each `Binding::Property`
(ADR 558) lowers to a `host.objects.batch`, which is one undo step:

- **Visibility variable**: target `…/layer[name="Badge"]`, path `visible`,
  coerced to `bool` by the row's schema.
- **Pixel-replacement variable**: target `…/layer[name="Photo"]`, path
  `pixelSource` (an `asset`), or `content` (bytes).
  - The bundle has no network door. The binding layer resolves a file path
    or URL to bytes or a `data:` URL.
  - The replacement is placed as is, centred.
  - The last applied source reads back from `pixelSource`.
- **Text-replacement variable**: there is no target. A raster type layer
  is pixels (see the gaps).

Persistence follows ADR 559 §6. The recipe goes in an `x-paged:bind` label
on the host frame (`hostOf` = the frame's canonical item address, e.g.
`rectangle:<id>`). The target is a selector over
layer names. The frame's image carries the last applied values, or the
previous ones while a bake is pending.

The spec drives two records through `host.objects`. The binding itself is
Wave 5.

## Panel fields that should become a shared `PropertyField`

These are built by the editor, not here.

- **Layers** (host Layers panel through the binding provider):
  - `layer.visible`, `layer.locked`, `layer.name`;
  - `layer.opacity` and `layer.blendMode`, which the host panel cannot
    show today because `LayerSummary` has neither;
  - `layer.clipped`, `mask.enabled`.
- **Tone section**: exposure, brightness/contrast, saturation, white
  balance, levels, blur σ, sharpen, hue rotate, vibrance, posterize,
  threshold, black & white, photo filter, colour balance, channel mixer,
  per-channel levels, curves. These map to the `adjustment.*` rows when the
  target is an adjustment layer.
  - The panel's uncommitted preview chain is session state. The field needs
    a transient (preview) mode.
- **Hue/Saturation section**: `adjustment.hueSaturation.*`.
- **Smart-object section**: `smartObject.source` (replace contents).
  - Scale stays a command, because it is a GPU re-render.
- **Not object properties**: tool options (brush size, hardness, flow,
  spacing, wand tolerance), the foreground and background colours, and the
  selection.

## Gaps

| Gap | Class |
|---|---|
| No text layer: type is rasterised into pixels, so the text-replacement variable has no target | not modelled (by design) |
| Non-identity composites are not baked headless; stored `pending` | constitution (GPU-only); needs a GPU realm, for example the Node CLI on a WebGPU runtime |
| Smart-object re-render and `scale` writes | GPU resample; `renderPending`, `scale` read-only |
| Pixel replacement only "as is" (no fit, fill or conform) | GPU resample |
| `http(s)` pixel sources | no fetch door (manifest `network: false`); the caller resolves |
| A placed *layered* PSD reads flat until opened in the editor | engine door: no stateless PSD→manifest export |
| No `group` kind; `order` is refused in a grouped stack | plugin follow-up |
| A write to the frame that is open in the image edit context is not merged into the live session | plugin follow-up (the session should observe its marker) |
| Adjustment rows carry no ranges or units | `ParamsLayout` has none (frozen ABI; needs an amendment or registry ranges) |
| 35 typed twins take no target and act on the edit-context session | plugin follow-up (explicit targets) |
| 16-bit layers: `content` read refused; replacement is 8-bit | plugin follow-up |
| After an InDesign round trip, the parts are gone: the frame opens flat from its baked image (the marker survives and points nowhere) | by design (ADR 559: parts are a cache) |
