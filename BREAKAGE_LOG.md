# BREAKAGE_LOG — paged.image vs. the plugin surface

Every place the published plugin surface (`@paged-media/plugin-api` v0.2
/ `plugin-sdk`) falls short of what paged.image needs. This log is BOTH
the API-v1 punch list (like draw's B-NN / web's W-NN) AND the resolution
of the spec's §2.2 gap table — the M0 phase-0 exit criterion is every
§2.2 row "covered" or an entry here with an RFC direction. Verified
against the published SDK + local plugin-sdk checkout on 2026-06-07;
evidence detail in `thoughts/docs/paged/plugin-image/a0-audit.md`.

Format: `I-NN · date · area · status`.

---

## §2.2 row dispositions

- Read access to placed-asset bytes + link metadata — **GAP** → I-04.
- Commit Operations / participate in undo via op log — **COVERED**:
  `host.document.mutate()` (mutate-never-throws, the full engine
  Mutation union) + `undo`/`redo`; Engine B's
  `SetParams`/`EditGraph`/`WriteBuffer` lower onto this surface (M2).
- Declarative panels + custom canvas region — panels **COVERED**
  (React expert leaf via `contribute.panel`); the embedded
  canvas/texture surface is **GAP** → I-01.
- Spawn dedicated workers + SharedArrayBuffer — **GAP** → I-02.
- OPFS quota for swap tier + pyramid cache — **GAP** → I-03.
- Register document/asset type handlers (PSD opens via plugin) —
  **GAP** → I-05.
- Provide image resources back to Vello — **GAP** → I-06.

---

- **I-01 · 2026-06-07 · panels / GPU surface · OPEN** — no plugin-owned
  `GPUCanvasContext`, texture-share handle, or custom canvas region.
  Panels are React-only; the only on-canvas affordance is
  `overlay.setToolPreview` (rect | polyline — draw B-07). A bundle
  cannot own a wgpu surface inside the editor viewport, nor hand a
  texture to the compositor without a CPU copy. THE single most
  consequential row (spec §13 note): a per-frame copy across this
  boundary breaks the interactive budgets (pointwise gesture < 8 ms,
  gaussian < 16 ms). RFC direction: plugin-owned `GPUCanvasContext` OR a
  texture-share contract (shared `GPUDevice` / importExternalTexture).
  Draft: `thoughts/docs/paged/plugin-image/rfc-gpu-surface.md` (with
  I-07; D-8 owner: plugin-integration track). Until resolved, Engine B
  viewport interactivity is degraded/blocked (M2 gate, not M0).

- **I-02 · 2026-06-07 · workers / SAB · OPEN** — no worker-spawn
  capability and no SharedArrayBuffer grant in the bundle contract. The
  wasm packaging contract (plugin-sdk `docs/wasm-packaging.md`, W-07)
  states SAB/threads are OFF in v1 (host-owned non-shared memory; the
  loader never sets `shared: true`). Engine A's CPU-worker decode pool
  (spec §7.1, wasm-bindgen-rayon over SAB) needs the grant. A-0 audit
  finding: the editor itself IS cross-origin isolated (COOP same-origin
  + COEP require-corp via `_headers` + the vite plugin; the render
  worker allocates camera/gesture SABs) — the platform can host SAB; the
  gap is purely the plugin CONTRACT. RFC direction: worker capability
  with COOP/COEP guarantees + a shared-memory grant in
  `capabilities.wasm`. Spec D-6 (plugin-owned pool) stands. M0/M1
  workaround: single-threaded decode bridge (landed as such by design).

- **I-03 · 2026-06-07 · storage / OPFS · OPEN** — `host.storage` is a
  JSON KV door (get/set/delete/keys; localStorage-backed in-process);
  no OPFS quota capability, no byte/blob store. A-0 audit finding: the
  editor uses ZERO OPFS today — no platform precedent. The Tier-2 swap
  tier + pyramid cache (spec §9.1, §7.3 `to_pyramid`) need it. RFC
  direction: storage capability with quota declaration + an OPFS/blob
  store distinct from the KV door. M0: residency Tier 2 is a typed stub.

- **I-04 · 2026-06-07 · assets / bytes door · OPEN** — document reads
  are structure/geometry/metadata only; there is NO placed-asset byte
  accessor (paged.web hit the same wall: W-06 — names cross the
  boundary, bytes don't). paged.image needs the raw bytes of placed
  assets (the PSD/JPEG/TIFF a frame links) to feed `image-pipeline`.
  RFC direction: capability-gated asset-bytes read door (link metadata +
  byte stream), shared design with W-06. M0 workaround: the engine is
  exercised through its own wasm surface + conformance corpus, not
  through placed-asset ingest.

- **I-05 · 2026-06-07 · contributions / importer-exporter · OPEN** —
  `editContexts`/`objectTypes` are declarable in the manifest schema but
  throw `PluginApiNotImplemented` at runtime (host reserved members;
  draw B-02 / web W-03 hit the same wall). There is no document/asset
  type handler registration, so "PSD opens via the plugin" has no door.
  RFC direction: importer/exporter registration capability + the
  edit-context registry. The M0 manifest deliberately declares neither.

- **I-06 · 2026-06-07 · renderer / texture provider · OPEN** — no
  contract to provide image resources (pyramid tiles for placed assets)
  back to the renderer. A-0 audit finding: core consumes whole
  `DecodedImage` RGBA8 buffers (lazy-decoded from `encoded` bytes) via
  peniko `ImageData`/`ImageBrush`/`draw_image` — no tile/pyramid source
  abstraction, no plugin-facing registration, no budget-arbitration
  contract with Vello's image pool (spec §9.1 expects one). RFC
  direction: texture/image resource provider contract; co-design with
  I-01's texture-share.

- **I-07 · 2026-06-07 · wasm lane / budgets + WebGPU reach · OPEN
  (RISK)** — the wasm packaging contract caps artifacts at 8 MiB each /
  16 MiB declared total (validator constants), 256 MiB memory ceiling,
  3 s load budget, and `loadBundleWasm` instantiates with NO ambient
  authority: no DOM, no network, and no `navigator.gpu` in the import
  object. Two consequences for a wgpu-based engine: (a) the built
  `image-js` release wasm must be measured against the 8 MiB budget
  every CI run (wgpu's webgpu backend is a thin browser-API wrapper, but
  codecs + engines add up); (b) a plugin wasm module CANNOT request a
  GPUAdapter itself — WebGPU is reachable only from the bundle's JS
  realm, which is exactly where wasm-bindgen's generated glue lives. So
  the engine's GPU device is created via the `image-js` wasm-bindgen
  surface running in the bundle realm (the `@paged-media/sdk` pattern),
  NOT via `loadBundleWasm`'s no-authority instantiation. Whether that
  realm is granted `navigator.gpu` long-term — and whether a
  host-provided `GPUDevice` import becomes the sanctioned path — is the
  RFC (pair of I-01, same draft). The existential row for the spec's
  GPU-only thesis inside the plugin sandbox.
