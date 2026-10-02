# paged.image — Core Layer Technical Specification

June 2026. Concept paper. Sections describe intent; where the implementation differs, `status.md` and the ADRs in `adr/` are authoritative.

Sections not relevant outside the original planning context have been removed; numbering is unchanged.

---

## 1. Purpose and scope

*Status note (2026-10-02): this section predates the implementation; see `status.md` for what is built (the tools, selections, brush engine and layer editing listed below as out of scope are built in this repo), [ADR 450](adr/450-gpu-only-kernels.md) for GPU-only execution, and [ADR 453](adr/453-tolerance-not-golden-bytes.md) for how GPU output is verified (CI has no GPU adapter, software or otherwise).*

`paged.image` is the raster subsystem of the Paged ecosystem: a Rust/WASM/
WebGPU image processing engine delivered as a **Paged plugin**, serving:

1. **Pipeline (Engine A)** — a libvips-class, demand-driven streaming engine
   for ingest, thumbnailing, color conversion, export baking, and batch.
2. **Graph (Engine B)** — a GEGL-class persistent tiled buffer engine for
   interactive, non-destructive bitmap editing (the "web Photoshop" track
   and the in-frame asset editor inside Paged documents).

Four properties are **constitutive** — they hold from M0 and are never
phased in:

- **PSD/PSB read and write** (§10.4). PSD is the interchange currency of the
  category. Round-trip safety ("Paged never destroys a PSD") is a launch
  property, achieved by the parse-preserve-round-trip discipline proven on
  IDML.
- **GPU-only execution** (§6, §9). There is exactly one production backend:
  WGSL compute through wgpu. No CPU kernel path ships; no fallback exists.
  Browser = WebGPU (Chrome-only, per platform decision); native ingest
  tooling = the same WGSL through wgpu on Vulkan/Metal/DX12; CI = pinned
  software adapter. What remains on the CPU is what is inherently CPU work:
  codec entropy coding, PSD structural parse/write, CMS transform
  *compilation*, and orchestration.
- **Plugin isolation** (§2). `paged.image` does not touch, patch, fork, or
  link against Paged core internals. Its only contact surface is the
  published plugin SDK (`@paged-media/plugin-api` / `plugin-sdk`) and other
  exposed package contracts. SDK gaps become RFCs against the plugin
  platform — never core modifications from this project.
- **100% tested and verified operations** (§12.2). No operation exists
  outside the registry; no registry row ships below its claimed conformance
  tier; coverage is a CI-computed invariant, not a goal.

**Out of scope** (separate companion specs): layer-tree *editing semantics*
and tools UX (the PSD-faithful layer *data model* is in scope — required for
round-trip), selections-as-UX, brush engine, history UI, plugin end-user API
surface via Boa, editor shell panels, IDML recipe/bake semantics.

### 1.1 Non-goals

- No server-side rendering. Evaluation is client-side, consistent with the
  Paged-wide decision. The same crates compile natively (napi-rs into
  a server-side tool) for optional bulk-ingest tooling only.
- No Emscripten, no C/C++ in the default build. C FFI codec escape hatches
  are native-only, non-default features (§10.3).
- No full libvips parity (~300 ops). Operations are tiered (§11); the long
  tail (mosaicing, FITS/Matlab/CSV, GUI-era ops) is excluded by design.
- No browser fallback paths of any kind (no WebGL, no CPU rendering, no
  nosimd builds).
- **No camera-RAW developing and no HEIC/HEIF ingestion.** `paged.image`'s
  lane is the *editable RGBA / PSD* pipeline — the published-design raster
  surface — not a RAW developer or an HEVC/HEIF container reader. Camera-RAW
  (CR2/CR3/NEF/ARW/DNG/…) is a whole demosaic + per-sensor color-science +
  highlight-recovery subsystem, not a codec, and its decoders are
  non-permissive (libraw is C++ LGPL/CDDL; the pure-Rust rawloader/rawler
  are LGPL-2.1 — off the cargo-deny allow-list). HEIC/HEIF decode needs
  libheif (C, LGPL-3) over an HEVC/AV1 decoder, hitting both the
  no-C-in-default-build rule (§1.1) and the same AV1 wall as AVIF. If a
  *placed* RAW/HEIC asset must be ingested, that is a document-link /
  host-side external-rasterize concern (a native pre-rasterize step handing
  RGBA in), never a codec adapter this crate ships. (Decode/encode of AVIF
  and JPEG XL are separately tracked deferrals, not non-goals — see §10.3
  and `registry/codecs.yaml`.)

---

## 2. Position in the Paged ecosystem: a plugin, not a core subsystem

`paged.image` is delivered as a **Paged plugin bundle** in its own
repository under the `paged-media` org. It is first-party in authorship and
third-party in discipline: it runs under exactly the rules every external
plugin runs under, and it is deliberately the heaviest stress test the
plugin platform has.

```
┌────────────────────────────────────────────────────────────────┐
│ Paged (untouched)                                              │
│  Layer 4  Plugin bundles ──────────────┐                       │
│  Layer 3  React shell                  │  capability-gated     │
│  Layer 2  Boa scripting                │  SDK boundary         │
│  Layer 1  Rust/WASM core (Vello, …)    │  (the ONLY boundary)  │
└────────────────────────────────────────┼───────────────────────┘
                                         │ @paged-media/plugin-api
                                         │ @paged-media/plugin-sdk
                                         ▼
┌────────────────────────────────────────────────────────────────┐
│ paged.image plugin bundle (own repo, own WASM module)          │
│  manifest (capabilities) · Boa-side glue · declarative panels  │
│  image-* crates (§4) compiled to a self-contained WASM module  │
└────────────────────────────────────────────────────────────────┘
```

### 2.1 Isolation contract

*Status note (2026-10-02): this section predates the implementation; see [ADR 460](adr/460-document-is-not-the-store.md) (edits live in the plugin session and leave through save-back, not as document Operations) and, in plugin-sdk, [ADR 315](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/315-isolation-contract.md) (the isolation contract) and [ADR 319](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/319-trust-line.md) (first-party bundles run in-process; the glue here is a TypeScript bundle, not Boa-side script).*

1. **Zero core contact.** No imports from `core/` or `editor/` internals, no
   patched Paged builds, no feature flags inside Paged existing for this
   plugin. CI builds against *published* SDK canaries only; a build that
   reaches around the SDK fails the pipeline.
2. **Capability-gated everything.** The manifest declares all needs
   (document read scopes, asset bytes, panel surfaces, worker spawn, OPFS
   quota, GPU surface). Read-broadly / write-narrowly applies: document
   reads via the standard read capability; all writes are committed
   **Operations submitted through the SDK mutation surface**. No
   back-channel into document state exists.
3. **Own runtime, own memory.** The raster engine is a self-contained WASM
   module with its own heap, worker pool, and GPU resources. One Boa
   sandbox per plugin, per platform rules; the Boa side is thin glue
   (panel logic, op submission, lifecycle) — pixels never cross into Boa.
4. **Gaps become RFCs, not hacks.** Where the current plugin SDK cannot
   express a need, this project files a plugin-platform RFC and waits or
   ships degraded — it does not modify core. Anticipated RFC candidates:
   §2.2.
5. **Renderer boundary.** Final page compositing remains Vello's, reached
   through the image/texture resource contract the SDK exposes. Frame-level
   affine transforms (IDML frame scale/rotate/crop) stay in Paged;
   `paged.image` hands over tiles/textures and recipe results, never draws
   into the page itself.

### 2.2 Required SDK surface (gap analysis → RFCs)

*Status note (2026-10-02): this section predates the implementation; see [ADR 459](adr/459-scene-layer-image-and-tiles.md) for how results reach the page (a scene-layer image plus a tile provider; the shared-GPU-texture row was deferred, see core [ADR 018](https://github.com/paged-media/core/blob/main/docs/adr/018-stage-b-gpu-texture-defer-record-only.md)), and, in plugin-sdk, [ADR 017](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/017-importer-exporter-door-shape.md) (importer/exporter registration) and [ADR 318](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/318-host-spawned-workers.md) (workers). The OPFS tier is a stub, and the viewer-grade bundle described at the end of this section is not built.*

| Need | Likely status in plugin spec | If missing |
|---|---|---|
| Read access to placed-asset bytes + link metadata | covered (read capability) | — |
| Commit Operations / participate in undo via op log | covered (mutation surface) | — |
| Declarative panels + custom canvas region for viewport | panels covered; embedded canvas/texture surface to verify | **RFC: plugin-owned `GPUCanvasContext` or texture-share handle** — the single most consequential row; a per-frame copy across this boundary breaks the §13 interactive budgets |
| Spawn dedicated workers + SharedArrayBuffer within bundle | to verify | RFC: worker capability with COOP/COEP guarantees |
| OPFS quota for swap tier + pyramid cache | to verify | RFC: storage capability with quota declaration |
| Register document/asset type handlers (PSD opens via plugin) | to verify | RFC: importer/exporter registration capability |
| Provide image resources back to Vello (pyramid tiles for placed assets) | to verify | RFC: texture/image resource provider contract |

If `paged.image` can be built without touching core, the plugin platform is
real — that is the dogfooding thesis of this project.

The viewer-grade subset (`image-core` + `image-pipeline` + `image-cms` +
`image-codecs`, read-only recipe playback, PSD `parsed/preserved` only) is
packaged as a viewer-compatible bundle without the mutation surface —
mirroring `EditorSession extends ViewerSession` at the plugin level.

---

## 3. Legal and methodological ground rules

*Status note (2026-10-02): this section predates the implementation; see [ADR 461](adr/461-clean-room-protocol.md) for the protocol as the repository states it, and [ADR 453](adr/453-tolerance-not-golden-bytes.md) for verification (no libvips or GEGL oracle runs in CI; the registry rows name the intended oracle).*

| Rule | Detail |
|---|---|
| **Clean-room** | libvips (LGPL-2.1+), GEGL/babl (LGPL-3+), GIMP (GPL-3+), Photoshop/PSD documentation of any provenance: **oracles and concept references only**. No transliteration of source. Implementation derives from published documentation, academic literature (Mitchell–Netravali, Lanczos, CIE colorimetry, Porter–Duff, W3C compositing & blending), and black-box behavior. |
| **Oracle use** | Native libvips and GEGL run in CI as differential-testing oracles (§12.4). Behavior is not copyrightable; expression is. PSD files carry their own oracle (the embedded merged composite). |
| **Provenance log** | Every kernel and PSD block handler records its specification sources in a `provenance:` block in its registry entry (paper, spec URL, oracle). Mirrors the IDML clean-room content protocol. |
| **Naming** | The streaming-pipeline crate group is product-neutral to maximize OSS adoption beyond Paged. The `vips` crate name on crates.io belongs to FFI bindings — not contested. |
| **License** | See `LICENSE.md`. CLA applies (`CLA.md`). |

### 3.1 Reference mounts: `references/gegl`, `references/libvips`

*Status note (2026-10-02): this section predates the implementation; see [ADR 461](adr/461-clean-room-protocol.md). `references/` is gitignored and never committed; the CI guard and the `Cargo.toml` `exclude` entries described below do not exist.*

The GEGL and libvips source trees are mounted **read-only** at
`references/gegl` and `references/libvips`. They
exist for **reference and inspiration only — their actual code is never
used.** Because the references sit physically next to the implementation, a
naive "everyone reads everything" workflow would weaken the clean-room
claim for the one place it legally matters (kernels and engine code). The
protocol therefore distinguishes two roles:

**Analyst role (may read `references/`):**
- Reads reference sources to understand *behavior, architecture, edge
  cases, parameter semantics, and test expectations*.
- Produces **behavior specifications** as internal design notes (prose, math,
  diagrams, tables of expected outputs) and proposes oracle test cases for
  `image-conformance`.
- Never writes implementation code in the same work session, and never
  pastes, transliterates, or closely paraphrases reference code or comments
  into any artifact. Analyst output contains *facts about behavior*, not
  expression.

**Implementer role (must not read `references/`):**
- Implements kernels, engines, and PSD handlers from: this spec, the
  behavior specs, the public documentation and academic
  literature (§3), and the oracle/differential tests.
- CI guard: implementation work branches fail if a diff touches
  `references/` or if tooling detects reads of `references/` paths during
  implementation work (`CLAUDE.md` states the rule).

**Hygiene:**
- `references/` is excluded from every published artifact: `exclude` in all
  `Cargo.toml` package manifests, `files` whitelist in `package.json`,
  excluded from the plugin bundle build, and excluded from source archives.
  Git history must never merge reference content outside `references/`.
- Architecture and concepts taken from the references (demand-driven
  regions, `GeglBuffer` semantics, babl's format-pair transforms) are ideas
  — not copyrightable — and are adopted freely with provenance notes.
  Expression is what the two-role split protects.
- Every behavior spec derived from reference reading cites which reference
  and what was extracted in its `provenance:` block (§12.3).

---

## 4. Crate architecture

*Status note (2026-10-02): this section predates the implementation; see `architecture.md` for the crates as built, and [ADR 450](adr/450-gpu-only-kernels.md) and [ADR 451](adr/451-one-kernel-definition.md) for the scalar references (the generated ones live in `image-kernels` behind the test-only `reference` feature, which only `image-conformance` enables). The TypeScript import rule is checked by `scripts/check-contract-imports.mjs`.*

Repository `paged-media/plugin-image` (isolation enforced by repo boundary +
CI building against published SDK canaries):

```
plugin-image/
├── manifest/            # plugin manifest, capability declarations, panel schemas
├── glue/                # Boa-side glue: lifecycle, panels, Operation submission via SDK
├── references/          # READ-ONLY: gegl/, libvips/ — §3.1 protocol; excluded from all artifacts
├── image-core/          # types: PixelFormat, Tile, TileMap, Region, ImageDesc
├── image-kernels/       # kernel definitions: WGSL source of truth + param blocks + metadata
├── image-pipeline/      # Engine A: demand-driven streaming evaluation (GPU dispatch)
├── image-graph/         # Engine B: persistent buffer DAG (editor, GPU dispatch)
├── image-gpu/           # wgpu device mgmt, tile residency, dispatch batching, WGSL assembly
├── image-psd/           # PSD/PSB structural parse, layer model, writer (§10.4)
├── image-cms/           # CMS engine per D-11 (qcms/moxcms/hybrid): transform compilation → GPU LUTs
├── image-codecs/        # Source/Target traits + format adapters (CPU by nature)
├── image-conformance/   # TEST-ONLY: scalar reference kernels, oracle harness, corpus
│                        #   runners, perceptual diff, perf gates — never shipped
└── image-js/            # wasm-bindgen surface consumed by glue/ and the viewer bundle
```

**Dependency rules (CI-enforced):**

1. `image-kernels` depends only on `image-core`. Neither engine is visible
   from kernels.
2. The scalar reference implementations live exclusively in
   `image-conformance` (dev-dependency tree); `wasm32` release builds prove
   by `cargo tree` check that no reference code is reachable.
3. **SDK rule:** the only `@paged-media/*` dependencies anywhere are
   `plugin-api`, `plugin-sdk`, and published package contracts —
   dependency-cruiser/cargo-deny fails the build on anything else.

The load-bearing constraint of the whole design: **kernels are pure
functions over tile slices with a frozen ABI** (§6), so Engine A, Engine B,
the conformance harness, and the WGSL assembly all consume one definition.

---

## 5. Core types (`image-core`)

### 5.1 PixelFormat (the babl lesson)

Every buffer, tile, and operation input/output carries an explicit, total
format descriptor. There are no implicit conversions anywhere in the system.

```rust
pub struct PixelFormat {
    pub channels: ChannelLayout,   // RGBA, GrayA, CMYK, CMYKA, Gray
    pub depth: SampleDepth,        // U8 | U16 | F16 | F32
    pub alpha: AlphaMode,          // None | Straight | Premultiplied
    pub transfer: Transfer,        // Linear | Gamma(TransferCurve)
    pub space: ColorSpaceRef,      // interned ICC profile or named space
}
```

- Conversions are **compiled paths** between two `PixelFormat`s, resolved
  once and cached (`image-cms`), never ad-hoc per-op code.
- `ColorSpaceRef` interns ICC profiles by content hash; equality is hash
  equality. Profiles travel with documents and survive serialization.

### 5.2 Working space

*Status note (2026-10-02): this section predates the implementation; see [ADR 455](adr/455-pixel-model.md) (the shipped ingest lane hands the kernels straight, not premultiplied, values with no transfer cast; no `BlendSpace` flag exists) and [ADR 457](adr/457-colour-management.md) (colour transforms run on the CPU at ingest).*

| Path | Format | Rationale |
|---|---|---|
| Production (GPU) | `RGBA · F16 · Premultiplied · Linear · doc-working-space` | `rgba16float` storage textures; premultiplied for correct filtering; linear for correct resampling/convolution |
| Conformance reference (test-only) | `RGBA · F32 · Premultiplied · Linear` | headroom for bit-stable goldens; F16 quantization applied as the final step before diffing against GPU output |
| PSD-compat blending | per-document flag `BlendSpace::Gamma` | Photoshop composites 8/16-bit documents in gamma space; PSD-origin documents default to gamma-space blending; linear is opt-in (§8.4) |

**Working-space policy (v1):** editing happens in an RGB working space —
the linearized form of the document profile (default: linear sRGB). CMYK
sources transform to working space at ingest and back at export;
soft-proofing is a display-chain pass (§10.1). Native CMYK *editing*
(per-ink documents) is deferred to a v2 decision; `PixelFormat`
already expresses it, so nothing in the core forecloses it.

### 5.3 Tiles

```rust
pub const TILE: u32 = 256;            // D-1

pub struct TileCoord { pub level: u8, pub x: i32, pub y: i32 }  // mip level + grid

pub struct Tile {
    pub format: PixelFormat,
    pub data: TileData,               // Gpu(TextureSlot) | Heap(Arc<[u8]>) | Swapped(OpfsKey)
    pub generation: u64,              // monotone; drives cache invalidation
}
```

- `TileMap`: sparse `HashMap<TileCoord, Arc<Tile>>` with copy-on-write via
  `Arc::make_mut`. Unallocated tiles read as a constant — sparse canvases
  cost nothing.
- Tiles are **mip-aware**: `level > 0` stores downsampled content. Engine B
  evaluates at the viewport's mip level (§8.3); Engine A uses levels for
  shrink-on-load.
- Per-tile `generation` integrates with salsa-style dependency tracking:
  a committed Operation bumps generations of touched tiles; downstream
  caches key on `(node_id, params_hash, input tile generations)`.

### 5.4 Regions

```rust
pub struct Region { pub x: i32, pub y: i32, pub w: u32, pub h: u32 }
```

All engine traffic is in regions (request ROI / validity); kernels never see
whole images.

---

## 6. Kernel model (`image-kernels`) — WGSL is the implementation

*Status note (2026-10-02): this section predates the implementation; see [ADR 450](adr/450-gpu-only-kernels.md) (GPU-only execution), [ADR 451](adr/451-one-kernel-definition.md) (one definition feeds both lanes), [ADR 452](adr/452-kernel-abi.md) (the ABI as amended to v1.1; `KernelDef` gained `inputs` and `module`) and [ADR 453](adr/453-tolerance-not-golden-bytes.md) (GPU output is checked against the scalar reference by the Rust harness in `image-conformance` where a GPU adapter is present; CI has none, there is no Playwright harness, and no libvips or GEGL oracle runner exists).*

A kernel is **a WGSL compute function plus a typed param block plus
metadata**. The WGSL is the production implementation; the scalar Rust twin
exists only in `image-conformance` as the golden source.

### 6.1 Kernel definition

```rust
pub struct KernelDef {
    pub id: &'static str,             // registry id: "point.curves", "conv.gaussian", …
    pub class: KernelClass,
    pub params: ParamsLayout,         // #[repr(C)] Pod + Hash + Serialize → WGSL uniform block 1:1
    pub wgsl: &'static str,           // entry point conforming to the fixed ABI (§9.2)
    pub mip_exact: bool,              // safe to evaluate at mip levels with scaled params?
    pub gpu_tolerance: Tolerance,     // perceptual/channel epsilon vs reference (§12.4)
}

pub enum KernelClass {
    Point,                                  // out(x,y) = f(in₀(x,y), …)
    Windowed { radius: (u16, u16) },        // needs expanded input window
    Resample { support: f32 },              // rational scale, kernel support
    Reduction(ReductionKind),               // min/max/avg/histogram → scalar/table
    Generator,                              // no inputs (gradients, noise, constants)
}
```

Properties:

- **Engine-agnostic.** Kernels know nothing about pull evaluation, dirty
  propagation, scheduling, or residency. Window expansion and tiling are
  engine concerns driven by `KernelClass` metadata.
- **Selection-ready from day one.** Every kernel ABI slot includes an
  optional grayscale mask; pointwise kernels apply
  `out = mix(in, f(in), mask)`, windowed kernels mask at write. Designed in
  now because retrofitting selections is what kills editor architectures.
  Engine A always binds the constant-1 mask.
- **Param blocks are Pod.** One `#[repr(C)]` definition is simultaneously
  the cache key (stable hash), the Operation payload (op log), and the WGSL
  uniform block. One definition, three uses.

### 6.2 Family codegen

The arithmetic/relational/boolean/linear/cast/band families (~half of the
libvips operation count) are produced by a `kernel_family!` macro expanding
`(operation × depth × channel-layout)` from one scalar expression per
operation. The macro emits **both** the WGSL body and the conformance-side
scalar reference from the same expression — one source of truth, T0 becomes
one meta-task.

### 6.3 Determinism strategy without a CPU production path

GPU floating point is not bit-stable across drivers and adapters. The
strategy:

- **Goldens come from the scalar reference** (test-only, `f32`, no
  fast-math, fixed reduction order, documented rounding per depth
  conversion) — bit-stable across platforms by construction.
- **GPU output is verified against goldens by tolerance**, declared per
  kernel (`gpu_tolerance`), via perceptual/channel diff. GPU output is
  never byte-golden-tested.
- **CI runs a pinned software adapter** (SwiftShader / Lavapipe, version-
  pinned in the CI image) so GPU-path runs are reproducible run-to-run;
  hardware-adapter runs happen on a scheduled matrix job to catch driver
  divergence early (results informational, not merge-blocking — D-10).

### 6.4 Definition of done (per kernel)

A kernel does not merge without **all** of:

1. WGSL implementation under the fixed ABI (§9.2)
2. Scalar reference in `image-conformance` (emitted by codegen for T0;
   handwritten for T1+)
3. `parity(ref↔oracle)` green where an oracle exists (libvips / GEGL / PSD
   composite, §12.4)
4. `parity(gpu↔ref)` green within declared tolerance on the pinned CI
   adapter, via the standard Playwright/WebGPU harness
5. Registry entry complete: class, oracle, `mip_exact`, tolerance,
   provenance, tier status (§12.3)

Removing the CPU production path (v0.4) deletes the former dual-backend tax:
SIMD optimization work, `pulp`/`multiversion` dispatch, and wasm `simd128`
kernel tuning are all gone. The scalar reference is written for clarity, not
speed.

---

## 7. Engine A — streaming pipeline (`image-pipeline`)

*Status note (2026-10-02): this section predates the implementation; see [ADR 454](adr/454-two-evaluation-engines.md). Decode workers are spawned through the host's worker door (plugin-sdk [ADR 318](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/318-host-spawned-workers.md)), not `wasm-bindgen-rayon`; the `to_pyramid` sink and the native ingest tool are not built.*

The libvips-shaped engine: bounded memory, one pass, source → sink.

### 7.1 Model

- A pipeline is a lazy DAG of `OpNode`s built by a fluent API; nothing
  executes until a sink pulls.
- Evaluation is **demand-driven by region**: the sink partitions output into
  work units (tile strips); ROIs propagate upstream, expanded per node by
  `KernelClass` window metadata; leaves answer from decode streams.
- **Division of labor:** codec decode/encode and PSD structural work run on
  the CPU worker pool (`wasm-bindgen-rayon` over SharedArrayBuffer in the
  browser; rayon natively) because entropy coding is inherently serial-ish
  CPU work. Every kernel stage (resample, CMS application, convolution,
  compose) is a GPU dispatch over the strip's tiles. The scheduler overlaps
  decode (CPU workers) with kernel execution (GPU queue) — the pipeline is
  a producer/consumer bridge between the two.
- **Operation cache:** memoizes `(op id, params hash, input content hashes)`
  → materialized regions, LRU-bounded. Aligned with salsa semantics so
  pipeline results participate in document-level dependency tracking
  ("asset pyramid depends on source bytes + ICC profile + recipe").
- **Native target:** identical code; wgpu selects Vulkan/Metal/DX12. The
  napi-rs ingest tool is therefore GPU-accelerated too —
  one backend everywhere.

### 7.2 Shrink-on-load

`thumbnail`-class planning is a pipeline feature, not an op: codec adapters
advertise native downscale capabilities (JPEG DCT scaling 1/2·1/4·1/8,
pyramid-TIFF level selection); the planner pushes the largest safe shrink
into the decoder and finishes with a `Resample` kernel (Lanczos3/Mitchell)
on the GPU for the residual factor. This single planning rule is the bulk of
libvips' famous thumbnail performance and the headline benchmark (§13).

### 7.3 Sinks (v1)

- `to_buffer(PixelFormat)` — materialize a region into a `TileMap`
  (feeds Engine B and the SDK texture-provider contract).
- `to_pyramid(store)` — write the mip pyramid (premultiplied, post-CMS);
  persisted via OPFS locally and server storage as a derived
  artifact.
- `to_encoder(format, options)` — GPU readback strip-by-strip into a codec
  encoder (export bake); the only structured readback path in the system.

---

## 8. Engine B — buffer graph (`image-graph`)

*Status note (2026-10-02): this section predates the implementation; see [ADR 454](adr/454-two-evaluation-engines.md) (the shipped editing surface is a layer stack folded through the `compose.*` kernels, with Engine B supplying the undo journal; no per-document `BlendSpace` exists) and [ADR 460](adr/460-document-is-not-the-store.md) (edits and their undo live in the plugin session, not in the document's operation log as §8.5 describes).*

The GEGL-shaped engine: persistent mutable state, incremental
re-evaluation, sub-frame interactive updates.

### 8.1 Model

```
SourceNode(PersistentBuffer) ─┐
SourceNode(PersistentBuffer) ─┼─► OpNode(kernel, params) ─► … ─► SinkNode(viewport)
GeneratorNode(gradient…)     ─┘
```

- `PersistentBuffer` = `TileMap` + `PixelFormat` + residency metadata — the
  Rust expression of `GeglBuffer`: sparse, tiled, mip-chained, swappable.
- `OpNode`s carry a kernel id + param block and an **output cache**
  (per-mip-level `TileMap` of computed tiles tagged with input generations).
- `SinkNode`s bind to display: the viewport sink owns the set of
  `(level, coord)` tiles currently visible and requests exactly those.

### 8.2 Invalidation and evaluation

1. A param change or buffer write computes a **damage region** (`Windowed`
   kernels inflate damage by radius; pointwise damage is identity).
2. Damage propagates downstream, invalidating cached tiles whose input
   generations no longer match.
3. The viewport sink re-requests its visible tile set; only invalid tiles
   recompute, upstream-first over the minimal subgraph, coalesced into
   batched GPU dispatches per node.

Budget: a pointwise param change over a 4K viewport touches ≤ ~140 visible
256² tiles at level 0 — one batched dispatch, well inside a frame (§13).

### 8.3 Mip-aware evaluation

Zoomed-out viewing evaluates the graph **at the viewing mip level**, not at
level 0 followed by downscale. Windowed kernels declare per-level parameter
scaling (gaussian σ halves per level); kernels where mip-space evaluation
diverges visibly declare `mip_exact: false` and force level-0 evaluation
with downsampling. The declaration is conformance-tested (§12.4,
`incremental-correct` includes mip equivalence checks for `mip_exact`
kernels).

### 8.4 Compositing semantics

Layer compositing is a kernel family (`compose.*`) executed like any other
node — the layer tree (companion spec) lowers to a graph. Commitments made
*now* because they shape kernels:

- Porter–Duff `over` on premultiplied data is the spine.
- The Photoshop blend-mode set is specified kernel-by-kernel against dual
  oracles (GEGL + PSD embedded composites), including the non-separable
  four (Hue/Saturation/Color/Luminosity) and the folklore formulas
  (Hard Mix, Linear Burn at varying opacities).
- `BlendSpace::{Linear, Gamma}` is per-document; both paths are first-class
  and conformance-tested. PSD-origin documents default to `Gamma`.

### 8.5 Mutation: Operations and Gestures

Direct mapping onto the Paged mutation model, expressed through the SDK
mutation surface (§2.1.2):

| Paged concept | image-graph realization |
|---|---|
| **Gesture** (ephemeral) | A param override on one `OpNode`, applied to the live graph only; recompute of visible invalid tiles; never serialized; replaced at high frequency (slider drag, filter preview) |
| **Operation** (committed) | `SetParams(node, block)` · `EditGraph(insert/remove/rewire)` · `WriteBuffer(buffer, sparse tile-delta payload)` — submitted through the SDK, appended to the document op log |
| **Undo** | Inverse op from the log; `WriteBuffer` undo restores COW-journaled `Arc<Tile>` snapshots — O(changed tiles), never O(canvas) |

A brush stroke (companion spec) is a stream of Gestures rendering into a
scratch tile set, committing on pointer-up as one `WriteBuffer` Operation
with the sparse delta as payload.

---

## 9. GPU execution layer (`image-gpu`) — the only execution layer

*Status note (2026-10-02): this section predates the implementation; see [ADR 450](adr/450-gpu-only-kernels.md) and [ADR 452](adr/452-kernel-abi.md) (the ABI, amended to v1.1 for handwritten modules). Tier 2 (OPFS) is a stub. Of the adapters in §9.3, neither CI lane nor the native ingest tool is set up; GPU tests run where a local adapter is present.*

### 9.1 Residency: three tiers

```
Tier 0  GPU texture pool   rgba16float texture arrays, TILE² slots, LRU
Tier 1  wasm heap          Arc<[u8]> tile bytes (also the COW/undo tier)
Tier 2  OPFS scratch       evicted cold tiles; sync access from workers
```

GPU texture memory lives **outside the wasm32 4 GB heap** — the tile pool
is deliberately the largest tier. Eviction: Tier 0 → Tier 1 on LRU
pressure; Tier 1 → Tier 2 by background sweeper. This is the modern
reincarnation of the Photoshop scratch disk and the mechanism that makes
100-megapixel × 30-layer documents feasible in a tab. Budget arbitration
with Vello's image resources happens through whatever resource contract the
SDK exposes (§2.2) — never through shared internals.

### 9.2 The WGSL ABI (frozen in M0 phase 0)

```
@group(0): input tile textures (arity per KernelClass)
@group(1): params uniform block   (repr(C) layout shared with Rust)
@group(2): selection mask texture (constant-1 default)
@group(3): output storage texture
workgroup: 16×16 over the tile grid
```

- T0/family kernels: WGSL emitted by `kernel_family!` (§6.2).
- Windowed/resample kernels: handwritten WGSL, separable passes where
  applicable (gaussian and Lanczos as two 1-D passes), validated against
  the scalar reference.
- Dispatch batching: all invalid tiles of one node coalesce into a single
  dispatch per pass; ping-pong between pool slots; readback only on
  explicit `to_cpu()` (export, undo journaling).

### 9.3 Adapters

| Context | Adapter | Notes |
|---|---|---|
| Browser (production) | WebGPU, Chrome-only | platform decision; no fallback of any kind |
| Native ingest (server-side tooling) | wgpu → Vulkan/Metal/DX12 | same WGSL, same results within tolerance |
| CI (merge-gating) | pinned SwiftShader/Lavapipe | reproducible run-to-run; version pinned in CI image |
| CI (scheduled matrix) | real adapters (NVIDIA/AMD/Intel/Apple) | informational driver-divergence watch (D-10) |

---

## 10. Color management, codecs, PSD

### 10.1 CMS (`image-cms`) — audit-gated engine choice

*Status note (2026-10-02): this section predates the implementation; see [ADR 457](adr/457-colour-management.md). The engine choice was made: the hybrid, qcms for display and moxcms for the print lane, behind one `CmsEngine` trait. "Apply on GPU" is not wired: no `cms.apply` kernel exists, and transforms run on the CPU at ingest (see [design/colour-management-at-ingest.md](design/colour-management-at-ingest.md)).*

**Status: open pending A-0 / D-11.** Paged core currently uses **qcms**
(FirefoxGraphics) for color management. The plugin's isolation contract
means it *may* choose a different CMS without touching core — but two
constraints govern the choice, and the A-0 existing-systems audit (M0 phase 0) decides
it with data rather than assumption:

1. **Cross-engine consistency is a conformance requirement.** A placed
   asset is color-managed by core (qcms, Vello render path) when displayed
   by Paged, and by `image-cms` when it flows through this plugin's
   pipeline or editor. If the two disagree visibly, users see color shifts
   between "the page" and "the image editor" — unacceptable for a
   print-grade tool. Registry feature `image.cms.core-consistency.*`:
   identical (image, source profile, destination profile, intent) through
   both engines must agree within declared ΔE tolerance, tested over the
   profile corpus. This requirement holds **whatever** engine is chosen.
2. **CMYK + v4 capability is non-negotiable for this plugin.** The audit
   must establish exactly what core's qcms build supports today (CMYK
   input/output? ICC v4 LUT pipelines? float transforms? rendering
   intents/BPC?). qcms is display-RGB-focused by heritage; print-grade
   CMYK ingest/soft-proof/export likely exceeds it.

**Decision matrix (D-11), resolved by the audit:**

| Option | Pros | Cons |
|---|---|---|
| qcms throughout (match core) | trivially consistent with core; one engine in the ecosystem | likely insufficient CMYK/v4/intent coverage; upstream is Firefox-driven |
| moxcms throughout | pure Rust, SIMD, CMYK-capable, intents/BPC | consistency with core's qcms must be *proven* per profile class (the §10.1.1 test); two engines in the ecosystem |
| Hybrid: qcms-equivalent display path + moxcms for CMYK ingest/proof/export | consistency where the eye compares; capability where print needs it | two code paths to conformance-test; boundary discipline needed |

Whatever the outcome: **compile on CPU, apply on GPU** stands. The chosen
engine builds the transform; GPU-path application bakes it into a 3D LUT
texture (shaper curves as 1D LUTs) sampled by a `cms.apply` kernel. Exact
(non-LUT) transforms run on CPU only where byte-exactness is contractual:
export encode and conformance goldens. LUT-vs-exact precision is
conformance-tested per profile class with declared ΔE tolerance. lcms2 FFI
remains a native CI differential oracle only, never shipped.

Display chain: working space → soft-proof (optional; CMYK target profile,
rendering intent + black-point compensation) → display profile — final GPU
passes on viewport tiles.

### 10.2 Codec adapter contract (`image-codecs`)

```rust
pub trait ImageSource {            // streaming decode (CPU workers)
    fn probe(&mut self) -> Result<SourceInfo>;          // dims, format, ICC, exif, mips
    fn native_shrink(&self) -> &[u32];                   // e.g. [1,2,4,8] for JPEG
    fn read_region(&mut self, roi: Region, shrink: u32, out: &mut TileSliceMut) -> Result<()>;
}
pub trait ImageTarget {            // streaming encode (CPU workers)
    fn begin(&mut self, info: TargetInfo) -> Result<()>;
    fn write_strip(&mut self, region: Region, data: &TileSliceRef) -> Result<()>;
    fn finish(&mut self) -> Result<EncodedStats>;
}
```

Sources/targets are sans-IO over a `ByteSource` abstraction (memory, OPFS,
JS `ReadableStream`, native file) so the same adapters serve browser and
native builds. Codecs are inherently CPU work and remain so — "GPU-only"
(§1) refers to kernel execution, not entropy coding.

### 10.3 Format matrix (v1 commitment)

*Status note (2026-10-02): this section predates the implementation; see [ADR 456](adr/456-codecs.md) and `registry/codecs.yaml`. The PNG adapter is `zune-png`. JPEG, PNG and PSD/PSB are built; TIFF, WebP, GIF and PDF/AI rasterization are not.*

| Format | Decode | Encode | Crate | Risk / note |
|---|---|---|---|---|
| JPEG (incl. CMYK/YCCK) | ✅ | ✅ | `zune-jpeg` / `jpeg-encoder` | low; APP14 Adobe handling conformance-tested |
| PNG | ✅ | ✅ | `zune-png` or `image-rs/png` (D-4) | low |
| TIFF (print profile: 16-bit, CMYK, JPEG-in-TIFF, clipping paths) | ✅ | ✅ | `image-tiff` + upstream patches | **highest-risk dependency**; `libtiff-sys` FFI behind native-only `codec-libtiff` feature as escape hatch |
| WebP | ✅ | lossless ✅ / lossy ⚠ | `image-webp` | lossy-encode gap acceptable v1 (export prefers JPEG/PNG/AVIF) |
| AVIF | ⏸ | ⏸ | `rav1d` (dec) / `ravif`+`rav1e` (enc) | **decode blocked** — rav1d is still C-API-only (2026-06-13), no permissive pure-Rust *library* AV1 decoder passes cargo-deny; **encode** is license-clear (ravif BSD-3 / rav1e BSD-2) but deferred on the 8 MiB wasm budget + no round-trip oracle. See `registry/codecs.yaml`. |
| PSD/PSB | ✅ full structural | ✅ round-trip-safe | `image-psd` (§10.4) | constitutive from M0; fidelity ladder per feature, lossless preservation always |
| PDF/AI (placed) | rasterize ✅ | — | `hayro` (Vello-ecosystem PDF) | evaluate vs need; PDFium rejected (Emscripten) |
| GIF | ✅ | ✅ | `image-rs/gif` + quantizer | low priority |
| JPEG XL | ⏸ | ⏸ | `jxl-oxide` (decode) | v2. Decode is now license-clear (`jxl-oxide`, MIT OR Apache-2.0, pure-Rust) — deferred on v2 scope + the wasm budget. **Encode blocked-on-ecosystem**: the only pure-Rust JXL encoder (`jxl-encoder`) is AGPL-3.0/commercial → fails cargo-deny; libjxl is C++. See `registry/codecs.yaml`. |

Corpus rule: before M1 freeze, run the format/feature frequency audit over
the project's IDML corpus `Links`; this matrix is adjusted by data, not taste.

### 10.4 PSD core (`image-psd`) — round-trip first

*Status note (2026-10-02): this section predates the implementation; see [ADR 458](adr/458-psd-preservation.md). The ecosystem oracle that exists is `psd-tools`, run locally on request (`image-conformance/tests/psd_ecosystem.rs`); it does not run in CI, and there is no GIMP lane.*

PSD support is engineered the way IDML was: **full structural parse +
lossless preservation + byte-faithful re-write first; semantic fidelity
second, feature by feature, tracked in the registry.** The honest claim at
every stage: "Paged never destroys a PSD." The conformance matrix shows
publicly which features it also renders and edits.

**Data model.** `image-psd` defines the PSD-faithful layer tree as plain
Rust data (no engine dependencies): header/color-mode data; layer records
(groups, clipping, blend mode, opacity, fill, flags); channel image data
(RAW/RLE/ZIP per depth); layer masks (raster + vector); additional layer
information blocks (`lsct`, `luni`, `lyid`, `lfx2`/`lmfx`, `SoCo`-family
fills, adjustment-layer parameter blocks, `TySh`/`Txt2` text records,
`SoLd`/`lnkD` smart objects, `vmsk`/`vsms` paths); global sections (image
resources incl. ICC, resolution, guides, slices; merged composite). This
model is the seed of the editor's layer-tree document model — the companion
editing-semantics spec *lowers from* this structure.

**Preservation invariant.** Every block the parser does not semantically
understand is retained as an opaque, position-independent byte payload
attached to its owner node and re-emitted verbatim on write. Understood
blocks re-encode from the model. Consequence: opening and saving a PSD
without touching an unmodeled feature is lossless *by construction* —
including Adobe-private and undocumented blocks. The writer maintains the
merged composite and section lengths so files stay valid for strict readers.

**Fidelity ladder** (registry namespace `image.psd.*`, adapted tiers):
`parsed → preserved → rendered → mutatable → round-trips`. Examples:
`image.psd.layer.blendmode.multiply` reaches `rendered` when the
`compose.*` kernel holds parity against reference composites;
`image.psd.text` may sit at `preserved` long-term (the `TySh`/`Txt2` text
engine is the deepest reverse-engineering rabbit hole in the format)
while remaining round-trip-safe; `image.psd.smartobject` reaches `rendered`
early (embedded payloads decode through `image-codecs`) and `mutatable`
only with the editing-semantics spec.

**Write semantics under editing.** When a document *is* edited, the writer
re-encodes the touched subtree from the model, refreshes the merged
composite via Engine A (flatten pipeline through `compose.*` kernels,
gamma/linear per document `BlendSpace`), recomputes RLE/ZIP channel data,
and preserves everything untouched. PSB (>30k px) shares the model with
widened offsets — one writer, two containers.

**Oracles.** Three-way validation: (1) byte-level — parse→write with zero
edits diffs structurally against the original; (2) render-level — flattened
output perceptually diffed against the file's **embedded merged composite**
(every PSD ships its own oracle); (3) ecosystem — round-tripped files
re-opened in CI by reference implementations (GIMP 3.x headless,
`psd-tools`), plus periodic manual Photoshop verification. A dedicated PSD
corpus (donated + synthesized + property-generated) feeds the fingerprinted
bug pipeline like the IDML corpus does.

---

## 11. Operation inventory and tiering

*Status note (2026-10-02): this section predates the implementation; see `registry/kernels.yaml` for the kernels that exist. `cms.apply` and the frequency (FFT) operations are not among them.*

Operations live in the registry as `image.*` features (§12.3). Tiers:

| Tier | Content | Schedule |
|---|---|---|
| **T0 — generated families** | arithmetic, relational, boolean, linear, casts, band ops, min/max/clamp | codegen; M0 |
| **T1 — crown jewels** | resample (`lanczos3`, `mitchell`, `cubic`, `nearest`, shrink planner), colour (`cms.apply`, premultiply, transfer), convolution (gaussian, unsharp, generic conv, separable planner), `compose.*` blend modes (required by the PSD merged-composite writer) | M1 |
| **T2 — editor-bearing** | curves, levels, HSL, exposure/WB, crop/flip/rotate90, gradient/noise generators, histogram/statistics reductions | M2 |
| **T3 — breadth** | morphology, median/rank, frequency (`rustfft` reference / GPU FFT), draw ops, distance transform | M3+ |
| **T∞ — never** | mosaicing/astronomy, FITS/Matlab/CSV/Analyze, ImageMagick delegation, GUI-era leftovers | excluded by design |

---

## 12. Conformance, testing, and the verification invariant

*Status note (2026-10-02): this section predates the implementation; see [ADR 453](adr/453-tolerance-not-golden-bytes.md) for how output is verified and plugin-sdk [ADR 317](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/317-registry-driven-dispatch.md) for registry-driven dispatch. The build-consumed registry is `registry/*.yaml` in this repo, and the test harness is the Rust crate `image-conformance`. This repo has no Playwright suite and no `#[feature_test]` macro; no libvips or GEGL oracle runner exists; CI runs the GPU parity tests without an adapter, so they skip.*

### 12.1 Identical environment to Paged core

`plugin-image` adopts the project's shared testing environment **without
deviation**:

- **Playwright is the sole browser-side runner**, Chrome WebGPU the only
  browser target; GPU conformance runs are ordinary Playwright suites.
- Tests carry `@feat:<id>` tags (Playwright) and `#[feature_test("id")]`
  (Rust) exactly as in the core repos.
- CI publishes `paged-results.json` to the project's feature registry;
  rows render on its status dashboard.
- The public subset feeds `conformance.public.json`, consumed by
  `<ConformanceMatrix>` MDX components on `docs.paged.media`.
- The fingerprinted GitHub-issue bug pipeline (auto-file on regression,
  auto-close on recovery) applies unchanged.
- The repo's `CLAUDE.md` carries this spec's invariants (clean-room/provenance §3, isolation
  contract §2.1, definition of done §6.4).

### 12.2 The 100% verification invariant

**Every operation is registry-listed; every registry row is tested at its
claimed tier; nothing else is reachable.** Enforced three ways:

1. **Registry-driven dispatch.** Kernel and PSD-block registration is
   generated *from* the registry YAML at build time. An implementation
   without a registry row does not get a dispatch entry — unregistered
   operations are unreachable by construction, not by policy.
2. **Coverage gate.** CI computes, per registry row, the presence of tests
   for every tier the row claims (`implemented` → unit;
   `parity(ref↔oracle)` → oracle diff; `parity(gpu↔ref)` → Playwright GPU
   diff; `incremental-correct` → invalidation property test; PSD tiers per
   §10.4). Coverage below 100% of claimed tiers fails the build — the
   invariant is a CI verdict, never a review judgment.
3. **Tier-regression gate.** A row's status can only move forward via a
   green run; a previously green tier turning red auto-files a
   fingerprinted issue and blocks merge to the release branch.

### 12.3 Registry integration

To be added to the project's feature registry alongside this spec's adoption:

- `registry/features/image.*.yaml` — namespaces: `image.kernel.*`,
  `image.pipeline.*`, `image.graph.*`, `image.codec.*`, `image.cms.*`,
  `image.psd.*`, `image.plugin.*` (SDK-integration features), each with
  dot-separated ids per standing convention.
- `plugin-image` CI gains the standard results-publishing job;
  the registry ingests its `paged-results.json` like any core repo.
- Conformance taxonomy mapping registered in the registry's docs:
  - kernels: `implemented → parity-ref-oracle → parity-gpu-ref → incremental-correct`
  - PSD: `parsed → preserved → rendered → mutatable → round-trips`
  - codecs/pipeline/plugin: the standard `parsed → rendered → mutatable →
    round-trips` ladder where applicable.

Sample rows:

```yaml
# registry/features/image.kernel.conv.yaml
id: image.kernel.conv.gaussian
title: Gaussian blur (separable, mip-scaled)
class: windowed
oracle: both            # libvips + GEGL
mip_exact: true
gpu_tolerance: { kind: channel_eps_f16, value: 2 }
status: planned
provenance:
  - "separable convolution: standard literature"
  - "sigma-per-mip scaling: derived, documented in a behavior spec"
tests:
  rust: ["image-conformance/tests/conv_gaussian.rs"]
  playwright: ["@feat:image.kernel.conv.gaussian"]
```

```yaml
# registry/features/image.psd.yaml
id: image.psd.layer.mask.raster
title: PSD raster layer masks
tier: preserved          # current; target: round-trips
oracle: embedded_composite
status: in-progress
provenance:
  - "Adobe PSD file format specification (public)"
  - "black-box corpus observation, recorded in a behavior spec"
tests:
  rust: ["image-conformance/tests/psd_blocks.rs::raster_mask"]
  corpus: ["psd-corpus/masks/*"]
```

### 12.4 Oracles and harnesses (in `image-conformance`, test-only)

- **Scalar reference** (golden source): per-kernel `f32` implementations,
  bit-stable per §6.3; emitted by codegen for T0, handwritten for T1+.
- **libvips** (native CI container): oracle for T0/T1 pipeline ops and the
  thumbnail planner.
- **GEGL** (native CI container): oracle for compositing/blend/filter ops.
- **PSD embedded composites + ecosystem readers** (§10.4).
- **Dual-oracle disagreement protocol:** where vips and GEGL disagree (edge
  handling, rounding, premultiply conventions), the registry row records
  both behaviors, the chosen convention, and the rationale. Disagreement is
  signal, not failure.
- **Differential harness:** per-kernel property tests (random tiles ×
  random valid params × all depths) against reference and oracle; corpus
  pipelines (thumbnail, CMYK→sRGB, sharpen-export) diffed end-to-end.
- **Incremental-correctness (Engine B):** random graph × random damage
  sequence → incremental result must equal from-scratch evaluation
  (reference path bit-exact; GPU path within tolerance). Engine B's failure
  mode is stale tiles, not wrong math; this tier is non-negotiable for
  every kernel class, and includes mip-equivalence checks for `mip_exact`
  kernels.

### 12.5 Dashboard

All `image.*` rows publish to the project's status dashboard; the public parity matrix
versus libvips — and the PSD round-trip guarantee matrix — are themselves
launch marketing for the standalone crates.

---

## 13. Performance gates (CI-enforced on the pinned adapter; informational on hardware matrix)

*Status note (2026-10-02): this section predates the implementation; see `status.md`. These gates are not built: no test or CI step enforces these budgets, and the one benchmark target in the repo is the PNG codec comparison in `image-conformance/benches/png_d4.rs`.*

| Benchmark | Target |
|---|---|
| `thumbnail` 8000×6000 JPEG → 256px (wasm, 4 workers + GPU) | ≥ wasm-vips ×2; within ×1.5 of native sharp |
| Pointwise gesture, 4K viewport, level-0 tiles | < 8 ms wall per update |
| Gaussian σ=5 full-viewport recompute | < 16 ms |
| 100 MP / 30-node graph, pan at level 2 | no level-0 evaluation triggered; < 4 ms residency misses per frame |
| Undo of 500-tile `WriteBuffer` | < 10 ms |
| PSD zero-edit round-trip, 500 MB PSB | streaming, peak heap < 512 MB |

Note: the interactive budgets assume the §2.2 GPU-surface contract avoids
per-frame copies across the plugin boundary — this is the single most
consequential SDK RFC.
