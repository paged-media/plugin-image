# CLAUDE.md — paged-media/plugin-image

Orientation for Claude sessions in **paged-media/plugin-image** — the
paged.image raster subsystem, delivered as a Paged plugin (private repo,
And The Next GmbH).

## What this is

A Rust/WASM/WebGPU image-processing engine in two shapes — **Engine A**
(libvips-class demand-driven streaming pipeline, `image-pipeline`) and
**Engine B** (GEGL-class persistent tiled buffer graph, `image-graph`) —
plus PSD/PSB round-trip (`image-psd`), GPU-only WGSL kernels
(`image-kernels`/`image-gpu`), CMS (`image-cms`), codec adapters
(`image-codecs`), and a TEST-ONLY conformance harness
(`image-conformance`). Shipped as a plugin bundle (`manifest/` + `glue/`)
consuming ONLY the published plugin SDK.

Spec (the authority): `thoughts/docs/paged/plugin-image/base-idea.md`.
A-0 audit + D-11 ruling: `thoughts/docs/paged/plugin-image/a0-audit.md`.
SDK gap tracker: the cross-repo RFI `thoughts/docs/paged/plugin-platform/rfi-core-sdk-gaps.md` (I-NN ids in §6; per-plugin BREAKAGE_LOG retired 2026-06-12).

## Project State & Feature Matrix (paged-media/state)

The canonical feature inventory, test linkage, and live status for ALL
Paged repos live in `paged-media/state` — dashboard:
https://state.paged.media, summary: `state/STATUS.md`. There is NO
feature matrix in this repo; do not create one.

Rules for every code change in this repo:

1. NEW CAPABILITY → registry entry. If your change adds or completes a
   feature, add/update its entry in `state/registry/features/*.yaml`
   (separate PR to paged-media/state, reference it from this PR).
   Feature IDs are immutable.
2. EVERY NEW TEST → feature tag. Playwright: `{ tag: ['@feat:<id>'] }`.
   Rust: `#[feature_test("<id>")]`. Untagged new tests fail CI.
   (Until the macro ships from state, use the naming convention
   `fn <feature_id_with_underscores>_…()` and the registry row's
   `tests:` pointer in `registry/*.yaml` here.)
3. STATUS CHANGE → registry, not prose. "X is now shipped/partial" is
   expressed by editing the registry entry, never by writing it into
   READMEs or docs.
4. NEVER hand-edit generated files: `state/data/*.json`,
   `state/STATUS.md`, docs conformance tables. They are overwritten on
   every generation.
5. BEFORE claiming a feature done: check its row on the dashboard (or
   run `/matrix`) — done means status reflects reality AND linked tests
   are green.
6. FOUND A BUG while working? If a test exposes it, just let it fail and
   push — the bug reporter files the issue. For untestable findings,
   open an issue with label `state-bug` + `feat:<id>`.

## Hard rules (this repo's constitution — spec §2/§3/§6)

- **CLEAN-ROOM / TWO-ROLE (§3.1).** `references/gegl` + `references/libvips`
  are read-only inspiration mounts. ANALYST agents may read `references/`
  and produce *behavior specs* into `thoughts/` (facts about behavior,
  never expression). IMPLEMENTER agents — everyone writing kernel,
  engine, PSD, or codec code — **MUST NOT read `references/`**, ever.
  Never paste, transliterate, or closely paraphrase reference code or
  comments into any artifact. Implementation derives from the spec, the
  behavior specs in `thoughts/`, public documentation and academic
  literature, and the oracle tests.
- **ISOLATION CONTRACT (§2.1).** Zero core contact. No imports from
  `core/` or `editor/` internals; the only `@paged-media/*` dependencies
  are `plugin-api`, `plugin-sdk`, and published package contracts
  (TS: `scripts/check-contract-imports.mjs`; Rust: `deny.toml` sources +
  the cargo-tree CI guards). SDK gaps become RFI §6 entries /
  plugin-platform RFCs — NEVER core modifications from this project.
- **GPU-ONLY EXECUTION (§6/§9).** One production backend: WGSL compute
  via wgpu. No CPU kernel path ships. The scalar Rust reference twins are
  TEST-ONLY (`image-kernels` feature `reference`, enabled solely by
  `image-conformance`'s dev-dependency); a wasm32 release build must
  prove by `cargo tree` that no reference code is reachable. What stays
  on CPU is inherently CPU work: codec entropy coding, PSD structural
  parse/write, CMS transform *compilation*, orchestration.
- **PRESERVATION INVARIANT (§10.4).** "Paged never destroys a PSD."
  Every unmodeled block is retained as opaque bytes attached to its
  owner node and re-emitted verbatim; zero-edit round-trip is
  **byte-identical** (the lazy-verbatim guard: unmodified typed nodes
  also re-emit their original source bytes).
- **PROVENANCE DISCIPLINE (§3).** Every kernel and PSD block handler
  records its specification sources in its row in `registry/*.yaml`
  here (paper, spec URL, oracle). No row, no dispatch.
- **DEFINITION OF DONE per kernel (§6.4).** WGSL under the frozen ABI +
  scalar reference + `parity(ref↔oracle)` where an oracle exists +
  `parity(gpu↔ref)` within declared tolerance + complete registry row.
  No green, no merge.
- **LICENSE ASYMMETRY.** Rust crates are dual MPL-2.0 OR PMEL — every
  `.rs` and `.wgsl` carries the 13-line MPL/PMEL header (copy from any
  `image-*/src/*.rs`). TS files (`manifest/`, `glue/`, `scripts/`) carry
  NO header (private-side convention, like plugin-draw/plugin-web).
  Don't cross that line.
- **Interface freeze.** `image-core` types, the `KernelDef`/WGSL ABI
  (`image-kernels/src/abi.rs`), and the codec traits are FROZEN (M0
  phase 0). Changes go through the orchestrator as versioned amendments,
  never drive-by edits.

## Two-registry split

- `paged-media/state` `registry/features/plugin-image.yaml` — the
  STATUS ledger (stage `plugin.image`; status planned/partial/shipped).
- `plugin-image/registry/*.yaml` (here) — the build-consumed
  kernel/PSD-block/codec metadata: class, `mip_exact`, `gpu_tolerance`,
  oracle, provenance, test pointers. `image-kernels/build.rs` generates
  the dispatch table FROM `registry/kernels.yaml` — an implementation
  without a row is unreachable by construction (§12.2). The ids mirror
  the state `image.*` ids so the two registries join by id.

## Layout

```
manifest/            plugin manifest (media.paged.image) + panel prototypes
glue/                the bundle: defineBundle + activate(host) + React panel
registry/            build-consumed kernel/PSD/codec metadata (see above)
image-core/          frozen types: PixelFormat, Tile/TileMap, Region, slices
image-kernels/       KernelDef + WGSL ABI + kernel_family! (T0 families)
image-gpu/           wgpu device, tile pool, residency, dispatch, WGSL assembly
image-pipeline/      Engine A: lazy DAG, demand-driven ROI, op cache, sinks
image-graph/         Engine B types + stubs (engine lands M2)
image-cms/           CmsEngine trait; qcms display backend (D-11: hybrid)
image-codecs/        ImageSource/ImageTarget/ByteSource + adapters
image-psd/           PSD/PSB model + parser + preservation writer
image-js/            wasm-bindgen surface (the bundle's compute artifact)
image-conformance/   TEST-ONLY: scalar refs, parity harness, PSD fixture
                     builder, property tests, benches — never shipped
references/          READ-ONLY clean-room mounts (gitignored; see §3.1)
```

## Commands

```bash
# Rust (the engine)
cargo build --workspace && cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

# GPU parity tests run on the local Metal adapter by default;
# select a backend explicitly with WGPU_BACKEND=metal|vulkan|gl.

# TS (the bundle) — install order: editor → plugin-sdk → plugin-image
pnpm install && pnpm test && pnpm typecheck
pnpm validate:manifest

# Dependency guards (CI runs these; run before claiming green)
cargo tree -p image-kernels --edges normal | grep -E 'image-(pipeline|graph|gpu|cms|js)' && echo LEAK
cargo tree -p image-js --target wasm32-unknown-unknown | grep -E 'image-conformance|proptest' && echo LEAK
cargo deny check

# wasm artifact (size-tracked against the 8 MiB budget — BREAKAGE I-07)
cargo build --release --target wasm32-unknown-unknown -p image-js

# Optional PSD ecosystem oracle (psd-tools): create once, then
# PAGED_PSD_ORACLE=1 cargo test -p image-conformance -- --ignored
python3 -m venv .venv && .venv/bin/pip install psd-tools
```
