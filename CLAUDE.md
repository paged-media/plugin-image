# CLAUDE.md — paged-media/plugin-image

Orientation for Claude sessions in **paged-media/plugin-image** — the
paged.image raster subsystem, delivered as a Paged plugin (public;
dual-licensed AGPL-3.0 OR PMEL, And The Next GmbH).

## What this is

A Rust/WASM/WebGPU image-processing engine in two shapes — **Engine A**
(libvips-class demand-driven streaming pipeline, `image-pipeline`) and
**Engine B** (GEGL-class persistent tiled buffer graph, `image-graph`) —
plus PSD/PSB round-trip (`image-psd`), GPU-only WGSL kernels
(`image-kernels`/`image-gpu`), CMS (`image-cms`), codec adapters
(`image-codecs`), and a TEST-ONLY conformance harness
(`image-conformance`). Shipped as a plugin bundle (`manifest/` + `glue/`)
consuming ONLY the published plugin SDK.

Spec (the authority): `docs/concept.md`.
A-0 audit + D-11 ruling: an internal audit note (its colour-engine ruling is `docs/adr/457-colour-management.md`).
SDK gap tracker: the cross-repo RFI (the internal gap register) (I-NN ids in §6; per-plugin BREAKAGE_LOG retired 2026-06-12).

## Project State & Feature Matrix (cockpit)

The feature inventory, test linkage and live status for ALL Paged repos are derived by
[Cockpit](https://github.com/drietsch/cockpit) from `~/paged/cockpit/` (`cockpit.toml` with
`root = ".."`; features in `cockpit/docs/features/<chapter>/<id>.md`). There is NO feature
matrix in this repo; do not create one.

Rules for every code change in this repo:

1. NEW CAPABILITY → feature file. If your change adds or completes a feature, add or update
   `cockpit/docs/features/<chapter>/<id>.md` (separate commit in `paged/cockpit`, referenced
   from this one). Feature ids are immutable; rename with `superseded_by`.
2. EVERY NEW TEST → feature link. Playwright: `{ tag: ['@feat:<id>'] }`. Rust: a test name
   ending in `__feat__<id_with_underscores>` or containing `[<id>]`. Otherwise an entry in
   `cockpit/test-map.yaml`.
3. STATUS CHANGE → `claims:` in the feature file, never prose. "X is now shipped/partial" is a
   claim edit; whether it *works* is computed from evidence and cannot be written.
4. BEFORE claiming a feature done: `cockpit feature <id> --json` (or its page in
   `cockpit serve`) — done means the linked tests are green and were produced after the
   latest implementation commit.
5. `cockpit validate --strict` is the gate (references resolve, required evidence present and
   fresh). `cockpit pull` fetches the newest CI artifacts; `cockpit status` is the summary.
6. FOUND A BUG while working? If a test exposes it, let it fail and push — the failure shows
   up as attention on its feature. Never commit `.cockpit/`.

## Hard rules (this repo's constitution — spec §2/§3/§6)

- **CLEAN-ROOM / TWO-ROLE (§3.1).** `references/gegl` + `references/libvips`
  are read-only inspiration mounts. ANALYST agents may read `references/`
  and produce *behavior specs* into `thoughts/` (facts about behavior,
  never expression). IMPLEMENTER agents — everyone writing kernel,
  engine, PSD, or codec code — **MUST NOT read `references/`**, ever.
  Never paste, transliterate, or closely paraphrase reference code or
  comments into any artifact. Implementation derives from the spec, the
  behavior specs in an internal notes repository, public documentation and academic
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

- `~/paged/cockpit/docs/features/image/<id>.md` (Cockpit, chapter `image`) — the
  STATUS ledger (component `plugin.image`; `claims:` planned/partial/shipped,
  health from evidence).
- `plugin-image/registry/*.yaml` (here) — the build-consumed
  kernel/PSD-block/codec metadata: class, `mip_exact`, `gpu_tolerance`,
  oracle, provenance, test pointers. `image-kernels/build.rs` generates
  the dispatch table FROM `registry/kernels.yaml` — an implementation
  without a row is unreachable by construction (§12.2). The ids mirror
  the Cockpit `image.*` ids so the two registries join by id.

## Layout

```
manifest/            plugin manifest (media.paged.image) + panel prototypes
glue/                the bundle: defineBundle + activate(host) + React panel
registry/            build-consumed kernel/PSD/codec metadata (see above)
image-core/          frozen types: PixelFormat, Tile/TileMap, Region, slices
image-kernels/       KernelDef + WGSL ABI + kernel_family! (T0 families)
image-gpu/           wgpu device, tile pool, residency, dispatch, WGSL assembly
image-pipeline/      Engine A: lazy DAG, demand-driven ROI, op cache, sinks
image-graph/         Engine B: tiled buffer graph — store, eval, tile cache,
                     set_params/gesture split, COW undo journal (journal.rs:
                     bounded, tile-granular, scoped per buffer)
image-cms/           CmsEngine trait; qcms display backend (D-11: hybrid)
image-codecs/        ImageSource/ImageTarget/ByteSource + adapters
image-psd/           PSD/PSB model + parser + preservation writer
image-js/            wasm-bindgen surface (the bundle's compute artifact) +
                     the LAYER GRAPH (layers.rs: the pixel-layer stack, its
                     compose.* fold, and the journal-backed undo)
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

# wasm artifact (size-tracked against the 100 MB app wasm budget — BREAKAGE I-07)
cargo build --release --target wasm32-unknown-unknown -p image-js

# Optional PSD ecosystem oracle (psd-tools): create once, then
# PAGED_PSD_ORACLE=1 cargo test -p image-conformance -- --ignored
python3 -m venv .venv && .venv/bin/pip install psd-tools

# Optional .abr corpus gate — LANE B, ANALYST-ONLY (it reads the mount).
# Skips loudly without it; the always-on LANE A (tests/abr_lane_a.rs)
# needs no corpus and runs in the ordinary suite.
#
# "Analyst-owned" is a convention about who maintains the EXPECTATIONS,
# not a barrier — the switch is not a credential. Running it is harmless:
# it discloses only aggregates the committed artifacts already publish.
# Do not build a gate that tries to detect who ran it.
PAGED_ABR_CORPUS=1 cargo test -p image-conformance --test abr_corpus -- --ignored --nocapture
```
