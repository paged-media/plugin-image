# paged.image

The raster subsystem of the Paged ecosystem, delivered as a **Paged
plugin**: a Rust/WASM/WebGPU image-processing engine serving a
libvips-class streaming pipeline (Engine A) and a GEGL-class persistent
tiled buffer graph (Engine B), with PSD/PSB round-trip as a constitutive
property — "Paged never destroys a PSD."

Concept / spec: `thoughts/docs/paged/plugin-image/base-idea.md` (v0.5).
First-party in authorship, third-party in discipline: this plugin runs
under exactly the rules every external plugin runs under, and is
deliberately the heaviest stress test the plugin platform has. Every
place the SDK falls short is a row in the cross-repo RFI
(`thoughts/docs/paged/plugin-platform/rfi-core-sdk-gaps.md`), never a core
modification. The old in-repo `BREAKAGE_LOG.md` was retired on 2026-06-12 —
every I-NN it held resolved into a platform `C-`/`K-` row (I-01→C-1,
I-02→K-3, I-04→C-5, I-06→C-6), and the RFI records that there is no
image-local residual left to fold.

## Packages

| Path | What |
|---|---|
| `manifest/` | plugin manifest `media.paged.image` + panel prototypes |
| `glue/` | the bundle: `defineBundle` + `activate(host)` + panel |
| `image-core/` | frozen types: `PixelFormat`, `Tile`/`TileMap`, `Region` |
| `image-kernels/` | `KernelDef` + frozen WGSL ABI + `kernel_family!` codegen |
| `image-gpu/` | wgpu device mgmt, tile pool, residency tiers, dispatch |
| `image-pipeline/` | Engine A — demand-driven streaming evaluation |
| `image-graph/` | Engine B — buffer graph, eval, tile cache, bounded COW undo journal |
| `image-cms/` | color management behind a swappable `CmsEngine` (D-11: hybrid) |
| `image-codecs/` | `ImageSource`/`ImageTarget` adapters (sans-IO) |
| `image-psd/` | PSD/PSB structural parse + preservation-invariant writer |
| `image-js/` | wasm-bindgen surface (the bundle's compute artifact) |
| `image-conformance/` | test-only: scalar references, parity harness, PSD fixture corpus |

## Clean-room

`references/gegl` and `references/libvips` are read-only inspiration
mounts (gitignored, never vendored). The two-role protocol applies:
analysts read references and write behavior specs into `thoughts/`;
implementers never read `references/`. See `CLAUDE.md` and spec §3.1.

## Setup

Sibling checkouts expected (`~/paged/{editor,plugin-sdk,plugin-image}`),
install order matters for the `link:` chain:

```bash
cd ~/paged/editor && pnpm install
cd ~/paged/plugin-sdk && pnpm install
cd ~/paged/plugin-image && pnpm install

# Engine
cargo build --workspace && cargo test --workspace

# Bundle
pnpm test && pnpm validate:manifest
```

## Milestones (spec §15)

- **M0** — skeleton, codegen proof (T0 families at gpu↔ref parity), PSD
  structural round-trip (`preserved`), SDK gap table closed, bundle
  loads via SDK with zero core changes.
- **M1** — crown-jewel kernels (resample/cms/conv/compose), codecs, PSD
  `rendered`, public crates announcement.
- **M2** — buffer graph + interactivity (Engine B), PSD `mutatable`.
- **M3** — breadth (T3 ops), selections plumbing, editor enablement.
- **M4** — editor enablement in the product sense: ingest → GPU adjust →
  in-frame composite, the selection and paint tool layers, the layer graph
  and undo journal, and save-back. This rung is what the code calls M4; the
  ladder above stopped at M3 while the shipped slice already used the name.
- **M5** (2026-08-06) — the Photoshop-catalog Phase-2 wave: the
  non-destructive spine (layer masks, adjustment layers, smart objects,
  clipping, groups), four new kernel classes, the four missing panels,
  the retouching family (clone, gradient-domain heal, content-aware
  fill), CMS rung 1 and the K-10 save-file adoption.

  The living status is NOT here — it is
  `thoughts/docs/paged/plugin-image/photoshop_clone_capability_catalog_with_paged_reuse.md`
  §36.4/§36.5 for the capability ledger and the `paged-media/state`
  registry for per-feature status. A milestone list in a README is a
  claim that goes stale the week after it is written; those two do not,
  because CI reads them.

## License

Dual-licensed **AGPL-3.0 OR the Paged Media Enterprise License (PMEL)** —
the same as the paged editor (a plugin is part of the editor app). The engine
(`paged-media/core`) and the plugin SDK (`paged-media/plugin-sdk`) it builds on
are MPL-2.0 OR PMEL. See [`LICENSE.md`](./LICENSE.md), [`LICENSE`](./LICENSE),
and [`CONTRIBUTING.md`](./CONTRIBUTING.md) (contributions under a CLA).

`SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-PMEL`
