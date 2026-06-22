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
place the SDK falls short is an entry in `BREAKAGE_LOG.md` (I-NN), never
a core modification.

## Packages

| Path | What |
|---|---|
| `manifest/` | plugin manifest `media.paged.image` + panel prototypes |
| `glue/` | the bundle: `defineBundle` + `activate(host)` + panel |
| `image-core/` | frozen types: `PixelFormat`, `Tile`/`TileMap`, `Region` |
| `image-kernels/` | `KernelDef` + frozen WGSL ABI + `kernel_family!` codegen |
| `image-gpu/` | wgpu device mgmt, tile pool, residency tiers, dispatch |
| `image-pipeline/` | Engine A — demand-driven streaming evaluation |
| `image-graph/` | Engine B — types now, engine in M2 |
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

## License

Dual-licensed **AGPL-3.0 OR the Paged Media Enterprise License (PMEL)** —
the same as the paged editor (a plugin is part of the editor app). The engine
(`paged-media/core`) and the plugin SDK (`paged-media/plugin-sdk`) it builds on
are MPL-2.0 OR PMEL. See [`LICENSE.md`](./LICENSE.md), [`LICENSE`](./LICENSE),
and [`CONTRIBUTING.md`](./CONTRIBUTING.md) (contributions under a CLA).

`SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-PMEL`
