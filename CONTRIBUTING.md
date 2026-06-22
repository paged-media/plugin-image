# Contributing

Thanks for your interest in this **paged plugin**. It's open, dual-licensed
**AGPL-3.0 OR PMEL** — the same as the paged editor (a plugin is part of the
editor app). The engine (`paged-media/core`) and the plugin SDK
(`paged-media/plugin-sdk`) it builds on are MPL-2.0 OR PMEL.

## License of contributions

By contributing you agree to the **Contributor License Agreement**
([`CLA.md`](./CLA.md)), which lets And The Next GmbH distribute your
contribution under **both** the AGPL-3.0 and the commercial PMEL. You retain
copyright to your contribution. A CLA bot will ask you to sign on your first PR.

New source files must carry the standard license header — copy it verbatim from
the top of any existing source file in this repo.

## Building & testing

See `CLAUDE.md` for the specifics. In general: Rust crates build/test with
`cargo build` / `cargo test` / `cargo clippy`; TypeScript packages with
`pnpm install` / `pnpm -r test` / `pnpm -r typecheck`. Format only the files you
touched.
