# ADR 461 — The clean-room two-role protocol

- **Status:** Accepted. Recorded retroactively on 2026-10-02 from the code at `f7d21e5`.
- **Scope:** a working rule for the whole repository; `registry/*.yaml`; `image-conformance/fixtures/abr/`

## Context

The README describes the two engines by naming existing projects: a "libvips-class
streaming pipeline" and a "GEGL-class persistent tiled buffer graph" (`README.md:5-6`).
Checkouts of such projects are kept beside the code under `references/`. `.gitignore`
describes that folder as "third-party material studied during incubation (read-only
mounts; never committed, never vendored into this repo)".

This ADR records a working method, not a mechanism in the code. The method is the first of the hard rules in
`CLAUDE.md` and has its own section in the README. Both state the rule and how to follow it. At `f7d21e5`
the repository does not record why; section 3.1 of [`concept.md`](../concept.md) states the original intent.

## Decision

Work on this repository is split into two roles with different reading rights.

- **Analysts** may read `references/`. They write behaviour specifications ("facts about
  behavior, never expression"), which are kept outside this repository.
- **Implementers** are "everyone writing kernel, engine, PSD, or codec code". They do not
  read `references/`. Their sources are the design spec, the analysts' behaviour
  specifications, public documentation, academic literature and the oracle tests.
- Reference code and comments are never pasted, transliterated or closely paraphrased
  "into any artifact".
- `references/` is git-ignored and is not a member of the Cargo workspace.
- Sources are written down. Every kernel and every PSD block handler "records its
  specification sources in its row in `registry/*.yaml`", and a kernel without a row is
  not dispatched.
- When a test needs facts that exist only in a reference corpus, an analyst publishes
  derived artifacts into the repository and the test reads those. The one case so far is
  the `.abr` brush corpus: counts, tables and one-way digests are committed, under the
  rule that "nothing may be published from which the corpus could be **reconstructed**".
  The test gate is split into a lane that runs everywhere on synthesised fixtures and an
  opt-in lane that needs the corpus.

## Evidence

- `CLAUDE.md:49-57`, `README.md:37-42` — the rule, in both places
- `.gitignore:1-3`, `Cargo.toml:1-3` — `references/` is ignored and "NEVER a member"
- `CLAUDE.md:76-78`, `registry/kernels.yaml:23`, `:34-35` — the provenance rule, the
  `provenance` field in the row schema, and one row's entry
- `image-psd/src/lib.rs:37-43`, `registry/psd-blocks.yaml:4-5` — the PSD crate's statement
  of what it is derived from; block handlers record provenance
- `image-conformance/fixtures/abr/README.md:9-44` — the published artifacts and their limit
- `image-conformance/fixtures/abr/README.md:105-122`, `CLAUDE.md:155-163` — the two test lanes

## Alternatives considered

- **Tests that read the corpus directly.** An earlier revision of the `.abr` behaviour
  specification told the implementer to wire the corpus into the test suite. The fixtures
  README records why that could not stand: "following it breaks the isolation rule,
  honouring the isolation rule leaves the gate unbuilt"
  (`image-conformance/fixtures/abr/README.md:17-21`). Derived artifacts were the answer.
- **A check on who runs the corpus lane.** Refused: "Do not build a gate that tries to
  detect who ran it" (`CLAUDE.md:162`).

## Consequences

Nothing in the repository checks who reads `references/`: no CI step or script refers to the folder.
For the corpus lane the repository says that being analyst-owned is "a convention about who maintains
the EXPECTATIONS, not a barrier" (`CLAUDE.md:159-160`). `.gitignore` and the workspace member list
keep `references/` out of commits and out of the build; they do not record who has read it.

The always-on lane reads only committed artifacts: it drives a synthesised-fixture builder
from the published profile (`image-conformance/fixtures/abr/README.md:105-109`). The corpus
lane reads the mount itself; it is `#[ignore]` and skips where the corpus is absent, which
includes CI (`image-conformance/fixtures/abr/README.md:117-119`).

A reader of this repository cannot follow every provenance entry to its source. The
behaviour specifications are not in the repository; `registry/abr.yaml:15-19` names one of
them as "THE SOURCE OF EVERY FACT BELOW". The design spec is published in part as
`docs/concept.md`. All 128 rows of `registry/kernels.yaml` carry a `provenance` list.

The two statements of the rule name two folders, `references/gegl` and
`references/libvips`. The fixtures README names two more under the same rule,
`references/abr-fixtures/` and `references/ag-psd` (`image-conformance/fixtures/abr/README.md:13`, `:39`).
The protocol governs what implementers read. Third-party code that ships is a separate
matter: Rust dependencies may come from crates.io only (`deny.toml:50-53`), and
`NOTICE.md:8-11` lists the dependencies whose licences require an attribution.

## Related

- [ADR 453](453-tolerance-not-golden-bytes.md) — the oracle tests implementers work from
- [ADR 458](458-psd-preservation.md) — the PSD crate, derived from the public file format specification
- [ADR 456](456-codecs.md) — the dependency rules in `deny.toml` and the attribution in `NOTICE.md`
- [ADR 317](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/317-registry-driven-dispatch.md) — the registry that carries the provenance rows
