# ADR 450 — Kernels execute on the GPU only; the scalar twin is for tests

- **Status:** Accepted. Recorded retroactively on 2026-10-02 from the code at `f7d21e5`.
- **Scope:** `image-kernels`, `image-gpu`, `image-js`, `image-conformance`, and the dependency guard in `.github/workflows/ci.yml`

## Context

A kernel is a pixel operation with a registry id (`math.add`, `conv.gaussian_h`,
`compose.multiply`). `image-kernels` holds the definitions and states that "WGSL is the
implementation" (`image-kernels/src/lib.rs:33`). Some kernels also have a scalar Rust
version, which the test crate uses to compute expected values
([ADR 451](451-one-kernel-definition.md), [ADR 453](453-tolerance-not-golden-bytes.md)).

The Rust version could become a fallback that ships when no GPU is present. The hard rules exclude that: "No CPU kernel path ships."
(`CLAUDE.md:65`). The rule cites the original design spec, published in part as [`concept.md`](../concept.md). The repository does not record why.

## Decision

Every registered kernel runs as a WGSL compute shader through `wgpu`, in `image-gpu`. No
Rust implementation of a kernel is compiled into the shipped wasm, and a missing GPU is
reported as an error.

- The generated scalar twins and their helper module compile only under the cargo feature
  `reference` of `image-kernels`, which only `image-conformance` enables. Handwritten
  references live in `image-conformance` itself. No other crate depends on that crate.
- `init_gpu` rejects when the realm has no `navigator.gpu`. A call that needs a dispatch
  and finds no device returns an error; the layer composite does the same.
- CI runs `cargo tree -p image-js --target wasm32-unknown-unknown` and fails when the
  output names `image-conformance` or `proptest`.
- The rule is about kernels. Other pixel work runs on the CPU: codec decode and encode, PSD parse and write, the
  colour transforms applied at ingest, selection and brush coverage, histogram and statistics, the 2× box downsample
  of the mip pyramid, the curves lookup table, the Jacobi solve behind the healing brush, and content-aware fill.
  Each site says so in a comment, except PSD parse and write, which the hard rules name (`CLAUDE.md:68-70`).

## Evidence

- `image-gpu/src/lib.rs:33-36` — the crate describes itself as the only execution layer
- `image-kernels/Cargo.toml:14-19`, `image-conformance/Cargo.toml:13-18` — the `reference`
  feature and the one dependency that turns it on
- `image-kernels/src/lib.rs:47-48`, `image-kernels/src/family.rs:118-133` — the reference
  helper module and the scalar twin exist only under that feature
- `image-js/src/lib.rs:172-190`, `image-js/src/lib.rs:514-519` — `init_gpu` rejects without
  WebGPU; the adjust runner errors without a device
- `image-js/src/layers.rs:1407-1413` — the layer composite refuses instead of blending on
  the CPU
- `.github/workflows/ci.yml:78-88` — the guard on the wasm32 dependency tree
- `image-js/src/ingest.rs:1137-1147`, `image-js/src/mip.rs:41-46`,
  `image-js/src/heal.rs:52-62`, `image-js/src/lib.rs:1645-1653` — four CPU pixel passes,
  each with its stated reason
- `CLAUDE.md:64-70` — the rule, and the work it names as staying on the CPU

## Alternatives considered

A CPU fallback for kernels is named and refused where it would have been used: the ingest
module says "(an absent adapter is an honest error)" (`image-js/src/ingest.rs:43`).

For the healing brush the opposite choice was weighed. A GPU version of the Jacobi
iteration would need one dispatch per sweep, "hundreds of round-trips per dab"
(`image-js/src/heal.rs:56`), so the solve stays on the CPU and its result is applied by the
registered kernel `math.add`.

## Consequences

Without WebGPU the plugin still decodes, shows the unadjusted image, composites a stack of
one plain layer, computes histograms and saves. Everything that dispatches a kernel is
unavailable, and the session reports that state to the panel
(`glue/src/session.ts:869-872`). The manifest declares `capabilities.gpu` with
`"realm": "bundle"` (`glue/manifest.json:27-29`); the device is created by the plugin's own
wasm.

The line between a kernel and CPU preparation is drawn by a comment at each site, not by a
type or a test. The CI guard looks for two crate names in a dependency tree; it does not
detect a CPU pixel loop written into a production crate.

Three CPU passes are described in the code as interim: the curves stage, the reductions in `image-gpu/src/reduce.rs` and the mip
pyramid each name a GPU version as future work. The curves comment, "no GPU LUT kernel exists yet" (`image-js/src/ingest.rs:1138`),
is out of date: a table kernel, `adjust.lut1d`, is registered (`image-kernels/src/families/adjust.rs:1616-1628`), but nothing in
`image-js` dispatches it and the adjustment chain still applies the curve with a CPU lookup.

Comments contradict the CI setup. `image-gpu/src/lib.rs:35`, `image-gpu/src/device.rs:34-35`
and `image-conformance/src/device.rs:34-35` speak of a software adapter in CI.
`.github/workflows/ci.yml:55-59` states that the runner has no adapter and that the GPU
tests skip, so CI does not execute any kernel. `image-kernels/Cargo.toml:17` calls the
enabling dependency a dev-dependency; it is a normal dependency of the test crate.

## Related

- [ADR 451](451-one-kernel-definition.md) — where the scalar twin comes from
- [ADR 453](453-tolerance-not-golden-bytes.md) — what the scalar twin is used for
- [ADR 308](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/308-plugin-wasm.md) — how the wasm that creates the device is loaded
