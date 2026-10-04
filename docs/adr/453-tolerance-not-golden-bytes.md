# ADR 453 — GPU output is verified against the scalar reference by tolerance, never by golden bytes

- **Status:** Accepted. Recorded retroactively on 2026-10-02 from the code at `f7d21e5`.
- **Scope:** `Tolerance` and `KernelDef.gpu_tolerance` in `image-kernels`; the harness and tests in `image-conformance`

## Context

Kernels run only on the GPU ([ADR 450](450-gpu-only-kernels.md)), and a scalar Rust
reference exists for many of them ([ADR 451](451-one-kernel-definition.md)). Something has
to define when a kernel's GPU output counts as correct.

The rule is stated in three places: on the `Tolerance` type, where GPU output is checked
against the reference by tolerance and not "byte-golden-tested"
(`image-kernels/src/lib.rs:78`); in the test crate, "BY TOLERANCE, never byte-golden"
(`image-conformance/src/lib.rs:36`); and in the per-kernel definition of done
(`CLAUDE.md:79-82`). None of the three gives a reason. The repository does not record why.

Two comments elsewhere describe the device dependence the rule has to live with. The
shared helper set avoids `pow` and other transcendentals because their "WGSL precision is
implementation-defined" (`image-kernels/src/abi.rs:70`). Test inputs exclude NaN and
infinity because the masked store is "fast-math/driver-dependent" for them
(`image-conformance/src/harness.rs:50`).

## Decision

A kernel is correct on the GPU when its output agrees with the scalar reference within the
tolerance the kernel declares. GPU output is never compared with stored bytes.

- `Tolerance` has three forms: `Exact`, `ChannelEpsF16(n)` (largest per-channel distance in
  f16 steps) and `PerceptualDeltaE`. At the pinned commit 32 kernels declare `Exact` and 96
  declare `ChannelEpsF16` with `n` from 1 to 16. No kernel declares `PerceptualDeltaE`.
- The input is quantised to f16 once. The GPU receives those bytes; the reference receives
  the same values widened to `f32`, computes in `f32`, and its result is quantised to f16
  as the last step before the comparison.
- Distance is the difference between two f16 bit patterns mapped onto a monotone integer
  line. NaN against a number is the maximum distance.
- Every module kernel has a row in a property table, and a test fails when one is missing. The properties are: identity under the
  kernel's documented no-op parameters within its declared tolerance (72 of the 95 rows; 4 rows check instead that a constant
  field is preserved, and 19 rows declare no identity and are excluded from this property), mask scoping for the modules that read
  the mask, determinism, and finite output. For a module kernel without a reference these are the only checks of its GPU output.
- The harness and the handwritten references live in `image-conformance`, which is never shipped; the generated
  twins are part of `image-kernels` and compile only under the `reference` feature that `image-conformance` enables.

## Evidence

- `image-kernels/src/lib.rs:76-87` — `Tolerance` and the statement of the rule
- `image-conformance/src/harness.rs:72-92`, `:106-153` — one quantised stimulus for both
  lanes; `parity` uploads, dispatches, reads back, runs the reference and diffs
- `image-conformance/src/quantize.rs:47-69` — the f16 step distance and the NaN rule
- `image-conformance/src/harness.rs:207-225` — `assert_within`; the ΔE arm is
  `unimplemented!`
- `image-conformance/tests/gpu_module_kernels.rs:33-67` — the property checks for module
  kernels
- `image-gpu/src/test_support.rs:62-86` (at `f7d21e5`: `image-conformance/src/device.rs`),
  `image-conformance/tests/gpu_module_kernels.rs:998-1013` — no adapter means the test
  skips, with a printed notice
- `.github/workflows/ci.yml:55-61` — the CI runner has no adapter

## Alternatives considered

Stored GPU output as the expected value is excluded by the rule itself.

A second tier, reference against an external implementation, is part of the definition of
done "where an oracle exists" (`CLAUDE.md:80`). `registry/kernels.yaml` names `vips`,
`gegl` or `both` as the oracle on 83 of its 128 rows. The repository contains no runner for
either, and every row has `status: implemented`, the level below the two parity levels the
file defines (`registry/kernels.yaml:22`).

## Consequences

The tolerance is part of a kernel's definition. `registry/kernels.yaml` repeats it per row,
but no code reads the registry's `gpu_tolerance` field; the value enforced is the one in
`KernelDef`.

CI does not run the comparison: the workflow states that the GPU parity tests skip themselves on the runner, which has no adapter
(`.github/workflows/ci.yml:55-59`). They execute only where an adapter exists; the commands section of the repository names the local Metal
adapter (`CLAUDE.md:136-137`). `.github/workflows/ci.yml:57-59` names a pinned software-rasteriser lane as a follow-up. No workflow sets
`WGPU_BACKEND` or `WGPU_FALLBACK`, although `image-gpu/src/device.rs:33-36` and `image-conformance/src/device.rs:33-37` describe CI as using them.

For kernels the scalar reference is the only standard: no test in the repository compares
it with another implementation. `image-conformance/src/lib.rs:40-43` lists
differential oracle runners that do not exist. The `gallery.*` family has no reference and
rests on the property checks alone.

## Related

- [ADR 450](450-gpu-only-kernels.md) — the reference is test-only
- [ADR 451](451-one-kernel-definition.md) — which kernels get a generated reference
- [ADR 452](452-kernel-abi.md) — the mask behaviour the property checks assert
- [ADR 317](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/317-registry-driven-dispatch.md) — the registry rows that carry the tolerance and oracle fields

## Amendment — 2026-10-04

Three statements above no longer match the code.

- The ΔE arm is implemented. `assert_within` bounds the mean CIEDE2000 between GPU output
  and reference when a kernel declares `PerceptualDeltaE` (`image-conformance/src/harness.rs:249-257`),
  using `image-conformance/src/delta_e.rs`, whose unit test reproduces the published
  CIEDE2000 test pairs of Sharma, Wu and Dalal (2005). This supersedes "the ΔE arm is
  `unimplemented!`".
- CI runs the comparison. The `gpu-sw` job runs the whole suite on a software Vulkan adapter
  for every pull request and the `gpu-metal` job on Apple silicon after merges to main, both
  with `REQUIRE_GPU=1`, which makes a missing adapter fail every device test instead of
  skipping it (`.github/workflows/ci.yml`, `image-gpu/src/test_support.rs:78-85`). This
  supersedes the paragraph beginning "CI does not run the comparison". The `rust` job still
  runs without an adapter, where the device tests skip.
- The four per-crate device helpers became one, `image_gpu::test_support::device_or_skip`,
  and `WGPU_FALLBACK` forces the fallback adapter only when set to `1`
  (`image-gpu/src/device.rs:63-68`).
