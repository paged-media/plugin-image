# ADR 451 — One kernel definition feeds both lanes

- **Status:** Accepted. Recorded retroactively on 2026-10-02 from the code at `f7d21e5`.
- **Scope:** `image-kernels` (`lib.rs`, `family.rs`, `abi.rs`, `reference_prelude.rs`, `families/`) and the tests in `image-conformance` that consume it

## Context

A kernel is used in two lanes. The GPU lane runs its WGSL in production
([ADR 450](450-gpu-only-kernels.md)). The reference lane runs a scalar Rust version in the
test crate and compares the two ([ADR 453](453-tolerance-not-golden-bytes.md)). The
comparison is only meaningful if both versions compute the same function.

The first kernels were per-pixel expressions; `math.linear`, in the first commit, is
`a * splat4(p.gain) + splat4(p.bias)`. For such kernels the crate header states the aim:
the macro emits the WGSL body and the scalar Rust twin from a single expression, "one
source of truth" (`image-kernels/src/lib.rs:41`). The macro's documentation gives the
reason: a token outside the shared vocabulary fails to compile on the Rust side, so "Rust
divergence is impossible by construction" (`image-kernels/src/family.rs:52`).

## Decision

A kernel is one `KernelDef` static: id, class, input arity, parameter layout, WGSL text,
the `module` flag, `mip_exact` and `gpu_tolerance`. Both engines, the WGSL assembler and
the test harness read that one value.

For an expression kernel, the `kernel_family!` macro takes one `eval:` expression and
emits three things from it:

- a `#[repr(C)]` parameter struct with a trailing `_abi_pad: u32`;
- the `KernelDef`, whose `wgsl` field is `stringify!` of the expression;
- under the cargo feature `reference`, a Rust function whose body is the same tokens,
  evaluated over `Px`, an `f32` RGBA value.

The expression is restricted to tokens valid in both languages: the samples `a` and `b`,
`p.<field>`, float literals with a decimal point, `+ - * /`, vector indexing (`a[0]`,
`a[p.channel]`), a fixed set of helper functions and the builtins `clamp`, `mix`, `min`,
`max`, `abs`, `floor`. The helpers exist twice, as `WGSL_PRELUDE` in `abi.rs` and as Rust
functions in `reference_prelude.rs`, where `Px` also carries the arithmetic operators.

At the pinned commit 33 of the 128 registered kernels are defined this way: the 16 `math.*`
kernels, `rel.*`, `bool.*`, `band.*`, `cast.*` and `adjust.invert_rgb`. The other 95 are
handwritten WGSL modules (`module: true`, [ADR 452](452-kernel-abi.md)). For those the
single `KernelDef` still serves every consumer, but a scalar reference, where one exists,
is written by hand in `image-conformance`.

## Evidence

- `image-kernels/src/lib.rs:105-129` — `KernelDef` and its `module` flag
- `image-kernels/src/family.rs:33-53` — what the macro emits and the token vocabulary
- `image-kernels/src/family.rs:99-116`, `:118-133` — `wgsl: stringify!($body)`, and the
  feature-gated function that evaluates the same `$body`
- `image-kernels/src/abi.rs:57-93`, `image-kernels/src/reference_prelude.rs:33-38` — the two
  halves of the helper set and the statement that they are kept in step
- `image-conformance/tests/golden_expansion.rs:43-46`, `:92-97` — a test pins both
  emissions of `math.linear`
- `image-conformance/tests/gpu_module_kernels.rs:36-43` — a handwritten module gets no
  generated twin
- `image-kernels/src/families/gen.rs:59-62` — a module family whose scalar reference is
  written by hand in its test file

## Alternatives considered

Writing the shader and the reference separately for every kernel. That is what the module
kernels do, because the expression vocabulary cannot state a window, a resample or a
coordinate-derived value (`image-kernels/src/families/gen.rs:59`; commit `9a245a9`). No
other alternative is recorded.

## Consequences

For the 33 expression kernels the shader and the reference cannot be edited apart. A token
outside the vocabulary stops the build of `image-conformance`, the only crate that turns
the `reference` feature on.

The two helper sets are separate source files and are kept in step by hand. The snapshot
test covers one kernel, `math.linear`; other drift shows up only in the parity tests,
which need a GPU adapter. Parameter fields of a macro kernel are limited to `f32`, `u32`
and `i32` (`image-kernels/src/family.rs:55-59`).

The by-construction guarantee does not reach the 95 module kernels, which are the
majority. Their reference, when present, is a second implementation that can disagree
with the shader in the ordinary way. The `gallery.*` family has no scalar reference in the
test tree; its GPU output is covered by property checks only
(`image-conformance/tests/gpu_module_kernels.rs`).

## Related

- [ADR 450](450-gpu-only-kernels.md) — the scalar twin never ships
- [ADR 452](452-kernel-abi.md) — the template an expression is spliced into, and the module form
- [ADR 453](453-tolerance-not-golden-bytes.md) — how the two lanes are compared
- [ADR 317](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/317-registry-driven-dispatch.md) — how a `KernelDef` becomes reachable for dispatch
