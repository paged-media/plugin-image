/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * This file is part of paged (https://paged.media) and is additionally
 * available under the Paged Media Enterprise License (PMEL). Full
 * copyright and license information is available in LICENSE.md which is
 * distributed with this source code.
 *
 *  @copyright  Copyright (c) And The Next GmbH
 *  @license    MPL-2.0 OR Paged Media Enterprise License (PMEL)
 */

//! Kernel model (spec §6) — WGSL is the implementation.
//!
//! A kernel is a WGSL compute function plus a typed param block plus
//! metadata. Kernels are engine-agnostic pure functions over tile
//! slices with a frozen ABI (`abi.rs`); Engine A, Engine B, the
//! conformance harness, and the WGSL assembly all consume this ONE
//! definition. The `kernel_family!` macro (`family.rs`) emits both the
//! WGSL body and (behind the `reference` feature, test-only) the
//! scalar Rust twin from a single expression — one source of truth.

pub mod abi;
#[macro_use]
pub mod family;
pub mod families;
#[cfg(feature = "reference")]
pub mod reference_prelude;

/// Frozen kernel ABI version (spec §9.2). Bumps are orchestrator-level
/// amendments.
pub const ABI_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelClass {
    /// out(x,y) = f(in₀(x,y), …)
    Point,
    /// Needs an expanded input window (ROI inflation in the engines).
    Windowed { radius: (u16, u16) },
    /// Rational scale with kernel support.
    Resample { support: f32 },
    /// min/max/avg/histogram → scalar/table.
    Reduction(ReductionKind),
    /// No inputs (gradients, noise, constants).
    Generator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionKind {
    Min,
    Max,
    Avg,
    Histogram,
}

/// Per-kernel GPU-vs-reference tolerance (spec §6.3): GPU output is
/// verified against the scalar-reference golden by tolerance, never
/// byte-golden-tested.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tolerance {
    /// Bit-identical after f16 quantization of the reference.
    Exact,
    /// Max per-channel distance in f16 ULPs.
    ChannelEpsF16(u32),
    /// Mean perceptual ΔE bound (color kernels, T1+).
    PerceptualDeltaE(f32),
}

/// One field of a param block — drives the WGSL `struct Params`
/// emission (1:1 with the `#[repr(C)]` Rust layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamField {
    pub name: &'static str,
    /// `f32` | `u32` | `i32` — the token is valid in both languages.
    pub wgsl_ty: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamsLayout {
    /// `size_of` the Rust param struct (uniform upload size).
    pub size: usize,
    pub fields: &'static [ParamField],
}

/// The kernel definition (spec §6.1, frozen M0 phase 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KernelDef {
    /// Registry id (dispatch key), e.g. `math.linear`, `conv.gaussian`.
    pub id: &'static str,
    pub class: KernelClass,
    /// Input texture arity (group 0): 1 = unary, 2 = binary,
    /// 0 = generator.
    pub inputs: u8,
    pub params: ParamsLayout,
    /// The kernel WGSL — interpreted per `module`: an expression body
    /// in the restricted DSL spliced into the ABI template
    /// (`kernel_family!` output), or a complete handwritten compute
    /// module conforming to the ABI binding interface (T1
    /// windowed/resample kernels, spec §9.2 "handwritten WGSL"). The
    /// WGSL is the production implementation either way.
    pub wgsl: &'static str,
    /// ABI v1.1 amendment (M1): `true` = `wgsl` is a complete module
    /// (see `abi` docs for the module-authoring contract); `false` =
    /// expression body. `kernel_family!` always emits `false`.
    pub module: bool,
    /// Safe to evaluate at mip levels with scaled params? (§8.3)
    pub mip_exact: bool,
    pub gpu_tolerance: Tolerance,
}

// The dispatch table, GENERATED from registry/kernels.yaml by build.rs
// (spec §12.2.1): a kernel without a registry row is unreachable by
// construction.
include!(concat!(env!("OUT_DIR"), "/registry_gen.rs"));

/// Dispatch lookup over the registry table.
pub fn lookup(id: &str) -> Option<&'static KernelDef> {
    registry().iter().copied().find(|d| d.id == id)
}

/// Every `KernelDef` defined in code, registry-listed or not. The
/// conformance gate asserts set-equality with `registry()` — both
/// directions of the 100% invariant (§12.2).
pub fn all_defined() -> Vec<&'static KernelDef> {
    families::ALL_FAMILIES
        .iter()
        .flat_map(|f| f.iter().copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_and_definitions_agree() {
        let mut reg: Vec<&str> = registry().iter().map(|d| d.id).collect();
        let mut def: Vec<&str> = all_defined().iter().map(|d| d.id).collect();
        reg.sort_unstable();
        def.sort_unstable();
        assert_eq!(
            reg, def,
            "registry/kernels.yaml and code-defined kernels must agree \
             (registry-driven dispatch, spec §12.2)"
        );
    }

    #[test]
    fn lookup_finds_linear() {
        let def = lookup("math.linear").expect("math.linear registered");
        assert_eq!(def.inputs, 1);
        assert!(matches!(def.class, KernelClass::Point));
    }
}
