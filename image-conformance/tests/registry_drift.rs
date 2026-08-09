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

//! THE KERNEL REGISTRY MUST MATCH THE KERNELS.
//!
//! `registry/kernels.yaml` is quoted as the metric for this plugin —
//! "101 kernels, 101 implemented" is how the catalog, the state registry
//! and every progress note describe the engine. Until this file existed,
//! NOTHING checked that claim: the YAML was prose that happened to sit
//! near the code, and the two could disagree indefinitely without a
//! single test noticing.
//!
//! They HAD disagreed. Six kernels (emboss, find_edges, motion, radial,
//! mosaic, selective_color) were written, compiled, and passed the whole
//! suite while absent from the registry — which means every count quoted
//! from it was quietly wrong for as long as that took to spot.
//!
//! The failure mode this prevents is not a missing kernel. It is a
//! CONFIDENT WRONG NUMBER: a registry that says 101 when there are 107
//! is worse than no registry, because it is consulted instead of the
//! code.
//!
//! Deliberately parsed with a line scanner rather than a YAML crate: the
//! two facts this needs (`- id:` and `status:`) are unambiguous at the
//! line level, and a dependency added for a drift test would be a
//! dependency the production crates carry forever.

use std::collections::BTreeSet;
use std::path::PathBuf;

use image_kernels::families::ALL_FAMILIES;

fn registry_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../registry/kernels.yaml")
}

/// Every `- id: …` in the registry, and separately those whose row says
/// `status: implemented`.
fn registry_ids() -> (BTreeSet<String>, BTreeSet<String>) {
    let text = std::fs::read_to_string(registry_path()).expect("read registry/kernels.yaml");
    let mut all = BTreeSet::new();
    let mut implemented = BTreeSet::new();
    let mut current: Option<String> = None;
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("- id: ") {
            let id = rest.trim().to_string();
            all.insert(id.clone());
            current = Some(id);
        } else if t == "status: implemented" {
            if let Some(id) = &current {
                implemented.insert(id.clone());
            }
        }
    }
    (all, implemented)
}

/// Every kernel id the CODE actually defines.
fn code_ids() -> BTreeSet<String> {
    ALL_FAMILIES
        .iter()
        .flat_map(|f| f.iter())
        .map(|k| k.id.to_string())
        .collect()
}

#[test]
fn image_editor_every_code_kernel_is_in_the_registry() {
    let code = code_ids();
    let (registry, _) = registry_ids();
    let missing: Vec<&String> = code.difference(&registry).collect();
    assert!(
        missing.is_empty(),
        "these kernels exist in code but not in registry/kernels.yaml, so every \
         count quoted from the registry is wrong: {missing:#?}"
    );
}

#[test]
fn image_editor_every_registry_kernel_exists_in_code() {
    let code = code_ids();
    let (registry, _) = registry_ids();
    let phantom: Vec<&String> = registry.difference(&code).collect();
    assert!(
        phantom.is_empty(),
        "these kernels are in registry/kernels.yaml but no `KernelDef` defines \
         them — the registry is advertising capability that does not exist, \
         which is the worse direction of the two: {phantom:#?}"
    );
}

#[test]
fn image_editor_a_registered_kernel_claiming_implemented_is_reachable() {
    // The status field is what the catalog reads to say "101/101
    // implemented". A row may honestly say `planned`; what it may not do
    // is say `implemented` about something with no code behind it.
    let code = code_ids();
    let (_, implemented) = registry_ids();
    let lying: Vec<&String> = implemented.difference(&code).collect();
    assert!(
        lying.is_empty(),
        "these rows say `status: implemented` and have no `KernelDef`: {lying:#?}"
    );
}
