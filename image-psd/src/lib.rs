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

//! PSD/PSB structural parse + preservation-invariant writer
//! (spec §10.4). The honest claim at every stage: **"Paged never
//! destroys a PSD."**
//!
//! # Clean-room provenance
//!
//! Implementation derives from the public Adobe Photoshop File Format
//! specification and black-box observation of synthesized corpus files
//! ONLY (spec §3); per-block provenance is recorded in
//! `registry/psd-blocks.yaml`. `references/` is never read by
//! implementers of this crate.
//!
//! # The preservation invariant (frozen M0 design)
//!
//! Three storage strategies, chosen per node:
//!
//! 1. **Typed-and-re-encoded** — fixed-width scalar structures (header,
//!    layer-record scalars). Re-encoding is byte-identical by
//!    construction.
//! 2. **Opaque-verbatim** — every block we don't model semantically
//!    (unknown resource ids, unknown additional-layer-info keys, color
//!    mode data, blend ranges, ZIP channel payloads, the merged
//!    composite). Stored as the exact source bytes, re-emitted
//!    verbatim.
//! 3. **Lazy-verbatim guard** — typed blocks ALSO retain their source
//!    bytes (`Raw = Option<Vec<u8>>`). An unmodified node re-emits its
//!    original bytes — so zero-edit round-trips stay byte-identical
//!    even when a producer used a non-canonical encoding our re-encoder
//!    would normalize. Only edited/constructed nodes (`Raw = None`)
//!    take the re-encode path; for those, structural equivalence is the
//!    contract (and byte-identity against our own canonical form).
//!
//! Channel pixel data stays as the on-disk compressed payload
//! (`ChannelData::Raw`) until something edits it — preservation AND the
//! 500 MB-PSB streaming budget (§13) both want that.

#![forbid(unsafe_code)]

pub mod compression;
pub mod container;
pub mod edit;
pub mod model;
pub mod reader;
pub mod writer;

mod emit;
mod parse;

pub use container::Container;
pub use model::PsdFile;

#[derive(Debug, thiserror::Error)]
pub enum PsdError {
    #[error("not a PSD/PSB file: {0}")]
    BadSignature(String),
    #[error("malformed {section}: {detail}")]
    Malformed {
        section: &'static str,
        detail: String,
    },
    #[error("truncated: needed {needed} bytes at offset {offset}, have {available}")]
    Truncated {
        offset: usize,
        needed: usize,
        available: usize,
    },
    #[error("unsupported: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, PsdError>;

impl PsdFile {
    /// Full structural parse (header → color mode → image resources →
    /// layer & mask info → merged composite). Every unmodeled block is
    /// retained opaquely; see the module docs.
    pub fn parse(bytes: &[u8]) -> Result<PsdFile> {
        parse::parse(bytes)
    }

    /// Serialize. Unmodified nodes re-emit their source bytes verbatim
    /// (zero-edit ⇒ byte-identical output); constructed/edited nodes
    /// re-encode canonically.
    pub fn write(&self) -> Result<Vec<u8>> {
        emit::write(self)
    }
}
