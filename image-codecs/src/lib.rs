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

//! Codec adapter contract (spec §10.2) — frozen M0 phase 0.
//!
//! Sources/targets are sans-IO over [`ByteSource`] so the same adapters
//! serve browser (memory / OPFS / ReadableStream) and native (file)
//! builds. Codecs are inherently CPU work and remain so — "GPU-only"
//! (spec §1) refers to kernel execution, not entropy coding.

#![forbid(unsafe_code)]

mod bytesource;
pub mod jpeg;
pub mod png;
pub mod raw;
mod source;
mod target;

pub use bytesource::{ByteSource, MemoryByteSource};
pub use jpeg::{JpegSource, JpegTarget};
pub use png::{PngSource, PngTarget};
pub use source::{ImageSource, SourceInfo};
pub use target::{EncodedStats, ImageTarget, TargetInfo};

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("malformed {format}: {detail}")]
    Malformed {
        format: &'static str,
        detail: String,
    },
    #[error("unsupported {format} feature: {detail}")]
    Unsupported {
        format: &'static str,
        detail: String,
    },
    #[error("read out of bounds: offset {offset} + len {len} > source len {source_len}")]
    OutOfBounds {
        offset: u64,
        len: usize,
        source_len: u64,
    },
    #[error("target sequencing error: {0}")]
    Sequencing(&'static str),
    #[error("io: {0}")]
    Io(String),
}

pub type Result<T> = std::result::Result<T, CodecError>;
