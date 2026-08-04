/*
 * This file is part of paged (https://paged.media).
 *
 * paged is free software: you may redistribute it and/or modify it under the
 * terms of the GNU Affero General Public License, version 3, as published by
 * the Free Software Foundation, OR under the Paged Media Enterprise License
 * (PMEL), a commercial license available from And The Next GmbH. Full
 * copyright and license information is available in LICENSE.md, distributed
 * with this source code.
 *
 * paged is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the licenses for details.
 *
 *  @copyright  Copyright (c) And The Next GmbH
 *  @license    AGPL-3.0-only OR Paged Media Enterprise License (PMEL)
 */

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

//! Adobe Photoshop `.abr` **brush-preset** reader.
//!
//! A sampled tip is literally the alpha bitmap a brush engine paints
//! with, and the descriptor tree around it is the dynamics model
//! Photoshop users already understand — which is why this lane exists.
//!
//! # Provenance
//!
//! The behaviour spec `thoughts/docs/paged/plugin-image/abr-brush-format-spec.md`
//! (revision 2), which was written by an ANALYST under the clean-room
//! two-role protocol and then verified against **3,215 real brush
//! presets in 9 licensed files**. Facts it carries are tagged `[OBS]`
//! (measured in those files), `[PUB]` (Adobe's own specification or an
//! independent public implementation) or `[REF]` (a single third-party
//! channel — treated here as plausible-but-unconfirmed and called out
//! wherever this reader leans on one). `references/` is never read by
//! implementers of this crate.
//!
//! Adobe has never published a specification for the modern (v6+) ABR
//! format. Everything in §5–§8 of the behaviour spec is
//! reverse-engineered by somebody; the document tracks by whom, per
//! claim, and so do the doc comments here.
//!
//! # Scope
//!
//! * **Modern container only** (major 6/7/9/10). The legacy v1/v2
//!   dialect is a structurally unrelated format that shares only the
//!   extension; it is rejected with a clear diagnostic, per the
//!   behaviour spec's own RECOMMENDATION (§1.2), and the public sources
//!   are rich enough that a later v1/v2 lane can be built from §1.2
//!   alone.
//! * **`patt` is retained as opaque bytes.** Pattern records drag in
//!   indexed-colour palettes, texture support is not needed for a first
//!   brush engine, and the section is empty in 7 of 9 fixtures. The
//!   texture `Idnt` then dangles, which the model already tolerates.
//!   The `patt` body is the same [`crate::vm_array`] container the
//!   sampled tips use, so most of the cost is already paid.
//! * **Nothing is written.** This is a reader. The preservation
//!   invariant is untouched.
//!
//! # Degradation posture
//!
//! One exotic brush must not poison a 500-brush library, and a pad byte
//! must not turn a clean EOF into a corrupt file. Structural framing
//! errors fail; everything else — an unknown section type, an
//! unrecognised tip class, an unresolvable sample reference, an
//! undecodable plane — degrades and lands in [`AbrFile::warnings`].

pub mod blend;
pub mod model;
pub mod phry;
pub mod samp;

use crate::descriptor::{read_versioned_descriptor, Descriptor, DESCRIPTOR_VERSION};
use crate::reader::ByteReader;
use crate::{PsdError, Result};

pub use blend::BlendMode;
pub use model::{
    AbrBrush, AbrTip, Airbrush, BristleShape, BristleTip, BrushPose, ColorDynamics, ComputedTip,
    ControlSource, Dynamics, ErodibleTip, HeightMap, SampledTipRef, Scatter, ShapeDynamics,
    Texture, ToolKind, ToolOptions, Transfer, ABSENT_PERCENT_DEFAULT, FRACTION_SCALE_PERCENT_KEYS,
    TOOL_OPTIONS_ABSENT_DEFAULT,
};
pub use phry::HierarchyNode;
pub use samp::AbrSample;

/// The 4-byte section signature (shared with PSD's own `8BIM` blocks).
pub const SECTION_SIGNATURE: [u8; 4] = *b"8BIM";

/// Major versions of the LEGACY, pre-descriptor dialect (`[PUB]`).
pub const LEGACY_VERSIONS: [i16; 2] = [1, 2];

/// Major versions of the MODERN, section-based dialect.
///
/// The set is **sparse and non-monotonic**: there is no version 3, 4, 5
/// or 8 in the wild, so a reader must match the explicit set and never
/// test `version >= 6` — a corrupt file whose first word is 11 or 4096
/// would otherwise be parsed as a section stream and read wildly out of
/// bounds (spec §1.2 trap). `[OBS]` for 6 (8 files) and 10 (1 file),
/// which parse byte-identically with no version-dependent branch
/// anywhere; `[REF]` that 7 and 9 behave the same — neither has been
/// seen.
pub const MODERN_VERSIONS: [i16; 4] = [6, 7, 9, 10];

/// Minor versions known to exist. Every one of the nine corpus fixtures
/// is minor 2; minor 1 is second-hand (spec §1.2, §14.2 item 4).
pub const KNOWN_MINOR_VERSIONS: [i16; 2] = [1, 2];

/// Everything the reader wants to tell you but must not fail the file
/// over.
///
/// Warnings exist because the behaviour spec repeatedly demands the
/// middle path: "do not fail the file, and do not fail silently either".
/// New long-form identifiers will keep appearing, and an anomaly that
/// never fires is cheap while one that fires silently is not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbrWarning {
    /// A minor version outside [`KNOWN_MINOR_VERSIONS`].
    UnknownMinorVersion(i16),
    /// The descriptor wrapper carried a version other than
    /// [`DESCRIPTOR_VERSION`]. Recorded, not enforced.
    UnexpectedDescriptorVersion { section: String, version: u32 },
    /// An unknown section type. Length-delimited, therefore trivially
    /// skippable; the body is retained in
    /// [`AbrFile::unknown_sections`] rather than failing the file.
    UnknownSection { kind: String, size: usize },
    /// A section body handler stopped before the section end. Harmless
    /// here because the reader always seeks to the declared end, but a
    /// signal that something is unmodelled.
    SectionTrailingBytes { kind: String, unread: usize },
    /// A sampled-tip record's rounded length exceeds the section body.
    SampleRecordOvershoot {
        declared: usize,
        rounded: usize,
        available: usize,
    },
    /// A structural parse did not account for exactly the record — the
    /// §2.2 self-check failing. `remaining` should equal `expected_pad`
    /// (the 0–3 bytes between the declared length and the rounded one);
    /// more means something after the array list was not modelled, less
    /// means the array list over-read.
    SampleRecordTrailingBytes {
        id: String,
        remaining: usize,
        expected_pad: usize,
    },
    /// More than one written plane in a tip record.
    MultiplePlanes { id: String, planes: usize },
    /// A tip record whose array list wrote no plane at all.
    SampleHasNoPlane { id: String },
    /// The redundant 32-bit depth disagrees with the 16-bit one.
    PlaneDepthDisagrees {
        id: String,
        pixel_depth: u32,
        depth: u16,
    },
    /// A 16-bit tip. No 16-bit tip exists in the corpus, so this decode
    /// path — raw only; 16-bit RLE is refused — is untested against a
    /// real file (spec §2.4 GAP).
    SixteenBitPlaneUnverified { id: String },
    /// A tip record that could not be decoded. The tip is unavailable;
    /// the rest of the library is not.
    SampleDecodeFailed { id: String, detail: String },
    /// Two `samp` records share an id. Never observed; the join would be
    /// ambiguous, and the FIRST record wins.
    DuplicateSampleId { id: String },
    /// A `samp` record no brush refers to. Never observed; harmless.
    OrphanSample { id: String },
    /// A `sampledData` value that resolved against no `samp` id.
    /// Reported rather than guessed at — there is deliberately no
    /// positional fallback (spec §3.3).
    UnresolvedSampleReference { brush_index: usize, id: String },
    /// A tip whose class id is none of the four known ones. The preset
    /// is retained as metadata.
    UnsupportedTipClass {
        brush_index: usize,
        class_id: String,
    },
    /// A descriptor whose class id was not the expected one.
    UnexpectedClassId { context: String, class_id: String },
    /// An ordinal outside its implicit table.
    OrdinalOutOfRange { key: String, value: i32 },
    /// A blend-mode identifier in neither vocabulary. Substituted with
    /// Normal.
    UnrecognisedBlendMode { key: String, value: String },
    /// A value whose unit or OSType was not the expected one.
    UnexpectedUnit { key: String, unit: String },
    /// `Dmtr` in a unit other than `#Pxl`. The value is kept verbatim;
    /// converting would need a resolution the file does not carry.
    NonPixelDiameter { unit: String },
    /// The `dtipsGridSize` and the `tdta` length disagree.
    HeightMapSizeMismatch {
        declared: u32,
        implied: u32,
        bytes: usize,
    },
    /// `useColorDynamics` was true. The whole colour-dynamics key set is
    /// `[REF]`-only: the gate was false on all 3,215 corpus brushes, so
    /// not one of its six keys has ever been observed (spec §6.6,
    /// §14.2 item 2).
    ColorDynamicsUnverified { brush_index: usize },
    /// The `phry` open/close token stream does not balance.
    HierarchyUnbalanced { opens: usize, closes: usize },
    /// The `phry` preset-leaf count does not match the brush count, so
    /// the positional binding was refused.
    HierarchyCountMismatch { presets: usize, brushes: usize },
}

/// A section the reader does not model, retained verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownSection {
    pub kind: [u8; 4],
    pub body: Vec<u8>,
}

/// A parsed `.abr` file.
#[derive(Debug, Clone, PartialEq)]
pub struct AbrFile {
    pub version: i16,
    pub minor_version: i16,
    /// Sampled tips in file order.
    pub samples: Vec<AbrSample>,
    /// Brush presets in **file order**, which is the presentation order
    /// in Photoshop's brush panel and is load-bearing: the `phry` folder
    /// tree refers to presets positionally and there is no id at this
    /// level. A reader that reorders or filters this list destroys the
    /// grouping (spec §3.2).
    pub brushes: Vec<AbrBrush>,
    /// The brush-panel folder tree, empty when the file has none.
    pub hierarchy: Vec<HierarchyNode>,
    /// The `patt` section body, retained opaquely (see the module docs).
    pub patterns_raw: Option<Vec<u8>>,
    pub unknown_sections: Vec<UnknownSection>,
    pub warnings: Vec<AbrWarning>,
}

impl AbrFile {
    /// Parse a whole `.abr` file.
    pub fn parse(bytes: &[u8]) -> Result<AbrFile> {
        let mut r = ByteReader::new(bytes);
        let version = r.i16()?;
        if LEGACY_VERSIONS.contains(&version) {
            return Err(PsdError::Unsupported(format!(
                "legacy .abr version {version}: a structurally unrelated, pre-descriptor format \
                 sharing only the file extension. Modern (6/7/9/10) files only; the legacy \
                 dialect is publicly documented and can be added as its own lane."
            )));
        }
        if !MODERN_VERSIONS.contains(&version) {
            return Err(PsdError::BadSignature(format!(
                ".abr version {version} is not one of {LEGACY_VERSIONS:?} ∪ {MODERN_VERSIONS:?}. \
                 There is no magic signature at offset 0, so the version word IS the integrity \
                 gate and the set is matched explicitly, never as `>= 6`."
            )));
        }
        let minor_version = r.i16()?;
        let mut warnings = Vec::new();
        if !KNOWN_MINOR_VERSIONS.contains(&minor_version) {
            warnings.push(AbrWarning::UnknownMinorVersion(minor_version));
        }

        let mut samples: Vec<AbrSample> = Vec::new();
        let mut brush_descriptors: Vec<Descriptor> = Vec::new();
        let mut hierarchy_roots: Vec<Descriptor> = Vec::new();
        let mut patterns_raw: Option<Vec<u8>> = None;
        let mut unknown_sections = Vec::new();

        // A section header is 12 bytes. Terminate when fewer remain OR
        // when the cursor has reached the end — and never let the
        // inter-section pad turn a clean EOF into an error: in 5 of 9
        // corpus files the file ends on the UNPADDED section end, so
        // applying the pad walks 1–3 bytes past EOF (spec §1.3 trap 12).
        while r.remaining() >= 12 {
            let sig = r.fourcc()?;
            if sig != SECTION_SIGNATURE {
                return Err(PsdError::Malformed {
                    section: "abr section",
                    detail: format!(
                        "expected `8BIM` at offset {}, found {:?}",
                        r.pos() - 4,
                        String::from_utf8_lossy(&sig)
                    ),
                });
            }
            let kind = r.fourcc()?;
            let size = r.u32()? as usize;
            // `sub` takes exactly `size` bytes and advances the parent
            // cursor past them — i.e. it IS the unconditional "seek to
            // section_start + size" the spec asks for, so a descriptor
            // that leaves trailing bytes cannot desynchronise the stream.
            let mut body = r.sub(size)?;
            match &kind {
                b"samp" => {
                    // An empty section is normal, not an error.
                    samples.extend(samp::parse_samp_section(&mut body, &mut warnings)?);
                }
                b"desc" => {
                    let (v, root) = read_versioned_descriptor(&mut body)?;
                    if v != DESCRIPTOR_VERSION {
                        warnings.push(AbrWarning::UnexpectedDescriptorVersion {
                            section: "desc".into(),
                            version: v,
                        });
                    }
                    // Accumulate across sections: do not assume one
                    // `samp` and one `desc` (spec §1.3 trap).
                    if let Some(list) = root.list(b"Brsh") {
                        for value in list {
                            if let Some(d) = value.as_descriptor() {
                                brush_descriptors.push(d.clone());
                            }
                        }
                    }
                }
                b"phry" => {
                    let (v, root) = read_versioned_descriptor(&mut body)?;
                    if v != DESCRIPTOR_VERSION {
                        warnings.push(AbrWarning::UnexpectedDescriptorVersion {
                            section: "phry".into(),
                            version: v,
                        });
                    }
                    hierarchy_roots.push(root);
                }
                b"patt" => {
                    if size > 0 {
                        patterns_raw = Some(body.take(body.remaining())?.to_vec());
                    }
                }
                other => {
                    warnings.push(AbrWarning::UnknownSection {
                        kind: String::from_utf8_lossy(other).into_owned(),
                        size,
                    });
                    unknown_sections.push(UnknownSection {
                        kind: *other,
                        body: body.take(body.remaining())?.to_vec(),
                    });
                }
            }
            if body.remaining() != 0 {
                warnings.push(AbrWarning::SectionTrailingBytes {
                    kind: String::from_utf8_lossy(&kind).into_owned(),
                    unread: body.remaining(),
                });
            }
            // The pad is computed from the DECLARED size, not from where
            // the body handler stopped — and the pad after the LAST
            // section is optional, so consume at most what is there.
            let pad = (4 - (size % 4)) % 4;
            let _ = r.take(pad.min(r.remaining()))?;
        }

        let mut brushes: Vec<AbrBrush> = brush_descriptors
            .iter()
            .enumerate()
            .map(|(i, d)| model::read_brush(d, i, &mut warnings))
            .collect();

        // ── the sample ↔ descriptor join (spec §3.3) ──────────────────
        //
        // Plain, exact, case-sensitive string equality — 3,205 of 3,205
        // references resolved that way in the corpus, with zero
        // prefix-retries needed and zero suffixed identifiers seen.
        // Samples ARE shared (many-to-one), so this is a LOOKUP, never a
        // pairing: one fixture has 2,053 brushes against 2,052 samples
        // precisely because a record is referenced twice.
        let mut index_of: Vec<(String, usize)> = Vec::with_capacity(samples.len());
        for (i, s) in samples.iter().enumerate() {
            if index_of.iter().any(|(id, _)| *id == s.id) {
                warnings.push(AbrWarning::DuplicateSampleId { id: s.id.clone() });
                continue;
            }
            index_of.push((s.id.clone(), i));
        }
        let lookup = |id: &str| index_of.iter().find(|(k, _)| k == id).map(|(_, i)| *i);
        for b in &mut brushes {
            b.resolve_samples(&lookup, &mut warnings);
        }
        let mut referenced = vec![false; samples.len()];
        for b in &brushes {
            for id in b.sampled_ids() {
                if let Some(i) = lookup(id) {
                    referenced[i] = true;
                }
            }
        }
        for (i, hit) in referenced.iter().enumerate() {
            if !hit {
                warnings.push(AbrWarning::OrphanSample {
                    id: samples[i].id.clone(),
                });
            }
        }

        let mut hierarchy = Vec::new();
        for root in &hierarchy_roots {
            hierarchy.extend(phry::decode(root, &mut warnings));
        }
        phry::bind_and_check(&mut hierarchy, brushes.len(), &mut warnings);

        Ok(AbrFile {
            version,
            minor_version,
            samples,
            brushes,
            hierarchy,
            patterns_raw,
            unknown_sections,
            warnings,
        })
    }

    /// The sample a brush's tip resolves to, if any.
    pub fn sample_for(&self, brush: &AbrBrush) -> Option<&AbrSample> {
        brush
            .tip
            .as_sampled()
            .and_then(|s| s.sample_index)
            .and_then(|i| self.samples.get(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_abr_container_version_set_is_matched_explicitly() {
        // Version 11 must NOT be treated as "newer than 6".
        for bad in [0i16, 3, 4, 5, 8, 11, 4096, -1] {
            let mut bytes = bad.to_be_bytes().to_vec();
            bytes.extend_from_slice(&2i16.to_be_bytes());
            let err = AbrFile::parse(&bytes).unwrap_err();
            assert!(
                matches!(err, PsdError::BadSignature(_)),
                "version {bad} produced {err:?}"
            );
        }
    }

    #[test]
    fn image_abr_container_legacy_versions_are_refused_by_name() {
        for legacy in LEGACY_VERSIONS {
            let mut bytes = legacy.to_be_bytes().to_vec();
            bytes.extend_from_slice(&[0, 0]);
            match AbrFile::parse(&bytes).unwrap_err() {
                PsdError::Unsupported(m) => assert!(m.contains("legacy"), "{m}"),
                other => panic!("expected Unsupported, got {other:?}"),
            }
        }
    }

    #[test]
    fn image_abr_container_an_empty_modern_file_parses_clean() {
        // Just the two version words: no sections at all.
        let mut bytes = 6i16.to_be_bytes().to_vec();
        bytes.extend_from_slice(&2i16.to_be_bytes());
        let f = AbrFile::parse(&bytes).unwrap();
        assert_eq!((f.version, f.minor_version), (6, 2));
        assert!(f.brushes.is_empty() && f.samples.is_empty());
        assert!(f.warnings.is_empty(), "{:?}", f.warnings);
    }

    #[test]
    fn image_abr_container_unknown_minor_version_warns_but_parses() {
        let mut bytes = 6i16.to_be_bytes().to_vec();
        bytes.extend_from_slice(&7i16.to_be_bytes());
        let f = AbrFile::parse(&bytes).unwrap();
        assert_eq!(f.warnings, vec![AbrWarning::UnknownMinorVersion(7)]);
    }
}
