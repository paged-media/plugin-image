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

//! The private PSD corpus, selected by CONTENT and classified.
//!
//! Two rules this module exists to enforce, because both have bitten
//! this workspace before:
//!
//! * **Select by magic, never by extension.** A file is a PSD/PSB when
//!   its first four bytes are `8BPS`. Extensions lie (images in the same
//!   corpus are misnamed by the hundred), and a lane that filters on
//!   `.psd` silently drops whatever was saved under another name.
//! * **Address by path RELATIVE to the corpus root, never by bare
//!   name.** Bare names collide across packs; a relative path does not.
//!
//! The corpus is private. Nothing here writes a file name anywhere
//! durable — the committed ledgers key files by SHA-256 and carry only
//! aggregates and the feature classification below.

use std::path::{Path, PathBuf};

use image_psd::model::addl::SectionKind;
use image_psd::PsdFile;

/// The `8BPS` signature every PSD and PSB starts with.
pub const PSD_MAGIC: [u8; 4] = *b"8BPS";

/// One corpus file: where it is, and how to name it in a log.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CorpusFile {
    /// Path relative to the corpus root (`/`-separated) — the only name a
    /// log line may use.
    pub rel: String,
    pub path: PathBuf,
}

/// Resolve the corpus root from the opt-in switch `var` (`1` = the
/// sibling checkout, anything else = an explicit root). `None`, with a
/// printed reason, when the switch is unset or the root is missing.
pub fn corpus_root(var: &str) -> Option<PathBuf> {
    let Some(switch) = std::env::var_os(var) else {
        eprintln!(
            "SKIP psd corpus lane: {var} unset (set it to 1, or to a corpus root, \
             and run with --ignored)"
        );
        return None;
    };
    let switch = switch.to_string_lossy().into_owned();
    let root = if switch == "1" || switch.is_empty() {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../corpus")
    } else {
        PathBuf::from(switch)
    };
    if !root.is_dir() {
        eprintln!("SKIP psd corpus lane: no corpus at {}", root.display());
        return None;
    }
    Some(root)
}

/// Does the file at `path` start with the PSD signature?
pub fn has_psd_magic(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut m = [0u8; 4];
    f.read_exact(&mut m).is_ok() && m == PSD_MAGIC
}

/// Every file under `root` whose CONTENT is a PSD/PSB, sorted by
/// relative path. VCS metadata is skipped; nothing else is.
pub fn psds_by_magic(root: &Path) -> Vec<CorpusFile> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() {
                if e.file_name() != ".git" {
                    stack.push(p);
                }
            } else if ft.is_file() && has_psd_magic(&p) {
                let rel = p
                    .strip_prefix(root)
                    .unwrap_or(&p)
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                out.push(CorpusFile { rel, path: p });
            }
        }
    }
    out.sort();
    out
}

/// What resource 0x0421 (version info) says about the merged composite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergedData {
    /// `hasRealMergedData` = 1: the composite is Photoshop's own render.
    Real,
    /// `hasRealMergedData` = 0: the composite is a placeholder (the file
    /// was saved without "maximize compatibility").
    NotReal,
    /// No 0x0421 resource at all — nothing vouches for the composite.
    Absent,
}

impl MergedData {
    pub fn as_str(self) -> &'static str {
        match self {
            MergedData::Real => "real",
            MergedData::NotReal => "flag-false",
            MergedData::Absent => "no-version-info",
        }
    }
}

/// Version-info resource id (Adobe Photoshop File Format specification,
/// Image Resource IDs: 0x0421 "Version Info … hasRealMergedData").
pub const RESOURCE_VERSION_INFO: u16 = 0x0421;

/// Read `hasRealMergedData` from the retained 0x0421 block.
///
/// The block is kept verbatim (`raw_block`: signature, id, padded Pascal
/// name, u32 size, data); the data is `u32 version`, then the flag byte.
pub fn merged_data(psd: &PsdFile) -> MergedData {
    let Some(block) = psd
        .resources
        .blocks
        .iter()
        .find(|b| b.id == RESOURCE_VERSION_INFO)
    else {
        return MergedData::Absent;
    };
    let Some(raw) = block.raw_block.as_deref() else {
        return MergedData::Absent;
    };
    // 4 sig + 2 id, then the Pascal name padded to an even total.
    let Some(&name_len) = raw.get(6) else {
        return MergedData::Absent;
    };
    let name_total = (1 + name_len as usize).next_multiple_of(2);
    let data_at = 6 + name_total + 4;
    match raw.get(data_at + 4) {
        Some(0) => MergedData::NotReal,
        Some(_) => MergedData::Real,
        None => MergedData::Absent,
    }
}

/// The layer features a file uses — the classification a ledger row
/// carries instead of a name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Features {
    pub color_mode: String,
    pub depth: u16,
    pub psb: bool,
    /// Layer records (dividers included).
    pub layer_records: usize,
    pub groups: bool,
    /// User or vector masks (`-2`/`-3` channels, a mask record, `vmsk`).
    pub masks: bool,
    pub clipping: bool,
    /// Adjustment layers (`levl`, `curv`, `hue2`, …).
    pub adjustments: bool,
    /// Fill layers (solid colour, gradient, pattern).
    pub fills: bool,
    /// Layer effects (`lrFX`, `lfx2`, `lmfx`).
    pub effects: bool,
    pub smart_objects: bool,
    pub text: bool,
    pub hidden_layers: bool,
    /// A layer below full opacity, or below full FILL opacity (`iOpa`).
    pub partial_opacity: bool,
    /// Any blend key other than `norm` (dividers' `pass` excluded).
    pub non_normal_blend: bool,
    /// Non-default "blend if" ranges.
    pub blend_if: bool,
}

const ADJUSTMENT_KEYS: [&[u8; 4]; 18] = [
    b"levl", b"curv", b"brit", b"blnc", b"hue2", b"hue ", b"selc", b"mixr", b"grdm", b"phfl",
    b"expA", b"vibA", b"blwh", b"nvrt", b"post", b"thrs", b"CgEd", b"clrL",
];
const FILL_KEYS: [&[u8; 4]; 3] = [b"SoCo", b"GdFl", b"PtFl"];
const EFFECT_KEYS: [&[u8; 4]; 3] = [b"lrFX", b"lfx2", b"lmfx"];
const SMART_KEYS: [&[u8; 4]; 3] = [b"SoLd", b"PlLd", b"SoLE"];
const TEXT_KEYS: [&[u8; 4]; 2] = [b"TySh", b"tySh"];
const VECTOR_MASK_KEYS: [&[u8; 4]; 2] = [b"vmsk", b"vsms"];

/// Is a blend-if range record the default (every range 0..255 open)?
/// Each channel pair is `src black-lo, black-hi, white-lo, white-hi`,
/// then the same for the destination; default is `00 00 FF FF`.
fn blend_ranges_default(raw: &[u8]) -> bool {
    raw.chunks_exact(4).all(|r| r == [0x00, 0x00, 0xFF, 0xFF])
}

pub fn classify(psd: &PsdFile) -> Features {
    let h = &psd.header;
    let mut f = Features {
        color_mode: format!("{:?}", h.color_mode),
        depth: h.depth,
        psb: matches!(psd.container, image_psd::Container::Psb),
        layer_records: psd.layer_mask.layers.len(),
        ..Features::default()
    };
    for l in &psd.layer_mask.layers {
        let has = |keys: &[&[u8; 4]]| l.addl.iter().any(|a| keys.iter().any(|k| **k == a.key));
        let divider = l.addl.iter().find_map(|a| a.lsct()).map(|d| d.kind);
        let is_divider = matches!(
            divider,
            Some(SectionKind::OpenFolder)
                | Some(SectionKind::ClosedFolder)
                | Some(SectionKind::BoundingDivider)
        );
        if is_divider {
            f.groups = true;
        }
        if l.channels.iter().any(|c| c.id == -2 || c.id == -3)
            || l.mask.is_some()
            || has(&VECTOR_MASK_KEYS)
        {
            f.masks = true;
        }
        if l.clipping != 0 {
            f.clipping = true;
        }
        f.adjustments |= has(&ADJUSTMENT_KEYS);
        f.fills |= has(&FILL_KEYS);
        f.effects |= has(&EFFECT_KEYS);
        f.smart_objects |= has(&SMART_KEYS);
        f.text |= has(&TEXT_KEYS);
        if l.flags & 0x02 != 0 && !matches!(divider, Some(SectionKind::BoundingDivider)) {
            f.hidden_layers = true;
        }
        let fill_below_full = l.addl.iter().any(|a| {
            a.key == *b"iOpa"
                && a.raw_block
                    .as_deref()
                    .and_then(|r| r.get(12))
                    .is_some_and(|v| *v != 255)
        });
        if l.opacity != 255 || fill_below_full {
            f.partial_opacity = true;
        }
        if !is_divider && l.blend_key != *b"norm" {
            f.non_normal_blend = true;
        }
        if !blend_ranges_default(&l.blend_ranges.raw) {
            f.blend_if = true;
        }
    }
    f
}

impl Features {
    /// The flags that are set, as a stable comma list (for summaries).
    pub fn tags(&self) -> Vec<&'static str> {
        let mut t = Vec::new();
        for (on, name) in [
            (self.groups, "groups"),
            (self.masks, "masks"),
            (self.clipping, "clipping"),
            (self.adjustments, "adjustments"),
            (self.fills, "fills"),
            (self.effects, "effects"),
            (self.smart_objects, "smart-objects"),
            (self.text, "text"),
            (self.hidden_layers, "hidden"),
            (self.partial_opacity, "partial-opacity"),
            (self.non_normal_blend, "non-normal-blend"),
            (self.blend_if, "blend-if"),
        ] {
            if on {
                t.push(name);
            }
        }
        t
    }
}

/// Per-channel absolute difference between two images.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ChannelDiff {
    pub max: u8,
    pub mean: f64,
    pub p99: u8,
    /// Percentage of pixels whose difference exceeds 2 levels.
    pub pct_over_2: f64,
}

/// How far our render is from a reference render.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CompositeDiff {
    /// R, G, B (each composited over white first, the way a flattened
    /// file shows a transparent pixel), then raw alpha.
    pub channels: [ChannelDiff; 4],
    pub delta_e_mean: f64,
    pub delta_e_p95: f64,
    pub delta_e_max: f64,
    /// Pixels the ΔE statistics were taken over (a deterministic
    /// stride subsample above [`DELTA_E_SAMPLE_CAP`]).
    pub delta_e_samples: usize,
}

/// ΔE00 is evaluated on at most this many pixels (a fixed-stride
/// subsample); the per-channel diffs always see every pixel.
pub const DELTA_E_SAMPLE_CAP: usize = 1 << 21;

/// How the REFERENCE buffer's colour channels relate to its alpha.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reference {
    /// Straight colour, like ours.
    Straight,
    /// Colour already MATTED against white (`c·α + 255·(1 − α)`), which
    /// is how a PSD's merged composite stores a transparent document —
    /// observed on every corpus file with transparency in the merged
    /// data. Comparing it as straight colour would charge that matte to
    /// the flatten.
    MattedWhite,
}

/// Compare our straight-alpha RGBA8 with a reference of the same extent.
///
/// # Panics
///
/// When the buffers differ in length or are not whole RGBA8 pixels.
pub fn compare_rgba8(ours: &[u8], theirs: &[u8], reference: Reference) -> CompositeDiff {
    assert_eq!(ours.len(), theirs.len(), "same extent");
    assert_eq!(ours.len() % 4, 0, "RGBA8");
    let n = ours.len() / 4;
    let over_white = |p: &[u8], c: usize| -> i32 {
        let a = p[3] as u32;
        ((p[c] as u32 * a + 255 * (255 - a) + 127) / 255) as i32
    };
    let mut hist = [[0u64; 256]; 4];
    for (a, b) in ours.chunks_exact(4).zip(theirs.chunks_exact(4)) {
        for c in 0..3 {
            let t = match reference {
                Reference::Straight => over_white(b, c),
                Reference::MattedWhite => b[c] as i32,
            };
            let d = (over_white(a, c) - t).unsigned_abs() as usize;
            hist[c][d] += 1;
        }
        hist[3][(a[3] as i32 - b[3] as i32).unsigned_abs() as usize] += 1;
    }
    let mut channels = [ChannelDiff::default(); 4];
    for (c, h) in hist.iter().enumerate() {
        if n == 0 {
            break;
        }
        let max = h.iter().rposition(|&v| v > 0).unwrap_or(0) as u8;
        let sum: u64 = h.iter().enumerate().map(|(d, &v)| d as u64 * v).sum();
        let over2: u64 = h[3..].iter().sum();
        let want = ((n as f64) * 0.99).ceil() as u64;
        let mut acc = 0u64;
        let mut p99 = 0u8;
        for (d, &v) in h.iter().enumerate() {
            acc += v;
            if acc >= want {
                p99 = d as u8;
                break;
            }
        }
        channels[c] = ChannelDiff {
            max,
            mean: sum as f64 / n as f64,
            p99,
            pct_over_2: 100.0 * over2 as f64 / n as f64,
        };
    }
    let stride = n.div_ceil(DELTA_E_SAMPLE_CAP).max(1);
    // ΔE is taken over white; a matted reference already is, so it is
    // handed over as opaque.
    let theirs_cmp: std::borrow::Cow<'_, [u8]> = match reference {
        Reference::Straight => std::borrow::Cow::Borrowed(theirs),
        Reference::MattedWhite => std::borrow::Cow::Owned(
            theirs
                .chunks_exact(4)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
        ),
    };
    let theirs = &theirs_cmp[..];
    let (sa, sb): (Vec<u8>, Vec<u8>) = if stride == 1 {
        (ours.to_vec(), theirs.to_vec())
    } else {
        let pick = |buf: &[u8]| -> Vec<u8> {
            buf.chunks_exact(4)
                .step_by(stride)
                .flatten()
                .copied()
                .collect()
        };
        (pick(ours), pick(theirs))
    };
    let de = crate::delta_e::delta_e_rgba8(&sa, &sb, [255, 255, 255]);
    CompositeDiff {
        channels,
        delta_e_mean: de.mean,
        delta_e_p95: de.p95,
        delta_e_max: de.max,
        delta_e_samples: de.count,
    }
}
