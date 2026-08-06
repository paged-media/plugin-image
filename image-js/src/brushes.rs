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

//! `.abr` brush presets, projected onto the parameters the paint engine
//! actually has — the bundle door over `image_psd::abr`.
//!
//! WHY THIS FILE EXISTS. The `.abr` reader is a large, corpus-gated,
//! spec-cited parser (3,215 presets across the analyst corpus), and until
//! now NOTHING called it. A wasm32 release build therefore eliminated it
//! wholesale: the capability existed in the repository and not in the
//! shipped artifact, which is the worst of both — the cost of carrying it
//! with none of the benefit, and a registry row that reads shipped.
//!
//! WHAT IT DELIBERATELY DOES NOT DO. The reader models far more than the
//! paint engine consumes: bristle physics, erodible height maps, dual
//! brushes, every jitter axis, the `phry` folder tree. Projecting all of
//! that would invent a UI for parameters no kernel reads. This door
//! surfaces exactly the four the brush machine takes — **size**,
//! **hardness**, **roundness/angle** (recorded, see below) and
//! **spacing** — plus the honest shape of what was skipped.
//!
//! Three honesty rules, each of which the corpus forced:
//!
//! * **Hardness is not universal.** Only a `computedBrush` carries
//!   `Hrdn`; a sampled (bitmap) tip has no such parameter, and an
//!   erodible tip's is optional. `null` means "the file does not say",
//!   never a substituted default — a preset that silently applied 0.5
//!   hardness to a bitmap tip would be fabricating.
//! * **Diameter is not always pixels.** `Dmtr` carries its own unit, and
//!   the reader keeps a non-`#Pxl` unit verbatim rather than rejecting
//!   it. The unit rides along so the panel can refuse to apply rather
//!   than apply a wrong number confidently.
//! * **An unsupported tip is still a preset.** `Unsupported`/`Missing`
//!   tips are LISTED, with `kind` saying so and no parameters — the
//!   reader's own §5 rule ("one exotic brush must not take out a
//!   500-brush library") carried through to the surface.
//!
//! The projection is a pure function over parsed bytes so it is testable
//! on the host; `mod wasm` is `#[cfg(target_arch = "wasm32")]` and never
//! type-checks there.

use image_psd::abr::{AbrBrush, AbrFile, AbrTip};
use image_psd::Result;

/// One preset, reduced to what the paint engine can act on.
#[derive(Debug, Clone, PartialEq)]
pub struct BrushPreset {
    /// Position in the file's `Brsh` list. **Load-bearing** — the folder
    /// tree refers to presets positionally, so this is the id.
    pub index: usize,
    pub name: String,
    /// `computed` | `sampled` | `bristle` | `erodible` | `unsupported` |
    /// `missing`. The panel shows it because the parameters available
    /// differ by kind, and a missing slider should be explicable.
    pub kind: &'static str,
    /// `Dmtr` as stored, in `diameter_unit`.
    pub diameter: Option<f64>,
    /// The four-byte unit tag, e.g. `#Pxl`. `None` when the tip carries
    /// no common block at all.
    pub diameter_unit: Option<String>,
    /// `Hrdn` as a 0..1 fraction where the tip has one.
    pub hardness: Option<f64>,
    /// `Spcn` as a fraction of the diameter.
    pub spacing: Option<f64>,
    /// `Intr` — when false, Photoshop spaces by cursor movement and
    /// `spacing` is inert. Surfaced so the panel does not apply a
    /// spacing the source disabled.
    pub spacing_enabled: Option<bool>,
    /// `Rndn` — minor/major axis ratio; 1.0 is circular.
    pub roundness: Option<f64>,
    pub angle_deg: Option<f64>,
}

impl BrushPreset {
    /// Whether the diameter can be read as pixels. A non-`#Pxl` unit is
    /// rare-to-unobserved but is not the same as absent, and the panel
    /// must not apply it as if it were px.
    pub fn diameter_is_pixels(&self) -> bool {
        self.diameter_unit.as_deref() == Some("#Pxl")
    }
}

/// Project a parsed `.abr` onto the paint engine's parameters.
pub fn presets_of(file: &AbrFile) -> Vec<BrushPreset> {
    file.brushes.iter().map(preset_of).collect()
}

fn preset_of(brush: &AbrBrush) -> BrushPreset {
    let kind = match &brush.tip {
        AbrTip::Computed(_) => "computed",
        AbrTip::Sampled(_) => "sampled",
        AbrTip::Bristle(_) => "bristle",
        AbrTip::Erodible(_) => "erodible",
        AbrTip::Unsupported { .. } => "unsupported",
        AbrTip::Missing => "missing",
    };
    // Hardness lives on two variants only, and on one of them it is
    // optional. Everything else reports `None` rather than a default.
    let hardness = match &brush.tip {
        AbrTip::Computed(t) => Some(t.hardness),
        AbrTip::Erodible(t) => t.hardness,
        _ => None,
    };
    let roundness = match &brush.tip {
        AbrTip::Computed(t) => Some(t.roundness),
        AbrTip::Sampled(t) => Some(t.roundness),
        _ => None,
    };
    let common = brush.tip.common();
    BrushPreset {
        index: brush.index,
        name: brush.name.clone(),
        kind,
        diameter: common.map(|c| c.diameter),
        diameter_unit: common.map(|c| unit_tag(&c.diameter_unit)),
        hardness,
        spacing: common.map(|c| c.spacing),
        spacing_enabled: common.map(|c| c.spacing_enabled),
        roundness,
        angle_deg: common.map(|c| c.angle_deg),
    }
}

/// A four-byte unit tag as text. Non-ASCII bytes are hex-escaped rather
/// than lossily replaced — the tag is diagnostic, and `#Pxl` is the only
/// value the panel acts on anyway.
fn unit_tag(raw: &[u8; 4]) -> String {
    if raw.iter().all(|b| b.is_ascii_graphic()) {
        String::from_utf8_lossy(raw).into_owned()
    } else {
        raw.iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// Parse `.abr` bytes and render the presets as JSON for the panel.
///
/// Shape: `{version, minorVersion, presetCount, sampleCount, warnings,
/// presets: [...]}`. `warnings` is the reader's own diagnostic list,
/// carried through rather than swallowed — a file that parsed WITH
/// complaints is a different state from one that parsed cleanly.
pub fn presets_json(bytes: &[u8]) -> Result<String> {
    let file = AbrFile::parse(bytes)?;
    Ok(render_json(&file))
}

fn render_json(file: &AbrFile) -> String {
    let presets = presets_of(file);
    let rows: Vec<String> = presets.iter().map(row_json).collect();
    // `AbrWarning` is a structured enum with no `Display` — its `Debug`
    // form carries the fields (which id, which counts) that make a
    // warning actionable, so it is what the panel shows.
    let warnings: Vec<String> = file
        .warnings
        .iter()
        .map(|w| json_string(&format!("{w:?}")))
        .collect();
    format!(
        "{{\"version\":{},\"minorVersion\":{},\"presetCount\":{},\"sampleCount\":{},\
         \"warnings\":[{}],\"presets\":[{}]}}",
        file.version,
        file.minor_version,
        presets.len(),
        file.samples.len(),
        warnings.join(","),
        rows.join(",")
    )
}

fn row_json(p: &BrushPreset) -> String {
    format!(
        "{{\"index\":{},\"name\":{},\"kind\":\"{}\",\"diameter\":{},\"diameterUnit\":{},\
         \"hardness\":{},\"spacing\":{},\"spacingEnabled\":{},\"roundness\":{},\"angle\":{}}}",
        p.index,
        json_string(&p.name),
        p.kind,
        json_num(p.diameter),
        p.diameter_unit
            .as_deref()
            .map_or_else(|| "null".to_string(), json_string),
        json_num(p.hardness),
        json_num(p.spacing),
        p.spacing_enabled
            .map_or_else(|| "null".to_string(), |b| b.to_string()),
        json_num(p.roundness),
        json_num(p.angle_deg),
    )
}

/// `f64` as JSON, with the non-finite trap closed: `NaN`/`Infinity` are
/// not JSON and would make `JSON.parse` throw on the panel side, turning
/// one odd preset into a whole-file failure. They render as `null`,
/// which is the same "the file does not say" the panel already handles.
fn json_num(v: Option<f64>) -> String {
    match v {
        Some(f) if f.is_finite() => format!("{f}"),
        _ => "null".to_string(),
    }
}

/// Minimal JSON string escape. `mod wasm` has its own copy for the PSD
/// doors; this one is host-testable, which is the point of the module.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image_psd::abr::model::{ComputedTip, SampledTipRef, TipCommon};
    use image_psd::descriptor::{Descriptor, Key};

    fn empty_descriptor() -> Descriptor {
        Descriptor {
            class_name: String::new(),
            class_id: Key::four(b"Brsh"),
            items: Vec::new(),
        }
    }

    fn common(diameter: f64, unit: [u8; 4]) -> TipCommon {
        TipCommon {
            diameter,
            diameter_unit: unit,
            angle_deg: 12.5,
            spacing: 0.25,
            spacing_enabled: true,
            flip_x: false,
            flip_y: false,
        }
    }

    fn brush(name: &str, index: usize, tip: AbrTip) -> AbrBrush {
        AbrBrush {
            index,
            name: name.to_string(),
            descriptor: empty_descriptor(),
            tip,
            wet_edges: None,
            noise: None,
            use_brush_size: None,
            brush_spacing: None,
            shape_dynamics: None,
            scatter: None,
            texture: None,
            dual_brush: None,
            color_dynamics: None,
            transfer: None,
            brush_pose: None,
            tool_options: None,
        }
    }

    fn file_of(brushes: Vec<AbrBrush>) -> AbrFile {
        AbrFile {
            version: 6,
            minor_version: 2,
            samples: Vec::new(),
            brushes,
            hierarchy: Vec::new(),
            patterns_raw: None,
            unknown_sections: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// The door must refuse a non-`.abr` by NAME, not by producing an
    /// empty preset list — "no brushes" and "not a brush file" are
    /// different answers and the panel shows different things for them.
    #[test]
    fn image_abr_engine_bridge_rejects_non_abr_bytes() {
        let err = presets_json(b"\x89PNG\r\n\x1a\n and then some").unwrap_err();
        let msg = err.to_string();
        assert!(
            !msg.is_empty(),
            "a rejection must carry a reason, got an empty message"
        );
    }

    /// An empty container is a legal file: version + minor + no sections.
    /// It must produce a parseable, EMPTY document rather than an error —
    /// otherwise the panel cannot distinguish "loaded, nothing in it"
    /// from "failed".
    #[test]
    fn image_abr_engine_bridge_empty_container_renders_empty_json() {
        let mut bytes = 6i16.to_be_bytes().to_vec();
        bytes.extend_from_slice(&2i16.to_be_bytes());
        let json = presets_json(&bytes).expect("an empty v6.2 container parses");
        assert!(json.contains("\"version\":6"), "{json}");
        assert!(json.contains("\"presetCount\":0"), "{json}");
        assert!(json.contains("\"presets\":[]"), "{json}");
    }

    /// Names come from an untrusted file and go straight into JSON. A
    /// quote or a newline in a preset name must not be able to break the
    /// document the panel parses.
    #[test]
    fn image_abr_engine_bridge_escapes_hostile_preset_names() {
        let escaped = json_string("he said \"hi\"\n\tand \\ left\u{1}");
        assert_eq!(escaped, "\"he said \\\"hi\\\"\\n\\tand \\\\ left\\u0001\"");
    }

    /// Non-finite parameters render as `null`, not as the literal `NaN`
    /// that would make `JSON.parse` throw for the whole file.
    #[test]
    fn image_abr_engine_bridge_non_finite_numbers_become_null() {
        assert_eq!(json_num(Some(f64::NAN)), "null");
        assert_eq!(json_num(Some(f64::INFINITY)), "null");
        assert_eq!(json_num(None), "null");
        assert_eq!(json_num(Some(30.0)), "30");
    }

    /// The unit tag decides whether the panel may treat the diameter as
    /// pixels at all.
    #[test]
    fn image_abr_engine_bridge_only_pxl_diameters_are_pixels() {
        let mut p = BrushPreset {
            index: 0,
            name: "t".into(),
            kind: "computed",
            diameter: Some(30.0),
            diameter_unit: Some("#Pxl".into()),
            hardness: Some(0.5),
            spacing: Some(0.25),
            spacing_enabled: Some(true),
            roundness: Some(1.0),
            angle_deg: Some(0.0),
        };
        assert!(p.diameter_is_pixels());
        p.diameter_unit = Some("#Prc".into());
        assert!(!p.diameter_is_pixels(), "a percentage is not a pixel count");
        p.diameter_unit = None;
        assert!(!p.diameter_is_pixels());
    }

    /// A four-byte tag with non-printable bytes is hex-escaped rather
    /// than turned into replacement characters — it is diagnostic output.
    #[test]
    fn image_abr_engine_bridge_binary_unit_tags_are_hex() {
        assert_eq!(unit_tag(b"#Pxl"), "#Pxl");
        assert_eq!(unit_tag(&[0x00, 0xff, 0x10, 0x41]), "00ff1041");
    }

    /// The projection itself: a computed tip yields every parameter the
    /// brush machine takes, in the units it takes them.
    #[test]
    fn image_abr_engine_bridge_projects_a_computed_tip_onto_brush_params() {
        let tip = AbrTip::Computed(ComputedTip {
            common: common(30.0, *b"#Pxl"),
            roundness: 0.8,
            hardness: 0.25,
        });
        let file = file_of(vec![brush("Soft Round", 0, tip)]);
        let p = &presets_of(&file)[0];
        assert_eq!(p.kind, "computed");
        assert_eq!(p.name, "Soft Round");
        assert_eq!(p.diameter, Some(30.0));
        assert!(p.diameter_is_pixels());
        assert_eq!(p.hardness, Some(0.25));
        assert_eq!(p.roundness, Some(0.8));
        assert_eq!(p.spacing, Some(0.25));
        assert_eq!(p.spacing_enabled, Some(true));
        assert_eq!(p.angle_deg, Some(12.5));
    }

    /// A SAMPLED tip has no `Hrdn` at all, and the honest answer is
    /// `null` — not a substituted default. This is the rule the panel's
    /// disabled Hardness row is built on, so it is pinned here.
    #[test]
    fn image_abr_engine_bridge_a_sampled_tip_reports_no_hardness() {
        let tip = AbrTip::Sampled(SampledTipRef {
            common: common(64.0, *b"#Pxl"),
            roundness: 1.0,
            name: Some("tip".into()),
            sampled_id: "0b0c4a97".into(),
            sample_index: Some(0),
        });
        let file = file_of(vec![brush("Chalk 60", 0, tip)]);
        let p = &presets_of(&file)[0];
        assert_eq!(p.kind, "sampled");
        assert_eq!(
            p.hardness, None,
            "a bitmap tip carries no hardness; inventing one would fabricate"
        );
        assert_eq!(p.diameter, Some(64.0), "but it does carry a size");
    }

    /// One exotic preset must not cost the library: an unsupported tip
    /// is LISTED, with its kind saying why it has no parameters, and the
    /// presets around it are untouched.
    #[test]
    fn image_abr_engine_bridge_an_unsupported_tip_is_listed_not_dropped() {
        let file = file_of(vec![
            brush(
                "ok",
                0,
                AbrTip::Computed(ComputedTip {
                    common: common(10.0, *b"#Pxl"),
                    roundness: 1.0,
                    hardness: 1.0,
                }),
            ),
            brush(
                "exotic",
                1,
                AbrTip::Unsupported {
                    class_id: "someFutureBrush".into(),
                },
            ),
            brush("headless", 2, AbrTip::Missing),
        ]);
        let presets = presets_of(&file);
        assert_eq!(presets.len(), 3, "no preset is dropped");
        assert_eq!(presets[1].kind, "unsupported");
        assert_eq!(presets[1].name, "exotic");
        assert_eq!(presets[1].diameter, None);
        assert_eq!(presets[2].kind, "missing");
        assert_eq!(presets[0].diameter, Some(10.0), "the good one still works");
    }

    /// THE WIRE, byte for byte.
    ///
    /// This exact string is also pinned in the bundle's
    /// `glue/test/brushes.spec.ts`, which feeds it to a fake wasm module
    /// to test the JSON→`BrushParams` mapping without needing real
    /// `.abr` bytes in Node. That split only stays honest if the two
    /// halves cannot drift silently — so if you change the shape here,
    /// this test fails FIRST and points at the spec that must change
    /// with it.
    #[test]
    fn image_abr_engine_bridge_the_wire_shape_is_pinned_on_both_sides() {
        let tip = AbrTip::Computed(ComputedTip {
            common: TipCommon {
                diameter: 30.0,
                diameter_unit: *b"#Pxl",
                angle_deg: 0.0,
                spacing: 0.25,
                spacing_enabled: true,
                flip_x: false,
                flip_y: false,
            },
            roundness: 1.0,
            hardness: 0.5,
        });
        let json = render_json(&file_of(vec![brush("Soft Round 30", 0, tip)]));
        assert_eq!(
            json,
            "{\"version\":6,\"minorVersion\":2,\"presetCount\":1,\"sampleCount\":0,\
             \"warnings\":[],\"presets\":[{\"index\":0,\"name\":\"Soft Round 30\",\
             \"kind\":\"computed\",\"diameter\":30,\"diameterUnit\":\"#Pxl\",\
             \"hardness\":0.5,\"spacing\":0.25,\"spacingEnabled\":true,\
             \"roundness\":1,\"angle\":0}]}"
        );
    }

    /// The JSON the panel parses: keys present, indices preserved, and a
    /// `null` where a parameter is genuinely absent.
    #[test]
    fn image_abr_engine_bridge_json_carries_indices_and_nulls() {
        let file = file_of(vec![
            brush(
                "A",
                0,
                AbrTip::Computed(ComputedTip {
                    common: common(30.0, *b"#Pxl"),
                    roundness: 1.0,
                    hardness: 0.5,
                }),
            ),
            brush("B", 1, AbrTip::Missing),
        ]);
        let json = render_json(&file);
        assert!(json.contains("\"presetCount\":2"), "{json}");
        assert!(json.contains("\"index\":0,\"name\":\"A\""), "{json}");
        assert!(json.contains("\"hardness\":0.5"), "{json}");
        // B has no tip, so every parameter is null — and `index` is still
        // its position, because the folder tree addresses presets by it.
        assert!(json.contains("\"index\":1,\"name\":\"B\""), "{json}");
        assert!(
            json.contains("\"kind\":\"missing\",\"diameter\":null"),
            "{json}"
        );
    }
}
