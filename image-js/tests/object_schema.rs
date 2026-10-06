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

//! The object model's generated schema (ADR 323): every settable
//! parameter of an ADJUSTMENT LAYER, and the blend modes a layer can take,
//! emitted as `glue/src/object-model/generated/image-schema.json` from the
//! two places the engine already defines them:
//!
//!   * the kernels' `ParamsLayout` (`image_kernels::lookup`) — which
//!     kernel, which field, how many components;
//!   * `AdjustParams::to_wire` — where each value lives in the block an
//!     adjustment layer is stored as, and its default.
//!
//! The one thing written by hand is the BINDING between the two (which
//! wire slot feeds which kernel field), and it is checked against both
//! sides, in both directions:
//!
//!   1. every bound kernel and field exists in the registry, with the
//!      component count its WGSL type has;
//!   2. every field of every bound kernel is bound, or named internal
//!      (the engine sets it; a user never does);
//!   3. every wire slot is covered exactly once;
//!   4. every slot is LIVE: perturbing it round-trips through
//!      `from_wire`/`to_wire`, so a slot the decoder ignores fails here.
//!
//! The emitted file is compared with the committed one; regenerate with
//! `UPDATE_IMAGE_SCHEMA=1 cargo test -p image-js --test object_schema`.
//!
//! This is a TEST, not a module of the crate, on purpose: nothing here
//! enters the wasm, so the committed artifact's source hash is untouched.

// Test names carry the cockpit feature link (`__feat__<id>`).
#![allow(non_snake_case)]

use std::collections::{BTreeMap, BTreeSet};

use image_js::ingest::AdjustParams;

/// How a row's value is typed on the object model.
#[derive(Clone, Copy, PartialEq)]
enum Ty {
    Number,
    /// A 0 | 1 slot: a stage gate or a boolean kernel field.
    Bool,
    /// `n` 256-entry tone tables behind a gate slot (CPU curves; no kernel).
    Lut {
        tables: usize,
        gate: usize,
    },
}

/// One binding: object-model path ← wire slot(s) → kernel field component.
struct Bind {
    path: &'static str,
    /// The kernel the value feeds (`None` = the CPU curves pass).
    kernel: Option<&'static str>,
    /// The kernel field (`None` = a stage gate, which decides whether the
    /// kernel runs at all rather than feeding one of its fields).
    field: Option<&'static str>,
    /// Component of a vector field.
    component: Option<usize>,
    offset: usize,
    ty: Ty,
    /// A value stored only while this gate slot is 1 (`posterize`,
    /// `threshold` are `Option`s on the engine side: off drops the value).
    gate: Option<usize>,
}

/// Fields the ENGINE sets, never a user: an ABI pad, the transfer
/// encoding exposure is told, the vibrance kernel's own saturation term
/// (kept 0 so the global saturation stays one control), the unsharp
/// threshold (fixed 0), the Gaussian radius (derived from σ) and the
/// hue/saturation vectors' fourth lane.
const INTERNAL: &[(&str, &str)] = &[
    ("adjust.exposure", "encoding"),
    ("adjust.vibrance", "saturation"),
    ("conv.unsharp", "threshold"),
    ("conv.gaussian_h", "radius"),
];

/// Offset of the extended block in the wire (16 head values + the curve LUT).
const E: usize = 16 + 256;

fn binds() -> Vec<Bind> {
    let mut v = Vec::new();
    let mut f = |path, kernel, field, component, offset, ty| {
        v.push(Bind {
            path,
            kernel,
            field,
            component,
            offset,
            ty,
            gate: None,
        })
    };
    use Ty::*;
    let k = Some;
    // ── the head block ──────────────────────────────────────────────
    f(
        "exposure.ev",
        k("adjust.exposure"),
        k("ev"),
        None,
        0,
        Number,
    );
    f(
        "brightnessContrast.brightness",
        k("adjust.brightness_contrast"),
        k("brightness"),
        None,
        1,
        Number,
    );
    f(
        "brightnessContrast.contrast",
        k("adjust.brightness_contrast"),
        k("contrast"),
        None,
        2,
        Number,
    );
    f(
        "saturation.sat",
        k("adjust.saturation"),
        k("sat"),
        None,
        3,
        Number,
    );
    f(
        "whiteBalance.temp",
        k("adjust.white_balance"),
        k("temp"),
        None,
        4,
        Number,
    );
    f(
        "whiteBalance.tint",
        k("adjust.white_balance"),
        k("tint"),
        None,
        5,
        Number,
    );
    for (i, field) in ["in_black", "in_white", "gamma", "out_black", "out_white"]
        .into_iter()
        .enumerate()
    {
        f(LEVELS[i], k("adjust.levels"), k(field), None, 6 + i, Number);
    }
    f(
        "gaussianBlur.sigma",
        k("conv.gaussian_h"),
        k("sigma"),
        None,
        11,
        Number,
    );
    f(
        "unsharp.amount",
        k("conv.unsharp"),
        k("amount"),
        None,
        12,
        Number,
    );
    f(
        "hueRotate.degrees",
        k("adjust.hue_rotate"),
        k("degrees"),
        None,
        13,
        Number,
    );
    f(
        "invert.enabled",
        k("adjust.invert_rgb"),
        None,
        None,
        14,
        Bool,
    );
    f(
        "curves.composite",
        None,
        None,
        None,
        16,
        Lut {
            tables: 1,
            gate: 15,
        },
    );
    // ── the extended block ──────────────────────────────────────────
    f(
        "vibrance.vibrance",
        k("adjust.vibrance"),
        k("vibrance"),
        None,
        E,
        Number,
    );
    for (i, field) in [
        "sh_cr", "sh_mg", "sh_yb", "mid_cr", "mid_mg", "mid_yb", "hi_cr", "hi_mg", "hi_yb",
    ]
    .into_iter()
    .enumerate()
    {
        f(
            COLOR_BALANCE[i],
            k("adjust.color_balance"),
            k(field),
            None,
            E + 1 + i,
            Number,
        );
    }
    f(
        "blackWhite.enabled",
        k("adjust.black_white"),
        None,
        None,
        E + 10,
        Bool,
    );
    for (i, field) in ["reds", "yellows", "greens", "cyans", "blues", "magentas"]
        .into_iter()
        .enumerate()
    {
        f(
            BLACK_WHITE[i],
            k("adjust.black_white"),
            k(field),
            None,
            E + 11 + i,
            Number,
        );
    }
    f(
        "posterize.enabled",
        k("adjust.posterize"),
        None,
        None,
        E + 17,
        Bool,
    );
    f(
        "posterize.levels",
        k("adjust.posterize"),
        k("levels"),
        None,
        E + 18,
        Number,
    );
    f(
        "threshold.enabled",
        k("adjust.threshold"),
        None,
        None,
        E + 19,
        Bool,
    );
    f(
        "threshold.threshold",
        k("adjust.threshold"),
        k("threshold"),
        None,
        E + 20,
        Number,
    );
    f(
        "photoFilter.density",
        k("adjust.photo_filter"),
        k("density"),
        None,
        E + 21,
        Number,
    );
    f(
        "photoFilter.fr",
        k("adjust.photo_filter"),
        k("fr"),
        None,
        E + 22,
        Number,
    );
    f(
        "photoFilter.fg",
        k("adjust.photo_filter"),
        k("fg"),
        None,
        E + 23,
        Number,
    );
    f(
        "photoFilter.fb",
        k("adjust.photo_filter"),
        k("fb"),
        None,
        E + 24,
        Number,
    );
    f(
        "photoFilter.preserve",
        k("adjust.photo_filter"),
        k("preserve"),
        None,
        E + 25,
        Bool,
    );
    for (i, field) in [
        "rr", "rg", "rb", "rc", "gr", "gg", "gb", "gc", "br", "bg", "bb", "bc",
    ]
    .into_iter()
    .enumerate()
    {
        f(
            CHANNEL_MIXER[i],
            k("adjust.channel_mixer"),
            k(field),
            None,
            E + 26 + i,
            Number,
        );
    }
    for (i, field) in [
        "r_in_black",
        "r_in_white",
        "r_gamma",
        "g_in_black",
        "g_in_white",
        "g_gamma",
        "b_in_black",
        "b_in_white",
        "b_gamma",
    ]
    .into_iter()
    .enumerate()
    {
        f(
            LEVELS_RGB[i],
            k("adjust.levels_rgb"),
            k(field),
            None,
            E + 38 + i,
            Number,
        );
    }
    for (c, path) in HS_MASTER.into_iter().enumerate() {
        f(
            path,
            k("adjust.hue_saturation"),
            k("master"),
            Some(c),
            E + 47 + c,
            Number,
        );
    }
    for r in 0..6 {
        for c in 0..3 {
            f(
                HS_RANGES[r * 3 + c],
                k("adjust.hue_saturation"),
                k("ranges"),
                Some(r * 4 + c),
                E + 50 + r * 3 + c,
                Number,
            );
        }
    }
    f(
        "hueSaturation.colorize[0]",
        k("adjust.hue_saturation"),
        k("colorize"),
        Some(0),
        E + 68,
        Bool,
    );
    for c in 1..4 {
        f(
            HS_COLORIZE[c - 1],
            k("adjust.hue_saturation"),
            k("colorize"),
            Some(c),
            E + 68 + c,
            Number,
        );
    }
    // ── version 3: the per-channel curves ───────────────────────────
    f(
        "curves.channels",
        None,
        None,
        None,
        E + 72 + 1,
        Lut {
            tables: 3,
            gate: E + 72,
        },
    );
    for b in &mut v {
        b.gate = match b.path {
            "posterize.levels" => Some(E + 17),
            "threshold.threshold" => Some(E + 19),
            _ => None,
        };
    }
    v
}

// Path spellings: the stage, then the kernel field in camelCase (vector
// components indexed), so a path names its kernel field mechanically.
const LEVELS: [&str; 5] = [
    "levels.inBlack",
    "levels.inWhite",
    "levels.gamma",
    "levels.outBlack",
    "levels.outWhite",
];
const COLOR_BALANCE: [&str; 9] = [
    "colorBalance.shCr",
    "colorBalance.shMg",
    "colorBalance.shYb",
    "colorBalance.midCr",
    "colorBalance.midMg",
    "colorBalance.midYb",
    "colorBalance.hiCr",
    "colorBalance.hiMg",
    "colorBalance.hiYb",
];
const BLACK_WHITE: [&str; 6] = [
    "blackWhite.reds",
    "blackWhite.yellows",
    "blackWhite.greens",
    "blackWhite.cyans",
    "blackWhite.blues",
    "blackWhite.magentas",
];
const CHANNEL_MIXER: [&str; 12] = [
    "channelMixer.rr",
    "channelMixer.rg",
    "channelMixer.rb",
    "channelMixer.rc",
    "channelMixer.gr",
    "channelMixer.gg",
    "channelMixer.gb",
    "channelMixer.gc",
    "channelMixer.br",
    "channelMixer.bg",
    "channelMixer.bb",
    "channelMixer.bc",
];
const LEVELS_RGB: [&str; 9] = [
    "levelsRgb.rInBlack",
    "levelsRgb.rInWhite",
    "levelsRgb.rGamma",
    "levelsRgb.gInBlack",
    "levelsRgb.gInWhite",
    "levelsRgb.gGamma",
    "levelsRgb.bInBlack",
    "levelsRgb.bInWhite",
    "levelsRgb.bGamma",
];
const HS_MASTER: [&str; 3] = [
    "hueSaturation.master[0]",
    "hueSaturation.master[1]",
    "hueSaturation.master[2]",
];
const HS_RANGES: [&str; 18] = [
    "hueSaturation.ranges[0][0]",
    "hueSaturation.ranges[0][1]",
    "hueSaturation.ranges[0][2]",
    "hueSaturation.ranges[1][0]",
    "hueSaturation.ranges[1][1]",
    "hueSaturation.ranges[1][2]",
    "hueSaturation.ranges[2][0]",
    "hueSaturation.ranges[2][1]",
    "hueSaturation.ranges[2][2]",
    "hueSaturation.ranges[3][0]",
    "hueSaturation.ranges[3][1]",
    "hueSaturation.ranges[3][2]",
    "hueSaturation.ranges[4][0]",
    "hueSaturation.ranges[4][1]",
    "hueSaturation.ranges[4][2]",
    "hueSaturation.ranges[5][0]",
    "hueSaturation.ranges[5][1]",
    "hueSaturation.ranges[5][2]",
];
const HS_COLORIZE: [&str; 3] = [
    "hueSaturation.colorize[1]",
    "hueSaturation.colorize[2]",
    "hueSaturation.colorize[3]",
];

/// Components a WGSL field type carries (`f32` 1, `vec4<f32>` 4,
/// `array<vec4<f32>, 6>` 24).
fn components(wgsl: &str) -> usize {
    let lanes = |t: &str| {
        if t.starts_with("vec2") {
            2
        } else if t.starts_with("vec3") {
            3
        } else if t.starts_with("vec4") {
            4
        } else {
            1
        }
    };
    match wgsl.strip_prefix("array<") {
        Some(rest) => {
            let (inner, n) = rest
                .trim_end_matches('>')
                .rsplit_once(',')
                .expect("array<T, N>");
            lanes(inner.trim()) * n.trim().parse::<usize>().expect("array length")
        }
        None => lanes(wgsl),
    }
}

/// camelCase of a kernel field (`in_black` → `inBlack`).
fn camel(s: &str) -> String {
    let mut out = String::new();
    let mut up = false;
    for ch in s.chars() {
        if ch == '_' {
            up = true;
        } else if up {
            out.extend(ch.to_uppercase());
            up = false;
        } else {
            out.push(ch);
        }
    }
    out
}

/// The f32's shortest round-trip spelling (`1`, `0.4`) — valid JSON.
fn json_num(v: f32) -> String {
    format!("{v}")
}

fn wire_len() -> usize {
    AdjustParams::default().to_wire().len()
}

#[test]
fn every_binding_names_a_registered_kernel_field__feat__image_object_model() {
    for b in binds() {
        let Some(kid) = b.kernel else { continue };
        let def = image_kernels::lookup(kid)
            .unwrap_or_else(|| panic!("{}: kernel {kid} is not registered", b.path));
        let Some(field) = b.field else { continue };
        let pf = def
            .params
            .fields
            .iter()
            .find(|p| p.name == field)
            .unwrap_or_else(|| panic!("{}: {kid} has no field {field}", b.path));
        let n = components(pf.wgsl_ty);
        match b.component {
            Some(c) => assert!(c < n, "{}: component {c} of a {n}-lane field", b.path),
            None => assert_eq!(
                n, 1,
                "{}: {kid}.{field} is a {n}-lane field; bind a component",
                b.path
            ),
        }
        // The path names its field mechanically.
        let tail = b.path.rsplit_once('.').unwrap().1;
        let bare = tail.split('[').next().unwrap();
        assert_eq!(
            bare,
            camel(field),
            "{}: the path must end in the field's camelCase",
            b.path
        );
    }
}

#[test]
fn every_field_of_every_bound_kernel_is_bound_or_internal__feat__image_object_model() {
    let binds = binds();
    let kernels: BTreeSet<&str> = binds.iter().filter_map(|b| b.kernel).collect();
    for kid in kernels {
        let def = image_kernels::lookup(kid).unwrap();
        for pf in def.params.fields {
            if pf.name.starts_with('_') || INTERNAL.contains(&(kid, pf.name)) {
                continue;
            }
            let bound: BTreeSet<usize> = binds
                .iter()
                .filter(|b| b.kernel == Some(kid) && b.field == Some(pf.name))
                .map(|b| b.component.unwrap_or(0))
                .collect();
            assert!(
                !bound.is_empty(),
                "{kid}.{} is neither bound nor internal",
                pf.name
            );
            let n = components(pf.wgsl_ty);
            if n > 1 {
                // Vector fields: every lane bound except the documented
                // fourth (pad) lane of the hue/saturation vectors.
                for c in 0..n {
                    let pad = kid == "adjust.hue_saturation" && pf.name != "colorize" && c % 4 == 3;
                    assert!(
                        pad || bound.contains(&c),
                        "{kid}.{}[{c}] is unbound",
                        pf.name
                    );
                }
            }
        }
    }
}

#[test]
fn every_wire_slot_is_covered_exactly_once__feat__image_object_model() {
    let mut owner: BTreeMap<usize, &str> = BTreeMap::new();
    let mut claim = |slot: usize, path: &'static str| {
        if let Some(prev) = owner.insert(slot, path) {
            panic!("wire slot {slot} is claimed by {prev} and {path}");
        }
    };
    for b in binds() {
        match b.ty {
            Ty::Lut { tables, gate } => {
                claim(gate, b.path);
                for s in b.offset..b.offset + 256 * tables {
                    claim(s, b.path);
                }
            }
            _ => claim(b.offset, b.path),
        }
    }
    let n = wire_len();
    let missing: Vec<usize> = (0..n).filter(|s| !owner.contains_key(s)).collect();
    assert!(missing.is_empty(), "unbound wire slots: {missing:?}");
    assert_eq!(owner.len(), n);
}

#[test]
fn every_wire_slot_is_live__feat__image_object_model() {
    let base = AdjustParams::default().to_wire();
    for b in binds() {
        // A gated value is only stored while its gate is open.
        let mut base = base.clone();
        if let Some(g) = b.gate {
            base[g] = 1.0;
        }
        let mut w = base.clone();
        match b.ty {
            Ty::Number => w[b.offset] += 0.25,
            Ty::Bool => w[b.offset] = if w[b.offset] == 0.0 { 1.0 } else { 0.0 },
            Ty::Lut { tables, gate } => {
                w[gate] = 1.0;
                for i in 0..256 * tables {
                    w[b.offset + i] = (255 - (i % 256)) as f32;
                }
            }
        }
        let back = AdjustParams::from_wire(&w).unwrap().to_wire();
        assert_eq!(back, w, "{}: slot {} does not round-trip", b.path, b.offset);
        assert_ne!(
            back, base,
            "{}: perturbing slot {} changed nothing",
            b.path, b.offset
        );
    }
}

/// The JSON the glue reads. Hand-written (no serde in this crate) and
/// deterministic, so the committed file is a stable diff.
fn emit() -> String {
    let base = AdjustParams::default().to_wire();
    let mut rows = Vec::new();
    for b in binds() {
        let mut o = vec![format!("\"path\":\"{}\"", b.path)];
        o.push(match b.kernel {
            Some(k) => format!("\"kernel\":\"{k}\""),
            None => "\"kernel\":null".into(),
        });
        o.push(match b.field {
            Some(f) => format!("\"field\":\"{f}\""),
            None => "\"field\":null".into(),
        });
        if let Some(c) = b.component {
            o.push(format!("\"component\":{c}"));
        }
        o.push(format!("\"offset\":{}", b.offset));
        if let Some(g) = b.gate {
            o.push(format!("\"gatedBy\":{g}"));
        }
        match b.ty {
            Ty::Number => {
                o.push("\"type\":\"number\"".into());
                o.push(format!("\"default\":{}", json_num(base[b.offset])));
            }
            Ty::Bool => {
                o.push("\"type\":\"bool\"".into());
                o.push(format!("\"default\":{}", base[b.offset] != 0.0));
            }
            Ty::Lut { tables, gate } => {
                o.push("\"type\":\"lut\"".into());
                o.push(format!("\"tables\":{tables}"));
                o.push(format!("\"gate\":{gate}"));
                o.push("\"default\":null".into());
            }
        }
        rows.push(format!("    {{{}}}", o.join(",")));
    }
    let blends: Vec<String> = image_js::stroke::blend_names()
        .iter()
        .map(|b| format!("\"{b}\""))
        .collect();
    let kernels: BTreeSet<&str> = binds().iter().filter_map(|b| b.kernel).collect();
    let kernels: Vec<String> = kernels.iter().map(|k| format!("\"{k}\"")).collect();
    format!(
        "{{\n  \"generatedBy\": \"image-js/tests/object_schema.rs\",\n  \"wireLength\": {},\n  \"kernels\": [{}],\n  \"blendModes\": [{}],\n  \"adjust\": [\n{}\n  ]\n}}\n",
        wire_len(),
        kernels.join(","),
        blends.join(","),
        rows.join(",\n")
    )
}

#[test]
fn the_committed_schema_is_current__feat__image_object_model() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../glue/src/object-model/generated/image-schema.json"
    );
    let fresh = emit();
    if std::env::var("UPDATE_IMAGE_SCHEMA").as_deref() == Ok("1") {
        std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap()).unwrap();
        std::fs::write(path, &fresh).unwrap();
        return;
    }
    let committed = std::fs::read_to_string(path).unwrap_or_default();
    assert!(
        committed == fresh,
        "{path} is stale — regenerate with UPDATE_IMAGE_SCHEMA=1 cargo test -p image-js --test object_schema"
    );
}
