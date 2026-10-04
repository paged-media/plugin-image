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

//! THE LIBVIPS ORACLE, replayed. `scripts/vips/record.mjs` ran libvips
//! over deterministic float stimuli for every kernel `registry/
//! kernels.yaml` names libvips as an oracle for; its outputs are committed
//! as `fixtures/vips/<id>.v` with provenance in `<id>.vips.json`. This
//! file regenerates the SAME stimuli (the formula is shared with the
//! recorder) and compares:
//!
//! * the SCALAR REFERENCE with libvips — on every run, no GPU — for the
//!   kernels that have one in `image-kernels` (the `kernel_family!`
//!   point kernels);
//! * the GPU kernel with libvips — every row, under the shared test
//!   device (`REQUIRE_GPU=1` turns a missing adapter into a failure).
//!
//! Each row's tolerance is MEASURED, recorded in `registry/vips-oracle.
//! yaml` (`oracle_tolerance` for the reference, `gpu_tolerance_measured`
//! for the GPU) and asserted here; re-measure with
//! `PAGED_VIPS_TOLERANCE=write`. A row may instead be `diverges: <reason>`.
//! `vips_oracle_table_covers_every_vips_row` keeps the table and the
//! registry in step. CI needs no libvips: the fixtures are committed.

use std::collections::BTreeMap;
use std::path::PathBuf;

use half::f16;
use image_conformance::Px;
use image_kernels::families::{adjust, arithmetic, band, boolean, conv, geom, linear, minmax};
use image_kernels::families::{morph, relational, resample};
use image_kernels::KernelDef;

const N: u32 = 64;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

// ───────────────────────────── stimuli ───────────────────────────────

/// `h = (7x + 13y + 29c + 31·seed + 17·((x·y) mod 11)) mod 257` — the
/// recorder's formula, verbatim.
fn hash(seed: u32, x: u32, y: u32, c: u32) -> u32 {
    (7 * x + 13 * y + 29 * c + 31 * seed + 17 * ((x * y) % 11)) % 257
}

fn stimulus(kind: &str, seed: u32) -> Vec<Px> {
    let mut out = Vec::with_capacity((N * N) as usize);
    for y in 0..N {
        for x in 0..N {
            out.push(Px(std::array::from_fn(|c| {
                let h = hash(seed, x, y, c as u32) as f32;
                match kind {
                    "unit" => h / 256.0,
                    "signed" => (h - 128.0) / 128.0,
                    "pos" => (64.0 + (h as u32 % 193) as f32) / 256.0,
                    "quarter" => (h as u32 % 5) as f32 / 4.0,
                    "binary" => (h as u32 % 2) as f32,
                    "opaque" if c == 3 => 1.0,
                    "opaque" => h / 256.0,
                    other => panic!("unknown stimulus kind {other}"),
                }
            })));
        }
    }
    out
}

// ─────────────────────────── the .v reader ───────────────────────────

/// A vips native file: 64-byte little-endian header, then the pixels.
/// Only what the recorder writes is accepted: 4-band float.
fn read_v(id: &str) -> (u32, u32, Vec<[f32; 4]>) {
    let path = root().join(format!("fixtures/vips/{id}.v"));
    let b = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        &b[0..4],
        &[0xb6, 0xa6, 0xf2, 0x08],
        "{id}: not a little-endian .v"
    );
    let i32_at = |o: usize| i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let (w, h, bands, fmt) = (i32_at(4), i32_at(8), i32_at(12), i32_at(20));
    assert_eq!(
        (bands, fmt),
        (4, 6),
        "{id}: expected 4-band float (BandFmt 6)"
    );
    let n = (w * h * 4) as usize;
    let px = b[64..64 + n * 4]
        .chunks_exact(16)
        .map(|p| {
            std::array::from_fn(|c| f32::from_le_bytes(p[c * 4..c * 4 + 4].try_into().unwrap()))
        })
        .collect();
    (w as u32, h as u32, px)
}

// ──────────────────────────── the engine ─────────────────────────────

/// How a row runs on our side.
struct Row {
    def: &'static KernelDef,
    params: Vec<u8>,
    inputs: Vec<&'static str>,
    /// `Some((w, h))` = a windowed/resample module over the whole
    /// stimulus as its window, producing this extent.
    window_out: Option<(u32, u32)>,
    /// The scalar reference, where `image-kernels` has one.
    reference: Option<Box<dyn Fn(Px, Px) -> Px>>,
}

fn bytes<P: bytemuck::Pod>(p: &P) -> Vec<u8> {
    bytemuck::bytes_of(p).to_vec()
}

macro_rules! point {
    ($def:expr, $p:expr, [$($k:literal),*], $r:path) => {{
        let p = $p;
        Row {
            def: &$def,
            params: bytes(&p),
            inputs: vec![$($k),*],
            window_out: None,
            reference: Some(Box::new(move |a, b| $r(a, b, &p))),
        }
    }};
}

fn module(
    def: &'static KernelDef,
    params: Vec<u8>,
    kind: &'static str,
    out: Option<(u32, u32)>,
) -> Row {
    Row {
        def,
        params,
        inputs: vec![kind],
        window_out: out,
        reference: None,
    }
}

fn row(id: &str) -> Row {
    use arithmetic::*;
    use band::*;
    use boolean::*;
    use linear::*;
    use minmax::*;
    use relational::*;
    match id {
        "math.linear" => point!(
            MATH_LINEAR,
            MathLinearParams::new(1.5, -0.125),
            ["unit"],
            math_linear
        ),
        "math.invert" => point!(MATH_INVERT, MathInvertParams::new(), ["unit"], math_invert),
        "math.add" => point!(MATH_ADD, MathAddParams::new(), ["unit", "unit"], math_add),
        "math.sub" => point!(MATH_SUB, MathSubParams::new(), ["unit", "unit"], math_sub),
        "math.mul" => point!(MATH_MUL, MathMulParams::new(), ["unit", "unit"], math_mul),
        "math.div" => point!(MATH_DIV, MathDivParams::new(), ["unit", "pos"], math_div),
        "math.add_const" => point!(
            MATH_ADD_CONST,
            MathAddConstParams::new(0.375),
            ["unit"],
            math_add_const
        ),
        "math.mul_const" => point!(
            MATH_MUL_CONST,
            MathMulConstParams::new(1.75),
            ["unit"],
            math_mul_const
        ),
        "math.abs" => point!(MATH_ABS, MathAbsParams::new(), ["signed"], math_abs),
        "math.sign" => point!(MATH_SIGN, MathSignParams::new(), ["signed"], math_sign),
        "math.neg" => point!(MATH_NEG, MathNegParams::new(), ["signed"], math_neg),
        "rel.eq" => point!(REL_EQ, RelEqParams::new(), ["quarter", "quarter"], rel_eq),
        "rel.ne" => point!(REL_NE, RelNeParams::new(), ["quarter", "quarter"], rel_ne),
        "rel.lt" => point!(REL_LT, RelLtParams::new(), ["quarter", "quarter"], rel_lt),
        "rel.le" => point!(REL_LE, RelLeParams::new(), ["quarter", "quarter"], rel_le),
        "rel.gt" => point!(REL_GT, RelGtParams::new(), ["quarter", "quarter"], rel_gt),
        "rel.ge" => point!(REL_GE, RelGeParams::new(), ["quarter", "quarter"], rel_ge),
        "bool.and" => point!(
            BOOL_AND,
            BoolAndParams::new(),
            ["binary", "binary"],
            bool_and
        ),
        "bool.or" => point!(BOOL_OR, BoolOrParams::new(), ["binary", "binary"], bool_or),
        "bool.xor" => point!(
            BOOL_XOR,
            BoolXorParams::new(),
            ["binary", "binary"],
            bool_xor
        ),
        "bool.not" => point!(BOOL_NOT, BoolNotParams::new(), ["binary"], bool_not),
        "band.extract" => point!(
            BAND_EXTRACT,
            BandExtractParams::new(1),
            ["unit"],
            band_extract
        ),
        "band.set_alpha" => point!(
            BAND_SET_ALPHA,
            BandSetAlphaParams::new(0.5),
            ["unit"],
            band_set_alpha
        ),
        "math.min" => point!(MATH_MIN, MathMinParams::new(), ["unit", "unit"], math_min),
        "math.max" => point!(MATH_MAX, MathMaxParams::new(), ["unit", "unit"], math_max),
        "math.clamp" => point!(
            MATH_CLAMP,
            MathClampParams::new(0.25, 0.75),
            ["unit"],
            math_clamp
        ),
        "math.min_const" => point!(
            MATH_MIN_CONST,
            MathMinConstParams::new(0.5),
            ["unit"],
            math_min_const
        ),
        "math.max_const" => point!(
            MATH_MAX_CONST,
            MathMaxConstParams::new(0.5),
            ["unit"],
            math_max_const
        ),
        "conv.box" => module(
            &conv::CONV_BOX,
            bytes(&conv::ConvBoxParams::new()),
            "unit",
            Some((62, 62)),
        ),
        "conv.gaussian_h" => module(
            &conv::CONV_GAUSSIAN_H,
            bytes(&conv::ConvGaussianParams::new(1.5, 5)),
            "unit",
            Some((16, 64)),
        ),
        "conv.gaussian_v" => module(
            &conv::CONV_GAUSSIAN_V,
            bytes(&conv::ConvGaussianParams::new(1.5, 5)),
            "unit",
            Some((64, 16)),
        ),
        "conv.unsharp" => Row {
            def: &conv::CONV_UNSHARP,
            params: bytes(&conv::ConvUnsharpParams::new(0.8, 0.0)),
            inputs: vec!["unit", "unit"],
            window_out: None,
            reference: None,
        },
        "resample.nearest" => module(
            &resample::RESAMPLE_NEAREST,
            bytes(&resample::ResampleParams::new(0.5, 0.5, 0.0, 0.0)),
            "unit",
            Some((128, 128)),
        ),
        "resample.mitchell" => module(
            &resample::RESAMPLE_MITCHELL,
            bytes(&resample::ResampleParams::new(2.0, 2.0, 0.0, 0.0)),
            "unit",
            Some((32, 32)),
        ),
        "resample.lanczos3" => module(
            &resample::RESAMPLE_LANCZOS3,
            bytes(&resample::ResampleParams::new(2.0, 2.0, 0.0, 0.0)),
            "unit",
            Some((32, 32)),
        ),
        "geom.flip_h" => module(
            &geom::GEOM_FLIP_H,
            bytes(&geom::FlipHParams::new(N)),
            "unit",
            Some((N, N)),
        ),
        "geom.flip_v" => module(
            &geom::GEOM_FLIP_V,
            bytes(&geom::FlipVParams::new(N)),
            "unit",
            Some((N, N)),
        ),
        "geom.rotate90_cw" => module(
            &geom::GEOM_ROTATE90_CW,
            bytes(&geom::Rotate90Params::new(N, N)),
            "unit",
            Some((N, N)),
        ),
        "geom.rotate90_ccw" => module(
            &geom::GEOM_ROTATE90_CCW,
            bytes(&geom::Rotate90Params::new(N, N)),
            "unit",
            Some((N, N)),
        ),
        "geom.crop" => module(
            &geom::GEOM_CROP,
            bytes(&geom::CropParams::new(8, 4)),
            "unit",
            Some((40, 32)),
        ),
        "geom.rotate_bilinear" => module(
            &geom::GEOM_ROTATE_BILINEAR,
            bytes(&geom::RotateBilinearParams::new(
                30.0,
                (32.0, 32.0),
                (32.0, 32.0),
            )),
            "unit",
            Some((N, N)),
        ),
        "morph.dilate" => module(
            &morph::MORPH_DILATE,
            bytes(&morph::MorphParams::new()),
            "unit",
            Some((62, 62)),
        ),
        "morph.erode" => module(
            &morph::MORPH_ERODE,
            bytes(&morph::MorphParams::new()),
            "unit",
            Some((62, 62)),
        ),
        "rank.median3" => module(
            &morph::RANK_MEDIAN3,
            bytes(&morph::MorphParams::new()),
            "unit",
            Some((62, 62)),
        ),
        "adjust.exposure" => module(
            &adjust::ADJUST_EXPOSURE,
            bytes(&adjust::AdjustExposureParams::new(0.5)),
            "opaque",
            None,
        ),
        "adjust.brightness_contrast" => module(
            &adjust::ADJUST_BRIGHTNESS_CONTRAST,
            bytes(&adjust::AdjustBrightnessContrastParams::new(0.1, 1.3)),
            "opaque",
            None,
        ),
        "adjust.levels" => module(
            &adjust::ADJUST_LEVELS,
            bytes(&adjust::AdjustLevelsParams::new(0.1, 0.9, 1.3, 0.05, 0.95)),
            "opaque",
            None,
        ),
        "adjust.levels_rgb" => module(
            &adjust::ADJUST_LEVELS_RGB,
            bytes(&adjust::AdjustLevelsRgbParams::new(
                [0.05, 0.95, 1.2],
                [0.1, 0.9, 0.8],
                [0.0, 1.0, 1.5],
            )),
            "opaque",
            None,
        ),
        "adjust.saturation" => module(
            &adjust::ADJUST_SATURATION,
            bytes(&adjust::AdjustSaturationParams::new(0.6)),
            "opaque",
            None,
        ),
        "adjust.hue_rotate" => module(
            &adjust::ADJUST_HUE_ROTATE,
            bytes(&adjust::AdjustHueRotateParams::new(30.0)),
            "opaque",
            None,
        ),
        "adjust.invert_rgb" => module(
            &adjust::ADJUST_INVERT_RGB,
            bytes(&adjust::AdjustInvertRgbParams::new()),
            "opaque",
            None,
        ),
        "adjust.white_balance" => module(
            &adjust::ADJUST_WHITE_BALANCE,
            bytes(&adjust::AdjustWhiteBalanceParams::new(0.3, -0.2)),
            "opaque",
            None,
        ),
        other => panic!("no engine mapping for vips row {other}"),
    }
}

fn f16_bytes(px: &[Px]) -> Vec<u8> {
    px.iter()
        .flat_map(|p| p.0)
        .flat_map(|c| f16::from_f32(c).to_bits().to_le_bytes())
        .collect()
}

fn quantized(px: &[Px]) -> Vec<Px> {
    px.iter()
        .map(|p| Px(p.0.map(|c| f16::from_f32(c).to_f32())))
        .collect()
}

/// The reference over the stimuli (f16-quantized, as the GPU sees them).
fn run_reference(r: &Row) -> Option<(u32, u32, Vec<[f32; 4]>)> {
    let f = r.reference.as_ref()?;
    let ins: Vec<Vec<Px>> = r
        .inputs
        .iter()
        .enumerate()
        .map(|(i, k)| quantized(&stimulus(k, i as u32 + 1)))
        .collect();
    let zero = Px([0.0; 4]);
    let out = (0..(N * N) as usize)
        .map(|i| f(ins[0][i], ins.get(1).map_or(zero, |b| b[i])).0)
        .collect();
    Some((N, N, out))
}

fn run_gpu(r: &Row) -> Option<(u32, u32, Vec<[f32; 4]>)> {
    let ctx = image_conformance::device::test_device()?;
    let ins: Vec<Vec<u8>> = r
        .inputs
        .iter()
        .enumerate()
        .map(|(i, k)| f16_bytes(&stimulus(k, i as u32 + 1)))
        .collect();
    let (w, h, out) = match r.window_out {
        Some((w, h)) => (
            w,
            h,
            image_gpu::execute_windowed_once(ctx, r.def, &ins[0], N, N, &r.params, None, w, h),
        ),
        None => {
            let tiles: Vec<image_gpu::TileInput<'_>> = ins
                .iter()
                .map(|b| image_gpu::TileInput { f16_bytes: b })
                .collect();
            (
                N,
                N,
                image_gpu::execute_tile_once(ctx, r.def, &tiles, &r.params, None, N, N),
            )
        }
    };
    let out = out.unwrap_or_else(|e| panic!("{}: {e}", r.def.id));
    let px = out
        .chunks_exact(8)
        .map(|t| std::array::from_fn(|c| f16::from_le_bytes([t[c * 2], t[c * 2 + 1]]).to_f32()))
        .collect();
    Some((w, h, px))
}

// ───────────────────────────── the table ─────────────────────────────

struct Entry {
    fixture: bool,
    diverges: Option<String>,
    oracle_tol: Option<f64>,
    gpu_tol: Option<f64>,
}

fn table_path() -> PathBuf {
    root().join("../registry/vips-oracle.yaml")
}

/// `registry/vips-oracle.yaml`, line-scanned (the rows are flat).
fn table() -> BTreeMap<String, Entry> {
    let text = std::fs::read_to_string(table_path()).expect("read registry/vips-oracle.yaml");
    let mut out = BTreeMap::new();
    let mut id: Option<String> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("- id: ") {
            id = Some(rest.trim().to_string());
            out.insert(
                rest.trim().to_string(),
                Entry {
                    fixture: false,
                    diverges: None,
                    oracle_tol: None,
                    gpu_tol: None,
                },
            );
            continue;
        }
        let Some(e) = id.as_ref().and_then(|i| out.get_mut(i)) else {
            continue;
        };
        let t = line.trim();
        if t.starts_with("fixture: ") {
            e.fixture = true;
        } else if let Some(r) = t.strip_prefix("diverges: ") {
            e.diverges = Some(r.trim_matches('"').to_string());
        } else if let Some(r) = t.strip_prefix("oracle_tolerance: ") {
            e.oracle_tol = r.parse().ok();
        } else if let Some(r) = t.strip_prefix("gpu_tolerance_measured: ") {
            e.gpu_tol = r.parse().ok();
        }
    }
    out
}

/// Ids whose kernels.yaml row says `oracle: vips` or `oracle: both`.
fn registry_vips_ids() -> Vec<String> {
    let text =
        std::fs::read_to_string(root().join("../registry/kernels.yaml")).expect("kernels.yaml");
    let mut out = Vec::new();
    let mut id: Option<String> = None;
    for line in text.lines() {
        let t = line.trim();
        if let Some(r) = t.strip_prefix("- id: ") {
            id = Some(r.trim().to_string());
        } else if t == "oracle: vips" || t == "oracle: both" {
            if let Some(i) = id.take() {
                out.push(i);
            }
        }
    }
    out
}

/// Fixture region (x, y, w, h) within our output, from the provenance.
fn region(id: &str, w: u32, h: u32) -> (u32, u32, u32, u32) {
    let text = std::fs::read_to_string(root().join(format!("fixtures/vips/{id}.vips.json")))
        .expect("provenance");
    let j = image_conformance::abr_corpus::json::Json::parse(&text).expect("json");
    match j.get("region").and_then(|r| r.as_array()) {
        Some(a) if a.len() == 4 => {
            let v: Vec<u32> = a.iter().map(|x| x.as_u64().expect("int") as u32).collect();
            (v[0], v[1], v[2], v[3])
        }
        _ => (0, 0, w, h),
    }
}

/// max |ours − vips| over the compared region.
fn distance(id: &str, ours: &(u32, u32, Vec<[f32; 4]>)) -> f64 {
    let (vw, vh, theirs) = read_v(id);
    let (w, h, px) = ours;
    assert_eq!((*w, *h), (vw, vh), "{id}: our extent vs vips's");
    let (rx, ry, rw, rh) = region(id, *w, *h);
    let mut worst = 0f64;
    for y in ry..ry + rh {
        for x in rx..rx + rw {
            let i = (y * w + x) as usize;
            for c in 0..4 {
                worst = worst.max((px[i][c] as f64 - theirs[i][c] as f64).abs());
            }
        }
    }
    worst
}

/// A measurement rounded UP to two significant figures, as text
/// (`49e-5`), so the table never carries float noise.
fn ceil_tol(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    let e = v.log10().floor() as i32 - 1;
    let k = (v / 10f64.powi(e)).ceil() as i64;
    format!("{k}e{e}")
}

/// A GPU re-measurement may exceed the recorded value by one f16 ULP at
/// magnitude 1-2: adapters differ in the last bit of transcendental and
/// divide results, and CI runs a software adapter.
const GPU_SLACK: f64 = 1.0 / 1024.0;

fn write_tolerances(measured: &BTreeMap<String, (Option<f64>, Option<f64>)>) {
    let text = std::fs::read_to_string(table_path()).expect("read table");
    let mut out = String::new();
    let mut id = String::new();
    for line in text.lines() {
        if let Some(r) = line.strip_prefix("- id: ") {
            id = r.trim().to_string();
        }
        let m = measured.get(&id);
        if line.starts_with("  oracle_tolerance: ") {
            match m.and_then(|m| m.0) {
                Some(v) => out.push_str(&format!("  oracle_tolerance: {}\n", ceil_tol(v))),
                None => out
                    .push_str("  oracle_tolerance: none (no scalar reference in image-kernels)\n"),
            }
        } else if line.starts_with("  gpu_tolerance_measured: ") {
            match m.and_then(|m| m.1) {
                Some(v) => out.push_str(&format!("  gpu_tolerance_measured: {}\n", ceil_tol(v))),
                None => {
                    out.push_str(line);
                    out.push('\n');
                }
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    std::fs::write(table_path(), out).expect("write table");
}

#[test]
#[allow(non_snake_case)]
fn vips_oracle_table_covers_every_vips_row__feat__image_conformance_harness() {
    let t = table();
    let reg = registry_vips_ids();
    let mut problems = Vec::new();
    for id in &reg {
        match t.get(id) {
            None => problems.push(format!(
                "{id}: oracle vips in kernels.yaml but no vips-oracle.yaml row"
            )),
            Some(e) if e.fixture == e.diverges.is_some() => problems.push(format!(
                "{id}: a row is EITHER a fixture or a diverges reason"
            )),
            Some(e) if e.fixture && !root().join(format!("fixtures/vips/{id}.v")).is_file() => {
                problems.push(format!("{id}: fixture missing"))
            }
            Some(_) => {}
        }
    }
    for id in t.keys() {
        if !reg.contains(id) {
            problems.push(format!(
                "{id}: in vips-oracle.yaml but not an oracle: vips row"
            ));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
#[allow(non_snake_case)]
fn engine_matches_libvips_within_the_measured_tolerance__feat__image_kernel_family_t0() {
    let write = std::env::var_os("PAGED_VIPS_TOLERANCE").is_some_and(|v| v == "write");
    let t = table();
    let mut measured = BTreeMap::new();
    let mut problems = Vec::new();
    for (id, e) in &t {
        if !e.fixture {
            continue;
        }
        let r = row(id);
        let ref_d = run_reference(&r).map(|o| distance(id, &o));
        let gpu_d = run_gpu(&r).map(|o| distance(id, &o));
        println!("{id:<28} reference {ref_d:?}  gpu {gpu_d:?}");
        if let (Some(d), Some(tol)) = (ref_d, e.oracle_tol) {
            if d > tol {
                problems.push(format!(
                    "{id}: reference is {d:.3e} from vips (tolerance {tol})"
                ));
            }
        }
        if let (Some(d), Some(tol)) = (gpu_d, e.gpu_tol) {
            if d > tol + GPU_SLACK {
                problems.push(format!("{id}: GPU is {d:.3e} from vips (measured {tol})"));
            }
        }
        if ref_d.is_some() && e.oracle_tol.is_none() && !write {
            problems.push(format!("{id}: reference distance never measured"));
        }
        measured.insert(id.clone(), (ref_d, gpu_d));
    }
    if write {
        write_tolerances(&measured);
        return;
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
