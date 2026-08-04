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

//! The typed brush-preset view over a `.abr` descriptor tree.
//!
//! **The descriptor tree is authoritative; this is a derived view.**
//! Every [`AbrBrush`] keeps its complete [`Descriptor`], so a key this
//! module does not model is not lost — that discipline is why the
//! fixture experiment behind the behaviour spec could find two keys
//! (`Rpt`, `brushGroup`) that every previously-published key vocabulary
//! was missing (spec §7.1). A fixed struct silently drops what it does
//! not know; this one cannot.
//!
//! # Provenance
//!
//! `thoughts/docs/paged/plugin-image/abr-brush-format-spec.md` §4.2
//! (unit rules), §5 (tips), §6 (dynamics), §7 (brush-level keys and
//! tool options), §9 (blend modes). Confidence tags are carried per item
//! in the doc comments below: `[OBS]` facts were measured on 3,215 real
//! brush presets, `[PUB]` come from Adobe's own specification, and
//! `[REF]` items rest on a single third-party channel and are called out
//! as unverified wherever this module relies on one.

use crate::descriptor::{units, Descriptor, Key};

use super::blend::BlendMode;
use super::AbrWarning;

/// The five `#Prc` keys that are stored as **0..1 fractions already**,
/// not on the 0..100 scale every other percent uses (spec §4.2
/// CONTRADICTION, §5.4 — `[OBS]`, though on only 3 bristle tips).
///
/// This is the single most destructive unit error available: read as
/// 0..100 and divided, a bristle `Lngt` of 1.37 (137%) becomes 1.37% —
/// an order of magnitude below Photoshop's documented 25% floor. The
/// *fact that the scales differ* is established; the membership of this
/// list is provisional on a thin corpus and is deliberately a named
/// constant so a future fixture can correct it in one place.
pub const FRACTION_SCALE_PERCENT_KEYS: [&[u8]; 5] =
    [b"Dnst", b"Lngt", b"clumping", b"thickness", b"stiffness"];

/// The value an ABSENT percent is assumed to take.
///
/// `[REF]` and **unverifiable from files** — absence tells you nothing
/// about the intended default (spec §4.2, §14.2 item 12). It is a named,
/// documented, testable constant precisely because it is a guess.
///
/// Two mitigations shrink its blast radius. The gate discipline (§6.2
/// `[OBS]`, 3,215/3,215) means an optional percent is essentially never
/// read on a live path. And where it genuinely could bite — `Mnm `, the
/// dynamics floor — this module does **not** apply it: [`Dynamics::minimum`]
/// stays `None`, because 38 of the 39 explicit `Mnm ` values in the
/// corpus are `0.0`, which is suggestive evidence *against* a 100%
/// default, and a renderer choosing its own floor is more honest than a
/// parser inventing one.
pub const ABSENT_PERCENT_DEFAULT: f64 = 1.0;

/// `toolOptions`' flow/opacity default when the key is absent (`[REF]`,
/// spec §7.2 — "absent means full, not none"). These are 0..100 plain
/// numbers, not unit floats.
pub const TOOL_OPTIONS_ABSENT_DEFAULT: f64 = 100.0;

/// Convert a raw `#Prc` value to a fraction, **per key**.
///
/// A blanket ÷100 is wrong (spec §4.2). Results are NOT clamped to
/// `[0, 1]`: observed raw values include `tiltScale` 200.0 and scatter
/// `jitter` 368.0, so percent-derived values legitimately exceed 1.
pub fn percent_to_fraction(key: &[u8], raw: f64) -> f64 {
    if FRACTION_SCALE_PERCENT_KEYS.contains(&key) {
        raw
    } else {
        raw / 100.0
    }
}

// ── shared tip keys (§5.1) ───────────────────────────────────────────

/// The six keys every tip variant carries — `[OBS]` on 3,218/3,218 tip
/// descriptors with exactly these types and units.
#[derive(Debug, Clone, PartialEq)]
pub struct TipCommon {
    /// `Dmtr` as stored. The unit was `#Pxl` in 3,218/3,218 cases; a
    /// non-pixel unit is kept verbatim and reported rather than
    /// rejected (spec §4.2: the strict unit check "buys nothing"), so
    /// check [`TipCommon::diameter_unit`] before treating this as px.
    pub diameter: f64,
    pub diameter_unit: [u8; 4],
    /// `Angl` in DEGREES. The unit is certain; the **sign convention and
    /// the composition order with `flip_x`/`flip_y` are `[UNSURE]`** —
    /// they are rendering semantics and cannot be settled by parsing
    /// (spec §5.1, §14.2 item 9).
    pub angle_deg: f64,
    /// `Spcn` as a FRACTION of the diameter (raw ÷ 100). This is the
    /// SHAPE-level spacing, the one that carries information; the
    /// brush-level `Spcn` is preserved on [`AbrBrush::brush_spacing`]
    /// and ignored (spec §5.1 — it occurred on 4 of 3,215 brushes and
    /// was the no-op 100.0 every time).
    pub spacing: f64,
    /// `Intr` — the **spacing-enabled** flag. The name suggests nothing
    /// of the sort; when false Photoshop spaces stamps by cursor
    /// movement instead of by `spacing` (spec §5.1 trap).
    pub spacing_enabled: bool,
    /// `flipX` on the SHAPE descriptor: a static mirror of the tip.
    /// Not to be confused with the brush-level `flipX`, which is the
    /// Flip X *Jitter* toggle (spec §6.2 trap; both occur in one file).
    pub flip_x: bool,
    pub flip_y: bool,
}

/// A `computedBrush` — the parametric elliptical "New Brush" tip.
#[derive(Debug, Clone, PartialEq)]
pub struct ComputedTip {
    pub common: TipCommon,
    /// `Rndn` as a fraction — minor/major axis ratio; 1.0 = circular.
    pub roundness: f64,
    /// `Hrdn` as a fraction — 1.0 = hard edge, 0.0 = maximally soft.
    ///
    /// **The falloff curve between the two is `[UNSURE]`** — it is
    /// Photoshop-internal and no source describes it (spec §5.2, §14.2
    /// item 10). Parsing gives the parameter; matching Photoshop's
    /// gradient is a rendering decision the brush engine owns.
    pub hardness: f64,
}

/// A `sampledBrush` — a bitmap tip, linked to a `samp` record.
#[derive(Debug, Clone, PartialEq)]
pub struct SampledTipRef {
    pub common: TipCommon,
    /// `Rndn` as a fraction, applied ON TOP OF the bitmap. Was 100.0 on
    /// 3,205 of 3,205 sampled tips, so this path has essentially no
    /// field coverage (spec §5.3).
    pub roundness: f64,
    /// `Nm  ` on the tip descriptor (distinct from the preset's name).
    pub name: Option<String>,
    /// `sampledData` — the UUID that joins to a `samp` record's id.
    pub sampled_id: String,
    /// Index into [`super::AbrFile::samples`], resolved by EXACT,
    /// case-sensitive string equality (spec §3.3 `[OBS]`, 3,205/3,205).
    /// `None` means the reference did not resolve — reported, never
    /// papered over with a positional guess.
    pub sample_index: Option<usize>,
}

/// Bristle shape, by position in the implicit table (spec §5.4).
///
/// Indices 0 and 6 are corroborated by Photoshop's own stock preset
/// names (`Round Point Stiff` → 0, `Flat Blunt Short Stiff` → 6, and
/// index 6 is only correct if the five before it are in the claimed
/// order). The remainder is inferred from the ordering those two pin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BristleShape {
    RoundPoint,
    RoundBlunt,
    RoundCurve,
    RoundAngle,
    RoundFan,
    FlatPoint,
    FlatBlunt,
    FlatCurve,
    FlatAngle,
    FlatFan,
}

impl BristleShape {
    /// `Shp ` is a plain integer indexing a table that exists nowhere in
    /// the file. Out of range yields `None` — never an out-of-bounds
    /// index (spec §5.4 trap).
    pub fn from_ordinal(o: i32) -> Option<BristleShape> {
        Some(match o {
            0 => BristleShape::RoundPoint,
            1 => BristleShape::RoundBlunt,
            2 => BristleShape::RoundCurve,
            3 => BristleShape::RoundAngle,
            4 => BristleShape::RoundFan,
            5 => BristleShape::FlatPoint,
            6 => BristleShape::FlatBlunt,
            7 => BristleShape::FlatCurve,
            8 => BristleShape::FlatAngle,
            9 => BristleShape::FlatFan,
            _ => return None,
        })
    }
}

/// A `dBrush` — the bristle brush.
#[derive(Debug, Clone, PartialEq)]
pub struct BristleTip {
    pub common: TipCommon,
    /// `Shp ` as stored.
    pub shape_ordinal: i32,
    /// The resolved shape, or `None` when the ordinal is out of range.
    pub shape: Option<BristleShape>,
    /// The five 0..1-fraction percents (see
    /// [`FRACTION_SCALE_PERCENT_KEYS`]). Values above 1 occur and are
    /// correct — `Lngt` 1.37 is 137%.
    pub density: f64,
    pub length: f64,
    pub clumping: f64,
    pub thickness: f64,
    pub stiffness: f64,
    pub physics: bool,
}

/// The erodible-tip height map: `grid_size² × 4` bytes of
/// **little-endian** float32 in 0..1 (spec §5.5 RESOLVED `[OBS]`).
///
/// This is the one place in `.abr` where the format's otherwise
/// universal big-endian rule does not hold: `tdta` payloads are opaque
/// blobs whose interior follows the producing subsystem's convention,
/// and this one used the host's. Decoding these bytes big-endian yields
/// `4.6e-41` and `4.9e+27`; the endianness is not a judgement call.
///
/// Row order (row-major vs column-major) and the wear semantics remain
/// unknown (§14.2 item 15), which is why [`HeightMap::raw`] is retained.
#[derive(Debug, Clone, PartialEq)]
pub struct HeightMap {
    pub grid_size: u32,
    pub values: Vec<f32>,
    /// The `tdta` payload verbatim.
    pub raw: Vec<u8>,
}

/// The airbrush parameters of a `dTips` tip.
///
/// The four percent-typed members are stored RAW and deliberately not
/// converted: only `0.0` and `1.0` were ever observed, which is
/// consistent with both the 0..100 and the 0..1 reading, so the scale is
/// genuinely undetermined (spec §5.5 NOTE, §14.2 item 16). Applying
/// ÷100 here would be inventing an answer.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Airbrush {
    /// `dtipsAirbrushCutoffAngle` — a bare `doub`, NOT a `#Ang` unit
    /// float, even though it is an angle (spec §5.5 trap `[OBS]`).
    pub cutoff_angle_deg: Option<f64>,
    pub granularity_raw: Option<f64>,
    pub streakiness_raw: Option<f64>,
    pub splat_size_raw: Option<f64>,
    /// A bare `long`.
    pub splat_count: Option<i32>,
}

/// A `dTips` — erodible and airbrush tips.
#[derive(Debug, Clone, PartialEq)]
pub struct ErodibleTip {
    pub common: TipCommon,
    /// `Shp ` as stored. The corpus contradicts the third-party
    /// reference here, which indexes this into the BRISTLE table: the
    /// preset named `Square Charcoal` carries 3, and erodible index 3 is
    /// *square* while bristle index 3 is *round angle* (spec §5.5
    /// CONTRADICTION). But the erodible table's exact membership is
    /// still `[UNSURE]` on three tips, so **no name is surfaced** — the
    /// integer is retained faithfully and nothing is resolved from it.
    pub shape_ordinal: i32,
    /// `dtipsType`. Correlates perfectly with the presence of the height
    /// map (0 ⇒ present, 1 ⇒ absent), which reads as an
    /// erodible-vs-airbrush FAMILY selector rather than a shape ordinal.
    /// Also retained, also unresolved (§14.2 item 14).
    pub tips_type: Option<i32>,
    pub physics: Option<bool>,
    pub length_ratio: Option<f64>,
    pub hardness: Option<f64>,
    /// `dtipsGridSize`. Coupled to the height map: both present together
    /// or both absent (spec §5.5 trap).
    pub grid_size: Option<i32>,
    pub height_map: Option<HeightMap>,
    pub airbrush: Airbrush,
}

/// The tip, dispatched by the nested descriptor's CLASS ID — there is no
/// variant field (spec §5 `[OBS]`, all four class ids observed).
#[derive(Debug, Clone, PartialEq)]
pub enum AbrTip {
    Computed(ComputedTip),
    Sampled(SampledTipRef),
    Bristle(BristleTip),
    Erodible(ErodibleTip),
    /// An unrecognised class id. The preset is RETAINED as metadata
    /// rather than poisoning the file — one exotic brush must not take
    /// out a 500-brush library (spec §5 trap).
    Unsupported {
        class_id: String,
    },
    /// The brush descriptor had no `Brsh` sub-descriptor at all.
    Missing,
}

impl AbrTip {
    pub fn common(&self) -> Option<&TipCommon> {
        Some(match self {
            AbrTip::Computed(t) => &t.common,
            AbrTip::Sampled(t) => &t.common,
            AbrTip::Bristle(t) => &t.common,
            AbrTip::Erodible(t) => &t.common,
            AbrTip::Unsupported { .. } | AbrTip::Missing => return None,
        })
    }

    pub fn as_sampled(&self) -> Option<&SampledTipRef> {
        match self {
            AbrTip::Sampled(t) => Some(t),
            _ => None,
        }
    }

    fn as_sampled_mut(&mut self) -> Option<&mut SampledTipRef> {
        match self {
            AbrTip::Sampled(t) => Some(t),
            _ => None,
        }
    }
}

// ── the dynamics primitive (§6.1) ────────────────────────────────────

/// The control source of a dynamic, by position in an implicit table.
///
/// **The ORDERING of this table is `[REF]` — reference-only.** The
/// observed ordinals (0, 2, 3, 4) are all in range, but nothing in any
/// file names them, and unlike the bristle table there are no
/// self-labelling stock presets to corroborate it (spec §6.1, §14.2
/// item 13). The ordinal itself is retained on [`Dynamics::control_ordinal`]
/// so a consumer that distrusts the naming can work from the integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlSource {
    Off,
    Fade,
    PenPressure,
    PenTilt,
    StylusWheel,
    InitialDirection,
    Direction,
    InitialRotation,
    Rotation,
}

impl ControlSource {
    pub fn from_ordinal(o: i32) -> Option<ControlSource> {
        Some(match o {
            0 => ControlSource::Off,
            1 => ControlSource::Fade,
            2 => ControlSource::PenPressure,
            3 => ControlSource::PenTilt,
            4 => ControlSource::StylusWheel,
            5 => ControlSource::InitialDirection,
            6 => ControlSource::Direction,
            7 => ControlSource::InitialRotation,
            8 => ControlSource::Rotation,
            _ => return None,
        })
    }
}

/// One dynamic parameter. Every dynamics site in the format has this
/// shape, and the file itself asserts it: all 220 observed dynamics
/// descriptors carry class id `brVr` (spec §6.1 `[OBS]`).
///
/// There is **no response curve** anywhere in `.abr` — 102 distinct keys
/// exist in the whole corpus and none is a curve, a point list or a
/// spline (spec §6.1 RESOLVED). A dynamic is exactly these four fields.
#[derive(Debug, Clone, PartialEq)]
pub struct Dynamics {
    /// `bVTy` as stored.
    pub control_ordinal: i32,
    /// The resolved control source; out-of-range degrades to
    /// [`ControlSource::Off`] and is reported.
    pub control: ControlSource,
    /// `fStp` — fade length in stamps, meaningful only under
    /// [`ControlSource::Fade`]. 207 of 220 carry 25.
    pub fade_steps: i32,
    /// `jitter` as a fraction. **Independent of `control`**: 41 dynamics
    /// descriptors carry control `off` together with a non-zero jitter,
    /// the largest 368% (spec §6.1 trap `[OBS]`). Whether Photoshop's
    /// *renderer* honours that combination is a behavioural question the
    /// format cannot answer.
    pub jitter: f64,
    /// `Mnm ` as a fraction — genuinely optional (39 of 220). Left
    /// `None` when absent; see [`ABSENT_PERCENT_DEFAULT`] for why no
    /// default is substituted here.
    pub minimum: Option<f64>,
}

/// Shape dynamics — gate `useTipDynamics`, FLATTENED onto the brush
/// descriptor (spec §6.2). Only the three dynamics primitives nest.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShapeDynamics {
    pub size: Option<Dynamics>,
    pub angle: Option<Dynamics>,
    pub roundness: Option<Dynamics>,
    pub minimum_diameter: Option<f64>,
    pub minimum_roundness: Option<f64>,
    pub tilt_scale: Option<f64>,
    /// The brush-level `flipX`: **Flip X Jitter**, random per-stamp
    /// mirroring — NOT the shape-level static mirror of the same name.
    pub flip_x_jitter: Option<bool>,
    pub flip_y_jitter: Option<bool>,
    pub brush_projection: Option<bool>,
}

/// Scattering — gate `useScatter`, also flattened (spec §6.3).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Scatter {
    pub scatter: Option<Dynamics>,
    pub count_dynamics: Option<Dynamics>,
    /// `Cnt ` — stamps per spacing interval. An 8-byte **`doub`**, not a
    /// `long` (spec §6.3 CONTRADICTION, 17/17): reading four bytes here
    /// desynchronises every subsequent key in the brush. Observed values
    /// are integral, so the meaning is a count — read as a double and
    /// round. NOT a percent.
    pub count: Option<f64>,
    pub both_axes: Option<bool>,
}

/// Texture — gate `useTexture` (spec §6.4).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Texture {
    /// `Txtr/Idnt` — resolves against the `patt` section, which this
    /// reader deliberately retains as opaque bytes (see
    /// [`super::AbrFile::patterns_raw`]), so this dangles by design.
    pub pattern_id: Option<String>,
    /// `Txtr/Nm  ` exactly as stored — frequently an Adobe
    /// `$$$/…=Default` localisation token (spec §6.4 `[OBS]`).
    pub pattern_name_raw: Option<String>,
    /// The same value with the localisation wrapper removed.
    pub pattern_name: Option<String>,
    pub blend_mode: Option<BlendMode>,
    pub scale: Option<f64>,
    pub depth: Option<f64>,
    pub minimum_depth: Option<f64>,
    pub depth_dynamics: Option<Dynamics>,
    /// `InvT`.
    pub invert: Option<bool>,
    /// `textureBrightness` — a **signed** bare `long`, centred on zero
    /// (observed 0, 6, 6, 6, −7), not a percent. Storing it unsigned
    /// loses half the range (spec §6.4 trap `[OBS]`).
    pub brightness: Option<i32>,
    /// `textureContrast` — likewise signed (observed 0, −20, −20, −20, 14).
    pub contrast: Option<i32>,
    /// `TxtC` — apply the texture to each tip separately.
    pub per_tip: Option<bool>,
}

/// Dual brush — the ONE group that is properly nested, and whose gate
/// lives INSIDE the nested descriptor (spec §6.5). The `dualBrush`
/// descriptor is present on every brush whether or not the feature is
/// used (3,215/3,215), so its presence tells you nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct DualBrush {
    /// `useDualBrush`, read from inside the sub-descriptor.
    pub enabled: bool,
    /// The dual brush's OWN complete tip, which can itself be a
    /// `sampledBrush` — so one preset can reference two `samp` records,
    /// and the join must run over every `sampledData` in the tree.
    pub tip: Option<Box<AbrTip>>,
    pub flip: Option<bool>,
    pub blend_mode: Option<BlendMode>,
    pub use_scatter: Option<bool>,
    pub spacing: Option<f64>,
    /// `Cnt ` — read as a number regardless of width. The behaviour
    /// spec's §6.5 table types this `long` while its own §6.3
    /// CONTRADICTION reports `doub` in 17/17 occurrences *including the
    /// 3 inside `dualBrush`*; the `[OBS]` finding wins.
    pub count: Option<f64>,
    pub both_axes: Option<bool>,
    pub count_dynamics: Option<Dynamics>,
    pub scatter_dynamics: Option<Dynamics>,
}

/// Colour dynamics — gate `useColorDynamics` (spec §6.6).
///
/// **The least-verified table in the behaviour spec.** `useColorDynamics`
/// was `false` on all 3,215 corpus brushes, so — per the gate discipline
/// — not one of these six keys was ever emitted and their names, types
/// and units rest entirely on a single third-party channel `[REF]`.
/// Reading them costs nothing and the generic tree is authoritative
/// anyway, but a [`AbrWarning::ColorDynamicsUnverified`] is raised
/// whenever the gate is actually true, so a consumer knows it is
/// standing on unverified ground.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ColorDynamics {
    pub foreground_background: Option<Dynamics>,
    /// `H   ` — hue jitter here, and the blend-mode code for "Hue"
    /// elsewhere. Any code table keyed by 4-char code must be scoped by
    /// context or it mis-resolves these (spec §6.6 trap).
    pub hue: Option<f64>,
    /// `Strt` — saturation jitter (and the blend code for "Saturation").
    pub saturation: Option<f64>,
    pub brightness: Option<f64>,
    pub purity: Option<f64>,
    pub per_tip: Option<bool>,
}

/// Transfer — gate `usePaintDynamics`. Photoshop's UI calls this panel
/// **Transfer**; the format does not (spec §6.7 trap). `wt`/`mx` are
/// mixer-only and are simply absent for non-mixer brushes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Transfer {
    /// `prVr` — FLOW dynamics, not "pressure".
    pub flow: Option<Dynamics>,
    pub opacity: Option<Dynamics>,
    pub wetness: Option<Dynamics>,
    pub mix: Option<Dynamics>,
}

/// Brush pose — gate `useBrushPose` (spec §6.8, one corpus brush).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BrushPose {
    pub override_angle: Option<bool>,
    pub override_tilt_x: Option<bool>,
    pub override_tilt_y: Option<bool>,
    pub override_pressure: Option<bool>,
    /// `brushPosePressure` — a `#Prc` unit float on the 0..100 scale,
    /// stored here as a fraction.
    pub pressure: Option<f64>,
    /// Bare `long`s, while the tip's `Angl` is a `#Ang` unit float —
    /// two angle representations in one file (spec §6.8 trap). NOT a
    /// normalised −1..1: the observed −54 and 23 are degrees on
    /// Photoshop's −60..60 tilt range.
    pub tilt_x: Option<i32>,
    pub tilt_y: Option<i32>,
    pub angle: Option<i32>,
}

/// The tool a `toolOptions` descriptor configures, carried by its CLASS
/// ID rather than a field (spec §7.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    /// `MixB` — `[REF]`, never observed.
    Mixer,
    /// `SmTl` — `[REF]`, never observed.
    Smudge,
    /// Everything else, including the observed `PbTl` paintbrush. The
    /// class-id set is open and the fallback is load-bearing, not
    /// defensive — an unknown tool class id is never an error.
    PlainBrush,
}

/// `toolOptions` — per-brush tool metadata (spec §7.2).
///
/// **Rare and near-untested: 1 occurrence in 3,215 brushes**, on a
/// paintbrush. Only the members the corpus actually pins are typed here;
/// the ~25 remaining `[REF]` keys are reachable through
/// [`ToolOptions::descriptor`] rather than modelled on a single
/// third-party channel's authority.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOptions {
    pub kind: ToolKind,
    pub class_id: String,
    /// `flow` — a 0..100 **plain number**, not a `#Prc` unit float
    /// (spec §7.2 trap `[OBS]`: a bare `long` of 100). Absent means
    /// [`TOOL_OPTIONS_ABSENT_DEFAULT`], i.e. full.
    pub flow: f64,
    /// `Opct` — likewise 0..100 plain.
    pub opacity: f64,
    /// `Md  ` — absent, or an empty identifier, both mean Normal.
    pub blend_mode: BlendMode,
    /// The complete sub-descriptor: everything this view does not model.
    pub descriptor: Descriptor,
}

/// One brush preset.
#[derive(Debug, Clone, PartialEq)]
pub struct AbrBrush {
    /// Position in the `desc` section's `Brsh` list. **Order is
    /// load-bearing** — the `phry` folder tree refers to presets
    /// positionally and carries no id (spec §3.2, §8.2).
    pub index: usize,
    /// `Nm  ` — present on 3,215/3,215.
    pub name: String,
    /// The complete brush descriptor. Authoritative; everything above is
    /// derived from it.
    pub descriptor: Descriptor,
    pub tip: AbrTip,
    /// `Wtdg` — wet edges. Not guessable from the name.
    pub wet_edges: Option<bool>,
    /// `Nose` — noise. Reads like a typo and is not one.
    pub noise: Option<bool>,
    pub use_brush_size: Option<bool>,
    /// The BRUSH-level `Spcn`, preserved for round-trip and otherwise
    /// ignored — the shape-level one carries the value (spec §5.1).
    pub brush_spacing: Option<f64>,
    pub shape_dynamics: Option<ShapeDynamics>,
    pub scatter: Option<Scatter>,
    pub texture: Option<Texture>,
    pub dual_brush: Option<DualBrush>,
    pub color_dynamics: Option<ColorDynamics>,
    pub transfer: Option<Transfer>,
    pub brush_pose: Option<BrushPose>,
    pub tool_options: Option<ToolOptions>,
}

impl AbrBrush {
    /// Every `sampledData` reference in this preset — the top tip's and
    /// the dual brush's — in tree order. The join must run over all of
    /// them, not just the top tip (spec §3.3 NOTE `[OBS]`).
    pub fn sampled_ids(&self) -> Vec<&str> {
        let mut out = Vec::new();
        if let Some(s) = self.tip.as_sampled() {
            out.push(s.sampled_id.as_str());
        }
        if let Some(db) = &self.dual_brush {
            if let Some(t) = &db.tip {
                if let Some(s) = t.as_sampled() {
                    out.push(s.sampled_id.as_str());
                }
            }
        }
        out
    }

    /// Resolve every `sampledData` reference against `index_of`, which
    /// maps a sample id to its position in [`super::AbrFile::samples`].
    ///
    /// EXACT, case-sensitive string equality — no prefix retry, and
    /// emphatically no positional fallback: exact matching resolved
    /// 3,205 of 3,205 references in the corpus, samples are SHARED
    /// (many-to-one), and one fixture has 2,053 brushes against 2,052
    /// samples, so positional pairing would silently mis-pair exactly
    /// the files that matter (spec §3.3).
    pub(crate) fn resolve_samples(
        &mut self,
        index_of: &dyn Fn(&str) -> Option<usize>,
        warnings: &mut Vec<AbrWarning>,
    ) {
        let index = self.index;
        let mut resolve = |tip: &mut AbrTip| {
            if let Some(s) = tip.as_sampled_mut() {
                s.sample_index = index_of(&s.sampled_id);
                if s.sample_index.is_none() {
                    warnings.push(AbrWarning::UnresolvedSampleReference {
                        brush_index: index,
                        id: s.sampled_id.clone(),
                    });
                }
            }
        };
        resolve(&mut self.tip);
        if let Some(db) = &mut self.dual_brush {
            if let Some(t) = &mut db.tip {
                resolve(t);
            }
        }
    }
}

// ── decoding ─────────────────────────────────────────────────────────

fn key_text(k: &Key) -> String {
    k.text_lossy()
}

/// A `#Prc` value as a fraction, scaled per key.
fn percent(d: &Descriptor, key: &[u8], w: &mut Vec<AbrWarning>) -> Option<f64> {
    let v = d.get(key)?;
    match v.as_unit_float() {
        Some((unit, raw)) => {
            if unit != units::PERCENT {
                w.push(AbrWarning::UnexpectedUnit {
                    key: String::from_utf8_lossy(key).into_owned(),
                    unit: String::from_utf8_lossy(&unit).into_owned(),
                });
            }
            Some(percent_to_fraction(key, raw))
        }
        // Not a unit float at all: take the number if there is one and
        // report the shape rather than dropping the value.
        None => v.as_number().map(|raw| {
            w.push(AbrWarning::UnexpectedUnit {
                key: String::from_utf8_lossy(key).into_owned(),
                unit: String::from_utf8_lossy(&v.ostype()).into_owned(),
            });
            percent_to_fraction(key, raw)
        }),
    }
}

/// A `#Ang` value in degrees.
fn angle(d: &Descriptor, key: &[u8], w: &mut Vec<AbrWarning>) -> Option<f64> {
    let v = d.get(key)?;
    match v.as_unit_float() {
        Some((unit, raw)) => {
            if unit != units::ANGLE {
                w.push(AbrWarning::UnexpectedUnit {
                    key: String::from_utf8_lossy(key).into_owned(),
                    unit: String::from_utf8_lossy(&unit).into_owned(),
                });
            }
            Some(raw)
        }
        None => v.as_number(),
    }
}

fn blend_mode(d: &Descriptor, key: &[u8], w: &mut Vec<AbrWarning>) -> Option<BlendMode> {
    let (_type_key, value) = d.enum_value(key)?;
    match BlendMode::from_key(value.as_bytes()) {
        Some(m) => Some(m),
        None => {
            // Never fail the file, and never fail silently: new
            // long-form ids will keep appearing (spec §9 step 4).
            w.push(AbrWarning::UnrecognisedBlendMode {
                key: String::from_utf8_lossy(key).into_owned(),
                value: key_text(value),
            });
            Some(BlendMode::Normal)
        }
    }
}

/// The dynamics primitive (§6.1).
fn dynamics(d: &Descriptor, key: &[u8], w: &mut Vec<AbrWarning>) -> Option<Dynamics> {
    let sub = d.descriptor(key)?;
    if !sub.class_id.matches(b"brVr") {
        w.push(AbrWarning::UnexpectedClassId {
            context: String::from_utf8_lossy(key).into_owned(),
            class_id: key_text(&sub.class_id),
        });
    }
    let control_ordinal = sub.i32(b"bVTy").unwrap_or(0);
    let control = match ControlSource::from_ordinal(control_ordinal) {
        Some(c) => c,
        None => {
            w.push(AbrWarning::OrdinalOutOfRange {
                key: "bVTy".into(),
                value: control_ordinal,
            });
            ControlSource::Off
        }
    };
    Some(Dynamics {
        control_ordinal,
        control,
        fade_steps: sub.i32(b"fStp").unwrap_or(0),
        jitter: percent(sub, b"jitter", w).unwrap_or(0.0),
        minimum: percent(sub, b"Mnm ", w),
    })
}

fn tip_common(d: &Descriptor, w: &mut Vec<AbrWarning>) -> TipCommon {
    let (diameter_unit, diameter) = d
        .unit_float(b"Dmtr")
        .unwrap_or((units::PIXELS, d.number(b"Dmtr").unwrap_or(0.0)));
    if diameter_unit != units::PIXELS {
        w.push(AbrWarning::NonPixelDiameter {
            unit: String::from_utf8_lossy(&diameter_unit).into_owned(),
        });
    }
    TipCommon {
        diameter,
        diameter_unit,
        angle_deg: angle(d, b"Angl", w).unwrap_or(0.0),
        spacing: percent(d, b"Spcn", w).unwrap_or(ABSENT_PERCENT_DEFAULT),
        spacing_enabled: d.bool(b"Intr").unwrap_or(false),
        flip_x: d.bool(b"flipX").unwrap_or(false),
        flip_y: d.bool(b"flipY").unwrap_or(false),
    }
}

/// Decode a tip from its nested `Brsh` descriptor, dispatching on the
/// CLASS ID (spec §5).
pub(crate) fn read_tip(d: &Descriptor, w: &mut Vec<AbrWarning>) -> AbrTip {
    let common = tip_common(d, w);
    let class = d.class_id.as_bytes();
    if class == b"computedBrush" {
        return AbrTip::Computed(ComputedTip {
            common,
            roundness: percent(d, b"Rndn", w).unwrap_or(ABSENT_PERCENT_DEFAULT),
            hardness: percent(d, b"Hrdn", w).unwrap_or(ABSENT_PERCENT_DEFAULT),
        });
    }
    if class == b"sampledBrush" {
        let sampled_id = d.text(b"sampledData").unwrap_or_default().to_string();
        return AbrTip::Sampled(SampledTipRef {
            common,
            roundness: percent(d, b"Rndn", w).unwrap_or(ABSENT_PERCENT_DEFAULT),
            name: d.text(b"Nm  ").map(|s| s.to_string()),
            sampled_id,
            sample_index: None,
        });
    }
    if class == b"dBrush" {
        let shape_ordinal = d.i32(b"Shp ").unwrap_or(0);
        let shape = BristleShape::from_ordinal(shape_ordinal);
        if shape.is_none() {
            w.push(AbrWarning::OrdinalOutOfRange {
                key: "Shp ".into(),
                value: shape_ordinal,
            });
        }
        return AbrTip::Bristle(BristleTip {
            common,
            shape_ordinal,
            shape,
            density: percent(d, b"Dnst", w).unwrap_or(0.0),
            length: percent(d, b"Lngt", w).unwrap_or(0.0),
            clumping: percent(d, b"clumping", w).unwrap_or(0.0),
            thickness: percent(d, b"thickness", w).unwrap_or(0.0),
            stiffness: percent(d, b"stiffness", w).unwrap_or(0.0),
            physics: d.bool(b"physics").unwrap_or(false),
        });
    }
    if class == b"dTips" {
        let grid_size = d.i32(b"dtipsGridSize");
        let height_map = d
            .raw_data(b"dtipsErodibleTipHeightMap")
            .map(|bytes| decode_height_map(bytes, grid_size, w));
        return AbrTip::Erodible(ErodibleTip {
            common,
            shape_ordinal: d.i32(b"Shp ").unwrap_or(0),
            tips_type: d.i32(b"dtipsType"),
            physics: d.bool(b"physics"),
            length_ratio: percent(d, b"dtipsLengthRatio", w),
            hardness: percent(d, b"dtipsHardness", w),
            grid_size,
            height_map,
            airbrush: Airbrush {
                // A bare `doub`, NOT a `#Ang` unit float: applying the
                // angle rule here would reject the file (§5.5 trap).
                cutoff_angle_deg: d.number(b"dtipsAirbrushCutoffAngle"),
                granularity_raw: raw_unit_value(d, b"dtipsAirbrushGranularity"),
                streakiness_raw: raw_unit_value(d, b"dtipsAirbrushStreakiness"),
                splat_size_raw: raw_unit_value(d, b"dtipsAirbrushSplatSize"),
                splat_count: d.i32(b"dtipsAirbrushSplatCount"),
            },
        });
    }
    AbrTip::Unsupported {
        class_id: key_text(&d.class_id),
    }
}

/// The raw value of a unit float, unconverted — for the four airbrush
/// percents whose scale is undetermined.
fn raw_unit_value(d: &Descriptor, key: &[u8]) -> Option<f64> {
    d.get(key).and_then(|v| {
        v.as_unit_float()
            .map(|(_, raw)| raw)
            .or_else(|| v.as_number())
    })
}

fn decode_height_map(bytes: &[u8], grid_size: Option<i32>, w: &mut Vec<AbrWarning>) -> HeightMap {
    // `gridSize² × 4` bytes of LITTLE-ENDIAN float32 (§5.5 RESOLVED).
    let declared = grid_size.filter(|g| *g > 0).map(|g| g as u32);
    let from_len = {
        let n = bytes.len() / 4;
        let root = (n as f64).sqrt().round() as u32;
        (root as usize * root as usize == n).then_some(root)
    };
    let grid = match (declared, from_len) {
        (Some(d), Some(f)) if d != f => {
            w.push(AbrWarning::HeightMapSizeMismatch {
                declared: d,
                implied: f,
                bytes: bytes.len(),
            });
            f
        }
        (Some(d), _) => d,
        (None, Some(f)) => f,
        (None, None) => 0,
    };
    let values = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    HeightMap {
        grid_size: grid,
        values,
        raw: bytes.to_vec(),
    }
}

// ── the gated groups' own vocabularies ───────────────────────────────
//
// Each list is exactly the set of keys its group reads, and each key
// appears in exactly one list. They exist so [`gated`] can report the
// never-observed "group key present, gate false" case by name; they are
// never used to decide whether a group is read — the gate does that.
//
// The two `flipX`/`flipY` entries are the BRUSH-level ones (Flip X/Y
// Jitter) and not the shape-level static mirrors of the same name: the
// latter live on the nested `Brsh` descriptor, which is a different
// descriptor entirely (spec §6.2 trap).

/// `useTipDynamics` (spec §6.2).
const SHAPE_DYNAMICS_KEYS: [&[u8]; 9] = [
    b"szVr",
    b"angleDynamics",
    b"roundnessDynamics",
    b"minimumDiameter",
    b"minimumRoundness",
    b"tiltScale",
    b"flipX",
    b"flipY",
    b"brushProjection",
];

/// `useScatter` (spec §6.3).
const SCATTER_KEYS: [&[u8]; 4] = [b"scatterDynamics", b"countDynamics", b"Cnt ", b"bothAxes"];

/// `useTexture` (spec §6.4).
const TEXTURE_KEYS: [&[u8]; 10] = [
    b"Txtr",
    b"textureBlendMode",
    b"textureScale",
    b"textureDepth",
    b"minimumDepth",
    b"textureDepthDynamics",
    b"InvT",
    b"textureBrightness",
    b"textureContrast",
    b"TxtC",
];

/// `useColorDynamics` (spec §6.6) — the `[REF]`-only vocabulary, never
/// emitted by any corpus file because the gate was false on all 3,215.
const COLOR_DYNAMICS_KEYS: [&[u8]; 6] = [
    b"clVr",
    b"H   ",
    b"Strt",
    b"Brgh",
    b"purity",
    b"colorDynamicsPerTip",
];

/// `usePaintDynamics` — Photoshop's **Transfer** panel (spec §6.7).
const TRANSFER_KEYS: [&[u8]; 4] = [b"prVr", b"opVr", b"wtVr", b"mxVr"];

/// `useBrushPose` (spec §6.8) — the one gate that is ABSENT on 36 of
/// 3,215 brushes, so absence must read as false.
const BRUSH_POSE_KEYS: [&[u8]; 8] = [
    b"overridePoseAngle",
    b"overridePoseTiltX",
    b"overridePoseTiltY",
    b"overridePosePressure",
    b"brushPosePressure",
    b"brushPoseTiltX",
    b"brushPoseTiltY",
    b"brushPoseAngle",
];

/// Read one brush preset from its descriptor.
pub(crate) fn read_brush(d: &Descriptor, index: usize, w: &mut Vec<AbrWarning>) -> AbrBrush {
    if !d.class_id.matches(b"brushPreset") {
        w.push(AbrWarning::UnexpectedClassId {
            context: "Brsh list element".into(),
            class_id: key_text(&d.class_id),
        });
    }
    let tip = match d.descriptor(b"Brsh") {
        Some(t) => read_tip(t, w),
        None => AbrTip::Missing,
    };
    if let AbrTip::Unsupported { class_id } = &tip {
        w.push(AbrWarning::UnsupportedTipClass {
            brush_index: index,
            class_id: class_id.clone(),
        });
    }

    // THE GATE DISCIPLINE (§6.2 [OBS], 3,215/3,215): when a gate is
    // false NONE of its group's keys are present, and when it is true
    // the keys that apply are. So: read the gate, and if it is false
    // skip the group entirely rather than inventing defaults for keys
    // that are not there. A group key present while its gate is false
    // never occurred once and is worth a diagnostic if it ever does.
    let shape_dynamics = gated(d, b"useTipDynamics", &SHAPE_DYNAMICS_KEYS, w, |d, w| {
        ShapeDynamics {
            size: dynamics(d, b"szVr", w),
            angle: dynamics(d, b"angleDynamics", w),
            roundness: dynamics(d, b"roundnessDynamics", w),
            minimum_diameter: percent(d, b"minimumDiameter", w),
            minimum_roundness: percent(d, b"minimumRoundness", w),
            tilt_scale: percent(d, b"tiltScale", w),
            flip_x_jitter: d.bool(b"flipX"),
            flip_y_jitter: d.bool(b"flipY"),
            brush_projection: d.bool(b"brushProjection"),
        }
    });

    let scatter = gated(d, b"useScatter", &SCATTER_KEYS, w, |d, w| Scatter {
        scatter: dynamics(d, b"scatterDynamics", w),
        count_dynamics: dynamics(d, b"countDynamics", w),
        count: d.number(b"Cnt "),
        both_axes: d.bool(b"bothAxes"),
    });

    let texture = gated(d, b"useTexture", &TEXTURE_KEYS, w, |d, w| {
        // `useTexture` true does not guarantee a `Txtr` descriptor
        // exists; both conditions are required (§6.4 trap).
        let txtr = d.descriptor(b"Txtr");
        Texture {
            pattern_id: txtr.and_then(|t| t.text(b"Idnt")).map(str::to_string),
            pattern_name_raw: txtr.and_then(|t| t.text(b"Nm  ")).map(str::to_string),
            pattern_name: txtr
                .and_then(|t| t.text_display(b"Nm  "))
                .map(str::to_string),
            blend_mode: blend_mode(d, b"textureBlendMode", w),
            scale: percent(d, b"textureScale", w),
            depth: percent(d, b"textureDepth", w),
            minimum_depth: percent(d, b"minimumDepth", w),
            depth_dynamics: dynamics(d, b"textureDepthDynamics", w),
            invert: d.bool(b"InvT"),
            brightness: d.i32(b"textureBrightness"),
            contrast: d.i32(b"textureContrast"),
            per_tip: d.bool(b"TxtC"),
        }
    });

    // Dual brush inverts the order: fetch the sub-descriptor FIRST, then
    // check the gate that lives inside it (§6.5).
    let dual_brush = d.descriptor(b"dualBrush").map(|db| {
        let enabled = db.bool(b"useDualBrush").unwrap_or(false);
        DualBrush {
            enabled,
            tip: db.descriptor(b"Brsh").map(|t| Box::new(read_tip(t, w))),
            flip: db.bool(b"Flip"),
            blend_mode: blend_mode(db, b"BlnM", w),
            use_scatter: db.bool(b"useScatter"),
            spacing: percent(db, b"Spcn", w),
            count: db.number(b"Cnt "),
            both_axes: db.bool(b"bothAxes"),
            count_dynamics: dynamics(db, b"countDynamics", w),
            scatter_dynamics: dynamics(db, b"scatterDynamics", w),
        }
    });

    let color_dynamics = gated(d, b"useColorDynamics", &COLOR_DYNAMICS_KEYS, w, |d, w| {
        ColorDynamics {
            foreground_background: dynamics(d, b"clVr", w),
            hue: percent(d, b"H   ", w),
            saturation: percent(d, b"Strt", w),
            brightness: percent(d, b"Brgh", w),
            purity: percent(d, b"purity", w),
            per_tip: d.bool(b"colorDynamicsPerTip"),
        }
    });
    if color_dynamics.is_some() {
        w.push(AbrWarning::ColorDynamicsUnverified { brush_index: index });
    }

    let transfer = gated(d, b"usePaintDynamics", &TRANSFER_KEYS, w, |d, w| Transfer {
        flow: dynamics(d, b"prVr", w),
        opacity: dynamics(d, b"opVr", w),
        wetness: dynamics(d, b"wtVr", w),
        mix: dynamics(d, b"mxVr", w),
    });

    let brush_pose = gated(d, b"useBrushPose", &BRUSH_POSE_KEYS, w, |d, w| BrushPose {
        override_angle: d.bool(b"overridePoseAngle"),
        override_tilt_x: d.bool(b"overridePoseTiltX"),
        override_tilt_y: d.bool(b"overridePoseTiltY"),
        override_pressure: d.bool(b"overridePosePressure"),
        pressure: percent(d, b"brushPosePressure", w),
        tilt_x: d.i32(b"brushPoseTiltX"),
        tilt_y: d.i32(b"brushPoseTiltY"),
        angle: d.i32(b"brushPoseAngle"),
    });

    let tool_options = d.descriptor(b"toolOptions").map(|t| {
        let class = t.class_id.as_bytes();
        let kind = if class == b"MixB" {
            ToolKind::Mixer
        } else if class == b"SmTl" {
            ToolKind::Smudge
        } else {
            // Including the observed `PbTl`: the class-id set is open
            // and unknown ids default to a plain brush, never an error.
            ToolKind::PlainBrush
        };
        ToolOptions {
            kind,
            class_id: key_text(&t.class_id),
            flow: t.number(b"flow").unwrap_or(TOOL_OPTIONS_ABSENT_DEFAULT),
            opacity: t.number(b"Opct").unwrap_or(TOOL_OPTIONS_ABSENT_DEFAULT),
            // Absent, or an empty identifier, both mean Normal.
            blend_mode: blend_mode(t, b"Md  ", w).unwrap_or(BlendMode::Normal),
            descriptor: t.clone(),
        }
    });

    AbrBrush {
        index,
        name: d.text(b"Nm  ").unwrap_or_default().to_string(),
        descriptor: d.clone(),
        tip,
        wet_edges: d.bool(b"Wtdg"),
        noise: d.bool(b"Nose"),
        use_brush_size: d.bool(b"useBrushSize"),
        brush_spacing: percent(d, b"Spcn", w),
        shape_dynamics,
        scatter,
        texture,
        dual_brush,
        color_dynamics,
        transfer,
        brush_pose,
        tool_options,
    }
}

/// Read a flattened group only when its gate is true, and report the
/// (never-observed) case of a group key present while the gate is false.
///
/// `group_keys` is the group's own vocabulary — the keys `read` would
/// consult. It is passed in rather than inferred because a closure
/// cannot be asked what it read, and it is worth passing: the gate
/// discipline (spec §6.2 `[OBS]`, 3,215/3,215) is the reason this reader
/// may skip a whole group on one boolean, so the one observation that
/// would undermine it deserves a diagnostic rather than silence. The
/// gate still decides — a stray key never causes the group to be read.
///
/// A gate that is **absent** counts as false. That is not a guess: 36 of
/// 3,215 corpus brushes carry no `useBrushPose` at all (profile
/// `gate_counts`), and none of them carries a pose key either.
fn gated<T>(
    d: &Descriptor,
    gate: &[u8],
    group_keys: &[&[u8]],
    w: &mut Vec<AbrWarning>,
    read: impl FnOnce(&Descriptor, &mut Vec<AbrWarning>) -> T,
) -> Option<T> {
    if d.bool(gate) == Some(true) {
        return Some(read(d, w));
    }
    let stray: Vec<String> = group_keys
        .iter()
        .filter(|k| d.contains(k))
        .map(|k| String::from_utf8_lossy(k).into_owned())
        .collect();
    if !stray.is_empty() {
        w.push(AbrWarning::GatedGroupKeysWithoutGate {
            gate: String::from_utf8_lossy(gate).into_owned(),
            keys: stray,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_abr_brush_model_percent_scale_splits_bristle_keys_from_the_rest() {
        // Stock bristle values, exactly as they sit in the file.
        assert_eq!(percent_to_fraction(b"Lngt", 1.37), 1.37);
        assert_eq!(percent_to_fraction(b"Dnst", 0.31), 0.31);
        assert_eq!(percent_to_fraction(b"stiffness", 0.85), 0.85);
        // …against the 0..100 siblings in the same file.
        assert_eq!(percent_to_fraction(b"Rndn", 100.0), 1.0);
        assert!((percent_to_fraction(b"tiltScale", 200.0) - 2.0).abs() < 1e-12);
        assert!((percent_to_fraction(b"jitter", 368.0) - 3.68).abs() < 1e-12);
    }

    #[test]
    fn image_abr_brush_model_percent_results_are_not_clamped() {
        // tiltScale 200% and jitter 368% both exceed 1.0 legitimately.
        assert!(percent_to_fraction(b"tiltScale", 200.0) > 1.0);
        assert!(percent_to_fraction(b"jitter", 368.0) > 1.0);
        assert!(percent_to_fraction(b"Lngt", 1.37) > 1.0);
    }

    #[test]
    fn image_abr_brush_model_bristle_shape_table_is_bounds_checked() {
        assert_eq!(
            BristleShape::from_ordinal(0),
            Some(BristleShape::RoundPoint)
        );
        // The strong corroboration: `Flat Blunt Short Stiff` carries 6.
        assert_eq!(BristleShape::from_ordinal(6), Some(BristleShape::FlatBlunt));
        assert_eq!(BristleShape::from_ordinal(9), Some(BristleShape::FlatFan));
        assert_eq!(BristleShape::from_ordinal(10), None);
        assert_eq!(BristleShape::from_ordinal(-1), None);
    }

    #[test]
    fn image_abr_brush_model_control_source_table_is_bounds_checked() {
        assert_eq!(ControlSource::from_ordinal(0), Some(ControlSource::Off));
        assert_eq!(
            ControlSource::from_ordinal(2),
            Some(ControlSource::PenPressure)
        );
        assert_eq!(
            ControlSource::from_ordinal(8),
            Some(ControlSource::Rotation)
        );
        assert_eq!(ControlSource::from_ordinal(9), None);
    }

    #[test]
    fn image_abr_brush_model_absent_percent_default_is_a_named_constant() {
        // The value itself is [REF] and unverified; the point of the
        // test is that it is named, documented and pinned rather than
        // scattered as a literal.
        assert_eq!(ABSENT_PERCENT_DEFAULT, 1.0);
        assert_eq!(TOOL_OPTIONS_ABSENT_DEFAULT, 100.0);
    }

    #[test]
    fn image_abr_height_map_is_little_endian_float32() {
        // 2×2 grid of 1.0 written little-endian.
        let mut bytes = Vec::new();
        for _ in 0..4 {
            bytes.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let mut w = Vec::new();
        let hm = decode_height_map(&bytes, Some(2), &mut w);
        assert_eq!(hm.grid_size, 2);
        assert_eq!(hm.values, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(hm.raw, bytes);
        assert!(w.is_empty());
        // The same bytes read big-endian are nonsense — which is how the
        // endianness was settled in the first place.
        let be = f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert!(be < 1e-30, "big-endian reading yields {be}");
    }

    #[test]
    fn image_abr_height_map_size_mismatch_is_reported() {
        let bytes = vec![0u8; 4 * 25]; // implies a 5×5 grid
        let mut w = Vec::new();
        let hm = decode_height_map(&bytes, Some(11), &mut w);
        assert_eq!(hm.grid_size, 5);
        assert!(matches!(
            w.as_slice(),
            [AbrWarning::HeightMapSizeMismatch { .. }]
        ));
    }
}
