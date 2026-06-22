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

//! The M0 PSD corpus: 11 named fixtures (the plan's "11 named
//! fixtures"). Each `pub fn` returns `(bytes, manifest)` where the bytes
//! come from this crate's INDEPENDENT emitter and the
//! [`FixtureManifest`] is the parser's answer key — every structural
//! fact a faithful parse must recover. The round-trip suite checks the
//! parser against the manifest (and, separately, byte-identity of
//! re-emit); the two code paths never share a bug because they share no
//! code (spec §10.4 oracle 1).
//!
//! Manifest types reuse image-psd's plain MODEL enums only ([`Container`],
//! [`Compression`]) — never its writer/reader — so "what to find" speaks
//! the same vocabulary as the parser's output without making the builder
//! self-referential.

use image_psd::container::Container;
use image_psd::model::Compression;

use super::channels::Plane;
use super::layers::{AddlSpec, ChannelSpec, LayerSpec, MaskSpec};
use super::PsdBuilder;

/// One additional-layer-info block the parser must surface, by its
/// on-disk identity (brief §6). Length is the STORED length (excludes
/// padding) so the manifest matches what the parser reads back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedAddl {
    pub sig: [u8; 4],
    pub key: [u8; 4],
    pub len: usize,
}

/// One layer the parser must recover, in flat-list order (brief §5/§7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedLayer {
    /// The legacy Pascal name in the layer record.
    pub name: String,
    /// The Unicode (`luni`) name, when the fixture carries one.
    pub unicode_name: Option<String>,
    /// Group nesting depth: 0 = document root, 1 = inside one group, etc.
    /// Folder + divider markers report the depth of their CONTENTS.
    pub depth: u32,
    pub blend_key: [u8; 4],
    pub opacity: u8,
    pub clipping: u8,
    pub layer_id: Option<u32>,
    pub has_mask: bool,
    /// `lsct` kind code when this entry is a group folder/divider
    /// (1 open / 2 closed / 3 bounding divider), else `None`.
    pub lsct_kind: Option<u32>,
}

impl ExpectedLayer {
    fn raster(name: &str, blend: [u8; 4], opacity: u8) -> Self {
        ExpectedLayer {
            name: name.to_string(),
            unicode_name: None,
            depth: 0,
            blend_key: blend,
            opacity,
            clipping: 0,
            layer_id: None,
            has_mask: false,
            lsct_kind: None,
        }
    }
}

/// A resource the parser must find (brief §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedResource {
    pub id: u16,
}

/// The parser's answer key for one fixture. Everything here is a fact
/// about the bytes; nothing references the builder's internals.
#[derive(Debug, Clone, PartialEq)]
pub struct FixtureManifest {
    pub name: &'static str,
    pub container: Container,
    pub width: u32,
    pub height: u32,
    /// Merged-composite channel count (header `channels`).
    pub channels: u16,
    pub composite_compression: Compression,
    /// `true` when the layer count is stored negative (brief §4a).
    pub transparency_in_merged: bool,
    pub layers: Vec<ExpectedLayer>,
    pub resources: Vec<ExpectedResource>,
    /// Unknown / opaque additional-layer-info blocks the parser must
    /// retain verbatim (brief §6) — by (sig, key, stored-len).
    pub unknown_addl: Vec<ExpectedAddl>,
}

/// Three solid RGB planes (composite channels) of the given dims.
fn rgb_planes(w: u32, h: u32, rgb: [u8; 3]) -> Vec<Plane> {
    rgb.iter().map(|&v| Plane::solid(w, h, v)).collect()
}

/// Three solid RGB layer channels (ids 0/1/2) of the given dims.
fn rgb_channels(w: u32, h: u32, rgb: [u8; 3], comp: Compression) -> Vec<ChannelSpec> {
    [0i16, 1, 2]
        .iter()
        .zip(rgb)
        .map(|(&id, v)| ChannelSpec {
            id,
            plane: Plane::solid(w, h, v),
            compression: comp,
        })
        .collect()
}

/// A plain raster [`LayerSpec`] (no mask, no extra addl) covering 0,0..w,h.
fn raster_layer(name: &str, w: u32, h: u32, rgb: [u8; 3], comp: Compression) -> LayerSpec {
    LayerSpec {
        top: 0,
        left: 0,
        bottom: h as i32,
        right: w as i32,
        name: name.to_string(),
        blend_key: *b"norm",
        opacity: 255,
        clipping: 0,
        flags: 0,
        mask: None,
        blend_ranges: Vec::new(),
        channels: rgb_channels(w, h, rgb, comp),
        extra_addl: Vec::new(),
    }
}

// ---------------------------------------------------------------------
// The 11 fixtures.
// ---------------------------------------------------------------------

/// 1. Flat RGB, no layers, RAW merged composite (brief §1/§11).
pub fn rgb8_flat() -> (Vec<u8>, FixtureManifest) {
    let (w, h) = (4, 4);
    let bytes = PsdBuilder::new(Container::Psd, w, h, 3)
        .composite(Compression::Raw, rgb_planes(w, h, [200, 100, 50]))
        .build();
    (
        bytes,
        FixtureManifest {
            name: "rgb8_flat",
            container: Container::Psd,
            width: w,
            height: h,
            channels: 3,
            composite_compression: Compression::Raw,
            transparency_in_merged: false,
            layers: Vec::new(),
            resources: Vec::new(),
            unknown_addl: Vec::new(),
        },
    )
}

/// 2. Flat RGB, RLE merged composite (brief §11/§12).
pub fn rgb8_flat_rle() -> (Vec<u8>, FixtureManifest) {
    let (w, h) = (4, 4);
    let bytes = PsdBuilder::new(Container::Psd, w, h, 3)
        .composite(Compression::Rle, rgb_planes(w, h, [10, 20, 30]))
        .build();
    (
        bytes,
        FixtureManifest {
            name: "rgb8_flat_rle",
            container: Container::Psd,
            width: w,
            height: h,
            channels: 3,
            composite_compression: Compression::Rle,
            transparency_in_merged: false,
            layers: Vec::new(),
            resources: Vec::new(),
            unknown_addl: Vec::new(),
        },
    )
}

/// 3. Three layers, one nested group (brief §5/§7). Flat list, bottom
///    first: bg, [divider, inner, folder], top.
pub fn multilayer_groups() -> (Vec<u8>, FixtureManifest) {
    let (w, h) = (8, 8);
    let bytes = PsdBuilder::new(Container::Psd, w, h, 3)
        .layer(raster_layer("bg", w, h, [0, 0, 0], Compression::Raw))
        .group_open("group", false)
        .layer(raster_layer("inner", w, h, [255, 0, 0], Compression::Raw))
        .group_close()
        .layer(raster_layer("top", w, h, [0, 255, 0], Compression::Raw))
        .composite(Compression::Raw, rgb_planes(w, h, [128, 128, 128]))
        .build();
    let layers = vec![
        ExpectedLayer::raster("bg", *b"norm", 255),
        // The bounding divider closes the group from below (brief §7).
        ExpectedLayer {
            depth: 1,
            lsct_kind: Some(3),
            ..ExpectedLayer::raster(super::GROUP_DIVIDER_NAME, *b"norm", 255)
        },
        ExpectedLayer {
            depth: 1,
            ..ExpectedLayer::raster("inner", *b"norm", 255)
        },
        ExpectedLayer {
            depth: 1,
            lsct_kind: Some(2),
            ..ExpectedLayer::raster("group", *b"norm", 255)
        },
        ExpectedLayer::raster("top", *b"norm", 255),
    ];
    (
        bytes,
        FixtureManifest {
            name: "multilayer_groups",
            container: Container::Psd,
            width: w,
            height: h,
            channels: 3,
            composite_compression: Compression::Raw,
            transparency_in_merged: false,
            layers,
            resources: Vec::new(),
            unknown_addl: Vec::new(),
        },
    )
}

/// 4. Unicode names: `luni` in BOTH count variants (brief §8) — one
///    layer's count excludes the trailing NUL, the other's includes it.
pub fn unicode_names() -> (Vec<u8>, FixtureManifest) {
    let (w, h) = (4, 4);
    let name_a = "café";
    let name_b = "naïve";
    let mut l0 = raster_layer("a", w, h, [1, 2, 3], Compression::Raw);
    l0.extra_addl.push(AddlSpec::Luni {
        name: name_a.to_string(),
        count_includes_null: false,
    });
    let mut l1 = raster_layer("b", w, h, [4, 5, 6], Compression::Raw);
    l1.extra_addl.push(AddlSpec::Luni {
        name: name_b.to_string(),
        count_includes_null: true,
    });
    let bytes = PsdBuilder::new(Container::Psd, w, h, 3)
        .layer(l0)
        .layer(l1)
        .composite(Compression::Raw, rgb_planes(w, h, [128, 128, 128]))
        .build();
    let layers = vec![
        ExpectedLayer {
            unicode_name: Some(name_a.to_string()),
            ..ExpectedLayer::raster("a", *b"norm", 255)
        },
        ExpectedLayer {
            unicode_name: Some(name_b.to_string()),
            ..ExpectedLayer::raster("b", *b"norm", 255)
        },
    ];
    (
        bytes,
        FixtureManifest {
            name: "unicode_names",
            container: Container::Psd,
            width: w,
            height: h,
            channels: 3,
            composite_compression: Compression::Raw,
            transparency_in_merged: false,
            layers,
            resources: Vec::new(),
            unknown_addl: Vec::new(),
        },
    )
}

/// 5. Layer ids via `lyid` (brief §9).
pub fn layer_ids() -> (Vec<u8>, FixtureManifest) {
    let (w, h) = (4, 4);
    let mut l0 = raster_layer("one", w, h, [1, 1, 1], Compression::Raw);
    l0.extra_addl.push(AddlSpec::Lyid(7));
    let mut l1 = raster_layer("two", w, h, [2, 2, 2], Compression::Raw);
    l1.extra_addl.push(AddlSpec::Lyid(42));
    let bytes = PsdBuilder::new(Container::Psd, w, h, 3)
        .layer(l0)
        .layer(l1)
        .composite(Compression::Raw, rgb_planes(w, h, [128, 128, 128]))
        .build();
    let layers = vec![
        ExpectedLayer {
            layer_id: Some(7),
            ..ExpectedLayer::raster("one", *b"norm", 255)
        },
        ExpectedLayer {
            layer_id: Some(42),
            ..ExpectedLayer::raster("two", *b"norm", 255)
        },
    ];
    (
        bytes,
        FixtureManifest {
            name: "layer_ids",
            container: Container::Psd,
            width: w,
            height: h,
            channels: 3,
            composite_compression: Compression::Raw,
            transparency_in_merged: false,
            layers,
            resources: Vec::new(),
            unknown_addl: Vec::new(),
        },
    )
}

/// 6. Several blend keys, opacities, and clipping (brief §5).
pub fn blend_opacity() -> (Vec<u8>, FixtureManifest) {
    let (w, h) = (4, 4);
    // (name, blend key, opacity, clipping)
    let specs: [(&str, [u8; 4], u8, u8); 4] = [
        ("base", *b"norm", 255, 0),
        ("mult", *b"mul ", 200, 0),
        ("screen", *b"scrn", 128, 0),
        ("clip", *b"over", 64, 1),
    ];
    let mut b = PsdBuilder::new(Container::Psd, w, h, 3);
    for (i, (name, key, op, clip)) in specs.iter().enumerate() {
        let mut l = raster_layer(name, w, h, [i as u8 * 40, 0, 0], Compression::Raw);
        l.blend_key = *key;
        l.opacity = *op;
        l.clipping = *clip;
        b = b.layer(l);
    }
    let bytes = b
        .composite(Compression::Raw, rgb_planes(w, h, [128, 128, 128]))
        .build();
    let layers = specs
        .iter()
        .map(|(name, key, op, clip)| ExpectedLayer {
            clipping: *clip,
            ..ExpectedLayer::raster(name, *key, *op)
        })
        .collect();
    (
        bytes,
        FixtureManifest {
            name: "blend_opacity",
            container: Container::Psd,
            width: w,
            height: h,
            channels: 3,
            composite_compression: Compression::Raw,
            transparency_in_merged: false,
            layers,
            resources: Vec::new(),
            unknown_addl: Vec::new(),
        },
    )
}

/// 7. A raster layer with a user mask: channel id -2 + mask data size 20
///    (brief §5).
pub fn raster_masks() -> (Vec<u8>, FixtureManifest) {
    let (w, h) = (4, 4);
    let mut l = raster_layer("masked", w, h, [9, 9, 9], Compression::Raw);
    // Append the user-mask channel (id -2). Its plane covers the mask
    // rect; here the mask rect equals the layer rect for simplicity.
    l.channels.push(ChannelSpec {
        id: -2,
        plane: Plane::solid(w, h, 255),
        compression: Compression::Raw,
    });
    l.mask = Some(MaskSpec {
        top: 0,
        left: 0,
        bottom: h as i32,
        right: w as i32,
        default_color: 0,
        flags: 0,
        real: None,
    });
    let bytes = PsdBuilder::new(Container::Psd, w, h, 3)
        .layer(l)
        .composite(Compression::Raw, rgb_planes(w, h, [128, 128, 128]))
        .build();
    let layers = vec![ExpectedLayer {
        has_mask: true,
        ..ExpectedLayer::raster("masked", *b"norm", 255)
    }];
    (
        bytes,
        FixtureManifest {
            name: "raster_masks",
            container: Container::Psd,
            width: w,
            height: h,
            channels: 3,
            composite_compression: Compression::Raw,
            transparency_in_merged: false,
            layers,
            resources: Vec::new(),
            unknown_addl: Vec::new(),
        },
    )
}

/// 8. Layers mixing RLE and RAW channels, and an RLE composite (brief
///    §10/§11) — exercises both decode paths in one file.
pub fn rle_and_raw_mix() -> (Vec<u8>, FixtureManifest) {
    let (w, h) = (6, 6);
    // Layer 0: all RAW. Layer 1: all RLE. Layer 2: mixed per channel.
    let l0 = raster_layer("raw", w, h, [255, 0, 0], Compression::Raw);
    let l1 = raster_layer("rle", w, h, [0, 255, 0], Compression::Rle);
    let l2 = LayerSpec {
        channels: vec![
            ChannelSpec {
                id: 0,
                plane: Plane::solid(w, h, 10),
                compression: Compression::Rle,
            },
            ChannelSpec {
                id: 1,
                plane: Plane::solid(w, h, 20),
                compression: Compression::Raw,
            },
            ChannelSpec {
                id: 2,
                plane: Plane::solid(w, h, 30),
                compression: Compression::Rle,
            },
        ],
        ..raster_layer("mixed", w, h, [0, 0, 255], Compression::Raw)
    };
    let bytes = PsdBuilder::new(Container::Psd, w, h, 3)
        .layer(l0)
        .layer(l1)
        .layer(l2)
        .composite(Compression::Rle, rgb_planes(w, h, [128, 128, 128]))
        .build();
    let layers = vec![
        ExpectedLayer::raster("raw", *b"norm", 255),
        ExpectedLayer::raster("rle", *b"norm", 255),
        ExpectedLayer::raster("mixed", *b"norm", 255),
    ];
    (
        bytes,
        FixtureManifest {
            name: "rle_and_raw_mix",
            container: Container::Psd,
            width: w,
            height: h,
            channels: 3,
            composite_compression: Compression::Rle,
            transparency_in_merged: false,
            layers,
            resources: Vec::new(),
            unknown_addl: Vec::new(),
        },
    )
}

/// 9. An unknown additional-layer-info block: a private key 'xpgi' with
///    a binary payload the parser must retain verbatim (brief §6/§10.4).
pub fn unknown_addl() -> (Vec<u8>, FixtureManifest) {
    let (w, h) = (4, 4);
    let payload = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01];
    let bytes = PsdBuilder::new(Container::Psd, w, h, 3)
        .layer(raster_layer("L", w, h, [7, 7, 7], Compression::Raw))
        .layer_addl_opaque(*b"8BIM", *b"xpgi", payload.clone())
        .composite(Compression::Raw, rgb_planes(w, h, [128, 128, 128]))
        .build();
    (
        bytes,
        FixtureManifest {
            name: "unknown_addl",
            container: Container::Psd,
            width: w,
            height: h,
            channels: 3,
            composite_compression: Compression::Raw,
            transparency_in_merged: false,
            layers: vec![ExpectedLayer::raster("L", *b"norm", 255)],
            resources: Vec::new(),
            unknown_addl: vec![ExpectedAddl {
                sig: *b"8BIM",
                key: *b"xpgi",
                len: payload.len(),
            }],
        },
    )
}

/// 10. An unmodeled image resource (id 0x0bb7) plus the modeled ICC and
///     resolution resources (brief §3).
pub fn unknown_resource() -> (Vec<u8>, FixtureManifest) {
    let (w, h) = (4, 4);
    let opaque_id = 0x0bb7u16;
    let icc = vec![0u8; 12]; // a stub ICC payload — opaque to us anyway
    let bytes = PsdBuilder::new(Container::Psd, w, h, 3)
        .resource_resolution(72.0)
        .resource_icc(icc)
        .resource_opaque(opaque_id, vec![0xAA, 0xBB, 0xCC])
        .composite(Compression::Raw, rgb_planes(w, h, [128, 128, 128]))
        .build();
    (
        bytes,
        FixtureManifest {
            name: "unknown_resource",
            container: Container::Psd,
            width: w,
            height: h,
            channels: 3,
            composite_compression: Compression::Raw,
            transparency_in_merged: false,
            layers: Vec::new(),
            resources: vec![
                ExpectedResource {
                    id: image_psd::model::resources::RES_RESOLUTION_INFO,
                },
                ExpectedResource {
                    id: image_psd::model::resources::RES_ICC_PROFILE,
                },
                ExpectedResource { id: opaque_id },
            ],
            unknown_addl: Vec::new(),
        },
    )
}

/// 11. PSB container: u64 length fields, u32 RLE row counts (brief §4/§5/
///     §10/§11). 64×64 is plenty — PSB is about field WIDTHS, not size.
pub fn psb_wide() -> (Vec<u8>, FixtureManifest) {
    let (w, h) = (64, 64);
    let bytes = PsdBuilder::new(Container::Psb, w, h, 3)
        .layer(raster_layer("psb", w, h, [12, 34, 56], Compression::Rle))
        .composite(Compression::Rle, rgb_planes(w, h, [200, 100, 50]))
        .build();
    (
        bytes,
        FixtureManifest {
            name: "psb_wide",
            container: Container::Psb,
            width: w,
            height: h,
            channels: 3,
            composite_compression: Compression::Rle,
            transparency_in_merged: false,
            layers: vec![ExpectedLayer::raster("psb", *b"norm", 255)],
            resources: Vec::new(),
            unknown_addl: Vec::new(),
        },
    )
}

/// All 11 fixtures, for corpus-wide round-trip drivers.
pub fn all() -> Vec<(Vec<u8>, FixtureManifest)> {
    vec![
        rgb8_flat(),
        rgb8_flat_rle(),
        multilayer_groups(),
        unicode_names(),
        layer_ids(),
        blend_opacity(),
        raster_masks(),
        rle_and_raw_mix(),
        unknown_addl(),
        unknown_resource(),
        psb_wide(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every fixture builds non-empty and carries the '8BPS' signature
    /// with the version its container declares (brief §1).
    #[test]
    fn psd_builder_fixtures_header_sanity() {
        for (bytes, manifest) in all() {
            assert!(
                bytes.len() > 26,
                "fixture {} too short: {} bytes",
                manifest.name,
                bytes.len()
            );
            assert_eq!(&bytes[0..4], b"8BPS", "fixture {} signature", manifest.name);
            let version = u16::from_be_bytes([bytes[4], bytes[5]]);
            assert_eq!(
                version,
                manifest.container.version(),
                "fixture {} version",
                manifest.name
            );
            // Dimensions in the header match the manifest (brief §1).
            let height = u32::from_be_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]);
            let width = u32::from_be_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]);
            assert_eq!(height, manifest.height, "fixture {} height", manifest.name);
            assert_eq!(width, manifest.width, "fixture {} width", manifest.name);
        }
    }

    /// `all()` lists exactly the 11 named fixtures with unique names.
    #[test]
    fn psd_builder_fixtures_count_is_eleven() {
        let fixtures = all();
        assert_eq!(fixtures.len(), 11);
        let mut names: Vec<&str> = fixtures.iter().map(|(_, m)| m.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 11, "fixture names must be unique");
    }

    /// The grouped fixture expands to the flat folder/divider list with
    /// the correct nesting depth (brief §7).
    #[test]
    fn psd_builder_fixtures_group_nesting() {
        let (_, m) = multilayer_groups();
        assert_eq!(m.layers.len(), 5);
        assert_eq!(m.layers[0].depth, 0); // bg at root
        assert_eq!(m.layers[1].lsct_kind, Some(3)); // bounding divider
        assert_eq!(m.layers[2].depth, 1); // inner member
        assert_eq!(m.layers[3].lsct_kind, Some(2)); // closed folder
        assert_eq!(m.layers[4].depth, 0); // top at root
    }

    /// The unknown-addl fixture pins the opaque block by identity (brief
    /// §6) and its stored length excludes padding.
    #[test]
    fn psd_builder_fixtures_unknown_addl_recorded() {
        let (_, m) = unknown_addl();
        assert_eq!(m.unknown_addl.len(), 1);
        assert_eq!(&m.unknown_addl[0].key, b"xpgi");
        assert_eq!(m.unknown_addl[0].len, 5);
    }

    /// PSB fixture declares the Psb container — the widened-field path.
    #[test]
    fn psd_builder_fixtures_psb_container() {
        let (_, m) = psb_wide();
        assert_eq!(m.container, Container::Psb);
    }
}
