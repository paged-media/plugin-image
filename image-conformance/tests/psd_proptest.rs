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

//! Property-based PSD round-trip (spec §10.4 oracle 1). Random small
//! layer stacks — 1..6 layers, an optional group, random blend keys,
//! per-channel RAW/RLE, random opacities, Unicode-ish names, an optional
//! unknown additional-layer-info block — are synthesized by the
//! INDEPENDENT builder, parsed by the production parser, re-emitted, and
//! checked for:
//!
//! * **byte-identity** — `write()` reproduces the builder's bytes (the
//!   zero-edit preservation target), and
//! * **semantic round-trip** — the legacy names (including the injected
//!   group divider/folder records), Unicode names, and opacities recover
//!   exactly, in flat bottom-first order.
//!
//! Cases are kept modest (~64) and dimensions tiny so the suite stays
//! fast: feat image.psd.roundtrip.

use image_conformance::psd_builder::{
    AddlSpec, ChannelSpec, LayerSpec, Plane, PsdBuilder, GROUP_DIVIDER_NAME,
};
use image_psd::container::Container;
use image_psd::model::{Compression, PsdFile};
use proptest::prelude::*;

/// The blend keys the generator draws from — a representative fixed set
/// of valid 4cc mode keys (brief §5).
const BLEND_KEYS: &[[u8; 4]] = &[
    *b"norm", *b"mul ", *b"scrn", *b"over", *b"diff", *b"dark", *b"lite",
];

/// A generated channel: RAW or RLE, with a fill value (the plane is solid
/// so PackBits has a deterministic, decodable shape either way).
#[derive(Debug, Clone)]
struct GenChannel {
    rle: bool,
    fill: u8,
}

/// A generated layer's intent, before lowering to a [`LayerSpec`].
#[derive(Debug, Clone)]
struct GenLayer {
    name: String,
    unicode: Option<String>,
    blend: [u8; 4],
    opacity: u8,
    channels: Vec<GenChannel>,
    /// At most one unknown additional-layer-info block (random payload).
    unknown_addl: Option<([u8; 4], Vec<u8>)>,
}

/// A name with a few Unicode-ish characters so the `luni` path is
/// exercised, while staying short and BMP-only (encodes to one UTF-16
/// unit each, keeping fixtures small).
fn name_strategy() -> impl Strategy<Value = String> {
    proptest::collection::vec(
        prop_oneof![
            (b'a'..=b'z').prop_map(|c| c as char),
            Just('é'),
            Just('ü'),
            Just('ñ'),
            Just('λ'),
            Just('Ω'),
        ],
        1..6usize,
    )
    .prop_map(|chars| chars.into_iter().collect())
}

fn channel_strategy() -> impl Strategy<Value = GenChannel> {
    (any::<bool>(), any::<u8>()).prop_map(|(rle, fill)| GenChannel { rle, fill })
}

fn blend_strategy() -> impl Strategy<Value = [u8; 4]> {
    (0..BLEND_KEYS.len()).prop_map(|i| BLEND_KEYS[i])
}

fn unknown_addl_strategy() -> impl Strategy<Value = Option<([u8; 4], Vec<u8>)>> {
    let key = prop_oneof![Just(*b"xpgi"), Just(*b"Xz99"), Just(*b"q0Op")];
    let payload = proptest::collection::vec(any::<u8>(), 0..64usize);
    proptest::option::of((key, payload))
}

fn layer_strategy() -> impl Strategy<Value = GenLayer> {
    (
        name_strategy(),
        proptest::option::of(name_strategy()),
        blend_strategy(),
        any::<u8>(),
        proptest::collection::vec(channel_strategy(), 3..=3usize),
        unknown_addl_strategy(),
    )
        .prop_map(
            |(name, unicode, blend, opacity, channels, unknown_addl)| GenLayer {
                name,
                unicode,
                blend,
                opacity,
                channels,
                unknown_addl,
            },
        )
}

/// A whole generated stack: 1..6 layers and an optional group window
/// `(start, len)` that wraps a contiguous run of them.
#[derive(Debug, Clone)]
struct GenStack {
    container: Container,
    layers: Vec<GenLayer>,
    /// `Some((start, len, open))` groups layers `start..start+len`.
    group: Option<(usize, usize, bool)>,
}

fn stack_strategy() -> impl Strategy<Value = GenStack> {
    let container = prop_oneof![Just(Container::Psd), Just(Container::Psb)];
    (
        container,
        proptest::collection::vec(layer_strategy(), 1..6usize),
    )
        .prop_flat_map(|(container, layers)| {
            let n = layers.len();
            // The optional group window must fit inside the stack; pick a
            // start and a length >= 1 within bounds, or no group at all.
            let group = if n >= 1 {
                prop_oneof![
                    Just(None),
                    (0..n, 1..=n, any::<bool>()).prop_map(move |(start, len, open)| {
                        let start = start.min(n.saturating_sub(1));
                        let len = len.min(n - start).max(1);
                        Some((start, len, open))
                    }),
                ]
                .boxed()
            } else {
                Just(None).boxed()
            };
            (Just(container), Just(layers), group)
        })
        .prop_map(|(container, layers, group)| GenStack {
            container,
            layers,
            group,
        })
}

const W: u32 = 4;
const H: u32 = 3;

/// Lower one [`GenLayer`] to a builder [`LayerSpec`] covering 0,0..W,H.
fn lower_layer(g: &GenLayer) -> LayerSpec {
    let channels = g
        .channels
        .iter()
        .enumerate()
        .map(|(i, c)| ChannelSpec {
            id: i as i16, // 0/1/2 composite channel ids
            plane: Plane::solid(W, H, c.fill),
            compression: if c.rle {
                Compression::Rle
            } else {
                Compression::Raw
            },
        })
        .collect();
    let mut extra_addl = Vec::new();
    if let Some(uni) = &g.unicode {
        extra_addl.push(AddlSpec::Luni {
            name: uni.clone(),
            count_includes_null: false,
        });
    }
    if let Some((key, payload)) = &g.unknown_addl {
        extra_addl.push(AddlSpec::Opaque {
            sig: *b"8BIM",
            key: *key,
            payload: payload.clone(),
        });
    }
    LayerSpec {
        top: 0,
        left: 0,
        bottom: H as i32,
        right: W as i32,
        name: g.name.clone(),
        blend_key: g.blend,
        opacity: g.opacity,
        clipping: 0,
        flags: 0,
        mask: None,
        blend_ranges: Vec::new(),
        channels,
        extra_addl,
    }
}

/// The flat bottom-first sequence of (legacy_name, opacity, unicode) the
/// builder will emit for this stack — including the divider (`</Layer
/// group>`, opacity 255, no unicode) and folder (group name, opacity 255)
/// records the builder injects for a group (brief §7).
fn expected_flat(stack: &GenStack) -> Vec<(String, u8, Option<String>)> {
    fn entry(g: &GenLayer) -> (String, u8, Option<String>) {
        (g.name.clone(), g.opacity, g.unicode.clone())
    }
    let mut out = Vec::new();
    match stack.group {
        None => out.extend(stack.layers.iter().map(entry)),
        Some((start, len, _open)) => {
            // before the group (root level)
            out.extend(stack.layers[..start].iter().map(entry));
            // group_open injects the bounding divider first (bottom-most)
            out.push((GROUP_DIVIDER_NAME.to_string(), 255, None));
            out.extend(stack.layers[start..start + len].iter().map(entry));
            // group_close injects the folder record above the members
            out.push(("grp".to_string(), 255, None));
            // after the group (root level)
            out.extend(stack.layers[start + len..].iter().map(entry));
        }
    }
    out
}

/// Build the PSD bytes for a generated stack via the independent builder.
fn build(stack: &GenStack) -> Vec<u8> {
    let mut b = PsdBuilder::new(stack.container, W, H, 3);
    match stack.group {
        None => {
            for g in &stack.layers {
                b = b.layer(lower_layer(g));
            }
        }
        Some((start, len, open)) => {
            for g in &stack.layers[..start] {
                b = b.layer(lower_layer(g));
            }
            b = b.group_open("grp", open);
            for g in &stack.layers[start..start + len] {
                b = b.layer(lower_layer(g));
            }
            b = b.group_close();
            for g in &stack.layers[start + len..] {
                b = b.layer(lower_layer(g));
            }
        }
    }
    // A RAW composite keeps the trailing section deterministic; the layer
    // channels carry the RLE/RAW variety the property cares about.
    let planes = (0..3).map(|_| Plane::solid(W, H, 128)).collect();
    b.composite(Compression::Raw, planes).build()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Any generated stack parses, re-emits byte-identically, and recovers
    /// its names/opacities/Unicode names through the parser.
    #[test]
    fn image_psd_roundtrip_random_stacks(stack in stack_strategy()) {
        let bytes = build(&stack);

        // Parse must succeed.
        let file = PsdFile::parse(&bytes)
            .map_err(|e| TestCaseError::fail(format!("parse failed: {e}")))?;

        // Byte-identity: zero-edit re-emit reproduces the source exactly.
        let reemit = file
            .write()
            .map_err(|e| TestCaseError::fail(format!("write failed: {e}")))?;
        prop_assert_eq!(
            &reemit,
            &bytes,
            "re-emit not byte-identical (len {} vs {})",
            reemit.len(),
            bytes.len()
        );

        // Semantic round-trip: flat-list legacy names, opacities, and the
        // Unicode `luni` names match the builder's intent in order.
        let want = expected_flat(&stack);
        prop_assert_eq!(
            file.layer_mask.layers.len(),
            want.len(),
            "layer count mismatch"
        );
        for (i, (got, (name, opacity, unicode))) in
            file.layer_mask.layers.iter().zip(&want).enumerate()
        {
            prop_assert_eq!(&got.name_legacy.text_lossy(), name, "layer[{}] legacy name", i);
            prop_assert_eq!(got.opacity, *opacity, "layer[{}] opacity", i);
            prop_assert_eq!(
                got.addl.iter().find_map(|a| a.unicode_name()),
                unicode.clone(),
                "layer[{}] unicode name",
                i
            );
        }
    }
}
