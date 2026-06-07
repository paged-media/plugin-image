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

//! gpu↔ref parity for the band family (channel plumbing): copy,
//! extract, set_alpha, broadcast_alpha. All four are pure lane
//! rearrangements ⇒ bit-exact. feat: band.copy / band.extract /
//! band.set_alpha / band.broadcast_alpha (registry/kernels.yaml).

use image_conformance::harness::{assert_within, parity, RefTile};
use image_conformance::Px;
use image_kernels::families::band::{
    band_broadcast_alpha, band_copy, band_extract, band_set_alpha, BandBroadcastAlphaParams,
    BandCopyParams, BandExtractParams, BandSetAlphaParams, BAND_BROADCAST_ALPHA, BAND_COPY,
    BAND_EXTRACT, BAND_SET_ALPHA,
};

/// A gradient where the four channels carry visibly distinct values —
/// so channel selection / broadcast can't pass by coincidence. Alpha
/// crosses the 0.5 line across the tile (the boolean-truthiness probe
/// other families lean on; harmless here but keeps the lane uniform).
fn gradient_tile(w: u32, h: u32) -> RefTile {
    RefTile::from_fn(w, h, |x, y| {
        Px([
            x as f32 / w as f32,
            y as f32 / h as f32,
            (x + y) as f32 / (w + h) as f32,
            (x + 1) as f32 / w as f32, // 1/w .. 1.0 — straddles 0.5
        ])
    })
}

/// A handful of explicit edge-case texels: opaque, transparent, and the
/// exact 0.5 alpha boundary, with each channel a distinct constant.
fn edge_tile() -> RefTile {
    let rows = [
        Px([0.0, 0.25, 0.5, 1.0]),   // fully opaque
        Px([0.75, 1.0, 0.125, 0.0]), // fully transparent
        Px([0.1, 0.2, 0.3, 0.5]),    // exactly on the 0.5 line
        Px([1.0, 0.0, 1.0, 0.0]),    // channels maximally separated
    ];
    RefTile::from_fn(4, rows.len() as u32, |x, y| {
        let _ = x;
        rows[y as usize]
    })
}

#[test]
fn band_copy_parity() {
    let tile = gradient_tile(image_core::TILE, image_core::TILE);
    let p = BandCopyParams::new();
    match parity(&BAND_COPY, band_copy, &[&tile], &p) {
        Some(r) => assert_within(r, &BAND_COPY),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn band_extract_parity() {
    // Oracle is per-channel (vips), so prove every channel selects.
    for channel in 0u32..4 {
        let tile = gradient_tile(image_core::TILE, image_core::TILE);
        let p = BandExtractParams::new(channel);
        match parity(&BAND_EXTRACT, band_extract, &[&tile], &p) {
            Some(r) => assert_within(r, &BAND_EXTRACT),
            None => {
                eprintln!("SKIP: no GPU adapter");
                break;
            }
        }
    }
}

#[test]
fn band_extract_edge_parity() {
    for channel in 0u32..4 {
        let tile = edge_tile();
        let p = BandExtractParams::new(channel);
        match parity(&BAND_EXTRACT, band_extract, &[&tile], &p) {
            Some(r) => assert_within(r, &BAND_EXTRACT),
            None => {
                eprintln!("SKIP: no GPU adapter");
                break;
            }
        }
    }
}

#[test]
fn band_set_alpha_parity() {
    let tile = gradient_tile(image_core::TILE, image_core::TILE);
    // A value that is NOT already present in the alpha gradient and that
    // is exact in f16, so the override is observable and bit-stable.
    let p = BandSetAlphaParams::new(0.375);
    match parity(&BAND_SET_ALPHA, band_set_alpha, &[&tile], &p) {
        Some(r) => assert_within(r, &BAND_SET_ALPHA),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn band_set_alpha_edge_parity() {
    let tile = edge_tile();
    let p = BandSetAlphaParams::new(1.0);
    match parity(&BAND_SET_ALPHA, band_set_alpha, &[&tile], &p) {
        Some(r) => assert_within(r, &BAND_SET_ALPHA),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn band_broadcast_alpha_parity() {
    let tile = gradient_tile(image_core::TILE, image_core::TILE);
    let p = BandBroadcastAlphaParams::new();
    match parity(&BAND_BROADCAST_ALPHA, band_broadcast_alpha, &[&tile], &p) {
        Some(r) => assert_within(r, &BAND_BROADCAST_ALPHA),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}

#[test]
fn band_broadcast_alpha_edge_parity() {
    let tile = edge_tile();
    let p = BandBroadcastAlphaParams::new();
    match parity(&BAND_BROADCAST_ALPHA, band_broadcast_alpha, &[&tile], &p) {
        Some(r) => assert_within(r, &BAND_BROADCAST_ALPHA),
        None => eprintln!("SKIP: no GPU adapter"),
    }
}
