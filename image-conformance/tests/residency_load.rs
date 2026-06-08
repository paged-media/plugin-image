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

//! Residency hardening under over-subscription (spec §15 M3): with a
//! working set LARGER than the GPU pool, the Tier-0↔Tier-1 ladder must
//! spill the LRU slot to the heap and fault it back byte-identically,
//! never panicking and never leaking a slot. The headline M3 item was an
//! "OPFS swap tier hardened", but OPFS (Tier 2) is the storage capability
//! the plugin SDK does not yet expose (BREAKAGE I-03) — so Tier-2 ops
//! must still return the typed error, and the HARDENING happens on the
//! Tier-0/1 path that the engines actually hit first.
//!
//! Three lanes:
//! (a) CPU bookkeeping — touch 3N distinct tiles through a pool of N
//!     slots; assert no panic/leak and that re-requested evicted tiles
//!     reload byte-identically (Tier-1 round-trip, no adapter needed);
//! (b) GPU round-trip — real `upload → evict → download == original`
//!     (SKIPs without an adapter, spec §6.3);
//! (c) Tier-2 ops still report the typed I-03 error.
//!
//! feat: image.residency.tier01.

use std::sync::Arc;

use image_conformance::device::test_device;
use image_core::{TextureSlot, TileCoord};
use image_gpu::{ResidencyManager, TexturePool, Tier, HEAP_TILE_BYTES};

fn coord(x: i32, y: i32) -> TileCoord {
    TileCoord { level: 0, x, y }
}

/// A distinct, deterministic full heap tile keyed by `seed` — every byte
/// derived from the seed so two tiles are byte-different and a mis-tracked
/// reload would not match.
fn tile_bytes(seed: u32) -> Arc<[u8]> {
    let mut v = vec![0u8; HEAP_TILE_BYTES];
    for (i, b) in v.iter_mut().enumerate() {
        // Mix the seed into every byte; +3 stride keeps adjacent bytes
        // distinct so a row/stride bug would surface.
        *b = ((i as u32)
            .wrapping_mul(31)
            .wrapping_add(seed.wrapping_mul(2_654_435_761))
            >> 7) as u8;
    }
    Arc::from(v.into_boxed_slice())
}

/// Lane (a): pure bookkeeping over the spill policy — no device. Models
/// the eviction loop a pool-of-N driver runs over 3N distinct tiles: when
/// "full", spill the least-recently-touched coord to Tier 1; on a
/// re-request of an evicted tile, fault it back in and assert the bytes
/// survived the heap round-trip verbatim.
#[test]
fn image_residency_tier01_oversubscribed_bookkeeping_no_leak_byte_identical() {
    const N: usize = 4; // pool slots
    let mut r = ResidencyManager::new();

    // The "originals" we will check evicted tiles against.
    let total = 3 * N;
    let originals: Vec<Arc<[u8]>> = (0..total as u32).map(tile_bytes).collect();

    // A tiny LRU model of the N Tier-0 slots: front = least-recently used.
    // Entry = (coord_index, slot). The manager mirrors the same coords.
    let mut lru: Vec<(usize, TextureSlot)> = Vec::with_capacity(N);
    // Free slot indices (start with all N).
    let mut free: Vec<u32> = (0..N as u32).rev().collect();

    for ci in 0..total {
        let c = coord(ci as i32, 0);
        // If full, spill the LRU coord to Tier 1 and reclaim its slot.
        if free.is_empty() {
            let (victim_ci, victim_slot) = lru.remove(0);
            let victim_coord = coord(victim_ci as i32, 0);
            let vacated = r.spill_to_tier1(victim_coord, Arc::clone(&originals[victim_ci]));
            assert_eq!(
                vacated,
                Some(victim_slot),
                "spill vacates exactly the victim's slot"
            );
            free.push(victim_slot.0);
            // Tier-0 count must have dropped by one — no leak.
            assert!(r.tier0_count() <= N, "Tier-0 never exceeds pool capacity");
        }
        let slot = TextureSlot(free.pop().expect("a slot is free after any spill"));
        // "Upload" + mark resident.
        r.mark_resident(c, slot);
        lru.push((ci, slot));
    }

    // No leak: exactly N tiles are Tier 0 (the last N touched), the rest
    // are Tier 1.
    assert_eq!(r.tier0_count(), N, "exactly N tiles resident at the end");
    let mut tier0 = 0usize;
    let mut tier1 = 0usize;
    for ci in 0..total {
        match r.tier(coord(ci as i32, 0)) {
            Some(Tier::Tier0(_)) => tier0 += 1,
            Some(Tier::Tier1(_)) => tier1 += 1,
            other => panic!("tile {ci} in unexpected tier: {other:?}"),
        }
    }
    assert_eq!(tier0, N, "N tiles on the GPU");
    assert_eq!(tier1, total - N, "the rest spilled to the heap");

    // Every evicted (Tier-1) tile reloads byte-identical to its original
    // (the fault-in the caller would upload).
    for (ci, original) in originals.iter().enumerate() {
        if let Some(bytes) = r.tier1_bytes(coord(ci as i32, 0)) {
            assert_eq!(
                &*bytes, &**original,
                "evicted tile {ci} reloads byte-identically"
            );
        }
    }
}

/// Lane (b): the real GPU `upload → evict → download == original` round
/// trip. Builds a pool of N slots, fills it, then drives one more
/// acquisition through `acquire_or_spill` to force a real LRU eviction
/// (texture readback to the heap) and asserts the evicted tile's pixels
/// survived the GPU round-trip byte-identically. SKIPs without an adapter.
#[test]
fn image_residency_tier01_gpu_evict_download_is_byte_identical() {
    let Some(ctx) = test_device() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    const N: u32 = 3;

    let mut pool = TexturePool::new(N);
    let mut r = ResidencyManager::new();

    // Fill the pool: N distinct tiles, each acquired and uploaded.
    let originals: Vec<Arc<[u8]>> = (0..N).map(tile_bytes).collect();
    for k in 0..N {
        let c = coord(k as i32, 0);
        let acq = r
            .acquire_or_spill(ctx, &mut pool, c)
            .expect("acquire within capacity");
        assert!(acq.spilled.is_none(), "no spill while the pool has room");
        r.upload(ctx, &mut pool, c, acq.slot, &originals[k as usize])
            .expect("upload tile");
        // Sanity: re-mark touched so the touch order is well-defined.
        pool.touch(acq.slot);
    }
    assert_eq!(pool.live(), N as usize, "pool full");
    assert_eq!(r.tier0_count(), N as usize, "all N on the GPU");

    // The LRU victim is coord(0,0) (touched first, never since). Acquire
    // one MORE tile to force its eviction → real texture download.
    let newc = coord(N as i32, 0);
    let acq = r
        .acquire_or_spill(ctx, &mut pool, newc)
        .expect("acquire under pressure");
    let (spilled_coord, spilled_bytes) = acq.spilled.expect("a tile was spilled");
    assert_eq!(spilled_coord, coord(0, 0), "LRU victim is the first tile");

    // (b) The spilled bytes — downloaded from the GPU — equal the bytes we
    // uploaded. byte-identical Tier-0→Tier-1 round trip.
    assert_eq!(
        &*spilled_bytes, &*originals[0],
        "upload → evict → download must be byte-identical"
    );
    // And the same bytes are reachable as the tile's Tier-1 residency.
    let reloaded = r
        .tier1_bytes(coord(0, 0))
        .expect("evicted tile is now on the heap");
    assert_eq!(&*reloaded, &*originals[0], "Tier-1 copy matches the upload");

    // No leak: still exactly N tiles resident (the freed slot was reused
    // for the new tile).
    assert_eq!(pool.live(), N as usize, "pool still full, slot reused");
    assert_eq!(r.tier0_count(), N as usize, "no slot leaked");

    // Re-request the evicted tile: fault it back in (acquire a slot,
    // upload its journalled bytes) — forces yet another eviction, and the
    // re-faulted tile must again round-trip byte-identically.
    let faulted = coord(0, 0);
    let bytes = r.tier1_bytes(faulted).expect("still on the heap");
    let acq2 = r
        .acquire_or_spill(ctx, &mut pool, faulted)
        .expect("re-acquire for fault-in");
    r.upload(ctx, &mut pool, faulted, acq2.slot, &bytes)
        .expect("re-upload faulted tile");
    assert!(
        matches!(r.tier(faulted), Some(Tier::Tier0(_))),
        "back on GPU"
    );
    // Download it straight back and confirm byte-identity end-to-end.
    let rt = r
        .download(ctx, &pool, faulted, acq2.slot)
        .expect("download faulted tile");
    assert_eq!(
        &*rt, &*originals[0],
        "fault-in → download is byte-identical to the original upload"
    );
}

/// Lane (c): Tier-2 (OPFS) ops still degrade with the typed I-03 error —
/// the hardening did not silently enable a tier the SDK cannot back.
#[test]
fn image_residency_tier01_tier2_ops_still_report_i03() {
    let mut r = ResidencyManager::new();
    let err = r
        .swap_out(coord(0, 0))
        .expect_err("Tier 2 swap_out unsupported");
    assert!(
        err.to_string().contains("I-03"),
        "Tier-2 error must name BREAKAGE I-03, got: {err}"
    );
    assert!(
        r.swap_in(coord(0, 0)).is_err(),
        "Tier 2 swap_in is unsupported too"
    );
}
