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

//! MODEL-BASED properties of the layer stack's undo journal.
//!
//! A generated program of operations — paint a rectangle into the
//! active layer (`edit_active`), undo, redo, add a layer, switch the
//! active layer — runs against the real `LayerStack` AND against a naive
//! model that keeps whole before/after copies of every edited layer.
//! After every step:
//!
//! * every layer's pixels equal the model's;
//! * undo/redo report exactly when the model says there is something to
//!   undo/redo, and land in the layer the edit was made on (which then
//!   becomes active) — not in whichever layer is selected;
//! * the journal's depth, redo depth and eviction count match the model,
//!   whose undo history is bounded by the journal's declared entry cap;
//! * the journal never exceeds either of its declared bounds.
//!
//! Programs are long enough (up to 80 steps) to cross the entry cap, so
//! the eviction path is exercised, not just the happy one.

use std::sync::Arc;

use image_core::Region;
use image_js::layers::LayerStack;
use image_js::pixels::Pixels;
use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

const W: u32 = 40;
const H: u32 = 24;

#[derive(Debug, Clone)]
enum Op {
    Paint {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        v: u8,
    },
    Undo,
    Redo,
    AddLayer,
    Select(usize),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        6 => (0..W, 0..H, 1..=W, 1..=H, any::<u8>()).prop_map(|(x, y, w, h, v)| Op::Paint {
            x,
            y,
            w: w.min(W - x),
            h: h.min(H - y),
            v,
        }),
        3 => Just(Op::Undo),
        2 => Just(Op::Redo),
        1 => Just(Op::AddLayer),
        1 => (0usize..8).prop_map(Op::Select),
    ]
}

/// One journaled edit, in the model: which layer, and its whole pixels
/// before and after.
#[derive(Clone)]
struct Edit {
    layer_id: u32,
    before: Vec<u8>,
    after: Vec<u8>,
}

struct Model {
    /// (id, pixels), bottom-first — mirrors the stack's order.
    layers: Vec<(u32, Vec<u8>)>,
    active: usize,
    undo: Vec<Edit>,
    redo: Vec<Edit>,
    dropped: u64,
    cap: usize,
}

impl Model {
    fn idx(&self, id: u32) -> usize {
        self.layers
            .iter()
            .position(|(i, _)| *i == id)
            .expect("layer")
    }
}

fn painted(src: &[u8], x: u32, y: u32, w: u32, h: u32, v: u8) -> Vec<u8> {
    let mut out = src.to_vec();
    for yy in y..y + h {
        for xx in x..x + w {
            let i = ((yy * W + xx) * 4) as usize;
            out[i..i + 4].copy_from_slice(&[v, v.wrapping_mul(3), v ^ 0x5a, 255]);
        }
    }
    out
}

fn check(stack: &LayerStack, m: &Model) -> Result<(), TestCaseError> {
    prop_assert_eq!(stack.len(), m.layers.len());
    for (l, (id, px)) in stack.layers().iter().zip(&m.layers) {
        prop_assert_eq!(l.id, *id);
        prop_assert!(l.rgba.raw() == px.as_slice(), "layer {} pixels diverge", id);
    }
    prop_assert_eq!(stack.active_index(), m.active);
    let h = stack.history();
    prop_assert_eq!(h.depth, m.undo.len(), "undo depth");
    prop_assert_eq!(h.redo_depth, m.redo.len(), "redo depth");
    prop_assert_eq!(h.can_undo, !m.undo.is_empty());
    prop_assert_eq!(h.can_redo, !m.redo.is_empty());
    prop_assert_eq!(h.dropped, m.dropped, "evictions");
    prop_assert!(
        h.depth <= h.max_entries,
        "depth {} over cap {}",
        h.depth,
        h.max_entries
    );
    prop_assert!(
        h.bytes <= h.max_bytes,
        "bytes {} over cap {}",
        h.bytes,
        h.max_bytes
    );
    Ok(())
}

fn run(ops: Vec<Op>) -> Result<(), TestCaseError> {
    let base: Vec<u8> = (0..W * H * 4).map(|i| (i * 7 % 251) as u8).collect();
    let mut stack = LayerStack::from_image(W, H, Arc::from(base.clone())).expect("stack");
    let cap = stack.history().max_entries;
    let mut m = Model {
        layers: vec![(stack.layers()[0].id, base)],
        active: 0,
        undo: Vec::new(),
        redo: Vec::new(),
        dropped: 0,
        cap,
    };
    check(&stack, &m)?;
    for op in ops {
        match op {
            Op::Paint { x, y, w, h, v } => {
                let (id, cur) = &m.layers[m.active];
                let id = *id;
                let next = painted(cur, x, y, w, h, v);
                let outcome = stack
                    .edit_active(
                        "paint",
                        Region::new(x as i32, y as i32, w, h),
                        Pixels::from_rgba8(Arc::from(next.clone())),
                    )
                    .expect("edit");
                prop_assert!(outcome.is_recorded(), "a non-empty edit must journal");
                m.undo.push(Edit {
                    layer_id: id,
                    before: cur.clone(),
                    after: next.clone(),
                });
                m.redo.clear();
                if m.undo.len() > m.cap {
                    m.undo.remove(0);
                    m.dropped += 1;
                }
                let a = m.active;
                m.layers[a].1 = next;
            }
            Op::Undo => {
                let got = stack.undo();
                prop_assert_eq!(got.is_some(), !m.undo.is_empty(), "undo availability");
                if let Some(e) = m.undo.pop() {
                    let i = m.idx(e.layer_id);
                    m.layers[i].1 = e.before.clone();
                    m.active = i;
                    m.redo.push(e);
                }
            }
            Op::Redo => {
                let got = stack.redo();
                prop_assert_eq!(got.is_some(), !m.redo.is_empty(), "redo availability");
                if let Some(e) = m.redo.pop() {
                    let i = m.idx(e.layer_id);
                    m.layers[i].1 = e.after.clone();
                    m.active = i;
                    m.undo.push(e);
                }
            }
            Op::AddLayer => {
                let at = stack.add("layer");
                let id = stack.layers()[at].id;
                m.layers.insert(at, (id, vec![0u8; (W * H * 4) as usize]));
                m.active = at;
            }
            Op::Select(k) => {
                let k = k % m.layers.len();
                stack.set_active(k).expect("select");
                m.active = k;
            }
        }
        check(&stack, &m)?;
    }
    Ok(())
}

#[test]
#[allow(non_snake_case)]
fn undo_redo_matches_a_naive_model_within_the_journal_bound__feat__image_editor_undo_journal() {
    let mut runner = TestRunner::new_with_rng(
        Config {
            cases: 48,
            failure_persistence: None,
            ..Config::default()
        },
        TestRng::from_seed(RngAlgorithm::ChaCha, &[0x10; 32]),
    );
    runner
        .run(&proptest::collection::vec(op(), 1..80), run)
        .unwrap();
}

#[test]
#[allow(non_snake_case)]
fn the_entry_cap_evicts_the_oldest_edit_and_says_so__feat__image_editor_undo_journal() {
    // Deterministic companion: cap + 5 edits, then undo until empty.
    let base = vec![0u8; (W * H * 4) as usize];
    let mut stack = LayerStack::from_image(W, H, Arc::from(base.clone())).expect("stack");
    let cap = stack.history().max_entries;
    let mut cur = base;
    let mut states = vec![cur.clone()];
    for k in 0..cap + 5 {
        cur = painted(&cur, (k as u32) % W, 0, 1, 1, k as u8 + 1);
        stack
            .edit_active(
                "paint",
                Region::new(((k as u32) % W) as i32, 0, 1, 1),
                Pixels::from_rgba8(Arc::from(cur.clone())),
            )
            .expect("edit");
        states.push(cur.clone());
    }
    assert_eq!(stack.history().depth, cap);
    assert_eq!(stack.history().dropped, 5);
    let mut undone = 0;
    while stack.undo().is_some() {
        undone += 1;
    }
    assert_eq!(undone, cap, "exactly the retained window undoes");
    assert!(
        stack.layers()[0].rgba.raw() == states[5].as_slice(),
        "undoing the whole window lands on the state after the 5 evicted edits"
    );
}
