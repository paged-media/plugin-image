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

//! PROPERTIES of the selection algebra (`SelectionCoverage::combine` and
//! `invert`), over generated coverage fields.
//!
//! On coverage the algebra is `add = max`, `intersect = min`,
//! `subtract = a·(1 − b)` and `invert = 255 − v`. `max`/`min` form a
//! distributive lattice, and `255 − v` is an order-reversing involution,
//! so these laws hold EXACTLY on every byte value — not just on binary
//! masks:
//!
//! * add and intersect are commutative, associative and idempotent;
//!   empty is add's identity and intersect's zero, full the reverse;
//! * they distribute over each other, and De Morgan holds through invert;
//! * invert is an involution;
//! * replace yields the shape;
//! * subtract never grows a selection; `a − ∅ = a`, `a − full = ∅`;
//!   on BINARY masks it is exactly `a ∩ ¬b`.

use image_gpu::{CombineMode, SelectionCoverage};
use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

const W: u32 = 9;
const H: u32 = 7;

fn runner() -> TestRunner {
    TestRunner::new_with_rng(
        Config {
            cases: 256,
            failure_persistence: None,
            ..Config::default()
        },
        TestRng::from_seed(RngAlgorithm::ChaCha, &[0x51; 32]),
    )
}

fn cov() -> impl Strategy<Value = SelectionCoverage> {
    proptest::collection::vec(any::<u8>(), (W * H) as usize)
        .prop_map(|d| SelectionCoverage::from_data(W, H, d).expect("sized"))
}

fn binary() -> impl Strategy<Value = SelectionCoverage> {
    proptest::collection::vec(any::<bool>(), (W * H) as usize).prop_map(|d| {
        SelectionCoverage::from_data(
            W,
            H,
            d.into_iter().map(|b| if b { 255 } else { 0 }).collect(),
        )
        .expect("sized")
    })
}

fn op(a: &SelectionCoverage, b: &SelectionCoverage, m: CombineMode) -> SelectionCoverage {
    let mut out = a.clone();
    out.combine(b, m);
    out
}

fn inv(a: &SelectionCoverage) -> SelectionCoverage {
    let mut out = a.clone();
    out.invert();
    out
}

use CombineMode::{Add, Intersect, Replace, Subtract};

#[test]
#[allow(non_snake_case)]
fn add_and_intersect_are_commutative_associative_idempotent__feat__image_selection_mask() {
    runner()
        .run(&(cov(), cov(), cov()), |(a, b, c)| {
            for m in [Add, Intersect] {
                prop_assert_eq!(op(&a, &b, m), op(&b, &a, m), "{:?} commutes", m);
                prop_assert_eq!(
                    op(&op(&a, &b, m), &c, m),
                    op(&a, &op(&b, &c, m), m),
                    "{:?} associates",
                    m
                );
                prop_assert_eq!(op(&a, &a, m), a.clone(), "{:?} is idempotent", m);
            }
            Ok(())
        })
        .unwrap();
}

#[test]
#[allow(non_snake_case)]
fn empty_and_full_are_the_identities_and_zeros__feat__image_selection_mask() {
    let (e, f) = (
        SelectionCoverage::empty(W, H),
        SelectionCoverage::full(W, H),
    );
    runner()
        .run(&cov(), |a| {
            prop_assert_eq!(op(&a, &e, Add), a.clone());
            prop_assert_eq!(op(&a, &f, Add), f.clone());
            prop_assert_eq!(op(&a, &f, Intersect), a.clone());
            prop_assert_eq!(op(&a, &e, Intersect), e.clone());
            prop_assert_eq!(op(&a, &e, Subtract), a.clone());
            prop_assert_eq!(op(&a, &f, Subtract), e.clone());
            Ok(())
        })
        .unwrap();
}

#[test]
#[allow(non_snake_case)]
fn add_and_intersect_distribute_and_obey_de_morgan__feat__image_selection_mask() {
    runner()
        .run(&(cov(), cov(), cov()), |(a, b, c)| {
            prop_assert_eq!(
                op(&a, &op(&b, &c, Add), Intersect),
                op(&op(&a, &b, Intersect), &op(&a, &c, Intersect), Add)
            );
            prop_assert_eq!(
                op(&a, &op(&b, &c, Intersect), Add),
                op(&op(&a, &b, Add), &op(&a, &c, Add), Intersect)
            );
            prop_assert_eq!(inv(&op(&a, &b, Add)), op(&inv(&a), &inv(&b), Intersect));
            prop_assert_eq!(inv(&op(&a, &b, Intersect)), op(&inv(&a), &inv(&b), Add));
            Ok(())
        })
        .unwrap();
}

#[test]
#[allow(non_snake_case)]
fn invert_is_an_involution_and_replace_yields_the_shape__feat__image_selection_mask() {
    runner()
        .run(&(cov(), cov()), |(a, b)| {
            prop_assert_eq!(inv(&inv(&a)), a.clone());
            prop_assert_eq!(op(&a, &b, Replace), b.clone());
            Ok(())
        })
        .unwrap();
}

#[test]
#[allow(non_snake_case)]
fn subtract_never_grows_and_is_intersect_with_the_complement_on_binary__feat__image_selection_mask()
{
    runner()
        .run(&(cov(), cov(), binary(), binary()), |(a, b, p, q)| {
            let d = op(&a, &b, Subtract);
            prop_assert!(
                d.data().iter().zip(a.data()).all(|(x, y)| x <= y),
                "a − b exceeds a somewhere"
            );
            prop_assert_eq!(op(&p, &q, Subtract), op(&p, &inv(&q), Intersect));
            prop_assert_eq!(op(&p, &p, Subtract), SelectionCoverage::empty(W, H));
            Ok(())
        })
        .unwrap();
}
