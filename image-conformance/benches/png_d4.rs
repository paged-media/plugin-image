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

//! D-4: zune-png vs image-rs/png decode/encode throughput over
//! synthetic PNGs (varying size / bit depth / filtering). Lands with
//! the M0 fan-out codecs unit; the winner is recorded in
//! registry/codecs.yaml with provenance and re-confirmed against the
//! Links corpus before the M1 freeze (spec §10.3 corpus rule).

fn main() {
    // criterion benches land with the codecs unit (D-4).
}
