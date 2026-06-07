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

//! The synthesized-PSD fixture builder — an INDEPENDENT byte emitter
//! (its own big-endian writer and its own PackBits encoder), a
//! deliberately separate code path from image-psd's production writer
//! so round-trip tests are never self-referential. The M0 corpus (the
//! 11 named fixtures) lives in `fixtures`.
//!
//! Lands with M0 fan-out unit U8 (emit/layers/channels/fixtures).
