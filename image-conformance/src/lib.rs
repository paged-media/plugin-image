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

//! The conformance harness (spec §12.4) — TEST-ONLY, never shipped.
//!
//! Goldens come from the scalar references (f32, fixed order); GPU
//! output is verified against them BY TOLERANCE, never byte-golden
//! (spec §6.3). f16 quantization of the reference is the final step
//! before diffing (§5.2).
//!
//! M0 fan-out adds: the PSD fixture builder (`psd_builder`, the
//! INDEPENDENT byte emitter), the proptest layer, the libvips/GEGL
//! differential oracle runners (CI containers), the D-4 criterion
//! bench, and the coverage gate.

pub mod compose_ref;
pub mod device;
pub mod harness;
pub mod psd_builder;
pub mod psd_render;
pub mod quantize;

pub use image_kernels::reference_prelude::Px;
