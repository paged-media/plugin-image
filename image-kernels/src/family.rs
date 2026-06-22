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

//! `kernel_family!` — dual emission from ONE expression (spec §6.2).
//!
//! Each invocation defines, from a single `eval:` body:
//!
//! 1. the `#[repr(C)]` param block (bytemuck `Pod`; byte-identity IS
//!    param identity — no `Hash` on f32 fields),
//! 2. the `KernelDef` static whose `wgsl` field is the stringified
//!    body (the PRODUCTION implementation, spliced into the ABI
//!    template by `abi::assemble`),
//! 3. behind `feature = "reference"` (test-only, enabled solely by
//!    image-conformance): the scalar Rust twin over
//!    [`reference_prelude::Px`].
//!
//! The body is written in a restricted DSL whose every token is valid
//! in BOTH languages: the idents `a`, `b` (vec4 samples), `p.<field>`
//! (params), float literals with a decimal point, the operators
//! `+ - * /`, the shared helpers (`splat4`) and the whitelisted
//! builtins (`clamp`, `mix`, `min`, `max`, `abs`, `floor`). Anything
//! outside the whitelist fails to compile in the reference arm — WGSL ≡
//! Rust divergence is impossible by construction, and the
//! golden-expansion conformance test snapshots both emissions.

/// See module docs. Layout note: param fields are restricted to
/// `f32`/`u32`/`i32` (all 4-byte aligned ⇒ no implicit padding ⇒ `Pod`
/// holds); the macro appends a trailing `_abi_pad: u32` so empty param
/// lists still form a valid (non-zero-sized) uniform block — the WGSL
/// struct emitted by `abi::assemble` appends the same field.
#[macro_export]
macro_rules! kernel_family {
    (
        $(#[doc = $doc:literal])*
        static $def:ident, params $params:ident, ref $refname:ident {
            id: $id:literal,
            class: $class:expr,
            inputs: $inputs:literal,
            params: { $( $pf:ident : $pty:ident ),* $(,)? },
            eval: |$a:ident, $b:ident, $p:ident| $body:expr,
            mip_exact: $mip:literal,
            tolerance: $tol:expr $(,)?
        }
    ) => {
        $(#[doc = $doc])*
        #[repr(C)]
        #[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
        pub struct $params {
            $( pub $pf: $pty, )*
            /// ABI tail pad — always 0 (see `kernel_family!` docs).
            pub _abi_pad: u32,
        }

        // Param-less kernels generate a zero-arg `new` (clippy wants
        // Default there; the uniform constructor shape wins).
        #[allow(clippy::new_without_default)]
        impl $params {
            pub fn new( $( $pf: $pty ),* ) -> Self {
                Self { $( $pf, )* _abi_pad: 0 }
            }

            /// The uniform upload / cache-key bytes (param identity is
            /// byte identity).
            pub fn as_bytes(&self) -> &[u8] {
                ::bytemuck::bytes_of(self)
            }
        }

        $(#[doc = $doc])*
        pub static $def: $crate::KernelDef = $crate::KernelDef {
            id: $id,
            class: $class,
            inputs: $inputs,
            params: $crate::ParamsLayout {
                size: ::core::mem::size_of::<$params>(),
                fields: &[
                    $( $crate::ParamField {
                        name: stringify!($pf),
                        wgsl_ty: stringify!($pty),
                    }, )*
                ],
            },
            wgsl: stringify!($body),
            module: false,
            mip_exact: $mip,
            gpu_tolerance: $tol,
        };

        /// Scalar reference twin (test-only golden source, spec §6.3).
        #[cfg(feature = "reference")]
        pub fn $refname(
            a: $crate::reference_prelude::Px,
            b: $crate::reference_prelude::Px,
            p: &$params,
        ) -> $crate::reference_prelude::Px {
            #[allow(unused_imports)]
            use $crate::reference_prelude::*;
            let $a = a;
            let $b = b;
            let $p = p;
            // Unary kernels ignore `b`; param-less kernels ignore `p`.
            let _ = (&$a, &$b, &$p);
            $body
        }
    };
}
