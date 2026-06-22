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

//! `PixelFormat` — the total, explicit format descriptor (spec §5.1,
//! "the babl lesson"). Every buffer, tile, and operation input/output
//! carries one; there are NO implicit conversions anywhere in the
//! system. Conversions are compiled paths between two `PixelFormat`s
//! (image-cms / cast kernels), never ad-hoc per-op code.

/// Channel layout. Spec-verbatim set (§5.1); codec-native layouts that
/// don't appear here (e.g. interleaved RGB without alpha) are described
/// by the codec's `SourceInfo` and converted at the slice boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelLayout {
    Gray,
    GrayA,
    Rgba,
    Cmyk,
    Cmyka,
}

impl ChannelLayout {
    pub const fn count(self) -> u8 {
        match self {
            ChannelLayout::Gray => 1,
            ChannelLayout::GrayA => 2,
            ChannelLayout::Rgba => 4,
            ChannelLayout::Cmyk => 4,
            ChannelLayout::Cmyka => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleDepth {
    U8,
    U16,
    F16,
    F32,
}

impl SampleDepth {
    pub const fn bytes(self) -> u8 {
        match self {
            SampleDepth::U8 => 1,
            SampleDepth::U16 | SampleDepth::F16 => 2,
            SampleDepth::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlphaMode {
    None,
    Straight,
    Premultiplied,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transfer {
    Linear,
    Gamma(TransferCurve),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferCurve {
    Srgb,
    Gamma22,
    Gamma18,
}

/// A named (non-ICC) color space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NamedSpace {
    /// Linear-light sRGB primaries — the default document working space
    /// (spec §5.2 working-space policy v1).
    LinearSrgb,
    Srgb,
}

/// Content hash of an interned ICC profile's bytes. Equality is hash
/// equality (§5.1); the resolver/interner lives in image-cms — core
/// only carries the identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IccHash(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpaceRef {
    Named(NamedSpace),
    Icc(IccHash),
}

/// The total format descriptor (spec §5.1, frozen M0 phase 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PixelFormat {
    pub channels: ChannelLayout,
    pub depth: SampleDepth,
    pub alpha: AlphaMode,
    pub transfer: Transfer,
    pub space: ColorSpaceRef,
}

impl PixelFormat {
    /// The GPU production working space (§5.2): rgba16float storage
    /// textures; premultiplied for correct filtering; linear for
    /// correct resampling/convolution.
    pub const GPU_WORKING: PixelFormat = PixelFormat {
        channels: ChannelLayout::Rgba,
        depth: SampleDepth::F16,
        alpha: AlphaMode::Premultiplied,
        transfer: Transfer::Linear,
        space: ColorSpaceRef::Named(NamedSpace::LinearSrgb),
    };

    /// The conformance reference working space (§5.2, test-only
    /// target): F32 headroom; f16 quantization is applied as the FINAL
    /// step before diffing against GPU output (§6.3).
    pub const REF_WORKING: PixelFormat = PixelFormat {
        depth: SampleDepth::F32,
        ..Self::GPU_WORKING
    };

    /// Bytes per pixel for heap-tile layouts (interleaved).
    pub const fn bytes_per_pixel(self) -> usize {
        self.channels.count() as usize * self.depth.bytes() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn working_spaces() {
        assert_eq!(PixelFormat::GPU_WORKING.bytes_per_pixel(), 8);
        assert_eq!(PixelFormat::REF_WORKING.bytes_per_pixel(), 16);
        assert_eq!(PixelFormat::REF_WORKING.channels, ChannelLayout::Rgba);
        assert_ne!(PixelFormat::GPU_WORKING, PixelFormat::REF_WORKING);
    }

    #[test]
    fn icc_equality_is_hash_equality() {
        let a = ColorSpaceRef::Icc(IccHash(42));
        let b = ColorSpaceRef::Icc(IccHash(42));
        assert_eq!(a, b);
    }
}
