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

//! ICC profile interner (spec §5.1): profiles are interned by content
//! hash; `ColorSpaceRef::Icc` equality IS hash equality. Profiles
//! travel with documents and survive serialization — the interner is
//! the lookup from identity back to bytes.

use std::collections::HashMap;
use std::sync::Arc;

use image_core::{ContentHash, IccHash};

use crate::Profile;

#[derive(Debug, Default)]
pub struct ProfileInterner {
    by_hash: HashMap<IccHash, Profile>,
}

impl ProfileInterner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, bytes: impl Into<Arc<[u8]>>) -> IccHash {
        let bytes: Arc<[u8]> = bytes.into();
        let hash = IccHash(ContentHash::of(&bytes).0);
        self.by_hash
            .entry(hash)
            .or_insert_with(|| Profile { hash, bytes });
        hash
    }

    pub fn get(&self, hash: IccHash) -> Option<&Profile> {
        self.by_hash.get(&hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_bytes_same_identity() {
        let mut i = ProfileInterner::new();
        let a = i.intern(vec![1u8, 2, 3].into_boxed_slice());
        let b = i.intern(vec![1u8, 2, 3].into_boxed_slice());
        let c = i.intern(vec![9u8].into_boxed_slice());
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(&*i.get(a).unwrap().bytes, &[1, 2, 3][..]);
    }
}
