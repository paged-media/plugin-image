# NOTICE — third-party attributions

paged.image itself is dual-licensed **MPL-2.0 OR PMEL** (Rust crates; see
`LICENSE` / `LICENSE.md`). This file records attributions that third-party
dependencies require us to reproduce when we distribute the compiled
artifact (`image-js` → the plugin's `.wasm`).

Only dependencies with an **affirmative notice obligation** are listed. The
permissive licenses that carry no such condition (MIT, Apache-2.0, BSD,
ISC, Zlib, …) are covered by the allow list in `deny.toml` and by the
copyright notices retained inside the distributed dependency sources.

---

## Independent JPEG Group (IJG)

> This software is based in part on the work of the Independent JPEG Group.

**Why it applies.** The JPEG encoder in the shipped wasm is the
`jpeg-encoder` crate, reached through `image-codecs` from both `image-js`
and `image-pipeline`. Its `src/fdct.rs` is a Rust port of the forward DCT
from libjpeg-turbo / the IJG's libjpeg — hence the crate's SPDX expression
`(MIT OR Apache-2.0) AND IJG`.

**What the license requires.** The IJG terms are permissive, not copyleft:
use, copying, modification and distribution are granted "for any purpose,
without fee", with no source-disclosure and no reciprocal-licensing
condition. There are three conditions, and only the second binds a
binary-only distribution such as ours:

1. If the *source* is distributed, the IJG README must accompany it with
   its copyright and no-warranty notice unaltered, and changes to the
   original files must be indicated.
2. **If only executable code is distributed, the accompanying
   documentation must state that "this software is based in part on the
   work of the Independent JPEG Group."** ← this file discharges it.
3. Use is permitted only if the user accepts full responsibility — i.e.
   the software is provided AS IS, without warranty.

Copyright (C) 1991-2020, Thomas G. Lane, Guido Vollbeding.
libjpeg-turbo modifications Copyright (C) 2015, 2020, D. R. Commander.

**Maintenance.** `deny.toml` allows the `IJG` identifier and points here.
If the JPEG dependency is replaced or removed, re-check whether this
section still applies before deleting it — and if another IJG-derived
codec is added, this wording already covers it.
