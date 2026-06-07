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

//! PackBits RLE — decode-tolerant, encode-canonical. We decode any
//! valid PackBits stream (including `-128` no-op bytes, which are
//! skipped); our encoder emits exactly one canonical form. Unedited
//! channels round-trip byte-identically anyway via the verbatim
//! payload (`ChannelData.bytes`), which is why canonical-only encoding
//! is safe (crate-level strategy 3).
//!
//! Provenance: Apple PackBits as referenced by the Adobe Photoshop
//! File Format specification ("RLE compressed ... with the byte counts
//! ... rows are padded to an even size"); TIFF 6.0 spec, PackBits
//! appendix.
//!
//! Canonical encode rule (one form, applied consistently):
//!   * A *replicate* run (control `1 - count`, then one byte) is emitted
//!     only for runs of **3 or more** identical bytes. A run of exactly
//!     two equal bytes is cheaper-or-equal encoded as literals (2 bytes
//!     literal vs. 2 bytes replicate), and folding it into a literal run
//!     keeps the form stable and idempotent.
//!   * Runs longer than 128 are split into 128-byte replicate chunks.
//!   * *Literal* runs (control `len - 1`, then the bytes) accumulate the
//!     non-replicable bytes and are flushed in 128-byte chunks.
//!   * `-128` (0x80) is never produced. We always decode it as a skip.

use crate::{PsdError, Result};

/// Decode one PackBits stream into `out`. `out.len()` must be the
/// exact expected unpacked size (PSD rows are independent streams).
///
/// Tolerant of `-128` no-op control bytes. Errors (`Malformed`) on any
/// stream that would over- or under-fill `out`, or that ends mid-run.
pub fn decode(src: &[u8], out: &mut [u8]) -> Result<()> {
    let mut si = 0usize; // cursor into the packed source
    let mut oi = 0usize; // cursor into the unpacked destination

    while si < src.len() {
        let ctrl = src[si] as i8;
        si += 1;

        if ctrl == -128 {
            // No-op filler — defined by the spec, skipped on decode.
            continue;
        }

        if ctrl >= 0 {
            // Literal run: copy the next `ctrl + 1` bytes verbatim.
            let n = ctrl as usize + 1;
            if si + n > src.len() {
                return Err(malformed(format!(
                    "literal run of {n} bytes overruns source ({} byte(s) left)",
                    src.len() - si
                )));
            }
            if oi + n > out.len() {
                return Err(malformed(format!(
                    "literal run of {n} bytes overruns output ({} byte(s) left)",
                    out.len() - oi
                )));
            }
            out[oi..oi + n].copy_from_slice(&src[si..si + n]);
            si += n;
            oi += n;
        } else {
            // Replicate run: repeat the next byte `1 - ctrl` times.
            let n = 1 - ctrl as isize;
            let n = n as usize; // ctrl ∈ [-127,-1] ⇒ n ∈ [2,128]
            if si >= src.len() {
                return Err(malformed("replicate run missing its value byte".into()));
            }
            let value = src[si];
            si += 1;
            if oi + n > out.len() {
                return Err(malformed(format!(
                    "replicate run of {n} bytes overruns output ({} byte(s) left)",
                    out.len() - oi
                )));
            }
            for slot in &mut out[oi..oi + n] {
                *slot = value;
            }
            oi += n;
        }
    }

    if oi != out.len() {
        return Err(malformed(format!(
            "stream produced {oi} byte(s), expected {}",
            out.len()
        )));
    }
    Ok(())
}

/// Encode `src` into the canonical PackBits form (see module docs).
pub fn encode(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // Pending literal bytes awaiting flush. Held until we either hit a
    // replicable run (≥3) or the 128-byte literal cap.
    let mut literals: Vec<u8> = Vec::new();
    let mut i = 0usize;

    while i < src.len() {
        // Measure the run of identical bytes starting at `i`.
        let run_byte = src[i];
        let mut run_len = 1usize;
        while i + run_len < src.len() && src[i + run_len] == run_byte {
            run_len += 1;
        }

        if run_len >= 3 {
            // Replicable: flush any pending literals, then emit the run
            // as one or more 128-byte replicate packets.
            flush_literals(&mut literals, &mut out);
            let mut remaining = run_len;
            while remaining > 0 {
                let chunk = remaining.min(128);
                // control = 1 - count, count ∈ [3,128] ⇒ ctrl ∈ [-127,-2]
                out.push((1i32 - chunk as i32) as u8);
                out.push(run_byte);
                remaining -= chunk;
            }
            i += run_len;
        } else {
            // Not worth a replicate packet (run of 1 or 2): treat the
            // bytes as literals, flushing whenever we reach the cap.
            for _ in 0..run_len {
                literals.push(run_byte);
                if literals.len() == 128 {
                    flush_literals(&mut literals, &mut out);
                }
            }
            i += run_len;
        }
    }

    flush_literals(&mut literals, &mut out);
    out
}

/// Emit a literal packet for the pending bytes (in ≤128-byte chunks)
/// and clear the buffer. A no-op when empty.
fn flush_literals(literals: &mut Vec<u8>, out: &mut Vec<u8>) {
    let mut start = 0usize;
    while start < literals.len() {
        let n = (literals.len() - start).min(128);
        // control = len - 1, len ∈ [1,128] ⇒ ctrl ∈ [0,127]
        out.push((n - 1) as u8);
        out.extend_from_slice(&literals[start..start + n]);
        start += n;
    }
    literals.clear();
}

#[inline]
fn malformed(detail: String) -> PsdError {
    PsdError::Malformed {
        section: "packbits",
        detail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: decode into an exactly-sized buffer.
    fn dec(src: &[u8], expected_len: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; expected_len];
        decode(src, &mut out)?;
        Ok(out)
    }

    #[test]
    fn image_psd_packbits_empty_input() {
        assert_eq!(encode(&[]), Vec::<u8>::new());
        assert_eq!(dec(&[], 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn image_psd_packbits_single_byte() {
        let data = [0x42u8];
        let enc = encode(&data);
        // One byte ⇒ a 1-byte literal packet: ctrl 0, then the byte.
        assert_eq!(enc, vec![0x00, 0x42]);
        assert_eq!(dec(&enc, 1).unwrap(), data);
    }

    #[test]
    fn image_psd_packbits_max_literal_run() {
        // 128 all-distinct bytes ⇒ a single literal packet (ctrl 127).
        let data: Vec<u8> = (0..128u16).map(|n| n as u8).collect();
        let enc = encode(&data);
        assert_eq!(enc.len(), 1 + 128);
        assert_eq!(enc[0], 127);
        assert_eq!(&enc[1..], &data[..]);
        assert_eq!(dec(&enc, 128).unwrap(), data);
    }

    #[test]
    fn image_psd_packbits_over_max_literal_run() {
        // 129 distinct-ish bytes ⇒ literal packet split at 128 + 1.
        let data: Vec<u8> = (0..129u16).map(|n| (n * 7) as u8).collect();
        // Guard against an accidental ≥3 replicate run in the synthetic data.
        assert!(!has_triple_run(&data));
        let enc = encode(&data);
        assert_eq!(enc[0], 127); // first packet: 128 literals
        assert_eq!(enc[1 + 128], 0); // second packet: 1 literal
        assert_eq!(dec(&enc, 129).unwrap(), data);
    }

    #[test]
    fn image_psd_packbits_max_replicate_run() {
        // 128 identical bytes ⇒ one replicate packet: ctrl -127, value.
        let data = vec![0xABu8; 128];
        let enc = encode(&data);
        assert_eq!(enc, vec![(1i32 - 128) as u8, 0xAB]); // 0x81, 0xAB
        assert_eq!(dec(&enc, 128).unwrap(), data);
    }

    #[test]
    fn image_psd_packbits_over_max_replicate_run() {
        // 300 identical ⇒ 128 + 128 + 44 replicate packets.
        let data = vec![0x7Fu8; 300];
        let enc = encode(&data);
        assert_eq!(
            enc,
            vec![
                (1i32 - 128) as u8,
                0x7F,
                (1i32 - 128) as u8,
                0x7F,
                (1i32 - 44) as u8,
                0x7F,
            ]
        );
        assert_eq!(dec(&enc, 300).unwrap(), data);
    }

    #[test]
    fn image_psd_packbits_decode_skips_noop_byte() {
        // A `-128` (0x80) control byte mid-stream is skipped: the output
        // is "AB" + "B" + "CC..." with the no-op in the middle.
        // [literal 1: A][noop][literal 1: B][replicate 3: C]
        let src = [0x00, b'A', 0x80, 0x00, b'B', (1i32 - 3) as u8, b'C'];
        let out = dec(&src, 5).unwrap();
        assert_eq!(out, b"ABCCC");
    }

    #[test]
    fn image_psd_packbits_alternating_literal_replicate() {
        // L L L (literal) R R R R (replicate) L (literal) ...
        let data = b"abcZZZZq".to_vec();
        let enc = encode(&data);
        // Expect: literal[abc], replicate[4×Z], literal[q].
        assert_eq!(
            enc,
            vec![2, b'a', b'b', b'c', (1i32 - 4) as u8, b'Z', 0, b'q'],
        );
        assert_eq!(dec(&enc, data.len()).unwrap(), data);
    }

    #[test]
    fn image_psd_packbits_two_byte_run_stays_literal() {
        // Exactly-two equal bytes do NOT become a replicate packet — our
        // canonical rule folds them into the surrounding literal run.
        let data = b"xyyz".to_vec();
        let enc = encode(&data);
        assert_eq!(enc, vec![3, b'x', b'y', b'y', b'z']);
        assert_eq!(dec(&enc, data.len()).unwrap(), data);
    }

    #[test]
    fn image_psd_packbits_ends_on_run_boundary() {
        // Stream ends exactly when a replicate run completes (no trailing
        // literal packet to flush).
        let data = b"abcDDD".to_vec();
        let enc = encode(&data);
        assert_eq!(enc, vec![2, b'a', b'b', b'c', (1i32 - 3) as u8, b'D']);
        // Last byte of the encoding is the replicate value — boundary end.
        assert_eq!(*enc.last().unwrap(), b'D');
        assert_eq!(dec(&enc, data.len()).unwrap(), data);
    }

    #[test]
    fn image_psd_packbits_roundtrip_assorted() {
        let cases: Vec<Vec<u8>> = vec![
            vec![],
            vec![0],
            vec![0xFF; 1],
            vec![5; 256],                           // all-same, > one max run
            (0..256u16).map(|n| n as u8).collect(), // all-distinct 256
            (0..400u16).map(|n| n as u8).collect(), // all-distinct 400
            b"the quick brown fox jumped".to_vec(),
            vec![1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6],
            {
                // mixed: long replicate, then long literal, then replicate
                let mut v = vec![9u8; 200];
                v.extend((0..200u16).map(|n| (n * 13) as u8));
                v.extend(vec![3u8; 130]);
                v
            },
        ];
        for data in &cases {
            let enc = encode(data);
            let dec = dec(&enc, data.len()).unwrap();
            assert_eq!(&dec, data, "roundtrip failed for {} bytes", data.len());
        }
    }

    #[test]
    fn image_psd_packbits_canonical_idempotence() {
        // encode(decode(canonical)) == canonical for every assorted case:
        // our encoder reproduces its own output exactly.
        let cases: Vec<Vec<u8>> = vec![
            vec![5; 256],
            (0..300u16).map(|n| n as u8).collect(),
            b"aaabbbcccdddeee".to_vec(),
            {
                let mut v = vec![0xAAu8; 300];
                v.extend(b"literals!");
                v.extend(vec![0xBBu8; 5]);
                v
            },
        ];
        for data in &cases {
            let canonical = encode(data);
            let round = dec(&canonical, data.len()).unwrap();
            let recanonical = encode(&round);
            assert_eq!(
                recanonical,
                canonical,
                "canonical encode not idempotent for {} bytes",
                data.len()
            );
        }
    }

    #[test]
    fn image_psd_packbits_malformed_truncated_literal() {
        // Literal packet claims 3 bytes but the source ends after 1.
        let src = [0x02, b'A'];
        let err = dec(&src, 3).unwrap_err();
        assert!(matches!(
            err,
            PsdError::Malformed {
                section: "packbits",
                ..
            }
        ));
    }

    #[test]
    fn image_psd_packbits_malformed_replicate_missing_value() {
        // Replicate control byte with no following value byte.
        let src = [(1i32 - 3) as u8];
        let err = dec(&src, 3).unwrap_err();
        assert!(matches!(
            err,
            PsdError::Malformed {
                section: "packbits",
                ..
            }
        ));
    }

    #[test]
    fn image_psd_packbits_malformed_output_overrun() {
        // A replicate run of 10 into a 3-byte buffer overruns the output.
        let src = [(1i32 - 10) as u8, b'Z'];
        let err = dec(&src, 3).unwrap_err();
        assert!(matches!(
            err,
            PsdError::Malformed {
                section: "packbits",
                ..
            }
        ));
    }

    #[test]
    fn image_psd_packbits_malformed_underrun() {
        // Stream produces fewer bytes than the buffer expects.
        let src = [0x00, b'A']; // emits 1 byte
        let err = dec(&src, 4).unwrap_err();
        assert!(matches!(
            err,
            PsdError::Malformed {
                section: "packbits",
                ..
            }
        ));
    }

    /// True if `data` contains any run of ≥3 identical consecutive bytes.
    fn has_triple_run(data: &[u8]) -> bool {
        data.windows(3).any(|w| w[0] == w[1] && w[1] == w[2])
    }
}
