# `.abr` corpus derived artifacts — ANALYST-published, IMPLEMENTER-legal

**Produced by:** an ANALYST-role session under the clean-room two-role
protocol (`plugin-image/CLAUDE.md` §3.1).
**Consumed by:** anyone, including IMPLEMENTER-role sessions and CI.
**Specified by:** `thoughts/docs/paged/plugin-image/abr-brush-format-spec.md`
§13.4 (what they are) and §14.3.1 (the gate they exist to make possible).

## Why these files exist

The `.abr` behaviour spec was verified against nine licensed `.abr` files —
3,215 brush presets, 3,202 sampled tips — plus a 238-image published-PNG
oracle. That corpus lives at `references/abr-fixtures/`, which is
**gitignored and inside the clean-room mount**: the ANALYST role may read
it, the IMPLEMENTER role may not, and CI has no copy of it at all.

Revision 2 of the spec nonetheless closed by telling the implementer to wire
the corpus into the test suite as the regression gate. That instruction had
no owner — following it breaks the isolation rule, honouring the isolation
rule leaves the gate unbuilt. It was reported back as errata item 8 and is
resolved by these files.

The resolution: the analyst reads the mount and emits **facts about
behaviour**. Those facts cross the boundary; the bytes do not.

## Why they are facts about behaviour and not expression

1. **What they describe is Adobe's file format.** Key names, OSTypes, unit
   codes, bounding rectangles and record counts are properties of bytes
   Photoshop wrote — the same class of claim as the rest of the behaviour
   spec.
2. **A digest is one-way.** `sha256(decoded coverage)` pins a decode
   exactly — one flipped bit anywhere in 3,202 records fails the gate —
   while carrying none of the picture. It cannot be inverted, so publishing
   it redistributes no third-party artwork. That is precisely why these
   files can be committed where the `.abr` files cannot.
3. **They were produced without the reference implementation.** The
   measuring parser was written from the behaviour spec. Nothing here is
   transcribed, transliterated or paraphrased from `references/ag-psd`.

The line that must not be crossed: nothing may be published from which the
corpus could be **reconstructed** — no pixel data, no RLE payloads, no
verbatim descriptor byte ranges. Neither file below carries any, and any
future extension of them must keep that line.

## The files

### `corpus-profile.json`

Whole-corpus derived tables:

- `files[]` — per fixture: byte size, major/minor version, the section
  kind/size list in file order, whether the file ends on the padded
  boundary, descriptor version, brush count, sampled-tip count.
- `ostype_counts`, `key_form_counts` — the OSType histogram and the
  long-form-vs-4-byte key split.
- `class_id_counts`, `tip_class_counts` — every class id that occurs, with
  counts.
- `key_ostype_counts`, `key_unit_counts` — the **key × OSType** and
  **key × unit** tables: 102 distinct keys as they actually occur.
- `gate_counts` — present / absent / true per gate. Note `useBrushPose`:
  absent on 36 of 3,215 brushes, which is why a missing gate must read as
  `false`.
- `ordinal_value_counts` — every observed value of `bVTy`, `fStp`, `Shp `
  (bristle and erodible separately) and `dtipsType`.
- `enum_pair_counts` — every observed enum type-key/value-key pair, in the
  dialect it arrived in.
- `dynamics_key_sets`, `join_counts`, `sampled_data_length_counts`.

### `corpus-record-ledger.tsv`

One row per sampled-tip record, 3,202 rows, in file order. Columns:

```
file  index  id  declared_len  pad_len  array_count  written_planes
top  left  bottom  right  w  h  depth  compression  decoded_bytes
sha256  png_oracle
```

`sha256` is the SHA-256 of the **decoded coverage mask** — row-major, one
byte per pixel, exactly `w * h` bytes, **not inverted**. `png_oracle`, on
238 rows, names the independently published transparent PNG whose *alpha
channel* hashes to the same value; that is the artefact that settles the
polarity question (spec §2.5), and it is 238 of 238 exact.

`pad_len` is `rounded_len - declared_len ∈ 0..=3`. It is in the ledger
because the spec's structural self-check is measured against the
**declared** extent, and 2,367 of 3,202 records carry a non-zero pad — a
check written against the rounded extent fails three quarters of the corpus.

## How to use them

**Lane A — implementer-owned, always on, no corpus needed.** Drive the
synthesised-fixture builder (`image-conformance/src/abr_builder`) from the
profile: emit a fixture for every `key|OSType` and `key|unit` pair, every
class id, every ordinal value, every container shape, the absent-gate case,
and each pad remainder. See spec §14.3.1 Lane A for the seven checks.

**Lane B — analyst-owned, opt-in, needs the mount.**

```bash
PAGED_ABR_CORPUS=1 cargo test -p image-conformance --test abr_corpus -- --ignored
```

Parses the real fixtures and compares them to these files row by row. It is
skipped, loudly, wherever `references/abr-fixtures/` is absent — which is
every machine that is not an analyst's.

## Regeneration

**Only an ANALYST session with the mount may regenerate these files**, and
the measuring parser must be written from the behaviour spec rather than
from anything in `references/`. A change to either file is a reviewable
diff, and that is the point: a Lane-B failure then means either the reader
regressed or the spec legitimately changed. **Never "fix" a Lane-B failure
by editing the expectation files.**
