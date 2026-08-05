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
  boundary, descriptor version, brush count, sampled-tip count. Two of
  those — `last_section_padded` and `descriptor_version` — carry a caveat;
  see `_notes`.
- `ostype_counts`, `key_form_counts` — the OSType histogram and the
  long-form-vs-4-byte key split.
- `class_id_counts`, `tip_class_counts` — every class id that occurs, with
  counts.
- `key_ostype_counts`, `key_unit_counts` — the **key × OSType** and
  **key × unit** tables: 102 distinct keys as they actually occur (103
  `key|OSType` pairs — one key occurs with two OSTypes).
- `gate_counts` — present / absent / true per gate. Note `useBrushPose`:
  absent on 36 of 3,215 brushes, which is why a missing gate must read as
  `false`.
- `ordinal_value_counts` — every observed value of `bVTy`, `fStp`, `Shp `
  (bristle and erodible separately) and `dtipsType`.
- `enum_pair_counts` — every observed enum type-key/value-key pair, in the
  dialect it arrived in.
- `dynamics_key_sets`, `join_counts`, `sampled_data_length_counts`.
- `png_oracle` — where the 238-image polarity oracle lives and how to point
  the gate at it. See "The PNG oracle" below.
- `_notes` — the two fields that carry a caveat rather than a plain value:
  `last_section_padded` and `descriptor_version`. See "Two fields that are
  not what they look like" below.

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

> **`png_oracle` is a path, not a bare filename** (changed 2026-08-05). It
> is relative to the root of the published PNG pack, which is a nested
> folder tree. See below.

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
CI and every checkout without the mount. Note that this is a *property of
the machine, not of the role*: on a machine that has the mount, this command
runs whoever types it. See "Running Lane B leaks nothing" below, which is
why that is acceptable.

B3 additionally needs `PAGED_ABR_PNG_DIR` — see "The PNG oracle".

## The PNG oracle

The 238-image polarity oracle is the **Mercator Settlement PNG Pack** by
K. M. Alexander — the same author's separate PNG export of the artwork in
`kma-mercator-settlement-brushes.abr`, released **CC0-1.0** (the pack's own
`_ReadMe_.txt` carries the dedication verbatim). It is what makes the
polarity of §2.5 *decidable* rather than merely probable: a second export
from a second pipeline, outside the `.abr` ecosystem entirely.

In the mount it sits at `references/abr-fixtures/mercator-png-oracle/`, and
it is a **nested tree**, not a flat directory — `Settlements/` (134),
`Landforms/` (56), `Flora/` (46), `Map Components/` (2), most with one
further level. That is why the ledger's `png_oracle` column now carries the
folder: a bare basename joined onto the directory opens nothing.

```bash
PAGED_ABR_CORPUS=1 \
PAGED_ABR_PNG_DIR="$PWD/references/abr-fixtures/mercator-png-oracle" \
  cargo test -p image-conformance --test abr_corpus -- --ignored --nocapture
```

**Why this section exists.** The ledger named the 238 files; nothing
published said where they were. B3 therefore defaulted to a directory that
does not exist under that name (`<corpus>/png-oracle`) and skipped itself —
on every machine, *including the analyst's*. It was the one Lane-B gate
that had never run. On 2026-08-05 it ran for the first time and passed:
**238 of 238 tips byte-identical to the published alpha, 0 matching the
inverted reading.** The mapping is bijective — every ledger name resolves,
no pack file is unused, no basename collides.

The remaining nicety is one line of `tests/abr_corpus.rs`: reading
`png_oracle.dir` from the profile instead of hard-coding `png-oracle` would
make B3 run with no environment variable at all. Reported rather than made,
so the artifact change and the code change stay separately reviewable.

## Two fields that are not what they look like

**`files[].last_section_padded` is undecidable when the last section's size
is a multiple of 4.** The flag records whether the final section is followed
by its 0–3 pad bytes. When `size % 4 == 0` that pad is *zero bytes long*, so
the padded and unpadded terminations emit **identical bytes** and nothing
can tell them apart. A value published for such a file would record how the
measuring parser broke the tie, not a fact about the file.

It happens not to arise here. Across the nine files the last section's size
mod 4 is `{1: 4, 2: 2, 3: 3}` — never 0 — so every published value *is*
observable, from the file's total byte count alone. Lane A asserts the flag
only where it is observable and covers both terminations with a dedicated
fixture. **Do not later "fix" a 4-aligned case: there is nothing there to be
right or wrong about.**

**`files[].descriptor_version` is per section, summarised per file.** A
`.abr` carries one descriptor version word per descriptor-bearing section —
`desc`, plus `phry` where present — so these nine files hold fourteen of
them (9 + 5). All fourteen are 16, so the single published number is
well-formed; it would stop being well-formed the day two sections disagreed.

Lane B's B1 asserts it via the *absence* of an
`UnexpectedDescriptorVersion` warning rather than by value, and **that is
correct, not a workaround.** The reader emits that warning for every
descriptor section whose version is not 16, and the warning carries both the
section name and the value — so the reader's silence is *logically
equivalent* to "every descriptor section in this file is at version 16",
which is exactly what the field claims. It is not a weaker assertion.

**The reader should not be changed to surface it.** A gate whose job is to
check the reader should measure independently — which is why B1 re-derives
the section list from the bytes rather than asking `AbrFile` for it. Adding
`AbrFile::descriptor_version` would let the gate check the reader against
itself. Lane B already has the independent measurement available for free:
`read_versioned_descriptor` returns the version, and B4 obtains it today and
discards it into `_version`.

## Running Lane B leaks nothing

Lane B compares a real parse against **aggregates this directory already
publishes**. Its output is pass/fail plus the counts in `corpus-profile.json`
and the digests in `corpus-record-ledger.tsv` — both committed, both
implementer-legal by construction. Running it therefore discloses nothing
that reading these two files does not, and an accidental run — someone
copying the documented command onto a machine that turns out to have the
mount — is a non-event.

"Analyst-owned" is a **convention about who maintains the expectations**,
not a mechanism that keeps anyone out. `PAGED_ABR_CORPUS=1` is a switch, not
a credential. That is deliberate and it is fine: what the clean-room rule
actually forbids is *reading `references/`* — opening, listing or grepping
the mount, and in particular `references/ag-psd` — and no test run does
that on anyone's behalf. **Do not add a gate that tries to detect who is
running Lane B.** It would be theatre: unenforceable, and aimed at a
disclosure that does not occur.

What *is* load-bearing is the ownership rule below — that the expectation
files are regenerated only by a session with the mount, and never edited to
make a red lane go green.

## Regeneration

**Only an ANALYST session with the mount may regenerate these files**, and
the measuring parser must be written from the behaviour spec rather than
from anything in `references/`. A change to either file is a reviewable
diff, and that is the point: a Lane-B failure then means either the reader
regressed or the spec legitimately changed. **Never "fix" a Lane-B failure
by editing the expectation files.**
