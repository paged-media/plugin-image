# ADR 456 — Codecs are pure-Rust, permissively licensed adapters behind I/O-free traits

- **Status:** Accepted. Recorded retroactively on 2026-10-02 from the code at `f7d21e5`.
- **Scope:** `image-codecs`, `registry/codecs.yaml`, `deny.toml`, `NOTICE.md`

## Context

The plugin decodes raster files into pixels and encodes edited pixels back into files. Kernels
run on the GPU ([ADR 450](450-gpu-only-kernels.md)); the codec crate's header sets codecs
apart from that rule: "Codecs are inherently CPU work and remain so"
(`image-codecs/src/lib.rs:37`). The same header gives the reason for the trait shape:
sources and targets do no I/O of their own "so the same adapters serve browser (memory /
OPFS / ReadableStream) and native (file) builds" (`image-codecs/src/lib.rs:35-37`).

The rules a codec dependency has to meet are written in the registry: "pure-Rust, no vendored
C, cargo-deny-permissive" (`registry/codecs.yaml:147-148`). `deny.toml` carries the allow
list of licences and denies crates from unknown registries and git sources.

## Decision

Decode and encode sit behind two traits, `ImageSource` (`probe`, `native_shrink`,
`read_region`) and `ImageTarget` (`begin`, `write_strip`, `finish`). A decoder reads its
input through `ByteSource`, a trait with `len` and `read_at` and no file or network type.
There is one adapter per format; the PNG and JPEG adapters wrap codecs from crates.io:

- PNG decode and encode: `zune-png` 0.5, chosen over the `png` crate by a benchmark kept in
  the repo. The registry records that it "wins every cell, decode and encode".
- JPEG decode: `zune-jpeg` 0.5.15. JPEG encode: `jpeg-encoder` 0.7.0.
- `RawSource` / `RawTarget`: an uncompressed interleaved buffer with no container.
- EXIF: a reader written in the crate for orientation, resolution and the colour-space tag.
  The registry states why no EXIF crate was taken: three tags need "a ~150-line bounded IFD
  walk", and a full crate would add a dependency and a second TIFF reader.

PSD is not behind these traits; it has its own crate ([ADR 458](458-psd-preservation.md)).

## Evidence

- `image-codecs/src/source.rs:73-86`, `image-codecs/src/target.rs:56-64` — the two traits
- `image-codecs/src/bytesource.rs:39-49`, `:64` — `ByteSource` and its only implementation
- `image-codecs/Cargo.toml:16-31`, `Cargo.toml:60` — the three codec dependencies; `zune-png` takes its version from the workspace entry
- `registry/codecs.yaml:24-36`, `:109-113` — the PNG benchmark result (the benchmark is
  `image-conformance/benches/png_d4.rs`); why the EXIF reader is hand-written
- `deny.toml:17-43`, `:50-53` — the licence allow list; unknown registries and git denied
- `NOTICE.md:15-23` — the IJG attribution, owed because `jpeg-encoder` ports libjpeg code
- `image-js/src/ingest.rs:530-546`, `image-js/src/saveback.rs:303-346` — the shipped decode
  (format chosen by magic bytes) and encode calls

## Alternatives considered

- **The `png` crate** lost the benchmark. It remains a dev-dependency of `image-conformance`
  as the comparison baseline only (`image-conformance/Cargo.toml:31-36`). **An EXIF crate**
  (the registry names kamadak-exif) was declined, as above.
- **AVIF**, not built (`registry/codecs.yaml:141-197`). Decode: `avif-decode` and `dav1d-rs`
  need C, `rav1d` exposes only a C API, and the licence of `rav1d-safe` is not on the allow
  list. Encode through `ravif` passes the licence rule but was deferred on wasm size and
  because, without a decoder, it cannot be tested by round trip.
- **JPEG XL**, not built (`registry/codecs.yaml:199-237`). Decode through `jxl-oxide` passes
  the licence rule and was deferred on scope and wasm size; the licence of the only
  pure-Rust encoder is not on the allow list.
- **Camera RAW and HEIC/HEIF**: "EXPLICIT NON-GOAL, not a deferral"
  (`registry/codecs.yaml:239-259`). The decoders named there are C, C++ or LGPL-licensed.

## Consequences

The plugin decodes PSD (through `image-psd`), PNG and JPEG, and encodes the same three.
`image-codecs` has no adapter for any other file format; unrecognised bytes are refused
(`image-js/src/ingest.rs:542-544`). A binary distribution has to carry the IJG sentence:
`glue/package.json:17` copies `NOTICE.md` into the npm package at publish and `:39` ships it.

The I/O-free seam is only partly used. `MemoryByteSource` is the one `ByteSource`;
`image-codecs/src/bytesource.rs:33-35` names file and OPFS backings as later work. Both
decoders decode the whole image on the first `read_region`, serve windows by copy, and
report `native_shrink` as `[1]` (`image-codecs/src/png/mod.rs:50-54`,
`image-codecs/src/jpeg/mod.rs:60-68`). PNG decode keeps 16-bit samples
(`image-codecs/src/png/decode.rs:158-166`), but PNG encode accepts 8-bit only and writes no
ICC profile (`image-codecs/src/png/encode.rs:127-133`, `:144-146`). JPEG encode takes RGB
and Gray, not CMYK (`image-codecs/src/jpeg/mod.rs:70-74`).

Several comments are behind the code. `image-codecs/src/png/mod.rs:38-40` and
`registry/codecs.yaml:37-38` say 16-bit PNG input is refused. `image-codecs/Cargo.toml:28-29`
and `registry/codecs.yaml:88-89` say the IJG term is not yet on the allow list; `deny.toml:42`
has it. The AVIF, JPEG XL and EXIF rows argue from an 8 MiB wasm budget; `scripts/build-wasm.sh:17`
states 100 MB for the whole application, and `glue/manifest.json:35` still declares 8 MiB.

## Related

- [ADR 450](450-gpu-only-kernels.md) — what runs on the GPU and what stays on the CPU
- [ADR 457](457-colour-management.md) — what happens to an embedded ICC profile after decode
- [ADR 460](460-document-is-not-the-store.md) — the exporters that call the encoders
- [ADR 308](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/308-plugin-wasm.md) — the wasm size budget
