# ADR 455 — The pixel model: explicit format, 16-bit float working tiles

- **Status:** Accepted. Recorded retroactively on 2026-10-02 from the code at `f7d21e5`.
- **Scope:** `PixelFormat` in `image-core`; the GPU texture format in `image-kernels` and `image-gpu`; `Pixels` and the upload and readback bridges in `image-js` and `image-pipeline`

## Context

`image-core` describes a pixel buffer by a `PixelFormat`: channel layout, sample depth,
alpha mode, transfer and colour space. Its header calls this "the babl lesson" and states
that the system has no implicit conversions (`image-core/src/format.rs:33-37`). The same
file defines the GPU working space and gives its reasons, premultiplied for filtering and
linear for resampling and convolution (`image-core/src/format.rs:133-135`).

`image-js` at first held decoded images and layers as bare 8-bit byte buffers. When 16-bit
sources were to keep their depth (August 2026), a new type was introduced. Its module
records why: with bare bytes, every `i * 4` and `chunks_exact(4)` would keep compiling and
read the wrong pixel (`image-js/src/pixels.rs:19-24`).

## Decision

Format is explicit, the GPU computes in `rgba16float`, and held pixels carry their depth.

- `PixelFormat` is a field of `Tile`, `TileMap`, tile slices and codec source info. A codec
  adapter refuses a slice whose format differs from its own.
- `PixelFormat::GPU_WORKING` is RGBA, F16, premultiplied, linear, linear sRGB primaries;
  `REF_WORKING` is the same with F32. Every kernel writes an `rgba16float` texture.
- Images and layers held by `image-js` are `Pixels`: straight RGBA at 8 or 16 bits per
  sample, with no `Deref` to bytes. A reader chooses `to_rgba8()` (keeps the high byte),
  `sample16()` (widens 8-bit as `(v << 8) | v`, so 255 becomes 65535) or `raw()`.
- A source its codec reports as 16-bit RGBA keeps its depth at decode, unless it carries a non-default EXIF orientation, in which case it is narrowed to 8 bits and flagged
  `depth_reduced` (`image-js/src/ingest.rs:665-677`). A 16-bit PSD is narrowed to 8 bits by the PSD composite decode and flagged the same way (`image-psd/src/composite.rs:123`).
  A source that reports a depth other than 8 bits in any other channel layout is refused with an `Unsupported` error (`image-js/src/ingest.rs:599-620`).

What the code does about the working space at the pinned commit:

- **Transfer.** No kernel and no upload or readback bridge converts between an encoded and
  a linear transfer. The PNG and JPEG adapters report an sRGB transfer; `image-js`
  re-declares the same bytes as linear and uploads them divided by 255 or 65535.
- **Alpha.** Engine-held pixels are straight. The layer fold, fills and strokes bracket
  their dispatches with `cast.premultiply` and `cast.unpremultiply`, skipped for a fully
  opaque window. The single-kernel door (`apply_point_kernel`) multiplies by alpha at
  upload and divides at readback when a pixel is not opaque. Engine A's bridge does neither.

## Evidence

- `image-core/src/format.rs:33-37`, `:122-150` — `PixelFormat`, `GPU_WORKING`, `REF_WORKING`
- `image-codecs/src/raw.rs:103-108`, `image-codecs/src/png/mod.rs:72-80` — the format check
  at the codec boundary; the sRGB transfer a codec reports
- `image-js/src/pixels.rs:15-31`, `:110-145` — `Pixels`, why it has no `Deref`, its views
- `image-pipeline/src/schedule.rs:473-478`, `image-js/src/ingest.rs:44-47`, `:501-510` — the
  decode bridge: no premultiply, no transfer cast; the buffer declared linear
- `image-js/src/lib.rs:2330-2361`, `:2437-2457` — the single-kernel door: alpha association,
  upload and readback at the source's depth
- `image-js/src/layers.rs:57-78`, `:1206-1215` — straight layers, premultiplied accumulator;
  an edit at another depth is refused
- `image-js/src/ingest.rs:596-621`, `:657-680` — which sources keep 16 bits
- `image-js/src/ingest.rs:1123-1129`, `image-js/src/layers.rs:491-498`, `:1517`, `:1585` — the 8-bit lanes

## Alternatives considered

Widening the bare byte buffers in place: rejected for the reason given in Context (commit
`775d7a6`). Widening by `v << 8` alone: rejected because white would no longer be white
(`image-js/src/pixels.rs:130-133`). Narrowing a 16-bit edit onto an 8-bit layer: replaced
by an error, "depths must match", because undo would restore tiles in another geometry
(`image-js/src/layers.rs:1206-1211`). For alpha, commit `78278e5` names two options it did
not take: premultiplying in the decode bridge, or changing the `adjust.*` kernels.

## Consequences

The declared working space and the data differ. `GPU_WORKING` says linear and
premultiplied; kernels receive encoded values and, through Engine A, straight ones. The
`alpha`, `transfer` and `space` fields are set but never read by production code, apart
from whole-descriptor comparison in the codec adapters. The ingest header calls the bridge
temporary (`image-js/src/ingest.rs:44-47`).

The `adjust.*` kernels are written for premultiplied input: each unpremultiplies, does its math and re-premultiplies, except `adjust.exposure`, which scales the premultiplied values directly
(`image-kernels/src/families/adjust.rs:33-39`). The adjustment chain reaches them through Engine A, whose bridge does not premultiply (`image-pipeline/src/schedule.rs:473-478`); the single-kernel
door premultiplies at upload when the image has a pixel that is not opaque (`image-js/src/lib.rs:2350-2360`). For such an image the two paths hand the same kernel different input values.

16-bit depth is carried by decode, by the single-kernel door, and by `LayerStack::edit_active` with its undo. The other paths are written for four bytes per pixel.
`adjust_rgba8` hands the image's raw buffer to a `RawSource` declared as 8-bit RGBA and reads back through an 8-bit sink (`image-js/src/ingest.rs:1123-1129`); `RawSource::new`
returns an error when the buffer length is not width × height × the declared bytes per pixel (`image-codecs/src/raw.rs:63-69`). `layers_open` (`image-js/src/lib.rs:2564`)
hands the raw buffer to `LayerStack::from_image`, which returns an error unless the buffer is four bytes per pixel (`image-js/src/layers.rs:491-498`); `from_image_px`, which
accepts either depth, has no caller outside tests. The multi-layer fold converts each layer with `rgba8_to_f16` and returns `f16_to_rgba8` (`image-js/src/layers.rs:1517`,
`:1585`). PSD save-back is 8-bit RGB (`image-js/src/saveback.rs:48-50`). A 16-bit source also skips the display colour transform (`image-js/src/ingest.rs:657-680`).

## Related

- [ADR 452](452-kernel-abi.md), [ADR 453](453-tolerance-not-golden-bytes.md) — the `rgba16float` output; f16 as the unit of comparison
- [ADR 454](454-two-evaluation-engines.md), [ADR 457](457-colour-management.md) — the layer stack and its journal; the colour transforms at decode
