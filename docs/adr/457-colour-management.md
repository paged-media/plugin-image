# ADR 457 — Colour management: one seam, one engine for display and one for print

- **Status:** Accepted. Recorded retroactively on 2026-10-02 from the code at `f7d21e5`.
- **Scope:** `image-cms`, and its two callers `image-js/src/display.rs` and `image-js/src/cmyk.rs`

## Context

Image files carry ICC profiles: RGB files tagged with a space other than sRGB, and CMYK
JPEGs. The plugin has to bring their pixels into one RGB space before its kernels run, and
what it shows inside a frame should agree with what the host renders around it.

The repo states why each engine was taken. qcms is "the exact build core uses (qcms 0.3,
`cmyk` + `iccv4-enabled`), so "the page" and "the image editor" agree by construction"
(`image-cms/src/qcms_engine.rs:34-36`). The same header (`:36-38`) lists the limits accepted:
"no BPC, 8-bit endpoints, Perceptual/RelCol-centric intents". The moxcms backend's header
says it was chosen for what core's qcms build lacks, and names handling of all four
rendering intents, CMYK ingest and ICC v4 pipelines; it is pure Rust with no C FFI
(`image-cms/src/moxcms_engine.rs:33-39`). lcms2 is a C library; `image-cms/Cargo.toml:31-33`
keeps it out of the shipped build under "the plugin's no-C-in-default-build rule".

## Decision

Colour management sits behind one trait, `CmsEngine`, with two backends: `QcmsEngine` for
the display path and `MoxcmsEngine` for the print lane. lcms2 is a test oracle only.

- `CmsEngine::compile(src, dst, intent, bpc)` returns a `CompiledTransform` that converts
  interleaved RGBA8 in place. `compile_cmyk_to_rgba8` is a second trait method whose default
  returns `CmsError::Unsupported`; only `MoxcmsEngine` overrides it.
- In the shipped path the transforms run on the CPU, once, at decode: "transform at decode,
  keep every kernel untouched" (`image-js/src/display.rs:26`). An 8-bit PNG or JPEG that is
  not CMYK and has an embedded profile goes through `QcmsEngine` to sRGB. A CMYK JPEG with
  an embedded CMYK profile goes through `MoxcmsEngine`. Both calls pass `Intent::Perceptual`
  and `bpc = false`. The destination is an sRGB profile that moxcms synthesises
  (`working_srgb_profile`); no ICC file is bundled.
- A profile problem does not fail the decode. Without a profile, or with one that does not
  compile, RGB pixels pass through unchanged and the image is reported as "sRGB assumed".
  CMYK without a usable profile is converted by the multiplicative ink formula: "there is
  no free, redistributable reference CMYK profile to assume" (`image-js/src/cmyk.rs:52-53`).

## Evidence

- `image-cms/src/lib.rs:186-214` — `CmsEngine`, with the defaulted CMYK method
- `Cargo.toml:56`, `image-cms/Cargo.toml:21`, `:28`, `:35` — qcms 0.3 with `cmyk` and
  `iccv4-enabled`, moxcms 0.8.1, lcms2 under `[dev-dependencies]`
- `core: crates/paged-color/Cargo.toml:24` — core's wasm qcms: same version, same features
- `image-js/src/ingest.rs:699-724` — the decode bridge: the CMYK cast, then the display
  transform for the other layouts
- `image-js/src/display.rs:93-114`, `image-js/src/cmyk.rs:87-102` — the two calls, with fallbacks
- `image-cms/tests/parity.rs:68`, `image-cms/tests/cmyk_lane.rs:84-90` — the bounds against
  lcms2: 3/255 per channel for RGB; 20/255, and 8/255 away from paper white, for CMYK

## Alternatives considered

lcms2 as a shipped engine: `image-cms/Cargo.toml:18-20` says "Unlike core, the plugin does
NOT ship lcms2 natively" and keeps it as an oracle. No other alternative is recorded.

## Consequences

Agreement with the page rests on `Cargo.toml:56` staying equal to core's declaration. The
plugin takes no dependency on core, so the two lines are maintained separately.

"Apply on GPU" is not built. `image-cms/src/lib.rs:39-43` and `:100-102` describe a baked
3D LUT sampled by a `cms.apply` kernel as the production path. `bake_lut` and `GpuLut`
exist, but `registry/kernels.yaml` has no `cms.apply` row and `bake_lut` is called only
from `image-cms/tests/parity.rs`.

Not every source is managed. The PSD composite decode does not surface the file's profile
and reports "sRGB assumed" (`image-js/src/ingest.rs:576-579`). A 16-bit RGBA PNG returns
before the transform and is reported the same way, with or without a profile (`:657-679`).
The CMYK fallback is not recorded separately. `cmyk8_to_rgba8` returns whether the
conversion was colour-managed; the caller discards that value (`image-js/src/ingest.rs:704`)
and sets `DisplayTreatment::Managed` for every CMYK source (`:720-721`), although
`image-js/src/cmyk.rs:57-60` says the fallback is flagged.

The print lane's range is not reachable by a user. Both shipped calls pass Perceptual;
`MoxcmsEngine::compile` (RGB to RGB) has no caller outside tests; no soft proof and no CMYK
export exist. Black-point compensation is recorded but has no effect in either backend
(`image-cms/src/qcms_engine.rs:40-43`, `image-cms/src/moxcms_engine.rs:58-67`).
The stated reason for moxcms includes CMYK and ICC v4, yet qcms is built with its `cmyk`
and `iccv4-enabled` features; `QcmsEngine` itself builds RGBA8-to-RGBA8 transforms only
(`image-cms/src/qcms_engine.rs:85-91`). Two comments are behind the code:
`registry/cms.yaml:86-90` says the CMYK lane is not wired into `image-js`, and
`image-js/src/ingest.rs:43-47` says the decode bridge does no CMS cast.

## Amendment — 2026-10-05: CMYK PSDs, the intent convention, the reported treatment

The decision stands (one seam, moxcms for the print lane, CPU at decode). What changed:

- **CMYK PSDs are converted, through one transform per file.** `image-js/src/cmyk.rs`
  `InkTransform` compiles the file's embedded profile (resource 1039) once; the merged
  composite (`decode_psd`) and every layer plate of a layered open
  (`PsdFile::layer_plates_rgba8_via`) go through it, so the two opens cannot disagree by
  construction. `image-psd` stays CMS-free: it hands back ink (`composite_cmyk8`) and takes the
  conversion as a function.
- **The intent is Perceptual, as a stated approximation of Photoshop's view.** Photoshop's
  default conversion of a CMYK document to RGB is Adobe ACE, relative colorimetric with
  black-point compensation (read from the app by `scripts/photoshop/probes/cmyk-stacks.jsx`).
  moxcms has no BPC; relative colorimetric without it is 28 levels off on Coated FOGRA39 ink
  patches, perceptual is within 4 (mean 0.44) — as close as lcms2's own relcol + BPC. The
  replay `image-conformance/tests/psd_cmyk_photoshop.rs` holds those numbers. A profile whose
  perceptual table does not map black to the PCS black would break the approximation; real
  BPC is the fix when moxcms offers it.
- **The treatment is reported as what it was.** `DisplayTreatment` gained `CmykConverted` and
  `CmykUncalibrated` (wire codes 3 and 4); the CMYK JPEG path now reports the device-formula
  fallback instead of `Managed`, which retires that item of the Consequences above.
- **The working space after a CMYK open is RGB.** Layers are blended in RGB after conversion.
  Photoshop blends a CMYK document's inks, so a non-Normal blend, a file without real merged
  data, or a flatten that strays from the converted composite declines the layered open
  (`image-psd/src/layer_pixels.rs`, `image-js/src/layers.rs` `cmyk_flatten_agrees`).

An RGB PSD's embedded profile is still not applied: doing so at `decode_psd` alone would make
the flattened open and the layered open (whose plates are not transformed) disagree.

## Related

- [ADR 003](https://github.com/paged-media/core/blob/main/docs/adr/003-lcms2-color.md) — core's engines: lcms2 native, qcms on wasm
- [ADR 455](455-pixel-model.md) — the working space the kernels declare
- [ADR 315](https://github.com/paged-media/plugin-sdk/blob/main/docs/adr/315-isolation-contract.md) — the rule under which the plugin takes no dependency on core
- [Colour management at ingest](../design/colour-management-at-ingest.md) — the design note for the display transform
