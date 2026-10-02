# The 16-bit stack — why it is not built, and the shape that makes it safe

*Status note (2026-10-02): the status line describes the engine's Rust API, which `image-js/tests/ingest_16bit.rs` exercises. Through the shipped wasm surface 16 bits are held only after a main-thread decode and kept only by the one-shot filter door; `layers_open`, the panel's adjustment chain, the in-frame image, tiles and save-back are 8-bit, and a 16-bit greyscale PNG is refused. See `status.md`.*

**Status:** **DONE 2026-08-09 — all five steps.** A 16-bit PNG opens at full depth and keeps every bit through an adjustment, a layer edit, and an undo. The only remaining narrowing is 16-bit Gray/GrayA, which has no widening arm and still says so via `depth_reduced`.
**Size:** smaller than this document first claimed — see the correction below.
**Blocked on:** nothing external. This is a decision about sequencing and risk.

Sections not relevant outside the original planning context have been removed.

## What already ships, so the gap is smaller than the row implies

*Status note (2026-10-02): this section and the next were written before the work was done; the layer store listed below as unbuilt is built, and `Layer.rgba` and `DecodedImage.rgba` are now the `Pixels` type in `image-js/src/pixels.rs` (the steps further down record it). See [ADR 455](../adr/455-pixel-model.md).*

The internal capability catalogue's row read as though 16-bit were one undifferentiated
wall. It is four things, and three are done:

| Rung | State |
|---|---|
| The kernel/tile ABI | **Done, always was.** f16 storage, f32 compute. The *engine* was never 8-bit-bound. |
| 16-bit PSD ingest | **Done.** Opens, narrowed by the high byte, reduction reported. |
| 16-bit PNG ingest | **Done 2026-08-09.** Used to error `Unsupported` — the file would not load. The blocker ("byte order unverified, no fixture") was false: PNG endianness is normative (§7.1) and `zune` returns assembled `u16`s anyway. `image-conformance/tests/codec_png_16bit.rs` now *measures* the order. |
| The adjustment-layer fold | **Done.** Stays in f16 rather than round-tripping through 8 per layer. Measured: >120 of 256 distinct levels survive a compress/expand pair, versus 26 before. |
| **The layer STORE** | **Unbuilt. This document.** |

Two claims in the old row were also simply wrong and are corrected
there: **JPEG was never a blocker** (the format is 8-bit — nothing to
narrow), and **it is not "a change to a core type"** — `SampleDepth::U16`
has existed in `image-core` all along. It is `image-js` work.

## Why it is dangerous to do blind, which is the real point

`Layer.rgba` is `Arc<[u8]>`. So is `DecodedImage.rgba`. Bare bytes.

That means **a wrong bytes-per-pixel assumption is not a compile error.**
Change the store to 16-bit and every one of the 161 sites that does
`i * 4`, `chunks_exact(4)`, `len() / 4`, or `w * h * 4` keeps building
and starts reading the wrong pixel. The failure is a silently corrupted
image, not a red build.

| File | Sites touching pixel bytes |
|---|---|
| `image-js/src/ingest.rs` | 63 |
| `image-js/src/layers.rs` | 52 |
| `image-js/src/saveback.rs` | 27 |
| `image-js/src/fill.rs` | 15 |
| `image-js/src/channels.rs` | 4 |

Plus the COW journal (tile-granular, sized in bytes), the brush/dab
engine, the PSD save-back's `replace_channel_pixels`, and every
`f16_to_rgba8` readback.

**This is the single worst shape a refactor can have**: large, mechanical,
and invisible to the compiler.

## The shape that makes it safe

Do not change `Arc<[u8]>` to a wider `Arc<[u8]>`. Introduce a type the
compiler can reason about, land it at 8-bit with **zero behaviour
change**, and only then switch the depth on.

**Step 1 — a newtype, still 8-bit. DONE.** `image-js/src/pixels.rs`.
Replace the bare buffer with:

```rust
pub struct Pixels {
    bytes: Arc<[u8]>,
    depth: SampleDepth,   // U8 today
}
impl Pixels {
    pub fn depth(&self) -> SampleDepth;
    pub fn bytes_per_pixel(&self) -> usize;   // 4 or 8 — never a literal
    pub fn as_rgba8(&self) -> Cow<'_, [u8]>;  // narrowing view, explicit
    pub fn sample(&self, i: usize, c: usize) -> u16;
}
```

Every site now **fails to compile** until it says which view it wants,
converting the invisible refactor into a mechanical one. Landed with the
suite green — 1026 Rust (the 1021 baseline plus 5 new `Pixels` tests),
210 vitest — and behaviour byte-identical at 8-bit.

**The touch-point estimate in this document was WRONG, and by a lot.**
It said 161, from a grep that counted comments, string literals and
`RGBA8` mentions. The compiler found **43**: 12 for `DecodedImage`, 11
for `Layer`, 20 more in `mod wasm`. The lesson is a recurring one
— a number asserted from a grep is not a measurement,
and the cheap way to find out was to make the change and let the
compiler answer.

**A gate lesson worth more than the step.** `cargo check --workspace
--all-targets` on the HOST reported zero errors while those 20 wasm-only
sites were still broken, because `mod wasm` is
`#[cfg(target_arch = "wasm32")]` and the host build does not compile the
code that ships. Same shape as the missing `#[wasm_bindgen]` bug found
the same day. **Any "green" that does not include a wasm32 build is a
claim about a different program.**

Two mistakes during the migration — a regex that over-applied to
`SmartSource.rgba` (still a plain `Arc<[u8]>`) and an unbalanced paren
in a test — were both caught by the compiler, which is exactly the
failure mode this step exists to convert.

**Step 2 — the upload bridge. DONE.** `apply_point_kernel` reads through `sample16` and normalises by 65535, so it is depth-agnostic. 8-bit is byte-identical to the old `/ 255.0` path because `(v << 8) | v` maps 255 to exactly 65535. **A trap here:** after `px_f16` changed to take a PIXEL index, both callers still passed BYTE offsets — both `usize`, so it compiled clean. The newtype cannot see inside a closure's own indexing; that one was caught by reading the call sites.

**Step 3 — the readback. DONE.** `f16_to_rgba16` and an unpremultiplying twin. This was the entire loss: the GPU always computed in f32 and stored `rgba16float`, and the only quantisation was the trip home — once per operation, so a chain of N adjustments quantised N times.

**Step 4 — the layered path. DONE.** `edit_active` takes `Pixels` and the undo restores through `from_raw(bytes, depth)`.

**The journal needed NO change** — this document predicted `image-graph` work and there was none. It stores opaque tile byte handles, and `FlatImage::new(w, h, bpp, …)` always took bytes-per-pixel as a *parameter* rather than baking in `4`. The only real bug to avoid was narrowing on the way in or out. **Second time this plan over-estimated** (see the touch-point count above); both times the cheap check was to look rather than reason.

**Undo was the sharp edge.** Restoring through an 8-bit view yields `0x1200` where `0x1234` went in — high byte kept, low byte silently gone — and it surfaces only AFTER an undo, about the worst place to hide a precision bug. Mutation-checked: putting the narrowing back fails the test.

**A depth mismatch is refused, not narrowed.** The journal snapshots tiles in one geometry and restores them in another, so a 16-bit edit onto an 8-bit layer corrupts an *undo* rather than merely degrading the edit.

**Step 5 — ingest stops narrowing. DONE for PNG.** 16-bit RGBA rides through at full depth and `depth_reduced` now reads `false`, because nothing is reduced. Other 16-bit layouts (Gray/GrayA) still narrow — they would need their own widening arms — and still say so. PSD save-back is unchanged.

**Step 6 — memory.** Every layer doubles. `image-js/src/layers.rs` already documents
canvas-extent layers as a deliberate simplification; at 16-bit the
memory argument gets sharper and may force per-layer bounds, which is a
separate decision.

## Verification this needs, beyond the suite

- **A level-count test**, not an eyeball: banding *is* loss of distinct
  levels. The adjustment fold's existing measurement (compress/expand,
  count survivors) is the template.
- **A round-trip at 16-bit**: a 16-bit PSD in, no edit, 16-bit out,
  byte-identical. The preservation invariant already demands this at
  8-bit; it must not weaken.
- **The GPU behavioural lane** (`image-conformance/tests/gpu_module_kernels.rs`) re-run — its
  stimulus is dyadic and lossless in f16, so it should be indifferent to
  the store's depth. If it is not, something is wrong.

## Postscript — what this document got wrong

Both of its size estimates were too high, and in the same way: asserted
from reading rather than measured by trying. 161 touch points were
actually 43. `image-graph` work turned out to be zero. The design
argument (make the compiler find the sites) was right and earned itself
several times over; the *cost* argument was not.

The counter-lesson is not "estimate less carefully" — it is that a
newtype migration is cheap to ATTEMPT and expensive to guess about,
because the compiler answers in one build what reading answers in an
afternoon.
