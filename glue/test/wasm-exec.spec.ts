import { describe, expect, it } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { ImageWasmModule } from "../src/engine";

/**
 * The EXECUTION half of the wasm gate. `wasm-surface.spec.ts` proves the
 * TS interface and the artifact's export list agree — a static text
 * check that would have caught the declared-but-absent incident, but
 * still never RUNS a single export. Every other spec in this directory
 * mocks the engine, so until this file the 128 registry kernels had
 * zero executions through the real module anywhere in CI: a build that
 * exported every name but computed garbage would have shipped green.
 *
 * This spec boots the COMMITTED artifact (glue/wasm — the copy the
 * published package ships; manifest/wasm holds the same bytes but is
 * gitignored build output) via initSync in Node, the same glue path
 * engine.ts uses, and drives a representative slice of the surface on a
 * synthetic 16×16 RGBA buffer, asserting real outputs.
 *
 * What Node CAN prove: the CPU lanes — ingest, histogram/stats
 * reductions, auto-enhance orchestration, the curve LUT, crop + tile
 * windowing, the PNG codec, and the whole selection family (marquee
 * rasterization, magic-wand flood, the Gaussian feather on the mask,
 * invert). What it CANNOT: pixel kernels are GPU-only by constitution
 * (spec §6 — no CPU path ships) and Node has no WebGPU, so for those
 * this spec pins the OTHER contract: they reject with the engine's own
 * honest no-GPU error. That still executes the real dispatch path — a
 * declared-but-absent export fails those assertions with "not a
 * function" instead of the engine's message, so the incident class is
 * caught at runtime too, not only by the d.ts diff.
 */
const HERE = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = join(HERE, "..", "wasm");
const WASM = join(WASM_DIR, "image_js_bg.wasm");
const built = existsSync(WASM);

// The CI half of the dual gate (the sheets engine-real pattern):
// REQUIRE_REAL_WASM=1 turns "artifact missing" from a skip into a hard
// failure. The artifact is COMMITTED, so CI needs no build step — this
// guard exists so a future .gitignore/LFS/checkout regression cannot
// silently drop the only real-execution lane back to zero.
if (process.env.REQUIRE_REAL_WASM === "1" && !built) {
  describe("real wasm execution — REQUIRED", () => {
    it("FAILS: REQUIRE_REAL_WASM=1 but the wasm artifact is missing", () => {
      throw new Error(
        `REQUIRE_REAL_WASM=1 but ${WASM} is missing — the committed ` +
          "artifact should be present in any checkout; rebuild with " +
          "scripts/build-wasm.sh (skipping is not allowed here)",
      );
    });
  });
}

/** Boot the real module (initSync is idempotent — repeat calls return
 *  the already-instantiated module, so per-test boots share one realm).
 *  Typed as `ImageWasmModule` deliberately: this drives the artifact
 *  through the SAME hand-written interface engine.ts trusts, so a call
 *  that compiles here and fails at runtime is exactly the drift the
 *  surface spec describes. */
async function boot(): Promise<ImageWasmModule> {
  const glue = (await import(
    /* @vite-ignore */ join(WASM_DIR, "image_js.js")
  )) as unknown as ImageWasmModule;
  glue.initSync({ module: readFileSync(WASM) });
  return glue;
}

const W = 16;
const H = 16;

/** Left half opaque black, right half opaque white — every reduction
 *  over it (histogram bins, channel min/max/mean, wand flood extents)
 *  has a hand-computable expectation. */
function halfAndHalf(): Uint8Array {
  const px = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const v = x < W / 2 ? 0 : 255;
      px[i] = v;
      px[i + 1] = v;
      px[i + 2] = v;
      px[i + 3] = 255;
    }
  }
  return px;
}

/** Await a call that may reject OR throw synchronously (the wasm door
 *  does both, depending on where the guard fires) and hand back the
 *  error text; null means it wrongly succeeded. */
async function refusal(fn: () => unknown): Promise<string | null> {
  try {
    await fn();
    return null;
  } catch (e) {
    return String(e);
  }
}

describe.skipIf(!built)("real wasm execution (committed artifact)", () => {
  it("boots and reports the surface truth (abi, kernel count, gpu state)", async () => {
    const glue = await boot();
    expect(glue.abi_version()).toBe(1);
    // The registry-generated dispatch table, read from the RUNNING
    // module — the "128 kernels" claim as a runtime readout rather than
    // prose. A floor, not an exact pin, so adding a kernel does not
    // touch this file.
    expect(glue.kernel_count()).toBeGreaterThanOrEqual(128);
    expect(glue.gpu_ready()).toBe(false);
  });

  it("ingests a synthetic buffer and reduces it correctly (histogram + stats)", async () => {
    const glue = await boot();
    const h = glue.ingest_rgba8(W, H, halfAndHalf());
    expect(h.width).toBe(W);
    expect(h.height).toBe(H);

    // 1024 u32 = r/g/b/luma × 256 bins; half the 256 pixels sit in bin
    // 0 and half in bin 255, every other bin empty. Only a real
    // reduction over the real buffer produces these numbers.
    const hist = glue.image_histogram(h.handle);
    expect(hist.length).toBe(1024);
    expect(hist[0]).toBe(128); // red bin 0
    expect(hist[255]).toBe(128); // red bin 255
    const lumaTotal = hist.slice(768).reduce((a: number, b: number) => a + b, 0);
    expect(lumaTotal).toBe(W * H);

    // Two-run equality: the reduction is deterministic.
    expect(Array.from(glue.image_histogram(h.handle))).toEqual(Array.from(hist));

    // channel_stats agrees with the histogram by construction — and
    // with our arithmetic: mean of {0,255} at 50/50 is 127.5.
    const stats = JSON.parse(glue.image_channel_stats(h.handle)) as Array<{
      name: string;
      min: number;
      max: number;
      mean: number;
    }>;
    const red = stats.find((s) => s.name === "red");
    expect(red).toMatchObject({ min: 0, max: 255 });
    expect(red?.mean).toBeCloseTo(127.5, 2);
    const alpha = stats.find((s) => s.name === "alpha");
    expect(alpha).toMatchObject({ min: 255, max: 255 });

    // A length that cannot be width*height*4 is a CLEAN error with the
    // real validator's message — a mock never had this check.
    expect(() => glue.ingest_rgba8(4, 4, new Uint8Array(7))).toThrow(
      /7 bytes for 4x4/,
    );
    glue.free_image(h.handle);
  });

  it("computes the tone LUT (point-op math on the curve points)", async () => {
    const glue = await boot();
    // The inversion curve (0→1, 1→0): a strictly descending 256-byte LUT.
    const inv = glue.curve_lut(new Float32Array([0, 1, 1, 0]));
    expect(inv.length).toBe(256);
    expect(inv[0]).toBe(255);
    expect(inv[255]).toBe(0);
    // The identity curve maps every level to itself at the endpoints
    // and stays monotonic between them.
    const id = glue.curve_lut(new Float32Array([0, 0, 1, 1]));
    expect(id[0]).toBe(0);
    expect(id[255]).toBe(255);
    for (let i = 1; i < 256; i++) expect(id[i]).toBeGreaterThanOrEqual(id[i - 1]);
  });

  it("crops the exact pixel window and reads it back through the tile door", async () => {
    const glue = await boot();
    const src = glue.ingest_rgba8(W, H, halfAndHalf());

    // Cut 4×4 out of the white half: every byte of the read-back is 255.
    const white = glue.crop_image(src.handle, 8, 0, 4, 4);
    expect([white.width, white.height]).toEqual([4, 4]);
    const whitePx = glue.image_tile_rgba8(white.handle, 0, 0, 4, 4);
    expect(whitePx.length).toBe(4 * 4 * 4);
    expect(Array.from(whitePx).every((b: number) => b === 255)).toBe(true);

    // Cut from the black half: RGB 0, alpha 255 — the crop took the
    // OTHER pixels, not merely different dimensions.
    const black = glue.crop_image(src.handle, 0, 0, 4, 4);
    const blackPx = glue.image_tile_rgba8(black.handle, 0, 0, 4, 4);
    for (let i = 0; i < blackPx.length; i += 4) {
      expect([blackPx[i], blackPx[i + 3]]).toEqual([0, 255]);
    }

    // A window fully outside the extent is a transparent miss (empty),
    // not an error and not torn bytes.
    expect(glue.image_tile_rgba8(white.handle, 100, 100, 4, 4).length).toBe(0);

    // The source survives its crops (the caller frees it — doc contract).
    expect(glue.image_tile_rgba8(src.handle, 0, 0, W, H).length).toBe(W * H * 4);
    glue.free_image(white.handle);
    glue.free_image(black.handle);
    glue.free_image(src.handle);
  });

  it("round-trips pixels through the real PNG codec, deterministically", async () => {
    const glue = await boot();
    const px = halfAndHalf();
    const png = glue.encode_image(px, W, H, "png");
    // A real PNG container, not a stub: the 8-byte signature's head.
    expect(Array.from(png.slice(0, 4))).toEqual([137, 80, 78, 71]);

    const dec = glue.decode_image(png);
    expect([dec.width, dec.height]).toEqual([W, H]);
    // PNG is lossless: decode returns the ingested bytes exactly.
    const back = glue.image_tile_rgba8(dec.handle, 0, 0, W, H);
    expect(Array.from(back)).toEqual(Array.from(px));

    // Entropy coding is deterministic: two encodes, identical bytes.
    expect(Array.from(glue.encode_image(px, W, H, "png"))).toEqual(
      Array.from(png),
    );
    glue.free_image(dec.handle);
  });

  it("drives the selection family: marquee, wand flood, invert, feather", async () => {
    const glue = await boot();
    const h = glue.ingest_rgba8(W, H, halfAndHalf());
    glue.selection_bind(h.handle);

    // Marquee rect (mode 0 = replace, the engine.ts wire code): the
    // rasterized coverage bounds are exactly the rect, fraction 64/256.
    glue.selection_set_rect(4, 4, 8, 8, 0);
    expect(Array.from(glue.selection_bounds())).toEqual([4, 4, 8, 8]);
    let s = glue.selection_stats();
    expect(s[0]).toBe(1); // has explicit selection
    expect(s[5]).toBeCloseTo(0.25, 3);

    // Magic wand seeded in the white half, tolerance 0, contiguous: the
    // flood finds exactly the right half — a real BFS over the real
    // pixels (a mock cannot know where the white pixels are).
    glue.selection_magic_wand(12, 8, 0, true, 0);
    expect(Array.from(glue.selection_bounds())).toEqual([8, 0, 8, 16]);
    s = glue.selection_stats();
    expect(s[5]).toBeCloseTo(0.5, 3);

    // Invert: the complement is the LEFT half, same area.
    glue.selection_invert();
    expect(Array.from(glue.selection_bounds())).toEqual([0, 0, 8, 16]);
    expect(glue.selection_stats()[5]).toBeCloseTo(0.5, 3);

    // Feather: a real Gaussian on the u8 coverage mask (CPU by design —
    // mask prep, not image processing). The hard edge must soften into
    // intermediate coverage values a binary mask cannot contain.
    const before = glue.selection_coverage_bytes();
    expect(
      Array.from(before).some((v: number) => v > 0 && v < 255),
    ).toBe(false);
    glue.selection_feather(2.0);
    const after = glue.selection_coverage_bytes();
    expect(after.length).toBe(W * H);
    expect(
      Array.from(after).filter((v: number) => v > 0 && v < 255).length,
    ).toBeGreaterThan(0);
    // …and feathering is deterministic in its readout.
    expect(Array.from(glue.selection_coverage_bytes())).toEqual(
      Array.from(after),
    );

    glue.selection_clear();
    glue.free_image(h.handle);
  });

  it("computes auto-enhance params (the identity guarantee on neutral input)", async () => {
    const glue = await boot();
    // The half-black/half-white buffer already spans the full luma range
    // with a neutral gray-world balance, so the documented guarantee
    // applies: the auto estimate is EXACTLY the identity [0, 1, 0, 0] —
    // never a wrong-looking correction.
    const h = glue.ingest_rgba8(W, H, halfAndHalf());
    expect(Array.from(glue.image_auto_enhance_params(h.handle))).toEqual([
      0, 1, 0, 0,
    ]);
    glue.free_image(h.handle);
  });

  it("returns the decode verbatim through the identity adjust door", async () => {
    const glue = await boot();
    const px = halfAndHalf();
    const h = glue.ingest_rgba8(W, H, px);
    // Identity is (exposure 0, brightness 0, contrast 1, saturation 1)
    // — engine.ts IDENTITY_PARAMS. The Stage-A door executes and hands
    // back the held pixels without a dispatch.
    const out = await glue.adjust_image(h.handle, 0, 0, 1, 1);
    expect(Array.from(out)).toEqual(Array.from(px));
    glue.free_image(h.handle);
  });

  it("refuses GPU-only kernels honestly in a no-WebGPU realm", async () => {
    const glue = await boot();
    const h = glue.ingest_rgba8(W, H, halfAndHalf());

    // Each refusal is the ENGINE's message, produced by the real entry
    // point running to its guard. A declared-but-absent export (the
    // wasm-surface incident) would read "is not a function" instead and
    // fail the match — so this is a runtime re-proof of that gate, one
    // per kernel family:
    expect(await refusal(() => glue.init_gpu())).toMatch(/WebGPU unavailable/);
    // T1 point op (brightness ≠ identity):
    expect(
      await refusal(() => glue.adjust_image(h.handle, 0, 0.5, 1, 1)),
    ).toMatch(/GPU not initialized/);
    // T1 resample:
    expect(
      await refusal(() => glue.resize_image(h.handle, 8, 8, "lanczos3")),
    ).toMatch(/GPU-only/);
    // Stylize — emboss, one of the ten that shipped declared-but-absent:
    expect(
      await refusal(() => glue.apply_emboss(h.handle, 45, 3)),
    ).toMatch(/GPU-only/);

    // The module survives its refusals — the realm is not poisoned.
    expect(glue.gpu_ready()).toBe(false);
    expect(glue.image_histogram(h.handle).length).toBe(1024);
    glue.free_image(h.handle);
  });
});
