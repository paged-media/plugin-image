/*
 * This file is part of paged (https://paged.media).
 *
 * paged is free software: you may redistribute it and/or modify it under the
 * terms of the GNU Affero General Public License, version 3, as published by
 * the Free Software Foundation, OR under the Paged Media Enterprise License
 * (PMEL), a commercial license available from And The Next GmbH. Full
 * copyright and license information is available in LICENSE.md, distributed
 * with this source code.
 *
 *  @copyright  Copyright (c) And The Next GmbH
 *  @license    AGPL-3.0-only OR Paged Media Enterprise License (PMEL)
 */

// The RETOUCH tools' glue (wave 6): dodge / burn / sponge, blur /
// sharpen, the spot healing brush and patch — the facade's wire, the
// session's options and its honest decline in Node, and the doors
// executed on the REAL image-js wasm (booted with initSync, no GPU).
//
// What is NOT here: retouched pixels. Every one of these strokes is a
// WGSL dispatch; the pixel proofs live in Rust (image-js/src/stroke.rs
// and retouch.rs, `…__feat__image_editor_*`).

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";
import manifestJson from "@paged-media/image-manifest/manifest.json";

import {
  DEFAULT_TONE,
  STROKE_TOOLS,
  TONE_TOOLS,
  wrapEngine,
  type ImageWasmModule,
} from "../src/engine";
import { ToneSection } from "../src/panels/sections/tone-section";
import { createImageSession } from "../src/session";
import { makeFakeEditor, mapBacking, psdBytes, shellStub, silentConsole } from "./helpers";

const WASM_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "wasm");
const WASM = join(WASM_DIR, "image_js_bg.wasm");

async function boot(): Promise<ImageWasmModule> {
  const glue = (await import(
    /* @vite-ignore */ join(WASM_DIR, "image_js.js")
  )) as unknown as ImageWasmModule;
  glue.initSync({ module: readFileSync(WASM) });
  return glue;
}

async function ingest() {
  const fake = makeFakeEditor();
  const handle = createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
    console: silentConsole,
    storage: mapBacking(),
    shell: shellStub(),
  });
  const session = createImageSession(handle.host);
  expect(await session.importBytes("t.psd", psdBytes())).toBe(true);
  return { handle, session };
}

/** Every element's props in a React tree, without a DOM. */
function propsOf(node: unknown, out: Record<string, unknown>[] = []) {
  if (node === null || node === undefined || typeof node !== "object") return out;
  if (Array.isArray(node)) {
    for (const n of node) propsOf(n, out);
    return out;
  }
  const el = node as { type?: unknown; props?: Record<string, unknown> };
  if (typeof el.type === "function") {
    return propsOf((el.type as (p: unknown) => unknown)(el.props), out);
  }
  if (el.props) {
    out.push(el.props);
    propsOf(el.props.children, out);
  }
  return out;
}

describe("dodge / burn / sponge", () => {
  it("the three are stroke tools that take tone options", () => {
    for (const t of ["dodge", "burn", "sponge"] as const) {
      expect(STROKE_TOOLS).toContain(t);
      expect(TONE_TOOLS).toContain(t);
    }
    expect(TONE_TOOLS).not.toContain("brush");
    expect(DEFAULT_TONE).toEqual({ range: "midtones", exposure: 0.5, saturate: false });
  });

  it("the facade passes the options through to brush_stroke_set_tone", () => {
    const calls: unknown[][] = [];
    const engine = wrapEngine({
      brush_stroke_set_tone: (...a: unknown[]) => calls.push(a),
    } as unknown as ImageWasmModule);
    engine.brushSetTone({ range: "highlights", exposure: 0.25, saturate: true });
    expect(calls).toEqual([["highlights", 0.25, true]]);
  });

  it("the real wasm refuses tone options with no stroke and names a bad range", async () => {
    if (!existsSync(WASM)) return;
    const wasm = await boot();
    expect(() => wasm.brush_stroke_set_tone("midtones", 0.5, false)).toThrow(
      /no stroke in progress/,
    );
    expect(() => wasm.brush_stroke_set_tone("darks", 0.5, false)).toThrow(/unknown tonal range/);
  });

  it("the session keeps the options (exposure clamped) and the strokes decline without a GPU", async () => {
    const { handle, session } = await ingest();
    session.setTone({ range: "shadows", exposure: 3 });
    expect(session.state().tone).toEqual({ range: "shadows", exposure: 1, saturate: false });
    for (const t of ["dodge", "burn", "sponge"] as const) {
      expect(await session.brushBegin(t)).toBe(false);
      expect(session.state().status).toMatch(/GPU-only/);
    }
    session.dispose();
    handle.dispose();
  });

  it("the panel section edits range, exposure and the sponge mode", () => {
    const patches: unknown[] = [];
    const all = propsOf(
      ToneSection({ tone: DEFAULT_TONE, disabled: false, onChange: (p) => patches.push(p) }),
    );
    const find = (k: string) => all.find((p) => k in p) as Record<string, unknown>;
    (find("data-image-tone-range").onChange as (e: unknown) => void)({
      target: { value: "highlights" },
    });
    (find("data-image-tone-exposure").onChange as (e: unknown) => void)({
      target: { value: "0.3" },
    });
    (find("data-image-tone-sponge").onChange as (e: unknown) => void)({
      target: { value: "saturate" },
    });
    expect(patches).toEqual([{ range: "highlights" }, { exposure: 0.3 }, { saturate: true }]);
  });

  it("the tools are in the manifest", () => {
    const tools = (manifestJson as PluginManifest).contributes?.tools ?? [];
    for (const t of ["dodge", "burn", "sponge"]) {
      expect(tools).toContain(`media.paged.image.tool.${t}`);
    }
  });
});

describe("blur / sharpen brushes", () => {
  it("are stroke tools without tone options, registered in the manifest", () => {
    const tools = (manifestJson as PluginManifest).contributes?.tools ?? [];
    for (const t of ["blur", "sharpen"] as const) {
      expect(STROKE_TOOLS).toContain(t);
      expect(TONE_TOOLS).not.toContain(t);
      expect(tools).toContain(`media.paged.image.tool.${t}`);
    }
  });

  it("the real wasm knows the tool names and still declines without a GPU", async () => {
    if (!existsSync(WASM)) return;
    const wasm = await boot();
    const img = wasm.ingest_rgba8(4, 4, new Uint8Array(64).fill(200));
    const p = [8, 0.5, 1, 1, 0.25, "normal", new Float32Array([0, 0, 0, 1]), "none"] as const;
    for (const t of ["blur", "sharpen"]) {
      expect(() => wasm.brush_stroke_begin(img.handle, t, ...p)).toThrow(/GPU-only/);
    }
    expect(() => wasm.brush_stroke_begin(img.handle, "smudge", ...p)).toThrow(/unknown paint tool/);
  });

  it("the session declines both without a GPU", async () => {
    const { handle, session } = await ingest();
    for (const t of ["blur", "sharpen"] as const) {
      expect(await session.brushBegin(t)).toBe(false);
      expect(session.state().status).toMatch(/GPU-only/);
    }
    session.dispose();
    handle.dispose();
  });
});

describe("spot healing brush", () => {
  it("is a stroke tool that needs no clone source", async () => {
    const { SAMPLING_TOOLS } = await import("../src/engine");
    expect(STROKE_TOOLS).toContain("spot-heal");
    expect(SAMPLING_TOOLS).not.toContain("spot-heal");
    const tools = (manifestJson as PluginManifest).contributes?.tools ?? [];
    expect(tools).toContain("media.paged.image.tool.spotHeal");
  });

  it("the real wasm knows the tool name and declines without a GPU", async () => {
    if (!existsSync(WASM)) return;
    const wasm = await boot();
    const img = wasm.ingest_rgba8(4, 4, new Uint8Array(64).fill(200));
    expect(() =>
      wasm.brush_stroke_begin(
        img.handle,
        "spot-heal",
        8,
        0.5,
        1,
        1,
        0.25,
        "normal",
        new Float32Array([0, 0, 0, 1]),
        "none",
      ),
    ).toThrow(/GPU-only/);
  });

  it("the session goes straight to the GPU check (no Alt-click demanded)", async () => {
    const { handle, session } = await ingest();
    expect(await session.brushBegin("spot-heal")).toBe(false);
    expect(session.state().status).toMatch(/GPU-only/);
    expect(session.state().status).not.toMatch(/clone source/);
    session.dispose();
    handle.dispose();
  });
});
