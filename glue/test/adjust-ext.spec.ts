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
 * paged is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the licenses for details.
 *
 *  @copyright  Copyright (c) And The Next GmbH
 *  @license    AGPL-3.0-only OR Paged Media Enterprise License (PMEL)
 */

// The EXTENDED adjust surface + the SAVE-BACK lane, over the REAL engine
// wasm where the lane is CPU (encode / PSD write) and over pure TS where
// it is packing. Node has no navigator.gpu, so anything needing a GPU
// dispatch (the extended chain itself, the generator fills) is covered
// natively in image-js/tests/ingest.rs — here we pin the wire, the
// gating, and the file lanes.

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";

import manifestJson from "@paged-media/image-manifest/manifest.json";

import {
  ADJUST_EXT_LEN,
  DEFAULT_BW_WEIGHTS,
  freshIdentityParams,
  isIdentity,
  packAdjustExt,
} from "../src/engine";
import { createImageSession } from "../src/session";
import {
  makeFakeEditor,
  mapBacking,
  psdBytes,
  shellStub,
  silentConsole,
} from "./helpers";

function makeHost(fake: ReturnType<typeof makeFakeEditor>) {
  return createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
    console: silentConsole,
    storage: mapBacking(),
    shell: shellStub(),
  });
}

describe("the extended adjust parameter wire", () => {
  it("packs the identity params into the neutral block", () => {
    const e = packAdjustExt(freshIdentityParams());
    expect(e).toHaveLength(ADJUST_EXT_LEN);
    expect(e[0]).toBe(0); // vibrance
    expect(Array.from(e.slice(1, 10))).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(e[10]).toBe(0); // black & white DISABLED
    expect(e[17]).toBe(0); // posterize off
    expect(e[19]).toBe(0); // threshold off
    expect(e[21]).toBe(0); // photo filter density 0 = off
    // Channel mixer identity matrix (rows at 26 / 30 / 34).
    expect(Array.from(e.slice(26, 38))).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
    // Per-channel levels identity {0, 1, 1} × 3 (at 38).
    expect(Array.from(e.slice(38, 47))).toEqual([0, 1, 1, 0, 1, 1, 0, 1, 1]);
  });

  it("packs every extended stage at its documented index", () => {
    const p = freshIdentityParams();
    p.vibrance = 0.25;
    p.colorBalance = {
      shadows: [1, 2, 3],
      midtones: [4, 5, 6],
      highlights: [7, 8, 9],
    };
    p.blackWhite = { enabled: true, weights: [...DEFAULT_BW_WEIGHTS] };
    p.posterizeLevels = 8;
    p.threshold = 0.4;
    p.photoFilter = { color: [0.1, 0.2, 0.3], density: 0.5, preserveLuminosity: false };
    p.channelMixer = { r: [1, 2, 3, 4], g: [5, 6, 7, 8], b: [9, 10, 11, 12] };
    p.levelsRgb = {
      r: { inBlack: 0.1, inWhite: 0.9, gamma: 1.5 },
      g: { inBlack: 0.2, inWhite: 0.8, gamma: 1.4 },
      b: { inBlack: 0.3, inWhite: 0.7, gamma: 1.3 },
    };
    const e = packAdjustExt(p);
    expect(e[0]).toBeCloseTo(0.25);
    expect(Array.from(e.slice(1, 10))).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(e[10]).toBe(1);
    expect(Array.from(e.slice(11, 17)).map((n) => Math.round(n * 100) / 100)).toEqual([
      ...DEFAULT_BW_WEIGHTS,
    ]);
    expect(e[17]).toBe(1);
    expect(e[18]).toBe(8);
    expect(e[19]).toBe(1);
    expect(e[20]).toBeCloseTo(0.4);
    expect(e[21]).toBeCloseTo(0.5);
    expect(e[25]).toBe(0); // preserve luminosity OFF
    expect(Array.from(e.slice(26, 38))).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    expect(Array.from(e.slice(38, 47)).map((n) => Math.round(n * 10) / 10)).toEqual([
      0.1, 0.9, 1.5, 0.2, 0.8, 1.4, 0.3, 0.7, 1.3,
    ]);
  });

  it("treats a GATED-OFF stage as identity whatever its other fields hold", () => {
    // The semantic-identity rule (mirrored by the Rust
    // AdjustParams::is_identity): a photo filter at density 0 and a
    // DISABLED black & white contribute nothing, so the chain is still a
    // no-op — the panel must not force a GPU dispatch for them.
    const p = freshIdentityParams();
    p.photoFilter = { color: [0, 1, 0], density: 0, preserveLuminosity: false };
    p.blackWhite = { enabled: false, weights: [9, 9, 9, 9, 9, 9] };
    expect(isIdentity(p)).toBe(true);

    p.photoFilter.density = 0.2;
    expect(isIdentity(p)).toBe(false);
  });

  it("sees every extended stage as non-identity once it is on", () => {
    const cases: Array<(p: ReturnType<typeof freshIdentityParams>) => void> = [
      (p) => (p.vibrance = 0.1),
      (p) => (p.colorBalance.midtones[1] = 0.1),
      (p) => (p.photoFilter.density = 0.1),
      (p) => (p.channelMixer.r[1] = 0.1),
      (p) => (p.levelsRgb.g.gamma = 1.2),
      (p) => (p.blackWhite.enabled = true),
      (p) => (p.posterizeLevels = 4),
      (p) => (p.threshold = 0.5),
    ];
    for (const mutate of cases) {
      const p = freshIdentityParams();
      mutate(p);
      expect(isIdentity(p)).toBe(false);
    }
  });

  it("freshIdentityParams deep-clones (a mutation never poisons the constant)", () => {
    const a = freshIdentityParams();
    const b = freshIdentityParams();
    a.levelsRgb.r.gamma = 3;
    a.colorBalance.shadows[0] = 3;
    a.channelMixer.b[3] = 3;
    a.blackWhite.weights[0] = 3;
    a.photoFilter.color[0] = 3;
    expect(isIdentity(b)).toBe(true);
    expect(b.levelsRgb.r.gamma).toBe(1);
    expect(b.colorBalance.shadows[0]).toBe(0);
  });
});

describe("the save-back lane (real engine wasm)", () => {
  it("re-encodes a PNG source's adjusted pixels through the PNG lane", async () => {
    // The fixture IS the encoder's own output — which also proves the
    // encode → decode round trip the save-back depends on.
    const { bootEngine } = await import("../src/engine");
    const engine = await bootEngine();
    const png = engine.encode(
      Uint8Array.from([255, 0, 0, 255, 0, 0, 255, 255]),
      2,
      1,
      "png",
    );
    expect(Array.from(png.slice(0, 4))).toEqual([0x89, 0x50, 0x4e, 0x47]);

    const fake = makeFakeEditor();
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    expect(await session.importBytes("shot.png", png)).toBe(true);
    // Identity params ⇒ no GPU needed; the encode itself is CPU.
    const back = await session.applyToFile();
    expect(back).not.toBeNull();
    expect(back!.fileName).toBe("shot.png");
    expect(back!.mimeType).toBe("image/png");
    expect(Array.from(back!.bytes.slice(0, 4))).toEqual([0x89, 0x50, 0x4e, 0x47]);
    expect(session.state().saveBack?.fileName).toBe("shot.png");

    session.dispose();
    handle.dispose();
  });

  it("writes a PSD source's adjusted composite back and says what it did", async () => {
    const fake = makeFakeEditor();
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    expect(await session.importBytes("art.psd", psdBytes())).toBe(true);
    expect(session.state().psd).not.toBeNull();

    const back = await session.applyToFile();
    expect(back).not.toBeNull();
    expect(back!.fileName).toBe("art.psd");
    expect(back!.mimeType).toBe("image/vnd.adobe.photoshop");
    expect(Array.from(back!.bytes.slice(0, 4))).toEqual([0x38, 0x42, 0x50, 0x53]);
    // The layerless fixture gains the synthesized single layer — the
    // note must SAY the structure changed, never stay silent.
    expect(back!.note).toContain("PSD save-back");
    expect(back!.note.toLowerCase()).toContain("flatten");
    // The written file re-ingests to the same pixels (the composite is
    // the adjusted result, and identity params ⇒ the decode verbatim).
    const round = createImageSession(handle.host);
    expect(await round.importBytes("again.psd", back!.bytes)).toBe(true);
    expect(round.state().source).toMatchObject({ width: 2, height: 1 });
    round.dispose();

    session.dispose();
    handle.dispose();
  });

  it("keeps a plain PSD export BYTE-IDENTICAL at identity (§10.4)", async () => {
    const fake = makeFakeEditor();
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    const original = psdBytes();
    expect(await session.importBytes("keep.psd", original)).toBe(true);
    expect(isIdentity(session.state().params)).toBe(true);

    const exported = await session.psdExportBytes();
    expect(exported).not.toBeNull();
    expect(Array.from(exported!.bytes)).toEqual(Array.from(original));

    session.dispose();
    handle.dispose();
  });

  it("registers the three save-back exporters", async () => {
    const { imageBundle } = await import("../src/index");
    const { loadBundle } = await import("@paged-media/plugin-sdk");
    const fake = makeFakeEditor();
    const loaded = loadBundle(() => fake.editor, imageBundle, {
      console: silentConsole,
      storage: mapBacking(),
      shell: shellStub(),
    });
    expect(fake.exporters.ids()).toEqual([
      "media.paged.image.exporter.psd",
      "media.paged.image.exporter.png",
      "media.paged.image.exporter.jpeg",
    ]);
    loaded.dispose();
    expect(fake.exporters.ids()).toHaveLength(0);
  });
});
