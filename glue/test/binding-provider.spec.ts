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

// ADR-023 phase D — the two providers, driven through the REAL shared
// registry the editor injects, not through the provider objects
// directly.
//
// That distinction is the point of this file. Calling
// `provider.readCollection(...)` in a test proves the function returns
// what it returns; going through `createBindingProviderRegistry()`
// proves the thing the host actually does — resolve "who answers this
// collection right now?" across loaded bundles, with the answer gated
// on the edit context being ACTIVE. A provider that works in isolation
// and is never consulted is the failure this seam exists to prevent,
// and only the registry path can see it.

import { describe, expect, it } from "vitest";

import {
  createBindingProviderRegistry,
  createBundleHost,
} from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";

import manifestJson from "@paged-media/image-manifest/manifest.json";

import { activate } from "../src/activate";
import { createImageSession } from "../src/session";
import { makeTextBindingProvider } from "../src/binding-provider/text-provider";
import { makeFakeEditor, mapBacking, psdBytes, shellStub, silentConsole } from "./helpers";

const PLUGIN = "media.paged.image";
const CTX = "rasterImage";

/** A bundle activated against a real registry, with the raster context
 *  reported ACTIVE the way the shell's own onEnter would. */
function boot(opts: { active?: boolean } = {}) {
  const registry = createBindingProviderRegistry();
  const fake = makeFakeEditor();
  const handle = createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
    console: silentConsole,
    storage: mapBacking(),
    shell: shellStub(),
    bindingProviders: registry,
  });
  const bundle = activate(handle.host);
  if (opts.active !== false) {
    registry.setContextActive(PLUGIN, CTX, "u1", true);
    // PRECONDITION, and it is load-bearing rather than decorative.
    //
    // Most assertions below are `resolved: false` / `handled: false` —
    // and those hold just as well when NOTHING IS REGISTERED. Mutation-
    // checked: disabling the registration branch in `activate` left 5
    // of these 10 tests green, i.e. half the file was asserting the
    // absence of a provider while claiming to assert its behaviour.
    //
    // Anchoring here makes every test that boots an active context
    // prove the providers exist BEFORE it interprets a refusal as
    // theirs. A refusal from an absent provider and a refusal from a
    // present one are different facts and must not share a test.
    expect(
      registry.activeProviders(),
      "both providers must be active before a refusal means anything",
    ).toHaveLength(2);
  }
  return { registry, fake, handle, bundle };
}

describe("the ADR-023 providers reach the registry at all", () => {
  it("registers BOTH providers, and only while the context is active", () => {
    const { registry, bundle, handle } = boot({ active: false });

    // Registered, but the context has not been entered — so nothing is
    // consulted. This is the LIFETIME rule: a provider borrows its
    // context's activation and is invisible outside it.
    expect(registry.activeProviders()).toHaveLength(0);

    registry.setContextActive(PLUGIN, CTX, "u1", true);
    const active = registry.activeProviders();
    expect(active).toHaveLength(2);

    // Exiting takes them away again — a stale provider answering for a
    // frame the user has left is the same bug in the other direction.
    registry.setContextActive(PLUGIN, CTX, "u1", false);
    expect(registry.activeProviders()).toHaveLength(0);

    bundle.dispose();
    handle.dispose();
  });

  it("declares the two lanes apart, so one cannot mask the other", () => {
    const { registry, bundle, handle } = boot();
    const provides = registry.activeProviders().map((p) => p.provides);

    const layers = provides.find((p) => p.collections?.includes("layers"));
    expect(layers, "a provider claims the layers collection").toBeDefined();
    expect(layers!.ops).toContain("layerMove");
    expect(layers!.ops).toContain("layerSetVisible");
    // The layers provider claims NO property paths — its whole surface
    // is rows + structural ops. Claiming paths it does not serve would
    // silence the character provider registered beside it.
    expect(layers!.paths ?? []).toHaveLength(0);

    const text = provides.find((p) => p.paths?.includes("characterFontFamily"));
    expect(text, "a provider claims the character paths").toBeDefined();
    expect(text!.collections ?? []).toHaveLength(0);

    bundle.dispose();
    handle.dispose();
  });
});

describe("the layers provider", () => {
  it("DECLINES with no stack open, rather than serving zero rows", async () => {
    // The distinction this asserts is the whole `absent`/`decline`
    // discipline applied to a collection: zero rows would CLAIM the
    // panel and assert "this raster frame has no layers", which is
    // false — there is no open stack at all, and the document's own
    // layers are then the truthful thing to show.
    const { registry, bundle, handle } = boot();
    const r = await registry.readCollection({
      collection: "layers",
      target: { kind: "selection", scope: "element" },
    } as never);
    expect(r.resolved).toBe(false);

    bundle.dispose();
    handle.dispose();
  });

  it("refuses a collection it never declared", async () => {
    const { registry, bundle, handle } = boot();
    const r = await registry.readCollection({
      collection: "swatches",
      target: { kind: "selection", scope: "element" },
    } as never);
    // Undeclared → the registry never even consults the provider, so
    // the host reads core. paged.image has no colour vocabulary to
    // offer the Swatches panel and must not appear to.
    expect(r.resolved).toBe(false);

    bundle.dispose();
    handle.dispose();
  });

  it("does not swallow a structural op it never declared", async () => {
    const { registry, bundle, handle } = boot();
    const r = await registry.applyMutation({
      op: "createSwatch",
      args: { name: "x" },
    } as never);
    // `handled: false` means core gets it. A provider that swallowed
    // every op would silently break the document's own panels while
    // its context happened to be open.
    expect(r.handled).toBe(false);

    bundle.dispose();
    handle.dispose();
  });
});

describe("the character provider", () => {
  it("declares a NON-EMPTY writablePaths — the lane nothing had exercised", () => {
    // paged.sheet declared `writablePaths: []` honestly (its engine has
    // no cell-style write API), so the host's property-WRITE path had
    // shipped without ever running against a real provider. This is the
    // assertion that says paged.image is the one that runs it.
    const { registry, bundle, handle } = boot();
    const text = registry
      .activeProviders()
      .map((p) => p.provides)
      .find((p) => p.paths?.includes("characterFontFamily"))!;

    expect(text.writablePaths).toBeDefined();
    expect(text.writablePaths!.length).toBeGreaterThan(0);
    expect(text.writablePaths).toContain("characterFontFamily");
    // Every writable path must also be a declared path — a write target
    // the provider never claimed to read is not a lane, it is a typo.
    for (const w of text.writablePaths!) expect(text.paths).toContain(w);

    bundle.dispose();
    handle.dispose();
  });

  it("owns the paths it cannot serve, so core never answers over a raster run", () => {
    const { registry, bundle, handle } = boot();
    const text = registry
      .activeProviders()
      .map((p) => p.provides)
      .find((p) => p.paths?.includes("characterFontFamily"))!;

    // Leading and paragraph alignment are meaningless on one rasterized
    // line — but they are DECLARED, so the host asks us and gets a
    // blank rather than asking core and painting the text caret's
    // paragraph over an image frame.
    expect(text.paths).toContain("characterLeading");
    expect(text.paths).toContain("paragraphJustification");
    // …and they are NOT writable, so the control renders read-only
    // instead of offering a commit that lands nowhere.
    expect(text.writablePaths).not.toContain("characterLeading");

    bundle.dispose();
    handle.dispose();
  });

  it("declines the FONT SIZE — the unit mismatch, asserted not assumed", () => {
    // `characterFontSize` binds to a length widget formatting in
    // points; raster type measures in pixels. The conversion IS
    // available (the composite path already computes points-per-pixel
    // from the frame box) — what is unresolved is the PRODUCT question
    // of what that field means. So this is owned-and-absent, a
    // different claim from "not mine".
    //
    // The test exists so that the day someone wires the bridge, it
    // fails and makes them state the decision rather than inherit it.
    const { registry, bundle, handle } = boot();
    const text = registry
      .activeProviders()
      .map((p) => p.provides)
      .find((p) => p.paths?.includes("characterFontFamily"))!;

    expect(text.paths).toContain("characterFontSize");
    expect(text.writablePaths).not.toContain("characterFontSize");

    bundle.dispose();
    handle.dispose();
  });

});

// ── the tool condition, ISOLATED ────────────────────────────────────
//
// These go through the provider directly rather than the registry, and
// that is deliberate. Routed through `activate()` the session has no
// ingested image, so a decline proves nothing about the TOOL — the
// provider would decline for want of a source either way. Mutation-
// checked and caught exactly that: replacing the tool check with
// `() => true` left the registry-level versions of these tests green.
//
// So the source is ingested first and the tool flag is the ONLY thing
// that moves between the two assertions.

describe("the character provider scopes by TOOL, not only by context", () => {
  async function ingestedSession() {
    const fake = makeFakeEditor();
    fake.placed.set("u1", psdBytes());
    fake.geometry.set("u1", {
      id: { kind: "rectangle", id: "u1" } as never,
      pageId: "pg1",
      bounds: [0, 0, 100, 200],
    });
    fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
    const handle = createBundleHost(
      () => fake.editor,
      manifestJson as PluginManifest,
      { console: silentConsole, storage: mapBacking(), shell: shellStub() },
    );
    const session = createImageSession(handle.host);
    expect(await session.ingestSelection(), "the fixture ingested").toBe(true);
    return { session, handle };
  }

  it("answers with the tool in hand and declines without it", async () => {
    const { session, handle } = await ingestedSession();
    let toolActive = false;
    const { provider } = makeTextBindingProvider(session, () =>
      toolActive ? "media.paged.image.tool.type" : null,
    );
    const req = {
      path: "characterFontFamily",
      target: { kind: "selection", scope: "content" },
    } as never;

    // Source present, tool absent → a decline that is ABOUT THE TOOL.
    expect((await provider.readProperty!(req)).kind).toBe("decline");

    toolActive = true;
    const answered = await provider.readProperty!(req);
    expect(answered.kind).toBe("value");
    expect((answered as { value: unknown }).value).toBe(
      session.state().type.family,
    );

    session.dispose();
    handle.dispose();
  });

  it("re-checks the tool on WRITE rather than trusting the read", async () => {
    // A panel can hold a value across a tool change; committing it then
    // would set type settings from a control the user has left.
    const { session, handle } = await ingestedSession();
    let toolActive = true;
    const { provider } = makeTextBindingProvider(session, () =>
      toolActive ? "media.paged.image.tool.type" : null,
    );
    const write = (value: string) =>
      provider.writeProperty!({
        path: "characterFontFamily",
        target: { kind: "selection", scope: "content" },
        value: value as never,
      } as never);

    expect((await write("Georgia")).kind).toBe("applied");
    expect(session.state().type.family, "the write LANDED").toBe("Georgia");

    toolActive = false;
    expect((await write("Futura")).kind).toBe("decline");
    expect(session.state().type.family, "and the refusal changed nothing").toBe(
      "Georgia",
    );

    session.dispose();
    handle.dispose();
  });

  it("refuses an empty face rather than deferring the failure to paint time", async () => {
    const { session, handle } = await ingestedSession();
    const { provider } = makeTextBindingProvider(
      session,
      () => "media.paged.image.tool.type",
    );
    const r = await provider.writeProperty!({
      path: "characterFontFamily",
      target: { kind: "selection", scope: "content" },
      value: "   " as never,
    } as never);
    expect(r.kind).toBe("decline");
    expect(session.state().type.family).not.toBe("");

    session.dispose();
    handle.dispose();
  });
});
