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

    // These two are STRUCTURALLY meaningless on a raster run — not
    // merely unbuilt — but they are DECLARED, so the host asks us and
    // gets a blank rather than asking core and painting the text
    // caret's paragraph over an image frame.
    //
    // Hyphenation needs LINE BREAKING to hyphenate at, and this lane
    // breaks only where the author typed a newline. Kerning method
    // (optical vs metrics) is a CHOICE the shaper does not expose — it
    // applies the face's own kerning and offers no alternative. Both
    // will stay here however far the lane grows, which is what makes
    // them the right examples after three earlier ones moved.
    //
    // NOTE the example has now moved TWICE in one day: first off
    // `characterLeading` (writable once the lane laid out more than one
    // line), then off `characterFontStyle` (writable once the lane
    // passed a style to the face door). A test whose example keeps
    // moving out from under it is the test doing its job — the claim is
    // about ownership-WITHOUT-value, so it needs a path that still has
    // none.
    //
    // THIRD MOVE, and the last one this example can make: leading,
    // then style, then underline all became writable as the lane grew.
    // `characterKerningMethod` is structurally absent rather than
    // merely unbuilt — the shaper applies the face's kerning and
    // exposes no choice of METHOD (optical vs metrics), so there is no
    // value to have. That is why it will not move.
    expect(text.paths).toContain("paragraphHyphenation");
    expect(text.paths).toContain("characterKerningMethod");
    // …and they are NOT writable, so the control renders read-only
    // instead of offering a commit that lands nowhere.
    expect(text.writablePaths).not.toContain("paragraphHyphenation");
    expect(text.writablePaths).not.toContain("characterKerningMethod");

    bundle.dispose();
    handle.dispose();
  });

  it("EVERY absent path is absent STRUCTURALLY, not merely unbuilt", () => {
    // THE CLOSING ASSERTION for the type lane. The interesting property
    // is not how many paths are served — it is that the ones which are
    // NOT have a reason that no amount of building would remove.
    //
    // Each of the eight below fails for one of three structural
    // reasons, and every one of them traces back to a single fact: a
    // raster layer is not a text COLUMN.
    //   · no measure to lay out against — indents, space before/after
    //   · no line BREAKING, so nothing to hyphenate or keep together
    //   · no CHOICE exposed by the shaper — kerning method
    // If a future lane ever wraps, the first two groups become real and
    // this test is where that decision surfaces.
    const { registry, bundle, handle } = boot();
    const text = registry
      .activeProviders()
      .map((p) => p.provides)
      .find((p) => p.paths?.includes("characterFontFamily"))!;

    const absent = (text.paths ?? []).filter(
      (p) => !(text.writablePaths ?? []).includes(p),
    );
    expect([...absent].sort()).toEqual(
      [
        // NOT `characterCase` — that one IS served (upper/lower), with
        // SMALL CAPS refused on write rather than faked by scaling
        // capitals. A refusal with a reason is a served path, not an
        // absent one.
        "characterKerningMethod",
        "paragraphFirstLineIndent",
        "paragraphHyphenation",
        "paragraphKeepLinesTogether",
        "paragraphLeftIndent",
        "paragraphRightIndent",
        "paragraphSpaceAfter",
        "paragraphSpaceBefore",
      ].sort(),
    );

    bundle.dispose();
    handle.dispose();
  });

  it("SERVES the font size in POINTS — the unit question, answered", () => {
    // This test used to assert the opposite, and said why: "the day
    // someone wires the bridge, it fails and makes them state the
    // decision rather than inherit it." The bridge is wired and the
    // decision is stated — `characterFontSize` means "the size unit of
    // whatever you are editing", surfaced in the panel's own unit.
    //
    // A pin that documents an open question is worth keeping precisely
    // because it becomes a liability the day the question is answered.
    const { registry, bundle, handle } = boot();
    const text = registry
      .activeProviders()
      .map((p) => p.provides)
      .find((p) => p.paths?.includes("characterFontFamily"))!;

    expect(text.paths).toContain("characterFontSize");
    // WRITABLE now — the half that changed. A size control the user can
    // read but not set would be a worse answer than a blank one.
    expect(text.writablePaths).toContain("characterFontSize");

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

  it("CONVERTS the font size through the layout's own points-per-pixel", async () => {
    // The conversion is the whole feature, so it is measured rather
    // than declared. `ptPerPx` is not recomputed here — it is the SAME
    // number `submitLayer` used to lay the composite out, which is why
    // a converted size cannot disagree with the pixels on screen.
    const { session, handle } = await ingestedSession();
    const { provider } = makeTextBindingProvider(
      session,
      () => "media.paged.image.tool.type",
    );
    const req = {
      path: "characterFontSize",
      target: { kind: "selection", scope: "content" },
    } as never;

    // BEFORE a composite there is no layout, so there is no conversion.
    // ABSENT rather than a 1:1 guess — a wrong number in a size field
    // is worse than a blank one, because it looks right.
    expect(session.state().ptPerPx).toBeNull();
    const cold = await provider.readProperty!(req);
    expect(cold.kind, "no layout ⇒ no size").toBe("absent");

    // Composite once so the session learns the frame's scale.
    await session.apply();
    const ptPerPx = session.state().ptPerPx;
    expect(ptPerPx, "the composite recorded the scale").not.toBeNull();
    expect(ptPerPx!).toBeGreaterThan(0);

    // READ: image px → points, through that exact number.
    const sizePx = session.state().type.sizePx;
    const read = await provider.readProperty!(req);
    expect(read.kind).toBe("value");
    expect((read as unknown as { value: number }).value).toBeCloseTo(
      sizePx * ptPerPx!,
      5,
    );

    // WRITE: points → image px, the inverse. Asserted as a ROUND TRIP
    // rather than against a hand-computed pixel count, so the test
    // cannot pass by mirroring the implementation's arithmetic.
    //
    // The target is DERIVED from the layout rather than picked: the
    // conversion lands on a PIXEL GRID, so a point size finer than one
    // pixel is not representable and comes back quantized. Writing a
    // literal 36 here failed on this fixture for exactly that reason —
    // its image is tiny against its frame, so 36 pt is a third of a
    // pixel. Twenty pixels' worth is representable by construction.
    const target = 20 * ptPerPx!;
    const wrote = await provider.writeProperty!({
      ...(req as object),
      value: target,
    } as never);
    expect(wrote.kind).toBe("applied");
    expect(session.state().type.sizePx).toBe(20);
    const back = await provider.readProperty!(req);
    expect((back as unknown as { value: number }).value).toBeCloseTo(target, 5);

    // THE QUANTIZATION, asserted rather than left as a footnote: a size
    // below one pixel clamps to one pixel instead of rounding to zero
    // and rasterizing nothing. The user sees the clamped value reflected
    // back, which is the honest signal that the grid is the limit.
    await provider.writeProperty!({
      ...(req as object),
      value: ptPerPx! / 4,
    } as never);
    expect(session.state().type.sizePx, "never below one pixel").toBe(1);

    // A nonsense size is refused, not rounded into existence.
    expect(
      (
        await provider.writeProperty!({
          ...(req as object),
          value: 0,
        } as never)
      ).kind,
    ).toBe("decline");

    session.dispose();
    handle.dispose();
  });

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
