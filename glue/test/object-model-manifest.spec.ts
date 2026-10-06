// @feat: image.object-model
//
// The manifest's `contributes.objectModel` and the schema files it names
// are GENERATED from what the bundle registers at runtime, so the two
// cannot drift (the runtime check compares inline schemas only; these are
// files). Regenerate with `UPDATE_OBJECT_MODEL=1 pnpm vitest run
// test/object-model-manifest.spec.ts`.

import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it } from "vitest";
import { createHeadlessHost, type HeadlessHost } from "@paged-media/plugin-sdk";

import { imageBundle } from "../src/index";
import { manifestObjectModel } from "../src/object-model/contribute";
import { KIND_ROWS, type ImageKind } from "../src/object-model/image-objects";
import { minimalIdml } from "./fixtures/minimal-idml";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const MANIFESTS = [join(ROOT, "manifest"), join(ROOT, "glue")];
const UPDATE = process.env.UPDATE_OBJECT_MODEL === "1";
const silent = { debug() {}, info() {}, warn() {}, error() {} };

let live: HeadlessHost | null = null;
afterEach(() => {
  live?.dispose();
  live = null;
});

const json = (v: unknown) => JSON.stringify(v, null, 2) + "\n";

describe("the manifest's object model is the registered one", () => {
  it("declares every registered kind and typed command, and the schema files hold the runtime rows", async () => {
    const h = (live = await createHeadlessHost({ console: silent, ...(UPDATE ? { capabilityMode: "warn" as const } : {}) }));
    await h.load(minimalIdml());
    h.loadBundle(imageBundle);
    const commands = (await h.objects.commands()).filter((c) => c.owner === "media.paged.image");
    const om = manifestObjectModel(
      commands.map((c) => ({ id: c.id, title: c.title, args: c.args, handler: () => null })),
    );
    for (const dir of MANIFESTS) {
      const path = join(dir, "manifest.json");
      const manifest = JSON.parse(readFileSync(path, "utf8")) as { contributes: Record<string, unknown> };
      const files = Object.fromEntries(
        (Object.keys(KIND_ROWS) as ImageKind[]).map((k) => [join(dir, "object-model", `${k}.json`), json(KIND_ROWS[k])]),
      );
      if (UPDATE) {
        manifest.contributes.objectModel = om;
        writeFileSync(path, json(manifest));
        mkdirSync(join(dir, "object-model"), { recursive: true });
        for (const [f, body] of Object.entries(files)) writeFileSync(f, body);
        continue;
      }
      expect(manifest.contributes.objectModel, `${path}: regenerate with UPDATE_OBJECT_MODEL=1`).toEqual(om);
      for (const [f, body] of Object.entries(files)) {
        expect(existsSync(f) ? readFileSync(f, "utf8") : null, `${f} is stale`).toBe(body);
      }
    }
    // Every untyped command has its typed twin (the shrink-only baseline
    // goes to zero for this plugin).
    const declared = JSON.parse(readFileSync(join(ROOT, "manifest", "manifest.json"), "utf8")) as {
      contributes: { commands: string[] };
    };
    const typed = new Set(commands.map((c) => c.id));
    expect(declared.contributes.commands.filter((id) => !typed.has(id))).toEqual([]);
  });
});
