// Registers paged.image's object model and its TYPED commands
// (`contribute.objectModel`, ADR 323).
//
// Every untyped command keeps working; each gets a typed twin with the
// same id, which takes it off the shrink-only untyped baseline. Most of
// them act on the image open in the edit context (the selection, the
// active layer) and take no arguments. Four become headless and name
// their target instead — `addLayer`, `addLayerMask`,
// `addLayerMaskHideAll`, `convertLayerToSmart` — and one is new,
// `replaceLayerPixels`. Those run through `host.objects` like any other
// caller, so each is one undo step.

import type {
  BundleHost,
  ObjectModelHandle,
  ObjectsSurface,
  TypedCommandContribution,
  ValueType,
} from "@paged-media/plugin-api";

import { bootEngine } from "../engine";
import {
  createImageObjects,
  KIND_CONTENT,
  KIND_PREFIX,
  KIND_ROWS,
  KIND_TITLES,
  parseImageAddress,
  imageAddress,
  type ImageCodec,
  type ImageKind,
  type ImageObjects,
} from "./image-objects";

/** One untyped command as `activate` registers it. */
export interface UntypedCommand {
  id: string;
  title: string;
  handler: (paged: unknown, payload?: unknown) => unknown;
}

const NO_ARGS: ValueType = { kind: "struct", fields: {} };
const LAYER_REF: ValueType = { kind: "ref", to: `${KIND_PREFIX}layer` };
const q = (kind: ImageKind) => `${KIND_PREFIX}${kind}`;

/** The typed twins whose arguments differ from "none", and the new ones. */
export const TYPED_ARGS: Record<string, { args: ValueType; title?: string }> = {
  "media.paged.image.command.addLayer": {
    args: { kind: "struct", fields: { image: { kind: "ref", to: q("image") }, name: { kind: "text" } } },
  },
  "media.paged.image.command.addLayerMask": { args: { kind: "struct", fields: { layer: LAYER_REF } } },
  "media.paged.image.command.addLayerMaskHideAll": { args: { kind: "struct", fields: { layer: LAYER_REF } } },
  "media.paged.image.command.convertLayerToSmart": { args: { kind: "struct", fields: { layer: LAYER_REF } } },
  "media.paged.image.command.replaceLayerPixels": {
    title: "Replace layer pixels",
    args: { kind: "struct", fields: { layer: LAYER_REF, source: { kind: "asset" } } },
  },
};

/** The CPU codec over the engine wasm (booted once; no GPU requested). */
export function engineCodec(): () => Promise<ImageCodec> {
  let booted: Promise<ImageCodec> | null = null;
  return () =>
    (booted ??= bootEngine().then((e) => ({
      decode(bytes: Uint8Array) {
        const d = e.decode(bytes);
        try {
          return { width: d.width, height: d.height, rgba: e.tile(d.handle, 0, 0, d.width, d.height) };
        } finally {
          e.freeImage(d.handle);
        }
      },
      encodePng: (rgba: Uint8Array, width: number, height: number) => e.encode(rgba, width, height, "png"),
    })));
}

const fail = (r: { applied: boolean; reason?: string; code?: string }) => {
  if (!r.applied) throw new Error(`${r.code ?? "failed"}: ${r.reason ?? "not applied"}`);
  return r;
};

function nativeHandlers(codec: () => Promise<ImageCodec>): Record<string, TypedCommandContribution["handler"]> {
  const frameOf = (layer: string) => {
    const a = parseImageAddress(layer);
    if (!a || a.kind !== "layer") throw new Error(`invalidValue: ${layer} is not a paged.image layer`);
    return a.frameId;
  };
  return {
    "media.paged.image.command.addLayer": async ({ objects }, args) => {
      const { image, name } = args as { image: string; name: string };
      return fail(await objects.batch([{ op: "create", kind: q("layer"), within: image, props: { name } }]));
    },
    "media.paged.image.command.addLayerMask": async ({ objects }, args) => {
      const { layer } = args as { layer: string };
      frameOf(layer);
      return fail(await objects.batch([{ op: "create", kind: q("mask"), within: layer }]));
    },
    "media.paged.image.command.addLayerMaskHideAll": async ({ objects }, args) => {
      const { layer } = args as { layer: string };
      const image = imageAddress(frameOf(layer));
      const [w, h] = await Promise.all([objects.get(image, "width"), objects.get(image, "height")]);
      if (w.kind !== "value" || h.kind !== "value") throw new Error(`unknownAddress: ${image}`);
      const [W, H] = [w.value as number, h.value as number];
      const black = new Uint8Array(W * H * 4);
      for (let i = 3; i < black.length; i += 4) black[i] = 255;
      const png = (await codec()).encodePng(black, W, H);
      const mask = layer.replace(`${KIND_PREFIX}layer/`, `${KIND_PREFIX}mask/`);
      return fail(
        await objects.batch([
          { op: "create", kind: q("mask"), within: layer },
          { op: "set", address: mask, path: "content", value: png },
        ]),
      );
    },
    "media.paged.image.command.convertLayerToSmart": async ({ objects }, args) => {
      const { layer } = args as { layer: string };
      frameOf(layer);
      return fail(await objects.batch([{ op: "create", kind: q("smartObject"), within: layer }]));
    },
    "media.paged.image.command.replaceLayerPixels": async ({ objects }, args) => {
      const { layer, source } = args as { layer: string; source: string };
      return fail(await objects.set(layer, "pixelSource", source));
    },
  };
}

/** The typed commands: one twin per untyped command, plus the new ones. */
export function typedCommands(
  untyped: readonly UntypedCommand[],
  codec: () => Promise<ImageCodec>,
): TypedCommandContribution[] {
  const native = nativeHandlers(codec);
  const out: TypedCommandContribution[] = untyped.map((c) => ({
    id: c.id,
    title: c.title,
    args: TYPED_ARGS[c.id]?.args ?? NO_ARGS,
    handler:
      native[c.id] ??
      (() => {
        // The interactive twin: the same action on the image in the edit
        // context. It returns nothing; its effect is the session's.
        c.handler(null, undefined);
        return null;
      }),
  }));
  for (const [id, spec] of Object.entries(TYPED_ARGS)) {
    if (out.some((c) => c.id === id)) continue;
    out.push({ id, title: spec.title ?? id, args: spec.args, handler: native[id]! });
  }
  return out;
}

/** `contributes.objectModel` as the manifest declares it (the kinds'
 *  rows live in `object-model/<kind>.json`, generated from KIND_ROWS). */
export function manifestObjectModel(commands: readonly TypedCommandContribution[]) {
  return {
    kinds: (Object.keys(KIND_ROWS) as ImageKind[]).map((kind) => ({
      kind,
      title: KIND_TITLES[kind],
      schema: `object-model/${kind}.json`,
    })),
    commands: commands.map((c) => ({ id: c.id, title: c.title, args: c.args })),
  };
}

/** The rows a kind's schema file holds (the registry adds `content`). */
export function schemaFileRows(kind: ImageKind) {
  return KIND_ROWS[kind];
}

export { KIND_CONTENT };

/**
 * Register the object model. Null when the host has no object registry
 * (`contribute.objectModel@1` false) — the bundle then works as before.
 */
export function contributeImageObjectModel(
  host: BundleHost,
  untyped: readonly UntypedCommand[],
  codec: () => Promise<ImageCodec> = engineCodec(),
): { handle: ObjectModelHandle; objects: ImageObjects } | null {
  const surface = host.contribute as Partial<BundleHost["contribute"]>;
  if (!host.supports("contribute.objectModel@1") || typeof surface.objectModel !== "function") return null;
  const objects = createImageObjects(host, codec);
  const handle = surface.objectModel({ kinds: objects.kinds, commands: typedCommands(untyped, codec) });
  return {
    handle,
    objects,
  };
}

export type { ObjectsSurface };
