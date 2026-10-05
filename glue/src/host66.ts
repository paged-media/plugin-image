/**
 * The protocol-66 host doors, probed. The contract (plugin-api 0.2.38)
 * now declares every door below, so the types are its own; what stays
 * here is the capability probe in front of each one. The bundle can
 * still be activated by a host that predates a door (an editor on an
 * older SDK, or a test double), so every accessor answers `null` /
 * `false` when the host does not announce the door or does not carry
 * the member, and every caller keeps its older path for that case —
 * nothing here is assumed.
 */
import type { ComponentType } from "react";
import type {
  BundleHost,
  ColorPickerProps,
  Disposable,
  ElementId,
  MutationOutcome,
  SceneImage,
  SceneImageTile,
  SceneLayerSurface,
  ToolSettingValue,
  ToolsSurface,
  WillSaveEvent,
} from "@paged-media/plugin-api";

export type { ColorPickerProps, SceneImage, SceneImageTile, ToolSettingValue };

type Mutation = Parameters<BundleHost["document"]["mutate"]>[0];

type ImageSceneSurface = Pick<SceneLayerSurface, "submitImage" | "submitImageTiles">;

/** The scene surface's image doors, or null on an older host. */
export function imageScene(surface: SceneLayerSurface): ImageSceneSurface | null {
  const s = surface as Partial<ImageSceneSurface>;
  return typeof s.submitImage === "function" && typeof s.submitImageTiles === "function"
    ? surface
    : null;
}

/** Whether the host takes scene images as bytes (no `number[]`). */
export const binaryScene = (host: BundleHost) => host.supports("rendering.sceneLayer.binary@1");

/** `host.parts.delete`, or null when the host cannot delete parts. */
export function partsDelete(host: BundleHost): ((path: string) => Promise<boolean>) | null {
  const parts = host.parts as Partial<BundleHost["parts"]>;
  if (!host.supports("storage.parts@2") || typeof parts.delete !== "function") return null;
  const del = parts as BundleHost["parts"];
  return (path) => del.delete(path);
}

/** Subscribe to the host's save of the document, when it announces one. */
export function onWillSave(
  host: BundleHost,
  listener: (e: WillSaveEvent) => void | Promise<void>,
): Disposable | null {
  const doc = host.document as Partial<BundleHost["document"]>;
  if (!host.supports("document.onWillSave@1") || typeof doc.onWillSave !== "function") return null;
  return (doc as BundleHost["document"]).onWillSave(listener);
}

/** Enter one of this bundle's edit contexts on `elementId`. Resolves
 *  false where the host cannot (the caller then says how to enter). */
export async function enterEditContext(
  host: BundleHost,
  type: string,
  elementId: ElementId,
): Promise<boolean> {
  const shell = host.shell as Partial<BundleHost["shell"]>;
  if (!host.supports("shell.enterEditContext@1") || typeof shell.enterEditContext !== "function") {
    return false;
  }
  try {
    return await (shell as BundleHost["shell"]).enterEditContext(type, elementId);
  } catch {
    return false;
  }
}

/** The host's tool-options values for this bundle's tools, or null. */
export function toolSettings(host: BundleHost): ToolsSurface | null {
  const tools = (host as Partial<Pick<BundleHost, "tools">>).tools as Partial<ToolsSurface> | undefined;
  if (
    !host.supports("tools.settings@1") ||
    !tools ||
    typeof tools.settings !== "function" ||
    typeof tools.onDidChangeSettings !== "function"
  ) {
    return null;
  }
  return tools as ToolsSurface;
}

/** The host's colour picker widget, or null (the panel's own then). */
export function hostColorPicker(host: BundleHost): ComponentType<ColorPickerProps> | null {
  const widgets = host.widgets as Partial<BundleHost["widgets"]>;
  return host.supports("widgets.colorPicker@1") && widgets.ColorPicker ? widgets.ColorPicker : null;
}

/** Whether a mutation's image bytes can cross as bytes (no JSON array),
 *  so the size of a baked image is not bounded by the JSON lane. */
export const binaryMutations = (host: BundleHost) => host.supports("document.mutateBinary@1");

/**
 * `mutation` with `bytes` as the image of its first `replaceImageBytes`
 * (whose own `bytes` is `[]`): over the binary lane where the host has
 * one, else the bytes spliced in as the `number[]` the JSON lane takes.
 */
export async function mutateWithBytes(
  host: BundleHost,
  mutation: Mutation,
  bytes: Uint8Array,
): Promise<MutationOutcome> {
  const doc = host.document as Partial<BundleHost["document"]>;
  if (binaryMutations(host) && typeof doc.mutateWithBytes === "function") {
    return (doc as BundleHost["document"]).mutateWithBytes(mutation, bytes);
  }
  return host.document.mutate(spliceBytes(mutation, bytes));
}

/** The JSON-lane form: the first `replaceImageBytes` gets the bytes. */
export function spliceBytes(mutation: Mutation, bytes: Uint8Array): Mutation {
  let done = false;
  const visit = (m: Mutation): Mutation => {
    const any = m as { op: string; args: Record<string, unknown> };
    if (!done && any.op === "replaceImageBytes") {
      done = true;
      return { ...any, args: { ...any.args, bytes: Array.from(bytes) } } as Mutation;
    }
    if (any.op === "batch") {
      const ops = (any.args.ops as Mutation[]).map(visit);
      return { ...any, args: { ...any.args, ops } } as Mutation;
    }
    return m;
  };
  return visit(mutation);
}
