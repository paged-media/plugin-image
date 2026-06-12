// @paged-media/image-glue — the paged.image plugin bundle.

import { defineBundle } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";

import { activate, PANEL_ID } from "./activate";
import manifestJson from "@paged-media/image-manifest/manifest.json";

export const imageBundle = defineBundle({
  manifest: manifestJson as PluginManifest,
  activate,
});

export { activate, PANEL_ID };
export {
  wrapEngine,
  isIdentity,
  IDENTITY_PARAMS,
  ENGINE_NOT_BUILT,
  type AdjustParams,
  type ImageEngine,
  type ImageWasmModule,
} from "./engine";
export {
  createImageSession,
  elementIdOf,
  type ImageSession,
  type ImageSessionState,
} from "./session";
