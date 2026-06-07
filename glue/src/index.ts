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
