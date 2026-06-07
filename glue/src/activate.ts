// The paged.image bundle entry. M0 scope (the honest slice the
// published SDK carries): the adjustments panel placeholder + the
// open command. The raster engines live in the Rust workspace
// (image-*); their wasm surface (image-js) joins the bundle when the
// engine lanes land — the GPU device is created in THIS realm, not
// inside loadBundleWasm's no-authority sandbox (BREAKAGE I-07). Asset
// ingest (I-04), PSD open-handler registration (I-05), and the
// viewport texture lane (I-01/I-06) are SDK gaps tracked in
// BREAKAGE_LOG.md — the panel says so rather than faking them.

import type { BundleHandle, BundleHost } from "@paged-media/plugin-api";
import { contributePanel } from "@paged-media/plugin-sdk";

import manifest from "@paged-media/image-manifest/manifest.json";

import { makeImagePanel } from "./panels/image-panel";

const PANEL_ID = "media.paged.image.panel.adjustments";

export function activate(host: BundleHost): BundleHandle {
  contributePanel(host, {
    id: PANEL_ID,
    title: "Image",
    icon: "panel-canvas",
    component: makeImagePanel(host),
    defaultDock: "right",
  });
  host.contribute.command({
    id: "media.paged.image.command.openImage",
    title: "Open image panel",
    category: "Image",
    handler: () => {
      host.shell.openPanel(PANEL_ID);
    },
  });
  host.log.info(`activated (apiVersion ${manifest.apiVersion})`);
  return { dispose() {} };
}

export { manifest, PANEL_ID };
