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

// The paged.image bundle entry — the M4 "editor enablement" slice. The
// platform doors this consumes (all probed, never assumed): C-5
// host.assets.getPlacedImage (a placed frame's ORIGINAL bytes), C-1
// Stage-A sceneLayer with the v41 image item (in-frame composite of the
// adjusted RGBA), K-2 contribute.importer (File/Open + drag-drop routes
// PSD/PNG/JPEG bytes here). The engine wasm (image-js: codec/PSD decode
// + the GPU-only Engine A adjustments) boots LAZILY in the bundle realm
// on first ingest — the GPU device is created THERE, not inside
// loadBundleWasm's no-authority sandbox (BREAKAGE I-07).

import type { BundleHandle, BundleHost } from "@paged-media/plugin-api";
import { contributePanel, contributeTool } from "@paged-media/plugin-sdk";

import manifest from "../manifest.json";

import { createImageSession } from "./session";
import { makeImagePanel } from "./panels/image-panel";
import { makeCropGesture } from "./crop-tool";

const PANEL_ID = "media.paged.image.panel.adjustments";
const CROP_TOOL_ID = "media.paged.image.tool.crop";

export function activate(host: BundleHost): BundleHandle {
  const session = createImageSession(host);

  contributePanel(host, {
    id: PANEL_ID,
    title: "Image",
    icon: "panel-canvas",
    component: makeImagePanel(session),
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

  // The selection-driven flow: "Adjust image" on a placed image frame —
  // ingest the frame's original bytes (C-5) and raise the panel; the
  // panel's committed Apply runs the GPU chain + the in-frame composite.
  host.contribute.command({
    id: "media.paged.image.command.adjustSelected",
    title: "Adjust image",
    category: "Image",
    handler: () => {
      host.shell.openPanel(PANEL_ID);
      void session.ingestSelection();
    },
  });

  // Auto-enhance: derive auto-levels + a gray-world white balance from the
  // ingested image's histogram (pure CPU readout) and set them on the panel
  // params. Like every edit it's PREVIEW-only — the user commits with Apply.
  host.contribute.command({
    id: "media.paged.image.command.autoEnhance",
    title: "Auto-enhance image",
    category: "Image",
    handler: () => {
      host.shell.openPanel(PANEL_ID);
      session.autoEnhance();
    },
  });

  // C-6 (I-06) — claim the ingested image's tile resource so the renderer
  // pulls level-0 tiles for the placed frame at its current scale (the
  // honest subset; the mip pyramid + Engine B window eval are the named
  // gap in tile-provider.ts). Degrades honestly when the host wires no
  // resource channel (rendering.resourceProvider@1 is false).
  host.contribute.command({
    id: "media.paged.image.command.claimTiles",
    title: "Serve image tiles to the renderer",
    category: "Image",
    handler: () => {
      host.shell.openPanel(PANEL_ID);
      session.claimTiles();
    },
  });

  // Crop + straighten TOOL (the on-canvas crop affordance). Registers
  // into the transform rail; its gesture drives the session's crop machine
  // (image_core::crop geometry) and renders the crop frame through the
  // LIVE host.overlay door. The COMMIT rides the commitCrop command (and
  // the panel button) so it's a deliberate, single action.
  contributeTool(host, {
    id: CROP_TOOL_ID,
    title: "Crop",
    icon: "tool-crop",
    group: CROP_TOOL_ID,
    section: "transform",
    // shift+x — yields "c" to the built-in Scissors tool (InDesign-canonical);
    // INV-REG-1 (editor registry-invariants) keeps tool shortcuts unique.
    shortcut: "shift+x",
    gesture: () => makeCropGesture(host, session),
  });

  // The crop commit command (also surfaced as the panel's "Apply crop"
  // button): cut the machine's rect out of the engine source + recomposite.
  host.contribute.command({
    id: "media.paged.image.command.commitCrop",
    title: "Apply crop",
    category: "Image",
    handler: () => {
      void session.commitCrop();
    },
  });

  // K-2 — the raster importer: opening/dropping a PSD/PNG/JPEG routes
  // its bytes HERE (decode into the session, raise the panel; it does
  // NOT replace the document). Degrades honestly on an older host.
  if (host.supports("contribute.importer@1")) {
    host.contribute.importer({
      id: "media.paged.image.importer.raster",
      title: "Image (PSD/PNG/JPEG)",
      extensions: [".psd", ".psb", ".png", ".jpg", ".jpeg"],
      mimeTypes: [
        "image/vnd.adobe.photoshop",
        "image/png",
        "image/jpeg",
      ],
      import: async ({ name, bytes }) => {
        host.shell.openPanel(PANEL_ID);
        await session.importBytes(name, bytes);
      },
    });
  }

  host.log.info(`activated (apiVersion ${manifest.apiVersion})`);

  return {
    dispose() {
      session.dispose();
    },
  };
}

export { manifest, PANEL_ID };
