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

// @paged-media/image-glue — the paged.image plugin bundle.

import { defineBundle } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";

import { activate, PANEL_ID } from "./activate";
import manifestJson from "../manifest.json";

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
export {
  createDecodePool,
  DECODE_WORKER_MODULE,
  type DecodePool,
  type DecodedRGBA,
} from "./decode-pool";
export type { DecodeReply, DecodeRequest } from "./decode-worker";
