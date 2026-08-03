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
  freshIdentityParams,
  packAdjustExt,
  ADJUST_EXT_LEN,
  DEFAULT_BW_WEIGHTS,
  DEFAULT_PHOTO_FILTER_COLOR,
  GRADIENT_KINDS,
  IDENTITY_PARAMS,
  IDENTITY_LEVELS_CHANNEL,
  ENGINE_NOT_BUILT,
  type AdjustParams,
  type BlackWhiteParams,
  type ChannelMixerParams,
  type ColorBalanceParams,
  type GradientKind,
  type ImageEngine,
  type ImageWasmModule,
  type LevelsChannel,
  type LevelsRgbParams,
  type PhotoFilterParams,
  type RasterFormat,
  type Rgba01,
} from "./engine";
export {
  createImageSession,
  elementIdOf,
  FEATHER_SIGMA_DEFAULT,
  FILL_NOISE_SEED_DEFAULT,
  type FillRequest,
  type ImageSession,
  type ImageSessionState,
  type SaveBackResult,
} from "./session";
export {
  createSelectionMachine,
  modeFromModifiers,
  WAND_TOLERANCE_DEFAULT,
  WAND_CONTIGUOUS_DEFAULT,
  type SelectionMachine,
  type SelectionShapeKind,
} from "./selection-machine";
export { makeSelectionGesture } from "./selection-tool";
export {
  selectionModeCode,
  type SelectionMode,
  type SelectionStats,
} from "./engine";
export {
  createDecodePool,
  DECODE_WORKER_MODULE,
  type DecodePool,
  type DecodedRGBA,
} from "./decode-pool";
export type { DecodeReply, DecodeRequest } from "./decode-worker";
