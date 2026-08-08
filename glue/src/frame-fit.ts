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

// The ONE image ↔ page transform every on-canvas image tool shares.
//
// The placed image is aspect-fit + centered inside the frame's content
// box (box = the frame's page-local bounds), exactly as the session's
// Apply lays out the C-1 Stage-A scene item. Crop, selection and brush
// all need the same mapping, and it MUST be the same mapping the
// composite uses or the overlay would drift off the pixels it claims to
// address — so it lives here once instead of three times.

import type { BundleHost, ElementGeometryItem } from "@paged-media/plugin-api";

/** The image→page aspect-fit transform for one frame box. */
export interface FitTransform {
  pageId: string;
  /** page-local pt of the image's top-left (0,0). */
  originX: number;
  originY: number;
  /** image-px → page-pt scale (uniform; aspect-fit). */
  scale: number;
}

/** Aspect-fit an `imgW`×`imgH` image into the frame `bounds`
 *  `[top,left,bottom,right]` (page-local pt), centered. */
export function fitInto(
  geom: ElementGeometryItem,
  imgW: number,
  imgH: number,
): FitTransform | null {
  const b = geom.bounds;
  if (!b) return null;
  // C-23 — a pasteboard frame reports `pageId: null`. A fit result is
  // consumed as a PLACEMENT on a page, so there is nothing to return
  // here; the caller already treats null as "cannot fit". Before C-23
  // the door omitted off-page frames entirely, so this path was reached
  // the same way — with a less honest reason.
  if (!geom.pageId) return null;
  const [top, left, bottom, right] = b;
  const boxW = Math.max(right - left, 1);
  const boxH = Math.max(bottom - top, 1);
  const scale = Math.min(boxW / imgW, boxH / imgH);
  const w = imgW * scale;
  const h = imgH * scale;
  return {
    pageId: geom.pageId,
    originX: left + (boxW - w) / 2,
    originY: top + (boxH - h) / 2,
    scale,
  };
}

/** image px → page-local pt (identity without a transform, so a tool
 *  degrades to raw page points rather than throwing). */
export function imageToPage(
  fit: FitTransform | null,
  p: [number, number],
): [number, number] {
  return fit
    ? [fit.originX + p[0] * fit.scale, fit.originY + p[1] * fit.scale]
    : p;
}

/** page-local pt → image px (the inverse of [`imageToPage`]). */
export function pageToImage(
  fit: FitTransform | null,
  p: [number, number],
): [number, number] {
  return fit
    ? [(p[0] - fit.originX) / fit.scale, (p[1] - fit.originY) / fit.scale]
    : p;
}

/** The composited frame's fit transform for the session's live source,
 *  read through `host.document.elementGeometry`. Null when nothing is
 *  ingested, no frame is targeted, or the geometry read fails (logged
 *  under `label` — an honest miss, never a throw into the gesture). */
export async function resolveFrameFit(
  host: BundleHost,
  source: { elementId: string | null; width: number; height: number } | null,
  label: string,
): Promise<FitTransform | null> {
  if (!source || !source.elementId) return null;
  try {
    const geom = await host.document.elementGeometry([
      { kind: "rectangle", id: source.elementId } as never,
    ]);
    return geom[0] ? fitInto(geom[0], source.width, source.height) : null;
  } catch (err) {
    host.log.debug(`${label}: frame geometry read failed`, err);
    return null;
  }
}
