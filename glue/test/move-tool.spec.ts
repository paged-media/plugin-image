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

// The raster Move tool's decisions: what moves (the selection or the
// layer), whether Alt copies, and the arrow-key steps.

import { describe, expect, it } from "vitest";

import { movePixels, nudgeFor } from "../src/move-tool";
import type { ImageSession } from "../src/session";

function sessionWith(selection: { x: number; y: number; w: number; h: number } | null) {
  const calls: string[] = [];
  const session = {
    state: () => ({ selection, source: { width: 10, height: 10 } }),
    moveSelection: async (dx: number, dy: number, vacate: number) => {
      calls.push(`moveSelection(${dx},${dy},${vacate})`);
      return true;
    },
    offsetLayer: async (dx: number, dy: number, edge: number) => {
      calls.push(`offsetLayer(${dx},${dy},${edge})`);
      return true;
    },
  } as unknown as ImageSession;
  return { session, calls };
}

describe("the raster Move tool", () => {
  it("moves the selected pixels when there is a selection; Alt copies", async () => {
    const { session, calls } = sessionWith({ x: 1, y: 1, w: 3, h: 3 });
    await movePixels(session, 2, -1, false);
    await movePixels(session, 2, -1, true);
    expect(calls).toEqual(["moveSelection(2,-1,0)", "moveSelection(2,-1,1)"]);
  });

  it("moves the active layer when nothing is selected", async () => {
    const { session, calls } = sessionWith(null);
    await movePixels(session, 5, 0, true);
    expect(calls).toEqual(["offsetLayer(5,0,0)"]);
  });

  it("a zero move does nothing (a click is not an edit)", async () => {
    const { session, calls } = sessionWith(null);
    expect(await movePixels(session, 0, 0, false)).toBe(false);
    expect(calls).toEqual([]);
  });

  it("arrows nudge 1 px, Shift+arrow 10, other keys nothing", () => {
    expect(nudgeFor("ArrowLeft", false)).toEqual([-1, 0]);
    expect(nudgeFor("ArrowDown", true)).toEqual([0, 10]);
    expect(nudgeFor("a", false)).toBeNull();
  });
});
