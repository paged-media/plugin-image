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

// The host's tool-options values reach the session in the session's own
// units (percent → 0..1), and a value of the wrong type is ignored.

import { describe, expect, it, vi } from "vitest";

import { applyToolSettings, toolOptionFields } from "../src/tool-options";
import type { ImageSession } from "../src/session";

const session = () =>
  ({
    setBrushParams: vi.fn(),
    setWandOptions: vi.fn(),
    setGradientKind: vi.fn(),
  }) as unknown as ImageSession & Record<string, ReturnType<typeof vi.fn>>;

describe("tool options", () => {
  it("maps brush percentages onto the brush's 0..1 parameters", () => {
    const s = session();
    applyToolSettings(s, "brush", { size: 40, hardness: 50, opacity: 80, flow: 25 });
    expect(s.setBrushParams).toHaveBeenCalledWith({ size: 40, hardness: 0.5, opacity: 0.8, flow: 0.25 });
  });

  it("passes wand tolerance and contiguity, and ignores mistyped values", () => {
    const s = session();
    applyToolSettings(s, "wand", { tolerance: 12, contiguous: "yes" });
    expect(s.setWandOptions).toHaveBeenCalledWith({ tolerance: 12 });
  });

  it("accepts only known gradient kinds", () => {
    const s = session();
    applyToolSettings(s, "gradient", { kind: "spiral" });
    expect(s.setGradientKind).not.toHaveBeenCalled();
    applyToolSettings(s, "gradient", { kind: "radial" });
    expect(s.setGradientKind).toHaveBeenCalledWith("radial");
  });

  it("declares fields for the six tools", () => {
    const f = toolOptionFields({ brush: "b", pencil: "p", eraser: "e", wand: "w", bucket: "k", gradient: "g" });
    expect(Object.keys(f).sort()).toEqual(["b", "e", "g", "k", "p", "w"]);
    expect(f.w.map((x) => x.key)).toEqual(["tolerance", "contiguous"]);
  });
});
