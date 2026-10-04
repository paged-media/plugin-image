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

import { describe, expect, it } from "vitest";

import { latestWins } from "../src/coalesce";

/** A run whose completion the test controls. */
function gated() {
  const gates: Array<() => void> = [];
  let runs = 0;
  const run = () =>
    new Promise<number>((resolve) => {
      runs += 1;
      const n = runs;
      gates.push(() => resolve(n));
    });
  return { run, gates, runs: () => runs };
}

describe("latestWins", () => {
  it("runs at once when idle", async () => {
    const g = gated();
    const f = latestWins(g.run);
    const p = f();
    expect(g.runs()).toBe(1);
    g.gates[0]();
    expect(await p).toBe(1);
  });

  it("a burst during one run costs exactly one trailing run, shared by every caller", async () => {
    const g = gated();
    const f = latestWins(g.run);
    const first = f();
    const burst = [f(), f(), f(), f()];
    expect(g.runs()).toBe(1);
    g.gates[0]();
    await first;
    await Promise.resolve();
    await Promise.resolve();
    expect(g.runs()).toBe(2);
    g.gates[1]();
    expect(await Promise.all(burst)).toEqual([2, 2, 2, 2]);
  });

  it("no caller settles on a run that started before its call", async () => {
    const g = gated();
    const f = latestWins(g.run);
    f();
    const late = f();
    g.gates[0]();
    let settled = false;
    void late.then(() => {
      settled = true;
    });
    await new Promise((r) => setTimeout(r, 0));
    expect(settled).toBe(false); // the trailing run has not finished
    g.gates[1]();
    expect(await late).toBe(2);
  });

  it("a failed run does not wedge the queue", async () => {
    let n = 0;
    const f = latestWins(async () => {
      n += 1;
      if (n === 1) throw new Error("boom");
      return n;
    });
    await expect(f()).rejects.toThrow("boom");
    expect(await f()).toBe(2);
  });
});
