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

// Counts what the SESSION asks of the engine: every `ImageEngine` method
// call by name, forwarded untouched to the real engine (nothing mocked —
// the wasm still answers). Budgets for the bundle are counts of these
// calls and of what reaches the host, never milliseconds.
//
// Install with `vi.mock("../../src/engine", …)` around the real
// `bootEngine` (see perf-budgets.spec.ts), so the session under test
// boots a counted engine without knowing it.

import type { ImageEngine } from "../../src/engine";

export interface EngineLog {
  readonly calls: Readonly<Record<string, number>>;
  count(method: string): number;
  reset(): void;
}

export function countingEngine(engine: ImageEngine): {
  engine: ImageEngine;
  log: EngineLog;
} {
  let calls: Record<string, number> = {};
  const log: EngineLog = {
    get calls() {
      return calls;
    },
    count: (m) => calls[m] ?? 0,
    reset: () => {
      calls = {};
    },
  };
  const proxy = new Proxy(engine, {
    get(target, prop, receiver) {
      const v = Reflect.get(target, prop, receiver) as unknown;
      if (typeof v !== "function" || typeof prop !== "string") return v;
      return (...args: unknown[]) => {
        calls[prop] = (calls[prop] ?? 0) + 1;
        return (v as (...a: unknown[]) => unknown).apply(target, args);
      };
    },
  });
  return { engine: proxy, log };
}
