// Latest-wins coalescing for work that re-reads the CURRENT state.
//
// A recomposite folds whatever the layer stack is at the moment it runs,
// so when ten slider steps arrive while one composite is in flight, nine
// of the composites they would trigger are already stale before they
// start. `latestWins(run)` returns a function that:
//
//   - runs `run` at once when nothing is in flight;
//   - while one IS in flight, schedules exactly ONE trailing run and
//     hands every caller in the meantime that same trailing promise.
//
// So a burst of N calls costs at most 2 runs, and every caller's promise
// settles after a run that started AFTER its own call — no caller is
// told "done" by a run that could not have seen its change.

export function latestWins<T>(run: () => Promise<T>): () => Promise<T> {
  let inFlight: Promise<T> | null = null;
  let trailing: Promise<T> | null = null;

  const start = (): Promise<T> => {
    const p = run().finally(() => {
      if (inFlight === p) inFlight = null;
    });
    inFlight = p;
    return p;
  };

  return () => {
    if (!inFlight) return start();
    if (!trailing) {
      const current = inFlight;
      trailing = current
        .catch(() => undefined)
        .then(() => {
          trailing = null;
          return start();
        });
    }
    return trailing;
  };
}
