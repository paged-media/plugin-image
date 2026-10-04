# ADR 463 — One undo list for pixels and the layer stack, reachable from the host

- **Status:** Accepted, 2026-10-04.
- **Scope:** `image-js/src/layers.rs` (the undo list), `image-js/src/lib.rs` (the exports), `glue/src/activate.ts` (the host hooks)

## Context

Undo covered pixel edits only, through a bounded tile journal keyed by layer id. Adding,
removing and reordering layers, opacity, blend, masks, groups, adjustment layers and smart
objects could not be undone. Removing a layer cleared the whole history, because the
journal's entries for that layer could never be replayed, and the canvas operations cleared
it too, because the journal's tiles address one canvas extent. The host's Cmd+Z never
reached the image: the edit context declared no undo hooks.

## Decision

The stack keeps ONE ordered list of steps. A pixel step is one journal entry (the journal
still holds its tiles); a structure step holds a snapshot of the layers, groups, active layer
and extent, which is cheap because layer pixels are shared `Arc` buffers.

Because undo is last in, first out, a pixel step is only ever replayed on the stack it was
recorded against: every structure change made after it has been undone first. So removing
a layer and rotating or resizing the canvas are ordinary undo steps and keep the history
before them. Both record themselves, so no caller can skip it.

Consecutive structure steps with the same merge key merge into one, so a slider drag is one
step. The list is bounded (200 structure steps; the journal bounds pixel steps by bytes) and
kept in step with the journal when it evicts or clears.

Inside the image's edit context, the host's undo and redo call the image's (ADR 012's
in-context hooks); when the image has nothing to undo, the hooks decline and the keystroke
goes to the document. Where the host reads them (protocol 66), the context also names the step, so the
Edit menu reads "Undo Brush stroke" rather than a generic "Undo".

## Evidence

- `image-js/src/layers.rs:1469-1492` — `recorded`: a structure step, with merging
- `image-js/src/layers.rs:1518-1530` — the journal and the list kept in step
- `image-js/src/layers.rs:1552-1583` — undo over both kinds of step
- `image-js/src/layers.rs:1015-1017` — removing a layer records itself
- `image-js/src/layers.rs:808-820` — the canvas operations record themselves
- `image-js/src/lib.rs:2680-2690` — every structural export goes through `recorded_edit`
- `glue/src/activate.ts:830-841` — the host undo hooks, declining when there is nothing to undo

## Alternatives considered

Clearing the history at every structural change, as before: it made layer work unsafe to
explore. Snapshotting pixels for every step instead of the tile journal: one 4000×3000 step
would hold 48 MB, against the journal's damaged tiles only.

## Consequences

A crop, resize or straighten still flattens the stack and drops the history (they replace
the image, not the stack). The history is not persisted: a reopened image starts with none
(ADR 462). After a commit, the document's own undo reverts the commit as one step, and the
image's undo list keeps the session's steps.

## Related

- [ADR 462](462-sessions-persist-in-parts.md) — commit and reopen
- [ADR 460](460-document-is-not-the-store.md) — superseded by ADR 462
