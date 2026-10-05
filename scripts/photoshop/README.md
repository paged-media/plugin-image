# Photoshop oracle lane

Adobe Photoshop is the oracle for paged.image's blend modes, adjustments,
filters and layer compositing — the way Illustrator is for paged.draw's
path operations. This directory RECORDS Photoshop's answers; CI REPLAYS
them from the committed fixtures and never needs Photoshop.

```bash
bash scripts/photoshop/run-probe.sh scripts/photoshop/probes/blend-modes.jsx
bash scripts/photoshop/run-probe.sh scripts/photoshop/probes/adjustments.jsx
bash scripts/photoshop/run-probe.sh scripts/photoshop/probes/filters.jsx
bash scripts/photoshop/run-probe.sh scripts/photoshop/probes/layer-stacks.jsx
bash scripts/photoshop/run-probe.sh scripts/photoshop/probes/vector-masks.jsx
# then re-measure and review the ledger diff:
PAGED_PHOTOSHOP_LEDGER=write cargo test -p image-conformance --test oracle_photoshop
```

| file | role |
| --- | --- |
| `run-probe.sh` | consent check, launch, ping, run under a timeout, judge, record |
| `lib/probe-lib.jsx` | ExtendScript helpers prepended to every probe (JSON, open/save, one case per duplicate, ActionManager) |
| `lib/stimuli.mjs` | the deterministic 128×128 16-bit untagged PNG stimuli |
| `lib/write-fixture.mjs` | judges the reply and the files, then copies them into `image-conformance/fixtures/photoshop/` with provenance |
| `probes/*.jsx` | one probe per family |

Replays: `image-conformance/tests/oracle_photoshop.rs` (blend modes,
adjustments, filters → `fixtures/photoshop/ledger.json`, every case
classified `agreement` / `convention` / `defect` / `no-counterpart`) and
`image-conformance/tests/psd_composite_photoshop.rs` (the layered PSDs) and
`image-conformance/tests/psd_vector_masks_photoshop.rs` (vector masks: the
merged composite, and the render Photoshop caches in each masked layer's
channel −2).

## Traps, and what each step does about them

- **Consent.** macOS gates Apple events per (controlling app, target
  app). The runner asks without prompting first and explains `-1743`
  (denied) / `-1744` (macOS is about to ask).
- **Silent hangs.** A sign-in screen, a licence dialog, crash recovery
  or an unanswered consent prompt all look like a hang. The runner pings
  with `app.version` under a 60 s timeout before sending a probe and
  pings again afterwards; `-1712` is reported as "a modal dialog is
  probably open". Never SIGKILL Photoshop — it comes back with a
  recovery dialog.
- **Files.** Photoshop reads and writes a `mktemp` directory under
  `$TMPDIR` without any prompt (verified with Photoshop 27.10 on
  macOS 26.4). The probe text itself is handed over as a STRING, so the
  app never has to read a script from disk.
- **Colour.** The stimuli are untagged 16-bit PNGs. With dialogs
  suppressed Photoshop opens them in the working RGB space without
  converting a number; every probe's outputs record the document profile
  (`none`) and Color Settings (recorded per fixture; the recording used
  "Europe General Purpose 3", whose RGB working space is sRGB
  IEC61966-2.1). Each probe carries an `identity` case that the replay
  holds to agreement — the proof that no conversion happened. The user's
  Color Settings are read, never changed.
- **Merged data.** `layer-stacks.jsx` sets "maximize compatibility" for
  the run (restored afterwards) so every PSD carries Photoshop's real
  merged composite; the replay checks resource 0x0421 says so.
- **Judge the artifact.** `osascript` exiting 0 says only that a string
  came back. `write-fixture.mjs` refuses a reply with any case error, a
  document left open, a missing output, or an output that is not a
  16-bit PNG of the stimulus size.
- **What the DOM cannot say.** The scripting DOM sets no fill rule, no
  multi-subpath component and no invert flag on a vector mask.
  `vector-masks.jsx` writes such a case with the DOM, patches the one
  field in the saved bytes (BINARY file I/O), has Photoshop OPEN the
  patched file and SAVE it again — so the recorded PSD and its composite
  are still Photoshop's own.
- **ExtendScript** is ES3 and the text crosses an Apple event: plain
  ASCII only (the runner checks), no `JSON`, no `let`.
