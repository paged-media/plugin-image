# libvips oracle lane

`record.mjs` runs libvips over deterministic float stimuli for every
kernel `registry/kernels.yaml` names libvips as an oracle for, and
commits the outputs (`image-conformance/fixtures/vips/<id>.v`, vips's
native format) with provenance (`<id>.vips.json`: vips version, the exact
commands, the stimulus formula, our parameters). It also regenerates
`registry/vips-oracle.yaml`, keeping the tolerances the replay measured.

```bash
node scripts/vips/record.mjs                    # every row (needs the vips CLI)
node scripts/vips/record.mjs math.add geom.crop # just these
PAGED_VIPS_TOLERANCE=write cargo test --release -p image-conformance --test oracle_vips
```

The replay, `image-conformance/tests/oracle_vips.rs`, needs no libvips:
it regenerates the stimuli from the shared formula and compares the
scalar reference (always) and the GPU kernel (under the test device) with
the recorded output, within the measured per-row tolerance.

Notes that cost a run to learn:

- The vips CLI reads a leading `-` as an option, so `-0.125` or
  `"-1 -1 -1 1"` must follow a `--`; the recorder puts options first as
  `--name=value` and every positional after `--`.
- A windowed kernel's window halo is its declared MAX radius
  (`conv.gaussian_*`: 24), not the radius parameter: a 64-wide window
  yields 16 output columns, centred.
- `vips affine` puts pixel centres on integers; the engine puts them at
  `+0.5`, so a rotation about the image centre uses `--idx=-31.5` etc.
- A row whose libvips operation is a different formula is recorded as
  `diverges: <reason>` (the resamplers: libvips reduce widens the filter
  by the shrink factor; the engine does not).
