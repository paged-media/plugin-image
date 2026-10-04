# Architecture decision records

An ADR records one load-bearing decision that has already been made: what was decided, what
in the code shows it, and what it obliges other code to do. It is a record, not a proposal.
When the code stops matching a record, the body is left as it is and a dated amendment is
added at the end.

ADR numbers are unique across the paged-media repositories, so a number names the same
record wherever it is cited. New records in this repository use 450–499. Lower numbers
predate that scheme; this repository holds none of them, and the ones its code rests on
live in other repositories. Records 450–461 were written on 2026-10-02 from the code as it
stood, for decisions made earlier; their status says so.

| ADR | Title | Status |
|---|---|---|
| [450](450-gpu-only-kernels.md) | Kernels execute on the GPU only; the scalar twin is for tests | Accepted, recorded retroactively 2026-10-02 |
| [451](451-one-kernel-definition.md) | One kernel definition feeds both lanes | Accepted, recorded retroactively 2026-10-02 |
| [452](452-kernel-abi.md) | The frozen shader kernel ABI (v1, amended to v1.1) | Accepted, recorded retroactively 2026-10-02 |
| [453](453-tolerance-not-golden-bytes.md) | GPU output is verified against the scalar reference by tolerance, never by golden bytes | Accepted, recorded retroactively 2026-10-02 (amended 2026-10-04) |
| [454](454-two-evaluation-engines.md) | Two evaluation engines over one kernel set | Accepted, recorded retroactively 2026-10-02 |
| [455](455-pixel-model.md) | The pixel model: explicit format, 16-bit float working tiles | Accepted, recorded retroactively 2026-10-02 |
| [456](456-codecs.md) | Codecs are pure-Rust, permissively licensed adapters behind I/O-free traits | Accepted, recorded retroactively 2026-10-02 |
| [457](457-colour-management.md) | Colour management: one seam, one engine for display and one for print | Accepted, recorded retroactively 2026-10-02 |
| [458](458-psd-preservation.md) | PSD is read and written by an own parser under a preservation rule | Accepted, recorded retroactively 2026-10-02 |
| [459](459-scene-layer-image-and-tiles.md) | Results reach the page as a retained scene-layer image plus a tile provider | Accepted, recorded retroactively 2026-10-02 |
| [460](460-document-is-not-the-store.md) | The document is not the store: edits live in the session and leave through save-back | Superseded by 462 (2026-10-04) |
| [461](461-clean-room-protocol.md) | The clean-room two-role protocol | Accepted, recorded retroactively 2026-10-02 |
| [462](462-sessions-persist-in-parts.md) | Raster sessions persist in container parts and bake into the frame | Accepted 2026-10-04 |
| [463](463-one-undo-list.md) | One undo list for pixels and the layer stack, reachable from the host | Accepted 2026-10-04 |

Decisions made in other repositories that this plugin's code rests on are listed in
[`../README.md`](../README.md).
