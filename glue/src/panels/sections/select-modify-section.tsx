// SELECT ▸ Modify, and the magic wand / paint bucket options.

import type { SelectionModifyOp } from "../../engine";
import type { ImageSession } from "../../session";
import type { WandOptions } from "../../selection-machine";
import { mono, note, row, sectionTitle } from "./styles";

const OPS: Array<[SelectionModifyOp, string]> = [
  ["expand", "Expand"],
  ["contract", "Contract"],
  ["border", "Border"],
  ["smooth", "Smooth"],
];

export function SelectModifySection({
  session,
  hasSelection,
  radius,
  onRadius,
  wand,
}: {
  session: ImageSession;
  hasSelection: boolean;
  radius: number;
  onRadius: (r: number) => void;
  wand: WandOptions;
}) {
  return (
    <>
      <div style={sectionTitle}>Selection</div>
      <div style={row}>
        <label htmlFor="pg-image-modify-radius">Modify by</label>
        <input
          id="pg-image-modify-radius"
          type="range"
          min={1}
          max={50}
          step={1}
          value={radius}
          onChange={(e) => onRadius(Number(e.target.value))}
        />
        <span style={mono}>{radius} px</span>
      </div>
      <div style={row}>
        {OPS.map(([op, label]) => (
          <button
            key={op}
            type="button"
            data-image-modify={op}
            disabled={!hasSelection}
            onClick={() => session.modifySelection(op, radius)}
          >
            {label}
          </button>
        ))}
      </div>
      <div style={row}>
        <label htmlFor="pg-image-wand-tolerance">Wand / bucket tolerance</label>
        <input
          id="pg-image-wand-tolerance"
          type="range"
          min={0}
          max={255}
          step={1}
          value={wand.tolerance}
          onChange={(e) => session.setWandOptions({ tolerance: Number(e.target.value) })}
        />
        <span style={mono}>{wand.tolerance}</span>
      </div>
      <div style={row}>
        <label htmlFor="pg-image-wand-contiguous">Contiguous</label>
        <input
          id="pg-image-wand-contiguous"
          type="checkbox"
          checked={wand.contiguous}
          onChange={(e) => session.setWandOptions({ contiguous: e.target.checked })}
        />
      </div>
      <div style={note}>
        Expand, contract and border make a hard-edged selection; smooth rounds
        corners and drops specks smaller than about the radius.
      </div>
    </>
  );
}
