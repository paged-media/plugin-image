// The DODGE / BURN / SPONGE options — Photoshop's options bar for the
// toning tools. Frozen into each stroke at pointer-down, like the brush's
// own parameters; the brush section's size, hardness, flow and spacing
// apply to these tools too (the sponge's strength IS the flow).

import { TONE_RANGES, type ToneOptions, type ToneRange } from "../../engine";
import { note, row, sectionTitle } from "./styles";

export function ToneSection({
  tone,
  disabled,
  onChange,
}: {
  tone: ToneOptions;
  disabled: boolean;
  onChange: (patch: Partial<ToneOptions>) => void;
}) {
  return (
    <>
      <div style={sectionTitle} data-image-tone-title>
        Dodge / burn / sponge
      </div>
      <div style={row}>
        <label htmlFor="pg-image-tone-range">Range</label>
        <select
          id="pg-image-tone-range"
          data-image-tone-range
          value={tone.range}
          disabled={disabled}
          onChange={(e) => onChange({ range: e.target.value as ToneRange })}
        >
          {TONE_RANGES.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </div>
      <div style={row}>
        <label htmlFor="pg-image-tone-exposure">Exposure</label>
        <input
          id="pg-image-tone-exposure"
          data-image-tone-exposure
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={tone.exposure}
          disabled={disabled}
          onChange={(e) => onChange({ exposure: Number(e.target.value) })}
        />
        <span>{Math.round(tone.exposure * 100)}%</span>
      </div>
      <div style={row}>
        <label htmlFor="pg-image-tone-sponge">Sponge</label>
        <select
          id="pg-image-tone-sponge"
          data-image-tone-sponge
          value={tone.saturate ? "saturate" : "desaturate"}
          disabled={disabled}
          onChange={(e) => onChange({ saturate: e.target.value === "saturate" })}
        >
          <option value="desaturate">desaturate</option>
          <option value="saturate">saturate</option>
        </select>
      </div>
      <div style={note}>
        Dodge lightens and burn darkens the chosen tonal range by the exposure;
        the sponge changes saturation at the brush&apos;s flow. All three paint
        with the brush tip and stay inside the selection.
      </div>
    </>
  );
}
