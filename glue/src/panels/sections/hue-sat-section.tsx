// HUE / SATURATION — the full dialog: Master or one of six colour ranges,
// each with hue, saturation and lightness; Colorize. Moves go through
// setParams, so they preview live and edit a bound adjustment layer.

import { HUE_SAT_RANGES, type HueSatParams } from "../../engine";
import type { ImageSession } from "../../session";
import { mono, note, row, sectionTitle } from "./styles";

const slider = (
  label: string,
  id: string,
  min: number,
  max: number,
  value: number,
  onChange: (v: number) => void,
  disabled: boolean,
  unit = "",
) => (
  <div style={row} key={id}>
    <label htmlFor={id}>{label}</label>
    <input
      id={id}
      type="range"
      min={min}
      max={max}
      step={1}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(Number(e.target.value))}
    />
    <span style={mono}>
      {value}
      {unit}
    </span>
  </div>
);

export function HueSatSection({
  session,
  hueSat,
  range,
  onRange,
  disabled,
}: {
  session: ImageSession;
  hueSat: HueSatParams;
  /** -1 = Master, 0..5 = the colour ranges. */
  range: number;
  onRange: (r: number) => void;
  disabled: boolean;
}) {
  const hsl = range < 0 ? hueSat.master : hueSat.ranges[range];
  const set = (k: 0 | 1 | 2, v: number) => {
    const next: HueSatParams = {
      master: [...hueSat.master],
      ranges: hueSat.ranges.map((r) => [...r] as [number, number, number]),
      colorize: { ...hueSat.colorize },
    };
    const target = range < 0 ? next.master : next.ranges[range];
    target[k] = k === 0 ? v : v / 100;
    session.setParams({ hueSat: next });
  };
  const setColorize = (patch: Partial<HueSatParams["colorize"]>) =>
    session.setParams({ hueSat: { ...hueSat, colorize: { ...hueSat.colorize, ...patch } } });
  const off = disabled || hueSat.colorize.on;
  return (
    <>
      <div style={sectionTitle}>Hue / Saturation</div>
      <div style={row}>
        <label htmlFor="pg-image-hs-range">Range</label>
        <select
          id="pg-image-hs-range"
          data-image-hs-range
          value={range}
          disabled={disabled}
          onChange={(e) => onRange(Number(e.target.value))}
        >
          <option value={-1}>Master</option>
          {HUE_SAT_RANGES.map((name, i) => (
            <option key={name} value={i}>
              {name}
            </option>
          ))}
        </select>
      </div>
      {slider("Hue", "pg-image-hs-hue", -180, 180, Math.round(hsl[0]), (v) => set(0, v), off, "°")}
      {slider("Saturation", "pg-image-hs-sat", -100, 100, Math.round(hsl[1] * 100), (v) => set(1, v), off)}
      {slider("Lightness", "pg-image-hs-light", -100, 100, Math.round(hsl[2] * 100), (v) => set(2, v), disabled)}
      <div style={row}>
        <label htmlFor="pg-image-hs-colorize">Colorize</label>
        <input
          id="pg-image-hs-colorize"
          type="checkbox"
          data-image-hs-colorize
          checked={hueSat.colorize.on}
          disabled={disabled}
          onChange={(e) => setColorize({ on: e.target.checked })}
        />
      </div>
      {hueSat.colorize.on &&
        slider("Colorize hue", "pg-image-hs-c-hue", 0, 360, Math.round(hueSat.colorize.hue), (v) => setColorize({ hue: v }), disabled, "°")}
      {hueSat.colorize.on &&
        slider(
          "Colorize saturation",
          "pg-image-hs-c-sat",
          0,
          100,
          Math.round(hueSat.colorize.saturation * 100),
          (v) => setColorize({ saturation: v / 100 }),
          disabled,
        )}
      <div style={note}>
        A colour range is fully affected within 15° of its centre and fades out by
        45°; greys belong to no range. Colorize tints every pixel with one hue.
      </div>
    </>
  );
}
