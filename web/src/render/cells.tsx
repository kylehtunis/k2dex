// Composed cells. 1:1 with the composed helpers in rendering_html.py:
//   slot_card / slot_card_empty / slot_strip -> <SlotCard /> etc.
//   excluded_row    -> <ExcludedRow />
//   comp_mon_cell   -> <CompMonCell />
//   pair_cell       -> <PairCell />
//   swap_cell       -> <SwapCell />
//   inline_mon      -> <InlineMon />
//   team_mini_strip -> <TeamMiniStrip />

import { extractItem, extractSpecies } from "./format";
import { SpriteBox } from "./Sprite";

// ----- Slot card / strip ----------------------------------------------

export interface SlotCardProps {
  name: string;
  indicator?: string;
}

export function SlotCard({ name, indicator = "s=1" }: SlotCardProps) {
  const species = extractSpecies(name);
  const item = extractItem(name);
  return (
    <div className="lab-slot">
      <div className="lab-slot-indicator">{indicator}</div>
      <SpriteBox name={name} size={72} className="lab-slot-sprite" />
      <div className="lab-slot-name">{species}</div>
      {item && (
        <>
          <hr className="lab-slot-divider" />
          <div className="lab-slot-item">@ {item}</div>
        </>
      )}
    </div>
  );
}

export interface SlotCardEmptyProps {
  ordinal: number;
}

export function SlotCardEmpty({ ordinal }: SlotCardEmptyProps) {
  return (
    <div className="lab-slot-empty">
      <div className="lab-slot-empty-label">free slot</div>
      <div className="lab-slot-empty-ord">·{ordinal}·</div>
    </div>
  );
}

export interface SlotStripProps {
  picked: readonly string[];
  teamSize?: number;
}

export function SlotStrip({ picked, teamSize = 6 }: SlotStripProps) {
  const empties = [];
  for (let i = picked.length + 1; i <= teamSize; i++) empties.push(i);
  return (
    <div className="lab-slot-strip">
      {picked.map((name) => (
        <SlotCard key={name} name={name} />
      ))}
      {empties.map((i) => (
        <SlotCardEmpty key={`empty-${i}`} ordinal={i} />
      ))}
    </div>
  );
}

// ----- Excluded row ----------------------------------------------------

export interface ExcludedRowProps {
  names: readonly string[];
  note?: string;
}

export function ExcludedRow({ names, note }: ExcludedRowProps) {
  if (names.length === 0 && !note) return null;
  return (
    <div className="lab-excluded-row">
      {names.length > 0 && <span className="lab-excluded-label">excluded</span>}
      {names.map((n) => (
        <span key={n} className="lab-excluded-tag">
          {extractSpecies(n)}
        </span>
      ))}
      {note && <span className="lab-excluded-note">— {note}</span>}
    </div>
  );
}

// ----- Comp mon / inline mon / pair / swap ----------------------------

export interface CompMonCellProps {
  name: string;
}

export function CompMonCell({ name }: CompMonCellProps) {
  const species = extractSpecies(name);
  const item = extractItem(name);
  return (
    <div className="lab-comp-mon">
      <SpriteBox name={name} size={36} />
      <div>
        <div className="lab-comp-mon-name">{species}</div>
        {item && <div className="lab-comp-mon-item">@ {item}</div>}
      </div>
    </div>
  );
}

export interface InlineMonProps {
  name: string;
  size?: number;
}

export function InlineMon({ name, size = 28 }: InlineMonProps) {
  const species = extractSpecies(name);
  const item = extractItem(name);
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <SpriteBox name={name} size={size} />
      <span className="lab-comp-mon-name">{species}</span>
      {item && (
        <span className="lab-comp-mon-item" style={{ marginLeft: 6 }}>
          @ {item}
        </span>
      )}
    </span>
  );
}

export interface PairCellProps {
  nameA: string;
  nameB: string;
}

export function PairCell({ nameA, nameB }: PairCellProps) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <InlineMon name={nameA} />
      <span
        style={{
          fontFamily: "var(--lab-font-mono)",
          color: "var(--lab-ink-muted)",
        }}
      >
        ×
      </span>
      <InlineMon name={nameB} />
    </div>
  );
}

export interface SwapCellProps {
  nameOut: string;
  nameIn: string;
}

export function SwapCell({ nameOut, nameIn }: SwapCellProps) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <InlineMon name={nameOut} />
      <span
        style={{
          fontFamily: "var(--lab-font-mono)",
          fontSize: 14,
          color: "var(--lab-accent)",
        }}
      >
        →
      </span>
      <InlineMon name={nameIn} />
    </div>
  );
}

export interface TeamMiniStripProps {
  names: readonly string[];
  size?: number;
}

export function TeamMiniStrip({ names, size = 24 }: TeamMiniStripProps) {
  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
      {names.map((n) => (
        <SpriteBox key={n} name={n} size={size} />
      ))}
    </div>
  );
}
