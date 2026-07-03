// Composed cells. 1:1 with the composed helpers in rendering_html.py:
//   slot_card / slot_card_empty / slot_strip -> <SlotCard /> etc.
//   excluded_row    -> <ExcludedRow />
//   comp_mon_cell   -> <CompMonCell />
//   pair_cell       -> <PairCell />
//   swap_cell       -> <SwapCell />
//   inline_mon      -> <InlineMon />
//   team_mini_strip -> <TeamMiniStrip />

import { type KeyboardEvent } from "react";
import { extractItem, extractSpecies } from "./format";
import { SpriteBox } from "./Sprite";
import { useFeatureModal } from "../components/FeatureModalContext";

// ----- Slot card / strip ----------------------------------------------

export interface SlotCardProps {
  name: string;
  indicator?: string;
}

export function SlotCard({ name, indicator = "s=1" }: SlotCardProps) {
  const species = extractSpecies(name);
  const item = extractItem(name);
  const fm = useFeatureModal();
  const wrapProps = fm
    ? {
        className: "lab-slot lab-feature-clickable",
        role: "button" as const,
        tabIndex: 0,
        onClick: () => fm.openFeature(name),
        onKeyDown: (e: KeyboardEvent) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fm.openFeature(name);
          }
        },
      }
    : { className: "lab-slot" };
  return (
    <div {...wrapProps}>
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

// ----- Included row ---------------------------------------------------

export interface IncludedRowProps {
  names: readonly string[];
  note?: string;
}

export function IncludedRow({ names, note }: IncludedRowProps) {
  if (names.length === 0 && !note) return null;
  return (
    <div className="lab-excluded-row">
      {names.length > 0 && <span className="lab-excluded-label">only</span>}
      {names.map((n) => (
        <span key={n} className="lab-included-tag">
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
  const fm = useFeatureModal();
  const inner = (
    <>
      <SpriteBox name={name} size={36} />
      <div>
        <div className="lab-comp-mon-name">{species}</div>
        {item && <div className="lab-comp-mon-item">@ {item}</div>}
      </div>
    </>
  );
  if (fm) {
    return (
      <button
        type="button"
        className="lab-comp-mon lab-feature-link"
        onClick={() => fm.openFeature(name)}
      >
        {inner}
      </button>
    );
  }
  return <div className="lab-comp-mon">{inner}</div>;
}

export interface InlineMonProps {
  name: string;
  size?: number;
}

export function InlineMon({ name, size = 32 }: InlineMonProps) {
  const species = extractSpecies(name);
  const item = extractItem(name);
  const fm = useFeatureModal();
  const style = { display: "inline-flex", alignItems: "center", gap: 6 } as const;
  const inner = (
    <>
      <SpriteBox name={name} size={size} />
      <span className="lab-comp-mon-name">{species}</span>
      {item && (
        <span className="lab-comp-mon-item" style={{ marginLeft: 6 }}>
          @ {item}
        </span>
      )}
    </>
  );
  if (fm) {
    return (
      <button
        type="button"
        className="lab-feature-link"
        style={style}
        onClick={() => fm.openFeature(name)}
      >
        {inner}
      </button>
    );
  }
  return <span style={style}>{inner}</span>;
}

export interface PairCellProps {
  nameA: string;
  nameB: string;
}

export function PairCell({ nameA, nameB }: PairCellProps) {
  return (
    <div className="lab-pair-cell">
      <InlineMon name={nameA} />
      <span className="lab-pair-sep">×</span>
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
    <div className="lab-swap-cell">
      <InlineMon name={nameOut} />
      <span className="lab-swap-arrow">→</span>
      <InlineMon name={nameIn} />
    </div>
  );
}

export interface TeamMiniStripProps {
  names: readonly string[];
  size?: number;
  /** Opt-in: wrap each sprite in a button that opens the feature modal. Off by
   * default so bare strips stay hover-only; only callers that want per-mon
   * drill-in (e.g. the /meta top-teams table) pass this. */
  interactive?: boolean;
}

export function TeamMiniStrip({
  names,
  size = 24,
  interactive = false,
}: TeamMiniStripProps) {
  const fm = useFeatureModal();
  const clickable = interactive && fm;
  return (
    <div className="lab-mini-strip">
      {names.map((n) =>
        clickable ? (
          <button
            key={n}
            type="button"
            className="lab-feature-link"
            onClick={() => fm.openFeature(n)}
          >
            <SpriteBox name={n} size={size} />
          </button>
        ) : (
          <SpriteBox key={n} name={n} size={size} />
        ),
      )}
    </div>
  );
}
