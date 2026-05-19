// Atomic render components for the lab-notebook visual language.
//
// 1:1 with the rendering_html.py atom functions:
//   stat / stat_strip   -> <Stat /> <StatStrip />
//   section_label       -> <SectionLabel />
//   page_title          -> <PageTitle />
//   score_chip          -> <ScoreChip />
//   signed_bar          -> <SignedBar />
//   mini_bar            -> <MiniBar />
//   corpus_cell         -> <CorpusCell />
//
// CSS class pairing: every `lab-*` className here corresponds to a
// selector in web/src/styles/components.css. Don't rename one without
// the other.

import type { ReactNode } from "react";
import { formatSigned, formatPct } from "./format";

// ----- Stat / StatStrip -------------------------------------------------

export interface StatProps {
  label: string;
  value: string;
  sub?: string;
  tooltip?: string;
}

export function Stat({ label, value, sub, tooltip }: StatProps) {
  return (
    <div>
      <div className="lab-stat-label">
        {label}
        {tooltip && (
          <span
            className="lab-stat-help"
            title={tooltip}
            role="img"
            aria-label={`Help: ${label}`}
          >
            ?
          </span>
        )}
      </div>
      <div className="lab-stat-value">{value}</div>
      {sub && <div className="lab-stat-sub">{sub}</div>}
    </div>
  );
}

export interface StatStripProps {
  cells: StatProps[];
  columns?: number;
}

export function StatStrip({ cells, columns }: StatStripProps) {
  const n = columns ?? cells.length;
  return (
    <div
      className="lab-stat-strip"
      style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }}
    >
      {cells.map((c, i) => (
        <Stat key={i} {...c} />
      ))}
    </div>
  );
}

// ----- Section label + page title --------------------------------------

export interface SectionLabelProps {
  num: string;       // "01", "02", …
  title: string;
  right?: ReactNode;
}

export function SectionLabel({ num, title, right }: SectionLabelProps) {
  return (
    <div className="lab-section">
      <div>
        <span className="lab-section-num">§{num}</span>
        <span className="lab-section-title">{title}</span>
      </div>
      {right && <div className="lab-section-right">{right}</div>}
    </div>
  );
}

export interface PageTitleProps {
  eyebrow: string;
  h1: string;
  /** Subtitle. Pass JSX (e.g. with inline <code> for math) directly. */
  subtitle?: ReactNode;
  rightCaption?: string;
}

export function PageTitle({ eyebrow, h1, subtitle, rightCaption }: PageTitleProps) {
  return (
    <div className="lab-page-title">
      <div>
        <div className="lab-eyebrow">{eyebrow}</div>
        <div className="lab-h1">{h1}</div>
        {subtitle && <div className="lab-subtitle">{subtitle}</div>}
      </div>
      {rightCaption && <div className="lab-corpus-caption">{rightCaption}</div>}
    </div>
  );
}

// ----- Score chip ------------------------------------------------------

export type ChipFmt = "signed" | "pct" | "count";

export interface ScoreChipProps {
  value: number;
  fmt?: ChipFmt;
  decimals?: number;
}

export function ScoreChip({ value, fmt = "signed", decimals }: ScoreChipProps) {
  if (fmt === "signed") {
    const d = decimals ?? 3;
    const color = value >= 0 ? "var(--lab-pos)" : "var(--lab-neg)";
    return (
      <span className="lab-chip" style={{ color }}>
        {formatSigned(value, d)}
      </span>
    );
  }
  if (fmt === "pct") {
    const d = decimals ?? 1;
    return <span className="lab-chip">{formatPct(value, d)}</span>;
  }
  // count
  const n = Math.trunc(value);
  const color = n === 0 ? "var(--lab-ink-muted)" : "var(--lab-ink)";
  return (
    <span className="lab-chip" style={{ color }}>
      {n}
    </span>
  );
}

// ----- Signed bar (bipolar around zero) --------------------------------

export interface SignedBarProps {
  value: number;
  maxValue: number;
  width?: number;
}

export function SignedBar({ value, maxValue, width = 80 }: SignedBarProps) {
  const half = Math.floor(width / 2);
  const mag = maxValue > 0 ? Math.min(Math.abs(value), maxValue) / maxValue : 0;
  const pixels = Math.round(mag * half);
  const fillStyle: React.CSSProperties =
    value >= 0
      ? { left: half, width: pixels, background: "var(--lab-pos)" }
      : { left: half - pixels, width: pixels, background: "var(--lab-neg)" };
  return (
    <div className="lab-bar-wrap" style={{ width }}>
      <div className="lab-bar-mid" style={{ left: half }} />
      <div className="lab-bar-fill" style={fillStyle} />
    </div>
  );
}

// ----- Mini bar (unipolar) ---------------------------------------------

export interface MiniBarProps {
  value: number;
  maxValue: number;
  width?: number;
}

export function MiniBar({ value, maxValue, width = 80 }: MiniBarProps) {
  const mag = maxValue > 0 ? Math.min(value, maxValue) / maxValue : 0;
  const pixels = Math.round(mag * width);
  return (
    <div className="lab-minibar-wrap" style={{ width }}>
      <div className="lab-minibar-fill" style={{ width: pixels }} />
    </div>
  );
}

// ----- Corpus cell -----------------------------------------------------

export interface CorpusCellProps {
  /** Min swap distance to the nearest observed roster. */
  delta: number;
  /** Count of the nearest observed roster (or, when delta=0, of the
   * exact team). */
  count: number;
}

export function CorpusCell({ delta, count }: CorpusCellProps) {
  if (delta === 0) {
    return <span className="lab-badge lab-badge-seen">{count}×</span>;
  }
  const cls = delta === 1 ? "lab-badge-near" : "lab-badge-far";
  return <span className={`lab-badge ${cls}`}>Δ{delta} ({count})</span>;
}
