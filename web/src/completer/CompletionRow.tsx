// One completion-table row: the four free mons rendered as a stack of
// CompMonCell, plus the standard observables columns.
//
// Pure presentation — caller provides numeric values already computed.

import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { CompMonCell } from "../render/cells";
import { CorpusCell } from "../render/atoms";
import { ScoreChip } from "../render/atoms";
import type { IsingModel } from "../sampler/types";

export interface CompletionRowProps {
  /** Order doesn't matter; row sorts them top-down by vocab order. */
  freeIdxs: readonly number[];
  /** Full team (fixed + free), used for the "Analyze" link. */
  fullTeam: readonly number[];
  scoreAdj: number;
  scoreRaw: number;
  coherence: number;
  corpus: { delta: number; count: number } | null;
  /** Optional %-share for top-K distribution mode (PT). Omitted in fast path. */
  freqPct?: number;
  isTopRow?: boolean;
  rank?: number;
  model: IsingModel;
}

export function CompletionRow({
  freeIdxs,
  fullTeam,
  scoreAdj,
  scoreRaw,
  coherence,
  corpus,
  freqPct,
  isTopRow,
  rank,
  model,
}: CompletionRowProps) {
  const sortedFree = [...freeIdxs].sort((a, b) => a - b);
  const analyzeUrl = `/analysis?team=${[...fullTeam].sort((a, b) => a - b).join(",")}`;
  return (
    <tr className={isTopRow ? "top-row" : undefined}>
      {rank !== undefined && <td className="rank">{rank}</td>}
      <td>
        <div className="lab-comp-pair">
          {sortedFree.map((i) => (
            <CompMonCell key={i} name={model.vocab[i]} />
          ))}
        </div>
      </td>
      <td>
        <Link to={analyzeUrl} className="lab-analyze-btn">
          Send to Analysis
        </Link>
      </td>
      {freqPct !== undefined && (
        <td className="num">
          <div className="lab-comp-freq">
            <div className="lab-comp-freq-pct">{freqPct.toFixed(1)}%</div>
          </div>
        </td>
      )}
      <td className="num">
        <ScoreChip value={scoreAdj} />
      </td>
      <td className="num">
        <ScoreChip value={scoreRaw} />
      </td>
      <td className="num">
        <ScoreChip value={coherence} />
      </td>
      <td className="num">
        {corpus !== null ? (
          <CorpusCell delta={corpus.delta} count={corpus.count} />
        ) : (
          <span style={{ color: "var(--lab-ink-muted)" }}>—</span>
        )}
      </td>
    </tr>
  );
}

export function CompletionTable({
  children,
  includeFreq,
  includeRank,
}: {
  children: ReactNode;
  includeFreq?: boolean;
  includeRank?: boolean;
}) {
  return (
    <table className="lab-comp-table">
      <thead>
        <tr>
          {includeRank && <th>#</th>}
          <th>Completion</th>
          <th></th>
          {includeFreq && <th className="num">Freq</th>}
          <th className="num">Score (adj)</th>
          <th className="num">Score (raw)</th>
          <th className="num">Coherence</th>
          <th className="num">Corpus</th>
        </tr>
      </thead>
      <tbody>{children}</tbody>
    </table>
  );
}
