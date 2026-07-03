// One completion-table row: the four free mons rendered as a stack of
// CompMonCell, plus the standard observables columns.
//
// Pure presentation — caller provides numeric values already computed.

import { type ReactNode, useCallback, useState } from "react";
import { Link } from "react-router-dom";
import { CompMonCell } from "../render/cells";
import { extractSpecies } from "../render/format";
import { CorpusCell } from "../render/atoms";
import { ScoreChip } from "../render/atoms";
import type { IsingModel } from "../sampler/types";
import { speciesToSlug } from "../render/sprite-url";
import { buildPartialPaste } from "../render/format";
import { encodeCore } from "../render/shareLink";
import { usePageState } from "../state/PageStateContext";

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
  /** Species-only mode: render each mon as its species, hiding the item that
   * was marginalized out (the underlying team is still a real completion). */
  hideItems?: boolean;
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
  hideItems = false,
}: CompletionRowProps) {
  const sortedFree = [...freeIdxs].sort((a, b) => a - b);
  const { completer } = usePageState();
  const analyzeUrl = `/analysis?t=${encodeCore(
    model.id,
    completer.fieldWeight,
    fullTeam,
    model,
  )}`;

  const [copied, setCopied] = useState(false);
  const handleCopyPaste = useCallback(() => {
    const paste = buildPartialPaste(fullTeam, model.vocab, speciesToSlug);
    navigator.clipboard.writeText(paste).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [fullTeam, model.vocab]);

  return (
    <tr className={isTopRow ? "top-row" : undefined}>
      {rank !== undefined && <td className="rank">{rank}</td>}
      <td className="mons">
        <div className="lab-comp-pair">
          {sortedFree.map((i) => (
            <CompMonCell
              key={i}
              name={hideItems ? extractSpecies(model.vocab[i]) : model.vocab[i]}
            />
          ))}
        </div>
      </td>
      <td className="actions">
        <div className="lab-comp-actions">
          <Link to={analyzeUrl} className="lab-analyze-btn">
            Send to Analysis
          </Link>
          <button
            type="button"
            className="lab-analyze-btn lab-copy-paste-btn"
            onClick={handleCopyPaste}
          >
            {copied ? "Copied!" : "Copy pokepaste"}
          </button>
        </div>
      </td>
      {freqPct !== undefined && (
        <td className="num" data-label="freq">
          <div className="lab-comp-freq">
            <div className="lab-comp-freq-pct">{freqPct.toFixed(1)}%</div>
          </div>
        </td>
      )}
      <td className="num" data-label="score adj">
        <ScoreChip value={scoreAdj} />
      </td>
      <td className="num" data-label="score raw">
        <ScoreChip value={scoreRaw} />
      </td>
      <td className="num" data-label="coherence">
        <ScoreChip value={coherence} />
      </td>
      <td className="num" data-label="corpus">
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
    <table className="lab-comp-table lab-table-cards">
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
