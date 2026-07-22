// One completion rendered as a stacked card: the full 6-mon team (pinned +
// completer-filled) gridded across the top, then the observables and action
// buttons underneath. Pure presentation — caller provides numbers already
// computed.

import { type ReactNode, useCallback, useState } from "react";
import { Link } from "react-router-dom";
import { extractAbility, extractItem, extractSpecies, buildPartialPaste } from "../render/format";
import { CorpusCell, ScoreChip } from "../render/atoms";
import { SpriteBox } from "../render/Sprite";
import { useFeatureModal } from "../components/FeatureModalContext";
import type { IsingModel } from "../sampler/types";
import { speciesToSlug } from "../render/sprite-url";
import { encodeCore } from "../render/shareLink";

/** One team member tile: sprite on top, name + item below. Clickable into the
 * feature modal wherever a provider is mounted. Pinned mons (locked by the
 * user) carry a subtle marker so completed picks read as the model's work. */
function CompMonTile({
  name,
  pinned,
  hideItem,
  hideAbility,
}: {
  name: string;
  pinned: boolean;
  hideItem: boolean;
  hideAbility: boolean;
}) {
  const species = extractSpecies(name);
  const item = hideItem ? null : extractItem(name);
  const ability = hideAbility ? null : extractAbility(name);
  const fm = useFeatureModal();
  const cls = `lab-comp-tile${pinned ? " is-pinned" : ""}`;
  const inner = (
    <>
      {pinned && <div className="lab-comp-tile-pin">pinned</div>}
      <SpriteBox name={name} size={52} className="lab-comp-tile-sprite" />
      <div className="lab-comp-tile-name">{species}</div>
      {item && <div className="lab-comp-tile-item">@ {item}</div>}
      {ability && <div className="lab-comp-tile-ability">{ability}</div>}
    </>
  );
  if (fm) {
    return (
      <button
        type="button"
        className={`${cls} lab-feature-link`}
        onClick={() => fm.openFeature(name)}
      >
        {inner}
      </button>
    );
  }
  return <div className={cls}>{inner}</div>;
}

export interface CompletionCardProps {
  /** Full team (pinned + free). Rendered as all six tiles. */
  fullTeam: readonly number[];
  /** The completer-filled members (order-agnostic); the rest are pinned. */
  freeIdxs: readonly number[];
  score: number;
  /** Hover title for the Score (its corpus percentile), when available. */
  scoreTitle?: string | null;
  coherence: number;
  /** Hover title for the Coherence, when available. */
  coherenceTitle?: string | null;
  corpus: { delta: number; count: number } | null;
  /** Optional %-share for top-K distribution mode (PT). Omitted in fast path. */
  freqPct?: number;
  isTopRow?: boolean;
  rank?: number;
  model: IsingModel;
  /** Species-only mode: hide the marginalized-out item on every tile. */
  hideItems?: boolean;
  /** Ability track deactivated: hide the marginalized-out ability on every tile. */
  hideAbility?: boolean;
}

export function CompletionCard({
  fullTeam,
  freeIdxs,
  score,
  scoreTitle = null,
  coherence,
  coherenceTitle = null,
  corpus,
  freqPct,
  isTopRow,
  rank,
  model,
  hideItems = false,
  hideAbility = false,
}: CompletionCardProps) {
  // Pinned tiles first (in vocab order), then the completer-filled ones.
  const freeSet = new Set(freeIdxs);
  const pinned = fullTeam.filter((i) => !freeSet.has(i)).sort((a, b) => a - b);
  const free = [...freeIdxs].sort((a, b) => a - b);
  const ordered = [...pinned, ...free];

  const analyzeUrl = `/analysis?t=${encodeCore(model.id, fullTeam, model)}`;

  const [copied, setCopied] = useState(false);
  const handleCopyPaste = useCallback(() => {
    const paste = buildPartialPaste(fullTeam, model.vocab, speciesToSlug);
    navigator.clipboard.writeText(paste).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [fullTeam, model.vocab]);

  const hasHead = rank !== undefined || freqPct !== undefined;

  return (
    <div className={`lab-comp-card${isTopRow ? " is-top" : ""}`}>
      {hasHead && (
        <div className="lab-comp-card-head">
          {rank !== undefined && (
            <div className="lab-comp-card-rank">#{rank}</div>
          )}
          {freqPct !== undefined && (
            <div className="lab-comp-card-freq">
              {freqPct.toFixed(1)}
              <span className="lab-comp-card-freq-unit">%</span>
            </div>
          )}
        </div>
      )}

      <div className="lab-comp-mon-grid">
        {ordered.map((i) => (
          <CompMonTile
            key={i}
            name={model.vocab[i]}
            pinned={freeSet.size > 0 && !freeSet.has(i)}
            hideItem={hideItems}
            hideAbility={hideAbility}
          />
        ))}
      </div>

      <div className="lab-comp-card-stats">
        <div className="lab-comp-stat">
          <span className="lab-comp-stat-label">Score</span>
          <ScoreChip value={score} title={scoreTitle ?? undefined} />
        </div>
        <div className="lab-comp-stat">
          <span className="lab-comp-stat-label">Coherence</span>
          <ScoreChip value={coherence} title={coherenceTitle ?? undefined} />
        </div>
        <div className="lab-comp-stat">
          <span className="lab-comp-stat-label">Corpus</span>
          {corpus !== null ? (
            <CorpusCell delta={corpus.delta} count={corpus.count} />
          ) : (
            <span style={{ color: "var(--lab-ink-muted)" }}>—</span>
          )}
        </div>
      </div>

      <div className="lab-comp-card-actions">
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
    </div>
  );
}

export function CompletionList({ children }: { children: ReactNode }) {
  return <div className="lab-comp-cards">{children}</div>;
}
