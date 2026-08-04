// Corpus-referenced percentiles: where a team's Score and Coherence fall
// among every roster observed in the tournament corpus. The reference
// distribution is the empirical corpus (team_counts.json), not the
// model's sampling distribution, so it is exact and cheap.
//
// Webapp-only (no Streamlit/Python mirror, no parity row).

import type { IsingModel, TeamCounts } from "../sampler/types";
import { parseTeamKey } from "./corpus";
import { teamObservables } from "./observables";

/** One observable's empirical corpus distribution. */
export interface WeightedDistribution {
  /** Distinct-roster values, ascending. */
  values: number[];
  /** Occurrence weight aligned with `values`. */
  weights: number[];
  totalWeight: number;
}

export interface CorpusScoreIndex {
  score: WeightedDistribution;
  coherence: WeightedDistribution;
}

function toDistribution(
  entries: Array<{ value: number; w: number }>,
): WeightedDistribution {
  entries.sort((a, b) => a.value - b.value);
  const values: number[] = [];
  const weights: number[] = [];
  let totalWeight = 0;
  for (const e of entries) {
    values.push(e.value);
    weights.push(e.w);
    totalWeight += e.w;
  }
  return { values, weights, totalWeight };
}

/** Score every observed roster once (fieldWeight = 1). Build per model +
 * corpus and memoize; a few thousand teams is a few ms. Returns null when
 * the corpus is unavailable. */
export function buildCorpusScoreIndex(
  model: IsingModel,
  teamCounts: TeamCounts | null,
): CorpusScoreIndex | null {
  if (teamCounts === null || teamCounts.size === 0) return null;
  const scores: Array<{ value: number; w: number }> = [];
  const coherences: Array<{ value: number; w: number }> = [];
  for (const [key, count] of teamCounts) {
    const obs = teamObservables(model, parseTeamKey(key));
    scores.push({ value: obs.scoreRaw, w: count });
    coherences.push({ value: obs.coherence, w: count });
  }
  return {
    score: toDistribution(scores),
    coherence: toDistribution(coherences),
  };
}

/** Weighted percentile rank (0–100) of `value` among the observed teams;
 * exact ties count half (standard percentile-rank convention). */
export function percentileRank(
  dist: WeightedDistribution,
  value: number,
): number {
  let below = 0;
  let equal = 0;
  for (let i = 0; i < dist.values.length; i++) {
    if (dist.values[i] < value) below += dist.weights[i];
    else if (dist.values[i] === value) equal += dist.weights[i];
    else break; // values are ascending
  }
  return (100 * (below + 0.5 * equal)) / dist.totalWeight;
}

/** 84 -> "84th", 21 -> "21st", 12 -> "12th". */
export function ordinal(n: number): string {
  const mod100 = n % 100;
  if (mod100 >= 11 && mod100 <= 13) return `${n}th`;
  switch (n % 10) {
    case 1:
      return `${n}st`;
    case 2:
      return `${n}nd`;
    case 3:
      return `${n}rd`;
    default:
      return `${n}th`;
  }
}

/** Rounded ordinal percentile label, e.g. "84th". */
export function percentileLabel(
  dist: WeightedDistribution,
  value: number,
): string {
  return ordinal(Math.round(percentileRank(dist, value)));
}

/** Hover-tooltip sentence, e.g. "84th percentile of tournament teams". */
export function percentileTitle(
  dist: WeightedDistribution,
  value: number,
): string {
  return `${percentileLabel(dist, value)} percentile`;
}
