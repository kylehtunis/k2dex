// Per-feature deep-dive data for the feature detail modal.
//
// Pure helpers over the already-loaded IsingModel + teamCounts — no fetches,
// no new artifacts. The feature-conditioned cousins of the /meta aggregates:
//   meta/couplings.ts  (all structural pairs)  -> featureCouplings (one row)
//   meta/topTeams.ts   (all top rosters)       -> featureCorpusAppearances
//
// Webapp-only; no Python counterpart, so no parity row.

import type { IsingModel, TeamCounts } from "../sampler/types";
import { isStructuralPair } from "../meta/couplings";
import { parseTeamKey } from "./corpus";
import type { TopTeam } from "../meta/topTeams";

export interface FeatureCoupling {
  /** Vocab index of the coupled partner. */
  idx: number;
  jValue: number;
}

export interface FeatureCouplings {
  /** Top positive couplings (synergies), strongest first. */
  synergies: FeatureCoupling[];
  /** Top negative couplings (antisynergies), most negative first. */
  antisynergies: FeatureCoupling[];
}

/** The structural couplings of feature `idx` — row `idx` of J with the same
 * same-species / same-item filter the /meta tables use — split into the top-N
 * synergies (J > 0) and antisynergies (J < 0). */
export function featureCouplings(
  model: IsingModel,
  idx: number,
  topN = 8,
): FeatureCouplings {
  const { V, J } = model;
  const base = idx * V;
  const all: FeatureCoupling[] = [];
  for (let j = 0; j < V; j++) {
    if (j === idx) continue;
    if (!isStructuralPair(model, idx, j)) continue;
    const jValue = J[base + j];
    if (jValue === 0) continue;
    all.push({ idx: j, jValue });
  }
  const synergies = all
    .filter((c) => c.jValue > 0)
    .sort((a, b) => b.jValue - a.jValue)
    .slice(0, topN);
  const antisynergies = all
    .filter((c) => c.jValue < 0)
    .sort((a, b) => a.jValue - b.jValue)
    .slice(0, topN);
  return { synergies, antisynergies };
}

export interface FeatureCorpus {
  /** Top-N most-played exact rosters containing the feature, count desc.
   * Members ordered by descending marginal m̂ (matches meta/topTeams). */
  teams: TopTeam[];
  /** Σ occurrence counts over every roster containing the feature. */
  totalAppearances: number;
  /** Number of distinct rosters containing the feature. */
  nTeams: number;
}

/** The feature-conditioned slice of the corpus: the most-played observed
 * rosters that include feature `idx`, plus aggregate appearance counts.
 * Returns empties when the corpus index is unavailable. */
export function featureCorpusAppearances(
  model: IsingModel,
  teamCounts: TeamCounts | null,
  idx: number,
  topN = 5,
): FeatureCorpus {
  if (teamCounts === null || teamCounts.size === 0) {
    return { teams: [], totalAppearances: 0, nTeams: 0 };
  }
  const { m } = model;
  const rows: { key: string; team: number[]; count: number }[] = [];
  let totalAppearances = 0;
  let nTeams = 0;
  for (const [key, count] of teamCounts) {
    const team = parseTeamKey(key);
    if (!team.includes(idx)) continue;
    nTeams++;
    totalAppearances += count;
    team.sort((a, b) => m[b] - m[a]);
    rows.push({ key, team, count });
  }
  // Count desc, ties broken by key for a stable order (matches topTeams).
  rows.sort((a, b) => b.count - a.count || (a.key < b.key ? -1 : 1));
  return {
    teams: rows.slice(0, topN).map(({ team, count }) => ({ team, count })),
    totalAppearances,
    nTeams,
  };
}

export interface FeatureRanks {
  /** 1-based rank by Bias (h), highest = #1. */
  biasRank: number;
  /** 1-based rank by empirical marginal (m̂), highest = #1. */
  marginalRank: number;
}

/** Where this feature sits among all V features by Bias and by usage. Ties
 * share a rank (counts strictly-greater + 1). */
export function featureRanks(model: IsingModel, idx: number): FeatureRanks {
  const { V, h, m } = model;
  const hi = h[idx];
  const mi = m[idx];
  let biasRank = 1;
  let marginalRank = 1;
  for (let k = 0; k < V; k++) {
    if (k === idx) continue;
    if (h[k] > hi) biasRank++;
    if (m[k] > mi) marginalRank++;
  }
  return { biasRank, marginalRank };
}
