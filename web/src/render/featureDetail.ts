// Per-feature deep-dive data for the feature detail modal.
//
// Pure helpers over the already-loaded IsingModel + teamCounts — no fetches,
// no new artifacts. The feature-conditioned cousins of the /meta aggregates:
//   meta/couplings.ts  (all structural pairs)  -> featureCouplings (one row)
//   meta/topTeams.ts   (all top rosters)       -> featureCorpusAppearances
//
// Webapp-only; no Python counterpart, so no parity row.

import type { IsingModel, SpeciesGraph, TeamCounts } from "../sampler/types";
import { parseTeamKey } from "./corpus";
import type { TopTeam } from "../meta/topTeams";

export interface SpeciesCoupling {
  /** Partner species name. */
  species: string;
  /** Species-level synergy (grand mean of the J block). */
  synergy: number;
}

export interface SpeciesCouplings {
  synergies: SpeciesCoupling[];
  antisynergies: SpeciesCoupling[];
}

/** Species-level couplings for species at `site`, ranked by synergy.
 * Uses the precomputed species graph. */
export function speciesCouplings(
  model: IsingModel,
  graph: SpeciesGraph,
  site: number,
  topN = 10,
): SpeciesCouplings {
  const mySpecies = model.sites[site];
  const S = graph.species.length;
  const myGraphIdx = graph.indexOf.get(mySpecies);
  if (myGraphIdx === undefined) return { synergies: [], antisynergies: [] };

  const all: SpeciesCoupling[] = [];
  for (let gi = 0; gi < S; gi++) {
    if (gi === myGraphIdx) continue;
    const partnerSpecies = graph.species[gi];
    const synergy = graph.synergy[myGraphIdx * S + gi];
    all.push({ species: partnerSpecies, synergy });
  }

  const synergies = all
    .filter((c) => c.synergy > 0)
    .sort((a, b) => b.synergy - a.synergy)
    .slice(0, topN);
  const antisynergies = all
    .filter((c) => c.synergy < 0)
    .sort((a, b) => a.synergy - b.synergy)
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

/** Site-level corpus appearances: rosters containing *any* feature of
 * the given site. Deduplicates rosters that match multiple features. */
export function siteCorpusAppearances(
  model: IsingModel,
  teamCounts: TeamCounts | null,
  site: number,
  topN = 5,
): FeatureCorpus {
  if (teamCounts === null || teamCounts.size === 0) {
    return { teams: [], totalAppearances: 0, nTeams: 0 };
  }
  const { m, siteFeatures } = model;
  const feats = new Set(siteFeatures[site]);
  const rows: { key: string; team: number[]; count: number }[] = [];
  let totalAppearances = 0;
  let nTeams = 0;
  for (const [key, count] of teamCounts) {
    const team = parseTeamKey(key);
    if (!team.some((f) => feats.has(f))) continue;
    nTeams++;
    totalAppearances += count;
    team.sort((a, b) => m[b] - m[a]);
    rows.push({ key, team, count });
  }
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
