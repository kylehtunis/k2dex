// Species-level projection of the feature-level co-occurrence baseline.
//
// The article talks in species ("Incineroar is a good partner"), but the
// co-occurrence engine (sampler/cooccurrence.ts) and the model both operate on
// (species, item) features. These helpers fold the feature-level matrix down to
// a species×species view: co-occurrence counts, appearance counts, teammate
// percentages, and a species-level greedy fill. All derived from the same
// corpus the model is fit on, so the comparison is apples-to-apples.

import type { IsingModel } from "../sampler/types";
import type { Cooccurrence } from "../sampler/cooccurrence";

export interface SpeciesCooccurrence {
  /** Number of distinct species (sites). */
  S: number;
  /** S×S flat row-major raw co-occurrence counts between species, symmetric,
   * zero diagonal. `siteC[a*S + b]` = teams containing both species a and b. */
  siteC: Float64Array;
  /** Per-species appearance count (teams containing that species). */
  siteAppear: Float64Array;
  /** Total teams in the corpus. */
  nTeams: number;
}

/** Fold the feature-level {@link Cooccurrence} down to species (site) level.
 * A species appears at most once per team, so its features are mutually
 * exclusive within a team — summing feature marginals/counts over a site gives
 * the species-level count exactly. */
export function aggregateToSpecies(
  model: IsingModel,
  cooc: Cooccurrence,
): SpeciesCooccurrence {
  const { V, siteOf } = model;
  const S = model.sites.length;
  const { C, m, nTeams } = cooc;

  const siteC = new Float64Array(S * S);
  for (let i = 0; i < V; i++) {
    const si = siteOf[i];
    const rowI = i * V;
    const rowSi = si * S;
    for (let j = 0; j < V; j++) {
      const cij = C[rowI + j];
      if (cij !== 0) siteC[rowSi + siteOf[j]] += cij;
    }
  }
  // Zero the diagonal (features of the same site never co-occur, but guard
  // against FP dust).
  for (let a = 0; a < S; a++) siteC[a * S + a] = 0;

  const siteAppear = new Float64Array(S);
  for (let i = 0; i < V; i++) siteAppear[siteOf[i]] += m[i] * nTeams;

  return { S, siteC, siteAppear, nTeams };
}

export interface Teammate {
  site: number;
  /** Raw co-occurrence count with the anchor species. */
  count: number;
  /** P(this species on team | anchor on team) — Smogon's "Teammates %". */
  pct: number;
}

/** Top teammates of `anchorSite` by co-occurrence, sorted descending. The
 * percentage is the conditional appearance rate, matching how Smogon /
 * Pikalytics present "Teammates". */
export function speciesTeammates(
  sc: SpeciesCooccurrence,
  anchorSite: number,
  topN: number,
): Teammate[] {
  const { S, siteC, siteAppear } = sc;
  const denom = siteAppear[anchorSite] || 1;
  const rowA = anchorSite * S;
  const out: Teammate[] = [];
  for (let b = 0; b < S; b++) {
    if (b === anchorSite) continue;
    const count = siteC[rowA + b];
    if (count <= 0) continue;
    out.push({ site: b, count, pct: count / denom });
  }
  out.sort((x, y) => y.count - x.count);
  return out.slice(0, topN);
}

/** Species-level co-occurrence score against a set of held-in species:
 * `score[b] = sum(siteC[a, b] for a in heldInSites)`. The species-level
 * counterpart of scoreCooccurrence. */
function speciesCoocScore(
  sc: SpeciesCooccurrence,
  heldInSites: readonly number[],
): Float64Array {
  const { S, siteC } = sc;
  const scores = new Float64Array(S);
  for (let b = 0; b < S; b++) {
    let s = 0;
    for (const a of heldInSites) s += siteC[a * S + b];
    scores[b] = s;
  }
  return scores;
}

/** Species-level greedy team fill by co-occurrence — the naive teambuilder,
 * one rung up from features: repeatedly add the highest-scoring species,
 * rescoring against the growing team, until `teamSize` species are chosen.
 * Species are unique by construction, so the only constraint is no repeats. */
export function speciesCoocGreedy(
  sc: SpeciesCooccurrence,
  heldInSites: readonly number[],
  excludedSites: ReadonlySet<number>,
  teamSize: number,
): number[] {
  const team = [...heldInSites];
  const inTeam = new Set(team);
  while (team.length < teamSize) {
    const scores = speciesCoocScore(sc, team);
    let bestSite = -1;
    let bestScore = -Infinity;
    for (let b = 0; b < sc.S; b++) {
      if (inTeam.has(b) || excludedSites.has(b)) continue;
      if (scores[b] > bestScore) {
        bestScore = scores[b];
        bestSite = b;
      }
    }
    if (bestSite < 0) break;
    team.push(bestSite);
    inTeam.add(bestSite);
  }
  return team;
}

/** The highest-marginal feature of a site (its most-used item build). Used to
 * turn a species selection into a concrete feature for scoring, and to seed
 * the model's mean-field run at a site pin. */
export function topFeatureOfSite(model: IsingModel, site: number): number {
  const feats = model.siteFeatures[site];
  let best = feats[0];
  let bestM = model.m[best];
  for (const f of feats) {
    if (model.m[f] > bestM) {
      bestM = model.m[f];
      best = f;
    }
  }
  return best;
}
