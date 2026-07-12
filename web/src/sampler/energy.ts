// Shared energy + constraint primitives.
//
// Mirrors the helpers at the top of sampling.py:
//   team_energy, build_constraint_sets, swap_violates_uniqueness,
//   initialize_state.
//
// Uniqueness machinery is dimension-agnostic: a team never holds two of the
// same site (species), and never two features sharing a value on a track
// flagged `unique` (e.g. the item track). Species-only models carry no tracks,
// so only the always-on site constraint applies (trivially satisfied since each
// feature is its own site).

import type { IsingModel } from "./types";
import type { Rng } from "./rng";

/** Raw Ising energy H(s) = -h·s - 0.5 s'Js. Lower = more probable.
 * `stateF` is the 0/1 float view of the team state. */
export function teamEnergy(
  stateF: Float64Array,
  J: Float64Array,
  h: Float64Array,
  V: number,
): number {
  // -h·s
  let hDot = 0;
  // 0.5 * s'Js. Iterate over the team membership; off-team entries are 0.
  // For simplicity, do the full V×V double loop with multiplications.
  // Hot path uses the delta math instead; this is only called at init.
  let quad = 0;
  for (let i = 0; i < V; i++) {
    hDot += h[i] * stateF[i];
    if (stateF[i] === 0) continue;
    const rowI = i * V;
    for (let j = 0; j < V; j++) {
      quad += J[rowI + j] * stateF[i] * stateF[j];
    }
  }
  return -hDot - 0.5 * quad;
}

export interface ConstraintSets {
  /** Site (species) indices occupied by the fixed slots. Sites are always
   * unique across a team. */
  usedSites: Set<number>;
  /** Values occupied by the fixed slots on each track, aligned to
   * `model.tracks` by index. Only tracks flagged `unique` are populated (and
   * checked); non-unique tracks keep an empty, unused set. */
  usedTrackValues: Set<string>[];
}

/** Sites + per-track unique values occupied by `fixed`. */
export function buildConstraintSets(
  fixed: Iterable<number>,
  model: IsingModel,
): ConstraintSets {
  const usedSites = new Set<number>();
  const usedTrackValues = model.tracks.map(() => new Set<string>());
  for (const i of fixed) {
    usedSites.add(model.siteOf[i]);
    for (let t = 0; t < model.tracks.length; t++) {
      if (!model.tracks[t].unique) continue;
      const v = model.trackValues[i][t];
      if (v !== null) usedTrackValues[t].add(v);
    }
  }
  return { usedSites, usedTrackValues };
}

/** True iff swapping `iIn` into position `outK` of `onNf` would create
 * a duplicate site (species) or a duplicate value on any unique track
 * (counting both the fixed and free slots of the team). */
export function swapViolatesUniqueness(
  iIn: number,
  outK: number,
  onNf: readonly number[],
  constraints: ConstraintSets,
  model: IsingModel,
): boolean {
  const site = model.siteOf[iIn];
  if (constraints.usedSites.has(site)) return true;
  for (let k = 0; k < onNf.length; k++) {
    if (k === outK) continue;
    if (model.siteOf[onNf[k]] === site) return true;
  }
  for (let t = 0; t < model.tracks.length; t++) {
    if (!model.tracks[t].unique) continue;
    const target = model.trackValues[iIn][t];
    if (target === null) continue;
    if (constraints.usedTrackValues[t].has(target)) return true;
    for (let k = 0; k < onNf.length; k++) {
      if (k === outK) continue;
      if (model.trackValues[onNf[k]][t] === target) return true;
    }
  }
  return false;
}

/** Greedy uniqueness-respecting sample of `nToFill` indices from
 * `available`. Returns null if no valid completion exists after
 * `maxAttempts` random restarts (e.g., user pinned conflicting items). */
export function initializeState(
  available: readonly number[],
  nToFill: number,
  constraints: ConstraintSets,
  model: IsingModel,
  rng: Rng,
  maxAttempts = 100,
): number[] | null {
  const nTracks = model.tracks.length;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const chosen: number[] = [];
    const usedSites = new Set(constraints.usedSites);
    const usedTrack = constraints.usedTrackValues.map((s) => new Set(s));
    const shuffled = rng.permutation(available);
    for (const idx of shuffled) {
      if (chosen.length === nToFill) break;
      if (usedSites.has(model.siteOf[idx])) continue;
      let conflict = false;
      for (let t = 0; t < nTracks; t++) {
        if (!model.tracks[t].unique) continue;
        const v = model.trackValues[idx][t];
        if (v !== null && usedTrack[t].has(v)) {
          conflict = true;
          break;
        }
      }
      if (conflict) continue;
      chosen.push(idx);
      usedSites.add(model.siteOf[idx]);
      for (let t = 0; t < nTracks; t++) {
        if (!model.tracks[t].unique) continue;
        const v = model.trackValues[idx][t];
        if (v !== null) usedTrack[t].add(v);
      }
    }
    if (chosen.length === nToFill) return chosen;
  }
  return null;
}

/** Resolve site-level pins (species fixed, track values free) to concrete seed
 * features: for each site, its highest-marginal feature that is available and
 * doesn't collide with the feature pins / earlier seeds on any unique track.
 * Returns one feature per `fixedSites` entry (aligned), or null if some site has
 * no placeable feature under the constraints.
 *
 * Used two ways: to seed the PT chains at site-pinned slots (where the seed's
 * track values then reroll during sampling), and to resolve a site pin to an
 * ordinary feature pin on the greedy fast path (which has no reroll machinery,
 * so the seed stays put). */
export function resolveSitePins(
  model: IsingModel,
  fixedSites: readonly number[],
  fixedFeatures: readonly number[],
  excluded: Iterable<number>,
): number[] | null {
  if (fixedSites.length === 0) return [];
  const exSet = new Set<number>(excluded);
  const usedSites = new Set<number>();
  const usedTrack = model.tracks.map(() => new Set<string>());
  const addUsed = (f: number) => {
    usedSites.add(model.siteOf[f]);
    for (let t = 0; t < model.tracks.length; t++) {
      if (!model.tracks[t].unique) continue;
      const v = model.trackValues[f][t];
      if (v !== null) usedTrack[t].add(v);
    }
  };
  for (const f of fixedFeatures) addUsed(f);
  const seeds: number[] = [];
  for (const site of fixedSites) {
    if (usedSites.has(site)) return null; // site already taken by a feature pin / duplicate
    let best = -1;
    let bestM = -Infinity;
    for (const f of model.siteFeatures[site]) {
      if (exSet.has(f)) continue;
      let conflict = false;
      for (let t = 0; t < model.tracks.length; t++) {
        if (!model.tracks[t].unique) continue;
        const v = model.trackValues[f][t];
        if (v !== null && usedTrack[t].has(v)) { conflict = true; break; }
      }
      if (conflict) continue;
      if (model.m[f] > bestM) { bestM = model.m[f]; best = f; }
    }
    if (best < 0) return null;
    seeds.push(best);
    addUsed(best);
  }
  return seeds;
}

/** Anchor-field tilt boost for static (feature-pin) surfaces:
 * `boost[j] = (alpha-1)·Σ_{p∈pins} J[p,j]`, zeroed on every feature of a
 * pinned site (pin↔pin couplings are untilted, and pins never move). Adding
 * this to hEff makes the pairwise energy equal
 * `H_alpha = H - (alpha-1)·Σ_{p,j free} J[p,j]s_j`, so meanfield / greedy on
 * the boosted field target the same tilted measure as the PT Potts kernel.
 * Mirrors `sampling.anchor_boost` (parity-gated indirectly via the meanfield
 * and greedy cases). */
export function anchorBoost(
  model: IsingModel,
  pins: readonly number[],
  anchorStrength: number,
): Float64Array {
  const { V, J, siteOf } = model;
  const boost = new Float64Array(V);
  if (anchorStrength === 1 || pins.length === 0) return boost;
  const pinSites = new Set<number>();
  for (const p of pins) pinSites.add(siteOf[p]);
  for (let j = 0; j < V; j++) {
    if (pinSites.has(siteOf[j])) continue;
    let s = 0;
    for (const p of pins) s += J[p * V + j];
    boost[j] = (anchorStrength - 1) * s;
  }
  return boost;
}

/** Build the list of vocab indices available to fill free team slots
 * (i.e. not fixed, not excluded). Convenience used by every sampler. */
export function availableIndices(
  model: IsingModel,
  fixed: Iterable<number>,
  excluded: Iterable<number>,
): number[] {
  const ex = new Set<number>();
  for (const i of fixed) ex.add(i);
  for (const i of excluded) ex.add(i);
  const out: number[] = [];
  for (let i = 0; i < model.V; i++) {
    if (!ex.has(i)) out.push(i);
  }
  return out;
}
