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
