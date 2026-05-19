// Shared energy + constraint primitives.
//
// Mirrors the helpers at the top of sampling.py:
//   team_energy, build_constraint_sets, swap_violates_uniqueness,
//   initialize_state.
//
// Uniqueness machinery (no-duplicate-species, no-duplicate-item) is
// gated on whether speciesOf / itemOf were passed. Phase 1 / Phase 2
// pass them as null and the checks become no-ops, mirroring Python.

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
  fixedSpecies: Set<string>;
  fixedItems: Set<string>;
}

/** Species + non-null item sets occupied by `fixed`. Empty when
 * speciesOf / itemOf are not provided. */
export function buildConstraintSets(
  fixed: Iterable<number>,
  speciesOf: readonly string[] | null,
  itemOf: readonly (string | null)[] | null,
): ConstraintSets {
  const fixedSpecies = new Set<string>();
  const fixedItems = new Set<string>();
  if (speciesOf === null && itemOf === null) {
    return { fixedSpecies, fixedItems };
  }
  for (const i of fixed) {
    if (speciesOf !== null) fixedSpecies.add(speciesOf[i]);
    if (itemOf !== null) {
      const it = itemOf[i];
      if (it !== null) fixedItems.add(it);
    }
  }
  return { fixedSpecies, fixedItems };
}

/** True iff swapping `iIn` into position `outK` of `onNf` would create
 * a duplicate species or duplicate non-null item (counting both the
 * fixed and free slots of the team). */
export function swapViolatesUniqueness(
  iIn: number,
  outK: number,
  onNf: readonly number[],
  fixedSpecies: Set<string>,
  fixedItems: Set<string>,
  speciesOf: readonly string[] | null,
  itemOf: readonly (string | null)[] | null,
): boolean {
  if (speciesOf !== null) {
    const target = speciesOf[iIn];
    if (fixedSpecies.has(target)) return true;
    for (let k = 0; k < onNf.length; k++) {
      if (k === outK) continue;
      if (speciesOf[onNf[k]] === target) return true;
    }
  }
  if (itemOf !== null) {
    const target = itemOf[iIn];
    if (target !== null) {
      if (fixedItems.has(target)) return true;
      for (let k = 0; k < onNf.length; k++) {
        if (k === outK) continue;
        if (itemOf[onNf[k]] === target) return true;
      }
    }
  }
  return false;
}

/** Greedy uniqueness-respecting sample of `nToFill` indices from
 * `available`. Returns null if no valid completion exists after
 * `maxAttempts` random restarts (e.g., user pinned conflicting items).
 *
 * Falls back to plain without-replacement sampling when both lookups
 * are null — the Phase 1/2 case. */
export function initializeState(
  available: readonly number[],
  nToFill: number,
  fixedSpecies: Set<string>,
  fixedItems: Set<string>,
  speciesOf: readonly string[] | null,
  itemOf: readonly (string | null)[] | null,
  rng: Rng,
  maxAttempts = 100,
): number[] | null {
  if (speciesOf === null && itemOf === null) {
    if (available.length < nToFill) return null;
    return rng.choice(available, nToFill);
  }
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const chosen: number[] = [];
    const usedSpecies = new Set(fixedSpecies);
    const usedItems = new Set(fixedItems);
    const shuffled = rng.permutation(available);
    for (const idx of shuffled) {
      if (chosen.length === nToFill) break;
      if (speciesOf !== null && usedSpecies.has(speciesOf[idx])) continue;
      if (itemOf !== null) {
        const it = itemOf[idx];
        if (it !== null && usedItems.has(it)) continue;
      }
      chosen.push(idx);
      if (speciesOf !== null) usedSpecies.add(speciesOf[idx]);
      if (itemOf !== null) {
        const it = itemOf[idx];
        if (it !== null) usedItems.add(it);
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
