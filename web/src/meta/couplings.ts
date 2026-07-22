// Filter + sort logic for the §03 extreme-couplings tables.
//
// Mirrors the upper-triangle mask + cross-species / cross-item filter
// from app.py:_render_meta. Phase 2 has unique species and all-null
// items, so both filters become no-ops there.

import type { IsingModel } from "../sampler/types";

export interface CouplingPair {
  /** Upper-triangle index (i < j) into the vocab. */
  i: number;
  j: number;
  jValue: number;
}

/** True when (i, j) is a *structural* coupling worth surfacing — i.e. not a
 * mechanical mutual exclusion. Same-species pairs, and same-value pairs on any
 * cross-slot-unique track (today: item only), couple purely because the two
 * builds can't co-exist on a team, so they carry no metagame signal. On
 * Species-only vocab (unique species, all-null items) both checks are no-ops,
 * so every off-diagonal pair is structural. Shared by filteredCouplings and
 * render/featureDetail. */
export function isStructuralPair(model: IsingModel, i: number, j: number): boolean {
  const { speciesOf, tracks, trackValues } = model;
  if (speciesOf[i] === speciesOf[j]) return false;
  for (let t = 0; t < tracks.length; t++) {
    if (!tracks[t].crossSlotUnique) continue;
    const vi = trackValues[i][t];
    const vj = trackValues[j][t];
    if (vi !== null && vj !== null && vi === vj) return false;
  }
  return true;
}

export interface ModulationEntry {
  featureA: number;
  featureB: number;
  jValue: number;
  /** How far this item-pair's coupling sits from the pair's species-level
   * synergy — the item-modulation residual for the pair. */
  deviation: number;
}

/** The item-pair couplings for a species pair, strongest |J| first: how each
 * pairing of their item builds shifts away from the pair's species-level
 * synergy. Non-structural pairs are dropped (see isStructuralPair). Shared by
 * the /meta §02 coupling table and the feature modal's coupling drill-down, so
 * both surfaces rank and filter identically. */
export function topModulationEntries(
  model: IsingModel,
  siteA: number,
  siteB: number,
  synergy: number,
  topN = 8,
): ModulationEntry[] {
  if (siteA === siteB) return [];
  const { siteFeatures, J, V } = model;
  const entries: ModulationEntry[] = [];
  for (const fa of siteFeatures[siteA]) {
    for (const fb of siteFeatures[siteB]) {
      if (!isStructuralPair(model, fa, fb)) continue;
      const jValue = J[fa * V + fb];
      entries.push({
        featureA: fa,
        featureB: fb,
        jValue,
        deviation: jValue - synergy,
      });
    }
  }
  entries.sort((a, b) => Math.abs(b.jValue) - Math.abs(a.jValue));
  return entries.slice(0, topN);
}

/** Iterate the strict upper triangle of J, dropping non-structural pairs
 * (see isStructuralPair). */
export function filteredCouplings(model: IsingModel): CouplingPair[] {
  const { V, J } = model;
  const out: CouplingPair[] = [];
  for (let i = 0; i < V; i++) {
    for (let j = i + 1; j < V; j++) {
      if (!isStructuralPair(model, i, j)) continue;
      out.push({ i, j, jValue: J[i * V + j] });
    }
  }
  return out;
}
