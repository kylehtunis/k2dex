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
 * mechanical mutual exclusion. Same-species pairs, and same-item pairs on
 * Species @ Item vocab, couple purely because the two builds can't co-exist on
 * a team, so they carry no metagame signal. On Species-only vocab (unique
 * species, all-null items) both checks are no-ops, so every off-diagonal pair
 * is structural. Shared by filteredCouplings and render/featureDetail. */
export function isStructuralPair(model: IsingModel, i: number, j: number): boolean {
  const { speciesOf, itemOf } = model;
  if (speciesOf[i] === speciesOf[j]) return false;
  const itI = itemOf[i];
  const itJ = itemOf[j];
  if (itI !== null && itJ !== null && itI === itJ) return false;
  return true;
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
