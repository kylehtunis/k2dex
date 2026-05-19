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

/** Iterate the strict upper triangle of J, optionally dropping same-
 * species and same-item pairs (Species @ Item only — those couplings
 * exist purely from mechanical mutual exclusion in VGC and don't
 * reflect model structure). */
export function filteredCouplings(model: IsingModel): CouplingPair[] {
  const { V, J, speciesOf, itemOf } = model;
  const out: CouplingPair[] = [];
  for (let i = 0; i < V; i++) {
    const spI = speciesOf[i];
    const itI = itemOf[i];
    for (let j = i + 1; j < V; j++) {
      if (speciesOf[j] === spI) continue;
      const itJ = itemOf[j];
      if (itI !== null && itJ !== null && itI === itJ) continue;
      out.push({ i, j, jValue: J[i * V + j] });
    }
  }
  return out;
}
