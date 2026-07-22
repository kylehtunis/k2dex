// Score every legal (out ∈ team, in ∉ team) single-swap from `team`
// and return the top-N by ΔE_adj. Independent suggestions, not a chain.
// Used by /analysis to surface "the model's N biggest critiques" of an
// observed team.
//
// Mirrors sampling.rank_single_swaps.

import type { IsingModel, SingleSwapEntry } from "./types";

export interface RankOpts {
  team: readonly number[];
  fieldWeight: number;
  topN?: number;
}

const DEFAULT_TOP_N = 20;

export function rankSingleSwaps(
  model: IsingModel,
  opts: RankOpts,
): SingleSwapEntry[] {
  const { V, J, h, siteOf, tracks, trackValues } = model;
  const fw = opts.fieldWeight;
  const topN = opts.topN ?? DEFAULT_TOP_N;

  const teamArr = [...opts.team];
  const teamMask = new Uint8Array(V);
  for (const i of teamArr) teamMask[i] = 1;

  const results: SingleSwapEntry[] = [];

  for (const outIdx of teamArr) {
    const others: number[] = [];
    for (const i of teamArr) if (i !== outIdx) others.push(i);

    // ΣJ[out, others]
    let jOutOthers = 0;
    for (const j of others) jOutOthers += J[outIdx * V + j];

    // Validity mask for in candidates.
    const valid = new Uint8Array(V);
    for (let i = 0; i < V; i++) valid[i] = teamMask[i] ? 0 : 1;
    // Site (species) uniqueness against the rest of the team.
    const othersSites = new Set<number>();
    for (const j of others) othersSites.add(siteOf[j]);
    for (let i = 0; i < V; i++) {
      if (valid[i] && othersSites.has(siteOf[i])) valid[i] = 0;
    }
    // Per-unique-track value uniqueness against the rest of the team.
    for (let t = 0; t < tracks.length; t++) {
      if (!tracks[t].crossSlotUnique) continue;
      const othersValues = new Set<string>();
      for (const j of others) {
        const v = trackValues[j][t];
        if (v !== null) othersValues.add(v);
      }
      for (let i = 0; i < V; i++) {
        const v = trackValues[i][t];
        if (valid[i] && v !== null && othersValues.has(v)) valid[i] = 0;
      }
    }

    const hOut = h[outIdx];
    for (let inIdx = 0; inIdx < V; inIdx++) {
      if (!valid[inIdx]) continue;
      let jInOthers = 0;
      const baseIn = inIdx * V;
      for (const j of others) jInOthers += J[baseIn + j];
      const dSumJ = jInOthers - jOutOthers;
      const dERaw = -(h[inIdx] - hOut) - dSumJ;
      const dEAdj = -fw * (h[inIdx] - hOut) - dSumJ;
      results.push({
        outIdx,
        inIdx,
        deltaEAdj: dEAdj,
        deltaERaw: dERaw,
        deltaSumJ: dSumJ,
      });
    }
  }

  results.sort((a, b) => a.deltaEAdj - b.deltaEAdj);
  return results.slice(0, topN);
}
