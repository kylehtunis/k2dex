// Greedy steepest-descent over single-swap moves.
//
// Mirrors sampling.greedy_optimize: at each step, evaluate every
// (non-pinned out-slot, valid in-candidate) swap, take the most
// improving one, stop at the first local minimum or after maxSwaps.
//
// Used as the second half of the fast path (MF marginals → greedy fill
// → greedy descent) in /completer, and as the swap-chain critique in
// /analysis.

import type { IsingModel, GreedyChainEntry } from "./types";

export interface GreedyOpts {
  startingTeam: readonly number[];
  pinned: readonly number[];
  excluded: readonly number[];
  fieldWeight: number;
  maxSwaps?: number;
}

const DEFAULT_MAX_SWAPS = 20;

function teamSumJ(J: Float64Array, V: number, team: readonly number[]): number {
  // 0.5 * s'Js for the team membership only.
  let s = 0;
  for (let a = 0; a < team.length; a++) {
    const ia = team[a];
    const base = ia * V;
    for (let b = 0; b < team.length; b++) {
      s += J[base + team[b]];
    }
  }
  return 0.5 * s;
}

function teamHSum(h: Float64Array, team: readonly number[]): number {
  let s = 0;
  for (const i of team) s += h[i];
  return s;
}

/** Greedy single-swap descent on E_adj. Deterministic given inputs. */
export function greedyOptimize(
  model: IsingModel,
  opts: GreedyOpts,
): { finalTeam: number[]; chain: GreedyChainEntry[] } {
  const { V, J, h, speciesOf, itemOf } = model;
  const fw = opts.fieldWeight;
  const maxSwaps = opts.maxSwaps ?? DEFAULT_MAX_SWAPS;

  const current = new Set<number>(opts.startingTeam);
  const pinned = new Set<number>(opts.pinned);
  const excluded = new Set<number>(opts.excluded);

  const eAdj = (team: number[]) =>
    -fw * teamHSum(h, team) - teamSumJ(J, V, team);
  const eRaw = (team: number[]) => -teamHSum(h, team) - teamSumJ(J, V, team);

  let energyAdj = eAdj([...current]);
  const chain: GreedyChainEntry[] = [];

  for (let step = 1; step <= maxSwaps; step++) {
    const currentArr = [...current];
    const currentMask = new Uint8Array(V);
    for (const i of currentArr) currentMask[i] = 1;

    let bestDelta = 0;
    let bestOut: number | null = null;
    let bestIn: number | null = null;

    for (const outIdx of currentArr) {
      if (pinned.has(outIdx)) continue;
      const others: number[] = [];
      for (const i of currentArr) if (i !== outIdx) others.push(i);

      // ΣJ[out, others]
      let jOutOthers = 0;
      for (const j of others) jOutOthers += J[outIdx * V + j];

      // Validity mask reused per outIdx — Phase-3 uniqueness depends on
      // `others`, so it differs for each out slot. The team mask excludes
      // current members; excluded list and uniqueness layer on top.
      const valid = new Uint8Array(V);
      for (let i = 0; i < V; i++) valid[i] = currentMask[i] ? 0 : 1;
      for (const ex of excluded) valid[ex] = 0;
      if (speciesOf !== null) {
        const othersSpecies = new Set<string>();
        for (const j of others) othersSpecies.add(speciesOf[j]);
        if (othersSpecies.size > 0) {
          for (let i = 0; i < V; i++) {
            if (valid[i] && othersSpecies.has(speciesOf[i])) valid[i] = 0;
          }
        }
      }
      if (itemOf !== null) {
        const othersItems = new Set<string>();
        for (const j of others) {
          const it = itemOf[j];
          if (it !== null) othersItems.add(it);
        }
        if (othersItems.size > 0) {
          for (let i = 0; i < V; i++) {
            const it = itemOf[i];
            if (valid[i] && it !== null && othersItems.has(it)) valid[i] = 0;
          }
        }
      }

      // Scan candidates. ΔE_adj[i] = -fw*(h[i]-h[out]) - (J[i,others].sum() - jOutOthers)
      const hOut = h[outIdx];
      for (let i = 0; i < V; i++) {
        if (!valid[i]) continue;
        let jInOthers = 0;
        const baseI = i * V;
        for (const j of others) jInOthers += J[baseI + j];
        const delta = -fw * (h[i] - hOut) - (jInOthers - jOutOthers);
        if (delta < bestDelta) {
          bestDelta = delta;
          bestOut = outIdx;
          bestIn = i;
        }
      }
    }

    if (bestOut === null || bestIn === null) break;

    current.delete(bestOut);
    current.add(bestIn);
    energyAdj += bestDelta;
    const teamAfter = [...current].sort((a, b) => a - b);
    chain.push({
      step,
      outIdx: bestOut,
      inIdx: bestIn,
      deltaEAdj: bestDelta,
      energyAdjAfter: energyAdj,
      energyRawAfter: eRaw(teamAfter),
      sumJAfter: teamSumJ(J, V, teamAfter),
      teamAfter,
    });
  }

  return {
    finalTeam: [...current].sort((a, b) => a - b),
    chain,
  };
}
