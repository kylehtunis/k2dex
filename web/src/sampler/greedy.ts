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
import { anchorBoost } from "./energy";

export interface GreedyOpts {
  startingTeam: readonly number[];
  pinned: readonly number[];
  excluded: readonly number[];
  fieldWeight: number;
  maxSwaps?: number;
  /** Anchor-field tilt alpha: pinned→free couplings enter the adjusted field
   * (alpha-1)-fold extra via `anchorBoost`, so the descent targets H_alpha.
   * Raw energies in the chain are unaffected. Default 1 (no tilt). */
  anchorStrength?: number;
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
  const { V, J, h, siteOf, tracks, trackValues } = model;
  const fw = opts.fieldWeight;
  const maxSwaps = opts.maxSwaps ?? DEFAULT_MAX_SWAPS;

  const current = new Set<number>(opts.startingTeam);
  const pinned = new Set<number>(opts.pinned);
  const excluded = new Set<number>(opts.excluded);

  // Adjusted field: fw*h plus the anchor-tilt boost (zero at alpha = 1).
  const hAdj = new Float64Array(V);
  for (let i = 0; i < V; i++) hAdj[i] = fw * h[i];
  const alpha = opts.anchorStrength ?? 1;
  if (alpha !== 1) {
    const boost = anchorBoost(model, opts.pinned, alpha);
    for (let i = 0; i < V; i++) hAdj[i] += boost[i];
  }

  const eAdj = (team: number[]) => -teamHSum(hAdj, team) - teamSumJ(J, V, team);
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
      // Site (species) uniqueness against the rest of the team.
      const othersSites = new Set<number>();
      for (const j of others) othersSites.add(siteOf[j]);
      if (othersSites.size > 0) {
        for (let i = 0; i < V; i++) {
          if (valid[i] && othersSites.has(siteOf[i])) valid[i] = 0;
        }
      }
      // Per-unique-track value uniqueness against the rest of the team.
      for (let t = 0; t < tracks.length; t++) {
        if (!tracks[t].crossSlotUnique) continue;
        const othersValues = new Set<string>();
        for (const j of others) {
          const v = trackValues[j][t];
          if (v !== null) othersValues.add(v);
        }
        if (othersValues.size === 0) continue;
        for (let i = 0; i < V; i++) {
          const v = trackValues[i][t];
          if (valid[i] && v !== null && othersValues.has(v)) valid[i] = 0;
        }
      }

      // Scan candidates. ΔE_adj[i] = -(hAdj[i]-hAdj[out]) - (J[i,others].sum() - jOutOthers)
      const hAdjOut = hAdj[outIdx];
      for (let i = 0; i < V; i++) {
        if (!valid[i]) continue;
        let jInOthers = 0;
        const baseI = i * V;
        for (const j of others) jInOthers += J[baseI + j];
        const delta = -(hAdj[i] - hAdjOut) - (jInOthers - jOutOthers);
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
