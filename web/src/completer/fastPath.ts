// Fast-path completer: MF marginals → uniqueness-respecting fill →
// greedy descent. Mirrors the post-v1 `/completer` fast path in
// app.py:_render_completer (the "popularity-fill bug" fix).

import { GREEDY_MAX_SWAPS, MF_MAX_ITERS, MF_TOL, TEAM_SIZE } from "../constants";
import { greedyOptimize } from "../sampler/greedy";
import { meanfieldMarginals } from "../sampler/meanfield";
import type { GreedyChainEntry, IsingModel } from "../sampler/types";

export interface FastPathInput {
  fixed: readonly number[];
  excludedSpecies: readonly string[]; // species-level
  fieldWeight: number;
}

export interface FastPathResult {
  finalTeam: readonly number[];
  chain: readonly GreedyChainEntry[];
  fixed: readonly number[];
  excluded: readonly number[];
  fieldWeight: number;
  mfIters: number;
}

export type FastPathError =
  | { kind: "over_constrained"; message: string }
  | { kind: "fill_failed"; message: string }
  | { kind: "too_many_pins"; message: string };

/** Expand species-level excludes to vocab-index-level (Phase 3 vocab
 * has multiple item variants per species; excluding a species kills
 * all of them). */
function expandExcludedSpecies(
  excludedSpecies: readonly string[],
  model: IsingModel,
): number[] {
  if (excludedSpecies.length === 0) return [];
  const set = new Set(excludedSpecies);
  const out: number[] = [];
  for (let i = 0; i < model.V; i++) {
    if (set.has(model.speciesOf[i])) out.push(i);
  }
  return out;
}

export function runFastPath(
  model: IsingModel,
  input: FastPathInput,
): { ok: true; result: FastPathResult } | { ok: false; error: FastPathError } {
  const { fixed, fieldWeight } = input;
  const excluded = expandExcludedSpecies(input.excludedSpecies, model);
  const kFree = TEAM_SIZE - fixed.length;
  if (kFree < 0) {
    return {
      ok: false,
      error: { kind: "too_many_pins", message: `Pinned more than ${TEAM_SIZE} mons.` },
    };
  }

  const mf = meanfieldMarginals(model, {
    fixed,
    excluded,
    fieldWeight,
    nIters: MF_MAX_ITERS,
    tol: MF_TOL,
  });
  if (mf === null) {
    return {
      ok: false,
      error: {
        kind: "over_constrained",
        message:
          "Not enough available Pokemon to fill the team after applying constraints.",
      },
    };
  }

  // Pop top valid candidates by marginal, honoring uniqueness against
  // both fixed mons and previously-picked candidates.
  const validIdxs: number[] = [];
  for (let i = 0; i < model.V; i++) if (mf.validMask[i]) validIdxs.push(i);
  validIdxs.sort((a, b) => mf.marginals[b] - mf.marginals[a]);

  const { siteOf, tracks, trackValues } = model;
  const usedSites = new Set<number>();
  const usedTrack = tracks.map(() => new Set<string>());
  const addUsed = (i: number) => {
    usedSites.add(siteOf[i]);
    for (let t = 0; t < tracks.length; t++) {
      if (!tracks[t].unique) continue;
      const v = trackValues[i][t];
      if (v !== null) usedTrack[t].add(v);
    }
  };
  const conflicts = (i: number): boolean => {
    if (usedSites.has(siteOf[i])) return true;
    for (let t = 0; t < tracks.length; t++) {
      if (!tracks[t].unique) continue;
      const v = trackValues[i][t];
      if (v !== null && usedTrack[t].has(v)) return true;
    }
    return false;
  };
  for (const i of fixed) addUsed(i);
  const initFree: number[] = [];
  for (const cand of validIdxs) {
    if (initFree.length === kFree) break;
    if (conflicts(cand)) continue;
    initFree.push(cand);
    addUsed(cand);
  }
  if (initFree.length < kFree) {
    return {
      ok: false,
      error: {
        kind: "fill_failed",
        message:
          "Could not fill the team — insufficient non-conflicting candidates.",
      },
    };
  }

  const startTeam = [...fixed, ...initFree].sort((a, b) => a - b);
  const { finalTeam, chain } = greedyOptimize(model, {
    startingTeam: startTeam,
    pinned: fixed,
    excluded,
    fieldWeight,
    maxSwaps: GREEDY_MAX_SWAPS,
  });
  return {
    ok: true,
    result: {
      finalTeam,
      chain,
      fixed,
      excluded,
      fieldWeight,
      mfIters: mf.iters,
    },
  };
}
