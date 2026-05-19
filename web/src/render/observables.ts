// Per-team observables + pairwise-J decomposition.
//
// Shared between /completer (post-run results) and /analysis (per-team
// diagnostics). Mirrors rendering.intra_team_sum_j + pairwise_j_rows
// and the inline observable math in app.py:_render_completer.
//
// Sign convention: scores are sign-flipped relative to the Hamiltonian
// space H(s) so higher = better team. Coherence is the pure J piece
// (= 0.5 s'Js), not sign-flipped.

import type { IsingModel } from "../sampler/types";

function teamHSum(h: Float64Array, team: readonly number[]): number {
  let s = 0;
  for (const i of team) s += h[i];
  return s;
}

/** 0.5 * s'Js for the team membership. Same as Python's
 * `intra_team_sum_j` — measures intra-team pairwise coherence. */
export function intraTeamSumJ(
  J: Float64Array,
  V: number,
  team: readonly number[],
): number {
  let s = 0;
  for (let a = 0; a < team.length; a++) {
    const base = team[a] * V;
    for (let b = 0; b < team.length; b++) s += J[base + team[b]];
  }
  return 0.5 * s;
}

export interface Observables {
  scoreAdj: number;
  scoreRaw: number;
  coherence: number;
}

export function teamObservables(
  model: IsingModel,
  team: readonly number[],
  fieldWeight: number,
): Observables {
  const hDot = teamHSum(model.h, team);
  const coherence = intraTeamSumJ(model.J, model.V, team);
  return {
    scoreAdj: fieldWeight * hDot + coherence,
    scoreRaw: hDot + coherence,
    coherence,
  };
}

export interface PairwiseJRow {
  rank: number;
  /** Vocab index of the first member. */
  idxA: number;
  /** Vocab index of the second member. */
  idxB: number;
  nameA: string;
  nameB: string;
  jValue: number;
  /** |J_ij| / Σ|J| over the team's C(team_size, 2) pairs. */
  pctOfAbsSum: number;
}

/** Pairwise J decomposition: C(team_size, 2) entries, sorted by |J|
 * descending. Mirrors rendering.pairwise_j_rows. Caller passes sorted
 * (or any-ordered) team indices. */
export function pairwiseJRows(
  team: readonly number[],
  vocab: readonly string[],
  J: Float64Array,
  V: number,
): PairwiseJRow[] {
  // Build (idxA, idxB, J[i,j]) tuples for each unordered pair.
  const pairs: Array<{ a: number; b: number; j: number }> = [];
  for (let i = 0; i < team.length; i++) {
    for (let k = i + 1; k < team.length; k++) {
      const a = team[i];
      const b = team[k];
      pairs.push({ a, b, j: J[a * V + b] });
    }
  }
  pairs.sort((p, q) => Math.abs(q.j) - Math.abs(p.j));
  let absSum = 0;
  for (const p of pairs) absSum += Math.abs(p.j);
  if (absSum === 0) absSum = 1;
  return pairs.map((p, r) => ({
    rank: r + 1,
    idxA: p.a,
    idxB: p.b,
    nameA: vocab[p.a],
    nameB: vocab[p.b],
    jValue: p.j,
    pctOfAbsSum: Math.abs(p.j) / absSum,
  }));
}
