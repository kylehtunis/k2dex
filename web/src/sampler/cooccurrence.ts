// Raw co-occurrence baseline — the "count how often they appear together"
// method every other teambuilder uses (Smogon's "Teammates %").
//
// This is the naive foil the Ising model is compared against in the
// "Why Not Just Count?" article. Mirrors the co-occurrence scorer in
// notebooks/validation.ipynb: C = X'X on the binary team matrix, and a
// partial team is scored by summing each candidate's co-occurrence counts
// with the pinned members.
//
// Mirrored 1:1 in k2dex/rendering.py (build_cooccurrence / score_cooccurrence),
// gated by tests/test_parity.py::test_cooccurrence_cases.

import type { IsingModel, TeamCounts } from "./types";
import { buildConstraintSets, occupy, violatesConstraints } from "./energy";
import { parseTeamKey } from "../render/corpus";

export interface Cooccurrence {
  /** V×V flat row-major co-occurrence counts, symmetric, zero diagonal.
   * `C[i*V + j]` = number of corpus teams containing both feature i and j. */
  C: Float64Array;
  /** Per-feature marginal appearance rate (appearances / nTeams). */
  m: Float64Array;
  /** Total number of teams (sum of counts). */
  nTeams: number;
}

/** Build the co-occurrence matrix and marginals from the corpus team-count
 * index. Each key is a sorted-index roster (see precompute.serialize_team_counts);
 * a team of size k contributes `count` to C[i,j] for each of its k*(k-1)/2 pairs
 * and to m[i] for each member. Diagonal stays zero. Mirrors the notebook's
 * `C = (X.T @ X)` with `np.fill_diagonal(C, 0)` and `m = X.mean(axis=0)`. */
export function buildCooccurrence(
  teamCounts: TeamCounts,
  V: number,
): Cooccurrence {
  const C = new Float64Array(V * V);
  const m = new Float64Array(V);
  let nTeams = 0;
  for (const [key, count] of teamCounts) {
    const team = parseTeamKey(key);
    nTeams += count;
    for (let a = 0; a < team.length; a++) {
      const ia = team[a];
      m[ia] += count;
      const rowA = ia * V;
      for (let b = a + 1; b < team.length; b++) {
        const ib = team[b];
        C[rowA + ib] += count;
        C[ib * V + ia] += count;
      }
    }
  }
  if (nTeams > 0) {
    for (let i = 0; i < V; i++) m[i] /= nTeams;
  }
  return { C, m, nTeams };
}

/** Co-occurrence score for every feature against a set of held-in (pinned)
 * features: `score[i] = sum(C[i, j] for j in heldIn)`. Higher = appears with
 * the pinned members more often. Mirrors the notebook's `score_cooccurrence`. */
export function scoreCooccurrence(
  C: Float64Array,
  V: number,
  heldIn: readonly number[],
): Float64Array {
  const scores = new Float64Array(V);
  for (let i = 0; i < V; i++) {
    const rowI = i * V;
    let s = 0;
    for (const j of heldIn) s += C[rowI + j];
    scores[i] = s;
  }
  return scores;
}

/** Greedy team fill by co-occurrence score — the "naive teambuilder":
 * repeatedly add the highest-scoring candidate that keeps the team legal
 * (unique species, unique per-unique-track values), rescoring against the
 * growing team each step, until the team is full or no candidate fits.
 *
 * Returns the completed team as sorted feature indices. `fixed` are the
 * pinned members; `excluded` are banned features. Deterministic given inputs
 * (ties broken by lowest feature index via the stable scan order). */
export function cooccurrenceGreedy(
  cooc: Cooccurrence,
  model: IsingModel,
  opts: { fixed: readonly number[]; excluded: readonly number[] },
): number[] {
  const { C } = cooc;
  const { V, teamSize } = model;

  const team = [...opts.fixed];
  const excluded = new Set(opts.excluded);

  // Uniqueness bookkeeping against the current team.
  const taken = buildConstraintSets(team, model);
  const inTeam = new Set(team);

  while (team.length < teamSize) {
    const scores = scoreCooccurrence(C, V, team);
    let bestIdx = -1;
    let bestScore = -Infinity;
    for (let i = 0; i < V; i++) {
      if (inTeam.has(i) || excluded.has(i)) continue;
      if (violatesConstraints(i, taken, model)) continue;
      if (scores[i] > bestScore) {
        bestScore = scores[i];
        bestIdx = i;
      }
    }
    if (bestIdx < 0) break; // no legal candidate remains
    team.push(bestIdx);
    inTeam.add(bestIdx);
    occupy(taken, bestIdx, model);
  }

  return team.sort((a, b) => a - b);
}
