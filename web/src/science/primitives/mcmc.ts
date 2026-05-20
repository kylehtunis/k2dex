// Single-spin Metropolis on a small graph. Toy code — not used by the
// production sampler.
//
// One sweep = V single-spin flip proposals (uniformly chosen sites).
// ΔH for flipping spin i: ΔH = (1 - 2 s_i) * (h_i + Σ_j J_ij s_j).
// Accept iff ΔH ≤ 0 or rand() < exp(-βΔH).

import type { Rng } from "../../sampler/rng";
import type { Graph, State } from "./graph";
import { graphEnergy } from "./graph";

export interface ChainOptions {
  sweeps: number;
  burnIn?: number;
  initialState?: State;
  record?: boolean;
}

export interface ChainResult {
  marginals: number[];
  finalState: State;
  history: State[];
  energies: number[];
}

function deltaH(g: Graph, s: State, i: number): number {
  let local = g.h[i];
  for (let j = 0; j < g.V; j++) local += g.J[i][j] * s[j];
  return -(1 - 2 * s[i]) * local;
}

export function runChain(
  g: Graph,
  T: number,
  rng: Rng,
  opts: ChainOptions,
): ChainResult {
  const { sweeps, burnIn = 0, initialState, record = false } = opts;
  const s: State = initialState
    ? initialState.slice()
    : Array.from({ length: g.V }, () => (rng.random() < 0.5 ? 0 : 1));
  const beta = 1 / Math.max(T, 1e-9);
  const counts = new Array(g.V).fill(0);
  const history: State[] = [];
  const energies: number[] = [];
  let kept = 0;
  for (let sw = 0; sw < burnIn + sweeps; sw++) {
    for (let step = 0; step < g.V; step++) {
      const i = rng.integers(g.V);
      const dH = deltaH(g, s, i);
      if (dH <= 0 || rng.random() < Math.exp(-beta * dH)) {
        s[i] = 1 - s[i];
      }
    }
    if (sw >= burnIn) {
      for (let i = 0; i < g.V; i++) if (s[i]) counts[i]++;
      kept++;
      if (record) {
        history.push(s.slice());
        energies.push(graphEnergy(g, s));
      }
    }
  }
  return {
    marginals: counts.map((c) => c / Math.max(kept, 1)),
    finalState: s,
    history,
    energies,
  };
}
