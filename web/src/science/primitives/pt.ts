// Toy parallel-tempered MCMC on a small graph. Each replica is a single-spin
// Metropolis chain at its own temperature; adjacent-rung swaps are attempted
// every `swapInterval` sweeps with acceptance Δβ * ΔH.
//
// This is the demonstration variant for /science S4. It is structurally
// different from web/src/sampler/pt.ts (swap-move MCMC for fixed-size subsets).
// Keeping them separate keeps the pedagogy honest.

import type { Rng } from "../../sampler/rng";
import type { Graph, State } from "./graph";
import { graphEnergy } from "./graph";

export interface PTOptions {
  sweeps: number;
  burnIn?: number;
  swapInterval?: number;
  record?: boolean;
}

export interface PTResult {
  /** Cold-chain (T = ladder[0]) marginals. */
  coldMarginals: number[];
  /** history[sweep][rungIndex] = originalReplicaId (for swap markers). */
  history: number[][];
  /** Total accepted replica exchanges. */
  swaps: number;
}

function singleSweep(g: Graph, s: State, beta: number, rng: Rng): void {
  for (let step = 0; step < g.V; step++) {
    const i = rng.integers(g.V);
    let local = g.h[i];
    for (let j = 0; j < g.V; j++) local += g.J[i][j] * s[j];
    const dH = -(1 - 2 * s[i]) * local;
    if (dH <= 0 || rng.random() < Math.exp(-beta * dH)) {
      s[i] = 1 - s[i];
    }
  }
}

export function runPT(
  g: Graph,
  ladder: number[],
  rng: Rng,
  opts: PTOptions,
): PTResult {
  const { sweeps, burnIn = 0, swapInterval = 5, record = false } = opts;
  const K = ladder.length;
  const betas = ladder.map((T) => 1 / Math.max(T, 1e-9));
  const states: State[] = Array.from({ length: K }, () =>
    Array.from({ length: g.V }, () => (rng.random() < 0.5 ? 0 : 1)),
  );
  const labels = Array.from({ length: K }, (_, k) => k);
  const counts = new Array(g.V).fill(0);
  const history: number[][] = [];
  let kept = 0;
  let swaps = 0;
  for (let sw = 0; sw < burnIn + sweeps; sw++) {
    for (let k = 0; k < K; k++) singleSweep(g, states[k], betas[k], rng);
    if (sw % swapInterval === 0) {
      for (let k = 0; k < K - 1; k++) {
        const dH =
          graphEnergy(g, states[k]) - graphEnergy(g, states[k + 1]);
        const dBeta = betas[k] - betas[k + 1];
        const logAcc = dBeta * dH;
        if (logAcc >= 0 || rng.random() < Math.exp(logAcc)) {
          [states[k], states[k + 1]] = [states[k + 1], states[k]];
          [labels[k], labels[k + 1]] = [labels[k + 1], labels[k]];
          swaps++;
        }
      }
    }
    if (sw >= burnIn) {
      for (let i = 0; i < g.V; i++) if (states[0][i]) counts[i]++;
      kept++;
      if (record) history.push(labels.slice());
    }
  }
  return {
    coldMarginals: counts.map((c) => c / Math.max(kept, 1)),
    history,
    swaps,
  };
}
