// Mean-field marginals on a small graph (binary spins ∈ {0, 1}).
//
// Fixed-point iteration: m_i ← σ(β (h_i + Σ_j J_ij m_j))
// where σ is the logistic sigmoid. Converges to a self-consistent
// solution; quality depends on graph topology and temperature.

import type { Graph } from "./graph";

export interface MFOptions {
  maxIters?: number;
  tol?: number;
  initialMarginals?: number[];
  record?: boolean;
}

export interface MFResult {
  marginals: number[];
  converged: boolean;
  iters: number;
  history: number[][];
}

function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  } else {
    const z = Math.exp(x);
    return z / (1 + z);
  }
}

export function meanFieldIterate(
  g: Graph,
  T: number,
  opts: MFOptions = {},
): MFResult {
  const { maxIters = 200, tol = 1e-6, initialMarginals, record = false } = opts;
  const beta = 1 / Math.max(T, 1e-9);
  const m = initialMarginals ? initialMarginals.slice() : new Array(g.V).fill(0.5);
  const history: number[][] = record ? [m.slice()] : [];
  let converged = false;
  let iter = 0;
  for (iter = 0; iter < maxIters; iter++) {
    const next = new Array(g.V);
    for (let i = 0; i < g.V; i++) {
      let s = g.h[i];
      for (let j = 0; j < g.V; j++) s += g.J[i][j] * m[j];
      next[i] = sigmoid(beta * s);
    }
    let maxDelta = 0;
    for (let i = 0; i < g.V; i++) {
      maxDelta = Math.max(maxDelta, Math.abs(next[i] - m[i]));
      m[i] = next[i];
    }
    if (record) history.push(m.slice());
    if (maxDelta < tol) {
      converged = true;
      iter++;
      break;
    }
  }
  return { marginals: m, converged, iters: iter, history };
}
