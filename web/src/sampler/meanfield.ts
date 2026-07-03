// Damped mean-field marginals for the Ising-pair model.
//
// Mirrors sampling.meanfield_marginals. Used as the "fast path" in
// /completer: MF marginals → greedy fill → greedy descent. Validated as
// a ranking-faithful proxy for swap-MCMC at fieldWeight=1, T=1 (see
// CLAUDE.md MF-vs-MCMC bullet).

import type { IsingModel, MeanfieldResult } from "./types";
import { buildConstraintSets } from "./energy";

export interface MeanfieldOpts {
  /** Pinned features (clamped to m=1). */
  fixed: readonly number[];
  /** Banned features (clamped to m=0). */
  excluded: readonly number[];
  /** Scales h. */
  fieldWeight: number;
  /** Max iterations of the damped fixed-point. */
  nIters?: number;
  /** Convergence tol on max |Δm| over free slots. */
  tol?: number;
  /** Damping coefficient on the m update. */
  damp?: number;
}

const DEFAULT_N_ITERS = 200;
const DEFAULT_TOL = 1e-5;
const DEFAULT_DAMP = 0.5;

function sigmoid(x: number): number {
  // Branch keeps the math accurate in both tails.
  if (x >= 0) {
    const e = Math.exp(-x);
    return 1 / (1 + e);
  } else {
    const e = Math.exp(x);
    return e / (1 + e);
  }
}

/** Damped mean-field iteration. Returns `null` when fewer than
 * `teamSize - fixed.length` candidates remain after applying fixed /
 * excluded / Phase-3 uniqueness against `fixed`. */
export function meanfieldMarginals(
  model: IsingModel,
  opts: MeanfieldOpts,
): MeanfieldResult | null {
  const { V, J, h, teamSize } = model;
  const nIters = opts.nIters ?? DEFAULT_N_ITERS;
  const tol = opts.tol ?? DEFAULT_TOL;
  const damp = opts.damp ?? DEFAULT_DAMP;

  const hEff = new Float64Array(V);
  for (let i = 0; i < V; i++) hEff[i] = opts.fieldWeight * h[i];

  const fixedMask = new Uint8Array(V);
  const excludedMask = new Uint8Array(V);
  for (const i of opts.fixed) fixedMask[i] = 1;
  for (const i of opts.excluded) excludedMask[i] = 1;

  // Uniqueness against the fixed mons: a candidate sharing a site (species)
  // or a unique-track value (item) with anything pinned is not eligible to
  // fill a free slot.
  const constraints = buildConstraintSets(opts.fixed, model);
  const uniqInvalid = new Uint8Array(V);
  for (let i = 0; i < V; i++) {
    if (constraints.usedSites.has(model.siteOf[i])) {
      uniqInvalid[i] = 1;
      continue;
    }
    for (let t = 0; t < model.tracks.length; t++) {
      if (!model.tracks[t].unique) continue;
      const v = model.trackValues[i][t];
      if (v !== null && constraints.usedTrackValues[t].has(v)) {
        uniqInvalid[i] = 1;
        break;
      }
    }
  }

  const validMask = new Uint8Array(V);
  let validCount = 0;
  for (let i = 0; i < V; i++) {
    if (!fixedMask[i] && !excludedMask[i] && !uniqInvalid[i]) {
      validMask[i] = 1;
      validCount++;
    }
  }
  const nToFill = teamSize - opts.fixed.length;
  if (validCount < nToFill) return null;

  const m = new Float64Array(V);
  const mNew = new Float64Array(V);
  for (let i = 0; i < V; i++) {
    if (fixedMask[i]) m[i] = 1;
    else if (excludedMask[i]) m[i] = 0;
    else m[i] = sigmoid(hEff[i]);
  }

  let itersUsed = nIters;
  for (let it = 0; it < nIters; it++) {
    // mNew[i] = sigmoid(hEff[i] + J[i, :] @ m)
    for (let i = 0; i < V; i++) {
      const base = i * V;
      let s = hEff[i];
      for (let j = 0; j < V; j++) s += J[base + j] * m[j];
      mNew[i] = sigmoid(s);
    }
    // Apply clamps.
    for (let i = 0; i < V; i++) {
      if (fixedMask[i]) mNew[i] = 1;
      else if (excludedMask[i]) mNew[i] = 0;
    }
    // Convergence check over free slots before damping (matches Python).
    let maxDelta = 0;
    for (let i = 0; i < V; i++) {
      if (!fixedMask[i] && !excludedMask[i]) {
        const d = Math.abs(mNew[i] - m[i]);
        if (d > maxDelta) maxDelta = d;
      }
    }
    // Damp + clamp.
    for (let i = 0; i < V; i++) {
      m[i] = damp * mNew[i] + (1 - damp) * m[i];
    }
    for (let i = 0; i < V; i++) {
      if (fixedMask[i]) m[i] = 1;
      else if (excludedMask[i]) m[i] = 0;
    }
    if (maxDelta < tol) {
      itersUsed = it + 1;
      break;
    }
  }

  return { marginals: m, validMask, iters: itersUsed };
}
