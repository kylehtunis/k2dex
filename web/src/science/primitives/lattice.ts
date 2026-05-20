// 2D Ising spin lattice with periodic boundaries and single-spin Metropolis.
//
// This is a toy primitive for the /science page — independent from the
// production sampler in web/src/sampler/. Single-spin flips here are
// pedagogically natural (the canonical Ising demo) but are NOT what the
// Pokemon code uses (that's swap moves under a size constraint).

import type { Rng } from "../../sampler/rng";

export type Spin = -1 | 1;
export type Lattice = Spin[][];

export function createLattice(rows: number, cols: number, rng: Rng): Lattice {
  const L: Lattice = [];
  for (let i = 0; i < rows; i++) {
    const row: Spin[] = [];
    for (let j = 0; j < cols; j++) {
      row.push(rng.random() < 0.5 ? -1 : 1);
    }
    L.push(row);
  }
  return L;
}

export function latticeMagnetization(L: Lattice): number {
  let sum = 0;
  let n = 0;
  for (const row of L) {
    for (const s of row) {
      sum += s;
      n++;
    }
  }
  return sum / n;
}

function neighborSum(L: Lattice, i: number, j: number): number {
  const R = L.length;
  const C = L[0].length;
  const up = L[(i - 1 + R) % R][j];
  const down = L[(i + 1) % R][j];
  const left = L[i][(j - 1 + C) % C];
  const right = L[i][(j + 1) % C];
  return up + down + left + right;
}

/** Run `sweeps` sweeps (each sweep = R*C single-spin Metropolis proposals). */
export function sweepLattice(L: Lattice, T: number, rng: Rng, sweeps: number): void {
  const R = L.length;
  const C = L[0].length;
  const N = R * C;
  const beta = 1 / Math.max(T, 1e-9);
  for (let sw = 0; sw < sweeps; sw++) {
    for (let step = 0; step < N; step++) {
      const i = rng.integers(R);
      const j = rng.integers(C);
      const s = L[i][j];
      const dE = 2 * s * neighborSum(L, i, j);
      if (dE <= 0 || rng.random() < Math.exp(-beta * dE)) {
        L[i][j] = -s as Spin;
      }
    }
  }
}
