// Tiny graph type for /science pedagogy widgets. Binary spins s_i in {0, 1}
// (matches the inverse-Ising convention used throughout this repo, NOT the
// physics ±1 convention used in lattice.ts).
//
// Energy: H(s) = -h·s - 0.5 s'Js. Equivalently for binary spins,
// H = -Σ h_i s_i - Σ_{i<j} J_ij s_i s_j.
//
// exactMarginals enumerates all 2^V states. Use ONLY on tiny graphs (V ≤ 12 or so).

export interface Graph {
  V: number;
  /** Symmetric, zero diagonal. */
  J: number[][];
  h: number[];
  /** Convenience: edges as [i, j, J_ij] tuples with i < j. */
  edges: ReadonlyArray<readonly [number, number, number]>;
}

export type State = number[];

export function makeGraph(
  V: number,
  edges: ReadonlyArray<readonly [number, number, number]>,
  h: number[],
): Graph {
  if (h.length !== V) throw new Error(`h length ${h.length} != V ${V}`);
  const J: number[][] = Array.from({ length: V }, () => new Array(V).fill(0));
  for (const [i, j, w] of edges) {
    if (i === j) throw new Error("self-loops disallowed");
    J[i][j] = w;
    J[j][i] = w;
  }
  return { V, J, h: h.slice(), edges: edges.slice() };
}

export function graphEnergy(g: Graph, s: State): number {
  let e = 0;
  for (let i = 0; i < g.V; i++) e -= g.h[i] * s[i];
  for (const [i, j, w] of g.edges) e -= w * s[i] * s[j];
  return e;
}

/** Exact marginals via brute enumeration. Only safe for V ≤ ~12. */
export function exactMarginals(g: Graph, T: number): number[] {
  if (g.V > 16) throw new Error(`exactMarginals: V=${g.V} too large`);
  const beta = 1 / Math.max(T, 1e-9);
  const counts = new Array(g.V).fill(0);
  let Z = 0;
  const s: State = new Array(g.V).fill(0);
  for (let bits = 0; bits < 1 << g.V; bits++) {
    for (let i = 0; i < g.V; i++) s[i] = (bits >> i) & 1;
    const w = Math.exp(-beta * graphEnergy(g, s));
    Z += w;
    for (let i = 0; i < g.V; i++) if (s[i]) counts[i] += w;
  }
  return counts.map((c) => c / Z);
}
