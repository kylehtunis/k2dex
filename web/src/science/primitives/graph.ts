// Tiny graph type for /science pedagogy widgets. Binary spins s_i in {0, 1}
// (matches the inverse-Ising convention used throughout this repo, NOT the
// physics ±1 convention used in lattice.ts).
//
// Energy: H(s) = -h·s - 0.5 s'Js. Equivalently for binary spins,
// H = -Σ h_i s_i - Σ_{i<j} J_ij s_i s_j.
//
// exactMarginals enumerates all 2^V states. Use ONLY on tiny graphs (V ≤ 12 or so).

import { Rng } from "../../sampler/rng";

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

/**
 * Erdős–Rényi giant component with Gaussian edge weights. Builds an ER(n, p)
 * graph with seeded RNG, finds the largest connected component, and returns
 * it as a Graph (h = 0 throughout) along with a deterministic circular layout.
 * Edge weights are i.i.d. N(0, sigma²) sampled via Box-Muller.
 */
export function randomGraph(
  seed: number,
  n: number,
  p: number,
  sigma: number,
): { graph: Graph; positions: { x: number; y: number }[] } {
  const rng = new Rng(seed);
  const adj: number[][] = Array.from({ length: n }, () => []);
  const rawEdges: Array<[number, number, number]> = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (rng.random() < p) {
        const u1 = Math.max(rng.random(), 1e-12);
        const u2 = rng.random();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        rawEdges.push([i, j, sigma * z]);
        adj[i].push(j);
        adj[j].push(i);
      }
    }
  }
  const comp = new Array<number>(n).fill(-1);
  const sizes: number[] = [];
  for (let start = 0; start < n; start++) {
    if (comp[start] !== -1) continue;
    const id = sizes.length;
    const stack = [start];
    let size = 0;
    while (stack.length) {
      const v = stack.pop()!;
      if (comp[v] !== -1) continue;
      comp[v] = id;
      size++;
      for (const w of adj[v]) if (comp[w] === -1) stack.push(w);
    }
    sizes.push(size);
  }
  let giant = 0;
  for (let k = 1; k < sizes.length; k++) if (sizes[k] > sizes[giant]) giant = k;
  const keep: number[] = [];
  const remap = new Array<number>(n).fill(-1);
  for (let v = 0; v < n; v++) {
    if (comp[v] === giant) {
      remap[v] = keep.length;
      keep.push(v);
    }
  }
  const V = keep.length;
  const edges: Array<[number, number, number]> = rawEdges
    .filter(([i, j]) => remap[i] !== -1 && remap[j] !== -1)
    .map(([i, j, w]) => [remap[i], remap[j], w]);
  const h = new Array<number>(V).fill(0);
  const graph = makeGraph(V, edges, h);
  const positions = Array.from({ length: V }, (_, k) => {
    const theta = (2 * Math.PI * k) / V - Math.PI / 2;
    return { x: Math.cos(theta), y: Math.sin(theta) };
  });
  return { graph, positions };
}

/**
 * One sweep = V single-spin Metropolis proposals on a graph state.
 * Mutates `s` in place. Pedagogical mirror of sweepLattice.
 */
export function sweepGraph(s: State, g: Graph, T: number, rng: Rng, sweeps: number): void {
  const beta = 1 / Math.max(T, 1e-9);
  for (let sw = 0; sw < sweeps; sw++) {
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
