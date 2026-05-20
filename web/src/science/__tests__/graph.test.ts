import { describe, expect, it } from "vitest";
import {
  exactMarginals,
  graphEnergy,
  makeGraph,
  randomGraph,
  sweepGraph,
} from "../primitives/graph";
import { Rng } from "../../sampler/rng";

describe("Graph + exact enumeration", () => {
  it("two-node ferromagnet: positive J pushes marginals up", () => {
    const g = makeGraph(2, [[0, 1, 2.0]], [0.5, 0.5]);
    const m = exactMarginals(g, 1.0);
    expect(m[0]).toBeGreaterThan(0.7);
    expect(m[1]).toBeGreaterThan(0.7);
    expect(Math.abs(m[0] - m[1])).toBeLessThan(1e-9);
  });

  it("two-node antiferromagnet at h=0: marginals equal by symmetry, <0.5", () => {
    // H(1,1) = +2 (penalised), H(0,1)=H(1,0)=H(0,0)=0.
    // m = (1+e^{-2})/(3+e^{-2}) ≈ 0.362.
    const g = makeGraph(2, [[0, 1, -2.0]], [0, 0]);
    const m = exactMarginals(g, 1.0);
    expect(Math.abs(m[0] - m[1])).toBeLessThan(1e-9);
    expect(m[0]).toBeLessThan(0.5);
  });

  it("graphEnergy: H(0,0) = 0 with h=0", () => {
    const g = makeGraph(2, [[0, 1, 1.0]], [0, 0]);
    expect(graphEnergy(g, [0, 0])).toBeCloseTo(0, 12);
  });

  it("graphEnergy: H(1,1) = -h0 - h1 - J for state=[1,1]", () => {
    const g = makeGraph(2, [[0, 1, 1.5]], [0.2, 0.3]);
    expect(graphEnergy(g, [1, 1])).toBeCloseTo(-0.5 - 1.5, 12);
  });
});

describe("randomGraph + sweepGraph", () => {
  it("randomGraph returns a connected giant component with positions", () => {
    const { graph, positions } = randomGraph(7, 16, 0.3, 1.0);
    expect(graph.V).toBeGreaterThan(0);
    expect(graph.V).toBeLessThanOrEqual(16);
    expect(positions.length).toBe(graph.V);
    // All listed edges must reference valid vertices.
    for (const [i, j] of graph.edges) {
      expect(i).toBeGreaterThanOrEqual(0);
      expect(j).toBeLessThan(graph.V);
    }
    // Connectedness: BFS from node 0 should reach all vertices.
    const seen = new Array<boolean>(graph.V).fill(false);
    const stack = [0];
    seen[0] = true;
    while (stack.length) {
      const v = stack.pop()!;
      for (let w = 0; w < graph.V; w++) {
        if (!seen[w] && graph.J[v][w] !== 0) {
          seen[w] = true;
          stack.push(w);
        }
      }
    }
    expect(seen.every(Boolean)).toBe(true);
  });

  it("randomGraph is deterministic for a given seed", () => {
    const a = randomGraph(3, 12, 0.4, 1.0);
    const b = randomGraph(3, 12, 0.4, 1.0);
    expect(a.graph.V).toBe(b.graph.V);
    expect(a.graph.J).toEqual(b.graph.J);
  });

  it("sweepGraph mutates state and respects T (high T → state changes a lot)", () => {
    const { graph } = randomGraph(11, 14, 0.35, 1.0);
    const rng = new Rng(1);
    const s = new Array<number>(graph.V).fill(0);
    sweepGraph(s, graph, 5.0, rng, 50);
    // At high T many spins should flip away from initial all-zero state.
    const ones = s.filter((x) => x === 1).length;
    expect(ones).toBeGreaterThan(0);
  });
});
