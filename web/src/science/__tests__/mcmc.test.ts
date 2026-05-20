import { describe, expect, it } from "vitest";
import { Rng } from "../../sampler/rng";
import { exactMarginals, makeGraph } from "../primitives/graph";
import { runChain } from "../primitives/mcmc";

describe("MCMC single-spin Metropolis on a graph", () => {
  it("estimated marginals approach exact marginals on a 4-node graph at T=1", () => {
    const g = makeGraph(
      4,
      [[0, 1, 1.5], [1, 2, 1.5], [2, 3, 1.5]],
      [0.0, 0.0, 0.0, 0.0],
    );
    const exact = exactMarginals(g, 1.0);
    const rng = new Rng(123);
    const { marginals } = runChain(g, 1.0, rng, { sweeps: 5000, burnIn: 1000 });
    for (let i = 0; i < g.V; i++) {
      expect(Math.abs(marginals[i] - exact[i])).toBeLessThan(0.05);
    }
  });

  it("history length matches sweeps", () => {
    const g = makeGraph(3, [[0, 1, 1]], [0, 0, 0]);
    const rng = new Rng(1);
    const { history } = runChain(g, 1.0, rng, { sweeps: 50, burnIn: 0, record: true });
    expect(history.length).toBe(50);
    expect(history[0].length).toBe(g.V);
  });
});
