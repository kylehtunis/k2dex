import { describe, expect, it } from "vitest";
import { exactMarginals, makeGraph } from "../primitives/graph";
import { meanFieldIterate } from "../primitives/mf";

describe("Mean-field on a small graph", () => {
  it("two-node weak coupling: MF tight vs exact", () => {
    const g = makeGraph(2, [[0, 1, 0.3]], [0.1, -0.1]);
    const exact = exactMarginals(g, 1.0);
    const { marginals, converged } = meanFieldIterate(g, 1.0, { maxIters: 200 });
    expect(converged).toBe(true);
    expect(Math.abs(marginals[0] - exact[0])).toBeLessThan(0.02);
    expect(Math.abs(marginals[1] - exact[1])).toBeLessThan(0.02);
  });

  it("records iteration trajectory when requested", () => {
    const g = makeGraph(3, [[0, 1, 0.5], [1, 2, 0.5]], [0, 0, 0]);
    const { history } = meanFieldIterate(g, 1.0, { maxIters: 50, record: true });
    expect(history.length).toBeGreaterThan(0);
    expect(history[0].length).toBe(g.V);
  });
});
