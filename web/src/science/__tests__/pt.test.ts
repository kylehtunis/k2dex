import { describe, expect, it } from "vitest";
import { Rng } from "../../sampler/rng";
import { exactMarginals, makeGraph } from "../primitives/graph";
import { runPT } from "../primitives/pt";

describe("Toy parallel tempering", () => {
  it("PT cold-chain marginals approach exact at low T on a bimodal graph", () => {
    const g = makeGraph(
      6,
      [
        [0, 1, 2.0],
        [1, 2, 2.0],
        [3, 4, 2.0],
        [4, 5, 2.0],
        [2, 3, -0.5],
      ],
      [0, 0, 0, 0, 0, 0],
    );
    const exact = exactMarginals(g, 0.7);
    const rng = new Rng(42);
    const { coldMarginals, swaps } = runPT(g, [0.7, 1.0, 1.5, 2.5], rng, {
      sweeps: 4000,
      burnIn: 1000,
      swapInterval: 5,
    });
    for (let i = 0; i < g.V; i++) {
      expect(Math.abs(coldMarginals[i] - exact[i])).toBeLessThan(0.08);
    }
    expect(swaps).toBeGreaterThan(0);
  });
});
