import { describe, expect, it } from "vitest";
import { exactMarginals, graphEnergy, makeGraph } from "../primitives/graph";

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
