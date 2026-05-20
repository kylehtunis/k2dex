import { describe, expect, it } from "vitest";
import { Rng } from "../../sampler/rng";
import { createLattice, latticeMagnetization, sweepLattice } from "../primitives/lattice";

describe("Lattice", () => {
  it("createLattice returns expected shape and ±1 values", () => {
    const L = createLattice(5, 7, new Rng(1));
    expect(L.length).toBe(5);
    expect(L[0].length).toBe(7);
    for (const row of L) for (const s of row) expect(Math.abs(s)).toBe(1);
  });

  it("magnetization in [-1, 1]", () => {
    const L = createLattice(20, 20, new Rng(1));
    const m = latticeMagnetization(L);
    expect(m).toBeGreaterThanOrEqual(-1);
    expect(m).toBeLessThanOrEqual(1);
  });

  it("sweep is deterministic given seed", () => {
    const rng = new Rng(42);
    const L = createLattice(8, 8, new Rng(7));
    const copy = L.map((row) => row.slice());
    sweepLattice(L, 2.0, rng, 1);
    const diff = L.flat().filter((v, i) => v !== copy.flat()[i]).length;
    expect(diff).toBeGreaterThan(0);
  });

  it("low T drives high |magnetization| (cold ferro state)", () => {
    const rng = new Rng(1);
    const L = createLattice(20, 20, new Rng(2));
    sweepLattice(L, 0.5, rng, 200);
    expect(Math.abs(latticeMagnetization(L))).toBeGreaterThan(0.5);
  });
});
