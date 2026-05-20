import { describe, expect, it } from "vitest";
import { energyAt, metropolisStep, project } from "../primitives/landscape";
import { Rng } from "../../sampler/rng";

describe("landscape primitive", () => {
  it("two wells at x=±1, y=0 with equal depth (symmetry)", () => {
    expect(energyAt(-1, 0)).toBeCloseTo(energyAt(1, 0), 9);
    // Wells should be lower-energy than the ridge between them.
    expect(energyAt(-1, 0)).toBeLessThan(energyAt(0, 0));
  });

  it("metropolisStep stays in domain", () => {
    const rng = new Rng(3);
    let p = { x: 0, y: 0 };
    for (let k = 0; k < 200; k++) p = metropolisStep(p, 1.0, 0.3, rng);
    expect(p.x).toBeGreaterThanOrEqual(-2.4);
    expect(p.x).toBeLessThanOrEqual(2.4);
    expect(p.y).toBeGreaterThanOrEqual(-2.4);
    expect(p.y).toBeLessThanOrEqual(2.4);
  });

  it("project: origin maps near center", () => {
    const P = { scaleXY: 50, scaleZ: 30, centerU: 200, centerV: 200 };
    const { u, v } = project({ x: 0, y: 0, z: 0 }, P);
    expect(u).toBeCloseTo(200, 9);
    expect(v).toBeCloseTo(200, 9);
  });
});
