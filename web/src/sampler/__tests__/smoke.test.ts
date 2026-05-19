// Smoke tests for the parts of the sampler that the parity baseline
// can't cover because they're stochastic. We just check that the
// samplers run end-to-end on the synthetic model, return well-formed
// output, and respect basic invariants (team size, fixed honored,
// uniqueness on Phase 3 features).

import { describe, expect, it } from "vitest";
import type { IsingModel } from "../types";
import { swapMcmc } from "../swap";
import { parallelTemperedMcmc } from "../pt";
import { unpackLowerTriangle } from "../model";

function buildSyntheticModel(): IsingModel {
  const V = 12;
  const TEAM_SIZE = 4;
  const speciesOf = ["A","A","B","B","C","C","D","D","E","E","F","F"];
  const itemOf: (string | null)[] = [null,"x",null,"y",null,"x",null,"y",null,"z",null,null];

  // J built from packed lower-triangle to also exercise unpackLowerTriangle.
  const lowerN = (V * (V - 1)) / 2;
  const lowerFlat = new Float32Array(lowerN);
  let k = 0;
  for (let i = 1; i < V; i++) {
    for (let j = 0; j < i; j++) {
      lowerFlat[k++] = 0.5 * Math.sin(0.7 * (i + 1)) * Math.cos(0.3 * (j + 1));
    }
  }
  const J = unpackLowerTriangle(lowerFlat, V);

  const h = new Float64Array(V);
  for (let i = 0; i < V; i++) h[i] = -0.6 + 0.1 * i;
  const m = new Float64Array(V);
  for (let i = 0; i < V; i++) m[i] = 0.05 + 0.04 * i;

  const vocab = speciesOf.map((s, i) => itemOf[i] === null ? s : `${s} @ ${itemOf[i]}`);
  const indexOf = new Map<string, number>();
  for (let i = 0; i < vocab.length; i++) indexOf.set(vocab[i], i);

  return {
    V, teamSize: TEAM_SIZE, vocab, speciesOf, itemOf, m, J, h, indexOf,
    nCorpusTeams: 0, name: "synthetic",
  };
}

function assertValidTeam(team: readonly number[], model: IsingModel, fixed: number[]) {
  expect(team.length).toBe(model.teamSize);
  for (const f of fixed) expect(team).toContain(f);
  // No duplicate species
  const species = new Set<string>();
  for (const i of team) {
    expect(species.has(model.speciesOf[i])).toBe(false);
    species.add(model.speciesOf[i]);
  }
  // No duplicate non-null items
  const items = new Set<string>();
  for (const i of team) {
    const it = model.itemOf[i];
    if (it === null) continue;
    expect(items.has(it)).toBe(false);
    items.add(it);
  }
}

describe("swapMcmc smoke", () => {
  it("produces valid teams and respects fixed", () => {
    const model = buildSyntheticModel();
    const fixed = [0];
    const result = swapMcmc(model, {
      fixed, excluded: [11], fieldWeight: 0.5,
      nSteps: 200, temperature: 1.0, seed: 12345,
    });
    expect(result).not.toBeNull();
    expect(result!.samples.length).toBe(200);
    expect(result!.acceptRate).toBeGreaterThanOrEqual(0);
    expect(result!.acceptRate).toBeLessThanOrEqual(1);
    // Every sample must be a valid team
    for (const team of result!.samples) assertValidTeam(team, model, fixed);
  });

  it("returns null when over-constrained", () => {
    const model = buildSyntheticModel();
    // Pin species A/B/C, exclude D/E/F; the only available indices
    // 1, 3, 5 are item-x/y/x variants of A/B/C respectively, all of
    // which collide with fixed species. nToFill=1 with no valid
    // candidate → initializeState exhausts retries → null.
    const result = swapMcmc(model, {
      fixed: [0, 2, 4], excluded: [6, 7, 8, 9, 10, 11], fieldWeight: 1.0,
      nSteps: 10, temperature: 1.0, seed: 1,
    });
    expect(result).toBeNull();
  });
});

describe("parallelTemperedMcmc smoke", () => {
  it("produces valid cold-chain samples post burn-in", () => {
    const model = buildSyntheticModel();
    const fixed: number[] = [];
    const result = parallelTemperedMcmc(model, {
      fixed, excluded: [], fieldWeight: 1.0,
      tLadder: [1.0, 1.5, 2.5, 4.0],
      nSteps: 500, burnIn: 100, swapInterval: 10, seed: 999,
    });
    expect(result).not.toBeNull();
    expect(result!.samples.length).toBe(400);
    expect(result!.localAccept).toBeGreaterThanOrEqual(0);
    expect(result!.localAccept).toBeLessThanOrEqual(1);
    expect(result!.swapAccept).toBeGreaterThanOrEqual(0);
    expect(result!.swapAccept).toBeLessThanOrEqual(1);
    for (const team of result!.samples) assertValidTeam(team, model, fixed);
  });
});

describe("unpackLowerTriangle", () => {
  it("reconstructs a symmetric matrix with zero diagonal", () => {
    const V = 5;
    const flat = new Float32Array([
      1.0,            // J[1,0]
      2.0, 3.0,       // J[2,0], J[2,1]
      4.0, 5.0, 6.0,  // J[3,0..2]
      7.0, 8.0, 9.0, 10.0,  // J[4,0..3]
    ]);
    const J = unpackLowerTriangle(flat, V);
    expect(J.length).toBe(V * V);
    // Symmetric
    for (let i = 0; i < V; i++) {
      for (let j = 0; j < V; j++) {
        expect(J[i * V + j]).toBeCloseTo(J[j * V + i]);
      }
    }
    // Zero diagonal
    for (let i = 0; i < V; i++) expect(J[i * V + i]).toBe(0);
    // Spot-check values
    expect(J[1 * V + 0]).toBeCloseTo(1.0);
    expect(J[3 * V + 1]).toBeCloseTo(5.0);
    expect(J[4 * V + 3]).toBeCloseTo(10.0);
  });

  it("throws on length mismatch", () => {
    expect(() => unpackLowerTriangle(new Float32Array(5), 4)).toThrow();
  });
});
