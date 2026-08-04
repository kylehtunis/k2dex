// Smoke tests for the parts of the sampler that the parity baseline
// can't cover because they're stochastic. We just check that the
// samplers run end-to-end on the synthetic model, return well-formed
// output, and respect basic invariants (team size, fixed honored,
// uniqueness on (species, item) features).

import { describe, expect, it } from "vitest";
import type { IsingModel } from "../types";
import { initChain, swapMcmc } from "../swap";
import { parallelTemperedMcmc } from "../pt";
import { unpackLowerTriangle, factoredFromSpeciesItem, withInactiveTracks } from "../model";
import {
  availableIndices,
  buildConstraintSets,
  resolveSitePins,
  teamEnergy,
} from "../energy";
import {
  buildSiteTables,
  pottsMoveSets,
  pottsSpeciesSwap,
  pottsTrackReroll,
  type PottsContext,
} from "../potts";
import { Rng } from "../rng";

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

  const factored = factoredFromSpeciesItem(speciesOf, itemOf);
  return {
    id: "synthetic", displayName: "Synthetic", regulation: "test",
    featureDimensions: 2, latestTournamentDate: "",
    V, teamSize: TEAM_SIZE, vocab, speciesOf, itemOf, ...factored, m, J, h, indexOf,
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

describe("parallelTemperedMcmc site pins", () => {
  it("keeps the site-pinned species on every sample and stays valid", () => {
    const model = buildSyntheticModel();
    const result = parallelTemperedMcmc(model, {
      fixed: [], fixedSites: [0], excluded: [], fieldWeight: 1.0,
      tLadder: [1.0, 1.5, 2.5, 4.0],
      nSteps: 400, burnIn: 100, swapInterval: 10, seed: 7,
    });
    expect(result).not.toBeNull();
    for (const team of result!.samples) {
      assertValidTeam(team, model, []);
      // Site 0 (species "A") is present in every sample; the item may vary.
      expect(team.some((i) => model.siteOf[i] === 0)).toBe(true);
    }
  });

  it("combines a feature pin and a site pin", () => {
    const model = buildSyntheticModel();
    // Feature pin index 3 = "B @ y" (locked exactly); site pin site 3 = "D".
    const result = parallelTemperedMcmc(model, {
      fixed: [3], fixedSites: [3], excluded: [], fieldWeight: 1.0,
      tLadder: [1.0, 2.0, 4.0],
      nSteps: 300, burnIn: 50, swapInterval: 10, seed: 42,
    });
    expect(result).not.toBeNull();
    for (const team of result!.samples) {
      assertValidTeam(team, model, [3]);
      expect(team.some((i) => model.siteOf[i] === 3)).toBe(true);
    }
  });
});

describe("anchor-field tilt", () => {
  it("stays valid with anchorStrength > 1 and mixed pins", () => {
    const model = buildSyntheticModel();
    const result = parallelTemperedMcmc(model, {
      fixed: [3], fixedSites: [0], excluded: [], fieldWeight: 1.0,
      tLadder: [1.0, 2.0, 4.0],
      nSteps: 400, burnIn: 100, swapInterval: 10, seed: 21,
      anchorStrength: 2.0,
    });
    expect(result).not.toBeNull();
    for (const team of result!.samples) {
      assertValidTeam(team, model, [3]);
      expect(team.some((i) => model.siteOf[i] === 0)).toBe(true);
    }
  });

  it("raises mean pin-integration vs alpha = 1", () => {
    const model = buildSyntheticModel();
    const pin = 9; // "E @ z"
    const meanT = (alpha: number): number => {
      const result = parallelTemperedMcmc(model, {
        fixed: [pin], excluded: [], fieldWeight: 1.0,
        tLadder: [1.0, 1.5, 2.5, 4.0],
        nSteps: 2000, burnIn: 500, swapInterval: 10, seed: 77,
        anchorStrength: alpha,
      });
      expect(result).not.toBeNull();
      let total = 0;
      for (const team of result!.samples) {
        for (const i of team) if (i !== pin) total += model.J[pin * model.V + i];
      }
      return total / result!.samples.length;
    };
    expect(meanT(3.0)).toBeGreaterThan(meanT(1.0));
  });

  it("keeps the tracked chain energy consistent with recomputed H_alpha", () => {
    const model = buildSyntheticModel();
    const { V, J, h } = model;
    const alpha = 2.0;
    const fixed = [3]; // "B @ y" feature pin
    const seeds = resolveSitePins(model, [0], fixed, []);
    expect(seeds).not.toBeNull();
    const available = availableIndices(model, fixed, []);
    const nToFill = model.teamSize - fixed.length - seeds!.length;

    const hEff = new Float64Array(V);
    for (let i = 0; i < V; i++) hEff[i] = h[i];
    const constraints = buildConstraintSets([...fixed, ...seeds!], model);
    const tables = buildSiteTables(model);
    const avail = new Uint8Array(V).fill(1);
    for (const f of fixed) avail[f] = 0;
    const lockedSlots = new Set<number>();
    for (let i = 0; i < seeds!.length; i++) lockedSlots.add(i);
    const ctx: PottsContext = {
      fixed,
      avail,
      tables,
      lockedSlots,
      anchorStrength: alpha,
      ...pottsMoveSets(tables, avail, nToFill + seeds!.length, lockedSlots),
    };

    // H_alpha recomputed from scratch: untilted team energy on hEff minus the
    // (alpha-1)-weighted pin<->free cross-coupling sum.
    const recompute = (c: { stateF: Float64Array; onNf: number[] }): number => {
      const pins: number[] = [...fixed];
      const free: number[] = [];
      for (let s = 0; s < c.onNf.length; s++) {
        (lockedSlots.has(s) ? pins : free).push(c.onNf[s]);
      }
      let cross = 0;
      for (const p of pins) for (const j of free) cross += J[p * V + j];
      return teamEnergy(c.stateF, J, hEff, V) - (alpha - 1) * cross;
    };

    const rng = new Rng(4242);
    const chain = initChain(model, fixed, available, nToFill, constraints, hEff, rng, seeds!);
    expect(chain).not.toBeNull();
    chain!.energy = recompute(chain!);
    for (let step = 0; step < 600; step++) {
      if (rng.random() < 0.5) {
        pottsTrackReroll(chain!, model, hEff, 1.0, tables, ctx, rng);
      } else {
        pottsSpeciesSwap(chain!, model, hEff, 1.0, tables, ctx, rng);
      }
    }
    expect(chain!.energy).toBeCloseTo(recompute(chain!), 8);
  });
});

describe("species-only sampling (degenerate item track)", () => {
  it("keeps species unique with no reroll; items may collide", () => {
    const base = buildSyntheticModel();
    const itemTrack = base.tracks.findIndex((t) => t.name === "item");
    expect(itemTrack).toBeGreaterThanOrEqual(0);
    // Deactivate the item track: it becomes degenerate (non-unique). With
    // pReroll 0 the sampler never rerolls it; species-swaps still marginalize.
    const model = withInactiveTracks(base, [itemTrack]);
    const result = parallelTemperedMcmc(model, {
      fixed: [], excluded: [], fieldWeight: 1.0,
      tLadder: [1.0, 2.0, 4.0],
      nSteps: 300, burnIn: 50, swapInterval: 10, seed: 3,
      pReroll: 0,
    });
    expect(result).not.toBeNull();
    for (const team of result!.samples) {
      expect(team.length).toBe(model.teamSize);
      // Species (site) uniqueness always holds; item uniqueness is relaxed.
      const species = new Set<string>();
      for (const i of team) {
        expect(species.has(model.speciesOf[i])).toBe(false);
        species.add(model.speciesOf[i]);
      }
    }
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
