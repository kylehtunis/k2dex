// Smoke tests for the parts of the sampler that the parity baseline
// can't cover because they're stochastic. We just check that the
// samplers run end-to-end on the synthetic model, return well-formed
// output, and respect basic invariants (team size, fixed honored,
// uniqueness on Phase 3 features).

import { describe, expect, it } from "vitest";
import type { IsingModel } from "../types";
import { initChain, swapMcmc } from "../swap";
import { parallelTemperedMcmc } from "../pt";
import type { TrackDef } from "../types";
import { unpackLowerTriangle, factoredFromSpeciesItem, deriveFactored, withInactiveTracks } from "../model";
import {
  availableIndices,
  buildConstraintSets,
  resolveSitePins,
  teamEnergy,
} from "../energy";
import {
  buildSiteTables,
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
    latestTournamentDate: "",
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
    const ctx: PottsContext = { fixed, avail, tables, lockedSlots, anchorStrength: alpha };

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

// 4 species carrying item + ability tracks; P's item x is split across two
// abilities so a partial pin (item locked, ability free) has room to reroll.
//   P(0): (x,a)(x,b)(y,a)  Q(1): (x,a)(z,b)  R(2): (y,b)(w,a)  S(3): (u,a)(v,b)
function buildTwoTrackModel(teamSize = 3): IsingModel {
  const sites = ["P", "Q", "R", "S"];
  const siteOf = [0, 0, 0, 1, 1, 2, 2, 3, 3];
  const items = ["x", "x", "y", "x", "z", "y", "w", "u", "v"];
  const abils = ["a", "b", "a", "a", "b", "b", "a", "a", "b"];
  const trackValues: (string | null)[][] = items.map((it, i) => [it, abils[i]]);
  const tracks: TrackDef[] = [
    { name: "item", cardinality: 1, crossSlotUnique: true, withinSlotUnique: false },
    { name: "ability", cardinality: 1, crossSlotUnique: false, withinSlotUnique: false },
  ];
  const V = siteOf.length;
  const lowerN = (V * (V - 1)) / 2;
  const lowerFlat = new Float32Array(lowerN);
  let k = 0;
  for (let i = 1; i < V; i++) {
    for (let j = 0; j < i; j++) {
      lowerFlat[k++] = 0.4 * Math.sin(0.8 * (i + 1)) * Math.cos(0.4 * (j + 1));
    }
  }
  const J = unpackLowerTriangle(lowerFlat, V);
  const h = new Float64Array(V);
  for (let i = 0; i < V; i++) h[i] = -0.4 + 0.1 * i;
  const m = new Float64Array(V).fill(0.15);
  const vocab = trackValues.map((tv, i) => `${sites[siteOf[i]]} @ ${tv[0]} (${tv[1]})`);
  const indexOf = new Map<string, number>();
  vocab.forEach((v, i) => indexOf.set(v, i));
  const derived = deriveFactored(sites, siteOf, trackValues);
  return {
    id: "t2", displayName: "t2", regulation: "test", latestTournamentDate: "",
    V, teamSize, vocab, sites, siteOf, tracks, trackValues, ...derived,
    m, J, h, indexOf, nCorpusTeams: 0, name: "t2",
  };
}

describe("partial pins (species + one track locked, others free)", () => {
  it("keeps the pinned item fixed while the ability rerolls", () => {
    const model = buildTwoTrackModel();
    // Pin site P (0) with item x locked, ability free.
    const result = parallelTemperedMcmc(model, {
      fixed: [], excluded: [], fieldWeight: 1.0,
      fixedSites: [0],
      sitePinTrackValues: [["x", null]],
      tLadder: [1.0, 2.0, 4.0],
      nSteps: 400, burnIn: 50, swapInterval: 10, seed: 9,
      pReroll: 0.6,
    });
    expect(result).not.toBeNull();
    const abilitiesSeen = new Set<string | null>();
    for (const team of result!.samples) {
      expect(team.length).toBe(model.teamSize);
      // Exactly one P feature present, and it holds the pinned item x.
      const pFeats = team.filter((i) => model.siteOf[i] === 0);
      expect(pFeats.length).toBe(1);
      expect(model.trackValues[pFeats[0]][0]).toBe("x"); // item locked
      abilitiesSeen.add(model.trackValues[pFeats[0]][1]);
      // Species (site) uniqueness across the team.
      const seen = new Set<number>();
      for (const i of team) {
        expect(seen.has(model.siteOf[i])).toBe(false);
        seen.add(model.siteOf[i]);
      }
    }
    // The free ability track actually explores both of x's abilities (a, b).
    expect(abilitiesSeen.size).toBeGreaterThan(1);
  });

  it("a pure site pin (no track values) still rerolls every track", () => {
    const model = buildTwoTrackModel();
    const result = parallelTemperedMcmc(model, {
      fixed: [], excluded: [], fieldWeight: 1.0,
      fixedSites: [0], // no sitePinTrackValues -> fully free
      tLadder: [1.0, 2.0, 4.0],
      nSteps: 400, burnIn: 50, swapInterval: 10, seed: 5,
      pReroll: 0.6,
    });
    expect(result).not.toBeNull();
    const itemsSeen = new Set<string | null>();
    for (const team of result!.samples) {
      const pFeats = team.filter((i) => model.siteOf[i] === 0);
      expect(pFeats.length).toBe(1);
      itemsSeen.add(model.trackValues[pFeats[0]][0]);
    }
    // Item is free here, so P visits more than one item value.
    expect(itemsSeen.size).toBeGreaterThan(1);
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
