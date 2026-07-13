// Smoke + invariant tests for the universal Potts move kernel (potts.ts).
// The deterministic pieces (buildSiteTables, siteConditional) are parity-gated
// against sampling.py; here we check their basic shape plus the stochastic
// moves' team invariants and the species-only degeneracy.

import { describe, expect, it } from "vitest";
import type { IsingModel } from "../types";
import { factoredFromSpeciesItem, unpackLowerTriangle } from "../model";
import {
  buildSiteTables,
  siteConditional,
  pottsSpeciesSwap,
  pottsTrackReroll,
  type PottsContext,
} from "../potts";
import { buildConstraintSets } from "../energy";
import { initChain } from "../swap";
import { Rng } from "../rng";

function buildModel(
  speciesOf: string[],
  itemOf: (string | null)[],
  teamSize: number,
): IsingModel {
  const V = speciesOf.length;
  const lowerN = (V * (V - 1)) / 2;
  const lowerFlat = new Float32Array(lowerN);
  let k = 0;
  for (let i = 1; i < V; i++) {
    for (let j = 0; j < i; j++) {
      lowerFlat[k++] = 0.4 * Math.sin(0.9 * (i + 1)) * Math.cos(0.5 * (j + 1));
    }
  }
  const J = unpackLowerTriangle(lowerFlat, V);
  const h = new Float64Array(V);
  for (let i = 0; i < V; i++) h[i] = -0.5 + 0.13 * i;
  const m = new Float64Array(V);
  for (let i = 0; i < V; i++) m[i] = 0.05 + 0.03 * i;
  const vocab = speciesOf.map((s, i) => (itemOf[i] === null ? s : `${s} @ ${itemOf[i]}`));
  const indexOf = new Map<string, number>();
  vocab.forEach((v, i) => indexOf.set(v, i));
  const factored = factoredFromSpeciesItem(speciesOf, itemOf);
  return {
    id: "t", displayName: "t", regulation: "test", featureDimensions: 2,
    latestTournamentDate: "", V, teamSize, vocab, speciesOf, itemOf, ...factored,
    m, J, h, indexOf, nCorpusTeams: 0, name: "t",
  };
}

// 6 species × 2 items (F itemless variants), team size 4.
function speciesItemModel(): IsingModel {
  const species: string[] = [];
  const items: (string | null)[] = [];
  for (const s of ["A", "B", "C", "D", "E", "F"]) {
    species.push(s, s);
  }
  items.push("x", "y", "y", "z", "z", "x", "x", "w", "w", "v", null, null);
  return buildModel(species, items, 4);
}

function speciesOnlyModel(): IsingModel {
  const species = ["A", "B", "C", "D", "E", "F"];
  const items: (string | null)[] = species.map(() => null);
  return buildModel(species, items, 4);
}

function ctxFor(model: IsingModel, fixed: number[], excluded: number[]): PottsContext {
  const tables = buildSiteTables(model);
  const avail = new Uint8Array(model.V).fill(1);
  for (const e of excluded) avail[e] = 0;
  for (const f of fixed) avail[f] = 0;
  return { fixed, avail, tables };
}

function assertValid(team: number[], model: IsingModel) {
  const sites = new Set<number>();
  const items = new Set<string>();
  for (const i of team) {
    expect(sites.has(model.siteOf[i])).toBe(false);
    sites.add(model.siteOf[i]);
    const it = model.itemOf[i];
    if (it !== null) {
      expect(items.has(it)).toBe(false);
      items.add(it);
    }
  }
}

describe("buildSiteTables", () => {
  it("groups features by site and assigns item ids by first appearance", () => {
    const model = speciesItemModel();
    const t = buildSiteTables(model);
    expect(t.nSites).toBe(6);
    // Each species owns two consecutive features.
    expect(t.siteFeatures[0]).toEqual([0, 1]);
    expect(t.siteFeatures[5]).toEqual([10, 11]);
    // itemId: "x" first at index 0 -> 0, "y" at 1 -> 1, etc.; itemless -> -1.
    expect(t.itemId[0]).toBe(0); // x
    expect(t.itemId[1]).toBe(1); // y
    expect(t.itemId[10]).toBe(-1); // itemless
    expect(t.itemId[11]).toBe(-1); // itemless
  });

  it("assigns all -1 item ids for a species-only model", () => {
    const t = buildSiteTables(speciesOnlyModel());
    expect(t.nSites).toBe(6);
    expect([...t.itemId]).toEqual([-1, -1, -1, -1, -1, -1]);
    for (let s = 0; s < 6; s++) expect(t.siteFeatures[s]).toEqual([s]);
  });
});

describe("siteConditional", () => {
  it("excludes candidates holding an item already retained", () => {
    const model = speciesItemModel();
    const tables = buildSiteTables(model);
    const hEff = new Float64Array(model.V);
    for (let i = 0; i < model.V; i++) hEff[i] = model.h[i];
    const avail = new Uint8Array(model.V).fill(1);
    // Retain feature 0 (A @ x, itemId 0). Site F's feature 10/11 are itemless
    // (never excluded); site B (feats 2,3 items y,y) — item y id is 1, not held.
    const cond = siteConditional(1, [0], [tables.itemId[0]], model, hEff, 1, tables, avail);
    // Site 1 = B, feats [2,3], items y (id1) both -> not conflicting with x.
    expect(cond.feats).toEqual([2, 3]);
    expect(cond.valid).toEqual([true, true]);
    expect(Number.isFinite(cond.logZ)).toBe(true);
  });

  it("degenerates to a single entry for a species-only site", () => {
    const model = speciesOnlyModel();
    const tables = buildSiteTables(model);
    const hEff = new Float64Array(model.V);
    for (let i = 0; i < model.V; i++) hEff[i] = model.h[i];
    const avail = new Uint8Array(model.V).fill(1);
    const cond = siteConditional(2, [0, 1], [-1, -1], model, hEff, 1, tables, avail);
    expect(cond.feats).toEqual([2]);
    expect(cond.valid).toEqual([true]);
    // Single-entry logZ equals its neg-energy.
    expect(cond.logZ).toBeCloseTo(cond.negE[0], 12);
  });
});

describe("pottsSpeciesSwap / pottsTrackReroll invariants", () => {
  it("keeps teams valid across many species-item moves", () => {
    const model = speciesItemModel();
    const rng = new Rng(7);
    const ctx = ctxFor(model, [], []);
    const constraints = buildConstraintSets([], model);
    const hEff = new Float64Array(model.V);
    for (let i = 0; i < model.V; i++) hEff[i] = model.h[i];
    const chain = initChain(model, [], [...Array(model.V).keys()], 4, constraints, hEff, rng)!;
    expect(chain).not.toBeNull();
    for (let s = 0; s < 400; s++) {
      if (rng.random() < 0.5) {
        pottsTrackReroll(chain, model, hEff, 1.0, ctx.tables, ctx, rng);
      } else {
        pottsSpeciesSwap(chain, model, hEff, 1.0, ctx.tables, ctx, rng);
      }
      const team: number[] = [];
      for (let i = 0; i < model.V; i++) if (chain.state[i]) team.push(i);
      expect(team.length).toBe(4);
      assertValid(team, model);
    }
  });

  it("honors a pinned feature and an excluded species", () => {
    const model = speciesItemModel();
    const rng = new Rng(3);
    const fixed = [0]; // A @ x pinned
    const excluded = [2, 3]; // exclude all of species B
    const ctx = ctxFor(model, fixed, excluded);
    const constraints = buildConstraintSets(fixed, model);
    const hEff = new Float64Array(model.V);
    for (let i = 0; i < model.V; i++) hEff[i] = model.h[i];
    const available: number[] = [];
    for (let i = 0; i < model.V; i++) if (i !== 0 && i !== 2 && i !== 3) available.push(i);
    const chain = initChain(model, fixed, available, 3, constraints, hEff, rng)!;
    for (let s = 0; s < 300; s++) {
      if (rng.random() < 0.5) {
        pottsTrackReroll(chain, model, hEff, 1.0, ctx.tables, ctx, rng);
      } else {
        pottsSpeciesSwap(chain, model, hEff, 1.0, ctx.tables, ctx, rng);
      }
      // Pin stays on; excluded species never appears.
      expect(chain.state[0]).toBe(1);
      expect(chain.state[2]).toBe(0);
      expect(chain.state[3]).toBe(0);
      const team: number[] = [];
      for (let i = 0; i < model.V; i++) if (chain.state[i]) team.push(i);
      expect(team.length).toBe(4);
      assertValid(team, model);
    }
  });

  it("species-only reroll is a no-op (no tracks)", () => {
    const model = speciesOnlyModel();
    const rng = new Rng(1);
    const ctx = ctxFor(model, [], []);
    const constraints = buildConstraintSets([], model);
    const hEff = new Float64Array(model.V);
    for (let i = 0; i < model.V; i++) hEff[i] = model.h[i];
    const chain = initChain(model, [], [...Array(model.V).keys()], 4, constraints, hEff, rng)!;
    const before = [...chain.state];
    pottsTrackReroll(chain, model, hEff, 1.0, ctx.tables, ctx, rng);
    expect([...chain.state]).toEqual(before);
  });
});

describe("energy tracking", () => {
  it("keeps chain.energy consistent with a full recompute after moves", () => {
    const model = speciesItemModel();
    const rng = new Rng(42);
    const ctx = ctxFor(model, [], []);
    const constraints = buildConstraintSets([], model);
    const hEff = new Float64Array(model.V);
    for (let i = 0; i < model.V; i++) hEff[i] = model.h[i];
    const chain = initChain(model, [], [...Array(model.V).keys()], 4, constraints, hEff, rng)!;
    for (let s = 0; s < 200; s++) {
      if (rng.random() < 0.5) {
        pottsTrackReroll(chain, model, hEff, 1.0, ctx.tables, ctx, rng);
      } else {
        pottsSpeciesSwap(chain, model, hEff, 1.0, ctx.tables, ctx, rng);
      }
    }
    // Recompute H = -h·s - 0.5 s'Js from scratch and compare to tracked energy.
    let hDot = 0;
    let quad = 0;
    for (let i = 0; i < model.V; i++) {
      if (!chain.state[i]) continue;
      hDot += hEff[i];
      for (let j = 0; j < model.V; j++) {
        if (chain.state[j]) quad += model.J[i * model.V + j];
      }
    }
    const recomputed = -hDot - 0.5 * quad;
    expect(chain.energy).toBeCloseTo(recomputed, 9);
  });
});
