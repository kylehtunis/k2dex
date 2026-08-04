// Unit tests for the feature-detail helpers (webapp-only; no parity row).
// A small hand-built model with known J / h / m and a tiny corpus index lets
// us assert species-level coupling ordering, corpus conditioning, and
// ranks exactly.

import { describe, expect, it } from "vitest";
import type { IsingModel, SpeciesGraph, TeamCounts } from "../../sampler/types";
import { factoredFromSpeciesItem } from "../../sampler/model";
import {
  featureCorpusAppearances,
  speciesCouplings,
  featureRanks,
} from "../featureDetail";

// V=6. Species A has two features (0, 1). Partners C, D, E are structural.
// Species B shares item "x" with A@x but is a different species so it IS
// structural at the species level.
function buildModel(): IsingModel {
  const V = 6;
  const speciesOf = ["A", "A", "B", "C", "D", "E"];
  const itemOf: (string | null)[] = ["x", null, "x", null, "y", null];
  const vocab = speciesOf.map((s, i) =>
    itemOf[i] === null ? s : `${s} @ ${itemOf[i]}`,
  );

  const J = new Float64Array(V * V);
  const set = (i: number, j: number, v: number) => {
    J[i * V + j] = v;
    J[j * V + i] = v;
  };
  set(0, 1, 9); // same species — species graph ignores
  set(0, 2, 0.6); // A@x <-> B@x
  set(1, 2, 0.4); // A   <-> B@x
  set(0, 3, 0.5); // A@x <-> C
  set(1, 3, 0.3); // A   <-> C
  set(0, 4, 0.8); // A@x <-> D
  set(1, 4, 0.2); // A   <-> D
  set(0, 5, -0.3); // A@x <-> E
  set(1, 5, -0.1); // A   <-> E

  const h = Float64Array.from([0.1, 0.5, -0.2, 0.3, 0.0, 0.4]);
  const m = Float64Array.from([0.2, 0.3, 0.1, 0.05, 0.25, 0.15]);

  const indexOf = new Map<string, number>();
  vocab.forEach((s, i) => indexOf.set(s, i));

  const factored = factoredFromSpeciesItem(speciesOf, itemOf);
  return {
    id: "fd", displayName: "FD", regulation: "test",
    featureDimensions: 2, latestTournamentDate: "",
    V, teamSize: 6, vocab, speciesOf, itemOf, ...factored, m, J, h, indexOf,
    nCorpusTeams: 20, name: "fd",
  };
}

// Build a mock SpeciesGraph matching the model above. Species in
// alphabetical order: [A, B, C, D, E]. Synergy = mean of each
// cross-species J block.
function buildGraph(): SpeciesGraph {
  const species = ["A", "B", "C", "D", "E"];
  const S = species.length;
  const synergy = new Float64Array(S * S);

  // A(feats 0,1) <-> B(feat 2): block is [[0.6],[0.4]], mean = 0.5
  const setSym = (a: number, b: number, syn: number) => {
    synergy[a * S + b] = syn;
    synergy[b * S + a] = syn;
  };
  setSym(0, 1, 0.5);   // A-B
  setSym(0, 2, 0.4);   // A-C: mean of [0.5, 0.3] = 0.4
  setSym(0, 3, 0.5);   // A-D: mean of [0.8, 0.2] = 0.5
  setSym(0, 4, -0.2);  // A-E: mean of [-0.3, -0.1] = -0.2

  const indexOf = new Map<string, number>();
  species.forEach((s, i) => indexOf.set(s, i));
  return { species, synergy, indexOf };
}

describe("speciesCouplings", () => {
  it("returns species-level synergies ranked by synergy desc", () => {
    const model = buildModel();
    const graph = buildGraph();
    // Site 0 = species A (features 0,1).
    const { synergies, antisynergies } = speciesCouplings(model, graph, 0);

    // Synergies: A-B (syn=0.5), A-D (syn=0.5), A-C (syn=0.4) — sorted by synergy desc
    expect(synergies.length).toBeGreaterThanOrEqual(3);
    expect(synergies[0].synergy).toBeCloseTo(0.5);
    expect(synergies[2].synergy).toBeCloseTo(0.4);

    // Antisynergies: A-E (syn=-0.2)
    expect(antisynergies).toHaveLength(1);
    expect(antisynergies[0].species).toBe("E");
    expect(antisynergies[0].synergy).toBeCloseTo(-0.2);
  });

  it("respects topN", () => {
    const model = buildModel();
    const graph = buildGraph();
    const { synergies } = speciesCouplings(model, graph, 0, 2);
    expect(synergies).toHaveLength(2);
  });
});

describe("featureCorpusAppearances", () => {
  const teamCounts: TeamCounts = new Map([
    ["0-2-3", 5],
    ["0-3-4", 3],
    ["1-2-5", 10], // no feature 0
    ["0-1-5", 2],
  ]);

  it("conditions on membership and aggregates", () => {
    const model = buildModel();
    const r = featureCorpusAppearances(model, teamCounts, 0);
    // Only the three rosters containing 0.
    expect(r.nTeams).toBe(3);
    expect(r.totalAppearances).toBe(10);
    // Ranked by count desc.
    expect(r.teams.map((t) => t.count)).toEqual([5, 3, 2]);
    // Members ordered by descending marginal (m[0]=0.2 leads its roster).
    expect(r.teams[0].team[0]).toBe(0);
  });

  it("honors topN and empty corpus", () => {
    const model = buildModel();
    expect(featureCorpusAppearances(model, teamCounts, 0, 2).teams).toHaveLength(2);
    expect(featureCorpusAppearances(model, null, 0).nTeams).toBe(0);
  });
});

describe("featureRanks", () => {
  it("counts strictly-greater + 1 for bias and usage", () => {
    const model = buildModel();
    // h[0]=0.1; greater: 0.5,0.3,0.4 -> rank 4.
    // m[0]=0.2; greater: 0.3,0.25  -> rank 3.
    expect(featureRanks(model, 0)).toEqual({ biasRank: 4, marginalRank: 3 });
  });
});
