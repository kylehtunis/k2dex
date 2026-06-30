// Unit tests for the feature-detail helpers (webapp-only; no parity row).
// A small hand-built model with known J / h / m and a tiny corpus index lets
// us assert structural filtering, coupling ordering, corpus conditioning, and
// ranks exactly.

import { describe, expect, it } from "vitest";
import type { IsingModel, TeamCounts } from "../../sampler/types";
import {
  featureCorpusAppearances,
  featureCouplings,
  featureRanks,
} from "../featureDetail";

// V=6. Feature 0 ("A @ x") shares species A with 1 and item x with 2, so both
// of those pairs are non-structural and must be filtered out regardless of how
// large their J is. Partners 3, 4, 5 are structural.
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
  set(0, 1, 9); // same species  -> filtered
  set(0, 2, 9); // same item "x" -> filtered
  set(0, 3, 0.5); // structural synergy
  set(0, 4, 0.8); // structural synergy (strongest)
  set(0, 5, -0.3); // structural antisynergy

  const h = Float64Array.from([0.1, 0.5, -0.2, 0.3, 0.0, 0.4]);
  const m = Float64Array.from([0.2, 0.3, 0.1, 0.05, 0.25, 0.15]);

  const indexOf = new Map<string, number>();
  vocab.forEach((s, i) => indexOf.set(s, i));

  return {
    id: "fd", displayName: "FD", regulation: "test",
    featureDimensions: 2, latestTournamentDate: "",
    V, teamSize: 6, vocab, speciesOf, itemOf, m, J, h, indexOf,
    nCorpusTeams: 20, name: "fd",
  };
}

describe("featureCouplings", () => {
  it("drops non-structural pairs and ranks by signed J", () => {
    const model = buildModel();
    const { synergies, antisynergies } = featureCouplings(model, 0);

    // Same-species (1) and same-item (2) partners are excluded despite J=9.
    const allIdxs = [...synergies, ...antisynergies].map((c) => c.idx);
    expect(allIdxs).not.toContain(1);
    expect(allIdxs).not.toContain(2);

    // Synergies strongest-first; antisynergies most-negative-first.
    expect(synergies.map((c) => c.idx)).toEqual([4, 3]);
    expect(synergies[0].jValue).toBeCloseTo(0.8);
    expect(antisynergies.map((c) => c.idx)).toEqual([5]);
    expect(antisynergies[0].jValue).toBeCloseTo(-0.3);
  });

  it("respects topN", () => {
    const model = buildModel();
    const { synergies } = featureCouplings(model, 0, 1);
    expect(synergies).toHaveLength(1);
    expect(synergies[0].idx).toBe(4);
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
