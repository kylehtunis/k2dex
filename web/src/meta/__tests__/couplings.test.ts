// The /meta §02 hierarchical drill-down on a 2-track (item + ability) model:
// the item tier must marginalize abilities with the empirical usage weights,
// and the attribute tier must be the exact residual around it (client-side
// mirror of potts.decompose_block_hierarchical's reconstruction invariant).

import { describe, expect, it } from "vitest";
import type { IsingModel, TrackDef } from "../../sampler/types";
import { deriveFactored } from "../../sampler/model";
import {
  groupByTrack,
  hasAttributeSplit,
  topAttributeEntries,
  topModulationEntries,
} from "../couplings";

// P(0): (x,a) (x,b) (y,a)    Q(1): (x,a) (z,b)
// P's item x spans two abilities (usage 3:1); everything else is a single
// ability, so only P has an attribute split.
const M = [0.3, 0.1, 0.2, 0.4, 0.2];
const PAIRS: [number, number, number][] = [
  [0, 3, 0.5], [0, 4, 0.8],
  [1, 3, -0.1], [1, 4, -0.4],
  [2, 3, 0.2], [2, 4, 0.3],
];

function buildModel(): IsingModel {
  const sites = ["parasect", "quagsire"];
  const siteOf = [0, 0, 0, 1, 1];
  const items = ["leftovers", "leftovers", "sitrus-berry", "leftovers", "life-orb"];
  const abils = ["dry-skin", "effect-spore", "dry-skin", "water-absorb", "unaware"];
  const trackValues: (string | null)[][] = items.map((it, i) => [it, abils[i]]);
  const tracks: TrackDef[] = [
    { name: "item", cardinality: 1, crossSlotUnique: true, withinSlotUnique: false },
    { name: "ability", cardinality: 1, crossSlotUnique: false, withinSlotUnique: false },
  ];
  const V = siteOf.length;
  const J = new Float64Array(V * V);
  for (const [i, j, v] of PAIRS) {
    J[i * V + j] = v;
    J[j * V + i] = v;
  }
  const vocab = trackValues.map((tv, i) => `${sites[siteOf[i]]} @ ${tv[0]} (${tv[1]})`);
  const indexOf = new Map<string, number>();
  vocab.forEach((v, i) => indexOf.set(v, i));
  return {
    id: "reg-test", displayName: "t", regulation: "test", latestTournamentDate: "",
    V, teamSize: 6, vocab, sites, siteOf, tracks, trackValues,
    ...deriveFactored(sites, siteOf, trackValues),
    m: new Float64Array(M), J, h: new Float64Array(V),
    indexOf, nCorpusTeams: 100, name: "t",
  };
}

const model = buildModel();

describe("groupByTrack", () => {
  it("groups by item with usage weights renormalized inside the group", () => {
    const groups = groupByTrack(model, model.siteFeatures[0], 0);
    expect(groups.map((g) => g.value)).toEqual(["leftovers", "sitrus-berry"]);
    expect(groups[0].features).toEqual([0, 1]);
    expect(groups[0].weights[0]).toBeCloseTo(0.75, 12);
    expect(groups[0].weights[1]).toBeCloseTo(0.25, 12);
    expect(groups[0].mass).toBeCloseTo(0.4, 12);
    expect(groups[0].rep).toBe(0); // most-used state represents the group
  });
});

describe("topModulationEntries (item tier)", () => {
  const entries = topModulationEntries(model, 0, 1, 0);

  it("drops the mechanically-excluded same-item pairing", () => {
    const items = entries.map((e) => [
      model.trackValues[e.featureA][0],
      model.trackValues[e.featureB][0],
    ]);
    expect(items).not.toContainEqual(["leftovers", "leftovers"]);
    expect(entries).toHaveLength(3);
  });

  it("marginalizes the ability with the empirical usage weights", () => {
    const e = entries.find(
      (x) => model.trackValues[x.featureA][0] === "leftovers"
        && model.trackValues[x.featureB][0] === "life-orb",
    )!;
    // 0.75 * J[(x,a),(z,b)] + 0.25 * J[(x,b),(z,b)]
    expect(e.jValue).toBeCloseTo(0.75 * 0.8 + 0.25 * -0.4, 12);
  });

  it("reports deviation from the species-level synergy", () => {
    const withSynergy = topModulationEntries(model, 0, 1, 0.25);
    for (const e of withSynergy) expect(e.deviation).toBeCloseTo(e.jValue - 0.25, 12);
  });
});

describe("topAttributeEntries (ability tier)", () => {
  const item = topModulationEntries(model, 0, 1, 0).find(
    (x) => model.trackValues[x.featureA][0] === "leftovers"
      && model.trackValues[x.featureB][0] === "life-orb",
  )!;
  const entries = topAttributeEntries(model, item.groupA, item.groupB, item.jValue, 1, 5);

  it("reconstructs the full coupling from the item tier plus the residual", () => {
    expect(entries).toHaveLength(2);
    for (const e of entries) {
      expect(item.jValue + e.deviation).toBeCloseTo(
        model.J[e.featureA * model.V + e.featureB],
        12,
      );
    }
  });

  it("has zero usage-weighted mean residual by construction", () => {
    const w = new Map([[0, 0.75], [1, 0.25]]);
    let acc = 0;
    for (const e of entries) acc += (w.get(e.featureA) ?? 0) * e.deviation;
    expect(acc).toBeCloseTo(0, 12);
  });

  it("drops values below the support floor", () => {
    // (x,b) carries m=0.1 -> 10 weighted appearances; a floor of 20 removes it,
    // leaving a single 1x1 cell with nothing to compare.
    expect(topAttributeEntries(model, item.groupA, item.groupB, item.jValue, 1, 20))
      .toEqual([]);
  });
});

describe("hasAttributeSplit", () => {
  it("is true only for a species carrying two abilities on one item build", () => {
    expect(hasAttributeSplit(model, 0, 1)).toBe(true);
    expect(hasAttributeSplit(model, 1, 1)).toBe(false);
  });
});
