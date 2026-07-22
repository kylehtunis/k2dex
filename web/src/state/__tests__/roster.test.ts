// Roster-slot helpers + share round-trip on a 2-track (item + ability) model.
// Guards the per-track pin derivation and the partial-pin share encoding, which
// have no other test coverage (the completer UI itself is untested).

import { describe, expect, it } from "vitest";
import type { IsingModel, TrackDef } from "../../sampler/types";
import { deriveFactored, unpackLowerTriangle } from "../../sampler/model";
import {
  derivePins,
  emptySlot,
  slotFeature,
  slotFromFeature,
  slotFromSlugs,
  setSlotTrack,
} from "../roster";
import type { RosterSlot } from "../PageStateContext";
import { encodeCore, decodeCore } from "../../render/shareLink";
import { buildSlugIndex } from "../../render/vocab-match";

// P's item x is split across abilities a/b (so an item-only pin is ambiguous);
// P's item y and every Q/R feature is a single ability (item pin fully resolves).
//   P(0): (x,a)(x,b)(y,a)   Q(1): (x,a)(z,b)   R(2): (w,a)(v,b)
function build2TrackModel(): IsingModel {
  const sites = ["parasect", "quagsire", "roserade"];
  const siteOf = [0, 0, 0, 1, 1, 2, 2];
  const items = ["leftovers", "leftovers", "sitrus-berry", "leftovers", "life-orb", "focus-sash", "sitrus-berry"];
  const abils = ["dry-skin", "effect-spore", "dry-skin", "water-absorb", "unaware", "natural-cure", "technician"];
  const trackValues: (string | null)[][] = items.map((it, i) => [it, abils[i]]);
  const tracks: TrackDef[] = [
    { name: "item", cardinality: 1, crossSlotUnique: true, withinSlotUnique: false },
    { name: "ability", cardinality: 1, crossSlotUnique: false, withinSlotUnique: false },
  ];
  const V = siteOf.length;
  const lowerFlat = new Float32Array((V * (V - 1)) / 2).fill(0.1);
  const J = unpackLowerTriangle(lowerFlat, V);
  const h = new Float64Array(V).fill(-0.2);
  const m = new Float64Array(V).fill(0.15);
  const vocab = trackValues.map((tv, i) => `${sites[siteOf[i]]} @ ${tv[0]} (${tv[1]})`);
  const indexOf = new Map<string, number>();
  vocab.forEach((v, i) => indexOf.set(v, i));
  const derived = deriveFactored(sites, siteOf, trackValues);
  return {
    id: "reg-test", displayName: "t", regulation: "test", latestTournamentDate: "",
    V, teamSize: 6, vocab, sites, siteOf, tracks, trackValues, ...derived,
    m, J, h, indexOf, nCorpusTeams: 0, name: "t",
  };
}

describe("slotFeature / derivePins", () => {
  const model = build2TrackModel();

  it("an empty slot (all tracks free) is a site pin", () => {
    const s = emptySlot(model, 0);
    expect(slotFeature(model, s)).toBeNull();
  });

  it("pinning an item that still spans two abilities stays a partial pin", () => {
    const s = setSlotTrack(emptySlot(model, 0), 0, "leftovers"); // P item x, 2 abilities
    expect(slotFeature(model, s)).toBeNull();
  });

  it("pinning an item whose ability is unique resolves to a feature", () => {
    const s = setSlotTrack(emptySlot(model, 0), 0, "sitrus-berry"); // P (y,a) unique
    expect(slotFeature(model, s)).toBe(2);
  });

  it("pinning both tracks resolves to the exact feature", () => {
    let s = setSlotTrack(emptySlot(model, 0), 0, "leftovers");
    s = setSlotTrack(s, 1, "effect-spore");
    expect(slotFeature(model, s)).toBe(1); // P (x,b)
  });

  it("derivePins splits feature / site / partial pins", () => {
    const roster: RosterSlot[] = [
      slotFromFeature(model, 3), // Q (x,a) — feature pin
      emptySlot(model, 1 + 1),   // R — site pin (all free)
      setSlotTrack(emptySlot(model, 0), 0, "leftovers"), // P item x, ability free — partial
    ];
    const { fixedIdxs, fixedSites, sitePinTrackValues } = derivePins(model, roster);
    expect(fixedIdxs).toEqual([3]);
    expect(fixedSites).toEqual([2, 0]);
    expect(sitePinTrackValues).toEqual([
      [null, null],           // R pure site pin
      ["leftovers", null],    // P partial: item locked, ability free
    ]);
  });
});

describe("share round-trip (encode → decode → slot)", () => {
  const model = build2TrackModel();
  const slugIndex = buildSlugIndex(model);

  function roundTrip(roster: RosterSlot[]): RosterSlot[] {
    const { fixedIdxs, fixedSites, sitePinTrackValues } = derivePins(model, roster);
    const token = encodeCore("reg-test", fixedIdxs, model, fixedSites, sitePinTrackValues);
    const decoded = decodeCore(token);
    expect(decoded).not.toBeNull();
    return decoded!.features
      .map((f) => slotFromSlugs(model, slugIndex, f.speciesSlug, f.itemSlug, f.abilitySlug))
      .filter((s): s is RosterSlot => s !== null);
  }

  it("round-trips a full feature pin (both tracks)", () => {
    const back = roundTrip([slotFromFeature(model, 1)]); // P (x,b)
    expect(derivePins(model, back).fixedIdxs).toEqual([1]);
  });

  it("round-trips a pure site pin", () => {
    const back = roundTrip([emptySlot(model, 2)]);
    const { fixedIdxs, fixedSites } = derivePins(model, back);
    expect(fixedIdxs).toEqual([]);
    expect(fixedSites).toEqual([2]);
  });

  it("round-trips a partial pin (item locked, ability free)", () => {
    const partial = setSlotTrack(emptySlot(model, 0), 0, "leftovers");
    const back = roundTrip([partial]);
    const { fixedSites, sitePinTrackValues } = derivePins(model, back);
    expect(fixedSites).toEqual([0]);
    expect(sitePinTrackValues).toEqual([["leftovers", null]]);
  });

  it("a legacy 2-segment token decodes as an ability-free pin", () => {
    // species~item with no ability segment: item x on P stays a partial pin.
    const decoded = decodeCore("reg-test.parasect~leftovers");
    const slot = slotFromSlugs(
      model, slugIndex, decoded!.features[0].speciesSlug,
      decoded!.features[0].itemSlug, decoded!.features[0].abilitySlug,
    )!;
    expect(slot.trackValues).toEqual(["leftovers", null]);
  });
});
