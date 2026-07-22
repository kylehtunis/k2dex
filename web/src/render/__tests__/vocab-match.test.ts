// Tests for clipboard-paste parsing, slug-based vocab matching, and the
// shareable-URL token codec. Uses a small synthetic model so the slug
// bridge (regional forme + item) is exercised against known vocab.

import { describe, expect, it } from "vitest";
import type { IsingModel } from "../../sampler/types";
import { factoredFromSpeciesItem } from "../../sampler/model";
import {
  buildSlugIndex,
  matchPaste,
  parsePokepaste,
  resolveFeature,
  resolveSpeciesSlug,
} from "../vocab-match";
import {
  decodeCompleter,
  decodeCore,
  encodeCompleter,
  encodeCore,
} from "../shareLink";

// Species-with-item model. Includes a regional forme (corpus name
// "Alolan Ninetales" vs Showdown "Ninetales-Alola") to exercise the
// slug bridge, two builds of Incineroar (different items), and an
// itemless entry.
function buildModel(): IsingModel {
  const species = [
    "Incineroar",
    "Incineroar",
    "Alolan Ninetales",
    "Calyrex-Shadow",
    "Amoonguss",
  ];
  const items: (string | null)[] = [
    "Sitrus Berry",
    "Assault Vest",
    "Light Clay",
    "Life Orb",
    null,
  ];
  const V = species.length;
  const vocab = species.map((s, i) => (items[i] === null ? s : `${s} @ ${items[i]}`));
  const indexOf = new Map<string, number>();
  vocab.forEach((v, i) => indexOf.set(v, i));
  // Marginals: Incineroar @ Sitrus is the most popular Incineroar build.
  const m = Float64Array.from([0.5, 0.2, 0.3, 0.4, 0.35]);
  const factored = factoredFromSpeciesItem(species, items);
  return {
    id: "test-species-item",
    displayName: "Test Species @ Item",
    regulation: "test",
    latestTournamentDate: "",
    V,
    teamSize: 6,
    vocab,
    speciesOf: species,
    itemOf: items,
    ...factored,
    m,
    J: new Float64Array(V * V),
    h: new Float64Array(V),
    indexOf,
    nCorpusTeams: 1000,
    name: "test-species-item",
  };
}

describe("parsePokepaste", () => {
  it("parses species lines and discards the rest", () => {
    const paste = [
      "Incineroar @ Sitrus Berry",
      "Ability: Intimidate",
      "Level: 50",
      "- Fake Out",
      "",
      "Calyrex-Shadow @ Life Orb",
      "Ability: As One",
      "",
      "Amoonguss",
      "Ability: Regenerator",
    ].join("\n");
    const parsed = parsePokepaste(paste);
    expect(parsed.map((p) => p.speciesSlug)).toEqual([
      "incineroar",
      "calyrex-shadow",
      "amoonguss",
    ]);
    expect(parsed[0].itemSlug).toBe("sitrus-berry");
    expect(parsed[2].itemSlug).toBeNull();
  });

  it("handles nickname and gender forms", () => {
    const paste = "Koko (Tapu Koko) (M) @ Life Orb\nAbility: Electric Surge";
    const parsed = parsePokepaste(paste);
    expect(parsed).toHaveLength(1);
    expect(parsed[0].rawSpecies).toBe("Tapu Koko");
    expect(parsed[0].itemSlug).toBe("life-orb");
  });

  it("strips Showdown mega forme suffixes to the base species slug", () => {
    // Corpus buckets megas under base species (slug "charizard"/"blastoise").
    expect(parsePokepaste("Charizard-Mega-Y @ Charizardite Y")[0].speciesSlug).toBe(
      "charizard",
    );
    expect(parsePokepaste("Charizard-Mega-X @ Charizardite X")[0].speciesSlug).toBe(
      "charizard",
    );
    expect(parsePokepaste("Blastoise-Mega @ Blastoisinite")[0].speciesSlug).toBe(
      "blastoise",
    );
  });

  it("collapses all three Mega Floette spellings to one slug", () => {
    // Corpus name "Eternal Flower Floette" slugs to "floette"; Showdown can
    // export it as plain Floette, Floette-Mega, or Floette-Eternal.
    for (const name of ["Floette", "Floette-Mega", "Floette-Eternal"]) {
      expect(parsePokepaste(`${name} @ Floettite`)[0].speciesSlug).toBe("floette");
    }
  });
});

describe("resolveFeature", () => {
  const model = buildModel();
  const slugIndex = buildSlugIndex(model);

  it("matches species + item exactly", () => {
    const r = resolveFeature(slugIndex, model, "incineroar", "assault-vest");
    expect(r.idx).toBe(1);
    expect(r.warning).toBeNull();
  });

  it("bridges regional forme naming", () => {
    // Showdown "Ninetales-Alola" -> slug "ninetales-alola" matches the
    // corpus "Alolan Ninetales".
    const r = resolveFeature(slugIndex, model, "ninetales-alola", "light-clay");
    expect(r.idx).toBe(2);
    expect(r.warning).toBeNull();
  });

  it("skips when item is out of vocab and species has multiple builds", () => {
    const r = resolveFeature(slugIndex, model, "incineroar", "leftovers", {
      species: "Incineroar",
      item: "Leftovers",
    });
    expect(r.idx).toBeNull();
    expect(r.warning).toContain("Incineroar");
    expect(r.warning).toContain("Leftovers");
  });

  it("warns and returns null for an unknown species", () => {
    const r = resolveFeature(slugIndex, model, "pikachu", null, {
      species: "Pikachu",
    });
    expect(r.idx).toBeNull();
    expect(r.warning).toContain("Pikachu");
  });

  it("resolves a species slug back to its display name", () => {
    expect(resolveSpeciesSlug(slugIndex, model, "ninetales-alola")).toBe(
      "Alolan Ninetales",
    );
    expect(resolveSpeciesSlug(slugIndex, model, "nope")).toBeNull();
  });
});

describe("matchPaste", () => {
  const model = buildModel();

  it("matches a full paste, deduping by species", () => {
    const paste = [
      "Incineroar @ Sitrus Berry",
      "",
      "Incineroar @ Assault Vest", // duplicate species — dropped
      "",
      "Calyrex-Shadow @ Life Orb",
    ].join("\n");
    const { idxs, errors, warnings } = matchPaste(model, paste);
    expect(idxs).toEqual([0, 3]);
    expect(errors).toEqual([]);
    expect(warnings).toEqual([]);
  });

  it("surfaces an error for empty input", () => {
    const { idxs, errors } = matchPaste(model, "   \n  ");
    expect(idxs).toEqual([]);
    expect(errors).toHaveLength(1);
  });

  it("warns on out-of-vocab species and skipped multi-build items", () => {
    const paste = "Incineroar @ Leftovers\n\nPikachu @ Light Ball";
    const { idxs, warnings } = matchPaste(model, paste);
    expect(idxs).toEqual([]);
    expect(warnings).toHaveLength(2);
  });
});

describe("shareLink core token", () => {
  const model = buildModel();

  it("round-trips a team to features", () => {
    const token = encodeCore("test-species-item", [3, 0], model);
    expect(token).toBe("test-species-item.incineroar~sitrus-berry_calyrex-shadow~life-orb");
    const decoded = decodeCore(token)!;
    expect(decoded.modelId).toBe("test-species-item");
    const slugIndex = buildSlugIndex(model);
    const idxs = decoded.features.map(
      (f) => resolveFeature(slugIndex, model, f.speciesSlug, f.itemSlug).idx,
    );
    expect(idxs).toEqual([0, 3]);
  });

  it("decodes legacy model codes to the unified per-regulation slug", () => {
    const decoded = decodeCore("si.3.incineroar~sitrus-berry")!;
    expect(decoded.modelId).toBe("reg-m-a");
    const decoded2 = decodeCore("s.5.amoonguss")!;
    expect(decoded2.modelId).toBe("reg-m-a");
    const decoded3 = decodeCore("reg-m-a-species-item.3.incineroar~sitrus-berry")!;
    expect(decoded3.modelId).toBe("reg-m-a");
  });

  it("drops the legacy Bias Adjustment segment on decode", () => {
    const legacy = decodeCore("reg-m-a.3.incineroar~sitrus-berry_amoonguss")!;
    const current = decodeCore("reg-m-a.incineroar~sitrus-berry_amoonguss")!;
    expect(legacy.features).toEqual(current.features);
    expect(legacy.modelId).toBe("reg-m-a");
  });

  it("encodes itemless features without a tilde", () => {
    const token = encodeCore("test-species", [4], model);
    expect(token).toBe("test-species.amoonguss");
    expect(decodeCore(token)!.features[0].itemSlug).toBeNull();
  });

  it("returns null on a malformed token", () => {
    expect(decodeCore("")).toBeNull();
    expect(decodeCore("garbage")).toBeNull();
    expect(decodeCore("X!Y.3.incineroar")).toBeNull();
  });
});

// Species-with-item-and-ability model (schema v4 with a real ability track;
// current committed artifacts don't have one yet, but the codec must already
// support it). Mirrors buildModel()'s Incineroar/Calyrex-Shadow/Amoonguss
// entries with an ability appended to each.
function buildAbilityModel(): IsingModel {
  const species = ["Incineroar", "Calyrex-Shadow", "Amoonguss"];
  const items: (string | null)[] = ["Sitrus Berry", "Life Orb", null];
  const abilities = ["Intimidate", "As One", "Regenerator"];
  const V = species.length;
  const vocab = species.map(
    (s, i) => (items[i] === null ? `${s} (${abilities[i]})` : `${s} @ ${items[i]} (${abilities[i]})`),
  );
  const indexOf = new Map<string, number>();
  vocab.forEach((v, i) => indexOf.set(v, i));
  const tracks = [
    { name: "item", cardinality: 1, crossSlotUnique: true, withinSlotUnique: false },
    { name: "ability", cardinality: 1, crossSlotUnique: false, withinSlotUnique: false },
  ];
  const trackValues = items.map((it, i) => [it, abilities[i]]);
  const siteFeatures = species.map((_, i) => [i]);
  return {
    id: "test-ability",
    displayName: "Test Species @ Item (Ability)",
    regulation: "test",
    latestTournamentDate: "",
    V,
    teamSize: 6,
    vocab,
    speciesOf: species,
    itemOf: items,
    sites: species,
    siteOf: species.map((_, i) => i),
    tracks,
    trackValues,
    siteFeatures,
    m: Float64Array.from([0.5, 0.4, 0.35]),
    J: new Float64Array(V * V),
    h: new Float64Array(V),
    indexOf,
    nCorpusTeams: 1000,
    name: "test-ability",
  };
}

describe("shareLink core token — ability track", () => {
  const model = buildAbilityModel();

  it("encodes species~item~ability for a feature pin with an ability locked", () => {
    const token = encodeCore("test-ability", [0], model);
    expect(token).toBe("test-ability.incineroar~sitrus-berry~intimidate");
    const decoded = decodeCore(token)!;
    expect(decoded.features[0]).toEqual({
      speciesSlug: "incineroar",
      itemSlug: "sitrus-berry",
      abilitySlug: "intimidate",
    });
  });

  it("keeps the item segment as 'none' for an itemless ability pin", () => {
    const token = encodeCore("test-ability", [2], model);
    expect(token).toBe("test-ability.amoonguss~none~regenerator");
    const decoded = decodeCore(token)!;
    expect(decoded.features[0].itemSlug).toBe("none");
    expect(decoded.features[0].abilitySlug).toBe("regenerator");
  });

  it("multi-word abilities slugify like items (spaces to hyphens)", () => {
    const token = encodeCore("test-ability", [1], model);
    expect(token).toBe("test-ability.calyrex-shadow~life-orb~as-one");
  });

  it("a legacy 2-segment (species~item) token decodes as ability-free", () => {
    const decoded = decodeCore("test-ability.incineroar~sitrus-berry")!;
    expect(decoded.features[0]).toEqual({
      speciesSlug: "incineroar",
      itemSlug: "sitrus-berry",
      abilitySlug: null,
    });
  });

  it("a bare species (site pin) token still decodes with a null item and ability", () => {
    const decoded = decodeCore("test-ability.amoonguss")!;
    expect(decoded.features[0]).toEqual({
      speciesSlug: "amoonguss",
      itemSlug: null,
      abilitySlug: null,
    });
  });
});

describe("shareLink completer token", () => {
  const model = buildModel();

  it("omits defaults and round-trips the primary inputs", () => {
    const params = encodeCompleter(
      {
        modelId: "test-species-item",
        fixedIdxs: [0],
        fixedSites: [],
        inactiveTracks: [],
        excludedSpecies: ["Amoonguss"],
        includedSpecies: [],
        usePT: true,
        temperature: 1.0, // default — omitted
        anchorStrength: 1.0, // default — omitted
        ptRuns: 10,
        ptLadder: 7,
        ptSweeps: 20000,
        ptSwapInterval: 10, // all default — omitted
        seed: null,
      },
      model,
    );
    expect(params.get("tmp")).toBeNull();
    expect(params.get("a")).toBeNull();
    expect(params.get("g")).toBeNull();
    expect(params.get("seed")).toBeNull();
    expect(params.get("anc")).toBeNull();
    expect(params.get("x")).toBe("amoonguss");

    const d = decodeCompleter(params)!;
    expect(d.usePT).toBe(true);
    expect(d.temperature).toBe(1.0);
    expect(d.ptRuns).toBe(10);
    expect(d.excludedSlugs).toEqual(["amoonguss"]);
    expect(d.seed).toBeNull();
  });

  it("round-trips the inclusion allow-list", () => {
    const params = encodeCompleter(
      {
        modelId: "test-species-item",
        fixedIdxs: [],
        fixedSites: [],
        inactiveTracks: [],
        excludedSpecies: [],
        includedSpecies: ["Amoonguss", "Incineroar"],
        usePT: true,
        temperature: 1.0,
        anchorStrength: 1.0,
        ptRuns: 10,
        ptLadder: 7,
        ptSweeps: 20000,
        ptSwapInterval: 10,
        seed: null,
      },
      model,
    );
    expect(params.get("i")).toBe("amoonguss_incineroar");
    expect(params.get("x")).toBeNull();
    expect(decodeCompleter(params)!.includedSlugs).toEqual([
      "amoonguss",
      "incineroar",
    ]);
  });

  it("carries greedy mode without a seed", () => {
    const params = encodeCompleter(
      {
        modelId: "test-species-item",
        fixedIdxs: [0],
        fixedSites: [],
        inactiveTracks: [],
        excludedSpecies: [],
        includedSpecies: [],
        usePT: false,
        temperature: 1.0,
        anchorStrength: 1.0,
        ptRuns: 10,
        ptLadder: 7,
        ptSweeps: 20000,
        ptSwapInterval: 10,
        seed: 42, // ignored in greedy mode
      },
      model,
    );
    expect(params.get("g")).toBe("1");
    expect(params.get("seed")).toBeNull();
    expect(decodeCompleter(params)!.usePT).toBe(false);
  });

  it("round-trips a non-default Anchor Strength, including greedy mode", () => {
    const base = {
      modelId: "test-species-item",
      fixedIdxs: [0],
      fixedSites: [] as number[],
      inactiveTracks: [] as number[],
      excludedSpecies: [] as string[],
      includedSpecies: [] as string[],
      temperature: 1.0,
      anchorStrength: 1.8,
      ptRuns: 10,
      ptLadder: 7,
      ptSweeps: 20000,
      ptSwapInterval: 10,
      seed: null,
    };
    const pt = encodeCompleter({ ...base, usePT: true }, model);
    expect(pt.get("anc")).toBe("1.8");
    expect(decodeCompleter(pt)!.anchorStrength).toBe(1.8);
    // Anchor Strength applies to the greedy path too, so it must survive
    // the greedy early-return.
    const greedy = encodeCompleter({ ...base, usePT: false }, model);
    expect(greedy.get("anc")).toBe("1.8");
    expect(decodeCompleter(greedy)!.anchorStrength).toBe(1.8);
    // Out-of-range values fall back to the default on decode.
    const bad = new URLSearchParams(pt);
    bad.set("anc", "99");
    expect(decodeCompleter(bad)!.anchorStrength).toBe(1.0);
  });

  it("encodes site pins as bare species slugs alongside feature pins", () => {
    const params = encodeCompleter(
      {
        modelId: "reg-m-a",
        fixedIdxs: [3], // feature pin: Calyrex-Shadow @ Life Orb
        fixedSites: [1], // site pin: Alolan Ninetales (any item)
        inactiveTracks: [],
        excludedSpecies: [],
        includedSpecies: [],
        usePT: true,
        temperature: 1.0,
        anchorStrength: 1.0,
        ptRuns: 10,
        ptLadder: 7,
        ptSweeps: 20000,
        ptSwapInterval: 10,
        seed: null,
      },
      model,
    );
    const d = decodeCompleter(params)!;
    const slugIndex = buildSlugIndex(model);
    const site = d.features.find((f) => f.itemSlug === null)!;
    const feature = d.features.find((f) => f.itemSlug !== null)!;
    expect(resolveSpeciesSlug(slugIndex, model, site.speciesSlug)).toBe("Alolan Ninetales");
    expect(resolveFeature(slugIndex, model, feature.speciesSlug, feature.itemSlug).idx).toBe(3);
  });

  it("round-trips species-only mode (deactivated tracks)", () => {
    const params = encodeCompleter(
      {
        modelId: "reg-m-a",
        fixedIdxs: [],
        fixedSites: [1],
        inactiveTracks: [0], // item track off
        excludedSpecies: [],
        includedSpecies: [],
        usePT: true,
        temperature: 1.0,
        anchorStrength: 1.0,
        ptRuns: 10,
        ptLadder: 7,
        ptSweeps: 20000,
        ptSwapInterval: 10,
        seed: null,
      },
      model,
    );
    expect(params.get("d")).toBe("0");
    expect(decodeCompleter(params)!.inactiveTracks).toEqual([0]);
  });

  it("encodes a non-default seed and advanced knobs", () => {
    const params = encodeCompleter(
      {
        modelId: "test-species-item",
        fixedIdxs: [0],
        fixedSites: [],
        inactiveTracks: [],
        excludedSpecies: [],
        includedSpecies: [],
        usePT: true,
        temperature: 0.7,
        anchorStrength: 1.0,
        ptRuns: 15,
        ptLadder: 7,
        ptSweeps: 20000,
        ptSwapInterval: 10,
        seed: 7,
      },
      model,
    );
    expect(params.get("seed")).toBe("7");
    expect(params.get("tmp")).not.toBeNull();
    expect(params.get("a")).toBe("15-7-20000-10");

    const d = decodeCompleter(params)!;
    expect(d.temperature).toBe(0.7);
    expect(d.ptRuns).toBe(15);
    expect(d.seed).toBe(7);
  });
});
