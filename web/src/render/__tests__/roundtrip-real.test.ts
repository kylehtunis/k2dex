import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { IsingModel } from "../../sampler/types";
import { buildSlugIndex, resolveFeature } from "../vocab-match";
import { encodeCore, decodeCore } from "../shareLink";

function loadMeta(name: string): IsingModel {
  const p = resolve(__dirname, `../../../public/models/${name}/meta.json`);
  const meta = JSON.parse(readFileSync(p, "utf8"));
  const V = meta.V as number;
  const vocab = meta.vocab as string[];
  const speciesOf = meta.species_of as string[];
  const itemOf = meta.item_of as (string | null)[];
  const indexOf = new Map<string, number>();
  vocab.forEach((v, i) => indexOf.set(v, i));
  return {
    id: meta.id ?? meta.name,
    displayName: meta.display_name ?? meta.name,
    regulation: meta.regulation ?? "",
    featureDimensions: meta.feature_dimensions ?? (itemOf.some((x: string | null) => x !== null) ? 2 : 1),
    latestTournamentDate: meta.latest_tournament_date ?? "",
    V,
    teamSize: meta.team_size,
    vocab,
    speciesOf,
    itemOf,
    m: new Float64Array(V),
    J: new Float64Array(V * V),
    h: new Float64Array(V),
    indexOf,
    nCorpusTeams: meta.n_corpus_teams,
    name: meta.id ?? meta.name,
  };
}

describe("real-model share-token round-trip", () => {
  for (const name of ["reg-m-a-species", "reg-m-a-species-item"]) {
    it(`every vocab index round-trips for ${name}`, () => {
      const model = loadMeta(name);
      const slugIndex = buildSlugIndex(model);
      const failures: string[] = [];
      for (let i = 0; i < model.V; i++) {
        const token = encodeCore(model.id, 0.3, [i], model);
        const decoded = decodeCore(token)!;
        const f = decoded.features[0];
        const r = resolveFeature(slugIndex, model, f.speciesSlug, f.itemSlug);
        if (r.idx === null) {
          failures.push(`${model.vocab[i]} -> token ${token} -> idx null`);
        }
      }
      expect(failures.slice(0, 20)).toEqual([]);
    });
  }
});
