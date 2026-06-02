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
    name: meta.name,
  };
}

describe("real-model share-token round-trip", () => {
  for (const name of ["species", "species_item"] as const) {
    it(`every vocab index round-trips for ${name}`, () => {
      const model = loadMeta(name);
      const slugIndex = buildSlugIndex(model);
      const failures: string[] = [];
      for (let i = 0; i < model.V; i++) {
        const token = encodeCore(model.name as "species" | "species_item", 0.3, [i], model);
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
