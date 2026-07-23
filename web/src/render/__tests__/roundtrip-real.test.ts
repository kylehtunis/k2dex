import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import type { IsingModel, TrackDef } from "../../sampler/types";
import { deriveFactored } from "../../sampler/model";
import { buildSlugIndex, resolveFeature } from "../vocab-match";
import { encodeCore, decodeCore } from "../shareLink";

const MODELS_DIR = resolve(__dirname, "../../../public/models");

/** Every committed model directory that carries a meta.json. Discovered (not
 * hard-coded) so the round-trip covers whatever regulations are currently built
 * — a regulation can be temporarily absent while its corpus is being rebuilt. */
function committedModels(): string[] {
  return readdirSync(MODELS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory() && existsSync(resolve(MODELS_DIR, d.name, "meta.json")))
    .map((d) => d.name);
}

function loadMeta(name: string): IsingModel {
  const p = resolve(MODELS_DIR, name, "meta.json");
  const meta = JSON.parse(readFileSync(p, "utf8"));
  const V = meta.V as number;
  const vocab = meta.vocab as string[];
  const sites = meta.sites as string[];
  const siteOf = meta.site_of as number[];
  const tracks = meta.tracks as TrackDef[];
  const trackValues = meta.track_values as (string | null)[][];
  const { siteFeatures, speciesOf, itemOf } = deriveFactored(sites, siteOf, trackValues);
  const indexOf = new Map<string, number>();
  vocab.forEach((v, i) => indexOf.set(v, i));
  return {
    id: meta.id ?? meta.name,
    displayName: meta.display_name ?? meta.name,
    regulation: meta.regulation ?? "",
    latestTournamentDate: meta.latest_tournament_date ?? "",
    V,
    teamSize: meta.team_size,
    vocab,
    sites,
    siteOf,
    tracks,
    trackValues,
    siteFeatures,
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
  for (const name of committedModels()) {
    it(`every vocab index round-trips for ${name}`, () => {
      const model = loadMeta(name);
      const slugIndex = buildSlugIndex(model);
      const failures: string[] = [];
      for (let i = 0; i < model.V; i++) {
        const token = encodeCore(model.id, [i], model);
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
