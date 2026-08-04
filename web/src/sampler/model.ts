// Model loader: fetches the precompute artifacts from /models/<name>/
// and reconstructs the in-memory IsingModel.
//
// File schema (mirrored from precompute.py):
//   meta.json         vocab, sites, site_of, tracks, track_values, scalars
//   J.bin             float32 lower triangle, V*(V-1)/2 entries
//                     ordering: [J[i,j] for i in 1..V-1 for j in 0..i-1]
//   h.bin             float32, V entries
//   m.bin             float32, V entries
//   team_counts.json  { "0-1-2-3-4-5": count, ... } (loaded separately)

import type { IsingModel, SpeciesGraph, TeamCounts } from "./types";

interface MetaJson {
  id?: string;
  name?: string;
  display_name?: string;
  regulation?: string;
  feature_dimensions?: number;
  latest_tournament_date?: string;
  V: number;
  team_size: number;
  n_corpus_teams: number;
  vocab: string[];
  sites: string[];
  site_of: number[];
  tracks: { name: string; unique: boolean }[];
  track_values: (string | null)[][];
  fit: { method: string; C?: number; lambda?: number; min_team_count: number };
  schema_version: number;
}

const SUPPORTED_SCHEMA_VERSIONS = [3];

/** Fetch a binary file as a Float32Array (assumes little-endian, native
 * to all platforms we care about). */
async function fetchFloat32(url: string): Promise<Float32Array> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status} ${r.statusText}`);
  const buf = await r.arrayBuffer();
  return new Float32Array(buf);
}

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

/** Reconstruct the symmetric V×V J matrix from its row-major strict
 * lower triangle. Output is a flat Float64Array of length V*V with
 * `J[i*V + j] === J[j*V + i]` and zero diagonal. */
export function unpackLowerTriangle(
  flat: Float32Array,
  V: number,
): Float64Array {
  const expected = (V * (V - 1)) / 2;
  if (flat.length !== expected) {
    throw new Error(
      `unpackLowerTriangle: got ${flat.length} entries, expected ${expected} for V=${V}`,
    );
  }
  const J = new Float64Array(V * V);
  let k = 0;
  // Same ordering as np.tril_indices(V, k=-1): (i,j) with i > j.
  for (let i = 1; i < V; i++) {
    const rowI = i * V;
    for (let j = 0; j < i; j++) {
      const v = flat[k++];
      J[rowI + j] = v;
      J[j * V + i] = v;
    }
  }
  return J;
}

/** Convenience views derived from the factored (sites + tracks) schema.
 * `siteFeatures` groups feature indices by site (a pure projection of
 * `siteOf`, not stored in meta.json); `speciesOf`/`itemOf` reconstruct the
 * old per-feature (species, item) arrays that display and render code use. */
export interface DerivedFactored {
  siteFeatures: number[][];
  speciesOf: string[];
  itemOf: (string | null)[];
}

/** The factored (sites + tracks) fields of an IsingModel, as loaded from a
 * v3 artifact or reconstructed from per-feature species/item arrays. */
export interface FactoredFields {
  sites: string[];
  siteOf: number[];
  tracks: { name: string; unique: boolean }[];
  trackValues: (string | null)[][];
  siteFeatures: number[][];
}

/** Inverse of {@link deriveFactored}: build the factored schema fields from
 * per-feature (species, item) arrays. A model with any non-null item carries a
 * single unique "item" track; an all-null species-only model carries no tracks.
 * Used to construct in-memory models from the legacy species/item view (tests,
 * ad-hoc models); the production loader reads the factored fields directly. */
export function factoredFromSpeciesItem(
  speciesOf: readonly string[],
  itemOf: readonly (string | null)[],
): FactoredFields {
  const sites: string[] = [];
  const siteIndex = new Map<string, number>();
  const siteOf: number[] = [];
  const siteFeatures: number[][] = [];
  for (let i = 0; i < speciesOf.length; i++) {
    const sp = speciesOf[i];
    let s = siteIndex.get(sp);
    if (s === undefined) {
      s = sites.length;
      siteIndex.set(sp, s);
      sites.push(sp);
      siteFeatures.push([]);
    }
    siteOf.push(s);
    siteFeatures[s].push(i);
  }
  const hasItems = itemOf.some((x) => x !== null);
  const tracks = hasItems ? [{ name: "item", unique: true }] : [];
  const trackValues: (string | null)[][] = hasItems
    ? itemOf.map((it) => [it])
    : itemOf.map(() => []);
  return { sites, siteOf, tracks, trackValues, siteFeatures };
}

export function deriveFactored(
  sites: readonly string[],
  siteOf: readonly number[],
  trackValues: readonly (readonly (string | null)[])[],
): DerivedFactored {
  const siteFeatures: number[][] = sites.map(() => []);
  const speciesOf: string[] = new Array(siteOf.length);
  const itemOf: (string | null)[] = new Array(siteOf.length);
  for (let i = 0; i < siteOf.length; i++) {
    const s = siteOf[i];
    siteFeatures[s].push(i);
    speciesOf[i] = sites[s];
    itemOf[i] = trackValues[i].length > 0 ? trackValues[i][0] : null;
  }
  return { siteFeatures, speciesOf, itemOf };
}

/** A sampling view of `model` with the given tracks made degenerate (their
 * `unique` flag cleared). The attribute toggle uses this: a deactivated track
 * carries no uniqueness constraint and the sampler doesn't reroll it (the
 * caller also sets pReroll=0), so the species-swap conditional sums over its
 * values freely — the exact marginal over that attribute. Only `tracks` is
 * changed; every other field (J/h/siteFeatures/…) is shared. Returns `model`
 * unchanged when nothing is deactivated. */
export function withInactiveTracks(
  model: IsingModel,
  inactive: readonly number[],
): IsingModel {
  if (inactive.length === 0) return model;
  const set = new Set(inactive);
  return {
    ...model,
    tracks: model.tracks.map((t, i) => (set.has(i) ? { ...t, unique: false } : t)),
  };
}

/** Load all four model artifacts in parallel. `basePath` is the URL
 * prefix that contains `<modelName>/{meta.json,J.bin,h.bin,m.bin}`;
 * defaults to "models" which combines with Vite's `base` to resolve
 * relative to the deployed site root. */
export async function loadModel(
  modelName: string,
  basePath = "models",
): Promise<IsingModel> {
  const base = `${import.meta.env.BASE_URL}${basePath}/${modelName}`;
  const [meta, jFlat, hF32, mF32] = await Promise.all([
    fetchJson<MetaJson>(`${base}/meta.json`),
    fetchFloat32(`${base}/J.bin`),
    fetchFloat32(`${base}/h.bin`),
    fetchFloat32(`${base}/m.bin`),
  ]);

  if (!SUPPORTED_SCHEMA_VERSIONS.includes(meta.schema_version)) {
    throw new Error(
      `Unsupported model schema version: ${meta.schema_version}`,
    );
  }
  if (hF32.length !== meta.V || mF32.length !== meta.V) {
    throw new Error(
      `Model size mismatch: V=${meta.V} but h.bin=${hF32.length}, m.bin=${mF32.length}`,
    );
  }

  const J = unpackLowerTriangle(jFlat, meta.V);
  const h = new Float64Array(hF32);
  const m = new Float64Array(mF32);
  const indexOf = new Map<string, number>();
  for (let i = 0; i < meta.vocab.length; i++) {
    indexOf.set(meta.vocab[i], i);
  }

  const id = meta.id ?? meta.name ?? modelName;
  const { siteFeatures, speciesOf, itemOf } = deriveFactored(
    meta.sites,
    meta.site_of,
    meta.track_values,
  );

  return {
    id,
    displayName: meta.display_name ?? id,
    regulation: meta.regulation ?? "",
    featureDimensions: meta.feature_dimensions ?? (meta.tracks.length + 1),
    latestTournamentDate: meta.latest_tournament_date ?? "",
    V: meta.V,
    teamSize: meta.team_size,
    vocab: meta.vocab,
    sites: meta.sites,
    siteOf: meta.site_of,
    tracks: meta.tracks,
    trackValues: meta.track_values,
    siteFeatures,
    speciesOf,
    itemOf,
    m,
    J,
    h,
    indexOf,
    nCorpusTeams: meta.n_corpus_teams,
    name: id,
  };
}

/** Load the corpus team-count index for `nearestObserved` queries.
 * Returns a Map keyed by sorted-index "-"-joined strings. */
export async function loadTeamCounts(
  modelName: string,
  basePath = "models",
): Promise<TeamCounts> {
  const base = `${import.meta.env.BASE_URL}${basePath}/${modelName}`;
  const obj = await fetchJson<Record<string, number>>(`${base}/team_counts.json`);
  const out = new Map<string, number>();
  for (const [k, v] of Object.entries(obj)) {
    out.set(k, v);
  }
  return out;
}

interface SpeciesGraphJson {
  species: string[];
  synergy_ut: number[];
}

/** Unpack a strict upper-triangle flat array into a symmetric S×S
 * Float64Array (row-major, zero diagonal).
 *
 * Throws on a length mismatch (as `unpackLowerTriangle` does): reading past
 * the end yields `undefined`, which a Float64Array silently stores as NaN, and
 * /meta would then render NaN synergies with nothing having errored. */
function unpackUpperTriangle(ut: number[], S: number): Float64Array {
  const expected = (S * (S - 1)) / 2;
  if (ut.length !== expected) {
    throw new Error(
      `species_graph: expected ${expected} upper-triangle entries for ${S} species, got ${ut.length}`,
    );
  }
  const out = new Float64Array(S * S);
  let k = 0;
  for (let i = 0; i < S; i++) {
    for (let j = i + 1; j < S; j++) {
      const v = ut[k++];
      out[i * S + j] = v;
      out[j * S + i] = v;
    }
  }
  return out;
}

/** Load the precomputed species-pair interaction graph. Returns null if
 * the artifact doesn't exist (species-only models don't have one). */
export async function loadSpeciesGraph(
  modelName: string,
  basePath = "models",
): Promise<SpeciesGraph | null> {
  const base = `${import.meta.env.BASE_URL}${basePath}/${modelName}`;
  const url = `${base}/species_graph.json`;
  const r = await fetch(url);
  if (!r.ok) return null;
  const data: SpeciesGraphJson = await r.json();
  const S = data.species.length;
  const indexOf = new Map<string, number>();
  for (let i = 0; i < S; i++) indexOf.set(data.species[i], i);
  return {
    species: data.species,
    synergy: unpackUpperTriangle(data.synergy_ut, S),
    indexOf,
  };
}
