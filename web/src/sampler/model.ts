// Model loader: fetches the precompute artifacts from /models/<name>/
// and reconstructs the in-memory IsingModel.
//
// File schema (mirrored from precompute.py):
//   meta.json         vocab, species_of, item_of, scalars
//   J.bin             float32 lower triangle, V*(V-1)/2 entries
//                     ordering: [J[i,j] for i in 1..V-1 for j in 0..i-1]
//   h.bin             float32, V entries
//   m.bin             float32, V entries
//   team_counts.json  { "0-1-2-3-4-5": count, ... } (loaded separately)

import type { IsingModel, TeamCounts } from "./types";

interface MetaJson {
  name: string;
  V: number;
  team_size: number;
  n_corpus_teams: number;
  vocab: string[];
  species_of: string[];
  item_of: (string | null)[];
  fit: { method: string; C: number; min_team_count: number; min_teams: number };
  schema_version: number;
}

const SCHEMA_VERSION = 1;

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

/** Load all four model artifacts in parallel. `basePath` is the URL
 * prefix that contains `<modelName>/{meta.json,J.bin,h.bin,m.bin}`;
 * defaults to "models" which combines with Vite's `base` to resolve
 * relative to the deployed site root. */
export async function loadModel(
  modelName: "species" | "species_item",
  basePath = "models",
): Promise<IsingModel> {
  const base = `${import.meta.env.BASE_URL}${basePath}/${modelName}`;
  const [meta, jFlat, hF32, mF32] = await Promise.all([
    fetchJson<MetaJson>(`${base}/meta.json`),
    fetchFloat32(`${base}/J.bin`),
    fetchFloat32(`${base}/h.bin`),
    fetchFloat32(`${base}/m.bin`),
  ]);

  if (meta.schema_version !== SCHEMA_VERSION) {
    throw new Error(
      `Model schema mismatch: file=${meta.schema_version}, runtime=${SCHEMA_VERSION}`,
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

  return {
    V: meta.V,
    teamSize: meta.team_size,
    vocab: meta.vocab,
    speciesOf: meta.species_of,
    itemOf: meta.item_of,
    m,
    J,
    h,
    indexOf,
    nCorpusTeams: meta.n_corpus_teams,
    name: meta.name,
  };
}

/** Load the corpus team-count index for `nearestObserved` queries.
 * Returns a Map keyed by sorted-index "-"-joined strings. */
export async function loadTeamCounts(
  modelName: "species" | "species_item",
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
