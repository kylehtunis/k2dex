import type { TrackDef } from "../sampler/types";

export interface ModelSummary {
  id: string;
  displayName: string;
  /** Resolved from the manifest's shared `types` block (description lives on
   * the product type, not the individual model). */
  description?: string;
  regulation: string;
  /** Product-tier type (e.g. "standard"). One artifact per (regulation, type). */
  type: string;
  /** Track (attribute) definitions the model carries, for the UI toggle.
   * Empty for species-only models. */
  tracks: TrackDef[];
  V: number;
  nCorpusTeams: number;
  latestTournamentDate: string;
  teamSize: number;
  isNew?: boolean;
}

export interface Manifest {
  schemaVersion: number;
  defaultModel: string;
  /** Product-type metadata, keyed by type name. Description is shared across
   * all regulations that carry that type. */
  types: Record<string, { description?: string }>;
  models: ModelSummary[];
}

interface ManifestJson {
  schema_version: number;
  default_model: string;
  types?: Record<string, { description?: string }>;
  models: Array<{
    id: string;
    display_name: string;
    regulation: string;
    type?: string;
    tracks?: TrackDef[];
    V: number;
    n_corpus_teams: number;
    latest_tournament_date: string;
    team_size: number;
    new?: boolean;
  }>;
}

let cached: Manifest | null = null;

export async function loadManifest(): Promise<Manifest> {
  if (cached) return cached;
  const url = `${import.meta.env.BASE_URL}models/manifest.json`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to load manifest: ${r.status}`);
  const raw: ManifestJson = await r.json();
  const types = raw.types ?? {};
  cached = {
    schemaVersion: raw.schema_version,
    defaultModel: raw.default_model,
    types,
    models: raw.models.map((m) => {
      const type = m.type ?? "standard";
      const description = types[type]?.description;
      return {
        id: m.id,
        displayName: m.display_name,
        ...(description !== undefined && { description }),
        regulation: m.regulation,
        type,
        tracks: m.tracks ?? [],
        V: m.V,
        nCorpusTeams: m.n_corpus_teams,
        latestTournamentDate: m.latest_tournament_date,
        teamSize: m.team_size,
        ...(m.new && { isNew: true }),
      };
    }),
  };
  return cached;
}

/** The model for a (regulation, type) cell. Defaults to the "standard" type,
 * the only tier today. Returns undefined when no such cell exists. */
export function modelForRegulation(
  manifest: Manifest,
  regulation: string,
  type = "standard",
): ModelSummary | undefined {
  return manifest.models.find(
    (m) => m.regulation === regulation && m.type === type,
  );
}
