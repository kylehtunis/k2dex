export interface ModelSummary {
  id: string;
  displayName: string;
  description?: string;
  regulation: string;
  featureDimensions: 1 | 2;
  /** Track (attribute) definitions the model carries, for the UI toggle.
   * Empty for species-only models. */
  tracks: { name: string; unique: boolean }[];
  V: number;
  nCorpusTeams: number;
  latestTournamentDate: string;
  teamSize: number;
  isNew?: boolean;
}

export interface Manifest {
  schemaVersion: number;
  defaultModel: string;
  models: ModelSummary[];
}

interface ManifestJson {
  schema_version: number;
  default_model: string;
  models: Array<{
    id: string;
    display_name: string;
    description?: string;
    regulation: string;
    feature_dimensions: 1 | 2;
    tracks?: { name: string; unique: boolean }[];
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
  cached = {
    schemaVersion: raw.schema_version,
    defaultModel: raw.default_model,
    models: raw.models.map((m) => ({
      id: m.id,
      displayName: m.display_name,
      ...(m.description !== undefined && { description: m.description }),
      regulation: m.regulation,
      featureDimensions: m.feature_dimensions,
      tracks: m.tracks ?? [],
      V: m.V,
      nCorpusTeams: m.n_corpus_teams,
      latestTournamentDate: m.latest_tournament_date,
      teamSize: m.team_size,
      ...(m.new && { isNew: true }),
    })),
  };
  return cached;
}
