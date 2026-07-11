// Single source of truth for the currently-loaded model.
//
// On mount, fetches models/manifest.json to discover available models,
// then loads the selected model's artifacts. Results are cached in
// component state so switching back and forth doesn't re-fetch.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { loadModel, loadSpeciesGraph, loadTeamCounts } from "../sampler/model";
import {
  buildCorpusScoreIndex,
  type CorpusScoreIndex,
} from "../render/corpusScore";
import type { IsingModel, SpeciesGraph, TeamCounts } from "../sampler/types";
import { loadManifest, type Manifest } from "./manifest";

interface CacheEntry {
  model: IsingModel;
  teamCounts: TeamCounts;
  speciesGraph: SpeciesGraph | null;
}

interface ModelContextValue {
  modelId: string;
  setModelId: (id: string) => void;
  model: IsingModel | null;
  teamCounts: TeamCounts | null;
  /** Precomputed species-pair interaction graph (APC-corrected synergy).
   * Null for species-only models or before load completes. */
  speciesGraph: SpeciesGraph | null;
  /** Empirical corpus Score distribution (every observed roster scored at
   * fw = 1), built once per loaded model. Feeds the Percentile displays. */
  corpusScoreIndex: CorpusScoreIndex | null;
  manifest: Manifest | null;
  status: "idle" | "loading" | "ready" | "error";
  error: Error | null;
}

const ModelContext = createContext<ModelContextValue | null>(null);

const STORAGE_KEY = "k2dex.modelId";
const LEGACY_STORAGE_KEY = "k2dex.phaseKey";

// Retired split-model ids -> the unified per-regulation model. After the
// schema collapse each regulation has exactly one artifact, so a returning
// visitor's stored id (or legacy /science phaseKey) maps to its regulation.
const LEGACY_MODEL_MAP: Record<string, string> = {
  species: "reg-m-a",
  species_item: "reg-m-a",
  "reg-m-a-species": "reg-m-a",
  "reg-m-a-species-item": "reg-m-a",
  "reg-m-a-species-item-weighted": "reg-m-a",
  "reg-m-b-experimental": "reg-m-b",
  "reg-m-b-species-item-boltzmann": "reg-m-b",
};

function readStoredModelId(manifest: Manifest): string {
  const inManifest = (id: string) => manifest.models.some((m) => m.id === id);
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v) {
      if (inManifest(v)) return v;
      const remapped = LEGACY_MODEL_MAP[v];
      if (remapped && inManifest(remapped)) return remapped;
    }

    const legacy = localStorage.getItem(LEGACY_STORAGE_KEY);
    if (legacy && LEGACY_MODEL_MAP[legacy] && inManifest(LEGACY_MODEL_MAP[legacy])) {
      return LEGACY_MODEL_MAP[legacy];
    }
  } catch { /* localStorage may be unavailable */ }
  return manifest.defaultModel;
}

export function ModelProvider({ children }: { children: ReactNode }) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [modelId, setModelIdState] = useState<string>("");
  const [cache, setCache] = useState<Record<string, CacheEntry>>({});
  const [status, setStatus] = useState<ModelContextValue["status"]>("idle");
  const [error, setError] = useState<Error | null>(null);

  // Load manifest on mount.
  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    loadManifest()
      .then((m) => {
        if (cancelled) return;
        setManifest(m);
        const id = readStoredModelId(m);
        setModelIdState(id);
        try {
          localStorage.setItem(STORAGE_KEY, id);
          localStorage.removeItem(LEGACY_STORAGE_KEY);
        } catch { /* ignore */ }
      })
      .catch((e: Error) => {
        if (cancelled) return;
        setError(e);
        setStatus("error");
      });
    return () => { cancelled = true; };
  }, []);

  const setModelId = useCallback((id: string) => {
    setModelIdState(id);
    try { localStorage.setItem(STORAGE_KEY, id); } catch { /* ignore */ }
  }, []);

  // Load model artifacts when modelId changes (and manifest is ready).
  useEffect(() => {
    if (!modelId || !manifest) return;
    if (cache[modelId]) {
      setStatus("ready");
      setError(null);
      return;
    }
    let cancelled = false;
    setStatus("loading");
    setError(null);
    Promise.all([loadModel(modelId), loadTeamCounts(modelId), loadSpeciesGraph(modelId)])
      .then(([model, teamCounts, speciesGraph]) => {
        if (cancelled) return;
        setCache((c) => ({ ...c, [modelId]: { model, teamCounts, speciesGraph } }));
        setStatus("ready");
      })
      .catch((e: Error) => {
        if (cancelled) return;
        setError(e);
        setStatus("error");
      });
    return () => { cancelled = true; };
  }, [modelId, manifest, cache]);

  const entry = cache[modelId] ?? null;
  const corpusScoreIndex = useMemo(
    () =>
      entry ? buildCorpusScoreIndex(entry.model, entry.teamCounts) : null,
    [entry],
  );
  const value: ModelContextValue = {
    modelId,
    setModelId,
    model: entry?.model ?? null,
    teamCounts: entry?.teamCounts ?? null,
    speciesGraph: entry?.speciesGraph ?? null,
    corpusScoreIndex,
    manifest,
    status,
    error,
  };
  return <ModelContext.Provider value={value}>{children}</ModelContext.Provider>;
}

export function useModel(): ModelContextValue {
  const ctx = useContext(ModelContext);
  if (ctx === null) {
    throw new Error("useModel must be used inside a <ModelProvider>");
  }
  return ctx;
}
