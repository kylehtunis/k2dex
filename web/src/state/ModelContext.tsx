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
  useState,
  type ReactNode,
} from "react";
import { loadModel, loadTeamCounts } from "../sampler/model";
import type { IsingModel, TeamCounts } from "../sampler/types";
import { loadManifest, type Manifest } from "./manifest";

interface CacheEntry {
  model: IsingModel;
  teamCounts: TeamCounts;
}

interface ModelContextValue {
  modelId: string;
  setModelId: (id: string) => void;
  model: IsingModel | null;
  teamCounts: TeamCounts | null;
  manifest: Manifest | null;
  status: "idle" | "loading" | "ready" | "error";
  error: Error | null;
}

const ModelContext = createContext<ModelContextValue | null>(null);

const STORAGE_KEY = "k2dex.modelId";
const LEGACY_STORAGE_KEY = "k2dex.phaseKey";

const LEGACY_MODEL_MAP: Record<string, string> = {
  species: "reg-m-a-species",
  species_item: "reg-m-a-species-item",
};

function readStoredModelId(manifest: Manifest): string {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v && manifest.models.some((m) => m.id === v)) return v;

    const legacy = localStorage.getItem(LEGACY_STORAGE_KEY);
    if (legacy && LEGACY_MODEL_MAP[legacy]) {
      const mapped = LEGACY_MODEL_MAP[legacy];
      if (manifest.models.some((m) => m.id === mapped)) return mapped;
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
    Promise.all([loadModel(modelId), loadTeamCounts(modelId)])
      .then(([model, teamCounts]) => {
        if (cancelled) return;
        setCache((c) => ({ ...c, [modelId]: { model, teamCounts } }));
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
  const value: ModelContextValue = {
    modelId,
    setModelId,
    model: entry?.model ?? null,
    teamCounts: entry?.teamCounts ?? null,
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
