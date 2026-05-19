// Single source of truth for the currently-loaded model.
//
// loadModel + loadTeamCounts hit /models/<name>/{...} once per choice;
// results are cached in component state so switching back and forth
// doesn't re-fetch. The "loading" status is exposed so pages can
// render a placeholder during the initial bytecode + asset load
// (Phase 3 model is ~600 KB; fast on broadband but worth indicating).

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

export type PhaseKey = "species" | "species_item";

interface CacheEntry {
  model: IsingModel;
  teamCounts: TeamCounts;
}

interface ModelContextValue {
  phaseKey: PhaseKey;
  setPhaseKey: (k: PhaseKey) => void;
  model: IsingModel | null;
  teamCounts: TeamCounts | null;
  status: "idle" | "loading" | "ready" | "error";
  error: Error | null;
}

const ModelContext = createContext<ModelContextValue | null>(null);

const STORAGE_KEY = "k2dex.phaseKey";

function readStoredPhase(): PhaseKey {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "species" || v === "species_item") return v;
  } catch { /* localStorage may be unavailable */ }
  return "species_item";
}

export function ModelProvider({ children }: { children: ReactNode }) {
  const [phaseKey, setPhaseKeyState] = useState<PhaseKey>(() => readStoredPhase());
  const [cache, setCache] = useState<Partial<Record<PhaseKey, CacheEntry>>>({});
  const [status, setStatus] = useState<ModelContextValue["status"]>("idle");
  const [error, setError] = useState<Error | null>(null);

  const setPhaseKey = useCallback((k: PhaseKey) => {
    setPhaseKeyState(k);
    try { localStorage.setItem(STORAGE_KEY, k); } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    if (cache[phaseKey]) {
      setStatus("ready");
      setError(null);
      return;
    }
    let cancelled = false;
    setStatus("loading");
    setError(null);
    Promise.all([loadModel(phaseKey), loadTeamCounts(phaseKey)])
      .then(([model, teamCounts]) => {
        if (cancelled) return;
        setCache((c) => ({ ...c, [phaseKey]: { model, teamCounts } }));
        setStatus("ready");
      })
      .catch((e: Error) => {
        if (cancelled) return;
        setError(e);
        setStatus("error");
      });
    return () => { cancelled = true; };
  }, [phaseKey, cache]);

  const entry = cache[phaseKey] ?? null;
  const value: ModelContextValue = {
    phaseKey,
    setPhaseKey,
    model: entry?.model ?? null,
    teamCounts: entry?.teamCounts ?? null,
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
