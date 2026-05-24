// Persists user-input state across tab switches so form values survive
// navigation. Only stores form inputs — computed results (run output,
// error messages, timers) are intentionally excluded.
//
// Resets all state when the active model changes (vocab indices from
// one model are meaningless under another).

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  PT_LADDER_LEVELS,
  PT_RUNS,
  PT_SWEEPS,
  PT_SWAP_INTERVAL,
} from "../constants";
import { useModel } from "./ModelContext";

export interface CompleterInputs {
  fixedIdxs: number[];
  excludedSpecies: string[];
  fieldWeight: number;
  temperature: number;
  usePT: boolean;
  ptRuns: number;
  ptLadder: number;
  ptSweeps: number;
  ptSwapInterval: number;
}

export interface AnalysisInputs {
  teamIdxs: number[];
  fieldWeight: number;
}

const COMPLETER_DEFAULTS: CompleterInputs = {
  fixedIdxs: [],
  excludedSpecies: [],
  fieldWeight: 0.3,
  temperature: 0.5,
  usePT: true,
  ptRuns: PT_RUNS,
  ptLadder: PT_LADDER_LEVELS,
  ptSweeps: PT_SWEEPS,
  ptSwapInterval: PT_SWAP_INTERVAL,
};

const ANALYSIS_DEFAULTS: AnalysisInputs = {
  teamIdxs: [],
  fieldWeight: 0.3,
};

interface PageStateContextValue {
  completer: CompleterInputs;
  setCompleter: (patch: Partial<CompleterInputs>) => void;
  analysis: AnalysisInputs;
  setAnalysis: (patch: Partial<AnalysisInputs>) => void;
}

const PageStateContext = createContext<PageStateContextValue | null>(null);

export function PageStateProvider({ children }: { children: ReactNode }) {
  const { model } = useModel();
  const phaseKey = model?.name ?? "—";
  const prevPhase = useRef(phaseKey);

  const [completer, setCompleterRaw] = useState<CompleterInputs>(
    () => ({ ...COMPLETER_DEFAULTS }),
  );
  const [analysis, setAnalysisRaw] = useState<AnalysisInputs>(
    () => ({ ...ANALYSIS_DEFAULTS }),
  );

  useEffect(() => {
    if (prevPhase.current !== phaseKey) {
      prevPhase.current = phaseKey;
      setCompleterRaw({ ...COMPLETER_DEFAULTS });
      setAnalysisRaw({ ...ANALYSIS_DEFAULTS });
    }
  }, [phaseKey]);

  const setCompleter = useCallback(
    (patch: Partial<CompleterInputs>) =>
      setCompleterRaw((prev) => ({ ...prev, ...patch })),
    [],
  );
  const setAnalysis = useCallback(
    (patch: Partial<AnalysisInputs>) =>
      setAnalysisRaw((prev) => ({ ...prev, ...patch })),
    [],
  );

  return (
    <PageStateContext.Provider
      value={{ completer, setCompleter, analysis, setAnalysis }}
    >
      {children}
    </PageStateContext.Provider>
  );
}

export function usePageState() {
  const ctx = useContext(PageStateContext);
  if (!ctx) throw new Error("usePageState outside PageStateProvider");
  return ctx;
}
