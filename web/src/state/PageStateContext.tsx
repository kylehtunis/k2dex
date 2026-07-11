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

/** One filled roster slot. `feature` is the flat vocab index when the user has
 * pinned a specific item (a feature pin); `null` when only the species is
 * chosen and the completer fills the item (a site pin). */
export interface RosterSlot {
  site: number;
  feature: number | null;
}

export interface CompleterInputs {
  /** Ordered roster: the source of truth for the 6-slot editor. Feature pins
   * (`feature != null`) and site pins (`feature == null`) are derived from it
   * for the sampler; empty slots are simply the absence of an entry. */
  roster: RosterSlot[];
  /** Deactivated attribute tracks (indices into model.tracks). A deactivated
   * track is degenerate: not pinned to a value, not rerolled, marginalized out
   * of the completions, and hidden. Empty = all attributes active. */
  inactiveTracks: number[];
  excludedSpecies: string[];
  /** Inclusion allow-list (species names). When non-empty, the completer may
   * only place these Pokémon (plus pinned ones); empty = all legal Pokémon. */
  includedSpecies: string[];
  temperature: number;
  usePT: boolean;
  ptRuns: number;
  ptLadder: number;
  ptSweeps: number;
  ptSwapInterval: number;
  /** Show the sampler-diagnostic observables (top-5 mass, acceptance rates)
   * that only matter when the advanced PT knobs are being tuned. */
  showDiagnostics: boolean;
}

export interface AnalysisInputs {
  /** Ordered roster (mirrors the completer). Analysis is feature-level, so a
   * complete team is the slots whose `feature` is set; a species-only slot is
   * simply an in-progress pick that doesn't count toward the team yet. */
  roster: RosterSlot[];
}

const COMPLETER_DEFAULTS: CompleterInputs = {
  roster: [],
  inactiveTracks: [],
  excludedSpecies: [],
  includedSpecies: [],
  temperature: 1.0,
  usePT: true,
  ptRuns: PT_RUNS,
  ptLadder: PT_LADDER_LEVELS,
  ptSweeps: PT_SWEEPS,
  ptSwapInterval: PT_SWAP_INTERVAL,
  showDiagnostics: false,
};

const ANALYSIS_DEFAULTS: AnalysisInputs = {
  roster: [],
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
  const currentId = model?.id ?? "—";
  const prevId = useRef(currentId);

  const [completer, setCompleterRaw] = useState<CompleterInputs>(
    () => ({ ...COMPLETER_DEFAULTS }),
  );
  const [analysis, setAnalysisRaw] = useState<AnalysisInputs>(
    () => ({ ...ANALYSIS_DEFAULTS }),
  );

  // Reset stale indices when the active model changes. Done during render
  // (not in an effect) so the reset lands before any child page's effects:
  // useEffect fires child-first, so a reset in a parent effect would run
  // *after* a child's decode effect and clobber a just-restored shared link.
  // See react.dev "You Might Not Need an Effect → Adjusting some state when
  // a prop changes". The "—" placeholder (model not yet loaded) is not a
  // real model, so we only reset when leaving an already-loaded model.
  if (prevId.current !== currentId) {
    const leavingLoadedModel = prevId.current !== "—";
    prevId.current = currentId;
    if (leavingLoadedModel) {
      setCompleterRaw({ ...COMPLETER_DEFAULTS });
      setAnalysisRaw({ ...ANALYSIS_DEFAULTS });
    }
  }

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
