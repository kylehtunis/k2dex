// Team completer page.
//
// Layout mirrors app.py:_render_completer:
//   PageTitle  — eyebrow + h1 + corpus caption
//   §01        — Starting roster (slot strip)
//   §02        — Excluded row (when non-empty)
//   §03        — Constraints (pin + exclude multiselects)
//   §04        — Sampler (sliders + full-sampler toggle + run button)
//   §05        — Observables (post-run)
//   §06        — Suggested completion table (post-run)
//
// Fast path (default): MF marginals → uniqueness-respecting greedy
// fill → greedy descent. Returns one team. Wired in this task.
//
// Full statistical sampler (toggle on): PT MCMC in a Web Worker. Wired
// in Task 21.

import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  FIELD_WEIGHT_OPTIONS,
  GREEDY_MAX_SWAPS,
  PT_BURN_IN,
  PT_HOT_T,
  PT_LADDER_LEVELS,
  PT_RUNS,
  PT_SWAP_INTERVAL,
  PT_SWEEPS,
  TEAM_SIZE,
  TEMPERATURE_OPTIONS,
  TOP_COMPLETIONS,
} from "../constants";
import { useModel } from "../state/ModelContext";
import { PageTitle, SectionLabel, StatStrip } from "../render/atoms";
import { ExcludedRow, SlotStrip } from "../render/cells";
import {
  SpeciesSelect,
  VocabSelect,
  speciesOptions,
  vocabOptions,
} from "../components/VocabSelect";
import { runFastPath, type FastPathResult } from "../completer/fastPath";
import { teamObservables } from "../render/observables";
import { nearestObserved } from "../render/corpus";
import {
  CompletionRow,
  CompletionTable,
} from "../completer/CompletionRow";
import { formatSigned } from "../render/format";
import { runPT, type PTDistEntry } from "../completer/ptDriver";

/** Fingerprint of the inputs that produced a run. Used to decide
 * whether the next button click is a fresh "Sample" (inputs changed)
 * or a "Re-run" with a new seed (inputs unchanged). */
interface PTInputFingerprint {
  fixed: readonly number[];
  excluded: readonly number[];
  fieldWeight: number;
  temperature: number;
  ptRuns: number;
  ptLadder: number;
  ptSweeps: number;
  ptSwapInterval: number;
}

type RunState =
  | {
      mode: "fast";
      result: FastPathResult;
      fieldWeight: number;
    }
  | {
      mode: "pt";
      dist: PTDistEntry[];
      nKept: number;
      localAccept: number;
      swapAccept: number;
      fixed: readonly number[];
      excluded: readonly number[];
      fieldWeight: number;
      temperature: number;
      elapsedMs: number;
      seed: number;
      fingerprint: PTInputFingerprint;
    };

export function CompleterPage() {
  const { model, teamCounts, status } = useModel();
  const [searchParams, setSearchParams] = useSearchParams();

  // Form state. Reset pins/excludes when the model phase changes; the
  // vocab is different and stale indices would refer to wrong mons.
  const phaseKey = model?.name ?? "—";
  const [fixedIdxs, setFixedIdxs] = useState<number[]>([]);

  // Pre-pin a species from the ?pinned= query param (set by the /science page).
  // Resolved once when the model becomes ready; cleared from the URL afterwards.
  useEffect(() => {
    if (status !== "ready" || !model) return;
    const pinned = searchParams.get("pinned");
    if (!pinned) return;
    // Find the highest-marginal vocab entry for this species.
    let bestIdx = -1;
    let bestM = -1;
    for (let i = 0; i < model.V; i++) {
      if (model.speciesOf[i] === pinned && model.m[i] > bestM) {
        bestM = model.m[i];
        bestIdx = i;
      }
    }
    if (bestIdx !== -1) setFixedIdxs([bestIdx]);
    setSearchParams({}, { replace: true });
  }, [status, model, searchParams, setSearchParams]);
  const [excludedSpecies, setExcludedSpecies] = useState<string[]>([]);
  const [fieldWeight, setFieldWeight] = useState(0.5);
  const [temperature, setTemperature] = useState(0.5);
  const [usePT, setUsePT] = useState(false);
  // PT knobs — mirror app.py's "Sampler parameters" expander. Defaults
  // match the locked constants; the expander lets power-users tune.
  //
  // Burn-in is locked at PT_BURN_IN (3,000) — it's a fixed equilibration
  // cost driven by ladder structure, not a user-meaningful knob. The
  // worker runs ptSweeps + PT_BURN_IN sweeps total and discards the
  // burn-in, so what's labelled "Samples per run" is what users actually
  // keep.
  const [ptRuns, setPtRuns] = useState(PT_RUNS);
  const [ptLadder, setPtLadder] = useState(PT_LADDER_LEVELS);
  const [ptSweeps, setPtSweeps] = useState(PT_SWEEPS);
  const [ptSwapInterval, setPtSwapInterval] = useState(PT_SWAP_INTERVAL);
  // Monotonically incrementing seed so "Re-run" gives a fresh draw from
  // the same target distribution. Increments before each PT submit.
  const [seedCounter, setSeedCounter] = useState(1);
  const [running, setRunning] = useState(false);
  const [runState, setRunState] = useState<RunState | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  /** Wall-clock elapsed for the active run (PT mode). Updated every
   * 250ms so the user sees a ticking number during the long sample. */
  const [elapsedMs, setElapsedMs] = useState(0);
  const elapsedTimer = useRef<number | null>(null);

  useEffect(() => {
    setFixedIdxs([]);
    setExcludedSpecies([]);
    setRunState(null);
    setErrorMsg(null);
    // Snap PT knobs back to defaults so a power-user's overrides
    // don't silently carry across phase switches.
    setPtRuns(PT_RUNS);
    setPtLadder(PT_LADDER_LEVELS);
    setPtSweeps(PT_SWEEPS);
    setPtSwapInterval(PT_SWAP_INTERVAL);
    setSeedCounter(1);
  }, [phaseKey]);

  const vocabOpts = useMemo(
    () => (model ? vocabOptions(model) : []),
    [model],
  );
  const speciesOpts = useMemo(
    () => (model ? speciesOptions(model) : []),
    [model],
  );

  // Timer cleanup on unmount / model switch. Lives up here (before any
  // early return) so the hook call order stays constant. Inlining the
  // clearInterval avoids capturing stopTimer, which is declared later
  // in the body — that closure would hit a TDZ on first render.
  useEffect(() => {
    return () => {
      if (elapsedTimer.current !== null) {
        window.clearInterval(elapsedTimer.current);
        elapsedTimer.current = null;
      }
    };
  }, [phaseKey]);

  if (status === "loading" || model === null) {
    return <p style={{ color: "var(--lab-ink-muted)" }}>Loading model…</p>;
  }

  const corpusCaption =
    `Limitless 2026 Reg M-A · ${model.nCorpusTeams.toLocaleString()} teams`;

  const fixedNames = fixedIdxs.map((i) => model.vocab[i]);
  const fixedSpeciesSet = new Set(fixedIdxs.map((i) => model.speciesOf[i]));
  const overlap = excludedSpecies.filter((s) => fixedSpeciesSet.has(s));
  const overlapError =
    overlap.length > 0
      ? `Cannot be both pinned and excluded: ${overlap.join(", ")}`
      : null;
  const ptTemperatureError =
    usePT && temperature >= PT_HOT_T
      ? `Temperature (${temperature}) must be strictly less than hot-T (${PT_HOT_T}) for the parallel-tempered sampler.`
      : null;

  const formError = overlapError ?? ptTemperatureError;
  const canRun = !formError && !running;

  // True iff the most recent PT result was generated from the exact
  // inputs currently in the form. Drives the "Sample" → "Re-run" label
  // swap on the primary button.
  const isPTRerun = (() => {
    if (!usePT || runState?.mode !== "pt") return false;
    const excludedNow: number[] = [];
    const excludedSet = new Set(excludedSpecies);
    for (let i = 0; i < (model?.V ?? 0); i++) {
      if (excludedSet.has(model!.speciesOf[i])) excludedNow.push(i);
    }
    const fp = runState.fingerprint;
    if (fp.fieldWeight !== fieldWeight) return false;
    if (fp.temperature !== temperature) return false;
    if (fp.ptRuns !== ptRuns) return false;
    if (fp.ptLadder !== ptLadder) return false;
    if (fp.ptSweeps !== ptSweeps) return false;
    if (fp.ptSwapInterval !== ptSwapInterval) return false;
    if (fp.fixed.length !== fixedIdxs.length) return false;
    for (let i = 0; i < fp.fixed.length; i++) {
      if (fp.fixed[i] !== fixedIdxs[i]) return false;
    }
    if (fp.excluded.length !== excludedNow.length) return false;
    for (let i = 0; i < fp.excluded.length; i++) {
      if (fp.excluded[i] !== excludedNow[i]) return false;
    }
    return true;
  })();

  const startTimer = () => {
    const t0 = performance.now();
    setElapsedMs(0);
    elapsedTimer.current = window.setInterval(() => {
      setElapsedMs(performance.now() - t0);
    }, 250);
    return t0;
  };
  const stopTimer = () => {
    if (elapsedTimer.current !== null) {
      window.clearInterval(elapsedTimer.current);
      elapsedTimer.current = null;
    }
  };

  const onRun = () => {
    if (!canRun || !model) return;
    setErrorMsg(null);
    setRunning(true);

    if (!usePT) {
      // Fast path — synchronous, ~50–200ms. queueMicrotask lets the
      // "Running…" state render once before the loop blocks.
      queueMicrotask(() => {
        const r = runFastPath(model, {
          fixed: fixedIdxs,
          excludedSpecies,
          fieldWeight,
        });
        if (r.ok) {
          setRunState({ mode: "fast", result: r.result, fieldWeight });
        } else {
          setErrorMsg(r.error.message);
          setRunState(null);
        }
        setRunning(false);
      });
      return;
    }

    // PT path — in a Web Worker.
    const excluded: number[] = [];
    const excludedSet = new Set(excludedSpecies);
    for (let i = 0; i < model.V; i++) {
      if (excludedSet.has(model.speciesOf[i])) excluded.push(i);
    }
    const t0 = startTimer();
    // Total sweeps include the locked burn-in so the user-displayed
    // "Samples per run" matches what's actually kept post burn-in.
    const seed = seedCounter;
    setSeedCounter(seedCounter + 1);
    const fingerprint: PTInputFingerprint = {
      fixed: [...fixedIdxs],
      excluded: [...excluded],
      fieldWeight,
      temperature,
      ptRuns,
      ptLadder,
      ptSweeps,
      ptSwapInterval,
    };
    runPT(model, {
      fixed: fixedIdxs,
      excluded,
      fieldWeight,
      coldT: temperature,
      hotT: PT_HOT_T,
      ladderLevels: ptLadder,
      nRuns: ptRuns,
      nSteps: ptSweeps + PT_BURN_IN,
      burnIn: PT_BURN_IN,
      swapInterval: ptSwapInterval,
      seed,
    }).then((r) => {
      stopTimer();
      const elapsedFinal = performance.now() - t0;
      setElapsedMs(elapsedFinal);
      if (r.ok) {
        setRunState({
          mode: "pt",
          dist: r.dist,
          nKept: r.nKept,
          localAccept: r.localAccept,
          swapAccept: r.swapAccept,
          fixed: fixedIdxs,
          excluded,
          fieldWeight,
          temperature,
          elapsedMs: elapsedFinal,
          seed,
          fingerprint,
        });
      } else {
        setErrorMsg(r.message);
        setRunState(null);
      }
      setRunning(false);
    });
  };

  return (
    <>
      <PageTitle
        eyebrow="notebook · /completer"
        h1="Team completer"
        rightCaption={corpusCaption}
      />

      <SectionLabel
        num="01"
        title={`Starting roster · ${fixedIdxs.length} of ${TEAM_SIZE} set`}
      />
      <SlotStrip picked={fixedNames} />
      <ExcludedRow names={excludedSpecies} />

      <SectionLabel num="02" title="Constraints" />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 24,
          marginBottom: 16,
        }}
      >
        <div>
          <label className="lab-form-label">Starting Roster</label>
          <VocabSelect
            options={vocabOpts}
            value={fixedIdxs}
            onChange={setFixedIdxs}
            maxSelections={TEAM_SIZE}
            placeholder="Choose Pokemon to include"
          />
        </div>
        <div>
          <label className="lab-form-label">Exclude (must NOT appear)</label>
          <SpeciesSelect
            options={speciesOpts}
            value={excludedSpecies}
            onChange={setExcludedSpecies}
            placeholder="Choose species to exclude"
          />
        </div>
      </div>

      <SectionLabel num="03" title="Sampler" />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 24,
          marginBottom: 12,
        }}
      >
        <div>
          <label className="lab-form-label">
            Bias Adjustment · {fieldWeight.toFixed(1)}
          </label>
          <div className="lab-form-caption">
            Scales the Bias before sampling. 1.0 = popularity bias at
            full strength. 0.0 = pure coherence,
            popularity disregarded. Useful operating range 0.2–0.8.
          </div>
          <input
            type="range"
            className="lab-slider"
            min={0}
            max={FIELD_WEIGHT_OPTIONS.length - 1}
            step={1}
            value={FIELD_WEIGHT_OPTIONS.indexOf(fieldWeight as 0)}
            onChange={(e) =>
              setFieldWeight(FIELD_WEIGHT_OPTIONS[Number(e.target.value)])
            }
          />
        </div>
        <div>
          <label className="lab-form-label">
            Temperature · {temperature}
          </label>
          <div className="lab-form-caption">
            Cold-chain target temperature for the statistical sampler.
            Lower = sharper Boltzmann distribution (fewer, more probable
            completions). Ignored when the full sampler is off. Hot chain
            is fixed at T={PT_HOT_T}.
          </div>
          <input
            type="range"
            className="lab-slider"
            min={0}
            max={TEMPERATURE_OPTIONS.length - 1}
            step={1}
            value={TEMPERATURE_OPTIONS.indexOf(temperature as 0.5)}
            onChange={(e) =>
              setTemperature(TEMPERATURE_OPTIONS[Number(e.target.value)])
            }
            disabled={!usePT}
          />
        </div>
      </div>
      <label className="lab-checkbox-row" style={{ marginBottom: 12 }}>
        <input
          type="checkbox"
          checked={usePT}
          onChange={(e) => setUsePT(e.target.checked)}
        />
        Full statistical sampler (slow)
      </label>
      {usePT && (
        <details className="lab-expander" style={{ marginBottom: 12 }}>
          <summary>Advanced sampler parameters</summary>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "28px 32px",
            }}
          >
            <div>
              <label className="lab-form-label">Runs · {ptRuns}</label>
              <div className="lab-form-caption">
                Independent PT runs. More runs → more samples + better
                distribution coverage. Runtime scales linearly.
              </div>
              <input
                type="range"
                className="lab-slider"
                min={1}
                max={10}
                step={1}
                value={ptRuns}
                onChange={(e) => setPtRuns(Number(e.target.value))}
              />
            </div>
            <div>
              <label className="lab-form-label">
                Ladder levels (K) · {ptLadder}
              </label>
              <div className="lab-form-caption">
                Replica chains from cold T to hot T = {PT_HOT_T}. More
                levels → smaller gaps between rungs → higher replica
                swap acceptance.
              </div>
              <input
                type="range"
                className="lab-slider"
                min={3}
                max={15}
                step={1}
                value={ptLadder}
                onChange={(e) => setPtLadder(Number(e.target.value))}
              />
            </div>
            <div>
              <label className="lab-form-label">
                Samples per run · {ptSweeps.toLocaleString()}
              </label>
              <div className="lab-form-caption">
                Cold-chain samples kept per PT run, post-burn-in. More
                samples → finer-grained distribution + better tail
                resolution. Burn-in ({PT_BURN_IN.toLocaleString()}) is
                discarded automatically before recording starts; runtime
                scales with samples + burn-in.
              </div>
              <input
                type="range"
                className="lab-slider"
                min={1000}
                max={50000}
                step={1000}
                value={ptSweeps}
                onChange={(e) => setPtSweeps(Number(e.target.value))}
              />
            </div>
            <div>
              <label className="lab-form-label">
                Swap interval · {ptSwapInterval}
              </label>
              <div className="lab-form-caption">
                Sweeps between replica-exchange attempts. Lower → more
                exchange attempts → faster cold-chain mixing, but each
                attempt is less likely to accept (smaller energy gap
                accumulated between attempts). Tune watching the Replica
                swap acceptance; healthy band is 20–80%.
              </div>
              <input
                type="range"
                className="lab-slider"
                min={1}
                max={30}
                step={1}
                value={ptSwapInterval}
                onChange={(e) =>
                  setPtSwapInterval(Number(e.target.value))
                }
              />
            </div>
          </div>
        </details>
      )}
      <button
        type="button"
        className="lab-button-primary"
        onClick={onRun}
        disabled={!canRun}
        style={{ width: "100%" }}
      >
        {running
          ? usePT
            ? `Running PT… ${(elapsedMs / 1000).toFixed(1)}s`
            : "Running…"
          : usePT
          ? isPTRerun
            ? "Re-run (new seed)"
            : "Sample"
          : "Suggest team"}
      </button>
      {formError && <div className="lab-form-error">{formError}</div>}
      {errorMsg && <div className="lab-form-error">{errorMsg}</div>}

      {runState?.mode === "fast" && (
        <FastResults runState={runState} model={model} teamCounts={teamCounts} />
      )}
      {runState?.mode === "pt" && (
        <PTResults runState={runState} model={model} teamCounts={teamCounts} />
      )}
    </>
  );
}

function FastResults({
  runState,
  model,
  teamCounts,
}: {
  runState: Extract<RunState, { mode: "fast" }>;
  model: ReturnType<typeof useModel>["model"] & object;
  teamCounts: ReturnType<typeof useModel>["teamCounts"];
}) {
  const { result, fieldWeight } = runState;
  const obs = teamObservables(model, result.finalTeam, fieldWeight);
  const corpus = nearestObserved(result.finalTeam, teamCounts);
  const fixedSet = new Set(result.fixed);
  const freeFinal = result.finalTeam.filter((i) => !fixedSet.has(i));
  return (
    <>
      <SectionLabel num="04" title="Observables" />
      <StatStrip
        cells={[
          {
            label: "Score (adj)",
            value: formatSigned(obs.scoreAdj),
            sub: `Bias Adj. = ${fieldWeight}`,
          },
          {
            label: "Coherence",
            value: formatSigned(obs.coherence),
            sub: "intra-team coupling",
          },
          {
            label: "Swaps taken",
            value: `${result.chain.length} / ${GREEDY_MAX_SWAPS}`,
            sub: "MF fill → local min",
          },
        ]}
      />
      <SectionLabel
        num="05"
        title="Suggested completion"
        right={`mean-field fill → greedy descent · pinned: ${result.fixed.length}`}
      />
      <CompletionTable>
        <CompletionRow
          freeIdxs={freeFinal}
          scoreAdj={obs.scoreAdj}
          scoreRaw={obs.scoreRaw}
          coherence={obs.coherence}
          corpus={corpus}
          isTopRow
          model={model}
        />
      </CompletionTable>
    </>
  );
}

function PTResults({
  runState,
  model,
  teamCounts,
}: {
  runState: Extract<RunState, { mode: "pt" }>;
  model: ReturnType<typeof useModel>["model"] & object;
  teamCounts: ReturnType<typeof useModel>["teamCounts"];
}) {
  const {
    dist, nKept, localAccept, swapAccept, fixed, fieldWeight, elapsedMs,
  } = runState;
  const top5Mass =
    nKept > 0
      ? (dist.slice(0, 5).reduce((s, e) => s + e.count, 0) / nKept) * 100
      : 0;
  const fixedSet = new Set(fixed);
  const topK = dist.slice(0, TOP_COMPLETIONS);
  return (
    <>
      <SectionLabel num="04" title="Observables · last run" />
      <StatStrip
        cells={[
          {
            label: "Cold samples",
            value: nKept.toLocaleString(),
            sub: "kept post burn-in",
          },
          {
            label: "Distinct",
            value: dist.length.toLocaleString(),
            sub: "completions",
          },
          {
            label: "Top-5 mass",
            value: `${top5Mass.toFixed(2)}%`,
            sub: "concentration",
            tooltip:
              "Fraction of all samples in the top-5 completions. " +
              "Healthy: 5%+ (upper bound depends on number of completions). " +
              "Too high → distribution is too steep; lower Temperature. " +
              "Below 5% → distribution is too flat; raise Temperature.",
          },
          {
            label: "Local accept",
            value: `${(localAccept * 100).toFixed(1)}%`,
            sub: "within-chain",
            tooltip:
              "Within-chain Metropolis-Hastings acceptance rate, averaged " +
              "across all replica chains. Healthy: 20–60%. " +
              "Below 15% → chains too cold; raise target Temperature. " +
              "Above 80% → chains too hot; lower target Temperature.",
          },
          {
            label: "Replica swap",
            value: `${(swapAccept * 100).toFixed(1)}%`,
            sub: "between chains",
            tooltip:
              "Acceptance rate for replica exchange between adjacent " +
              "temperature rungs. Healthy: 20–80%. " +
              "Below 15% → rungs are spaced too far apart; increase ladder levels. " +
              "Above 80% → rungs are too close; decrease ladder levels.",
          },
          {
            label: "Runtime",
            value: `${(elapsedMs / 1000).toFixed(1)}s`,
            sub: "wall clock",
          },
        ]}
      />
      <SectionLabel
        num="05"
        title="Top completions"
        right={`ordered by sample frequency · ${Math.min(TOP_COMPLETIONS, dist.length)} of ${dist.length.toLocaleString()} shown`}
      />
      <CompletionTable includeFreq includeRank>
        {topK.map((entry, idx) => {
          const freeIdxs = entry.team.filter((i) => !fixedSet.has(i));
          const obs = teamObservables(model, entry.team, fieldWeight);
          const corpus = nearestObserved(entry.team, teamCounts);
          const freqPct = nKept > 0 ? (entry.count / nKept) * 100 : 0;
          return (
            <CompletionRow
              key={entry.team.join("-")}
              rank={idx + 1}
              freeIdxs={freeIdxs}
              scoreAdj={obs.scoreAdj}
              scoreRaw={obs.scoreRaw}
              coherence={obs.coherence}
              corpus={corpus}
              freqPct={freqPct}
              isTopRow={idx === 0}
              model={model}
            />
          );
        })}
      </CompletionTable>
    </>
  );
}
