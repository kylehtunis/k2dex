// Team completer page.
//
// Layout mirrors app.py:_render_completer:
//   PageTitle  — eyebrow + h1 + corpus caption
//   §01        — Starting roster (slot strip)
//   §02        — Excluded row (when non-empty)
//   §03        — Constraints (pin + exclude multiselects)
//   §04        — Sampler (greedy toggle + sliders + run button)
//   §05        — Observables (post-run)
//   §06        — Suggested completion table (post-run)
//
// Full statistical sampler (default): PT MCMC in a Web Worker. Returns a
// distribution of completions; PT options (temperature + advanced) are
// shown.
//
// Greedy sampling (toggle on): MF marginals → uniqueness-respecting
// greedy fill → greedy descent. Returns one team, fast, but can descend
// into incoherent basins at low Bias Adjustment with few pins — so
// checking it bumps Bias Adjustment to 0.8 and hides the PT options.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  FIELD_WEIGHT_OPTIONS,
  GREEDY_MAX_SWAPS,
  PT_BURN_IN,
  PT_HOT_T,
  TEAM_SIZE,
  TEMPERATURE_OPTIONS,
  TOP_COMPLETIONS,
} from "../constants";
import { useModel } from "../state/ModelContext";
import { usePageState, type RosterSlot } from "../state/PageStateContext";
import { PageTitle, SectionLabel, StatStrip } from "../render/atoms";
import { ExcludedRow } from "../render/cells";
import { SpeciesSelect, speciesOptions } from "../components/VocabSelect";
import { RosterEditor } from "../components/RosterEditor";
import { runFastPath, type FastPathResult } from "../completer/fastPath";
import { withInactiveTracks } from "../sampler/model";
import { teamObservables } from "../render/observables";
import { nearestObserved } from "../render/corpus";
import {
  CompletionCard,
  CompletionList,
} from "../completer/CompletionCard";
import { formatSigned } from "../render/format";
import {
  buildSlugIndex,
  matchPaste,
  resolveFeature,
  resolveSpeciesSlug,
} from "../render/vocab-match";
import { decodeCompleter, encodeCompleter } from "../render/shareLink";
import { runPT, type PTDistEntry } from "../completer/ptDriver";

/** Fingerprint of the inputs that produced a run. Used to decide
 * whether the next button click is a fresh "Sample" (inputs changed)
 * or a "Re-run" with a new seed (inputs unchanged). */
interface PTInputFingerprint {
  fixed: readonly number[];
  fixedSites: readonly number[];
  speciesOnly: boolean;
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
      hideItems: boolean;
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
      hideItems: boolean;
      fingerprint: PTInputFingerprint;
    };

export function CompleterPage() {
  const { model, teamCounts, status, modelId, setModelId } = useModel();
  const [searchParams, setSearchParams] = useSearchParams();
  const { completer, setCompleter } = usePageState();

  const {
    roster, inactiveTracks, excludedSpecies, fieldWeight,
    temperature, usePT, ptRuns, ptLadder, ptSweeps, ptSwapInterval,
  } = completer;
  const setRoster = (next: RosterSlot[]) => setCompleter({ roster: next });
  // Reset every input on the page back to an empty query.
  const clearAll = () =>
    setCompleter({ roster: [], excludedSpecies: [], inactiveTracks: [] });

  // Pin arrays derived from the ordered roster for the sampler + share links.
  // Feature pins (item chosen) vs site pins (item left to the completer).
  const fixedIdxs = roster.filter((s) => s.feature !== null).map((s) => s.feature as number);
  const fixedSites = roster.filter((s) => s.feature === null).map((s) => s.site);

  // Attribute toggle. Today the only track is "item"; deactivating it drives
  // species-only mode (marginalize + hide the item, no reroll, no uniqueness).
  const itemTrackIdx = model ? model.tracks.findIndex((t) => t.name === "item") : -1;
  const speciesOnly = itemTrackIdx >= 0 && inactiveTracks.includes(itemTrackIdx);
  const effectiveModel = useMemo(
    () => (model ? withInactiveTracks(model, inactiveTracks) : null),
    [model, inactiveTracks],
  );

  // Toggle a track. Deactivating clears every slot's item (it's now hidden and
  // filled by the completer), turning feature pins into species-only pins.
  const toggleTrack = (ti: number, active: boolean) => {
    if (!model) return;
    if (active) {
      setCompleter({ inactiveTracks: inactiveTracks.filter((t) => t !== ti) });
      return;
    }
    setCompleter({
      inactiveTracks: [...inactiveTracks.filter((t) => t !== ti), ti],
      roster: roster.map((s) => ({ site: s.site, feature: null })),
    });
  };

  const currentModelId = model?.id ?? "—";

  // Identity of the URL state currently applied / being written, so the
  // decode and live-sync effects don't clobber each other.
  const appliedRef = useRef<string | null>(null);

  // Decode a shared link (or the legacy ?pinned= from /science) into state.
  // Switches the model first if the token names a different one, then
  // applies once the matching model is ready.
  useEffect(() => {
    if (status !== "ready" || !model) return;
    const identity = searchParams.toString();
    if (!identity || identity === appliedRef.current) return;

    if (searchParams.get("t")) {
      const d = decodeCompleter(searchParams);
      if (!d) {
        appliedRef.current = identity;
        return;
      }
      if (d.modelId !== modelId) {
        setModelId(d.modelId);
        return; // re-run once the new model is ready
      }
      const slugIndex = buildSlugIndex(model);
      const newRoster: RosterSlot[] = [];
      for (const f of d.features) {
        if (f.itemSlug === null) {
          // Bare mon = site-level pin: resolve the species to a site index.
          const name = resolveSpeciesSlug(slugIndex, model, f.speciesSlug);
          if (name) {
            const site = model.sites.indexOf(name);
            if (site >= 0) newRoster.push({ site, feature: null });
          }
        } else {
          const r = resolveFeature(slugIndex, model, f.speciesSlug, f.itemSlug);
          if (r.idx !== null) newRoster.push({ site: model.siteOf[r.idx], feature: r.idx });
        }
      }
      const excluded: string[] = [];
      for (const slug of d.excludedSlugs) {
        const name = resolveSpeciesSlug(slugIndex, model, slug);
        if (name) excluded.push(name);
      }
      appliedRef.current = identity;
      const inactive = d.inactiveTracks.filter((t) => t >= 0 && t < model.tracks.length);
      setCompleter({
        roster: newRoster.slice(0, TEAM_SIZE),
        inactiveTracks: inactive,
        excludedSpecies: excluded,
        fieldWeight: d.fieldWeight,
        usePT: d.usePT,
        temperature: d.temperature,
        ptRuns: d.ptRuns,
        ptLadder: d.ptLadder,
        ptSweeps: d.ptSweeps,
        ptSwapInterval: d.ptSwapInterval,
      });
      if (d.seed !== null) {
        setSeedCounter(d.seed);
        setRestoredSeed(d.seed);
      }
      return;
    }

    // Legacy ?pinned= (set by the /science page): site-pin the species
    // (species locked, item free — the natural "I want this Pokemon" pin).
    const pinned = searchParams.get("pinned");
    if (!pinned) return;
    const site = model.sites.indexOf(pinned);
    appliedRef.current = identity;
    if (site >= 0) setCompleter({ roster: [{ site, feature: null }] });
  }, [status, model, modelId, searchParams, setCompleter, setModelId]);

  // Ephemeral state — not persisted across tab switches.
  const [seedCounter, setSeedCounter] = useState(1);
  // A seed restored from a shared link but not yet consumed by a run.
  // Keeps the URL advertising the reproducible seed until the user Samples.
  const [restoredSeed, setRestoredSeed] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const [runState, setRunState] = useState<RunState | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const elapsedTimer = useRef<number | null>(null);

  // Clear per-model run state when the active model changes. Done during
  // render (not in an effect) so it can't clobber the decode effect that
  // restores a shared link's seed: useEffect runs after render, and this
  // reset is declared after decode, so as an effect it would fire second
  // and wipe the just-restored seed. The "—" placeholder isn't a loaded
  // model, so we only reset when leaving one that was already loaded.
  const prevRunPhase = useRef(currentModelId);
  if (prevRunPhase.current !== currentModelId) {
    const leavingLoadedModel = prevRunPhase.current !== "—";
    prevRunPhase.current = currentModelId;
    if (leavingLoadedModel) {
      setRunState(null);
      setErrorMsg(null);
      setSeedCounter(1);
      setRestoredSeed(null);
    }
  }

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
  }, [currentModelId]);

  const corpusCaption = model
    ? `Reg ${model.regulation} · ${model.nCorpusTeams.toLocaleString()} teams`
    : undefined;

  const totalPins = roster.length;
  const fixedSpeciesSet = model
    ? new Set(roster.map((s) => model.sites[s.site]))
    : new Set<string>();
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
    if (fp.speciesOnly !== speciesOnly) return false;
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
    if (fp.fixedSites.length !== fixedSites.length) return false;
    for (let i = 0; i < fp.fixedSites.length; i++) {
      if (fp.fixedSites[i] !== fixedSites[i]) return false;
    }
    if (fp.excluded.length !== excludedNow.length) return false;
    for (let i = 0; i < fp.excluded.length; i++) {
      if (fp.excluded[i] !== excludedNow[i]) return false;
    }
    return true;
  })();

  // Seed to advertise in the URL: the displayed PT run's seed while inputs
  // still match it, otherwise a seed restored from a link but not yet run.
  const seedForUrl = !usePT
    ? null
    : runState?.mode === "pt" && isPTRerun
      ? runState.seed
      : restoredSeed;

  // Live-sync the URL from the form (debounced so slider drags settle into
  // one write). Skipped while an incoming link is still pending decode and
  // when there's nothing worth sharing (empty roster + no excludes).
  const fixedKey = fixedIdxs.join(",");
  const fixedSitesKey = fixedSites.join(",");
  const inactiveKey = inactiveTracks.join(",");
  const excludedKey = [...excludedSpecies].sort().join(",");
  const shareParams = useMemo(() => {
    if (!model) return null;
    if (totalPins === 0 && excludedSpecies.length === 0) return null;
    return encodeCompleter(
      {
        modelId,
        fieldWeight,
        fixedIdxs,
        fixedSites,
        inactiveTracks,
        excludedSpecies,
        usePT,
        temperature,
        ptRuns,
        ptLadder,
        ptSweeps,
        ptSwapInterval,
        seed: seedForUrl,
      },
      model,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    model, modelId, fieldWeight, fixedKey, fixedSitesKey, inactiveKey,
    excludedKey, usePT, temperature, ptRuns, ptLadder, ptSweeps,
    ptSwapInterval, seedForUrl,
  ]);
  useEffect(() => {
    if (!shareParams) return;
    if (searchParams.get("t") && searchParams.toString() !== appliedRef.current) {
      return; // incoming link not yet decoded
    }
    const next = shareParams.toString();
    if (next === searchParams.toString()) return;
    const handle = setTimeout(() => {
      appliedRef.current = next;
      setSearchParams(shareParams, { replace: true });
    }, 300);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shareParams]);

  // Import a pokepaste from the clipboard into the starting roster.
  const [importMsg, setImportMsg] = useState<{
    error: string | null;
    warnings: string[];
  } | null>(null);
  const handleImport = useCallback(async () => {
    if (!model) return;
    try {
      const text = await navigator.clipboard.readText();
      const { idxs, errors, warnings } = matchPaste(model, text);
      // A paste is a concrete roster of feature pins (species + item).
      if (idxs.length > 0) {
        setRoster(
          idxs.slice(0, TEAM_SIZE).map((i) => ({ site: model.siteOf[i], feature: i })),
        );
      }
      setImportMsg({ error: errors[0] ?? null, warnings });
    } catch {
      setImportMsg({ error: "Couldn't read the clipboard.", warnings: [] });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

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
    if (!canRun || !model || !effectiveModel) return;
    setErrorMsg(null);
    setRunning(true);
    // The displayed run now governs the URL seed; drop any armed link seed.
    setRestoredSeed(null);

    if (!usePT) {
      // Fast path — synchronous, ~50–200ms. queueMicrotask lets the
      // "Running…" state render once before the loop blocks.
      queueMicrotask(() => {
        const r = runFastPath(effectiveModel, {
          fixed: fixedIdxs,
          fixedSites,
          excludedSpecies,
          fieldWeight,
        });
        if (r.ok) {
          setRunState({ mode: "fast", result: r.result, fieldWeight, hideItems: speciesOnly });
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
      fixedSites: [...fixedSites],
      speciesOnly,
      excluded: [...excluded],
      fieldWeight,
      temperature,
      ptRuns,
      ptLadder,
      ptSweeps,
      ptSwapInterval,
    };
    runPT(effectiveModel, {
      fixed: fixedIdxs,
      fixedSites,
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
      // Species-only: don't reroll the item, marginalize it into species sets.
      pReroll: speciesOnly ? 0 : undefined,
      projectToSites: speciesOnly,
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
          hideItems: speciesOnly,
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
        eyebrow="VGC Tool"
        h1="Team completer"
        rightCaption={corpusCaption}
      />

      {status === "loading" || model === null ? (
        <p style={{ color: "var(--lab-ink-muted)" }}>Loading model…</p>
      ) : <>

      <SectionLabel
        num="01"
        title={`Starting roster · ${totalPins} of ${TEAM_SIZE} set`}
        right="pick a Pokémon per slot; leave the item unset to let the completer fill it"
      />
      <RosterEditor
        model={model}
        roster={roster}
        onChange={setRoster}
        itemActive={!speciesOnly}
        teamSize={TEAM_SIZE}
      />
      <div style={{ marginBottom: 12, display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button
          type="button"
          className="lab-analyze-btn lab-copy-paste-btn"
          onClick={handleImport}
        >
          Import from clipboard
        </button>
        <button
          type="button"
          className="lab-analyze-btn lab-copy-paste-btn"
          onClick={clearAll}
          disabled={roster.length === 0 && excludedSpecies.length === 0 && inactiveTracks.length === 0}
        >
          Clear all
        </button>
        {importMsg?.error && (
          <div className="lab-form-error">{importMsg.error}</div>
        )}
        {importMsg?.warnings.map((w, i) => (
          <div className="lab-form-note" key={i}>{w}</div>
        ))}
      </div>
      <ExcludedRow names={excludedSpecies} />

      <SectionLabel num="02" title="Constraints" />
      <div style={{ marginBottom: 16 }}>
        <label className="lab-form-label">Exclude (must NOT appear)</label>
        <SpeciesSelect
          options={speciesOpts}
          value={excludedSpecies}
          onChange={(v) => setCompleter({ excludedSpecies: v })}
          placeholder="Choose Pokémon to exclude"
          ariaLabel="Exclude Pokémon"
        />
      </div>

      {model.tracks.length > 0 && (
        <details className="lab-expander" style={{ marginBottom: 16 }}>
          <summary>Advanced options</summary>
          <div style={{ marginTop: 8 }}>
            <label className="lab-form-label">Excluded attributes</label>
            <div className="lab-form-caption">
              All attributes are active by default. Exclude one to build by
              Pokémon only: the model still accounts for it under the hood, but
              it's marginalized out and hidden from the results.
            </div>
            <div className="lab-attr-toggles">
              {model.tracks.map((t, ti) => (
                <label key={t.name} className="lab-checkbox-row">
                  <input
                    type="checkbox"
                    checked={inactiveTracks.includes(ti)}
                    onChange={(e) => toggleTrack(ti, !e.target.checked)}
                  />
                  {t.name.charAt(0).toUpperCase() + t.name.slice(1)}
                </label>
              ))}
            </div>
          </div>
        </details>
      )}

      <SectionLabel num="03" title="Sampler" />
      <div
        className={usePT ? "lab-form-grid" : "lab-form-grid lab-form-grid-1"}
        style={{ marginBottom: 12 }}
      >
        <div>
          <label className="lab-form-label">
            Bias Adjustment · {fieldWeight.toFixed(1)}
          </label>
          <div className="lab-form-caption">
            Scales the Bias before sampling. 1.0 = popularity bias at
            full strength. 0.0 = pure coherence,
            popularity disregarded. Useful operating range 0.2–0.8. The higher this 
            value, the more the model will prioritize the most commonly used Pokémon.
          </div>
          <input
            type="range"
            className="lab-slider"
            aria-label="Bias Adjustment"
            min={0}
            max={FIELD_WEIGHT_OPTIONS.length - 1}
            step={1}
            value={FIELD_WEIGHT_OPTIONS.indexOf(fieldWeight as 0)}
            onChange={(e) =>
              setCompleter({ fieldWeight: FIELD_WEIGHT_OPTIONS[Number(e.target.value)] })
            }
          />
        </div>
        {usePT && (
          <div>
            <label className="lab-form-label">
              Temperature · {temperature}
            </label>
            <div className="lab-form-caption">
              Controls the tradeoff between exploring more teams (higher) and concentrating
              on the most likely teams (lower). The range 0.3-0.7 is generally reasonable. Raise if you
              aren't getting enough unique teams and lower if the top 5 mass drops below ~5-10%.
            </div>
            <input
              type="range"
              className="lab-slider"
              aria-label="Temperature"
              min={0}
              max={TEMPERATURE_OPTIONS.length - 1}
              step={1}
              value={TEMPERATURE_OPTIONS.indexOf(temperature as 0.5)}
              onChange={(e) =>
                setCompleter({ temperature: TEMPERATURE_OPTIONS[Number(e.target.value)] })
              }
            />
          </div>
        )}
      </div>
      <label className="lab-checkbox-row" style={{ marginBottom: 12 }}>
        <input
          type="checkbox"
          checked={!usePT}
          onChange={(e) => {
            const greedy = e.target.checked;
            // Greedy descends into incoherent basins at low Bias Adjustment
            // with few pins, so seed it high (0.8); the full sampler is fine
            // at the discrimination-optimal 0.3 (energy_discrimination.ipynb).
            setCompleter({ usePT: !greedy, fieldWeight: greedy ? 0.8 : 0.3 });
          }}
        />
        Greedy sampling (faster, only one result per query)
      </label>
      {usePT && (
        <details className="lab-expander" style={{ marginBottom: 12 }}>
          <summary>Advanced sampler parameters</summary>
          <div className="lab-form-grid lab-form-grid-airy">
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
                max={25}
                step={1}
                value={ptRuns}
                onChange={(e) => setCompleter({ ptRuns: Number(e.target.value) })}
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
                onChange={(e) => setCompleter({ ptLadder: Number(e.target.value) })}
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
                onChange={(e) => setCompleter({ ptSweeps: Number(e.target.value) })}
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
                  setCompleter({ ptSwapInterval: Number(e.target.value) })
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
      </>}
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
      <CompletionList>
        <CompletionCard
          freeIdxs={freeFinal}
          fullTeam={result.finalTeam}
          scoreAdj={obs.scoreAdj}
          scoreRaw={obs.scoreRaw}
          coherence={obs.coherence}
          corpus={corpus}
          isTopRow
          model={model}
          hideItems={runState.hideItems}
        />
      </CompletionList>
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
      <CompletionList>
        {topK.map((entry, idx) => {
          const freeIdxs = entry.team.filter((i) => !fixedSet.has(i));
          const obs = teamObservables(model, entry.team, fieldWeight);
          const corpus = nearestObserved(entry.team, teamCounts);
          const freqPct = nKept > 0 ? (entry.count / nKept) * 100 : 0;
          return (
            <CompletionCard
              key={entry.team.join("-")}
              rank={idx + 1}
              freeIdxs={freeIdxs}
              fullTeam={entry.team}
              scoreAdj={obs.scoreAdj}
              scoreRaw={obs.scoreRaw}
              coherence={obs.coherence}
              corpus={corpus}
              freqPct={freqPct}
              isTopRow={idx === 0}
              model={model}
              hideItems={runState.hideItems}
            />
          );
        })}
      </CompletionList>
    </>
  );
}
export default CompleterPage;
