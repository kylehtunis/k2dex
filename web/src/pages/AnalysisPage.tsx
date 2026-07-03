// Team analysis page.
//
// Layout mirrors app.py:_render_analysis:
//   PageTitle
//   §01  Team       (slot strip + multiselect, exactly TEAM_SIZE mons)
//   §02  Observables strip (Score adj/raw, Coherence, Corpus)
//   §03  Pairwise coupling decomposition (15 rows)
//   §04  Top single swaps from this team (TOP_SINGLE_SWAPS rows, independent)
//   §05  Greedy critique — single-swap chain
//
// All math is deterministic (no MCMC); recomputes inline when inputs change.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  FIELD_WEIGHT_OPTIONS,
  GREEDY_MAX_SWAPS,
  TEAM_SIZE,
  TOP_SINGLE_SWAPS,
} from "../constants";
import { useModel } from "../state/ModelContext";
import { usePageState, type RosterSlot } from "../state/PageStateContext";
import { PageTitle, SectionLabel, StatStrip } from "../render/atoms";
import { RosterEditor } from "../components/RosterEditor";
import type { IsingModel } from "../sampler/types";
import {
  intraTeamSumJ,
  pairwiseJRows,
  teamObservables,
} from "../render/observables";
import { nearestObserved } from "../render/corpus";
import { rankSingleSwaps } from "../sampler/rank";
import { greedyOptimize } from "../sampler/greedy";
import { buildPartialPaste, formatSigned } from "../render/format";
import { speciesToSlug } from "../render/sprite-url";
import { buildSlugIndex, matchPaste, resolveFeature } from "../render/vocab-match";
import { decodeCore, encodeCore } from "../render/shareLink";
import { PairwiseJTable } from "../analysis/PairwiseJTable";
import { SwapsTable } from "../analysis/SwapsTable";
import { ChainTable } from "../analysis/ChainTable";
import { ScrollX } from "../components/ScrollX";

/** Feature indices → roster slots (each a feature pin). Analysis is
 * feature-level, so every entry carries a concrete item. */
function idxsToRoster(model: IsingModel, idxs: readonly number[]): RosterSlot[] {
  return idxs.map((i) => ({ site: model.siteOf[i], feature: i }));
}

export function AnalysisPage() {
  const { model, teamCounts, status, modelId, setModelId } = useModel();
  const [searchParams, setSearchParams] = useSearchParams();
  const { analysis, setAnalysis } = usePageState();
  const { roster, fieldWeight } = analysis;
  // The complete team = roster slots with an item pinned. Species-only slots
  // are in-progress picks that don't count until an item is chosen.
  const teamIdxs = useMemo(
    () => roster.filter((s) => s.feature !== null).map((s) => s.feature as number),
    [roster],
  );

  // The share token currently reflected in (or being applied from) the URL.
  // Guards the decode/live-sync handshake: decode marks a token applied,
  // live-sync skips writing while an unapplied incoming token is pending.
  const appliedTokenRef = useRef<string | null>(null);

  // Decode a shared link into state. Accepts the slug token (?t=) and a
  // legacy ?team= index list. Switches the model first if the token names a
  // different one, then applies once the matching model is ready.
  useEffect(() => {
    if (status !== "ready" || !model) return;
    const token = searchParams.get("t");
    const legacyTeam = searchParams.get("team");
    const key = token ?? (legacyTeam ? `team:${legacyTeam}` : null);
    if (!key || key === appliedTokenRef.current) return;

    if (token) {
      const decoded = decodeCore(token);
      if (!decoded) {
        appliedTokenRef.current = key;
        return;
      }
      if (decoded.modelId !== modelId) {
        setModelId(decoded.modelId);
        return; // re-run once the new model is ready
      }
      const slugIndex = buildSlugIndex(model);
      const idxs: number[] = [];
      for (const f of decoded.features) {
        const r = resolveFeature(slugIndex, model, f.speciesSlug, f.itemSlug);
        if (r.idx !== null) idxs.push(r.idx);
      }
      appliedTokenRef.current = key;
      setAnalysis({
        roster: idxsToRoster(model, idxs.slice(0, TEAM_SIZE)),
        fieldWeight: decoded.fieldWeight,
      });
      return;
    }

    // Legacy ?team=idx,idx fallback.
    const idxs = legacyTeam!
      .split(",")
      .map(Number)
      .filter((i) => !isNaN(i) && i >= 0 && i < model.V);
    appliedTokenRef.current = key;
    if (idxs.length > 0)
      setAnalysis({ roster: idxsToRoster(model, idxs.slice(0, TEAM_SIZE)) });
  }, [status, model, modelId, searchParams, setAnalysis, setModelId]);

  // Live-sync the URL from state (debounced so a slider drag settles into
  // one write). Skipped while an incoming token is still pending decode.
  const teamKey = useMemo(
    () => [...teamIdxs].sort((a, b) => a - b).join(","),
    [teamIdxs],
  );
  useEffect(() => {
    if (!model || teamIdxs.length === 0) return;
    const incoming = searchParams.get("t");
    if (incoming && incoming !== appliedTokenRef.current) return;
    const token = encodeCore(modelId, fieldWeight, teamIdxs, model);
    if (token === searchParams.get("t")) return;
    const handle = setTimeout(() => {
      appliedTokenRef.current = token;
      setSearchParams({ t: token }, { replace: true });
    }, 300);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, modelId, teamKey, fieldWeight]);

  // Import a pokepaste from the clipboard into the team.
  const [importMsg, setImportMsg] = useState<{
    error: string | null;
    warnings: string[];
  } | null>(null);
  const handleImport = useCallback(async () => {
    if (!model) return;
    try {
      const text = await navigator.clipboard.readText();
      const { idxs, errors, warnings } = matchPaste(model, text);
      if (idxs.length > 0)
        setAnalysis({ roster: idxsToRoster(model, idxs.slice(0, TEAM_SIZE)) });
      setImportMsg({ error: errors[0] ?? null, warnings });
    } catch {
      setImportMsg({ error: "Couldn't read the clipboard.", warnings: [] });
    }
  }, [model, setAnalysis]);

  const teamSorted = useMemo(
    () => [...teamIdxs].sort((a, b) => a - b),
    [teamIdxs],
  );

  // Phase 3 uniqueness validation (inert under Species vocab — species
  // are unique by construction and itemOf is all-null there).
  const uniquenessError = useMemo<string | null>(() => {
    if (!model) return null;
    const seenSp = new Map<string, string>();
    const seenIt = new Map<string, string>();
    for (const i of teamIdxs) {
      const sp = model.speciesOf[i];
      if (seenSp.has(sp)) {
        return `Two entries for species ${sp} (${seenSp.get(sp)}, ${model.vocab[i]}). Pick one variant.`;
      }
      seenSp.set(sp, model.vocab[i]);
      const it = model.itemOf[i];
      if (it !== null) {
        if (seenIt.has(it)) {
          return `Two mons holding ${it} (${seenIt.get(it)}, ${model.vocab[i]}). Items must be unique.`;
        }
        seenIt.set(it, model.vocab[i]);
      }
    }
    return null;
  }, [model, teamIdxs]);

  const teamComplete = teamIdxs.length === TEAM_SIZE;

  const [copied, setCopied] = useState(false);
  const handleCopyPaste = useCallback(() => {
    if (!model || teamIdxs.length === 0) return;
    const paste = buildPartialPaste(teamIdxs, model.vocab, speciesToSlug);
    navigator.clipboard.writeText(paste).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [model, teamIdxs]);

  // Heavy diagnostics — only compute when the team is valid + complete.
  const diagnostics = useMemo(() => {
    if (!model || !teamComplete || uniquenessError) return null;
    const obs = teamObservables(model, teamSorted, fieldWeight);
    const pjRows = pairwiseJRows(teamSorted, model.vocab, model.J, model.V);
    const swaps = rankSingleSwaps(model, {
      team: teamSorted,
      fieldWeight,
      topN: TOP_SINGLE_SWAPS,
    });
    const greedy = greedyOptimize(model, {
      startingTeam: teamSorted,
      pinned: [],
      excluded: [],
      fieldWeight,
      maxSwaps: GREEDY_MAX_SWAPS,
    });
    const finalObs = teamObservables(model, greedy.finalTeam, fieldWeight);
    const finalSumJ = intraTeamSumJ(model.J, model.V, greedy.finalTeam);
    return { obs, pjRows, swaps, greedy, finalObs, finalSumJ };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, teamSorted.join(","), fieldWeight, teamComplete, uniquenessError]);

  const corpus =
    model && teamComplete && !uniquenessError
      ? nearestObserved(teamSorted, teamCounts)
      : null;

  const corpusCaption = model
    ? `Reg ${model.regulation} · ${model.nCorpusTeams.toLocaleString()} teams`
    : undefined;

  return (
    <>
      <PageTitle
        eyebrow="VGC TOOL"
        h1="Team analysis"
        rightCaption={corpusCaption}
      />

      {status === "loading" || model === null ? (
        <p style={{ color: "var(--lab-ink-muted)" }}>Loading model…</p>
      ) : <>

      <SectionLabel
        num="01"
        title={`Team · ${teamIdxs.length} of ${TEAM_SIZE} set`}
        right="pick a Pokémon and item for each of the six slots"
      />
      <RosterEditor
        model={model}
        roster={roster}
        onChange={(next) => setAnalysis({ roster: next })}
        itemActive
        teamSize={TEAM_SIZE}
        itemPlaceholder="item"
        emptyHint={null}
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
          onClick={handleCopyPaste}
          disabled={!teamComplete}
        >
          {copied ? "Copied!" : "Copy pokepaste"}
        </button>
        <button
          type="button"
          className="lab-analyze-btn lab-copy-paste-btn"
          onClick={() => setAnalysis({ roster: [] })}
          disabled={roster.length === 0}
        >
          Clear all
        </button>
      </div>
      {importMsg?.error && (
        <div className="lab-form-error">{importMsg.error}</div>
      )}
      {importMsg?.warnings.map((w, i) => (
        <div className="lab-form-note" key={i}>{w}</div>
      ))}

      {uniquenessError && (
        <div className="lab-form-error">{uniquenessError}</div>
      )}
      {!teamComplete && !uniquenessError && (
        <p style={{ color: "var(--lab-ink-muted)", fontStyle: "italic" }}>
          Pick {TEAM_SIZE} Pokémon to analyze (have {teamIdxs.length}).
        </p>
      )}

      {teamComplete && !uniquenessError && diagnostics && (
        <>
          <div style={{ marginTop: 16, marginBottom: 12 }}>
            <label className="lab-form-label">
              Bias Adjustment · {fieldWeight.toFixed(1)}
            </label>
            <p className="lab-form-caption" style={{ marginBottom: 6 }}>
              Rescales the Bias before computing Score (adj) and choosing
              greedy swaps. Score (raw) always uses Bias Adj. = 1.
            </p>
            <input
              type="range"
              className="lab-slider"
              aria-label="Bias Adjustment"
              min={0}
              max={FIELD_WEIGHT_OPTIONS.length - 1}
              step={1}
              value={FIELD_WEIGHT_OPTIONS.indexOf(fieldWeight as 0)}
              onChange={(e) =>
                setAnalysis({ fieldWeight: FIELD_WEIGHT_OPTIONS[Number(e.target.value)] })
              }
            />
          </div>

          <SectionLabel num="02" title="Observables" />
          <StatStrip
            cells={[
              {
                label: "Score (adj)",
                value: formatSigned(diagnostics.obs.scoreAdj),
                sub: `Bias Adj. = ${fieldWeight}`,
                tooltip:
                  "Hamiltonian-space score sign-flipped so higher = better, " +
                  "with h rescaled by the Bias Adjustment slider.",
              },
              {
                label: "Score (raw)",
                value: formatSigned(diagnostics.obs.scoreRaw),
                sub: "Bias Adj. = 1.0",
                tooltip:
                  "Same as Score (adj) but always at Bias Adj. = 1. The " +
                  "model's data-calibrated team score.",
              },
              {
                label: "Coherence",
                value: formatSigned(diagnostics.obs.coherence),
                sub: "intra-team coupling",
                tooltip:
                  "Σ J_ij over the C(team_size,2) unordered pairs. Positive " +
                  "= synergistic archetype; negative = balance team the " +
                  "pairwise model can't see what makes it work.",
              },
              ...(corpus !== null
                ? [
                    {
                      label: "Corpus",
                      value:
                        corpus.delta === 0
                          ? `${corpus.count}×`
                          : `Δ${corpus.delta} (${corpus.count})`,
                      sub:
                        corpus.delta === 0
                          ? "exact roster seen"
                          : "to nearest observed",
                      tooltip:
                        "Distance to the nearest observed roster in the " +
                        "ingest corpus. Δ0 = the exact team appeared in " +
                        "tournament data; Δk = k swaps away from the " +
                        "nearest realized team.",
                    },
                  ]
                : []),
            ]}
          />

          <SectionLabel
            num="03"
            title="Pairwise coupling decomposition"
            right={`C(${TEAM_SIZE}, 2) = ${diagnostics.pjRows.length} unordered pairs · sorted by |Coupling|`}
          />
          <ScrollX>
            <PairwiseJTable rows={diagnostics.pjRows} />
          </ScrollX>

          <SectionLabel
            num="04"
            title="Top single swaps from this team"
            right="independent one-step alternatives · ranked by ΔScore (adj)"
          />
          {diagnostics.swaps.length === 0 ? (
            <p style={{ color: "var(--lab-ink-muted)", fontStyle: "italic" }}>
              No legal single swap exists from this team.
            </p>
          ) : (
            <SwapsTable
              swaps={diagnostics.swaps}
              teamIdx={teamSorted}
              model={model}
              teamCounts={teamCounts}
              onAcceptSwap={(out, inn) =>
                setAnalysis({
                  roster: roster.map((s) =>
                    s.feature === out
                      ? { site: model.siteOf[inn], feature: inn }
                      : s,
                  ),
                })
              }
            />
          )}

          <SectionLabel
            num="05"
            title="Greedy critique · single-swap chain"
            right={`converged in ${diagnostics.greedy.chain.length} of max ${GREEDY_MAX_SWAPS} swaps`}
          />
          <StatStrip
            cells={[
              {
                label: "Swaps taken",
                value: `${diagnostics.greedy.chain.length} / ${GREEDY_MAX_SWAPS}`,
                sub: "to local min",
              },
              {
                label: "Δ Score (adj)",
                value: formatSigned(
                  diagnostics.finalObs.scoreAdj - diagnostics.obs.scoreAdj,
                ),
                sub: "vs starting",
              },
              {
                label: "Δ Score (raw)",
                value: formatSigned(
                  diagnostics.finalObs.scoreRaw - diagnostics.obs.scoreRaw,
                ),
                sub: "vs starting",
              },
              {
                label: "Δ Coherence",
                value: formatSigned(
                  diagnostics.finalSumJ - diagnostics.obs.coherence,
                ),
                sub: "vs starting",
              },
            ]}
          />
          {diagnostics.greedy.chain.length === 0 ? (
            <p style={{ color: "var(--lab-ink-muted)", fontStyle: "italic" }}>
              No improving single-swap exists — this team is a local maximum
              of Score (adj). Try a different Bias Adjustment to see if the
              model re-ranks under a different objective.
            </p>
          ) : (
            <ChainTable
              startingTeam={teamSorted}
              chain={diagnostics.greedy.chain}
              startScoreAdj={diagnostics.obs.scoreAdj}
              startScoreRaw={diagnostics.obs.scoreRaw}
              startSumJ={diagnostics.obs.coherence}
              model={model}
              teamCounts={teamCounts}
            />
          )}
        </>
      )}
      </>}
    </>
  );
}
export default AnalysisPage;
