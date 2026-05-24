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

import { useEffect, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import {
  FIELD_WEIGHT_OPTIONS,
  GREEDY_MAX_SWAPS,
  TEAM_SIZE,
  TOP_SINGLE_SWAPS,
} from "../constants";
import { useModel } from "../state/ModelContext";
import { usePageState } from "../state/PageStateContext";
import { PageTitle, SectionLabel, StatStrip } from "../render/atoms";
import { SlotStrip } from "../render/cells";
import { VocabSelect, vocabOptions } from "../components/VocabSelect";
import {
  intraTeamSumJ,
  pairwiseJRows,
  teamObservables,
} from "../render/observables";
import { nearestObserved } from "../render/corpus";
import { rankSingleSwaps } from "../sampler/rank";
import { greedyOptimize } from "../sampler/greedy";
import { formatSigned } from "../render/format";
import { PairwiseJTable } from "../analysis/PairwiseJTable";
import { SwapsTable } from "../analysis/SwapsTable";
import { ChainTable } from "../analysis/ChainTable";

export function AnalysisPage() {
  const { model, teamCounts, status } = useModel();
  const [searchParams, setSearchParams] = useSearchParams();
  const { analysis, setAnalysis } = usePageState();
  const { teamIdxs, fieldWeight } = analysis;

  // Pre-populate from ?team= query param (set by the completer's "Analyze" button).
  useEffect(() => {
    if (status !== "ready" || !model) return;
    const teamParam = searchParams.get("team");
    if (!teamParam) return;
    const idxs = teamParam
      .split(",")
      .map(Number)
      .filter((i) => !isNaN(i) && i >= 0 && i < model.V);
    if (idxs.length > 0) setAnalysis({ teamIdxs: idxs.slice(0, TEAM_SIZE) });
    setSearchParams({}, { replace: true });
  }, [status, model, searchParams, setSearchParams, setAnalysis]);

  const vocabOpts = useMemo(
    () => (model ? vocabOptions(model) : []),
    [model],
  );

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

  if (status === "loading" || model === null) {
    return <p style={{ color: "var(--lab-ink-muted)" }}>Loading model…</p>;
  }

  const corpusCaption =
    `Reg M-A · ${model.nCorpusTeams.toLocaleString()} teams`;
  const teamNames = teamIdxs.map((i) => model.vocab[i]);

  return (
    <>
      <PageTitle
        eyebrow="notebook · /analysis"
        h1="Team analysis"
        rightCaption={corpusCaption}
      />

      <SectionLabel
        num="01"
        title={`Team · ${teamIdxs.length} of ${TEAM_SIZE} set`}
      />
      <SlotStrip picked={teamNames} />
      <div style={{ marginBottom: 16 }}>
        <label className="lab-form-label">Your team (exactly {TEAM_SIZE})</label>
        <VocabSelect
          options={vocabOpts}
          value={teamIdxs}
          onChange={(v) => setAnalysis({ teamIdxs: v })}
          maxSelections={TEAM_SIZE}
          placeholder={`Choose your team of ${TEAM_SIZE}`}
        />
      </div>

      {uniquenessError && (
        <div className="lab-form-error">{uniquenessError}</div>
      )}
      {!teamComplete && !uniquenessError && (
        <p style={{ color: "var(--lab-ink-muted)", fontStyle: "italic" }}>
          Pick {TEAM_SIZE} Pokemon to analyze (have {teamIdxs.length}).
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
          <PairwiseJTable rows={diagnostics.pjRows} />

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
    </>
  );
}
