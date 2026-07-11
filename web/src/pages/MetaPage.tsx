// Meta data page. Section order diverges from app.py:_render_meta — the
// webapp leads with the empirical top-teams table, then the model's
// extreme couplings / biases:
//   PageTitle
//   §01  Top teams                 (top META_TOP_TEAMS by corpus count)
//        — webapp-only; no Streamlit counterpart
//   §02  Extreme couplings         (top META_TOP_PAIRS species pairs by APC-corrected synergy)
//   §03  Extreme features by Bias  (top META_TOP_FEATURES, both directions)

import { useMemo } from "react";
import { META_TOP_FEATURES, META_TOP_PAIRS, META_TOP_TEAMS } from "../constants";
import { useModel } from "../state/ModelContext";
import { PageTitle, SectionLabel } from "../render/atoms";
import { FeatureBiasTable } from "../meta/FeatureBiasTable";
import {
  SpeciesCouplingsTable,
  type SpeciesCouplingRow,
} from "../meta/ExtremeCouplingsTable";
import { TopTeamsTable } from "../meta/TopTeamsTable";
import { topTeams } from "../meta/topTeams";
import type { SpeciesGraph } from "../sampler/types";

function buildSpeciesCouplingRows(
  graph: SpeciesGraph,
): { posSorted: SpeciesCouplingRow[]; negSorted: SpeciesCouplingRow[]; maxSynergy: number } {
  const S = graph.species.length;
  const all: SpeciesCouplingRow[] = [];
  for (let i = 0; i < S; i++) {
    for (let j = i + 1; j < S; j++) {
      const synergy = graph.synergy[i * S + j];
      const corrected = graph.corrected[i * S + j];
      all.push({ a: i, b: j, synergy, corrected });
    }
  }
  let maxSynergy = 0;
  for (const r of all) {
    const a = Math.abs(r.synergy);
    if (a > maxSynergy) maxSynergy = a;
  }
  if (maxSynergy === 0) maxSynergy = 1;

  const posSorted = all
    .filter((r) => r.synergy > 0)
    .sort((a, b) => b.synergy - a.synergy);
  const negSorted = all
    .filter((r) => r.synergy < 0)
    .sort((a, b) => a.synergy - b.synergy);

  return { posSorted, negSorted, maxSynergy };
}

export function MetaPage() {
  const { model, teamCounts, speciesGraph, corpusScoreIndex, status } = useModel();

  const teams = useMemo(() => {
    if (!model || !teamCounts) return null;
    const rows = topTeams(teamCounts, META_TOP_TEAMS, model.m);
    const maxCount = rows.length > 0 ? rows[0].count : 1;
    return { rows, maxCount };
  }, [model, teamCounts]);

  const speciesCouplings = useMemo(() => {
    if (!speciesGraph) return null;
    return buildSpeciesCouplingRows(speciesGraph);
  }, [speciesGraph]);

  const sorted = useMemo(() => {
    if (!model) return null;
    const V = model.V;
    const orderDesc = Array.from({ length: V }, (_, i) => i);
    orderDesc.sort((a, b) => model.h[b] - model.h[a]);
    const orderAsc = [...orderDesc].reverse();

    let maxM = 0;
    for (let i = 0; i < V; i++) if (model.m[i] > maxM) maxM = model.m[i];

    return { orderDesc, orderAsc, maxM };
  }, [model]);

  const corpusCaption = model
    ? `Reg ${model.regulation} · ${model.nCorpusTeams.toLocaleString()} teams`
    : undefined;

  return (
    <>
      <PageTitle
        eyebrow="VGC Tool"
        h1="Metagame Model"
        rightCaption={corpusCaption}
      />

      {status === "loading" || model === null || sorted === null ? (
        <p style={{ color: "var(--lab-ink-muted)" }}>Loading model…</p>
      ) : <>

      {teams && teams.rows.length > 0 && (
        <>
          <SectionLabel
            num="01"
            title="Top teams"
            right={`top ${META_TOP_TEAMS} · ranked by corpus count`}
          />
          <TopTeamsTable
            rows={teams.rows}
            maxCount={teams.maxCount}
            nCorpusTeams={model.nCorpusTeams}
            model={model}
            scoreIndex={corpusScoreIndex}
          />
        </>
      )}

      {speciesCouplings && speciesGraph && (
        <>
          <SectionLabel
            num="02"
            title="Extreme couplings"
            right={`top ${META_TOP_PAIRS} each direction · ranked by species synergy (APC-corrected)`}
          />
          <div className="lab-split-pair">
            <div>
              <div className="lab-subheading lab-subheading-pos">
                Top Positive Synergy
              </div>
              <SpeciesCouplingsTable
                rows={speciesCouplings.posSorted.slice(0, META_TOP_PAIRS)}
                maxSynergy={speciesCouplings.maxSynergy}
                graph={speciesGraph}
                model={model}
              />
            </div>
            <div>
              <div className="lab-subheading lab-subheading-neg">
                Top Negative Synergy
              </div>
              <SpeciesCouplingsTable
                rows={speciesCouplings.negSorted.slice(0, META_TOP_PAIRS)}
                maxSynergy={speciesCouplings.maxSynergy}
                graph={speciesGraph}
                model={model}
              />
            </div>
          </div>
        </>
      )}

      <SectionLabel
        num="03"
        title="Extreme features by Bias"
        right={`top ${META_TOP_FEATURES} each direction · ranked by Bias`}
      />
      <div className="lab-split-pair">
        <div>
          <div className="lab-subheading lab-subheading-pos">
            Top Positive Bias · most popular
          </div>
          <FeatureBiasTable
            order={sorted.orderDesc.slice(0, META_TOP_FEATURES)}
            maxM={sorted.maxM}
            model={model}
          />
        </div>
        <div>
          <div className="lab-subheading lab-subheading-neg">
            Top Negative Bias · most unlikely
          </div>
          <FeatureBiasTable
            order={sorted.orderAsc.slice(0, META_TOP_FEATURES)}
            maxM={sorted.maxM}
            model={model}
          />
        </div>
      </div>
      </>}
    </>
  );
}
export default MetaPage;
