// Meta data page:
//   PageTitle
//   §01  Top teams                 (top META_TOP_TEAMS by corpus count)
//   §02  Extreme couplings         (top META_TOP_PAIRS species pairs by signed synergy)

import { useMemo } from "react";
import { META_TOP_PAIRS, META_TOP_TEAMS } from "../constants";
import { useModel } from "../state/ModelContext";
import { PageTitle, SectionLabel } from "../render/atoms";
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
      all.push({ a: i, b: j, synergy: graph.synergy[i * S + j] });
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

      {status === "loading" || model === null ? (
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
            right={`top ${META_TOP_PAIRS} each direction · ranked by species synergy`}
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
      </>}
    </>
  );
}
export default MetaPage;
