// Meta data page. Section order diverges from app.py:_render_meta — the
// webapp leads with the empirical top-teams table, then the model's
// extreme couplings / biases:
//   PageTitle
//   §01  Top teams                 (top META_TOP_TEAMS by corpus count)
//        — webapp-only; no Streamlit counterpart
//   §02  Extreme couplings         (top META_TOP_PAIRS, both directions)
//   §03  Extreme features by Bias  (top META_TOP_FEATURES, both directions)

import { useMemo } from "react";
import { META_TOP_FEATURES, META_TOP_PAIRS, META_TOP_TEAMS } from "../constants";
import { useModel } from "../state/ModelContext";
import { PageTitle, SectionLabel } from "../render/atoms";
import { FeatureBiasTable } from "../meta/FeatureBiasTable";
import { ExtremeCouplingsTable } from "../meta/ExtremeCouplingsTable";
import { TopTeamsTable } from "../meta/TopTeamsTable";
import { filteredCouplings } from "../meta/couplings";
import { topTeams } from "../meta/topTeams";

export function MetaPage() {
  const { model, teamCounts, status } = useModel();

  // Top rosters by raw corpus count. Keyed off both the model (for the
  // m̂ member ordering + vocab) and the corpus index.
  const teams = useMemo(() => {
    if (!model || !teamCounts) return null;
    const rows = topTeams(teamCounts, META_TOP_TEAMS, model.m);
    const maxCount = rows.length > 0 ? rows[0].count : 1;
    return { rows, maxCount };
  }, [model, teamCounts]);

  // Sort orders — only depend on h / J, so memoize against the model identity.
  const sorted = useMemo(() => {
    if (!model) return null;
    const V = model.V;
    const orderDesc = Array.from({ length: V }, (_, i) => i);
    orderDesc.sort((a, b) => model.h[b] - model.h[a]);
    const orderAsc = [...orderDesc].reverse();

    const couplings = filteredCouplings(model);
    const posSorted = [...couplings].sort((a, b) => b.jValue - a.jValue);
    const negSorted = [...couplings].sort((a, b) => a.jValue - b.jValue);

    // Max |J| over filtered set — used to calibrate the SignedBar in
    // both ±Coupling tables.
    let maxJ = 0;
    for (const p of couplings) {
      const a = Math.abs(p.jValue);
      if (a > maxJ) maxJ = a;
    }
    if (maxJ === 0) maxJ = 1;

    let maxM = 0;
    for (let i = 0; i < V; i++) if (model.m[i] > maxM) maxM = model.m[i];

    return { orderDesc, orderAsc, posSorted, negSorted, maxJ, maxM };
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
          />
        </>
      )}

      <SectionLabel
        num="02"
        title="Extreme couplings"
        right={`top ${META_TOP_PAIRS} each direction · ranked by Coupling`}
      />
      <div className="lab-split-pair">
        <div>
          <div className="lab-subheading lab-subheading-pos">
            Top Positive Coupling · synergies
          </div>
          <ExtremeCouplingsTable
            rows={sorted.posSorted.slice(0, META_TOP_PAIRS)}
            maxJ={sorted.maxJ}
            model={model}
          />
        </div>
        <div>
          <div className="lab-subheading lab-subheading-neg">
            Top Negative Coupling · antisynergies
          </div>
          <ExtremeCouplingsTable
            rows={sorted.negSorted.slice(0, META_TOP_PAIRS)}
            maxJ={sorted.maxJ}
            model={model}
          />
        </div>
      </div>

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
