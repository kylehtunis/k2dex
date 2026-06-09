// Meta data page.
//
// Layout mirrors app.py:_render_meta:
//   PageTitle
//   §01  Fitted model stat strip
//   §02  Extreme features by Bias  (top META_TOP_FEATURES, both directions)
//   §03  Extreme couplings         (top META_TOP_PAIRS, both directions)
//   §04  Distributional diagnostics (J histogram + h line plot)

import { useMemo } from "react";
import { META_TOP_FEATURES, META_TOP_PAIRS } from "../constants";
import { useModel } from "../state/ModelContext";
import { PageTitle, SectionLabel, StatStrip } from "../render/atoms";
import { FeatureBiasTable } from "../meta/FeatureBiasTable";
import { ExtremeCouplingsTable } from "../meta/ExtremeCouplingsTable";
import { filteredCouplings } from "../meta/couplings";

export function MetaPage() {
  const { model, status } = useModel();

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

      <SectionLabel num="01" title="Fitted model" />
      <StatStrip
        cells={[
          { label: "Model", value: model.displayName, sub: model.featureDimensions === 1 ? "PL · species" : "PL · item-pair" },
          {
            label: "Vocab",
            value: model.V.toLocaleString(),
            sub: "unique entries",
          },
          {
            label: "Corpus",
            value: model.nCorpusTeams.toLocaleString(),
            sub: "teams observed",
          },
        ]}
      />

      <SectionLabel
        num="02"
        title="Extreme features by Bias"
        right={`top ${META_TOP_FEATURES} each direction · ranked by Bias`}
      />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 24,
          marginBottom: 12,
        }}
      >
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

      <SectionLabel
        num="03"
        title="Extreme couplings"
        right={`top ${META_TOP_PAIRS} each direction · ranked by Coupling`}
      />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 24,
          marginBottom: 12,
        }}
      >
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
      </>}
    </>
  );
}
export default MetaPage;
