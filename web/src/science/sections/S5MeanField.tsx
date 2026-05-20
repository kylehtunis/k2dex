// S5: mean-field iteration animated on the same 6-node graph as S4.
// Overlays exact marginals from enumeration and PT cold-chain.

import { useMemo, useState } from "react";
import { BlockMath, InlineMath } from "../widgets/Math";
import { LinePlot } from "../widgets/LinePlot";
import { Rng } from "../../sampler/rng";
import { exactMarginals, makeGraph } from "../primitives/graph";
import { meanFieldIterate } from "../primitives/mf";
import { runPT } from "../primitives/pt";

function buildGraph(scale: number) {
  return makeGraph(
    6,
    [
      [0, 1, 2.0 * scale],
      [1, 2, 2.0 * scale],
      [3, 4, 2.0 * scale],
      [4, 5, 2.0 * scale],
      [2, 3, -0.5 * scale],
    ],
    [0, 0, 0, 0, 0, 0],
  );
}

const NODE_COLORS = ["#1f4e8c", "#9c2a2a", "#2f7d4f", "#b3791c", "#6a3d9a", "#666"];

export function S5MeanField() {
  const [scale, setScale] = useState(1.0);
  const T = 1.0;

  const { mfHistory, exact, ptCold, maxDelta } = useMemo(() => {
    const g = buildGraph(scale);
    const mf = meanFieldIterate(g, T, { maxIters: 80, tol: 1e-6, record: true });
    const ex = exactMarginals(g, T);
    const ptR = runPT(g, [T, 1.5, 2.5, 4.0], new Rng(7), {
      sweeps: 2000,
      burnIn: 500,
      swapInterval: 5,
    });
    const finalMF = mf.history[mf.history.length - 1] ?? ex;
    const delta = Math.max(...ex.map((e, i) => Math.abs(e - finalMF[i])));
    return { mfHistory: mf.history, exact: ex, ptCold: ptR.coldMarginals, maxDelta: delta };
  }, [scale]);

  const mfSeries = Array.from({ length: 6 }, (_, i) => ({
    data: mfHistory.map((row) => row[i]),
    color: NODE_COLORS[i],
  }));
  const exactSeries = Array.from({ length: 6 }, (_, i) => ({
    data: new Array(mfHistory.length).fill(exact[i]),
    color: NODE_COLORS[i],
    dashed: true,
  }));

  return (
    <section id="s5-mf" className="lab-science-section">
      <h2>5. Mean field: the cheap proxy</h2>
      <p>
        Parallel tempering is the gold standard, but it's expensive — thousands of sweeps
        across <InlineMath formula="K" /> chains, often per query. Mean field gives a much
        cheaper approximation: replace each spin's neighbors with their <em>average</em>{" "}
        values, then iterate until the averages stop changing:
      </p>
      <BlockMath formula="m_i \leftarrow \sigma\!\bigl(\beta(h_i + {\textstyle\sum_j} J_{ij} m_j)\bigr)" />
      <p>
        It converges in a handful of iterations and is fully deterministic. The tradeoff:
        MF ignores correlations beyond what marginals capture, so it under-estimates the
        impact of strong couplings. In practice, MF agrees with PT on the top-1 completion
        in about <strong>85%</strong> of cases — close enough that the{" "}
        <em>/completer</em> page uses MF by default.
      </p>
      <div className="lab-science-controls">
        <label>
          coupling scale = {scale.toFixed(2)}{" "}
          <input
            type="range"
            min={0.2}
            max={2.5}
            step={0.05}
            value={scale}
            onChange={(e) => setScale(Number(e.target.value))}
          />
        </label>
        <span>max |MF − exact| = {maxDelta.toFixed(3)}</span>
      </div>
      <figure>
        <LinePlot
          width={640}
          height={260}
          series={[...exactSeries, ...mfSeries]}
          yDomain={[0, 1]}
          xLabel="MF iteration"
          yLabel="marginal"
        />
        <figcaption>
          Solid: mean-field trajectory per node. Dashed: exact marginal (truth). PT
          cold-chain: [{ptCold.map((v) => v.toFixed(2)).join(", ")}].
        </figcaption>
      </figure>
    </section>
  );
}
