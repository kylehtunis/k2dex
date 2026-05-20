// S4: side-by-side single-chain vs PT on a bimodal 6-node graph.
// Cold-T slider drives both. Right panel shows replica-exchange swap markers.

import { useMemo, useState } from "react";
import { BlockMath, InlineMath } from "../widgets/Math";
import { ChainStrip } from "../widgets/ChainStrip";
import { GraphView } from "../widgets/GraphView";
import { Rng } from "../../sampler/rng";
import { exactMarginals, makeGraph } from "../primitives/graph";
import { runChain } from "../primitives/mcmc";
import { runPT } from "../primitives/pt";

const GRAPH = makeGraph(
  6,
  [
    [0, 1, 2.0],
    [1, 2, 2.0],
    [3, 4, 2.0],
    [4, 5, 2.0],
    [2, 3, -0.5],
  ],
  [0, 0, 0, 0, 0, 0],
);

const NODE_POS = [
  { x: 60, y: 60 },
  { x: 60, y: 140 },
  { x: 60, y: 220 },
  { x: 260, y: 60 },
  { x: 260, y: 140 },
  { x: 260, y: 220 },
];

function marginalFill(m: number): string {
  const light = 232;
  const dark = 26;
  const v = Math.round(light + (dark - light) * m);
  return `rgb(${v},${v},${Math.round(223 + (26 - 223) * m)})`;
}

export function S4PT() {
  const [Tcold, setTcold] = useState(0.7);
  const [seed, setSeed] = useState(1);

  const { single, pt, exact } = useMemo(() => {
    const ladder = [Tcold, 1.0, 1.5, 2.5];
    const r1 = new Rng(seed);
    const single = runChain(GRAPH, Tcold, r1, { sweeps: 2000, burnIn: 500 });
    const r2 = new Rng(seed + 1000);
    const pt = runPT(GRAPH, ladder, r2, {
      sweeps: 2000,
      burnIn: 500,
      swapInterval: 5,
      record: true,
    });
    const exact = exactMarginals(GRAPH, Tcold);
    return { single, pt, exact };
  }, [Tcold, seed]);

  const makeNodes = (marginals: number[]) =>
    NODE_POS.map((p, i) => ({
      id: i,
      label: String(i),
      x: p.x,
      y: p.y,
      fill: marginalFill(marginals[i]),
    }));

  const graphEdges = GRAPH.edges.map(([i, j, w]) => ({ i, j, weight: w }));

  return (
    <section id="s4-pt" className="lab-science-section">
      <h2>4. Parallel tempering: escaping basins at low T</h2>
      <p>
        At low temperatures the Metropolis chain can get stuck in a local energy minimum:
        it samples that basin densely but never visits the others. The frequencies you see
        reflect <em>where the chain started</em>, not the true Boltzmann distribution.
      </p>
      <p>
        Parallel tempering runs <InlineMath formula="K" /> chains at temperatures{" "}
        <InlineMath formula="T_1 < T_2 < \dots < T_K" />. Hot chains roam freely; cold
        chains concentrate. Periodically the algorithm proposes a swap between adjacent
        rungs:
      </p>
      <BlockMath formula="p_\text{swap} = \min\bigl(1,\ e^{(\beta_k - \beta_{k+1})(H_{k+1} - H_k)}\bigr)" />
      <p>
        The cold-chain marginals are now correct Boltzmann draws, not basin artifacts. This
        is why the "Full statistical sampler" toggle in <em>/completer</em> uses PT.
      </p>
      <div className="lab-science-controls">
        <label>
          cold T = {Tcold.toFixed(2)}{" "}
          <input
            type="range"
            min={0.3}
            max={2.0}
            step={0.05}
            value={Tcold}
            onChange={(e) => setTcold(Number(e.target.value))}
          />
        </label>
        <button type="button" onClick={() => setSeed((s) => s + 1)}>
          Resample
        </button>
      </div>
      <div className="lab-science-row">
        <figure>
          <GraphView nodes={makeNodes(single.marginals)} edges={graphEdges} width={320} height={280} />
          <figcaption>
            Single chain at cold T
            <br />
            (node darkness = marginal)
          </figcaption>
        </figure>
        <figure>
          <GraphView nodes={makeNodes(pt.coldMarginals)} edges={graphEdges} width={320} height={280} />
          <figcaption>PT cold-chain marginals</figcaption>
        </figure>
        <figure>
          <GraphView nodes={makeNodes(exact)} edges={graphEdges} width={320} height={280} />
          <figcaption>Exact (truth)</figcaption>
        </figure>
      </div>
      <figure style={{ marginTop: 16 }}>
        <ChainStrip history={pt.history.slice(0, 400)} K={4} width={640} height={120} />
        <figcaption>
          Replica-exchange timeline (first 400 post-burn-in sweeps). Each row = a
          temperature rung; color = which original chain occupies that rung. Color flips =
          accepted swaps ({pt.swaps} total).
        </figcaption>
      </figure>
    </section>
  );
}
