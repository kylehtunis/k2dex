// Bridge section: the Ising model doesn't need a lattice. Same dynamics on
// an arbitrary graph (here an Erdős–Rényi giant component with Gaussian edge
// weights), driven by a T slider, mirroring the Lattice section's interaction.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BlockMath, InlineMath } from "../widgets/Math";
import { GraphView } from "../widgets/GraphView";
import { randomGraph, springLayout, sweepGraph } from "../primitives/graph";
import type { State } from "../primitives/graph";
import { Rng } from "../../sampler/rng";
import { LinePlot } from "../widgets/LinePlot";

const ER_N = 14;
const ER_P = 0.32;
const ER_SIGMA = 1.0;
const SWEEP_BATCH = 10;
const RUN_INTERVAL_MS = 100;
const VIEW_SIZE = 380;
const VIEW_RADIUS = 150;

function randSeed() {
  return Math.floor(Math.random() * 2 ** 30);
}

function graphMagnetization(state: State): number {
  return state.reduce((a, s) => a + s, 0) / state.length;
}

export function Graph() {
  const [graphSeed, setGraphSeed] = useState(() => randSeed());
  const { graph, positions } = useMemo(() => {
    const result = randomGraph(graphSeed, ER_N, ER_P, ER_SIGMA);
    const layoutRng = new Rng(graphSeed ^ 0xa5a5a5);
    const sprung = springLayout(result.graph, layoutRng, 240);
    return { graph: result.graph, positions: sprung };
  }, [graphSeed]);
  const initialState = useMemo<State>(
    () => new Array(graph.V).fill(0).map((_, i) => (i % 2 === 0 ? 1 : 0)),
    [graph.V],
  );
  const rngRef = useRef(new Rng(randSeed()));
  const [state, setState] = useState<State>(initialState.slice());
  const [T, setT] = useState(1.0);
  const [running, setRunning] = useState(false);
  const [magHistory, setMagHistory] = useState<number[]>(() => [
    graphMagnetization(initialState),
  ]);

  useEffect(() => {
    rngRef.current = new Rng(randSeed());
    setState(initialState.slice());
    setMagHistory([graphMagnetization(initialState)]);
    setRunning(false);
  }, [graphSeed, initialState]);

  const step = useCallback(() => {
    setState((prev) => {
      const next = prev.slice();
      sweepGraph(next, graph, T, rngRef.current, SWEEP_BATCH);
      setMagHistory((h) => [...h, graphMagnetization(next)].slice(-200));
      return next;
    });
  }, [graph, T]);

  const reset = useCallback(() => {
    rngRef.current = new Rng(randSeed());
    setState(initialState.slice());
    setMagHistory([graphMagnetization(initialState)]);
  }, [initialState]);

  useEffect(() => {
    if (!running) return;
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      if (now - last >= RUN_INTERVAL_MS) {
        last = now;
        step();
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running, step]);

  const nodes = positions.map((p, i) => ({
    id: i,
    label: String(i),
    x: VIEW_SIZE / 2 + VIEW_RADIUS * p.x,
    y: VIEW_SIZE / 2 + VIEW_RADIUS * p.y,
    active: state[i] === 1,
  }));
  const edges = graph.edges.map(([i, j, w]) => ({ i, j, weight: w }));

  return (
    <section id="graph" className="lab-science-section">
      <h2>Beyond the lattice</h2>
      <p>
        The 2d lattice model works well for explaining magnets, but we can also think of it as a special case of a more general model.
        The spins don't need to be in a grid, so we can represent any topology we want as a network of nodes (spins) and edges (couplings).
        We can also say that each spin has a bias <InlineMath formula="h_i" />, representing a preference for how it wants to spin 
        (for example, caused by an external magnetic field).
      </p>
      <p>The energy formula becomes:</p>
      <BlockMath formula="H(s) = -\sum_i h_i s_i - \sum_{i < j} J_{ij}\, s_i s_j" />
      <p>
        If you plug in 0 for <InlineMath formula="h_i" /> and 1 for <InlineMath formula="J_{ij}" />, you'll get the exact energy formula we used for the lattice version.
      </p>
      <p>
        Couplings can also be negative, meaning that those spins prefer to be <i>opposite</i>, rather than aligned.
        Also, from now on we'll think of spins as on or off (1 or 0) instead of +1 or -1.
        It makes more sense for what's coming up, and it doesn't change the math.
      </p>
      <p>
        The figure below is a randomly generated network representing a more general Ising model.
        While math can predict the Boltzmann distribution of the simple 2d lattice analytically,
        that isn't yet possible for arbitrary graphs. Try a few random graphs at different temperatures to see how even a tiny system can behave completely unpredictably.
      </p>
      <div className="lab-science-controls">
        <label>
          T = {T.toFixed(2)}{" "}
          <input
            type="range"
            min={0.1}
            max={4}
            step={0.05}
            value={T}
            onChange={(e) => setT(Number(e.target.value))}
          />
        </label>
        <button type="button" onClick={() => setRunning((r) => !r)}>
          {running ? "Pause" : "Play"}
        </button>
        <button type="button" onClick={step} disabled={running}>
          Step {SWEEP_BATCH} sweeps
        </button>
        <button type="button" onClick={reset}>
          Reset
        </button>
        <button type="button" onClick={() => setGraphSeed(randSeed())}>
          New graph
        </button>
      </div>
      <div className="lab-science-row">
        <figure>
          <GraphView
            nodes={nodes}
            edges={edges}
            width={VIEW_SIZE}
            height={VIEW_SIZE}
            nodeRadius={14}
            maxStrokeWidth={4}
          />
          <figcaption>
            Random graph ({graph.V} nodes, {graph.edges.length} edges). Dark node
            = spin on; light = off. Blue edge: J &gt; 0 (prefer agreement). Red
            edge: J &lt; 0 (prefer disagreement). Thickness ∝ |J|.
          </figcaption>
        </figure>
        <figure style={{ marginRight: 24 }}>
          <LinePlot
            width={360}
            height={160}
            series={[{ data: magHistory, color: "#1f4e8c" }]}
            yDomain={[0, 1]}
            xLabel="sweeps (last 200)"
            yLabel="frac on"
          />
          <figcaption>Magnetization over time</figcaption>
        </figure>
      </div>
    </section>
  );
}
