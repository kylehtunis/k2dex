// Bridge section: the Ising model doesn't need a lattice. Same dynamics on
// an arbitrary graph (here an Erdős–Rényi giant component with Gaussian edge
// weights), driven by a T slider, mirroring the Lattice section's interaction.

import { useCallback, useMemo, useRef, useState } from "react";
import { BlockMath, InlineMath } from "../widgets/Math";
import { GraphView } from "../widgets/GraphView";
import { randomGraph, sweepGraph } from "../primitives/graph";
import type { State } from "../primitives/graph";
import { Rng } from "../../sampler/rng";

const GRAPH_SEED = 11;
const ER_N = 14;
const ER_P = 0.32;
const ER_SIGMA = 1.0;
const SWEEP_BATCH = 10;
const VIEW_SIZE = 360;
const VIEW_RADIUS = 130;

export function Graph() {
  const { graph, positions } = useMemo(
    () => randomGraph(GRAPH_SEED, ER_N, ER_P, ER_SIGMA),
    [],
  );
  const initialState = useMemo<State>(
    () => new Array(graph.V).fill(0).map((_, i) => (i % 2 === 0 ? 1 : 0)),
    [graph.V],
  );
  const rngRef = useRef(new Rng(31));
  const [state, setState] = useState<State>(initialState.slice());
  const [T, setT] = useState(1.0);

  const step = useCallback(() => {
    setState((prev) => {
      const next = prev.slice();
      sweepGraph(next, graph, T, rngRef.current, SWEEP_BATCH);
      return next;
    });
  }, [graph, T]);

  const reset = useCallback(() => {
    rngRef.current = new Rng(31);
    setState(initialState.slice());
  }, [initialState]);

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
      <h2>Beyond the lattice: the same model on any graph</h2>
      <p>
        Nothing about the Ising story required a grid. The same energy works on{" "}
        <em>any</em> graph of spins:
      </p>
      <BlockMath formula="H(s) = -\sum_i h_i s_i - \sum_{i < j} J_{ij}\, s_i s_j" />
      <p>
        The lattice was a special case where every edge has the same{" "}
        <InlineMath formula="J" /> and each spin has four neighbors. Drop that
        restriction and the graph can be irregular, edge weights can vary in
        sign and magnitude, and "neighbor" is whatever the structure of the
        problem says it is. The graph below is a small random network with
        Gaussian edge weights — some couplings positive (blue, agree), some
        negative (red, disagree). Same Metropolis dynamics, same{" "}
        <InlineMath formula="p(s) \propto e^{-H(s)/T}" />.
      </p>
      <div className="lab-science-controls">
        <label>
          T = {T.toFixed(2)}{" "}
          <input
            type="range"
            min={0.2}
            max={4}
            step={0.05}
            value={T}
            onChange={(e) => setT(Number(e.target.value))}
          />
        </label>
        <button type="button" onClick={step}>
          Step {SWEEP_BATCH} sweeps
        </button>
        <button type="button" onClick={reset}>
          Reset
        </button>
      </div>
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
      <p>
        This generality is the whole point. The graph of Supreme Court justices
        and the graph of competitive Pokémon are not lattices — they're
        irregular structures with their own positive and negative couplings.
        Once we can fit J from data (next), the same machinery applies to both.
      </p>
    </section>
  );
}
