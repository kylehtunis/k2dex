// S2: forward Ising on a 2D lattice. Slider for T, button to step
// sweeps, live magnetization-vs-sweep plot.

import { useCallback, useRef, useState } from "react";
import { BlockMath, InlineMath } from "../widgets/Math";
import { LinePlot } from "../widgets/LinePlot";
import { SpinGrid } from "../widgets/SpinGrid";
import { createLattice, latticeMagnetization, sweepLattice } from "../primitives/lattice";
import type { Lattice } from "../primitives/lattice";
import { Rng } from "../../sampler/rng";

const SIZE = 20;
const SWEEP_BATCH = 10;

export function S2Lattice() {
  const rngRef = useRef(new Rng(42));
  const [lattice, setLattice] = useState<Lattice>(() =>
    createLattice(SIZE, SIZE, new Rng(7)),
  );
  const [T, setT] = useState(2.3);
  const [magHistory, setMagHistory] = useState<number[]>([
    latticeMagnetization(createLattice(SIZE, SIZE, new Rng(7))),
  ]);

  const step = useCallback(() => {
    setLattice((prev) => {
      const next = prev.map((row) => row.slice()) as Lattice;
      sweepLattice(next, T, rngRef.current, SWEEP_BATCH);
      setMagHistory((h) => [...h, latticeMagnetization(next)].slice(-200));
      return next;
    });
  }, [T]);

  const reset = useCallback(() => {
    rngRef.current = new Rng(42);
    const L = createLattice(SIZE, SIZE, new Rng(7));
    setLattice(L);
    setMagHistory([latticeMagnetization(L)]);
  }, []);

  return (
    <section id="s2-lattice" className="lab-science-section">
      <h2>2. The Ising model on a 2D lattice</h2>
      <p>
        Each cell holds a spin <InlineMath formula="s_i \in \{-1, +1\}" />. Neighboring
        spins prefer to agree: the energy of a configuration is
      </p>
      <BlockMath formula="H(s) = -\sum_{\langle i,j \rangle} s_i s_j" />
      <p>
        and the probability of seeing configuration <InlineMath formula="s" /> is{" "}
        <InlineMath formula="p(s) \propto e^{-H(s)/T}" />. Low{" "}
        <InlineMath formula="T" />: spins align (magnetized). High{" "}
        <InlineMath formula="T" />: thermal noise wins.
      </p>
      <div className="lab-science-controls">
        <label>
          T = {T.toFixed(2)}{" "}
          <input
            type="range"
            min={0.5}
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
      <div className="lab-science-row">
        <figure>
          <SpinGrid lattice={lattice} cell={14} />
          <figcaption>Spin configuration (dark = +1, light = −1)</figcaption>
        </figure>
        <figure>
          <LinePlot
            width={420}
            height={200}
            series={[{ data: magHistory, color: "#1f4e8c" }]}
            yDomain={[-1, 1]}
            xLabel="sweeps (last 200)"
            yLabel="m"
          />
          <figcaption>Magnetization over time</figcaption>
        </figure>
      </div>
    </section>
  );
}
