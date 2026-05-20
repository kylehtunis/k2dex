// Forward Ising on a 2D lattice. Slider for T, button to step
// sweeps, live magnetization-vs-sweep plot.

import { useCallback, useRef, useState } from "react";
import { BlockMath, InlineMath } from "../widgets/Math";
import { LinePlot } from "../widgets/LinePlot";
import { SpinGrid } from "../widgets/SpinGrid";
import { createLattice, latticeMagnetization, sweepLattice } from "../primitives/lattice";
import type { Lattice } from "../primitives/lattice";
import { Rng } from "../../sampler/rng";

const SIZE = 40;
const SWEEP_BATCH = 20;

// Onsager's exact 2D Ising critical temperature: T_c = 2 / ln(1 + sqrt(2)).
const T_C = 2 / Math.log(1 + Math.sqrt(2));

interface TChoice {
  value: number;
  label: string;
  hint: string;
  critical?: boolean;
}

const T_CHOICES: TChoice[] = [
  { value: 1.0, label: "1.0", hint: "deep cold" },
  { value: 1.8, label: "1.8", hint: "cold" },
  { value: T_C, label: "T_c", hint: "critical", critical: true },
  { value: 2.6, label: "2.6", hint: "warm" },
  { value: 3.5, label: "3.5", hint: "hot" },
];

export function Lattice() {
  const rngRef = useRef(new Rng(42));
  const [lattice, setLattice] = useState<Lattice>(() =>
    createLattice(SIZE, SIZE, new Rng(7)),
  );
  const [T, setT] = useState(T_C);
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
    <section id="lattice" className="lab-science-section">
      <h2>The Ising model on a 2D lattice</h2>
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
        <div className="lab-t-picker" role="radiogroup" aria-label="Temperature">
          {T_CHOICES.map((c) => {
            const selected = Math.abs(c.value - T) < 1e-9;
            return (
              <button
                key={c.label}
                type="button"
                role="radio"
                aria-checked={selected}
                onClick={() => setT(c.value)}
                className={
                  "lab-t-btn" +
                  (selected ? " is-selected" : "") +
                  (c.critical ? " is-critical" : "")
                }
                title={`T = ${c.value.toFixed(3)} (${c.hint})`}
              >
                <span className="lab-t-btn-label">{c.label}</span>
                <span className="lab-t-btn-hint">{c.hint}</span>
              </button>
            );
          })}
        </div>
        <button type="button" onClick={step}>
          Step {SWEEP_BATCH} sweeps
        </button>
        <button type="button" onClick={reset}>
          Reset
        </button>
      </div>
      <p className="lab-science-note">
        At the critical temperature <InlineMath formula="T_c = 2/\ln(1+\sqrt{2}) \approx 2.269" />{" "}
        a 2D Ising lattice undergoes a phase transition: below{" "}
        <InlineMath formula="T_c" /> the system settles into long-range order (one sign dominates,{" "}
        <InlineMath formula="|m| > 0" />); above <InlineMath formula="T_c" /> thermal noise wins
        and <InlineMath formula="m \to 0" />. Right at <InlineMath formula="T_c" /> the lattice is
        scale-free — patches of every size coexist.
      </p>
      <div className="lab-science-row">
        <figure>
          <SpinGrid lattice={lattice} cell={8} />
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
