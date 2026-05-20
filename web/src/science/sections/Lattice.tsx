// Forward Ising on a 2D lattice. Slider for T, button to step
// sweeps, live magnetization-vs-sweep plot.

import { useCallback, useEffect, useRef, useState } from "react";
import { BlockMath, InlineMath } from "../widgets/Math";
import { LinePlot } from "../widgets/LinePlot";
import { SpinGrid } from "../widgets/SpinGrid";
import { createLattice, latticeMagnetization, sweepLattice } from "../primitives/lattice";
import type { Lattice } from "../primitives/lattice";
import { Rng } from "../../sampler/rng";

const SIZE = 32;
const SWEEP_BATCH = 20;
const RUN_INTERVAL_MS = 100;

function randSeed() {
  return Math.floor(Math.random() * 2 ** 30);
}

export function Lattice() {
  const rngRef = useRef(new Rng(randSeed()));
  const initSeedRef = useRef(randSeed());
  const [lattice, setLattice] = useState<Lattice>(() =>
    createLattice(SIZE, SIZE, new Rng(initSeedRef.current)),
  );
  const [T, setT] = useState(2.3);
  const [magHistory, setMagHistory] = useState<number[]>([
    latticeMagnetization(createLattice(SIZE, SIZE, new Rng(initSeedRef.current))),
  ]);
  const [running, setRunning] = useState(false);

  const step = useCallback(() => {
    setLattice((prev) => {
      const next = prev.map((row) => row.slice()) as Lattice;
      sweepLattice(next, T, rngRef.current, SWEEP_BATCH);
      setMagHistory((h) => [...h, latticeMagnetization(next)].slice(-200));
      return next;
    });
  }, [T]);

  const reset = useCallback(() => {
    const newSeed = randSeed();
    initSeedRef.current = newSeed;
    rngRef.current = new Rng(randSeed());
    const L = createLattice(SIZE, SIZE, new Rng(newSeed));
    setLattice(L);
    setMagHistory([latticeMagnetization(L)]);
  }, []);

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

  return (
    <section id="lattice" className="lab-science-section">
      <h2>The Ising model on a 2D lattice</h2>
      <p>
        The Ising model is a way to describe that same behavior mathematically.
        We create a grid of cells, called a lattice, with each cell containing a "spin" that can point either up or down.
      </p>
      <p>
        The energy of a configuration of spins <InlineMath formula="s_i" /> with value <InlineMath formula="\pm 1" /> is given by the formula
      </p>
      <BlockMath formula="H(\vec{s}) = -\sum_{\langle i,j \rangle} s_i s_j" />
      <p>
        and the probability of seeing configuration <InlineMath formula="\vec{s}" /> is{" "}
        <InlineMath formula="p(\vec{s}) \propto e^{-H(\vec{s})/T}" />. This is called the <b>Boltzmann distribution</b> (remember that name, it'll come up again).
      </p>
      <p>
        In basic terms, it means that the energy of the system is lower when more neighbors are aligned, and that lower energy states are more likely to occur, especially at low temperatures.
      </p>
      <p>
        The widget below lets you run a simulation of the Ising model and see how it behaves at high and low temperatures.
        This temperature slider does the <i>exact same thing</i> as the one in the Team Completer, as I'll explain further down.
        Take a moment to play with the simulation. See if you can use it to understand why magnets will sometimes flip their poles when they heat up and cool back down.
      </p>
      <div className="lab-science-controls">
        <label>
          T = {T.toFixed(2)}{" "}
          <input
            type="range"
            min={0}
            max={5}
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
      </div>
      <div className="lab-science-row">
        <figure>
          <SpinGrid lattice={lattice} cell={10} />
          <figcaption>Spin configuration (dark = +1, light = −1)</figcaption>
        </figure>
        <figure>
          <LinePlot
            width={360}
            height={160}
            series={[{ data: magHistory, color: "#1f4e8c" }]}
            yDomain={[-1, 1]}
            xLabel="sweeps (last 200)"
            yLabel="m"
          />
          <LinePlot
            width={360}
            height={160}
            series={[{ data: magHistory.map(Math.abs), color: "#9c2a2a" }]}
            yDomain={[0, 1]}
            xLabel="sweeps (last 200)"
            yLabel="|m|"
          />
          <figcaption>Magnetization (top) and its magnitude (bottom)</figcaption>
        </figure>
      </div>
    </section>
  );
}
