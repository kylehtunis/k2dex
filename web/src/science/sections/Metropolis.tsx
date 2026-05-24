// Same lattice as the Lattice section, but exposes the Metropolis decision
// step-by-step. Two-stage UI: Propose (compute ΔE and p) → Evaluate (roll the
// RNG, accept or reject). Mass-update via sweeps is a separate button.

import { useCallback, useRef, useState } from "react";
import { BlockMath, InlineMath } from "../widgets/Math";
import { SpinGrid } from "../widgets/SpinGrid";
import { createLattice } from "../primitives/lattice";
import type { Lattice, Spin } from "../primitives/lattice";
import { Rng } from "../../sampler/rng";

const SIZE = 12;

type Phase =
  | { kind: "idle" }
  | {
      kind: "pending";
      i: number;
      j: number;
      dE: number;
      pAccept: number;
    }
  | {
      kind: "evaluated";
      i: number;
      j: number;
      dE: number;
      pAccept: number;
      outcome: "accept" | "reject";
    };

function neighborSum(L: Lattice, i: number, j: number): number {
  const R = L.length;
  const C = L[0].length;
  return (
    L[(i - 1 + R) % R][j] +
    L[(i + 1) % R][j] +
    L[i][(j - 1 + C) % C] +
    L[i][(j + 1) % C]
  );
}

function randSeed() {
  return Math.floor(Math.random() * 2 ** 30);
}

export function Metropolis() {
  const initSeedRef = useRef(randSeed());
  const rngRef = useRef(new Rng(randSeed()));
  const [lattice, setLattice] = useState<Lattice>(() =>
    createLattice(SIZE, SIZE, new Rng(initSeedRef.current)),
  );
  const [T, setT] = useState(2.3);
  const TRef = useRef(T);
  TRef.current = T;
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });

  const propose = useCallback(() => {
    const i = rngRef.current.integers(SIZE);
    const j = rngRef.current.integers(SIZE);
    const s = lattice[i][j];
    const dE = 2 * s * neighborSum(lattice, i, j);
    const pAccept = dE <= 0 ? 1 : Math.exp(-dE / Math.max(T, 1e-9));
    setPhase({ kind: "pending", i, j, dE, pAccept });
  }, [lattice, T]);

  const evaluate = useCallback(() => {
    if (phase.kind !== "pending") return;
    const { i, j, dE } = phase;
    const pAccept = dE <= 0 ? 1 : Math.exp(-dE / Math.max(TRef.current, 1e-9));
    const accept = dE <= 0 || rngRef.current.random() < pAccept;
    if (accept) {
      setLattice((prev) => {
        const next = prev.map((row) => row.slice()) as Lattice;
        next[i][j] = -next[i][j] as Spin;
        return next;
      });
    }
    setPhase({
      kind: "evaluated",
      i,
      j,
      dE,
      pAccept,
      outcome: accept ? "accept" : "reject",
    });
  }, [phase]);

  const reset = useCallback(() => {
    const newSeed = randSeed();
    initSeedRef.current = newSeed;
    rngRef.current = new Rng(randSeed());
    setLattice(createLattice(SIZE, SIZE, new Rng(newSeed)));
    setPhase({ kind: "idle" });
  }, []);

  const highlight =
    phase.kind === "idle" ? null : { i: phase.i, j: phase.j };
  const stepLabel =
    phase.kind === "pending" ? "Evaluate flip →" : "Propose a flip →";
  const stepAction = phase.kind === "pending" ? evaluate : propose;
  const captionTail =
    phase.kind === "pending"
      ? " (pending)"
      : phase.kind === "evaluated"
        ? ` (${phase.outcome})`
        : "";

  return (
    <section id="mcmc" className="lab-science-section">
      <h3>Metropolis</h3>
      <p>
        I've let you play with a few Ising simulations now, but I haven't actually explained how to go from mathematical equations to simulation.
        The key is the Markov Chain Monte Carlo (MCMC) method. The most basic version of MCMC, called the Metropolis algorithm,
        just looks at one spin at a time and decides whether to flip it based on one rule:
      </p>
      <BlockMath formula="\text{accept proposal with } p = \min\bigl(1,\ e^{-\Delta E / T}\bigr)" />
      <p>
        In other words, if flipping the spin would lower the energy of the system, then we flip it.
        If not, we still <i>might</i> flip it with a probability that increases as the energy difference gets smaller or as the temperature gets higher.
        Given infinite time, this process will find itself in configuration <InlineMath formula="\vec{s}" /> at the exact rate predicted by the Boltzmann distribution.
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
        <button
          type="button"
          onClick={stepAction}
        >
          {stepLabel}
        </button>
        <button type="button" onClick={reset}>
          Reset
        </button>
      </div>
      <div className="lab-science-row">
        <figure>
          <SpinGrid lattice={lattice} cell={20} highlight={highlight} />
          <figcaption>
            Highlighted = current proposal{captionTail}
          </figcaption>
        </figure>
        <div>
          {phase.kind === "idle" ? (
            <p style={{ color: "#888" }}>
              Click "Propose a flip" to set up a single Metropolis step.
            </p>
          ) : (
            <table className="lab-science-mcmc-readout">
              <tbody>
                <tr>
                  <th>cell</th>
                  <td>
                    ({phase.i}, {phase.j})
                  </td>
                </tr>
                <tr>
                  <th>ΔE</th>
                  <td>{phase.dE.toFixed(2)}</td>
                </tr>
                <tr>
                  <th>p(accept)</th>
                  <td>{(phase.dE <= 0 ? 1 : Math.exp(-phase.dE / Math.max(T, 1e-9))).toFixed(3)}</td>
                </tr>
                <tr>
                  <th>status</th>
                  <td>
                    {phase.kind === "pending"
                      ? "awaiting RNG"
                      : phase.outcome}
                  </td>
                </tr>
              </tbody>
            </table>
          )}
        </div>
      </div>
    </section>
  );
}
