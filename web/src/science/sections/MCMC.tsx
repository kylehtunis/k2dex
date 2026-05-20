// Same lattice as the Lattice section, but exposes the Metropolis decision
// step-by-step. Two-stage UI: Propose (compute ΔE and p) → Evaluate (roll the
// RNG, accept or reject). Mass-update via sweeps is a separate button.

import { useCallback, useRef, useState } from "react";
import { BlockMath, InlineMath } from "../widgets/Math";
import { SpinGrid } from "../widgets/SpinGrid";
import { createLattice, sweepLattice } from "../primitives/lattice";
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

export function MCMC() {
  const rngRef = useRef(new Rng(99));
  const [lattice, setLattice] = useState<Lattice>(() =>
    createLattice(SIZE, SIZE, new Rng(11)),
  );
  const [T, setT] = useState(2.3);
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
    const { i, j, dE, pAccept } = phase;
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

  const runSweeps = useCallback(() => {
    setLattice((prev) => {
      const next = prev.map((r) => r.slice()) as Lattice;
      sweepLattice(next, T, rngRef.current, 5);
      return next;
    });
    setPhase({ kind: "idle" });
  }, [T]);

  const reset = useCallback(() => {
    rngRef.current = new Rng(99);
    setLattice(createLattice(SIZE, SIZE, new Rng(11)));
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
      <h2>Sampling: Metropolis MCMC</h2>
      <p>
        We can't enumerate <InlineMath formula="2^{400}" /> lattice configurations. Markov
        chain Monte Carlo gets around this by random-walking through configurations in a
        way that, in the long run, visits each one with the correct Boltzmann weight.
      </p>
      <p>The Metropolis rule:</p>
      <BlockMath formula="\text{accept proposal with } p = \min\bigl(1,\ e^{-\Delta E / T}\bigr)" />
      <p>
        Each step has two halves. <strong>Propose</strong>: pick a random spin and compute
        the energy change <InlineMath formula="\Delta E" /> if we flipped it; that determines
        the accept probability <InlineMath formula="p" />. <strong>Evaluate</strong>: roll a
        uniform random number; if it lands under <InlineMath formula="p" /> the flip happens,
        otherwise the spin stays put. Energy-lowering flips always accepted; energy-raising
        flips sometimes accepted (with probability that shrinks as{" "}
        <InlineMath formula="T" /> drops).
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
        <button type="button" onClick={stepAction}>
          {stepLabel}
        </button>
        <button type="button" onClick={runSweeps}>
          Run 5 sweeps
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
                  <td>{phase.pAccept.toFixed(3)}</td>
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
