// S3: same lattice, but exposes the Metropolis decision step-by-step.
// Mode 1: "single step" — highlight the proposed cell, show ΔE and accept
// probability, accept or reject. Mode 2: "run" — mass-update via sweeps.

import { useCallback, useRef, useState } from "react";
import { BlockMath, InlineMath } from "../widgets/Math";
import { SpinGrid } from "../widgets/SpinGrid";
import { createLattice, sweepLattice } from "../primitives/lattice";
import type { Lattice, Spin } from "../primitives/lattice";
import { Rng } from "../../sampler/rng";

const SIZE = 12;

interface Proposal {
  i: number;
  j: number;
  dE: number;
  pAccept: number;
  outcome: "accept" | "reject";
}

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

export function S3MCMC() {
  const rngRef = useRef(new Rng(99));
  const [lattice, setLattice] = useState<Lattice>(() =>
    createLattice(SIZE, SIZE, new Rng(11)),
  );
  const [T, setT] = useState(2.3);
  const [proposal, setProposal] = useState<Proposal | null>(null);

  const propose = useCallback(() => {
    setLattice((prev) => {
      const i = rngRef.current.integers(SIZE);
      const j = rngRef.current.integers(SIZE);
      const s = prev[i][j];
      const dE = 2 * s * neighborSum(prev, i, j);
      const pAccept = dE <= 0 ? 1 : Math.exp(-dE / Math.max(T, 1e-9));
      const accept = dE <= 0 || rngRef.current.random() < pAccept;
      setProposal({ i, j, dE, pAccept, outcome: accept ? "accept" : "reject" });
      if (accept) {
        const next = prev.map((row) => row.slice()) as Lattice;
        next[i][j] = -s as Spin;
        return next;
      }
      return prev;
    });
  }, [T]);

  const runSweeps = useCallback(() => {
    setLattice((prev) => {
      const next = prev.map((r) => r.slice()) as Lattice;
      sweepLattice(next, T, rngRef.current, 5);
      return next;
    });
    setProposal(null);
  }, [T]);

  const reset = useCallback(() => {
    rngRef.current = new Rng(99);
    setLattice(createLattice(SIZE, SIZE, new Rng(11)));
    setProposal(null);
  }, []);

  return (
    <section id="s3-mcmc" className="lab-science-section">
      <h2>3. Sampling: Metropolis MCMC</h2>
      <p>
        We can't enumerate <InlineMath formula="2^{400}" /> lattice configurations. Markov
        chain Monte Carlo gets around this by random-walking through configurations in a
        way that, in the long run, visits each one with the correct Boltzmann weight.
      </p>
      <p>The Metropolis rule:</p>
      <BlockMath formula="\text{accept proposal with } p = \min\bigl(1,\ e^{-\Delta E / T}\bigr)" />
      <p>
        At each step: pick a random spin, compute the energy change{" "}
        <InlineMath formula="\Delta E" /> if we flipped it, accept the flip with
        probability <InlineMath formula="p" />. Energy-lowering flips always accepted;
        energy-raising flips sometimes accepted (with probability that shrinks as{" "}
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
        <button type="button" onClick={propose}>
          Propose one flip
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
          <SpinGrid
            lattice={lattice}
            cell={20}
            highlight={proposal ? { i: proposal.i, j: proposal.j } : null}
          />
          <figcaption>
            Highlighted = most recent proposal
            {proposal ? ` (${proposal.outcome})` : ""}
          </figcaption>
        </figure>
        <div>
          {proposal ? (
            <table className="lab-science-mcmc-readout">
              <tbody>
                <tr>
                  <th>cell</th>
                  <td>
                    ({proposal.i}, {proposal.j})
                  </td>
                </tr>
                <tr>
                  <th>ΔE</th>
                  <td>{proposal.dE.toFixed(2)}</td>
                </tr>
                <tr>
                  <th>p(accept)</th>
                  <td>{proposal.pAccept.toFixed(3)}</td>
                </tr>
                <tr>
                  <th>outcome</th>
                  <td>{proposal.outcome}</td>
                </tr>
              </tbody>
            </table>
          ) : (
            <p style={{ color: "#888" }}>
              Click "Propose one flip" to see a single Metropolis step.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
