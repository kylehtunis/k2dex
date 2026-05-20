// Inverse Ising on SCOTUS votes. Two linked widgets:
//   1. Fitted-J graph at N = all votes.
//   2. Completer mirror: pin some justices to "voted with the bloc"; the
//      same J/h give P(s_i = 1 | pinned) for each unpinned justice by exact
//      enumeration over 2^(9 - |pinned|) configurations.

import { useEffect, useMemo, useState } from "react";
import { GraphView } from "../widgets/GraphView";
import { JUSTICE_POSITIONS } from "../data/scotusLayout";

interface Fit {
  n_used: number;
  J: number[][];
  h: number[];
}

interface FitsPayload {
  [key: string]: Fit;
}

interface VotesPayload {
  justices: string[];
  votes: number[][];
}

const BASE_URL = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");

type PinValue = 0 | 1;

/**
 * Exact conditional marginals P(s_i = 1 | s_pinned) under the fitted
 * pairwise model. Energy: H(s) = -Σ h_i s_i - Σ_{i<j} J_ij s_i s_j.
 * Pinned justices are clamped to their pinned value (0 or 1).
 */
function conditionalMarginals(
  J: number[][],
  h: number[],
  pinned: ReadonlyMap<number, PinValue>,
): number[] {
  const V = h.length;
  const unpinned: number[] = [];
  for (let i = 0; i < V; i++) if (!pinned.has(i)) unpinned.push(i);
  const K = unpinned.length;
  const s = new Array<number>(V).fill(0);
  for (const [i, v] of pinned) s[i] = v;
  let Z = 0;
  const numer = new Array<number>(V).fill(0);
  for (let bits = 0; bits < 1 << K; bits++) {
    for (let k = 0; k < K; k++) s[unpinned[k]] = (bits >> k) & 1;
    let H = 0;
    for (let i = 0; i < V; i++) if (s[i]) H -= h[i];
    for (let i = 0; i < V; i++) {
      if (!s[i]) continue;
      for (let j = i + 1; j < V; j++) {
        if (s[j]) H -= J[i][j];
      }
    }
    const w = Math.exp(-H);
    Z += w;
    for (const i of unpinned) if (s[i]) numer[i] += w;
  }
  return numer.map((n) => n / Z);
}

export function SCOTUS() {
  const [fits, setFits] = useState<FitsPayload | null>(null);
  const [votes, setVotes] = useState<VotesPayload | null>(null);
  const [pinned, setPinned] = useState<Map<number, PinValue>>(new Map());

  useEffect(() => {
    Promise.all([
      fetch(`${BASE_URL}/scotus/fits.json`).then((r) => r.json()),
      fetch(`${BASE_URL}/scotus/votes.json`).then((r) => r.json()),
    ]).then(([f, v]) => {
      setFits(f);
      setVotes(v);
    });
  }, []);

  const fit = fits?.all ?? null;

  const { nodes, edges } = useMemo(() => {
    if (!fit || !votes) return { nodes: [], edges: [] };
    const nodes = votes.justices.map((label, i) => ({
      id: i,
      label: label.split(" ")[0],
      x: JUSTICE_POSITIONS[i].x,
      y: JUSTICE_POSITIONS[i].y,
    }));
    const edges: { i: number; j: number; weight: number }[] = [];
    for (let i = 0; i < votes.justices.length; i++) {
      for (let j = i + 1; j < votes.justices.length; j++) {
        if (Math.abs(fit.J[i][j]) > 0.05) {
          edges.push({ i, j, weight: fit.J[i][j] });
        }
      }
    }
    return { nodes, edges };
  }, [fit, votes]);

  const completions = useMemo(() => {
    if (!fit) return null;
    return conditionalMarginals(fit.J, fit.h, pinned);
  }, [fit, pinned]);

  if (!fits || !votes || !fit) {
    return (
      <section id="scotus" className="lab-science-section">
        <h2>Inverse Ising: the Supreme Court</h2>
        <p>Loading SCOTUS data…</p>
      </section>
    );
  }

  // Cycle: unpinned → 1 (conservative side) → 0 (liberal side) → unpinned.
  const cyclePin = (i: number) => {
    setPinned((prev) => {
      const next = new Map(prev);
      const cur = next.get(i);
      if (cur === undefined) next.set(i, 1);
      else if (cur === 1) next.set(i, 0);
      else next.delete(i);
      return next;
    });
  };

  const ranked =
    completions === null
      ? []
      : votes.justices
          .map((label, i) => ({ i, label, p: completions[i] }))
          .filter((r) => !pinned.has(r.i))
          .sort((a, b) => b.p - a.p);

  return (
    <section id="scotus" className="lab-science-section">
      <h2>Flipping the problem: inverse Ising on the Supreme Court</h2>
      <p>
        So far we've assumed someone hands us the couplings <em>J</em> and asks
        us to sample. But where does <em>J</em> come from in practice? You only
        get to <em>observe</em> samples — votes, teams, biological states — and
        you have to recover the couplings from data. That's the{" "}
        <em>inverse</em> Ising problem. The standard tool is pseudo-likelihood:
        for each spin, fit a logistic regression predicting its state from all
        the others. The regression coefficients become that spin's row of{" "}
        <em>J</em>.
      </p>
      <p>
        We applied this to {votes.votes.length} non-unanimous Rehnquist-court
        decisions (the same data Lee, Broderick &amp; Frey used in their 2015
        paper). Each vote is 9 bits. The fitted graph below shows positive
        couplings (blue) between justices who vote together more than chance,
        and negative couplings (red) between those who tend to disagree.
      </p>
      <figure>
        <GraphView
          nodes={nodes}
          edges={edges}
          width={460}
          height={420}
          nodeRadius={26}
        />
        <figcaption>
          Fitted J on all {fit.n_used} non-unanimous votes. Edges with |J| &gt;
          0.05 shown.
        </figcaption>
      </figure>
      <h3 className="lab-science-subhead">Same machinery: completing a vote</h3>
      <p>
        Once we have <em>J</em> and <em>h</em>, predicting a missing spin
        conditional on observed ones is just another marginal computation —{" "}
        <em>exactly</em> what <em>/completer</em> does with Pokémon teams.
        Click any justice below to pin them: once for <span className="lab-scotus-swatch lab-scotus-swatch-1" />{" "}
        <em>conservative</em> side, again for <span className="lab-scotus-swatch lab-scotus-swatch-0" />{" "}
        <em>liberal</em> side, again to clear. The model returns the
        conditional probability each unpinned justice votes conservative
        given that pattern. With only nine spins we compute it exactly by
        enumeration — no MF or PT needed.
      </p>
      <div className="lab-scotus-completer">
        <div className="lab-scotus-pins" role="group" aria-label="Pin justices">
          {votes.justices.map((label, i) => {
            const v = pinned.get(i);
            const cls =
              v === 1
                ? "lab-scotus-pin is-pinned-1"
                : v === 0
                  ? "lab-scotus-pin is-pinned-0"
                  : "lab-scotus-pin";
            return (
              <button
                key={i}
                type="button"
                className={cls}
                onClick={() => cyclePin(i)}
                aria-pressed={v !== undefined}
              >
                {label.split(" ")[0]}
              </button>
            );
          })}
          {pinned.size > 0 && (
            <button
              type="button"
              className="lab-scotus-pin lab-scotus-pin-clear"
              onClick={() => setPinned(new Map())}
            >
              clear
            </button>
          )}
        </div>
        <div className="lab-scotus-ranks">
          {ranked.length === 0 ? (
            <p className="lab-science-note">
              All nine pinned. Unpin one to see predictions.
            </p>
          ) : (
            ranked.map(({ i, label, p }) => (
              <div key={i} className="lab-scotus-rank-row">
                <span className="lab-scotus-rank-name">
                  {label.split(" ")[0]}
                </span>
                <div className="lab-scotus-rank-bar">
                  <div
                    className="lab-scotus-rank-fill"
                    style={{ width: `${p * 100}%` }}
                  />
                </div>
                <span className="lab-scotus-rank-pct">
                  {(p * 100).toFixed(0)}%
                </span>
              </div>
            ))
          )}
        </div>
        <p className="lab-science-note">
          Bars show P(votes conservative) for each unpinned justice. With
          nothing pinned, you see baseline marginals — how often each justice
          voted conservative overall. Pin Scalia conservative and the other
          conservatives (Rehnquist, Thomas) jump near 100%; pin Ginsburg
          liberal and Breyer and Stevens drop sharply. Pin one of each — e.g.
          Scalia conservative, Ginsburg liberal — and the swing justices
          (O'Connor, Kennedy) settle in the middle. Same lift / suppression
          dynamic as Pokémon teammates, on a graph you can hold in your head.
        </p>
      </div>
    </section>
  );
}
