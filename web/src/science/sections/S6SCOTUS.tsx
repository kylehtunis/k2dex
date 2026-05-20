// S6: inverse Ising on SCOTUS votes. User picks a checkpoint N; widget
// shows fitted J for justices at that data budget. Folds the
// "fit J from data" concept into a real-world application.

import { useEffect, useMemo, useState } from "react";
import { GraphView } from "../widgets/GraphView";
import { JUSTICE_POSITIONS } from "../data/scotusLayout";

interface FitsPayload {
  [key: string]: { n_used: number; J: number[][]; h: number[] };
}

interface VotesPayload {
  justices: string[];
  votes: number[][];
}

const CHECKPOINTS = ["10", "50", "100", "500", "all"] as const;
type Checkpoint = (typeof CHECKPOINTS)[number];

const BASE_URL = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");

export function S6SCOTUS() {
  const [fits, setFits] = useState<FitsPayload | null>(null);
  const [votes, setVotes] = useState<VotesPayload | null>(null);
  const [checkpoint, setCheckpoint] = useState<Checkpoint>("all");

  useEffect(() => {
    Promise.all([
      fetch(`${BASE_URL}/scotus/fits.json`).then((r) => r.json()),
      fetch(`${BASE_URL}/scotus/votes.json`).then((r) => r.json()),
    ]).then(([f, v]) => {
      setFits(f);
      setVotes(v);
    });
  }, []);

  const { nodes, edges } = useMemo(() => {
    if (!fits || !votes) return { nodes: [], edges: [] };
    const fit = fits[checkpoint];
    const nodes = votes.justices.map((label, i) => ({
      id: i,
      label: label.split(" ")[0], // last name only — fits in node circle
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
  }, [fits, votes, checkpoint]);

  if (!fits || !votes) {
    return (
      <section id="s6-scotus" className="lab-science-section">
        <h2>6. Inverse Ising: the Supreme Court</h2>
        <p>Loading SCOTUS data…</p>
      </section>
    );
  }

  return (
    <section id="s6-scotus" className="lab-science-section">
      <h2>6. Flipping the problem: inverse Ising on the Supreme Court</h2>
      <p>
        So far we've assumed someone hands us the couplings <em>J</em> and asks us to
        sample. But where does <em>J</em> come from in practice? You only get to{" "}
        <em>observe</em> samples — votes, teams, biological states — and you have to recover
        the couplings from data. That's the <em>inverse</em> Ising problem. The standard
        tool is pseudo-likelihood: for each spin, fit a logistic regression predicting its
        state from all the others. The regression coefficients become that spin's row of{" "}
        <em>J</em>.
      </p>
      <p>
        We applied this to {votes.votes.length} non-unanimous Rehnquist-court decisions
        (the same data Lee, Broderick &amp; Frey used in their 2015 paper). Each vote is 9
        bits. The fitted graph shows positive couplings (blue) between justices who vote
        together more than chance, and negative couplings (red) between those who tend to
        disagree. The same machinery powers <em>/completer</em> — 574 Pokemon-item features
        instead of 9 justices, ~7,000 team rosters instead of ~{votes.votes.length} votes.
      </p>
      <div className="lab-science-controls">
        <label>
          N votes seen:{" "}
          <select
            value={checkpoint}
            onChange={(e) => setCheckpoint(e.target.value as Checkpoint)}
          >
            {CHECKPOINTS.map((c) => (
              <option key={c} value={c}>
                {c === "all" ? `all (${fits.all.n_used})` : c}
              </option>
            ))}
          </select>
        </label>
        <span style={{ color: "#888", fontSize: 13 }}>
          Showing edges with |J| &gt; 0.05
        </span>
      </div>
      <figure>
        <GraphView nodes={nodes} edges={edges} width={460} height={420} nodeRadius={26} />
        <figcaption>
          Fitted J at N = {fits[checkpoint].n_used}. At N = 10 the graph is mostly noise;
          by N = 500 the ideological blocs are stable.
        </figcaption>
      </figure>
    </section>
  );
}
