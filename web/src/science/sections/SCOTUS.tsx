// Inverse Ising on SCOTUS votes. Two linked widgets:
//   1. Fitted-J graph at N = all votes.
//   2. Completer mirror: pin justices to conservative/liberal; show top
//      configurations by energy with corpus-observation counts, plus per-
//      justice conditional marginals.

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

// Short two-letter labels for in-row config display.
const JUSTICE_ABBREV = ["Re", "St", "OC", "Sc", "Ke", "So", "Th", "Gi", "Br"];

// Drop sprite URLs in later; null falls back to the abbreviation chip.
const JUSTICE_SPRITES: (string | null)[] = [
  null, null, null, null, null, null, null, null, null,
];

const TOP_N_CONFIGS = 6;

type PinValue = 0 | 1;

function configEnergy(
  J: number[][],
  h: number[],
  s: number[],
): number {
  let H = 0;
  for (let i = 0; i < h.length; i++) if (s[i]) H -= h[i];
  for (let i = 0; i < h.length; i++) {
    if (!s[i]) continue;
    for (let j = i + 1; j < h.length; j++) {
      if (s[j]) H -= J[i][j];
    }
  }
  return H;
}

/** Conditional marginals P(s_i = 1 | s_pinned) via exact enumeration. */
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
    const w = Math.exp(-configEnergy(J, h, s));
    Z += w;
    for (const i of unpinned) if (s[i]) numer[i] += w;
  }
  return numer.map((n) => n / Z);
}

interface RankedConfig {
  bits: number;
  spins: number[];
  H: number;
}

/** Top-N configurations consistent with pins, sorted by lowest energy first. */
function topConfigurations(
  J: number[][],
  h: number[],
  pinned: ReadonlyMap<number, PinValue>,
  topN: number,
): RankedConfig[] {
  const V = h.length;
  const unpinned: number[] = [];
  for (let i = 0; i < V; i++) if (!pinned.has(i)) unpinned.push(i);
  const K = unpinned.length;
  const s = new Array<number>(V).fill(0);
  for (const [i, v] of pinned) s[i] = v;
  const out: RankedConfig[] = [];
  for (let bits = 0; bits < 1 << K; bits++) {
    for (let k = 0; k < K; k++) s[unpinned[k]] = (bits >> k) & 1;
    const H = configEnergy(J, h, s);
    let fullBits = 0;
    for (let i = 0; i < V; i++) if (s[i]) fullBits |= 1 << i;
    out.push({ bits: fullBits, spins: s.slice(), H });
  }
  out.sort((a, b) => a.H - b.H);
  return out.slice(0, topN);
}

function configBits(row: readonly number[]): number {
  let bits = 0;
  for (let i = 0; i < row.length; i++) if (row[i]) bits |= 1 << i;
  return bits;
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

  const corpusCounts = useMemo(() => {
    const m = new Map<number, number>();
    if (!votes) return m;
    for (const row of votes.votes) {
      const k = configBits(row);
      m.set(k, (m.get(k) ?? 0) + 1);
    }
    return m;
  }, [votes]);

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
        edges.push({ i, j, weight: fit.J[i][j] });
      }
    }
    return { nodes, edges };
  }, [fit, votes]);

  const marginals = useMemo(() => {
    if (!fit) return null;
    return conditionalMarginals(fit.J, fit.h, pinned);
  }, [fit, pinned]);

  const topConfigs = useMemo(() => {
    if (!fit) return [] as RankedConfig[];
    return topConfigurations(fit.J, fit.h, pinned, TOP_N_CONFIGS);
  }, [fit, pinned]);

  if (!fits || !votes || !fit) {
    return (
      <section id="scotus" className="lab-science-section">
        <h3>Fitting a model to data</h3>
        <p>Loading SCOTUS data…</p>
      </section>
    );
  }

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
    marginals === null
      ? []
      : votes.justices
          .map((label, i) => ({ i, label, p: marginals[i] }))
          .filter((r) => !pinned.has(r.i))
          .sort((a, b) => b.p - a.p);

  const renderJusticeSlot = (i: number, vote: PinValue, isPinned: boolean) => {
    const cls =
      "lab-scotus-slot " +
      (vote === 1 ? "lab-scotus-slot-c" : "lab-scotus-slot-l") +
      (isPinned ? " is-pinned" : "");
    const sprite = JUSTICE_SPRITES[i];
    return (
      <div key={i} className={cls} title={votes.justices[i]}>
        {sprite ? (
          <img src={sprite} alt={votes.justices[i]} />
        ) : (
          <span className="lab-scotus-slot-label">{JUSTICE_ABBREV[i]}</span>
        )}
      </div>
    );
  };

  return (
    <section id="scotus" className="lab-science-section">
      <h3>Fitting a model to data</h3>
      <p>
        In all the examples so far, we've talked about what the model is <i>doing</i>, but where does it come <i>from</i>?
        Well, suppose you have a system that you want to model using an Ising simulation.
        You can observe the system and count all the times it appears in each state,
        but for even a simple system like our 32x32 lattice, there are 2^1024 possible states, more than the number of protons in the universe.
        Good luck even seeing the same one twice, let alone trying to get good estimates of their probabilities.
      </p>
      <p>
        So, the task is to estimate the parameters of a system by observing an infinitesimal subset of the possible states.
        This is called the <em>inverse</em> Ising problem
        (and by now you should be getting a sense how this all might eventually tie back into Pokemon teams).
      </p>
      <p>
        The standard tool for this is pseudo-likelihood estimation:
        for each spin, fit a logistic regression predicting its state from all
        the others. The regression coefficients become that spin's row of{" "}
        <em>J</em>. To see how it works, we'll look at a real-world example that's small enough that we
        <i>can</i> observe a significant portion of the possible configurations: voting patterns of the nine justices of the United States Supreme Court.
      </p>
      <h3>The Supreme Court</h3>
      <p>
        This idea to fit Supreme Court votes to an Ising model comes from a 2015
        paper (cite), and here I'm using the same method and data. Each vote is represented by a network of 9 spins. The fitted graph below shows positive
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
      </figure>
      <h4 className="lab-science-subhead">Completing a vote</h4>
      <p>
        Once we have <em>J</em> and <em>h</em>, predicting unobserved spins
        conditional on observed ones is just another marginal computation —{" "}
        <em>exactly</em> what <em>/completer</em> does with Pokémon teams.
        Click any justice to pin them: once for <span className="lab-scotus-swatch lab-scotus-swatch-1" />{" "}
        <em>conservative</em>, again for <span className="lab-scotus-swatch lab-scotus-swatch-0" />{" "}
        <em>liberal</em>, again to clear. With only nine spins we can enumerate
        all 2<sup>9</sup> = 512 configurations exactly and rank them by energy.
        The lowest-energy configurations are the ones the model thinks are most
        likely to occur.
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

        <div className="lab-scotus-configs">
          <div className="lab-scotus-configs-head">
            <span className="lab-scotus-configs-head-spins">
              Top-{TOP_N_CONFIGS} configurations (lowest energy first)
            </span>
            <span className="lab-scotus-configs-head-h">H</span>
            <span className="lab-scotus-configs-head-score">score</span>
            <span className="lab-scotus-configs-head-obs">observed</span>
          </div>
          {topConfigs.map((cfg) => {
            const count = corpusCounts.get(cfg.bits) ?? 0;
            return (
              <div key={cfg.bits} className="lab-scotus-config-row">
                <div className="lab-scotus-config-spins">
                  {cfg.spins.map((vote, i) =>
                    renderJusticeSlot(i, vote as PinValue, pinned.has(i)),
                  )}
                </div>
                <span className="lab-scotus-config-h">
                  {cfg.H.toFixed(2)}
                </span>
                <span className="lab-scotus-config-score">
                  {(-cfg.H).toFixed(2)}
                </span>
                <span className="lab-scotus-config-obs">
                  {count > 0 ? `${count}×` : "—"}
                </span>
              </div>
            );
          })}
        </div>
        <p className="lab-science-note">
          The same Boltzmann distribution we used for spin lattices: lower H =
          more probable. The "score" column is just −H, the same convention the
          Pokémon completer uses so that higher = better. "Observed" counts how
          many times that exact 9-vote pattern appeared in the {votes.votes.length} raw votes.
        </p>

        <h4 className="lab-scotus-subhead">
          Conditional marginals for unpinned justices
        </h4>
        <div className="lab-scotus-ranks">
          {ranked.length === 0 ? (
            <p className="lab-science-note">
              All nine pinned — every justice's vote is fixed.
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
          Blue fill = P(votes conservative); red remainder = P(votes liberal).
          Pin Scalia conservative and the other conservatives (Rehnquist, Thomas)
          jump near 100%. Pin Ginsburg liberal and Stevens / Breyer drop sharply.
          Pin one of each — Scalia conservative, Ginsburg liberal — and the
          swing justices (O'Connor, Kennedy) settle in the middle.
        </p>
      </div>
    </section>
  );
}
