// Closing section: bridges the toy/SCOTUS machinery to the real model.
// The figure is a force-directed graph of the top species, drawn from the
// species-only (Phase 2) J — one node per species. A slider hides couplings
// below a threshold and recomputes the spring layout, so the strongest
// structure emerges as you raise it. Not animated — just layout. The prose's
// spin/parameter counts read from the Species @ Item (Phase 3) model via the
// shared context; the figure loads the species model independently.

import { useEffect, useMemo, useState } from "react";
import { GraphView } from "../widgets/GraphView";
import { useModel } from "../../state/ModelContext";
import { loadModel } from "../../sampler/model";
import type { IsingModel } from "../../sampler/types";
import { spriteUrl } from "../../render/sprite-url";
import { makeGraph, springLayout } from "../primitives/graph";
import { Rng } from "../../sampler/rng";

// Candidate nodes: the N most-used species (by their best build's marginal).
const TOP_SPECIES = 32;
const LAYOUT_SEED = 0x5eed;
const LAYOUT_ITERS = 320;
const VIEW_SIZE = 600;
const VIEW_RADIUS = 200;
const NODE_RADIUS = 18;
const SPRITE_SIZE = NODE_RADIUS * 2.2; // must match GraphView's sprite sizing
const SPRITE_OPACITY = 1.0;

// One representative vocab index per species (its highest-marginal build),
// since the figure is species-level but the model is (species, item).
interface SpeciesRep {
  species: string;
  vocabIdx: number;
  m: number;
}

// Push apart any sprites closer than `minSep` pixels. A few relaxation passes
// after the spring layout: dense coupling cores otherwise collapse on top of
// each other, and sprites (unlike dots) can't usefully overlap.
function relaxOverlaps(
  pts: { x: number; y: number }[],
  minSep: number,
  iters: number,
  size: number,
  pad: number,
) {
  for (let it = 0; it < iters; it++) {
    for (let a = 0; a < pts.length; a++) {
      for (let b = a + 1; b < pts.length; b++) {
        let dx = pts[b].x - pts[a].x;
        let dy = pts[b].y - pts[a].y;
        let d = Math.hypot(dx, dy);
        if (d < 1e-6) {
          dx = a - b || 1;
          dy = 1;
          d = Math.hypot(dx, dy);
        }
        if (d < minSep) {
          const push = (minSep - d) / 2;
          pts[a].x -= (dx / d) * push;
          pts[a].y -= (dy / d) * push;
          pts[b].x += (dx / d) * push;
          pts[b].y += (dy / d) * push;
        }
      }
    }
    for (const p of pts) {
      p.x = Math.min(size - pad, Math.max(pad, p.x));
      p.y = Math.min(size - pad, Math.max(pad, p.y));
    }
  }
}

export function Pokemon() {
  const { model, phaseKey, setPhaseKey, status } = useModel();

  // Force Phase 3 so the prose's spin/parameter counts read live from the
  // Species @ Item model.
  useEffect(() => {
    if (phaseKey !== "species_item") setPhaseKey("species_item");
  }, []);

  const ready =
    model !== null && status === "ready" && phaseKey === "species_item";

  // The figure uses the species-only (Phase 2) model — one node per species,
  // with J directly between species. Loaded independently of the shared
  // context, which stays on species_item for the prose counts above.
  const [figModel, setFigModel] = useState<IsingModel | null>(null);
  useEffect(() => {
    let cancelled = false;
    loadModel("species").then((m) => {
      if (!cancelled) setFigModel(m);
    });
    return () => {
      cancelled = true;
    };
  }, []);
  const figReady = figModel !== null;

  // Live model dimensions for the prose.
  const nSpins = ready ? model!.V : null;
  const nParams = nSpins !== null ? nSpins + (nSpins * (nSpins - 1)) / 2 : null;
  const nTeams = ready ? model!.nCorpusTeams : null;

  // Top species by their dominant build's marginal.
  const reps = useMemo<SpeciesRep[]>(() => {
    if (!figReady) return [];
    const best = new Map<string, SpeciesRep>();
    for (let i = 0; i < figModel!.V; i++) {
      const sp = figModel!.speciesOf[i];
      const cur = best.get(sp);
      if (!cur || figModel!.m[i] > cur.m) {
        best.set(sp, { species: sp, vocabIdx: i, m: figModel!.m[i] });
      }
    }
    return [...best.values()]
      .sort((a, b) => b.m - a.m)
      .slice(0, Math.min(TOP_SPECIES, best.size));
  }, [figReady, figModel]);

  // All species–species couplings among the top set, by |J| descending.
  // a, b are local indices into `reps`.
  const candidateEdges = useMemo(() => {
    if (!figReady || reps.length === 0) return [];
    const V = figModel!.V;
    const out: { a: number; b: number; J: number }[] = [];
    for (let x = 0; x < reps.length; x++) {
      for (let y = x + 1; y < reps.length; y++) {
        const w = figModel!.J[reps[x].vocabIdx * V + reps[y].vocabIdx];
        if (w !== 0) out.push({ a: x, b: y, J: w });
      }
    }
    out.sort((p, q) => Math.abs(q.J) - Math.abs(p.J));
    return out;
  }, [figReady, figModel, reps]);

  // "all" shows both signs (faithful to the model, but negative couplings
  // dominate); "positive" drops exclusions for the cleaner "works together" view.
  const [mode, setMode] = useState<"all" | "positive">("all");
  const shownCandidates = useMemo(
    () =>
      mode === "positive"
        ? candidateEdges.filter((e) => e.J > 0)
        : candidateEdges,
    [candidateEdges, mode],
  );

  // Threshold on |J|, in fixed units. Default 0 (show everything); the slider
  // range is hardcoded per mode (positive couplings top out lower than the
  // negative ones the "all" view includes). Reset to 0 when switching modes.
  const [threshold, setThreshold] = useState(0);
  useEffect(() => {
    setThreshold(0);
  }, [mode]);

  const thr = threshold;
  const sliderMax = mode === "positive" ? 1 : 3;

  // Visible edges + the nodes they touch, laid out fresh each threshold.
  const { nodes, edges } = useMemo(() => {
    if (reps.length === 0) return { nodes: [], edges: []};
    const visible = shownCandidates.filter((e) => Math.abs(e.J) >= thr);
    const used = [...new Set(visible.flatMap((e) => [e.a, e.b]))].sort(
      (a, b) => a - b,
    );
    const compact = new Map(used.map((local, k) => [local, k]));
    const layoutEdges = visible.map(
      (e) => [compact.get(e.a)!, compact.get(e.b)!, e.J] as const,
    );
    const g = makeGraph(
      used.length,
      layoutEdges,
      new Array(used.length).fill(0),
    );
    const pos = springLayout(g, new Rng(LAYOUT_SEED), LAYOUT_ITERS);
    const pts = pos.map((p) => ({
      x: VIEW_SIZE / 2 + VIEW_RADIUS * p.x,
      y: VIEW_SIZE / 2 + VIEW_RADIUS * p.y,
    }));
    relaxOverlaps(pts, SPRITE_SIZE * 0.9, 120, VIEW_SIZE, SPRITE_SIZE / 2);
    const nodes = used.map((local, k) => ({
      id: local,
      label: "",
      x: pts[k].x,
      y: pts[k].y,
      sprite: spriteUrl(figModel!.vocab[reps[local].vocabIdx]),
    }));
    const edges = visible.map((e) => ({ i: e.a, j: e.b, weight: e.J }));
    return { nodes, edges, hiddenCount: reps.length - used.length };
  }, [shownCandidates, thr, reps, figModel]);

  return (
    <section id="pokemon" className="lab-science-section">
      <h3>The same machinery, on Pokémon</h3>
      <p>
        Alright, it's finally time to actually talk about Pokémon! By now you
        should have a good understanding of the methods, so all that's left is to
        connect the dots. Starting with the data: thanks to the amazing people over
        at Limitless VGC, we have a huge dataset of real competitive teams that
        players have brought to tournaments. Limiting to the current regulation,
        doubles tournaments, and only events with at least 64 participants, we
        get about{" "}
        {nTeams !== null
          ? (Math.round(nTeams / 1000) * 1000).toLocaleString()
          : "13,000"}{" "}
        teams. These are the observations the model is fit to.
      </p>
      <p>
        In the SCOTUS example there were 9 spins to fit, one per justice. The
        current VGC regulation, Pokémon Champions Regulation M-A, has roughly
        200 species and formes and about 100 held items. Modeling species alone
        is around 200 spins; every Species @ Item combination is closer to 20,000. With
        one parameter per spin plus one per pair of spins (for the coupling), 
        that full model needs to fit about 200 million parameters (nearly a gigabyte), 
        all of which still has to run in the browser! So we keep only the spins
        that appear in 5 or more teams, bringing the Species @ Item model down to
        {" "}<strong>{nSpins !== null ? nSpins.toLocaleString() : "—"} spins</strong>{" "}
        and about{" "}
        <strong>
          {nParams !== null
            ? (Math.round(nParams / 10000) * 10000).toLocaleString()
            : "—"}{" "}
          parameters
        </strong>
        , about 1% of the full pairwise space. Much nicer.
      </p>
      <p>
        The largest difference between the Pokémon and SCOTUS/Ising cases is that
        in those earlier examples every combination of spins was valid. Any spin
        could be up or down regardless of the others, and any justice could vote
        liberal or conservative regardless of the others. In Pokémon that's not
        true: only six Pokémon can be on a team at once, and duplicate species or
        items are forbidden. These states aren't just unlikely, they're
        literally impossible. To enforce this, we make a small modification to
        the sampler that still produces the correct Boltzmann distribution over
        the <i>valid</i> states. Instead of considering one spin at a time and
        proposing to flip it, we start with six spins on and consider two spins
        at a time, one on and one off. The proposal is to <i>swap</i> their
        states, keeping exactly six on at all times. If a swap would violate the
        species or item constraints it is rejected outright (without this rule
        such states would rarely appear anyway since the model assigns them very
        high energy, but it's good to enforce it explicitly).
      </p>
      <p>
        The figure below shows how the couplings in the fitted model connect the
        top species together. Use the slider to hide the weaker couplings and
        watch the structure of the strongest relationships emerge.
      </p>
      <div className="lab-science-controls">
        <div className="lab-t-picker" role="radiogroup" aria-label="Couplings">
          <button
            type="button"
            role="radio"
            aria-checked={mode === "all"}
            className={"lab-t-btn" + (mode === "all" ? " is-selected" : "")}
            onClick={() => setMode("all")}
            disabled={!figReady}
          >
            <span className="lab-t-btn-label">All Couplings</span>
            <span className="lab-t-btn-hint">see what the model sees</span>
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={mode === "positive"}
            className={"lab-t-btn" + (mode === "positive" ? " is-selected" : "")}
            onClick={() => setMode("positive")}
            disabled={!figReady}
          >
            <span className="lab-t-btn-label">Positive couplings only</span>
            <span className="lab-t-btn-hint">visualize synergies</span>
          </button>
        </div>
        <label>
          Hide couplings below {thr.toFixed(2)}{" "}
          <input
            type="range"
            className="lab-slider"
            min={0}
            max={sliderMax}
            step={0.05}
            value={thr}
            onChange={(e) => setThreshold(Number(e.target.value))}
            disabled={!figReady}
          />
        </label>
      </div>
      <figure>
        {!figReady ? (
          <p style={{ color: "var(--lab-ink-muted)" }}>Loading the live model…</p>
        ) : nodes.length === 0 ? (
          <p style={{ color: "var(--lab-ink-muted)" }}>
            No couplings above {thr.toFixed(2)}. Lower the slider to bring
            species back.
          </p>
        ) : (
          <GraphView
            nodes={nodes}
            edges={edges}
            width={VIEW_SIZE}
            height={VIEW_SIZE}
            nodeRadius={NODE_RADIUS}
            maxStrokeWidth={6}
            showLabels={false}
            spriteOpacity={SPRITE_OPACITY}
          />
        )}
        {figReady && nodes.length > 0 && (
          <figcaption>
            The {reps.length} most-used species, connected by their pairwise
            coupling strengths from the fitted Species model <em>J</em>.{" "}
            {nodes.length} shown at this threshold
            . Blue = positive (co-occurs), red = negative (excludes); thickness
            ∝ strength.
          </figcaption>
        )}
      </figure>
      <p>
        And that's it! Hopefully this page has given you a feel for how k2dex
        builds and analyzes teams using statistical physics. Building this system
        and this page has been a ton of fun for me, and I hope you find it both
        interesting as an application of statistical physics and useful as a
        teambuilding resource!
      </p>
    </section>
  );
}
