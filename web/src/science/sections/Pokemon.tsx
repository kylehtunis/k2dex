// Closing section: bridges the toy/SCOTUS machinery to the real model.
// The figure is a force-directed graph of the top species, drawn from the
// active model's J. A slider hides couplings below a threshold and
// recomputes the spring layout, so the strongest structure emerges as you
// raise it. Not animated, just layout. The prose's spin/parameter counts
// read from the active model. The graph itself (reps, layout, render) is
// shared with the Metagame Model page via CouplingGraph.tsx.

import { useCallback, useEffect, useState } from "react";
import { useModel } from "../../state/ModelContext";
import { CouplingGraph, type CouplingEdge } from "../../components/CouplingGraph";

const TOP_SPECIES = 32;
const VIEW_SIZE = 600;

export function Pokemon() {
  const { model, status } = useModel();

  const ready = model !== null && status === "ready";

  const nSpins = ready ? model!.V : null;
  const nParams = nSpins !== null ? nSpins + (nSpins * (nSpins - 1)) / 2 : null;
  const nTeams = ready ? model!.nCorpusTeams : null;
  const regulation = ready ? model!.regulation : null;

  const [mode, setMode] = useState<"all" | "positive">("all");

  const [threshold, setThreshold] = useState(0);
  useEffect(() => {
    setThreshold(0);
  }, [mode]);

  const sliderMax = mode === "positive" ? 1 : 3;

  const filterEdge = useCallback(
    (e: CouplingEdge) => {
      if (mode === "positive" && e.J <= 0) return false;
      return Math.abs(e.J) >= threshold;
    },
    [mode, threshold],
  );

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
        In the SCOTUS example there were 9 spins to fit, one per justice.
        {regulation && <> The current VGC regulation, Pokémon Champions Regulation {regulation}, has</>}
        {!regulation && <> The current VGC regulation has</>}
        {" "}roughly 200 species and formes and about 100 held items. Modeling species alone
        is around 200 spins; every Species @ Item combination is closer to 20,000. With
        one parameter per spin plus one per pair of spins (for the coupling),
        that full model needs to fit about 200 million parameters (nearly a gigabyte),
        all of which still has to run in the browser! So we keep only the spins
        that appear in 5 or more teams, bringing the model down to
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
            disabled={!ready}
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
            disabled={!ready}
          >
            <span className="lab-t-btn-label">Positive couplings only</span>
            <span className="lab-t-btn-hint">visualize synergies</span>
          </button>
        </div>
        <label>
          Hide couplings below {threshold.toFixed(2)}{" "}
          <input
            type="range"
            className="lab-slider"
            min={0}
            max={sliderMax}
            step={0.05}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            disabled={!ready}
          />
        </label>
      </div>
      {!ready ? (
        <p style={{ color: "var(--lab-ink-muted)" }}>Loading the live model…</p>
      ) : (
        <CouplingGraph
          model={model!}
          filterEdge={filterEdge}
          topSpecies={TOP_SPECIES}
          viewSize={VIEW_SIZE}
          renderCaption={({ reps, visibleNodes }) => (
            <>
              The {reps.length} most-used species, connected by their pairwise
              coupling strengths from the fitted model <em>J</em>.{" "}
              {visibleNodes} shown at this threshold. Blue = positive
              (co-occurs), red = negative (excludes); thickness ∝ strength.
            </>
          )}
        />
      )}
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
