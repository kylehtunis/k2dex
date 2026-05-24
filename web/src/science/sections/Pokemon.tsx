// Bridge from toy to live model. Picker selects a species; renders
// that species' top ±10 couplings from the fitted Phase 3 J as an ego graph.
// Deep-links to /completer with that species pre-pinned.

import { useEffect, useMemo, useState } from "react";
import Select, { type SingleValue } from "react-select";
import { BlockMath } from "../widgets/Math";
import { GraphView } from "../widgets/GraphView";
import { useModel } from "../../state/ModelContext";
import { speciesOptions } from "../../components/VocabSelect";
import { spriteUrl } from "../../render/sprite-url";
import { extractSpecies } from "../../render/format";

interface EgoEntry {
  partnerIdx: number;
  partnerLabel: string;
  J: number;
}

export function Pokemon() {
  const { model, phaseKey, setPhaseKey, status } = useModel();
  const [species, setSpecies] = useState<string>("");

  // Force Phase 3 on mount so the ego graph shows item-pair J.
  useEffect(() => {
    if (phaseKey !== "species_item") setPhaseKey("species_item");
  }, []);

  const speciesOpts = useMemo(
    () => (model ? speciesOptions(model) : []),
    [model],
  );

  // Default to the most popular species once the model is loaded.
  useEffect(() => {
    if (!species && speciesOpts.length > 0) {
      setSpecies(speciesOpts[0].value);
    }
  }, [speciesOpts, species]);

  const ego: EgoEntry[] = useMemo(() => {
    if (!model || status !== "ready") return [];
    const V = model.V;
    // Pick the vocab entry for this species with the highest marginal.
    let seedIdx = -1;
    let seedM = -1;
    for (let i = 0; i < V; i++) {
      if (model.speciesOf[i] === species && model.m[i] > seedM) {
        seedM = model.m[i];
        seedIdx = i;
      }
    }
    if (seedIdx === -1) return [];
    // Build J row, skip same-species entries, sort by |J|.
    const entries: EgoEntry[] = [];
    for (let j = 0; j < V; j++) {
      if (j === seedIdx) continue;
      if (model.speciesOf[j] === species) continue;
      const w = model.J[seedIdx * V + j];
      if (w === 0) continue;
      entries.push({
        partnerIdx: j,
        partnerLabel: model.vocab[j],
        J: w,
      });
    }
    entries.sort((a, b) => Math.abs(b.J) - Math.abs(a.J));
    return entries.slice(0, 20);
  }, [model, status, species]);

  // Circular layout: seed in center, partners around the rim.
  const graphNodes = useMemo(() => {
    const cx = 270;
    const cy = 270;
    const r = 210;
    const out: {
      id: number;
      label: string;
      x: number;
      y: number;
      active?: boolean;
      sprite?: string;
    }[] = [
      {
        id: -1,
        label: species,
        x: cx,
        y: cy,
        active: true,
        sprite: spriteUrl(species),
      },
    ];
    const n = ego.length;
    ego.forEach((e, k) => {
      const theta = (k / Math.max(n, 1)) * Math.PI * 2 - Math.PI / 2;
      const partnerSpecies = extractSpecies(e.partnerLabel);
      out.push({
        id: e.partnerIdx,
        label: partnerSpecies,
        x: cx + r * Math.cos(theta),
        y: cy + r * Math.sin(theta),
        sprite: spriteUrl(e.partnerLabel),
      });
    });
    return out;
  }, [ego, species]);

  const graphEdges = useMemo(
    () => ego.map((e) => ({ i: -1, j: e.partnerIdx, weight: e.J })),
    [ego],
  );

  return (
    <section id="pokemon" className="lab-science-section">
      <h2>Same machinery, but Pokemon</h2>
      <p>
        Alright, now it's finally time to actually talk about Pokemon.
        By now you should have a good understanding of the methods I'm using, so all that's left is to connect the dots.
        Starting from the data: thanks to the amazing people over at Limitless VGC, we have a huge dataset of actual competitive teams that have been brought to real tournaments.
        Limiting to the current regulation, doubles tournaments, and only tournaments with &gt=64 participants, we get about 14,000 total teams.
        These become the obersvations that our model tries to fit. 
      </p>
      <p>
        In the SCOTUS example, there were 9 spins we had to fit (one per justice).
        In the current VGC regulation as of the time of writing this, Pokemon Champions Regulation M-A, there are about 200 different species and formes and aboutn 100 held items.
        If we model all of them, that would be 200 spins that would need to be fit for the Species corpus and 20,000 for the Species @ Item corpus. 
        The model requires us to fit one parameter per spin and one parameter per combination of spins, so a model with all species-item pairs would have on the order of 400 million parameters (~1.6 GB) <i>and</i> still need to run in the browser!
        To reduce that number, we'll consider only spins that appear 5 or more times in the tournament data, which gets us down to xxx spins and xxx parameters for the Species @ Item model.
        Much nicer.
      </p>
      <p>
        The largest difference between the Pokemon and SCOTUS/Ising case is that for those earlier examples, all spin combinations were valid.
        Each individual spin could possibly be up or down regardless of the others, and any justice could vote liberal or conservative regardless of the others.
        In Pokemon, that's not the case: only six Pokemon can be on a team at once, and duplicate species or items are forbidden.
        These states aren't just unlikley, they are literally impossible.
        To enforce these constraints, we make a small modification to the sampling algorithm that still results in the proper Boltzmann distribution over <i>valid</i> states.
        Instead of considering one spin at a time and proposing to flip it, we start with six spins On and consider two spins at a time, one On and one Off.
        The proposal is to <i>swap</i> their states, maintaining exactly six On at all times.
        If the proposed swap would violate the species or item constraints, it is automatically rejected 
        (without this rule, such states would be quite unlikely to show up in results since the model would assign them very high energy, but it's still good to enforce it explicitly).
      </p>
      <p>
        The widget below shows how the couplings in the fitted model connect the top species together. 
        Use the slider to hide weaker couplings to see the structure of the strongest relationships start to emerge.
      </p>
      <div className="lab-science-controls">
        <label className="lab-pokemon-species-label">
          <span>Species:</span>
          <div className="lab-pokemon-species-select">
            <Select
              classNamePrefix="lab-select"
              options={speciesOpts}
              value={
                species
                  ? speciesOpts.find((o) => o.value === species) ?? null
                  : null
              }
              onChange={(sel: SingleValue<{ label: string; value: string }>) =>
                setSpecies(sel?.value ?? "")
              }
              isClearable={false}
              placeholder="Pick a species…"
              menuPortalTarget={document.body}
              styles={{ menuPortal: (base) => ({ ...base, zIndex: 9999 }) }}
              />
          </div>
        </label>
      </div>
      <figure>
        {status !== "ready" ? (
          <p style={{ color: "#888" }}>Loading the live Phase 3 model (~2.5 MB)…</p>
        ) : !species ? (
          <p style={{ color: "#888" }}>Pick a species above.</p>
        ) : ego.length === 0 ? (
          <p style={{ color: "#888" }}>
            No (species, item) entries found for {species} in the current vocab.
          </p>
        ) : (
          <GraphView
          nodes={graphNodes as any}
          edges={graphEdges as any}
          width={540}
          height={540}
          nodeRadius={26}
          />
        )}
        <figcaption>
          Top ±{Math.min(ego.length, 20)} couplings for {species} from the fitted Phase 3{" "}
          <em>J</em>. Blue = positive (co-occurs), red = negative (excludes).
        </figcaption>
      </figure>
      <p>
        And that's it! Hopefully this page has given you a good understanding of how k2dex is able to build and analyze teams using statistical physics.
        Building this system and this page has been a ton of fun for me, and I hope you find it both useful as a player and interesting as an application of scientific theory to competitive Pokemon.
      </p>
    </section>
  );
}
