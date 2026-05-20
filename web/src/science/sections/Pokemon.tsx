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
      <h2>Same machinery, real Pokémon</h2>
      <p>
        The Pokémon completer uses the same pseudo-likelihood fit we just applied to SCOTUS,
        but on a binary indicator matrix where each column is a (species, held item) pair
        and each row is one team roster from the live tournament corpus. The fitted{" "}
        <em>J</em> has the same meaning: positive entries between features that appear
        together more than chance, negative entries between features that exclude each other.
      </p>
      <p>
        One adjustment: teams have a hard size constraint (six Pokémon, no duplicates), and
        flipping a single spin breaks it. The production sampler uses{" "}
        <strong>swap moves</strong> — turn one slot off, turn another on, in one atomic step.
        The acceptance rule is identical; only the proposal distribution changes:
      </p>
      <BlockMath formula="\Delta H = h_{i_\text{out}} - h_{i_\text{in}} + (J_{i_\text{out}} - J_{i_\text{in}}) \cdot s + J_{i_\text{in},\, i_\text{out}}" />
      <p>
        Pick a species below to see its top ±10 couplings from the live Phase 3 model.
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
    </section>
  );
}
