// S7: bridge from toy to live model. Picker selects a species; renders
// that species' top ±10 couplings from the fitted Phase 3 J as an ego graph.
// Deep-links to /completer with that species pre-pinned.

import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { BlockMath } from "../widgets/Math";
import { GraphView } from "../widgets/GraphView";
import { useModel } from "../../state/ModelContext";

const STARTER_SPECIES = [
  "Calyrex-Shadow",
  "Calyrex-Ice",
  "Zacian",
  "Miraidon",
  "Koraidon",
  "Urshifu",
  "Incineroar",
  "Amoonguss",
  "Tornadus",
  "Rillaboom",
];

interface EgoEntry {
  partnerIdx: number;
  partnerLabel: string;
  J: number;
}

export function S7Pokemon() {
  const { model, phaseKey, setPhaseKey, status } = useModel();
  const [species, setSpecies] = useState(STARTER_SPECIES[0]);

  // Force Phase 3 on mount so the ego graph shows item-pair J.
  useEffect(() => {
    if (phaseKey !== "species_item") setPhaseKey("species_item");
  }, []);

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
    const cx = 240;
    const cy = 240;
    const r = 180;
    const out: { id: number; label: string; x: number; y: number; active?: boolean }[] =
      [{ id: -1, label: species.split("-")[0], x: cx, y: cy, active: true }];
    const n = ego.length;
    ego.forEach((e, k) => {
      const theta = (k / Math.max(n, 1)) * Math.PI * 2 - Math.PI / 2;
      const rawLabel = e.partnerLabel.split(" @ ")[0];
      out.push({
        id: e.partnerIdx,
        label: rawLabel.split("-")[0],
        x: cx + r * Math.cos(theta),
        y: cy + r * Math.sin(theta),
      });
    });
    return out;
  }, [ego, species]);

  const graphEdges = useMemo(
    () => ego.map((e) => ({ i: -1, j: e.partnerIdx, weight: e.J })),
    [ego],
  );

  return (
    <section id="s7-pokemon" className="lab-science-section">
      <h2>7. Same machinery, real Pokémon</h2>
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
        Pick a species below to see its top ±10 couplings from the live Phase 3 model. Then
        try the completer seeded with that pick.
      </p>
      <div className="lab-science-controls">
        <label>
          Species:{" "}
          <select value={species} onChange={(e) => setSpecies(e.target.value)}>
            {STARTER_SPECIES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <Link
          to={`/completer?pinned=${encodeURIComponent(species)}`}
          className="lab-button"
        >
          Try a team with this seed →
        </Link>
      </div>
      <figure>
        {status !== "ready" ? (
          <p style={{ color: "#888" }}>Loading the live Phase 3 model (~2.5 MB)…</p>
        ) : ego.length === 0 ? (
          <p style={{ color: "#888" }}>
            No (species, item) entries found for {species} in the current vocab.
          </p>
        ) : (
          <GraphView
            nodes={graphNodes as any}
            edges={graphEdges as any}
            width={480}
            height={480}
            nodeRadius={22}
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
