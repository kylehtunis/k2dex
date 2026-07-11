// §02 of /meta: species-pair coupling table, ranked by APC-corrected
// Frobenius norm. Each row shows the signed synergy (species-level
// coupling) and can expand to show the top item-modulation entries for
// that pair, computed client-side from J.

import { useCallback, useMemo, useState } from "react";
import { ScoreChip, SignedBar } from "../render/atoms";
import { SpriteBox } from "../render/Sprite";
import { ScrollX } from "../components/ScrollX";
import { InlineMon } from "../render/cells";
import type { IsingModel, SpeciesGraph } from "../sampler/types";

export interface SpeciesCouplingRow {
  /** Index into SpeciesGraph.species (alphabetical). */
  a: number;
  b: number;
  synergy: number;
  corrected: number;
}

export interface SpeciesCouplingsTableProps {
  rows: readonly SpeciesCouplingRow[];
  maxSynergy: number;
  graph: SpeciesGraph;
  model: IsingModel;
}

function SpeciesMon({ species }: { species: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <SpriteBox name={species} size={32} />
      <span className="lab-comp-mon-name">{species}</span>
    </span>
  );
}

interface ModulationEntry {
  featureA: number;
  featureB: number;
  jValue: number;
  deviation: number;
}

function topModulationEntries(
  model: IsingModel,
  speciesA: string,
  speciesB: string,
  synergy: number,
  topN = 8,
): ModulationEntry[] {
  const { siteFeatures, siteOf, J, V } = model;
  const idxA = model.sites.indexOf(speciesA);
  const idxB = model.sites.indexOf(speciesB);
  if (idxA < 0 || idxB < 0) return [];
  const featA = siteFeatures[idxA];
  const featB = siteFeatures[idxB];
  // Only compute for same-site pairs once (skip if same site — shouldn't happen).
  if (idxA === idxB) return [];
  const entries: ModulationEntry[] = [];
  for (const fa of featA) {
    for (const fb of featB) {
      if (siteOf[fa] === siteOf[fb]) continue;
      const itA = model.itemOf[fa];
      const itB = model.itemOf[fb];
      if (itA !== null && itB !== null && itA === itB) continue;
      const jValue = J[fa * V + fb];
      entries.push({ featureA: fa, featureB: fb, jValue, deviation: jValue - synergy });
    }
  }
  entries.sort((a, b) => Math.abs(b.deviation) - Math.abs(a.deviation));
  return entries.slice(0, topN);
}

export function SpeciesCouplingsTable({
  rows,
  maxSynergy,
  graph,
  model,
}: SpeciesCouplingsTableProps) {
  const [expanded, setExpanded] = useState<string | null>(null);

  const toggleExpand = useCallback((key: string) => {
    setExpanded((prev) => (prev === key ? null : key));
  }, []);

  return (
    <ScrollX>
    <table className="lab-comp-table lab-table-pairs">
      <thead>
        <tr>
          <th className="num">#</th>
          <th>pair</th>
          <th className="num">Synergy</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, rank) => {
          const specA = graph.species[r.a];
          const specB = graph.species[r.b];
          const key = `${r.a}-${r.b}`;
          const isOpen = expanded === key;
          return (
            <ExpandableRow
              key={key}
              rank={rank}
              speciesA={specA}
              speciesB={specB}
              synergy={r.synergy}
              maxSynergy={maxSynergy}
              isOpen={isOpen}
              onToggle={() => toggleExpand(key)}
              model={model}
            />
          );
        })}
      </tbody>
    </table>
    </ScrollX>
  );
}

function ExpandableRow({
  rank,
  speciesA,
  speciesB,
  synergy,
  maxSynergy,
  isOpen,
  onToggle,
  model,
}: {
  rank: number;
  speciesA: string;
  speciesB: string;
  synergy: number;
  maxSynergy: number;
  isOpen: boolean;
  onToggle: () => void;
  model: IsingModel;
}) {
  const modEntries = useMemo(
    () => (isOpen ? topModulationEntries(model, speciesA, speciesB, synergy) : []),
    [isOpen, model, speciesA, speciesB, synergy],
  );

  const maxDev = useMemo(() => {
    if (modEntries.length === 0) return 1;
    let m = 0;
    for (const e of modEntries) {
      const a = Math.abs(e.deviation);
      if (a > m) m = a;
    }
    return m || 1;
  }, [modEntries]);

  return (
    <>
      <tr
        className={`lab-species-coupling-row${isOpen ? " lab-expanded" : ""}`}
        onClick={onToggle}
        style={{ cursor: "pointer" }}
      >
        <td className="rank">{(rank + 1).toString().padStart(2, "0")}</td>
        <td className="pair">
          <div className="lab-pair-cell">
            <SpeciesMon species={speciesA} />
            <span className="lab-pair-sep">&times;</span>
            <SpeciesMon species={speciesB} />
          </div>
        </td>
        <td className="num" data-label="synergy">
          <div className="lab-coupling-val">
            <SignedBar value={synergy} maxValue={maxSynergy} width={80} />
            <ScoreChip value={synergy} />
          </div>
        </td>
      </tr>
      {isOpen && modEntries.length > 0 && (
        <tr className="lab-modulation-detail">
          <td colSpan={3}>
            <div className="lab-modulation-list">
              <div className="lab-modulation-header">Item modulation (deviation from base synergy)</div>
              <table className="lab-modulation-table">
                <thead>
                  <tr>
                    <th>item pair</th>
                    <th className="num">J</th>
                    <th className="num">deviation</th>
                  </tr>
                </thead>
                <tbody>
                  {modEntries.map((e) => (
                    <tr key={`${e.featureA}-${e.featureB}`}>
                      <td className="pair">
                        <div className="lab-pair-cell lab-pair-cell-compact">
                          <InlineMon name={model.vocab[e.featureA]} size={24} />
                          <span className="lab-pair-sep">&times;</span>
                          <InlineMon name={model.vocab[e.featureB]} size={24} />
                        </div>
                      </td>
                      <td className="num">
                        <ScoreChip value={e.jValue} />
                      </td>
                      <td className="num">
                        <div className="lab-coupling-val">
                          <SignedBar value={e.deviation} maxValue={maxDev} width={50} />
                          <ScoreChip value={e.deviation} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
