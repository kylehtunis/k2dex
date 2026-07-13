// §02 of /meta: species-pair coupling table, ranked by APC-corrected
// Frobenius norm. Each row shows the signed synergy (species-level
// coupling) and can expand to show the top item-modulation entries for
// that pair, computed client-side from J.

import { useCallback, useMemo, useState } from "react";
import { ScoreChip, SignedBar } from "../render/atoms";
import { ScrollX } from "../components/ScrollX";
import { InlineMon } from "../render/cells";
import { topModulationEntries } from "./couplings";
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
  const modEntries = useMemo(() => {
    if (!isOpen) return [];
    const siteA = model.sites.indexOf(speciesA);
    const siteB = model.sites.indexOf(speciesB);
    if (siteA < 0 || siteB < 0) return [];
    return topModulationEntries(model, siteA, siteB, synergy);
  }, [isOpen, model, speciesA, speciesB, synergy]);

  const maxJ = useMemo(() => {
    if (modEntries.length === 0) return 1;
    let m = 0;
    for (const e of modEntries) {
      const a = Math.abs(e.jValue);
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
            <InlineMon name={speciesA} interactive={false} />
            <span className="lab-pair-sep">&times;</span>
            <InlineMon name={speciesB} interactive={false} />
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
              <table className="lab-modulation-table">
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
                        <div className="lab-coupling-val">
                          <SignedBar value={e.jValue} maxValue={maxJ} width={50} />
                          <ScoreChip value={e.jValue} />
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
