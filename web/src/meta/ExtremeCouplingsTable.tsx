// §02 of /meta: species-pair coupling table, ranked by APC-corrected
// Frobenius norm. Each row shows the signed synergy (species-level coupling)
// and expands into the hierarchical drill-down computed client-side from J:
// species synergy -> item modulation (every other attribute marginalized out)
// -> optionally one more tier for the selected attribute, nested under an item
// pair. The attribute of interest is chosen by a radio inside the drill-down;
// its state lives in MetaPage so it survives collapsing one row and opening
// another.

import { useCallback, useEffect, useId, useMemo, useState } from "react";
import { ScoreChip, SignedBar } from "../render/atoms";
import { ScrollX } from "../components/ScrollX";
import { InlineMon } from "../render/cells";
import { SpriteBox } from "../render/Sprite";
import { META_SUPPORT_MIN_COUNT } from "../constants";
import { hasAttributeSplit, topAttributeEntries, topModulationEntries } from "./couplings";
import type { IsingModel, SpeciesGraph } from "../sampler/types";

/** Track index the drill-down breaks down by. 0 (item) is the base tier and
 * adds no nesting; any later track nests inside each item pair. */
export type MetaAttribute = number;

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
  attribute: MetaAttribute;
  onAttributeChange: (attribute: MetaAttribute) => void;
}

export function SpeciesCouplingsTable({
  rows,
  maxSynergy,
  graph,
  model,
  attribute,
  onAttributeChange,
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
              attribute={attribute}
              onAttributeChange={onAttributeChange}
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
  attribute,
  onAttributeChange,
}: {
  rank: number;
  speciesA: string;
  speciesB: string;
  synergy: number;
  maxSynergy: number;
  isOpen: boolean;
  onToggle: () => void;
  model: IsingModel;
  attribute: MetaAttribute;
  onAttributeChange: (attribute: MetaAttribute) => void;
}) {
  const sites = useMemo(() => {
    const siteA = model.sites.indexOf(speciesA);
    const siteB = model.sites.indexOf(speciesB);
    return siteA < 0 || siteB < 0 ? null : { siteA, siteB };
  }, [model, speciesA, speciesB]);

  const modEntries = useMemo(() => {
    if (!isOpen || !sites) return [];
    return topModulationEntries(model, sites.siteA, sites.siteB, synergy);
  }, [isOpen, model, sites, synergy]);

  // The attribute tier is only offered when the residual can be non-zero: at
  // least one of the two species carries more than one value on that track
  // inside some item build.
  const attrSplit = useMemo(() => {
    if (!isOpen || !sites || attribute === 0) return false;
    return (
      hasAttributeSplit(model, sites.siteA, attribute) ||
      hasAttributeSplit(model, sites.siteB, attribute)
    );
  }, [isOpen, model, sites, attribute]);

  const maxJ = useMemo(() => {
    let m = 0;
    for (const e of modEntries) {
      const a = Math.abs(e.jValue);
      if (a > m) m = a;
    }
    return m || 1;
  }, [modEntries]);

  const [openItemPair, setOpenItemPair] = useState<string | null>(null);

  // Selecting an attribute opens the top-ranked item pairing straight away, so
  // the choice visibly changes the drill-down instead of only arming a second
  // click. Switching back to item (or to a pair with nothing to split) closes.
  const entryKey = (e: (typeof modEntries)[number]) => `${e.featureA}-${e.featureB}`;
  const firstKey = modEntries.length > 0 ? entryKey(modEntries[0]) : null;
  useEffect(() => {
    setOpenItemPair(attrSplit ? firstKey : null);
  }, [attrSplit, firstKey]);

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
              <AttributePicker
                model={model}
                attribute={attribute}
                onChange={onAttributeChange}
              />
              <table className="lab-modulation-table">
                <tbody>
                  {modEntries.map((e) => {
                    const key = entryKey(e);
                    const drillable = attrSplit;
                    const nestedOpen = drillable && openItemPair === key;
                    return (
                      <ItemPairRow
                        key={key}
                        model={model}
                        entry={e}
                        maxJ={maxJ}
                        drillable={drillable}
                        isOpen={nestedOpen}
                        onToggle={() =>
                          setOpenItemPair((prev) => (prev === key ? null : key))
                        }
                        attribute={attribute}
                      />
                    );
                  })}
                </tbody>
              </table>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

/** One item-pair row, plus its nested attribute-residual tier when open. */
function ItemPairRow({
  model,
  entry,
  maxJ,
  drillable,
  isOpen,
  onToggle,
  attribute,
}: {
  model: IsingModel;
  entry: ReturnType<typeof topModulationEntries>[number];
  maxJ: number;
  drillable: boolean;
  isOpen: boolean;
  onToggle: () => void;
  attribute: MetaAttribute;
}) {
  const attrEntries = useMemo(() => {
    if (!isOpen) return [];
    return topAttributeEntries(
      model,
      entry.groupA,
      entry.groupB,
      entry.jValue,
      attribute,
      META_SUPPORT_MIN_COUNT,
    );
  }, [isOpen, model, entry, attribute]);

  const maxDev = useMemo(() => {
    let m = 0;
    for (const e of attrEntries) {
      const a = Math.abs(e.deviation);
      if (a > m) m = a;
    }
    return m || 1;
  }, [attrEntries]);

  return (
    <>
      <tr
        className={`lab-mod-pair-row${drillable ? " lab-mod-drillable" : ""}${
          isOpen ? " lab-expanded" : ""
        }`}
        onClick={drillable ? onToggle : undefined}
        style={drillable ? { cursor: "pointer" } : undefined}
      >
        <td className="lab-mod-caret" aria-hidden="true">
          {drillable ? (isOpen ? "▾" : "▸") : ""}
        </td>
        <td className="pair">
          <div className="lab-pair-cell lab-pair-cell-compact">
            <InlineMon name={model.vocab[entry.featureA]} size={24} />
            <span className="lab-pair-sep">&times;</span>
            <InlineMon name={model.vocab[entry.featureB]} size={24} />
          </div>
        </td>
        <td className="num">
          <div className="lab-coupling-val">
            <SignedBar value={entry.jValue} maxValue={maxJ} width={50} />
            <ScoreChip value={entry.jValue} />
          </div>
        </td>
      </tr>
      {isOpen && (
        <tr className="lab-attr-detail">
          <td colSpan={3}>
            {attrEntries.length === 0 ? (
              <div className="lab-attr-empty">
                Not enough corpus support to split this pairing by{" "}
                {model.tracks[attribute].name}.
              </div>
            ) : (
              <table className="lab-attr-table">
                <tbody>
                  {attrEntries.map((e) => (
                    <tr key={`${e.featureA}-${e.featureB}`}>
                      <td className="pair">
                        <div className="lab-mod-pair">
                          <AttrCell model={model} feature={e.featureA} track={attribute} />
                          <span className="lab-pair-sep">&times;</span>
                          <AttrCell model={model} feature={e.featureB} track={attribute} />
                        </div>
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
            )}
          </td>
        </tr>
      )}
    </>
  );
}

/** Sprite + the feature's value on `track`. The species and item are fixed for
 * the whole nested tier, so only the attribute value varies per row. */
function AttrCell({
  model,
  feature,
  track,
}: {
  model: IsingModel;
  feature: number;
  track: number;
}) {
  return (
    <span className="lab-mod-item">
      <SpriteBox name={model.vocab[feature]} size={22} />
      <span className="lab-mod-item-label">
        {model.trackValues[feature][track] ?? "—"}
      </span>
    </span>
  );
}

/** Attribute-of-interest radio. Rendered inside the drill-down (it only means
 * anything there), but its state is owned by the page so both the positive and
 * negative tables agree and the choice survives switching rows. */
function AttributePicker({
  model,
  attribute,
  onChange,
}: {
  model: IsingModel;
  attribute: MetaAttribute;
  onChange: (attribute: MetaAttribute) => void;
}) {
  // Every mounted picker needs its own radio-group name: two open drill-downs
  // (one per table) sharing a name makes the browser treat them as one group,
  // so checking either visually unchecks the other even though both render the
  // same controlled value.
  const groupName = useId();
  if (model.tracks.length < 2) return null;
  return (
    <div className="lab-attr-picker" onClick={(e) => e.stopPropagation()}>
      <span className="lab-attr-picker-label">break down by</span>
      {model.tracks.map((t, i) => (
        <label key={t.name} className="lab-attr-option">
          <input
            type="radio"
            name={groupName}
            checked={attribute === i}
            onChange={() => onChange(i)}
          />
          <span>{t.name}</span>
        </label>
      ))}
    </div>
  );
}
