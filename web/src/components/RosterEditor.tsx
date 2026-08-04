// Factored roster editor for the completer. A grid of `teamSize` slots; each
// filled slot is a species picker + an optional item picker. Leaving a slot's
// item unset makes it a site pin (species locked, item filled by the completer);
// choosing an item makes it a feature pin. Empty slots are filled entirely by
// the completer. This is the factored counterpart to the flat feature list —
// species and item are separate dimensions, and "unspecified" is the default
// (not an explicit "any item" choice).

import { useEffect, useMemo, useRef } from "react";
import Select, { type SelectInstance, type SingleValue } from "react-select";
import { MENU_PORTAL_TARGET } from "./portalTarget";
import type { IsingModel } from "../sampler/types";
import type { RosterSlot } from "../state/PageStateContext";
import { SpriteBox } from "../render/Sprite";

interface Opt {
  label: string;
  value: string;
}

const portalStyles = { menuPortal: (base: Record<string, unknown>) => ({ ...base, zIndex: 9999 }) };

/** Item label for a feature: its item-track value verbatim (e.g. "Life Orb",
 * "None") — treated as an ordinary item string, no special-casing. */
function itemLabel(model: IsingModel, feature: number): string {
  const v = model.trackValues[feature][0];
  return v ?? "—";
}

export function RosterEditor({
  model,
  roster,
  onChange,
  itemActive,
  teamSize = 6,
  itemPlaceholder = "item (optional)",
  emptyHint = "empty",
}: {
  model: IsingModel;
  roster: readonly RosterSlot[];
  onChange: (next: RosterSlot[]) => void;
  itemActive: boolean;
  teamSize?: number;
  /** Placeholder for the item picker. The completer says the completer fills
   * an unset item; analysis (feature-level) leaves it plain. */
  itemPlaceholder?: string;
  /** Hint shown in trailing inert slots. `null` renders just the ordinal —
   * used on analysis, where an empty slot isn't "filled by" anything. */
  emptyHint?: string | null;
}) {
  const sitePop = useMemo(() => {
    const pop = new Float64Array(model.sites.length);
    for (let i = 0; i < model.V; i++) pop[model.siteOf[i]] += model.m[i];
    return pop;
  }, [model]);

  const usedKey = roster.map((s) => s.site).join(",");
  const speciesOptions = useMemo(() => {
    const used = new Set(roster.map((s) => s.site));
    return Array.from({ length: model.sites.length }, (_, s) => s)
      .filter((s) => !used.has(s))
      .sort((a, b) => sitePop[b] - sitePop[a])
      .map((s) => ({ label: model.sites[s], value: `${s}` }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, sitePop, usedKey]);

  // Per-slot select instances, so a species pick can jump the cursor to its
  // item picker and a clear can send it back to the species picker. Populated
  // by ref callbacks; entries null out on unmount.
  const speciesRefs = useRef<(SelectInstance<Opt> | null)[]>([]);
  const itemRefs = useRef<(SelectInstance<Opt> | null)[]>([]);
  const pendingFocus = useRef<{ kind: "species" | "item"; slot: number } | null>(null);

  // After a roster change re-renders, apply the queued focus. The target
  // mounts in the same commit, so its ref is set before this effect runs;
  // attempt once and clear so a stale target can't grab focus on a later
  // unrelated render.
  useEffect(() => {
    const p = pendingFocus.current;
    if (!p) return;
    pendingFocus.current = null;
    const inst = p.kind === "item" ? itemRefs.current[p.slot] : speciesRefs.current[p.slot];
    inst?.focus();
  });

  // After choosing a species, jump to that slot's item picker; with no item
  // track, jump to the next species picker (the next mon / the add slot).
  const focusAfterSpecies = (slot: number) => {
    pendingFocus.current = itemActive
      ? { kind: "item", slot }
      : slot + 1 < teamSize
        ? { kind: "species", slot: slot + 1 }
        : null;
  };

  const setSpecies = (slot: number, site: number | null) => {
    if (site === null) {
      // Clearing a slot removes the mon; send focus to the species picker that
      // now occupies this index (next mon shifted up, or the add slot).
      pendingFocus.current = { kind: "species", slot };
      onChange(roster.filter((_, i) => i !== slot));
    } else {
      // New species → item resets to unset (items are species-specific).
      focusAfterSpecies(slot);
      onChange(roster.map((s, i) => (i === slot ? { site, feature: null } : s)));
    }
  };
  const setItem = (slot: number, feature: number | null) =>
    onChange(roster.map((s, i) => (i === slot ? { ...s, feature } : s)));
  const addSpecies = (site: number) => {
    focusAfterSpecies(roster.length);
    onChange([...roster, { site, feature: null }]);
  };

  const slots = [];
  for (let i = 0; i < teamSize; i++) {
    if (i < roster.length) {
      const slot = roster[i];
      const spriteName =
        slot.feature !== null ? model.vocab[slot.feature] : model.sites[slot.site];
      const itemOptions: Opt[] = model.siteFeatures[slot.site]
        .slice()
        .sort((a, b) => model.m[b] - model.m[a])
        .map((f) => ({
          label: itemLabel(model, f),
          value: `${f}`,
        }));
      const itemValue: Opt | null =
        slot.feature !== null
          ? { label: itemLabel(model, slot.feature), value: `${slot.feature}` }
          : null;
      slots.push(
        <div className="lab-roster-slot" key={`slot-${i}`}>
          <SpriteBox name={spriteName} size={56} className="lab-roster-slot-sprite" />
          <Select
            ref={(el) => {
              speciesRefs.current[i] = el;
            }}
            classNamePrefix="lab-select"
            className="lab-roster-species"
            options={speciesOptions}
            value={{ label: model.sites[slot.site], value: `${slot.site}` }}
            onChange={(o: SingleValue<Opt>) => setSpecies(i, o ? Number(o.value) : null)}
            isClearable
            openMenuOnFocus
            tabSelectsValue={false}
            placeholder="Pokémon"
            aria-label={`Slot ${i + 1} Pokémon`}
            menuPortalTarget={MENU_PORTAL_TARGET}
            styles={portalStyles}
          />
          {itemActive && (
            <Select
              ref={(el) => {
                itemRefs.current[i] = el;
              }}
              classNamePrefix="lab-select"
              className="lab-roster-item"
              options={itemOptions}
              value={itemValue}
              onChange={(o: SingleValue<Opt>) => setItem(i, o ? Number(o.value) : null)}
              isClearable
              openMenuOnFocus
              tabSelectsValue={false}
              placeholder={itemPlaceholder}
              aria-label={`Slot ${i + 1} item`}
              menuPortalTarget={MENU_PORTAL_TARGET}
              styles={portalStyles}
            />
          )}
        </div>,
      );
    } else if (i === roster.length) {
      slots.push(
        <div className="lab-roster-slot lab-roster-slot-empty" key={`add-${i}`}>
          <div className="lab-roster-slot-ord">·{i + 1}·</div>
          <Select
            ref={(el) => {
              speciesRefs.current[i] = el;
            }}
            classNamePrefix="lab-select"
            className="lab-roster-species"
            options={speciesOptions}
            value={null}
            onChange={(o: SingleValue<Opt>) => o && addSpecies(Number(o.value))}
            placeholder="add a Pokémon"
            aria-label={`Slot ${i + 1}, add a Pokémon`}
            menuPortalTarget={MENU_PORTAL_TARGET}
            styles={portalStyles}
          />
        </div>,
      );
    } else {
      slots.push(
        <div className="lab-roster-slot lab-roster-slot-empty lab-roster-slot-inert" key={`empty-${i}`}>
          <div className="lab-roster-slot-ord">·{i + 1}·</div>
          {emptyHint && <div className="lab-roster-slot-hint">{emptyHint}</div>}
        </div>,
      );
    }
  }

  return <div className="lab-roster-grid">{slots}</div>;
}
