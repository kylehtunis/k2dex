// Factored roster editor for the completer. A grid of `teamSize` slots; each
// filled slot is a species picker plus one optional picker per attribute track
// (item, ability, …). Leaving a track unset makes the completer fill it (a site
// or partial pin); setting every track that narrows the species to a single
// feature makes a feature pin. Empty slots are filled entirely by the completer.
// "Unspecified" is the default (not an explicit "any" choice).

import { useEffect, useMemo, useRef } from "react";
import Select, { type SelectInstance, type SingleValue } from "react-select";
import { MENU_PORTAL_TARGET } from "./portalTarget";
import type { IsingModel } from "../sampler/types";
import type { RosterSlot } from "../state/PageStateContext";
import { emptySlot, setSlotTrack, slotFeature } from "../state/roster";
import { SpriteBox } from "../render/Sprite";

interface Opt {
  label: string;
  value: string;
}

const portalStyles = { menuPortal: (base: Record<string, unknown>) => ({ ...base, zIndex: 9999 }) };

/** Distinct values on track `t` among the slot's features that match every
 * *other* pinned track (a cascade, so a chosen combination always exists),
 * sorted by usage (summed marginal mass over the features holding each value,
 * descending) so popular items/abilities lead. A `null` value renders as an em
 * dash. */
function trackOptions(model: IsingModel, slot: RosterSlot, t: number): Opt[] {
  const massByValue = new Map<string | null, number>();
  const order: (string | null)[] = [];
  for (const f of model.siteFeatures[slot.site]) {
    let ok = true;
    for (let o = 0; o < slot.trackValues.length; o++) {
      if (o !== t && slot.trackValues[o] !== null && model.trackValues[f][o] !== slot.trackValues[o]) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    const v = model.trackValues[f][t];
    if (!massByValue.has(v)) order.push(v);
    massByValue.set(v, (massByValue.get(v) ?? 0) + model.m[f]);
  }
  return order
    .sort((a, b) => (massByValue.get(b) ?? 0) - (massByValue.get(a) ?? 0))
    .map((v) => ({ label: v ?? "—", value: v ?? "" }));
}

export function RosterEditor({
  model,
  roster,
  onChange,
  trackActive = () => true,
  teamSize = 6,
  itemPlaceholder = "item (optional)",
  emptyHint = "empty",
}: {
  model: IsingModel;
  roster: readonly RosterSlot[];
  onChange: (next: RosterSlot[]) => void;
  /** Whether a track's picker is shown/editable. The completer passes the
   * "not in Excluded attributes" predicate; analysis leaves every track active.
   * Track 0 (item) always shows when active (even for a single-item species);
   * later tracks (ability) show only when the species is non-degenerate. */
  trackActive?: (track: number) => boolean;
  teamSize?: number;
  /** Placeholder for the first (item) track picker. */
  itemPlaceholder?: string;
  /** Hint shown in trailing inert slots. `null` renders just the ordinal. */
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

  // Per-slot select instances so a species pick can jump the cursor to its first
  // track picker and a clear can send it back to the species picker.
  const speciesRefs = useRef<(SelectInstance<Opt> | null)[]>([]);
  const firstTrackRefs = useRef<(SelectInstance<Opt> | null)[]>([]);
  const pendingFocus = useRef<{ kind: "species" | "track"; slot: number } | null>(null);

  useEffect(() => {
    const p = pendingFocus.current;
    if (!p) return;
    pendingFocus.current = null;
    const inst = p.kind === "track" ? firstTrackRefs.current[p.slot] : speciesRefs.current[p.slot];
    inst?.focus();
  });

  // The visible track indices for a slot: every active track. A degenerate
  // track (one value for this species) still shows its picker — a one-option
  // dropdown — so the editor is consistent across species (some carry stray
  // extra ability values from teamsheet contamination, and hiding the picker
  // only for the clean ones was confusingly inconsistent).
  const visibleTracks = (): number[] => {
    const out: number[] = [];
    for (let t = 0; t < model.tracks.length; t++) {
      if (trackActive(t)) out.push(t);
    }
    return out;
  };

  const focusAfterSpecies = (slot: number) => {
    pendingFocus.current = visibleTracks().length > 0
      ? { kind: "track", slot }
      : slot + 1 < teamSize
        ? { kind: "species", slot: slot + 1 }
        : null;
  };

  const setSpecies = (slot: number, site: number | null) => {
    if (site === null) {
      pendingFocus.current = { kind: "species", slot };
      onChange(roster.filter((_, i) => i !== slot));
    } else {
      // New species → tracks reset to unset (values are species-specific).
      focusAfterSpecies(slot);
      onChange(roster.map((s, i) => (i === slot ? emptySlot(model, site) : s)));
    }
  };
  const setTrack = (slot: number, track: number, value: string | null) =>
    onChange(roster.map((s, i) => (i === slot ? setSlotTrack(s, track, value) : s)));
  const addSpecies = (site: number) => {
    focusAfterSpecies(roster.length);
    onChange([...roster, emptySlot(model, site)]);
  };

  const slots = [];
  for (let i = 0; i < teamSize; i++) {
    if (i < roster.length) {
      const slot = roster[i];
      const feat = slotFeature(model, slot);
      const spriteName = feat !== null ? model.vocab[feat] : model.sites[slot.site];
      const tracks = visibleTracks();
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
          {tracks.map((t, ti) => {
            const v = slot.trackValues[t];
            return (
              <Select
                key={`track-${t}`}
                ref={(el) => {
                  if (ti === 0) firstTrackRefs.current[i] = el;
                }}
                classNamePrefix="lab-select"
                className="lab-roster-item"
                options={trackOptions(model, slot, t)}
                value={v !== null ? { label: v, value: v } : null}
                onChange={(o: SingleValue<Opt>) => setTrack(i, t, o ? o.value : null)}
                isClearable
                openMenuOnFocus
                tabSelectsValue={false}
                placeholder={t === 0 ? itemPlaceholder : `${model.tracks[t].name} (optional)`}
                aria-label={`Slot ${i + 1} ${model.tracks[t].name}`}
                menuPortalTarget={MENU_PORTAL_TARGET}
                styles={portalStyles}
              />
            );
          })}
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
            aria-label={`Slot ${i + 1} — add a Pokémon`}
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
