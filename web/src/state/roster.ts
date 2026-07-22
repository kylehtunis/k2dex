// Roster-slot helpers shared by the completer, analysis, editor, and share
// links. A RosterSlot holds a species (site) plus a per-track pinned value
// (string) or null = free. The completer resolves each slot to either a fully
// determined feature pin or a site/partial pin the sampler fills.
//
// Single-track invariant: for a model whose only track is "item", these reduce
// to the old {site, feature} behaviour — a slot with the item chosen resolves
// to that unique feature; a slot with the item free resolves to a site pin.

import type { IsingModel } from "../sampler/types";
import type { RosterSlot } from "./PageStateContext";
import { itemToSlug, abilityToSlug } from "../render/sprite-url";
import { resolveSpeciesSlug, type SlugIndex } from "../render/vocab-match";

/** A fresh slot for `site` with every track free. */
export function emptySlot(model: IsingModel, site: number): RosterSlot {
  return { site, trackValues: model.tracks.map(() => null) };
}

/** A slot fully pinned to feature `f` (every track set to `f`'s values). */
export function slotFromFeature(model: IsingModel, f: number): RosterSlot {
  return { site: model.siteOf[f], trackValues: model.trackValues[f].slice() };
}

/** The site's features matching every pinned track value (free tracks impose
 * nothing). */
export function matchingFeatures(model: IsingModel, slot: RosterSlot): number[] {
  return model.siteFeatures[slot.site].filter((f) =>
    slot.trackValues.every((v, t) => v === null || model.trackValues[f][t] === v),
  );
}

/** The unique feature a slot resolves to, or null when 0 or >1 features match —
 * i.e. the slot is a site pin (all free) or a partial pin (some tracks free,
 * still ambiguous) the sampler must fill. A pinned track that already narrows
 * the species to a single feature (e.g. an item whose sole ability is implied)
 * counts as fully determined. */
export function slotFeature(model: IsingModel, slot: RosterSlot): number | null {
  const ms = matchingFeatures(model, slot);
  return ms.length === 1 ? ms[0] : null;
}

/** Whether a slot resolves to a concrete feature (a completed team member). */
export function slotComplete(model: IsingModel, slot: RosterSlot): boolean {
  return slotFeature(model, slot) !== null;
}

/** Clear a track's pinned value on every slot (used when a track is
 * deactivated: its value is hidden and filled by the completer). */
export function clearTrack(roster: readonly RosterSlot[], track: number): RosterSlot[] {
  return roster.map((s) => ({
    site: s.site,
    trackValues: s.trackValues.map((v, t) => (t === track ? null : v)),
  }));
}

/** Set one track's pinned value on a slot (null = free). */
export function setSlotTrack(slot: RosterSlot, track: number, value: string | null): RosterSlot {
  return { site: slot.site, trackValues: slot.trackValues.map((v, t) => (t === track ? value : v)) };
}

/** Reconstruct a slot from share-link slugs (`speciesSlug` plus optional
 * per-track `itemSlug` / `abilitySlug`). A `null` slug leaves that track free,
 * so a bare species is a site pin, `species~item` a partial (or full) pin, and
 * `species~item~ability` a full feature pin. Returns null when the species
 * isn't in the model. Unmatched track slugs leave that track free. */
export function slotFromSlugs(
  model: IsingModel,
  slugIndex: SlugIndex,
  speciesSlug: string,
  itemSlug: string | null,
  abilitySlug: string | null,
): RosterSlot | null {
  const name = resolveSpeciesSlug(slugIndex, model, speciesSlug);
  if (!name) return null;
  const site = model.sites.indexOf(name);
  if (site < 0) return null;
  const trackValues: (string | null)[] = model.tracks.map(() => null);
  const feats = model.siteFeatures[site];
  if (itemSlug !== null && model.tracks.length > 0) {
    for (const f of feats) {
      const it = model.trackValues[f][0];
      const slug = it === null || it === "None" ? "none" : itemToSlug(it);
      if (slug === itemSlug) {
        trackValues[0] = it;
        break;
      }
    }
  }
  const abIdx = model.tracks.findIndex((t) => t.name === "ability");
  if (abilitySlug !== null && abIdx >= 0) {
    for (const f of feats) {
      const ab = model.trackValues[f][abIdx];
      if (ab !== null && abilityToSlug(ab) === abilitySlug) {
        trackValues[abIdx] = ab;
        break;
      }
    }
  }
  return { site, trackValues };
}

/** Derived sampler pins from a roster: fully-determined slots become feature
 * pins (`fixedIdxs`); the rest become site pins (`fixedSites`) carrying their
 * per-track pinned values (`sitePinTrackValues`, null tracks free). */
export function derivePins(model: IsingModel, roster: readonly RosterSlot[]): {
  fixedIdxs: number[];
  fixedSites: number[];
  sitePinTrackValues: (string | null)[][];
} {
  const fixedIdxs: number[] = [];
  const fixedSites: number[] = [];
  const sitePinTrackValues: (string | null)[][] = [];
  for (const slot of roster) {
    const f = slotFeature(model, slot);
    if (f !== null) {
      fixedIdxs.push(f);
    } else {
      fixedSites.push(slot.site);
      sitePinTrackValues.push(slot.trackValues.slice());
    }
  }
  return { fixedIdxs, fixedSites, sitePinTrackValues };
}
