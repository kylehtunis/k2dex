// Filter + sort logic for the §03 extreme-couplings tables.
//
// Mirrors the upper-triangle mask + cross-species / cross-item filter
// from app.py:_render_meta. Phase 2 has unique species and all-null
// items, so both filters become no-ops there.

import type { IsingModel } from "../sampler/types";

export interface CouplingPair {
  /** Upper-triangle index (i < j) into the vocab. */
  i: number;
  j: number;
  jValue: number;
}

/** True when (i, j) is a *structural* coupling worth surfacing — i.e. not a
 * mechanical mutual exclusion. Same-species pairs, and same-value pairs on any
 * cross-slot-unique track (today: item only), couple purely because the two
 * builds can't co-exist on a team, so they carry no metagame signal. On
 * Species-only vocab (unique species, all-null items) both checks are no-ops,
 * so every off-diagonal pair is structural. Shared by filteredCouplings and
 * render/featureDetail. */
export function isStructuralPair(model: IsingModel, i: number, j: number): boolean {
  const { speciesOf, tracks, trackValues } = model;
  if (speciesOf[i] === speciesOf[j]) return false;
  for (let t = 0; t < tracks.length; t++) {
    if (!tracks[t].crossSlotUnique) continue;
    const vi = trackValues[i][t];
    const vj = trackValues[j][t];
    if (vi !== null && vj !== null && vi === vj) return false;
  }
  return true;
}

// ---------- Hierarchical species -> item -> attribute drill-down ----------
//
// The client-side mirror of potts.decompose_block_hierarchical, ranked rather
// than returned as matrices. A species pair's block is read at two tiers:
//
//   item tier      the ability-marginalized item block — each item pair's
//                  coupling is the usage-weighted mean over the states sharing
//                  that item, so near-duplicate ability variants collapse into
//                  one row instead of repeating.
//   attribute tier the residual within one item pair: how each attribute value
//                  (ability) deviates from that item pair's item-level figure.
//                  Its usage-weighted mean is zero by construction, so it is
//                  exactly the conditional-on-item attribute effect.
//
// Weights come from the data marginal `m` (empirical appearance rate), matching
// potts.py's usage weighting, and support figures are `m` scaled by the corpus
// size — the same approximation as FittedModel.appearances.

/** Item is track 0 by construction (loaders.ITEM_TRACK) and is the tier every
 * other attribute nests under, both here and in potts.py. */
const ITEM_TRACK = 0;

/** A usage-weighted group of a site's features sharing one track value. */
export interface FeatureGroup {
  /** The shared track value (null when the track carries no value). */
  value: string | null;
  features: number[];
  /** Per-feature weights within the group, summing to 1 (uniform if massless). */
  weights: number[];
  /** Most-used feature in the group — the representative for sprite/labels. */
  rep: number;
  /** Unnormalized marginal mass of the group (before renormalization). */
  mass: number;
}

/** Group `features` by their value on `track`, in first-appearance order, with
 * usage weights renormalized inside each group. Mirrors potts._item_groups. */
export function groupByTrack(
  model: IsingModel,
  features: readonly number[],
  track: number,
): FeatureGroup[] {
  const byValue = new Map<string | null, FeatureGroup>();
  const order: FeatureGroup[] = [];
  for (const f of features) {
    const v = model.trackValues[f][track];
    let g = byValue.get(v);
    if (!g) {
      g = { value: v, features: [], weights: [], rep: f, mass: 0 };
      byValue.set(v, g);
      order.push(g);
    }
    g.features.push(f);
    g.weights.push(model.m[f]);
    g.mass += model.m[f];
    if (model.m[f] > model.m[g.rep]) g.rep = f;
  }
  for (const g of order) {
    if (g.mass > 1e-12) {
      g.weights = g.weights.map((w) => w / g.mass);
    } else {
      g.weights = g.weights.map(() => 1 / g.features.length);
    }
  }
  return order;
}

/** The usage-weighted mean coupling between two groups of features — one cell
 * of the marginalized block. */
function groupCoupling(model: IsingModel, a: FeatureGroup, b: FeatureGroup): number {
  const { J, V } = model;
  let acc = 0;
  for (let ia = 0; ia < a.features.length; ia++) {
    const row = a.features[ia] * V;
    const wa = a.weights[ia];
    for (let ib = 0; ib < b.features.length; ib++) {
      acc += wa * b.weights[ib] * J[row + b.features[ib]];
    }
  }
  return acc;
}

export interface ModulationEntry {
  /** Representative features of the two groups (sprite + label rendering). */
  featureA: number;
  featureB: number;
  jValue: number;
  /** How far this pair sits from its parent tier's figure — the species-level
   * synergy for item entries, the item-pair coupling for attribute entries. */
  deviation: number;
  /** Weighted corpus appearances of the rarer of the two groups. */
  support: number;
}

/** Item-tier entries for a species pair, strongest |J| first: how each pairing
 * of their item builds shifts away from the pair's species-level synergy, with
 * every other attribute marginalized out. Mechanically-excluded pairs are
 * dropped (see isStructuralPair). Shared by the /meta §02 coupling table and
 * the feature modal's coupling drill-down, so both surfaces rank and filter
 * identically. `groups` is the parallel per-entry group pair, which the /meta
 * attribute tier drills into. */
export function topModulationEntries(
  model: IsingModel,
  siteA: number,
  siteB: number,
  synergy: number,
  topN = 8,
): (ModulationEntry & { groupA: FeatureGroup; groupB: FeatureGroup })[] {
  if (siteA === siteB) return [];
  const groupsA = groupByTrack(model, model.siteFeatures[siteA], ITEM_TRACK);
  const groupsB = groupByTrack(model, model.siteFeatures[siteB], ITEM_TRACK);
  const entries: (ModulationEntry & { groupA: FeatureGroup; groupB: FeatureGroup })[] = [];
  for (const ga of groupsA) {
    for (const gb of groupsB) {
      if (!isStructuralPair(model, ga.rep, gb.rep)) continue;
      const jValue = groupCoupling(model, ga, gb);
      entries.push({
        featureA: ga.rep,
        featureB: gb.rep,
        jValue,
        deviation: jValue - synergy,
        support: Math.min(ga.mass, gb.mass) * model.nCorpusTeams,
        groupA: ga,
        groupB: gb,
      });
    }
  }
  entries.sort((a, b) => Math.abs(b.jValue) - Math.abs(a.jValue));
  return entries.slice(0, topN);
}

/** Attribute-tier entries nested under one item pair: each pairing of the two
 * groups' values on `track`, ranked by how far it deviates from the item-level
 * coupling `itemJ`. Any further tracks are marginalized out the same way the
 * item tier marginalizes this one, so the tier isolates `track`'s conditional
 * effect. Groups whose weighted corpus support falls below `minSupport` are
 * dropped — a residual read off two or three teams is noise. */
export function topAttributeEntries(
  model: IsingModel,
  groupA: FeatureGroup,
  groupB: FeatureGroup,
  itemJ: number,
  track: number,
  minSupport: number,
  topN = 8,
): ModulationEntry[] {
  const subA = groupByTrack(model, groupA.features, track)
    .filter((g) => g.mass * model.nCorpusTeams >= minSupport);
  const subB = groupByTrack(model, groupB.features, track)
    .filter((g) => g.mass * model.nCorpusTeams >= minSupport);
  if (subA.length * subB.length < 2) return [];
  const entries: ModulationEntry[] = [];
  for (const ga of subA) {
    for (const gb of subB) {
      const jValue = groupCoupling(model, ga, gb);
      entries.push({
        featureA: ga.rep,
        featureB: gb.rep,
        jValue,
        deviation: jValue - itemJ,
        support: Math.min(ga.mass, gb.mass) * model.nCorpusTeams,
      });
    }
  }
  entries.sort((a, b) => Math.abs(b.deviation) - Math.abs(a.deviation));
  return entries.slice(0, topN);
}

/** True when a site carries more than one value on `track` inside at least one
 * item group — i.e. the attribute residual is not zero by construction. A
 * species whose every item build has the same ability has nothing to drill. */
export function hasAttributeSplit(model: IsingModel, site: number, track: number): boolean {
  for (const g of groupByTrack(model, model.siteFeatures[site], ITEM_TRACK)) {
    if (groupByTrack(model, g.features, track).length > 1) return true;
  }
  return false;
}

/** Iterate the strict upper triangle of J, dropping non-structural pairs
 * (see isStructuralPair). */
export function filteredCouplings(model: IsingModel): CouplingPair[] {
  const { V, J } = model;
  const out: CouplingPair[] = [];
  for (let i = 0; i < V; i++) {
    for (let j = i + 1; j < V; j++) {
      if (!isStructuralPair(model, i, j)) continue;
      out.push({ i, j, jValue: J[i * V + j] });
    }
  }
  return out;
}
