// Universal Potts move kernel for the PT sampler.
//
// Per-chain mirror of the batched training kernel in k2dex/models.py
// (`_build_site_tables`, `_site_conditional`, `_potts_species_swap_sweep`,
// `_potts_track_reroll_sweep`). The deterministic pieces (`buildSiteTables`,
// `siteConditional`) are parity-gated against sampling.py; the stochastic moves
// (`pottsSpeciesSwap`, `pottsTrackReroll`) are JS smoke-tested only.
//
// The model is treated as a Potts model: sites = species, states = a species'
// full features (points in the product of the per-track value alphabets: item ×
// ability × ...), relative to the absent reference. A species-swap integrates
// the whole state out (Metropolized-Gibbs, accept on Z_B / Z_A, then draw the
// full state from the exact conditional); a per-track reroll resamples one
// on-team slot's value on a single track (pinning the others) from its exact
// conditional. For a species-only model (no tracks) each site has one feature,
// so the species-swap acceptance Z_B / Z_A reduces to exp(-ΔH/T) — algebraically
// identical to the atomic `localSwapStep` — and no reroll occurs.

import type { IsingModel } from "./types";
import type { Rng } from "./rng";
import type { ChainState } from "./swap";

export interface SiteTables {
  /** Number of distinct sites (species). */
  nSites: number;
  /** site index -> its feature (flat vocab index) list, ascending. */
  siteFeatures: number[][];
  /** Per-feature item id for the primary unique track: distinct integer per
   * item string (first-appearance order), -1 for itemless / no unique track.
   * Mirrors `models._group_ids(item_of, none_sentinel=True)`. */
  itemId: Int32Array;
  /** Number of attribute tracks (item, ability, ...). 0 for species-only. */
  nTracks: number;
  /** Per-feature per-track dense value id, flat row-major `f*nTracks + t`.
   * Unlike `itemId` this is dense (null / itemless gets a real id) so it can
   * pin a track to an exact value. Mirrors `models._build_site_tables`'
   * `sp_track_vid`. */
  trackVid: Int32Array;
}

/** Index of the primary unique track (the item track), or -1 if the model has
 * no unique track (species-only). The per-chain conditional enforces exclusion
 * on this one track, matching the single item dimension of the training kernel.
 * (Multi-unique-track exclusion in the conditional is a future extension.) */
function primaryUniqueTrack(model: IsingModel): number {
  for (let t = 0; t < model.tracks.length; t++) {
    if (model.tracks[t].crossSlotUnique) return t;
  }
  return -1;
}

/** Precompute per-site feature arrays and the per-feature item id used by
 * `siteConditional`. Deterministic; parity-gated against
 * `sampling.build_site_tables`. */
export function buildSiteTables(model: IsingModel): SiteTables {
  const { V, siteOf, sites, trackValues, tracks } = model;
  const nSites = sites.length;
  const siteFeatures: number[][] = sites.map(() => []);
  for (let f = 0; f < V; f++) siteFeatures[siteOf[f]].push(f);

  const t = primaryUniqueTrack(model);
  const itemId = new Int32Array(V);
  if (t < 0) {
    itemId.fill(-1);
  } else {
    const idOf = new Map<string, number>();
    for (let f = 0; f < V; f++) {
      const v = trackValues[f][t];
      if (v === null) {
        itemId[f] = -1;
        continue;
      }
      let id = idOf.get(v);
      if (id === undefined) {
        id = idOf.size;
        idOf.set(v, id);
      }
      itemId[f] = id;
    }
  }

  // Dense per-track value ids (first-appearance order in feature order, per
  // track). null is a real value here so a track can be pinned to "no item".
  const nTracks = tracks.length;
  const trackVid = new Int32Array(V * nTracks);
  for (let tr = 0; tr < nTracks; tr++) {
    const idOf = new Map<string | null, number>();
    for (let f = 0; f < V; f++) {
      const v = trackValues[f][tr];
      let id = idOf.get(v);
      if (id === undefined) {
        id = idOf.size;
        idOf.set(v, id);
      }
      trackVid[f * nTracks + tr] = id;
    }
  }
  return { nSites, siteFeatures, itemId, nTracks, trackVid };
}

export interface SiteConditional {
  /** Candidate item-state flat indices for the site. */
  feats: number[];
  /** -E_slot / T at each candidate (E_slot = -h_eff[f] - Σ_{r∈R} J[f,r]). */
  negE: Float64Array;
  /** True where the candidate is available and clears item-exclusion vs R. */
  valid: boolean[];
  /** Tempered log item-partition log Σ_valid exp(negE); -Infinity if none. */
  logZ: number;
}

/** Per-chain item conditional for placing `site` alongside retained members
 * `rFeat` (with item ids `rItemId`). `avail[f]` gates feature-level
 * availability (excluded features are unavailable). `rWeights` (optional,
 * default all-1) scales each retained member's coupling — the anchor-field
 * tilt puts weight alpha on pin↔free couplings. `pinValues` (optional, length
 * nTracks, -1 = free) restricts candidates to states matching every pinned
 * track — a joint species-swap draw passes all-free (or undefined); a per-track
 * reroll of track `t` pins every other track. Deterministic; parity-gated
 * against `sampling.site_conditional`. */
export function siteConditional(
  site: number,
  rFeat: readonly number[],
  rItemId: readonly number[],
  model: IsingModel,
  hEff: Float64Array,
  invTemp: number,
  tables: SiteTables,
  avail: Uint8Array,
  rWeights?: ArrayLike<number>,
  pinValues?: ArrayLike<number>,
): SiteConditional {
  const { V, J } = model;
  const { nTracks, trackVid } = tables;
  const feats = tables.siteFeatures[site];
  const M = feats.length;
  const negE = new Float64Array(M);
  const valid: boolean[] = new Array(M);
  let mx = -Infinity;
  for (let j = 0; j < M; j++) {
    const f = feats[j];
    // E_slot = -h_eff[f] - Σ_{r∈R} w_r · J[f, r]; negE = -E_slot / T.
    let jSum = 0;
    const base = f * V;
    if (rWeights === undefined) {
      for (let r = 0; r < rFeat.length; r++) jSum += J[base + rFeat[r]];
    } else {
      for (let r = 0; r < rFeat.length; r++) jSum += rWeights[r] * J[base + rFeat[r]];
    }
    negE[j] = (hEff[f] + jSum) * invTemp;
    // Availability + item-exclusion: a candidate holding a real item already
    // held by a retained member is invalid; itemless (id < 0) never conflicts.
    let ok = avail[f] === 1;
    if (ok) {
      const iid = tables.itemId[f];
      if (iid >= 0) {
        for (let r = 0; r < rItemId.length; r++) {
          if (rItemId[r] === iid) {
            ok = false;
            break;
          }
        }
      }
    }
    // Pin restriction: drop candidates whose value on any pinned track differs.
    if (ok && pinValues !== undefined) {
      for (let t = 0; t < nTracks; t++) {
        if (pinValues[t] >= 0 && trackVid[f * nTracks + t] !== pinValues[t]) {
          ok = false;
          break;
        }
      }
    }
    valid[j] = ok;
    if (ok && negE[j] > mx) mx = negE[j];
  }
  let logZ = -Infinity;
  if (mx > -Infinity) {
    let s = 0;
    for (let j = 0; j < M; j++) if (valid[j]) s += Math.exp(negE[j] - mx);
    logZ = Math.log(s) + mx;
  }
  return { feats, negE, valid, logZ };
}

/** Context carried across Potts moves within one PT run. */
export interface PottsContext {
  /** Pinned (feature-locked) vocab indices, always on the team. */
  fixed: readonly number[];
  /** Length-V mask, 1 where a feature may be placed (not excluded). */
  avail: Uint8Array;
  tables: SiteTables;
  /** onNf slot indices whose species is site-pinned: the species never swaps,
   * but the item track still rerolls. Empty/undefined = no site pins (the
   * species-swap then considers every free slot, exactly as before). */
  lockedSlots?: ReadonlySet<number>;
  /** Anchor-field tilt alpha: pin↔free couplings (pins = fixed features and
   * locked slots' current features) are scaled by alpha in every conditional
   * and slot energy, targeting H_alpha = H - (alpha-1)·Σ_{p,j free} J[p,j]s_j.
   * Pin↔pin and free↔free couplings are untouched. Default 1 (no tilt). */
  anchorStrength?: number;
  /** Per-track relative weight for which track a reroll targets (length
   * nTracks, from POTTS_TRACK_REROLL_WEIGHTS by track name). A zero weight
   * disables rerolling that track (e.g. an excluded attribute). Undefined ⇒
   * every track weighted equally. */
  trackRerollWeights?: Float64Array;
}

/** The retained team (all on-team features except the slot at `onNfPos` of the
 * chain's free slots) as (feats, itemIds, isPin flags). A retained member is a
 * pin when it is a fixed feature or sits in a locked (site-pinned) slot. */
function retained(
  chain: ChainState,
  onNfPos: number,
  ctx: PottsContext,
  tables: SiteTables,
): { rFeat: number[]; rItemId: number[]; rPin: boolean[] } {
  const rFeat: number[] = [];
  const rPin: boolean[] = [];
  for (const f of ctx.fixed) {
    rFeat.push(f);
    rPin.push(true);
  }
  for (let k = 0; k < chain.onNf.length; k++) {
    if (k === onNfPos) continue;
    rFeat.push(chain.onNf[k]);
    rPin.push(ctx.lockedSlots?.has(k) ?? false);
  }
  const rItemId = rFeat.map((f) => tables.itemId[f]);
  return { rFeat, rItemId, rPin };
}

/** Anchor-tilt coupling weights for one move: weight alpha on couplings with
 * exactly one pinned endpoint (retained pin ↔ free mover, or retained free ↔
 * pinned mover), 1 otherwise. Undefined at alpha = 1 (no tilt fast path). */
function anchorWeights(
  rPin: readonly boolean[],
  moverPinned: boolean,
  alpha: number,
): Float64Array | undefined {
  if (alpha === 1) return undefined;
  const w = new Float64Array(rPin.length);
  for (let i = 0; i < rPin.length; i++) {
    w[i] = rPin[i] !== moverPinned ? alpha : 1;
  }
  return w;
}

/** Draw one index j ∝ exp(negE[j]) over valid entries. Assumes at least one
 * valid entry. */
function sampleCategorical(
  negE: Float64Array,
  valid: boolean[],
  rng: Rng,
): number {
  let mx = -Infinity;
  for (let j = 0; j < negE.length; j++) if (valid[j] && negE[j] > mx) mx = negE[j];
  let sum = 0;
  for (let j = 0; j < negE.length; j++) if (valid[j]) sum += Math.exp(negE[j] - mx);
  let target = rng.random() * sum;
  for (let j = 0; j < negE.length; j++) {
    if (!valid[j]) continue;
    target -= Math.exp(negE[j] - mx);
    if (target <= 0) return j;
  }
  // Numerical fallback: last valid index.
  for (let j = negE.length - 1; j >= 0; j--) if (valid[j]) return j;
  return 0;
}

function slotEnergy(
  f: number,
  rFeat: readonly number[],
  model: IsingModel,
  hEff: Float64Array,
  rWeights?: ArrayLike<number>,
): number {
  const base = f * model.V;
  let jSum = 0;
  if (rWeights === undefined) {
    for (let r = 0; r < rFeat.length; r++) jSum += model.J[base + rFeat[r]];
  } else {
    for (let r = 0; r < rFeat.length; r++) jSum += rWeights[r] * model.J[base + rFeat[r]];
  }
  return -hEff[f] - jSum;
}

/** One Metropolized-Gibbs species-swap proposal on a random free slot. Mutates
 * `chain` in place on accept (state, stateF, onNf, energy; offNf is not
 * maintained — the PT path does not read it). Returns whether a move was
 * proposed (passed the site-availability pre-check) and whether it accepted. */
export function pottsSpeciesSwap(
  chain: ChainState,
  model: IsingModel,
  hEff: Float64Array,
  T: number,
  tables: SiteTables,
  ctx: PottsContext,
  rng: Rng,
  maxTries = 16,
): { proposed: boolean; accepted: boolean } {
  if (chain.onNf.length === 0) return { proposed: false, accepted: false };
  const invTemp = 1 / T;
  const { siteOf } = model;

  const outK = rng.integers(chain.onNf.length);
  // Site-pinned slots keep their species; only their item track rerolls.
  if (ctx.lockedSlots?.has(outK)) return { proposed: false, accepted: false };
  const outFeat = chain.onNf[outK];
  const siteA = siteOf[outFeat];
  const { rFeat, rItemId, rPin } = retained(chain, outK, ctx, tables);
  // The moving slot is free (locked slots returned above), so retained pins
  // carry the anchor weight.
  const rWeights = anchorWeights(rPin, false, ctx.anchorStrength ?? 1);

  // Present sites (retained team + the out slot's own species): the proposal B
  // must be an off-team species.
  const present = new Set<number>();
  for (const f of rFeat) present.add(siteOf[f]);
  present.add(siteA);
  if (present.size >= tables.nSites) return { proposed: false, accepted: false };

  let siteB = rng.integers(tables.nSites);
  let tries = 0;
  while (present.has(siteB) && tries < maxTries) {
    siteB = rng.integers(tables.nSites);
    tries++;
  }
  if (present.has(siteB)) return { proposed: false, accepted: false };

  const condA = siteConditional(siteA, rFeat, rItemId, model, hEff, invTemp, tables, ctx.avail, rWeights);
  const condB = siteConditional(siteB, rFeat, rItemId, model, hEff, invTemp, tables, ctx.avail, rWeights);
  if (!Number.isFinite(condB.logZ) || !Number.isFinite(condA.logZ)) {
    return { proposed: false, accepted: false };
  }

  const logRatio = condB.logZ - condA.logZ;
  const accept = logRatio >= 0 || rng.random() < Math.exp(logRatio);
  if (!accept) return { proposed: true, accepted: false };

  const choice = sampleCategorical(condB.negE, condB.valid, rng);
  const newFeat = condB.feats[choice];
  const dH =
    slotEnergy(newFeat, rFeat, model, hEff, rWeights) -
    slotEnergy(outFeat, rFeat, model, hEff, rWeights);
  chain.state[outFeat] = 0;
  chain.state[newFeat] = 1;
  chain.stateF[outFeat] = 0;
  chain.stateF[newFeat] = 1;
  chain.onNf[outK] = newFeat;
  chain.energy += dH;
  return { proposed: true, accepted: true };
}

/** Pick a track to reroll ∝ `weights` (length nTracks). Returns -1 when every
 * weight is zero (nothing rerollable). Undefined weights ⇒ uniform. */
function pickTrack(nTracks: number, weights: Float64Array | undefined, rng: Rng): number {
  if (nTracks === 0) return -1;
  let total = 0;
  for (let t = 0; t < nTracks; t++) total += weights ? weights[t] : 1;
  if (total <= 0) return -1;
  let target = rng.random() * total;
  for (let t = 0; t < nTracks; t++) {
    target -= weights ? weights[t] : 1;
    if (target <= 0) return t;
  }
  return nTracks - 1;
}

/** One Gibbs per-track reroll on a random free slot: pick a track (∝
 * `ctx.trackRerollWeights`), pin every other track to the slot's current
 * value, and resample that one track from the exact conditional given the rest
 * of the team (species and the untouched tracks unchanged; always accepted).
 * Mutates `chain` in place (offNf not maintained). No-op when the model has no
 * tracks or no track is rerollable. */
export function pottsTrackReroll(
  chain: ChainState,
  model: IsingModel,
  hEff: Float64Array,
  T: number,
  tables: SiteTables,
  ctx: PottsContext,
  rng: Rng,
): void {
  const { nTracks, trackVid } = tables;
  if (chain.onNf.length === 0 || nTracks === 0) return;
  const track = pickTrack(nTracks, ctx.trackRerollWeights, rng);
  if (track < 0) return;
  const invTemp = 1 / T;
  const outK = rng.integers(chain.onNf.length);
  const outFeat = chain.onNf[outK];
  const site = model.siteOf[outFeat];
  const { rFeat, rItemId, rPin } = retained(chain, outK, ctx, tables);
  // A locked (site-pinned) slot's reroll is a pin move: its couplings to free
  // retained members carry the anchor weight; couplings to other pins do not.
  const moverPinned = ctx.lockedSlots?.has(outK) ?? false;
  const rWeights = anchorWeights(rPin, moverPinned, ctx.anchorStrength ?? 1);

  // Pin every track except the rerolled one to the slot's current values, so
  // the conditional ranges only over states that differ in track `track`.
  const pinValues = new Int32Array(nTracks);
  for (let t = 0; t < nTracks; t++) pinValues[t] = trackVid[outFeat * nTracks + t];
  pinValues[track] = -1;

  const cond = siteConditional(
    site, rFeat, rItemId, model, hEff, invTemp, tables, ctx.avail, rWeights, pinValues);
  if (!Number.isFinite(cond.logZ)) return; // no valid state (shouldn't happen: current is valid)
  const choice = sampleCategorical(cond.negE, cond.valid, rng);
  const newFeat = cond.feats[choice];
  if (newFeat === outFeat) return;
  const dH =
    slotEnergy(newFeat, rFeat, model, hEff, rWeights) -
    slotEnergy(outFeat, rFeat, model, hEff, rWeights);
  chain.state[outFeat] = 0;
  chain.state[newFeat] = 1;
  chain.stateF[outFeat] = 0;
  chain.stateF[newFeat] = 1;
  chain.onNf[outK] = newFeat;
  chain.energy += dH;
}
