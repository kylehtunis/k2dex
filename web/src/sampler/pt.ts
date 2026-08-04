// Parallel-tempered MCMC. K chains run at temperatures `tLadder`
// (sorted ascending; chain 0 is the target / cold chain). Every
// `swapInterval` sweeps, propose adjacent-chain replica-exchange swaps
// with `min(1, exp((1/T_lo - 1/T_hi) * (E_lo - E_hi)))`. Cold-chain
// samples are kept post burn-in.
//
// The inner move is the universal Potts kernel (potts.ts): each sweep is a
// species-swap, or — when the model has tracks — with probability `pReroll` an
// item track-reroll instead. For a species-only model this degenerates to the
// atomic swap. `localSwapStep` (swap.ts) is retained for the `swapMcmc` utility
// but is no longer the PT inner move.
//
// Mirrors sampling.parallel_tempered_mcmc.

import type { IsingModel, PTResult, TeamIndices } from "./types";
import { Rng } from "./rng";
import { availableIndices, buildConstraintSets, resolveSitePins } from "./energy";
import { initChain, snapshotTeam } from "./swap";
import type { ChainState } from "./swap";
import {
  buildSiteTables,
  pottsSpeciesSwap,
  pottsTrackReroll,
  type PottsContext,
  pottsMoveSets,
} from "./potts";

export interface PTOpts {
  fixed: readonly number[];
  excluded: readonly number[];
  fieldWeight: number;
  tLadder: readonly number[];
  nSteps: number;
  burnIn: number;
  swapInterval: number;
  seed: number;
  /** Probability an item-track model rerolls (vs species-swaps) each sweep.
   * Ignored for species-only models. Default 0.5. */
  pReroll?: number;
  /** Site-level pins (species fixed, track values free). Each is seeded to its
   * best placeable feature, then its slot is locked against species-swaps but
   * still rerolled. Empty = feature-level pins only. */
  fixedSites?: readonly number[];
  /** Anchor-field tilt alpha ("Anchor Strength"): samples from
   * H_alpha = H - (alpha-1)·Σ_{p∈pins, j free} J[p,j]s_j, concentrating mass
   * on teams that couple well to the pins (feature + site). 1 = no tilt
   * (default); no effect when nothing is pinned. */
  anchorStrength?: number;
}

export function parallelTemperedMcmc(
  model: IsingModel,
  opts: PTOpts,
): PTResult | null {
  const { V, h, teamSize } = model;
  const rng = new Rng(opts.seed);
  const K = opts.tLadder.length;

  // Resolve site-level pins to concrete seed features (species fixed, item
  // free). They occupy free slots that are locked against species-swaps but
  // still rerolled; the remaining slots fill randomly.
  const seeds = resolveSitePins(model, opts.fixedSites ?? [], opts.fixed, opts.excluded);
  if (seeds === null) return null;

  const available = availableIndices(model, opts.fixed, opts.excluded);
  const nToFill = teamSize - opts.fixed.length - seeds.length;
  if (nToFill < 0 || available.length < nToFill) return null;

  const hEff = new Float64Array(V);
  for (let i = 0; i < V; i++) hEff[i] = opts.fieldWeight * h[i];

  // The random fill must avoid the feature pins' AND the seeds' sites/values.
  const constraints = buildConstraintSets([...opts.fixed, ...seeds], model);

  // Potts move context: site tables, availability mask, fixed pins, and the
  // site-pinned slot indices (0..seeds.length-1, front of onNf).
  const tables = buildSiteTables(model);
  const avail = new Uint8Array(V).fill(1);
  for (const e of opts.excluded) avail[e] = 0;
  for (const f of opts.fixed) avail[f] = 0; // pinned features are retained, never re-placed
  // Seed features are NOT masked off: their slot rerolls among the site's items.
  const lockedSlots = new Set<number>();
  for (let i = 0; i < seeds.length; i++) lockedSlots.add(i);
  const anchorStrength = opts.anchorStrength ?? 1;
  const moveSets = pottsMoveSets(tables, avail, nToFill + seeds.length, lockedSlots);
  if (moveSets.usableSites.length === 0) return null;
  const ctx: PottsContext = {
    fixed: opts.fixed,
    avail,
    tables,
    lockedSlots,
    anchorStrength,
    ...moveSets,
  };
  const hasTracks = model.tracks.length > 0;
  const pReroll = opts.pReroll ?? 0.5;

  const chains: ChainState[] = [];
  for (let k = 0; k < K; k++) {
    const c = initChain(
      model,
      opts.fixed,
      available,
      nToFill,
      constraints,
      hEff,
      rng,
      seeds,
    );
    if (c === null) return null;
    // initChain's energy is the untilted H(hEff); correct it to H_alpha so
    // move deltas (tilted slot energies) and replica exchange stay consistent.
    if (anchorStrength !== 1) {
      // Pin↔free cross-coupling sum; pin↔pin pairs are excluded from the tilt.
      const pins: number[] = [...opts.fixed];
      const free: number[] = [];
      for (let s = 0; s < c.onNf.length; s++) {
        (lockedSlots.has(s) ? pins : free).push(c.onNf[s]);
      }
      let cross = 0;
      for (const p of pins) {
        const base = p * V;
        for (const j of free) cross += model.J[base + j];
      }
      c.energy -= (anchorStrength - 1) * cross;
    }
    chains.push(c);
  }

  const samples: TeamIndices[] = new Array(opts.nSteps);
  let localAccept = 0;
  let localPropose = 0;
  let swapAccept = 0;
  let swapPropose = 0;

  for (let step = 0; step < opts.nSteps; step++) {
    // One Potts move per chain at its own temperature: a species-swap, or an
    // item track-reroll (chosen with probability pReroll when tracks exist).
    // Reroll is a Gibbs step (always accepted) and is not counted toward the
    // species-swap MH acceptance statistic.
    for (let k = 0; k < K; k++) {
      if (hasTracks && rng.random() < pReroll) {
        pottsTrackReroll(chains[k], model, hEff, opts.tLadder[k], tables, ctx, rng);
      } else {
        const r = pottsSpeciesSwap(
          chains[k],
          model,
          hEff,
          opts.tLadder[k],
          tables,
          ctx,
          rng,
        );
        if (r.proposed) localPropose++;
        if (r.accepted) localAccept++;
      }
    }

    // Periodic replica-exchange between adjacent T levels.
    if (step > 0 && step % opts.swapInterval === 0) {
      for (let k = 0; k < K - 1; k++) {
        const tLo = opts.tLadder[k];
        const tHi = opts.tLadder[k + 1];
        const betaDiff = 1 / tLo - 1 / tHi; // > 0 (cold has higher beta)
        const delta = betaDiff * (chains[k].energy - chains[k + 1].energy);
        swapPropose++;
        if (delta >= 0 || rng.random() < Math.exp(delta)) {
          const tmp = chains[k];
          chains[k] = chains[k + 1];
          chains[k + 1] = tmp;
          swapAccept++;
        }
      }
    }

    samples[step] = snapshotTeam(chains[0]);
  }

  return {
    samples: samples.slice(opts.burnIn),
    localAccept: localPropose > 0 ? localAccept / localPropose : 0,
    swapAccept: swapPropose > 0 ? swapAccept / swapPropose : 0,
  };
}
