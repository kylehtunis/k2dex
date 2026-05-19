// Parallel-tempered MCMC. K chains run at temperatures `tLadder`
// (sorted ascending; chain 0 is the target / cold chain). Every
// `swapInterval` sweeps, propose adjacent-chain replica-exchange swaps
// with `min(1, exp((1/T_lo - 1/T_hi) * (E_lo - E_hi)))`. Cold-chain
// samples are kept post burn-in.
//
// Mirrors sampling.parallel_tempered_mcmc.

import type { IsingModel, PTResult, TeamIndices } from "./types";
import { Rng } from "./rng";
import { availableIndices, buildConstraintSets } from "./energy";
import { initChain, localSwapStep, snapshotTeam } from "./swap";
import type { ChainState } from "./swap";

export interface PTOpts {
  fixed: readonly number[];
  excluded: readonly number[];
  fieldWeight: number;
  tLadder: readonly number[];
  nSteps: number;
  burnIn: number;
  swapInterval: number;
  seed: number;
}

export function parallelTemperedMcmc(
  model: IsingModel,
  opts: PTOpts,
): PTResult | null {
  const { V, h, teamSize, speciesOf, itemOf } = model;
  const rng = new Rng(opts.seed);
  const K = opts.tLadder.length;

  const available = availableIndices(model, opts.fixed, opts.excluded);
  const nToFill = teamSize - opts.fixed.length;
  if (available.length < nToFill) return null;

  const hEff = new Float64Array(V);
  for (let i = 0; i < V; i++) hEff[i] = opts.fieldWeight * h[i];

  const { fixedSpecies, fixedItems } = buildConstraintSets(
    opts.fixed,
    speciesOf,
    itemOf,
  );

  const chains: ChainState[] = [];
  for (let k = 0; k < K; k++) {
    const c = initChain(
      model,
      opts.fixed,
      available,
      nToFill,
      fixedSpecies,
      fixedItems,
      hEff,
      rng,
    );
    if (c === null) return null;
    chains.push(c);
  }

  const samples: TeamIndices[] = new Array(opts.nSteps);
  let localAccept = 0;
  let localPropose = 0;
  let swapAccept = 0;
  let swapPropose = 0;

  for (let step = 0; step < opts.nSteps; step++) {
    // One local MH move per chain at its own temperature.
    for (let k = 0; k < K; k++) {
      const r = localSwapStep(
        chains[k],
        model,
        hEff,
        opts.tLadder[k],
        fixedSpecies,
        fixedItems,
        rng,
      );
      if (r.proposed) localPropose++;
      if (r.accepted) localAccept++;
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
