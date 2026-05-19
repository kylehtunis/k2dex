// Constant-temperature swap-move MCMC and the shared chain primitives
// used by pt.ts.
//
// Mirrors sampling._SwapChainState, _init_chain, _local_swap_step, and
// the public swap_mcmc. The swap move flips one currently-on mon off
// and one currently-off mon on simultaneously, preserving the
// team-size-6 constraint a single-spin flip would break.

import type { IsingModel, SwapMcmcResult, TeamIndices } from "./types";
import type { Rng } from "./rng";
import { Rng as RngImpl } from "./rng";
import {
  availableIndices,
  buildConstraintSets,
  initializeState,
  swapViolatesUniqueness,
  teamEnergy,
} from "./energy";

export interface ChainState {
  /** Boolean state, length V. */
  state: Uint8Array;
  /** Float64 view of state for the matvec inner loop. */
  stateF: Float64Array;
  /** Non-fixed mons currently on the team. */
  onNf: number[];
  /** Non-fixed mons currently off the team. */
  offNf: number[];
  /** Running H(state) under h_eff. Updated incrementally by accepts. */
  energy: number;
}

export interface SwapMcmcOpts {
  fixed: readonly number[];
  excluded: readonly number[];
  fieldWeight: number;
  nSteps: number;
  temperature: number;
  seed: number;
}

/** Build a ChainState with `fixed` clamped on and `nToFill` free slots
 * filled via uniqueness-respecting initialization. Returns null if no
 * valid initial state can be built (e.g., user over-constrained items). */
export function initChain(
  model: IsingModel,
  fixed: readonly number[],
  available: readonly number[],
  nToFill: number,
  fixedSpecies: Set<string>,
  fixedItems: Set<string>,
  hEff: Float64Array,
  rng: Rng,
): ChainState | null {
  const { V, J, speciesOf, itemOf } = model;
  const state = new Uint8Array(V);
  for (const i of fixed) state[i] = 1;

  let onNf: number[];
  let offNf: number[];
  if (nToFill > 0) {
    const init = initializeState(
      available,
      nToFill,
      fixedSpecies,
      fixedItems,
      speciesOf,
      itemOf,
      rng,
    );
    if (init === null) return null;
    for (const i of init) state[i] = 1;
    onNf = init;
    const onSet = new Set(init);
    offNf = [];
    for (const i of available) if (!onSet.has(i)) offNf.push(i);
  } else {
    onNf = [];
    offNf = [];
  }

  const stateF = new Float64Array(V);
  for (let i = 0; i < V; i++) stateF[i] = state[i];

  return {
    state,
    stateF,
    onNf,
    offNf,
    energy: teamEnergy(stateF, J, hEff, V),
  };
}

/** One MH swap proposal at temperature T. Mutates `chain` in place on
 * accept. Returns (proposed, accepted) where proposed=false means the
 * proposal was pre-rejected by the uniqueness check (no MH draw was
 * actually evaluated). Callers do NOT count uniqueness rejections
 * toward acceptance statistics. */
export function localSwapStep(
  chain: ChainState,
  model: IsingModel,
  hEff: Float64Array,
  T: number,
  fixedSpecies: Set<string>,
  fixedItems: Set<string>,
  rng: Rng,
): { proposed: boolean; accepted: boolean } {
  if (chain.offNf.length === 0 || chain.onNf.length === 0) {
    return { proposed: false, accepted: false };
  }
  const outK = rng.integers(chain.onNf.length);
  const inK = rng.integers(chain.offNf.length);
  const iOut = chain.onNf[outK];
  const iIn = chain.offNf[inK];
  if (
    swapViolatesUniqueness(
      iIn,
      outK,
      chain.onNf,
      fixedSpecies,
      fixedItems,
      model.speciesOf,
      model.itemOf,
    )
  ) {
    return { proposed: false, accepted: false };
  }
  // ΔH = h_eff[out] - h_eff[in] + (J[out] - J[in]) · state_f + J[in, out]
  const V = model.V;
  const baseOut = iOut * V;
  const baseIn = iIn * V;
  const J = model.J;
  const stateF = chain.stateF;
  let dotDiff = 0;
  for (let j = 0; j < V; j++) {
    dotDiff += (J[baseOut + j] - J[baseIn + j]) * stateF[j];
  }
  const deltaH = hEff[iOut] - hEff[iIn] + dotDiff + J[baseIn + iOut];
  const accept = deltaH <= 0 || rng.random() < Math.exp(-deltaH / T);
  if (accept) {
    chain.state[iOut] = 0;
    chain.state[iIn] = 1;
    chain.stateF[iOut] = 0;
    chain.stateF[iIn] = 1;
    chain.onNf[outK] = iIn;
    chain.offNf[inK] = iOut;
    chain.energy += deltaH;
  }
  return { proposed: true, accepted: accept };
}

/** Snapshot a chain's current team as a sorted index list. */
export function snapshotTeam(chain: ChainState): TeamIndices {
  const team: number[] = [];
  for (let i = 0; i < chain.state.length; i++) {
    if (chain.state[i]) team.push(i);
  }
  return team;
}

/** Constant-T swap-move MCMC. Returns null if a valid initial state
 * can't be constructed (over-constrained). */
export function swapMcmc(
  model: IsingModel,
  opts: SwapMcmcOpts,
): SwapMcmcResult | null {
  const { V, h, teamSize, speciesOf, itemOf } = model;
  const rng = new RngImpl(opts.seed);

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

  if (nToFill === 0) {
    // Team fully determined by `fixed`; no swaps possible.
    const team = [...opts.fixed].sort((a, b) => a - b);
    const samples: TeamIndices[] = new Array(opts.nSteps);
    for (let i = 0; i < opts.nSteps; i++) samples[i] = team;
    return { samples, acceptRate: 0 };
  }

  const chain = initChain(
    model,
    opts.fixed,
    available,
    nToFill,
    fixedSpecies,
    fixedItems,
    hEff,
    rng,
  );
  if (chain === null) return null;

  let accepted = 0;
  let proposed = 0;
  const samples: TeamIndices[] = new Array(opts.nSteps);
  for (let step = 0; step < opts.nSteps; step++) {
    const r = localSwapStep(
      chain,
      model,
      hEff,
      opts.temperature,
      fixedSpecies,
      fixedItems,
      rng,
    );
    if (r.proposed) proposed++;
    if (r.accepted) accepted++;
    samples[step] = snapshotTeam(chain);
  }
  return {
    samples,
    acceptRate: proposed > 0 ? accepted / proposed : 0,
  };
}
