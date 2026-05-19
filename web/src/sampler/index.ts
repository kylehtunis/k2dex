// Barrel for the sampler family. Public surface mirrors what app.py
// imports from sampling.py:
//
//   loadModel             — fetch + parse precompute artifacts
//   meanfieldMarginals    — fast-path step 1
//   greedyOptimize        — fast-path step 2 + /analysis critique
//   swapMcmc              — single-chain MCMC
//   parallelTemperedMcmc  — /completer "full sampler" mode
//   rankSingleSwaps       — /analysis top-N independent swaps
//   teamEnergy            — H(s), exposed so render layer can sign-flip
//
// Sampler-internal helpers (initChain, localSwapStep, etc.) are
// available via the module path but not re-exported here.

export type {
  IsingModel,
  TeamCounts,
  TeamIndices,
  SamplerInputs,
  GreedyChainEntry,
  MeanfieldResult,
  SwapMcmcResult,
  PTResult,
  SingleSwapEntry,
} from "./types";

export { loadModel, loadTeamCounts, unpackLowerTriangle } from "./model";
export { Rng } from "./rng";
export { teamEnergy, buildConstraintSets, availableIndices } from "./energy";
export { meanfieldMarginals } from "./meanfield";
export { greedyOptimize } from "./greedy";
export { swapMcmc } from "./swap";
export { parallelTemperedMcmc } from "./pt";
export { rankSingleSwaps } from "./rank";
