// Shared types for the sampler family.
//
// Energy convention is Hamiltonian-space throughout this module:
// lower energy = more probable. Sign-flip to "Score" happens at the
// render layer only, exactly as in the Python sampling.py.

export interface IsingModel {
  /** Vocab size (number of features). */
  readonly V: number;
  /** Team size constraint (always 6 for VGC). */
  readonly teamSize: number;
  /** Vocab display strings (Phase 2: bare species; Phase 3: "Species @ Item"). */
  readonly vocab: readonly string[];
  /** Per-feature species name, for uniqueness constraints. */
  readonly speciesOf: readonly string[];
  /** Per-feature item name; null for itemless features. */
  readonly itemOf: readonly (string | null)[];
  /** Empirical per-feature marginal (Float32 promoted to Float64 on load). */
  readonly m: Float64Array;
  /** Pairwise couplings, flat row-major V*V, symmetric, zero diagonal. */
  readonly J: Float64Array;
  /** Bias / field, length V. */
  readonly h: Float64Array;
  /** Map vocab string -> index, built once on load. */
  readonly indexOf: ReadonlyMap<string, number>;
  /** Number of corpus teams the model was fit on. */
  readonly nCorpusTeams: number;
  /** Display name of the model ("species" | "species_item"). */
  readonly name: string;
}

/** Corpus team-count index. Keys are sorted-index "-"-joined strings
 * matching the JSON output of `precompute.serialize_team_counts`. */
export type TeamCounts = Map<string, number>;

/** Common sampler input shape. Most samplers take this plus algorithm-
 * specific knobs (T, n_steps, etc.). */
export interface SamplerInputs {
  /** Pinned features (must be on the team). */
  fixed: readonly number[];
  /** Banned features (must be off the team). */
  excluded: readonly number[];
  /** Scales h: 0 = pure pairwise (no popularity prior), 1 = full popularity. */
  fieldWeight: number;
}

/** A team state as a sorted index list. Used for sample collection and
 * uniqueness hashing (cheaper than carrying full V-length booleans). */
export type TeamIndices = readonly number[];

export interface GreedyChainEntry {
  step: number;
  outIdx: number;
  inIdx: number;
  deltaEAdj: number;
  energyAdjAfter: number;
  energyRawAfter: number;
  sumJAfter: number;
  teamAfter: TeamIndices;
}

export interface MeanfieldResult {
  /** Per-feature marginal after the damped fixed-point. */
  marginals: Float64Array;
  /** True where candidate is eligible to fill the remaining team slots. */
  validMask: Uint8Array;
  /** Number of iterations actually performed (<= nIters). */
  iters: number;
}

export interface SwapMcmcResult {
  /** Length-`nSteps` array of sorted team indices, one per sweep. */
  samples: TeamIndices[];
  /** Fraction of proposed (not pre-rejected) MH steps that were accepted. */
  acceptRate: number;
}

export interface PTResult {
  /** Cold-chain samples post-burn-in. */
  samples: TeamIndices[];
  /** Mean local MH acceptance across all chains. */
  localAccept: number;
  /** Mean replica-exchange acceptance. */
  swapAccept: number;
}

export interface SingleSwapEntry {
  outIdx: number;
  inIdx: number;
  deltaEAdj: number;
  deltaERaw: number;
  deltaSumJ: number;
}
