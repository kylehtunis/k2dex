// Front-end mirror of app.py module-level constants. These are the
// locked completer/analysis/meta knobs — not user-tunable in the v1
// webapp (kept here so any future re-introduction has one place to
// touch). Phase 2/3 fit-time constants live in the per-model
// meta.json instead (loaded at runtime via loadModel).

export const CURRENT_REGULATION = "M-B";

export const TEAM_SIZE = 6;

export const TEMPERATURE_OPTIONS = [
  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0,
] as const;

export const TOP_COMPLETIONS = 10;
export const TOP_SINGLE_SWAPS = 10;
export const GREEDY_MAX_SWAPS = 10;
export const META_TOP_PAIRS = 25;
export const META_TOP_TEAMS = 20;

// Anchor Strength (anchor-field tilt alpha): how strongly completions build
// around the pinned Pokémon. 1 = the untilted model conditional.
export const DEFAULT_ANCHOR = 1.0;
export const ANCHOR_MIN = 1.0;
export const ANCHOR_MAX = 3.0;
export const ANCHOR_STEP = 0.1;
/** The article MiniCompleter's default alpha. Tuned on a headless niche-pin
 * sweep (Camerupt / Vivillon / Torkoal / Crabominable, reg-m-a): 2.0 makes the
 * completion decisively pin-centric without degrading raw score or variety;
 * 3.0 starts pulling in fringe partners. */
export const ANCHOR_ARTICLE_DEFAULT = 2.0;

export const PT_HOT_T = 3.0;
export const PT_LADDER_LEVELS = 7;
export const PT_RUNS = 10;
export const PT_SWEEPS = 20000;
export const PT_BURN_IN = 5000;
export const PT_SWAP_INTERVAL = 10;

export const MF_MAX_ITERS = 200;
export const MF_TOL = 1e-5;

// Potts move kernel (per-track reroll allocation). Mirror of the Python
// POTTS_* constants in k2dex/constants.py. Each per-chain sweep is a species
// swap, or with probability POTTS_REROLL_PROB a track (attribute) reroll; the
// track to reroll is picked weighted by POTTS_TRACK_REROLL_WEIGHTS (relative
// weights by track name, need not sum to 1). Item outweighs ability because it
// carries most of the modelled signal. Tunable.
export const POTTS_REROLL_PROB = 0.5;
export const POTTS_TRACK_REROLL_WEIGHTS: Record<string, number> = {
  item: 0.8,
  ability: 0.2,
};

// /meta drill-down support gate: minimum corpus appearances for a cell to be
// shown in the hierarchical drill-down. Mirror of META_SUPPORT_MIN_COUNT.
export const META_SUPPORT_MIN_COUNT = 5;
