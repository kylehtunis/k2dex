// Front-end mirror of app.py module-level constants. These are the
// locked completer/analysis/meta knobs — not user-tunable in the v1
// webapp (kept here so any future re-introduction has one place to
// touch). Phase 2/3 fit-time constants live in the per-model
// meta.json instead (loaded at runtime via loadModel).

export const CURRENT_REGULATION = "M-A";

export const TEAM_SIZE = 6;

export const FIELD_WEIGHT_OPTIONS = [
  0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
] as const;

export const TEMPERATURE_OPTIONS = [
  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0,
] as const;

export const TOP_COMPLETIONS = 10;
export const TOP_SINGLE_SWAPS = 10;
export const GREEDY_MAX_SWAPS = 10;
export const META_TOP_FEATURES = 25;
export const META_TOP_PAIRS = 25;

export const PT_HOT_T = 3.0;
export const PT_LADDER_LEVELS = 7;
export const PT_RUNS = 10;
export const PT_SWEEPS = 20000;
export const PT_BURN_IN = 5000;
export const PT_SWAP_INTERVAL = 10;

export const MF_MAX_ITERS = 200;
export const MF_TOL = 1e-5;
