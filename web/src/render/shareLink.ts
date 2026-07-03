// Shareable-URL token codec for the Analysis and Completer pages.
//
// State is encoded by slug (not raw vocab index) so links survive corpus
// rebuilds (precompute.py reorders vocab). Tokens use URL-unreserved
// separators (. _ ~) so they never need percent-encoding, and slugs only
// contain [a-z0-9-]. base64/gzip would inflate a payload this short, so
// the readable delimited form is also the most wieldy.
//
// Core token `t` (shared by both pages): "<modelSlug>.<fwIndex>.<mons>"
//   <modelSlug>  the model's id slug, now a per-regulation slug (e.g. "reg-m-a")
//   <fwIndex>    index into FIELD_WEIGHT_OPTIONS
//   <mons>       "_"-joined; each mon is one of:
//                  "speciesSlug~itemSlug"  a feature pin (species + item locked;
//                                          itemless builds encode "...~none")
//                  "speciesSlug"           a site pin (species locked, item free)
//                Feature mons come first (ascending index) so pre-site-pin
//                tokens stay byte-identical; site mons are appended.
//
// Legacy model codes (the short "s"/"si" and the retired split-model slugs)
// decode to the unified per-regulation slug for backward compatibility.
//
// Completer adds default-omitting params: x (excluded), g (greedy),
// tmp (temperature index), a (advanced PT knobs), seed (reproduces the
// exact PT run while inputs still match it).

import {
  FIELD_WEIGHT_OPTIONS,
  TEMPERATURE_OPTIONS,
  PT_RUNS,
  PT_LADDER_LEVELS,
  PT_SWEEPS,
  PT_SWAP_INTERVAL,
} from "../constants";
import { speciesToSlug, itemToSlug } from "./sprite-url";
import type { IsingModel } from "../sampler/types";

const LEGACY_CODE_TO_SLUG: Record<string, string> = {
  s: "reg-m-a",
  si: "reg-m-a",
  "reg-m-a-species": "reg-m-a",
  "reg-m-a-species-item": "reg-m-a",
  "reg-m-a-species-item-weighted": "reg-m-a",
  "reg-m-b-experimental": "reg-m-b",
  "reg-m-b-species-item-boltzmann": "reg-m-b",
};

const DEFAULT_TEMPERATURE = 0.5;

export interface FeatureSlug {
  speciesSlug: string;
  /** null = a bare mon = site-level pin (species locked, item free); a string
   * (incl. "none") = a feature pin with that item locked. */
  itemSlug: string | null;
}

function featureSlug(model: IsingModel, i: number): string {
  const sp = speciesToSlug(model.speciesOf[i]);
  const it = model.itemOf[i];
  return it === null ? sp : `${sp}~${itemToSlug(it)}`;
}

function fieldWeightIndex(fieldWeight: number): number {
  const exact = (FIELD_WEIGHT_OPTIONS as readonly number[]).indexOf(fieldWeight);
  if (exact >= 0) return exact;
  let best = 0;
  let bestD = Infinity;
  FIELD_WEIGHT_OPTIONS.forEach((v, i) => {
    const d = Math.abs(v - fieldWeight);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

/** Build the shared core token from feature pins (`idxs`) and optional site
 * pins (`siteIdxs`, site indices). Feature mons keep ascending-index order and
 * come first; site mons are appended, so a feature-only token is unchanged. */
export function encodeCore(
  modelId: string,
  fieldWeight: number,
  idxs: readonly number[],
  model: IsingModel,
  siteIdxs: readonly number[] = [],
): string {
  const featureMons = [...idxs]
    .sort((a, b) => a - b)
    .map((i) => featureSlug(model, i));
  const siteMons = [...siteIdxs]
    .sort((a, b) => a - b)
    .map((s) => speciesToSlug(model.sites[s]));
  const mons = [...featureMons, ...siteMons].join("_");
  return `${modelId}.${fieldWeightIndex(fieldWeight)}.${mons}`;
}

export interface DecodedCore {
  modelId: string;
  fieldWeight: number;
  features: FeatureSlug[];
}

function parseMon(s: string): FeatureSlug {
  const tilde = s.indexOf("~");
  return tilde < 0
    ? { speciesSlug: s, itemSlug: null }
    : { speciesSlug: s.slice(0, tilde), itemSlug: s.slice(tilde + 1) };
}

/** Resolve the model slug from the first token segment, handling legacy codes. */
function resolveModelSlug(code: string): string | null {
  if (LEGACY_CODE_TO_SLUG[code]) return LEGACY_CODE_TO_SLUG[code];
  if (/^[a-z0-9_-]+$/.test(code)) return code;
  return null;
}

/** Decode the shared core token; null on a malformed/empty token. */
export function decodeCore(t: string | null): DecodedCore | null {
  if (!t) return null;
  const parts = t.split(".");
  if (parts.length < 2) return null;
  const modelId = resolveModelSlug(parts[0]);
  if (!modelId) return null;
  const fieldWeight = FIELD_WEIGHT_OPTIONS[Number(parts[1])] ?? 0.3;
  const monsStr = parts.slice(2).join(".");
  const features = monsStr
    ? monsStr.split("_").filter(Boolean).map(parseMon)
    : [];
  return { modelId, fieldWeight, features };
}

export interface CompleterShareState {
  modelId: string;
  fieldWeight: number;
  fixedIdxs: readonly number[];
  /** Site-level pins (site indices). Encoded as bare species slugs. */
  fixedSites: readonly number[];
  /** Deactivated attribute-track indices (species-only mode). Encoded as `d`. */
  inactiveTracks: readonly number[];
  excludedSpecies: readonly string[];
  usePT: boolean;
  temperature: number;
  ptRuns: number;
  ptLadder: number;
  ptSweeps: number;
  ptSwapInterval: number;
  /** Last PT run's seed, or null when not reproducible (greedy / stale). */
  seed: number | null;
}

/** Encode completer state to URL params, omitting empty/default extras. */
export function encodeCompleter(
  s: CompleterShareState,
  model: IsingModel,
): URLSearchParams {
  const p = new URLSearchParams();
  p.set("t", encodeCore(s.modelId, s.fieldWeight, s.fixedIdxs, model, s.fixedSites));
  if (s.excludedSpecies.length) {
    p.set(
      "x",
      [...s.excludedSpecies].map(speciesToSlug).sort().join("_"),
    );
  }
  if (s.inactiveTracks.length) {
    p.set("d", [...s.inactiveTracks].sort((a, b) => a - b).join("-"));
  }
  if (!s.usePT) {
    p.set("g", "1");
    return p;
  }
  if (s.temperature !== DEFAULT_TEMPERATURE) {
    const tmp = (TEMPERATURE_OPTIONS as readonly number[]).indexOf(s.temperature);
    if (tmp >= 0) p.set("tmp", String(tmp));
  }
  if (
    s.ptRuns !== PT_RUNS ||
    s.ptLadder !== PT_LADDER_LEVELS ||
    s.ptSweeps !== PT_SWEEPS ||
    s.ptSwapInterval !== PT_SWAP_INTERVAL
  ) {
    p.set("a", [s.ptRuns, s.ptLadder, s.ptSweeps, s.ptSwapInterval].join("-"));
  }
  if (s.seed !== null) p.set("seed", String(s.seed));
  return p;
}

export interface DecodedCompleter {
  modelId: string;
  fieldWeight: number;
  features: FeatureSlug[];
  inactiveTracks: number[];
  excludedSlugs: string[];
  usePT: boolean;
  temperature: number;
  ptRuns: number;
  ptLadder: number;
  ptSweeps: number;
  ptSwapInterval: number;
  seed: number | null;
}

/** Decode completer URL params; null when there's no valid core token. */
export function decodeCompleter(params: URLSearchParams): DecodedCompleter | null {
  const core = decodeCore(params.get("t"));
  if (!core) return null;
  const usePT = params.get("g") !== "1";
  const x = params.get("x");
  const excludedSlugs = x ? x.split("_").filter(Boolean) : [];
  const d = params.get("d");
  const inactiveTracks = d
    ? d.split("-").map(Number).filter((n) => Number.isInteger(n) && n >= 0)
    : [];

  let temperature = DEFAULT_TEMPERATURE;
  const tmp = params.get("tmp");
  if (tmp !== null) {
    const v = TEMPERATURE_OPTIONS[Number(tmp)];
    if (v !== undefined) temperature = v;
  }

  let ptRuns = PT_RUNS;
  let ptLadder = PT_LADDER_LEVELS;
  let ptSweeps = PT_SWEEPS;
  let ptSwapInterval = PT_SWAP_INTERVAL;
  const a = params.get("a");
  if (a) {
    const nums = a.split("-").map(Number);
    if (nums.length === 4 && nums.every((n) => Number.isFinite(n))) {
      [ptRuns, ptLadder, ptSweeps, ptSwapInterval] = nums;
    }
  }

  const seedStr = params.get("seed");
  const seed =
    usePT && seedStr !== null && Number.isFinite(Number(seedStr))
      ? Number(seedStr)
      : null;

  return {
    modelId: core.modelId,
    fieldWeight: core.fieldWeight,
    features: core.features,
    inactiveTracks,
    excludedSlugs,
    usePT,
    temperature,
    ptRuns,
    ptLadder,
    ptSweeps,
    ptSwapInterval,
    seed,
  };
}
