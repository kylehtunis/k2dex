// Parallel-tempered MCMC worker.
//
// Runs in a Web Worker so the main thread stays responsive during the
// 10–30s PT sample. Aggregates samples into a frequency distribution
// before posting back — saves bandwidth and matches Python's behavior
// (sampling.py emits raw samples; app.py:parallel_tempered_distribution
// folds them into a Counter, which is what /completer renders).

import { deriveFactored } from "../sampler/model";
import { parallelTemperedMcmc } from "../sampler/pt";
import type { IsingModel, TeamIndices, TrackDef } from "../sampler/types";

export interface PTRequest {
  /** Slim view of the model. Worker reconstructs an IsingModel-shaped
   * value from these (it doesn't need vocab/indexOf for the math). The
   * factored fields (sites/siteOf/tracks/trackValues) drive the Potts moves;
   * speciesOf/itemOf/siteFeatures are rederived from them in the worker. */
  modelData: {
    V: number;
    teamSize: number;
    J: Float64Array;
    h: Float64Array;
    m: Float64Array;
    vocab: readonly string[];
    sites: readonly string[];
    siteOf: readonly number[];
    tracks: readonly TrackDef[];
    trackValues: readonly (readonly (string | null)[])[];
  };
  fixed: readonly number[];
  /** Site-level pins (species fixed, some/all tracks free to reroll). */
  fixedSites?: readonly number[];
  /** Per-`fixedSites` entry, its per-track pinned value (partial pins); null /
   * null-track = free. */
  sitePinTrackValues?: readonly (readonly (string | null)[] | null)[];
  excluded: readonly number[];
  fieldWeight: number;
  /** Cold target T (smallest), at index 0 in the rebuilt ladder. */
  coldT: number;
  /** Hot end-cap T (largest). */
  hotT: number;
  /** Number of temperature rungs between cold and hot (inclusive). */
  ladderLevels: number;
  /** Number of independent PT runs to aggregate. */
  nRuns: number;
  nSteps: number;
  burnIn: number;
  swapInterval: number;
  seed: number;
  /** Item-track reroll probability per sweep (ignored for species-only). */
  pReroll?: number;
  /** Anchor-field tilt alpha ("Anchor Strength"); 1 = no tilt. */
  anchorStrength?: number;
  /** Deactivated attribute-track indices. Their reroll is disabled and they are
   * marginalized out of the completion distribution: completions bucket by
   * (site, active-track values), each bucket keeping its most-frequent real
   * feature-team as the representative. Empty ⇒ bucket by full feature set. */
  inactiveTracks?: readonly number[];
}

export interface PTResponse {
  ok: true;
  /** Sorted descending by count. Each `team` is sorted ascending indices. */
  dist: Array<{ team: number[]; count: number }>;
  nKept: number;
  localAccept: number;
  swapAccept: number;
  ladder: number[];
}

export interface PTError {
  ok: false;
  message: string;
}

function buildLadder(coldT: number, hotT: number, levels: number): number[] {
  if (levels < 2) return [coldT, hotT];
  const out: number[] = [];
  for (let k = 0; k < levels; k++) {
    const frac = k / (levels - 1);
    out.push(coldT * Math.pow(hotT / coldT, frac));
  }
  return out;
}

function aggregate(
  allSamples: TeamIndices[][],
): { dist: Array<{ team: number[]; count: number }>; nKept: number } {
  const counts = new Map<string, { team: number[]; count: number }>();
  let total = 0;
  for (const samples of allSamples) {
    for (const team of samples) {
      total++;
      const sorted = [...team].sort((a, b) => a - b);
      const key = sorted.join(",");
      const entry = counts.get(key);
      if (entry) entry.count++;
      else counts.set(key, { team: sorted, count: 1 });
    }
  }
  const dist = [...counts.values()].sort((a, b) => b.count - a.count);
  return { dist, nKept: total };
}

/** Like `aggregate`, but buckets by each member's projected key (site plus its
 * value on every *active* track) — the inactive tracks are marginalized out.
 * Each bucket's `team` is its most-frequent real feature-team (a genuine sampled
 * completion), so downstream observables and corpus lookups stay meaningful; the
 * UI hides the inactive tracks' columns. With no active tracks this is the
 * species-set projection; with every track active it matches `aggregate`. */
function aggregateProjected(
  allSamples: TeamIndices[][],
  siteOf: readonly number[],
  trackValues: readonly (readonly (string | null)[])[],
  activeTracks: readonly number[],
): { dist: Array<{ team: number[]; count: number }>; nKept: number } {
  const memberKey = (f: number): string =>
    `${siteOf[f]}#${activeTracks.map((t) => trackValues[f][t] ?? "").join("|")}`;
  interface Bucket {
    count: number;
    reps: Map<string, { team: number[]; count: number }>;
  }
  const buckets = new Map<string, Bucket>();
  let total = 0;
  for (const samples of allSamples) {
    for (const team of samples) {
      total++;
      const sorted = [...team].sort((a, b) => a - b);
      const projKey = sorted.map(memberKey).sort().join(",");
      let bucket = buckets.get(projKey);
      if (!bucket) {
        bucket = { count: 0, reps: new Map() };
        buckets.set(projKey, bucket);
      }
      bucket.count++;
      const featKey = sorted.join(",");
      const rep = bucket.reps.get(featKey);
      if (rep) rep.count++;
      else bucket.reps.set(featKey, { team: sorted, count: 1 });
    }
  }
  const dist = [...buckets.values()]
    .map((b) => {
      let best = { team: [] as number[], count: -1 };
      for (const rep of b.reps.values()) if (rep.count > best.count) best = rep;
      return { team: best.team, count: b.count };
    })
    .sort((a, b) => b.count - a.count);
  return { dist, nKept: total };
}

self.onmessage = (e: MessageEvent<PTRequest>) => {
  const req = e.data;
  try {
    const md = req.modelData;
    const { siteFeatures, speciesOf, itemOf } = deriveFactored(
      md.sites,
      md.siteOf,
      md.trackValues,
    );
    const model: IsingModel = {
      id: "",
      displayName: "",
      regulation: "",
      latestTournamentDate: "",
      V: md.V,
      teamSize: md.teamSize,
      vocab: md.vocab,
      sites: md.sites,
      siteOf: md.siteOf,
      tracks: md.tracks,
      trackValues: md.trackValues,
      siteFeatures,
      speciesOf,
      itemOf,
      m: md.m,
      J: md.J,
      h: md.h,
      indexOf: new Map(),
      nCorpusTeams: 0,
      name: "",
    };
    const ladder = buildLadder(req.coldT, req.hotT, req.ladderLevels);
    const allSamples: TeamIndices[][] = [];
    let localAcceptSum = 0;
    let swapAcceptSum = 0;
    for (let run = 0; run < req.nRuns; run++) {
      const res = parallelTemperedMcmc(model, {
        fixed: req.fixed,
        fixedSites: req.fixedSites,
        sitePinTrackValues: req.sitePinTrackValues,
        excluded: req.excluded,
        fieldWeight: req.fieldWeight,
        tLadder: ladder,
        nSteps: req.nSteps,
        burnIn: req.burnIn,
        swapInterval: req.swapInterval,
        seed: req.seed + run, // independent stream per run
        pReroll: req.pReroll,
        anchorStrength: req.anchorStrength,
        inactiveTracks: req.inactiveTracks,
      });
      if (res === null) {
        const reply: PTError = {
          ok: false,
          message: "Not enough available Pokemon to fill the team after applying constraints.",
        };
        (self as unknown as Worker).postMessage(reply);
        return;
      }
      allSamples.push(res.samples);
      localAcceptSum += res.localAccept;
      swapAcceptSum += res.swapAccept;
    }
    const inactive = new Set(req.inactiveTracks ?? []);
    const activeTracks = md.tracks
      .map((_t, t) => t)
      .filter((t) => !inactive.has(t));
    const { dist, nKept } = inactive.size > 0
      ? aggregateProjected(allSamples, md.siteOf, md.trackValues, activeTracks)
      : aggregate(allSamples);
    const reply: PTResponse = {
      ok: true,
      dist,
      nKept,
      localAccept: localAcceptSum / req.nRuns,
      swapAccept: swapAcceptSum / req.nRuns,
      ladder,
    };
    (self as unknown as Worker).postMessage(reply);
  } catch (err: unknown) {
    const reply: PTError = {
      ok: false,
      message: err instanceof Error ? err.message : String(err),
    };
    (self as unknown as Worker).postMessage(reply);
  }
};
