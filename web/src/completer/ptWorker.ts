// Parallel-tempered MCMC worker.
//
// Runs in a Web Worker so the main thread stays responsive during the
// 10–30s PT sample. Aggregates samples into a frequency distribution
// before posting back — saves bandwidth and matches Python's behavior
// (sampling.py emits raw samples; app.py:parallel_tempered_distribution
// folds them into a Counter, which is what /completer renders).

import { parallelTemperedMcmc } from "../sampler/pt";
import type { IsingModel, TeamIndices } from "../sampler/types";

export interface PTRequest {
  /** Slim view of the model. Worker reconstructs an IsingModel-shaped
   * value from these (it doesn't need vocab/indexOf for the math). */
  modelData: {
    V: number;
    teamSize: number;
    J: Float64Array;
    h: Float64Array;
    m: Float64Array;
    vocab: readonly string[];
    speciesOf: readonly string[];
    itemOf: readonly (string | null)[];
  };
  fixed: readonly number[];
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

self.onmessage = (e: MessageEvent<PTRequest>) => {
  const req = e.data;
  try {
    const model: IsingModel = {
      V: req.modelData.V,
      teamSize: req.modelData.teamSize,
      vocab: req.modelData.vocab,
      speciesOf: req.modelData.speciesOf,
      itemOf: req.modelData.itemOf,
      m: req.modelData.m,
      J: req.modelData.J,
      h: req.modelData.h,
      indexOf: new Map(), // not used by the sampler
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
        excluded: req.excluded,
        fieldWeight: req.fieldWeight,
        tLadder: ladder,
        nSteps: req.nSteps,
        burnIn: req.burnIn,
        swapInterval: req.swapInterval,
        seed: req.seed + run, // independent stream per run
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
    const { dist, nKept } = aggregate(allSamples);
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
