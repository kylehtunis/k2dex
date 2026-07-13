// Main-thread driver for the PT worker. Wraps the postMessage /
// onmessage dance in a Promise so callers can `await runPT(...)`.
//
// Each call spawns a fresh worker, runs once, terminates. Cheaper
// than keeping a long-lived worker around since the PT run dominates
// the lifecycle anyway.

import type { IsingModel } from "../sampler/types";
import type { PTRequest, PTResponse, PTError } from "./ptWorker";

export interface PTDistEntry {
  team: number[];
  count: number;
}

export interface PTSuccess {
  ok: true;
  dist: PTDistEntry[];
  nKept: number;
  localAccept: number;
  swapAccept: number;
  ladder: number[];
}

export interface PTFailure {
  ok: false;
  message: string;
}

export function runPT(
  model: IsingModel,
  opts: {
    fixed: readonly number[];
    fixedSites?: readonly number[];
    excluded: readonly number[];
    fieldWeight: number;
    coldT: number;
    hotT: number;
    ladderLevels: number;
    nRuns: number;
    nSteps: number;
    burnIn: number;
    swapInterval: number;
    seed?: number;
    pReroll?: number;
    anchorStrength?: number;
    projectToSites?: boolean;
  },
): Promise<PTSuccess | PTFailure> {
  return new Promise((resolve) => {
    const worker = new Worker(
      new URL("./ptWorker.ts", import.meta.url),
      { type: "module" },
    );
    worker.onmessage = (e: MessageEvent<PTResponse | PTError>) => {
      worker.terminate();
      resolve(e.data);
    };
    worker.onerror = (ev) => {
      worker.terminate();
      resolve({ ok: false, message: ev.message || "worker error" });
    };

    const req: PTRequest = {
      modelData: {
        V: model.V,
        teamSize: model.teamSize,
        J: model.J,
        h: model.h,
        m: model.m,
        vocab: model.vocab,
        sites: model.sites,
        siteOf: model.siteOf,
        tracks: model.tracks,
        trackValues: model.trackValues,
      },
      fixed: opts.fixed,
      fixedSites: opts.fixedSites,
      excluded: opts.excluded,
      fieldWeight: opts.fieldWeight,
      coldT: opts.coldT,
      hotT: opts.hotT,
      ladderLevels: opts.ladderLevels,
      nRuns: opts.nRuns,
      nSteps: opts.nSteps,
      burnIn: opts.burnIn,
      swapInterval: opts.swapInterval,
      seed: opts.seed ?? 0x5eed,
      pReroll: opts.pReroll,
      anchorStrength: opts.anchorStrength,
      projectToSites: opts.projectToSites,
    };
    worker.postMessage(req);
  });
}
