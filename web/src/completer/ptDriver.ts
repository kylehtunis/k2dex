// Main-thread driver for the PT worker. Wraps the postMessage /
// onmessage dance in a Promise so callers can `await run.promise`.
//
// Each call spawns a fresh worker, runs once, terminates. Cheaper
// than keeping a long-lived worker around since the PT run dominates
// the lifecycle anyway.
//
// A run must be cancellable: it takes seconds, and its result is only valid
// for the model it was launched against. Callers hold the returned handle and
// cancel it when they unmount or switch models, otherwise a landing result
// would be applied to a model whose vocab indices mean something else.

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
  /** Set when the caller cancelled the run. The result is meaningless and
   * must not be rendered or reported as an error — the caller that cancelled
   * owns whatever state comes next. */
  cancelled?: boolean;
}

export interface PTRun {
  promise: Promise<PTSuccess | PTFailure>;
  /** Terminate the worker and settle the promise as cancelled. Idempotent. */
  cancel(): void;
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
): PTRun {
  let settle!: (r: PTSuccess | PTFailure) => void;
  const promise = new Promise<PTSuccess | PTFailure>((resolve) => {
    settle = resolve;
  });

  const worker = new Worker(
    new URL("./ptWorker.ts", import.meta.url),
    { type: "module" },
  );
  let done = false;
  const finish = (r: PTSuccess | PTFailure) => {
    if (done) return;
    done = true;
    worker.terminate();
    settle(r);
  };

  worker.onmessage = (e: MessageEvent<PTResponse | PTError>) => finish(e.data);
  worker.onerror = (ev) =>
    finish({ ok: false, message: ev.message || "worker error" });

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

  return {
    promise,
    cancel: () => finish({ ok: false, message: "cancelled", cancelled: true }),
  };
}
