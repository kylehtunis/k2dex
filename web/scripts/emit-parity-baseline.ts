// Emit a parity baseline that the Python test suite loads and verifies.
//
// Constructs a small synthetic IsingModel deterministically, runs the
// JS sampler from fixed inputs, writes ../../tests/parity_baseline.json
// at the repo root. Python's tests/test_parity.py reads it and asserts
// Python's sampling.py reproduces the same outputs within tolerance.
//
// Run: cd web && npx vite-node scripts/emit-parity-baseline.ts
//
// The synthetic model is hand-constructed (not RNG-generated) so it's
// language-agnostic. V is small (~12) so MCMC/PT runs are cheap in
// both implementations.

import { writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import type { IsingModel, TeamCounts } from "../src/sampler/types";
import { meanfieldMarginals } from "../src/sampler/meanfield";
import { greedyOptimize } from "../src/sampler/greedy";
import { rankSingleSwaps } from "../src/sampler/rank";
import { nearestObserved, teamKey } from "../src/render/corpus";
import { speciesToSlug } from "../src/render/sprite-url";
import { intraTeamSumJ, pairwiseJRows } from "../src/render/observables";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, "..", "..");
const OUT_PATH = resolve(REPO_ROOT, "tests", "parity_baseline.json");

// --- Synthetic model (Phase 3-shaped: speciesOf + itemOf set) ----------

const V = 12;
const TEAM_SIZE = 4;

// Pair structure: 6 species × 2 item variants. Indices 0..11 layout:
//   0,1 -> species "A", items null / "x"
//   2,3 -> species "B", items null / "y"
//   4,5 -> species "C", items null / "x"
//   6,7 -> species "D", items null / "y"
//   8,9 -> species "E", items null / "z"
//   10,11 -> species "F", items null / null   (two itemless variants of F)
const SPECIES = ["A", "B", "C", "D", "E", "F"];
const SPECIES_OF: string[] = [];
const ITEM_OF: (string | null)[] = [];
for (let s = 0; s < 6; s++) {
  SPECIES_OF.push(SPECIES[s], SPECIES[s]);
}
ITEM_OF.push(null, "x"); // A
ITEM_OF.push(null, "y"); // B
ITEM_OF.push(null, "x"); // C
ITEM_OF.push(null, "y"); // D
ITEM_OF.push(null, "z"); // E
ITEM_OF.push(null, "w"); // F — second variant gets an item so vocab strings stay unique
                          // (Python's nearest_observed keys on vocab strings; real corpora
                          // never have duplicate vocab entries by construction).

// Hand-built J: smooth analytic function so JS and Python both compute
// it byte-identically. Symmetric, zero diagonal.
function buildJ(): Float64Array {
  const J = new Float64Array(V * V);
  for (let i = 0; i < V; i++) {
    for (let j = 0; j < i; j++) {
      const v = 0.5 * Math.sin(0.7 * (i + 1)) * Math.cos(0.3 * (j + 1));
      J[i * V + j] = v;
      J[j * V + i] = v;
    }
  }
  return J;
}

// Hand-built h: linear ramp, mixed signs.
function buildH(): Float64Array {
  const h = new Float64Array(V);
  for (let i = 0; i < V; i++) {
    h[i] = -0.6 + 0.1 * i;
  }
  return h;
}

// Per-feature "empirical" marginal: not load-bearing for parity tests,
// but the IsingModel type requires it. Use a smooth function.
function buildM(): Float64Array {
  const m = new Float64Array(V);
  for (let i = 0; i < V; i++) {
    m[i] = 0.05 + 0.04 * i;
  }
  return m;
}

const J = buildJ();
const h = buildH();
const m = buildM();

const vocab = SPECIES_OF.map((s, i) => {
  const it = ITEM_OF[i];
  return it === null ? s : `${s} @ ${it}`;
});

const indexOf = new Map<string, number>();
for (let i = 0; i < vocab.length; i++) indexOf.set(vocab[i], i);

const model: IsingModel = {
  V,
  teamSize: TEAM_SIZE,
  vocab,
  speciesOf: SPECIES_OF,
  itemOf: ITEM_OF,
  m,
  J,
  h,
  indexOf,
  nCorpusTeams: 0,
  name: "synthetic",
};

// --- Test cases --------------------------------------------------------

interface MfCase {
  name: string;
  input: {
    fixed: number[];
    excluded: number[];
    fieldWeight: number;
    nIters: number;
    tol: number;
    damp: number;
  };
  expected: {
    marginals: number[];
    validMask: number[];
    iters: number;
  } | null;
}

interface GreedyCase {
  name: string;
  input: {
    startingTeam: number[];
    pinned: number[];
    excluded: number[];
    fieldWeight: number;
    maxSwaps: number;
  };
  expected: {
    finalTeam: number[];
    chain: Array<{
      step: number;
      outIdx: number;
      inIdx: number;
      deltaEAdj: number;
      energyAdjAfter: number;
      energyRawAfter: number;
      sumJAfter: number;
      teamAfter: number[];
    }>;
  };
}

interface RankCase {
  name: string;
  input: {
    team: number[];
    fieldWeight: number;
    topN: number;
  };
  expected: Array<{
    outIdx: number;
    inIdx: number;
    deltaEAdj: number;
    deltaERaw: number;
    deltaSumJ: number;
  }>;
}

const mfCases: MfCase[] = [];
for (const c of [
  { name: "mf_fw_1.0_no_pins", fixed: [], excluded: [], fieldWeight: 1.0 },
  { name: "mf_fw_0.5_one_pin", fixed: [0], excluded: [11], fieldWeight: 0.5 },
  { name: "mf_fw_0.0_pin_uniqueness", fixed: [0, 2], excluded: [], fieldWeight: 0.0 },
]) {
  const opts = {
    fixed: c.fixed,
    excluded: c.excluded,
    fieldWeight: c.fieldWeight,
    nIters: 200,
    tol: 1e-5,
    damp: 0.5,
  };
  const r = meanfieldMarginals(model, opts);
  mfCases.push({
    name: c.name,
    input: opts,
    expected: r === null ? null : {
      marginals: Array.from(r.marginals),
      validMask: Array.from(r.validMask),
      iters: r.iters,
    },
  });
}

const greedyCases: GreedyCase[] = [];
for (const c of [
  {
    name: "greedy_fw_1.0_no_pins",
    startingTeam: [0, 2, 4, 6],
    pinned: [],
    excluded: [],
    fieldWeight: 1.0,
  },
  {
    name: "greedy_fw_0.5_pinned",
    startingTeam: [0, 2, 4, 6],
    pinned: [0],
    excluded: [11],
    fieldWeight: 0.5,
  },
  {
    name: "greedy_fw_0.0_pure_J",
    startingTeam: [1, 3, 5, 7],
    pinned: [],
    excluded: [],
    fieldWeight: 0.0,
  },
]) {
  const opts = { ...c, maxSwaps: 10 };
  const r = greedyOptimize(model, opts);
  greedyCases.push({
    name: c.name,
    input: opts,
    expected: {
      finalTeam: r.finalTeam,
      chain: r.chain.map((e) => ({ ...e, teamAfter: [...e.teamAfter] })),
    },
  });
}

const rankCases: RankCase[] = [];
for (const c of [
  { name: "rank_fw_1.0", team: [0, 2, 4, 6], fieldWeight: 1.0, topN: 10 },
  { name: "rank_fw_0.5", team: [1, 3, 5, 7], fieldWeight: 0.5, topN: 10 },
]) {
  const r = rankSingleSwaps(model, c);
  rankCases.push({ name: c.name, input: c, expected: r });
}

// --- intra_team_sum_j + pairwise_j_rows cases ------------------------

interface ObsCase {
  name: string;
  team: number[];
  expected: {
    intraTeamSumJ: number;
    pairwise: Array<{
      rank: number;
      idxA: number;
      idxB: number;
      jValue: number;
      pctOfAbsSum: number;
    }>;
  };
}

const obsCases: ObsCase[] = [];
for (const c of [
  { name: "obs_team_a", team: [0, 2, 4, 6] },
  { name: "obs_team_b", team: [1, 3, 5, 7] },
  { name: "obs_team_mixed", team: [0, 3, 4, 9] },
]) {
  const rows = pairwiseJRows(c.team, vocab, J, V);
  const intra = intraTeamSumJ(J, V, c.team);
  obsCases.push({
    name: c.name,
    team: c.team,
    expected: {
      intraTeamSumJ: intra,
      pairwise: rows.map((r) => ({
        rank: r.rank,
        idxA: r.idxA,
        idxB: r.idxB,
        jValue: r.jValue,
        pctOfAbsSum: r.pctOfAbsSum,
      })),
    },
  });
}

// --- nearest_observed cases ------------------------------------------

// Hand-built tiny corpus. Three rosters, with one having a clear
// neighbor at delta=1. Keys are sorted-index "-"-joined.
const corpusEntries: Array<{ team: number[]; count: number }> = [
  { team: [0, 2, 4, 6], count: 10 },
  { team: [0, 2, 4, 8], count: 3 },   // delta=1 from above (swap 6 → 8)
  { team: [1, 3, 5, 7], count: 7 },
];
const teamCounts: TeamCounts = new Map();
for (const e of corpusEntries) teamCounts.set(teamKey(e.team), e.count);

interface CorpusCase {
  name: string;
  team: number[];
  expected: { delta: number; count: number } | null;
}

// --- speciesToSlug cases ---------------------------------------------

const slugCases: Array<{ input: string; expected: string }> = [
  "Calyrex-Shadow",
  "Blastoise-Mega",
  "Arcanine-Hisui",
  "Urshifu-Rapid-Strike",
  "Chien-Pao",
  "Ho-Oh",
  "Porygon-Z",
  "Iron Hands",
  "Farfetch'd",
  "Mr. Mime",
  "Type: Null",
  "Nidoran-F",
  "Tapu Koko",
  "Necrozma-Dawn-Wings",
  "Eternal Flower Floette",
].map((input) => ({ input, expected: speciesToSlug(input) }));

const corpusCases: CorpusCase[] = [];
for (const c of [
  { name: "corpus_exact_match", team: [0, 2, 4, 6] },             // delta=0, count=10
  { name: "corpus_one_swap", team: [0, 2, 4, 10] },               // delta=1, nearest is [0,2,4,6] count=10 over [0,2,4,8] count=3
  { name: "corpus_two_swaps_tiebreak", team: [0, 2, 9, 11] },     // delta=2 from [0,2,4,6] (count=10) and [0,2,4,8] (count=3); tiebreak by count
  { name: "corpus_far", team: [10, 11, 6, 7] },                   // larger distance
]) {
  const r = nearestObserved(c.team, teamCounts);
  corpusCases.push({ name: c.name, team: c.team, expected: r });
}

// --- Write baseline ---------------------------------------------------

const baseline = {
  schema_version: 1,
  description: "Parity baseline emitted by web/scripts/emit-parity-baseline.ts. "
    + "Python tests/test_parity.py runs the same inputs through sampling.py and "
    + "asserts agreement within tolerance. Regenerate after modifying either side.",
  model: {
    V,
    teamSize: TEAM_SIZE,
    vocab,
    speciesOf: SPECIES_OF,
    itemOf: ITEM_OF,
    J: Array.from(J),
    h: Array.from(h),
    m: Array.from(m),
  },
  mf: mfCases,
  greedy: greedyCases,
  rank: rankCases,
  corpus: {
    rosters: corpusEntries,
    cases: corpusCases,
  },
  slugs: slugCases,
  obs: obsCases,
};

mkdirSync(dirname(OUT_PATH), { recursive: true });
writeFileSync(OUT_PATH, JSON.stringify(baseline, null, 2));
console.log(`Wrote ${OUT_PATH}`);
console.log(`  ${mfCases.length} MF cases, ${greedyCases.length} greedy cases, ${rankCases.length} rank cases, ${corpusCases.length} corpus cases, ${slugCases.length} slug cases, ${obsCases.length} obs cases`);
