import { describe, it, expect } from "vitest";
import {
  meanPairwiseDifference,
  noveltyScore,
} from "../../completer/completionStats";
import {
  buildCorpusScoreIndex,
  ordinal,
  percentileRank,
} from "../corpusScore";
import type { IsingModel, TeamCounts } from "../../sampler/types";

describe("meanPairwiseDifference", () => {
  it("is null with fewer than two teams", () => {
    expect(meanPairwiseDifference([])).toBeNull();
    expect(meanPairwiseDifference([[0, 1, 2]])).toBeNull();
  });

  it("is 0 for identical teams and team_size for disjoint teams", () => {
    expect(meanPairwiseDifference([[0, 1, 2], [0, 1, 2]])).toBe(0);
    expect(meanPairwiseDifference([[0, 1, 2], [3, 4, 5]])).toBe(3);
  });

  it("averages over all pairs", () => {
    // Pairs: (a,b) differ 1, (a,c) differ 2, (b,c) differ 2 -> mean 5/3.
    const a = [0, 1, 2];
    const b = [0, 1, 3];
    const c = [0, 4, 5];
    expect(meanPairwiseDifference([a, b, c])).toBeCloseTo(5 / 3);
  });
});

describe("noveltyScore", () => {
  it("is null without corpus lookups", () => {
    expect(noveltyScore([5, 3], [null, null])).toBeNull();
    expect(noveltyScore([], [])).toBeNull();
  });

  it("is 0 for all-seen and 100 for all far-from-corpus", () => {
    expect(noveltyScore([5, 3], [0, 0])).toBe(0);
    expect(noveltyScore([5, 3], [3, 7])).toBe(100);
  });

  it("weights by sample count and caps delta", () => {
    // 9 samples at delta 0, 1 sample at delta 3 (cap) -> 10%.
    expect(noveltyScore([9, 1], [0, 3])).toBeCloseTo(10);
    // delta beyond the cap contributes the same as the cap.
    expect(noveltyScore([9, 1], [0, 6])).toBeCloseTo(10);
    // delta 1 counts as 1/3 novel.
    expect(noveltyScore([1], [1])).toBeCloseTo(100 / 3);
  });
});

describe("corpus score percentile", () => {
  // Minimal model: J = 0, h[i] = i, so a single-member team's Score is
  // exactly its vocab index. teamObservables only touches J, h, V.
  const V = 4;
  const model = {
    V,
    J: new Float64Array(V * V),
    h: Float64Array.from([0, 1, 2, 3]),
  } as unknown as IsingModel;
  const teamCounts: TeamCounts = new Map([
    ["0", 1],
    ["1", 1],
    ["2", 2],
  ]);

  it("builds sorted weighted score + coherence distributions", () => {
    const index = buildCorpusScoreIndex(model, teamCounts)!;
    expect(index.score.values).toEqual([0, 1, 2]);
    expect(index.score.weights).toEqual([1, 1, 2]);
    expect(index.score.totalWeight).toBe(4);
    // J = 0 -> every corpus team has coherence 0.
    expect(index.coherence.values).toEqual([0, 0, 0]);
    expect(index.coherence.totalWeight).toBe(4);
  });

  it("returns null without a corpus", () => {
    expect(buildCorpusScoreIndex(model, null)).toBeNull();
    expect(buildCorpusScoreIndex(model, new Map())).toBeNull();
  });

  it("ranks with ties counting half", () => {
    const index = buildCorpusScoreIndex(model, teamCounts)!;
    expect(percentileRank(index.score, -1)).toBe(0);
    expect(percentileRank(index.score, 1.5)).toBe(50); // above weights 1+1 of 4
    expect(percentileRank(index.score, 1)).toBe(37.5); // 1 below + half of 1 tied
    expect(percentileRank(index.score, 99)).toBe(100);
    // All corpus coherences tie at 0 -> a 0-coherence team sits at 50.
    expect(percentileRank(index.coherence, 0)).toBe(50);
  });
});

describe("ordinal", () => {
  it("formats English ordinals", () => {
    expect(ordinal(1)).toBe("1st");
    expect(ordinal(2)).toBe("2nd");
    expect(ordinal(3)).toBe("3rd");
    expect(ordinal(4)).toBe("4th");
    expect(ordinal(11)).toBe("11th");
    expect(ordinal(12)).toBe("12th");
    expect(ordinal(13)).toBe("13th");
    expect(ordinal(21)).toBe("21st");
    expect(ordinal(84)).toBe("84th");
    expect(ordinal(100)).toBe("100th");
  });
});
