// Corpus index lookups for the `corpus` column rendered on /completer
// and /analysis.
//
// Mirrors rendering.nearest_observed. Team-counts here are loaded from
// team_counts.json with sorted-index "-"-joined keys (see
// precompute.serialize_team_counts). We never reconstruct the
// vocab-string roster set on the client — the index is keyed by vocab
// indices, period.

import type { TeamCounts } from "../sampler/types";

/** Canonical sorted-index "-"-joined key. Matches the schema written
 * by precompute.serialize_team_counts. */
export function teamKey(team: readonly number[]): string {
  return [...team].sort((a, b) => a - b).join("-");
}

interface NearestObservedResult {
  delta: number;
  count: number;
}

/** Joint (min swap distance, count of the nearest observed roster).
 *
 * delta == 0  -> the exact team is in the corpus; count is its own occurrence
 * delta >= 1  -> count of the nearest observed roster (tie-broken by
 *                highest count — most-played among equally-near).
 *
 * Returns null when teamCounts is empty / unavailable. Mirrors
 * rendering.nearest_observed exactly. */
export function nearestObserved(
  team: readonly number[],
  teamCounts: TeamCounts | null,
): NearestObservedResult | null {
  if (teamCounts === null || teamCounts.size === 0) return null;

  const key = teamKey(team);
  const exact = teamCounts.get(key);
  if (exact !== undefined) return { delta: 0, count: exact };

  const teamSize = team.length;
  const teamSet = new Set(team);
  let bestDelta: number | null = null;
  let bestCount = 0;
  for (const [k, c] of teamCounts) {
    // Parse roster key once per entry. For ~6000 entries × team_size=6
    // this is ~36k integer parses per call, well under 5ms in V8.
    let intersection = 0;
    let start = 0;
    while (start < k.length) {
      let end = k.indexOf("-", start);
      if (end < 0) end = k.length;
      const idx = +k.slice(start, end);
      if (teamSet.has(idx)) intersection++;
      start = end + 1;
    }
    const delta = teamSize - intersection;
    if (
      bestDelta === null ||
      delta < bestDelta ||
      (delta === bestDelta && c > bestCount)
    ) {
      bestDelta = delta;
      bestCount = c;
    }
  }
  if (bestDelta === null) return null;
  return { delta: bestDelta, count: bestCount };
}
