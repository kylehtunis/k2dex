// Ranking logic for the §03 top-teams table.
//
// The corpus is shipped to the client as team_counts.json: a
// Map<sortedIndexKey, count> where the key is the "-"-joined sorted
// vocab indices of an exact observed roster (see
// precompute.serialize_team_counts). This is the same index already
// loaded for `nearestObserved`; here we just rank it by raw occurrence.

import type { TeamCounts } from "../sampler/types";
import { parseTeamKey } from "../render/corpus";

export interface TopTeam {
  /** Vocab indices of the roster (in display order — most-popular
   * member first by marginal m̂). */
  team: number[];
  /** Raw occurrence count of this exact roster in the corpus. */
  count: number;
}

/** Top-`k` exact rosters by occurrence count, descending.
 *
 * Each team's members are ordered by descending marginal `m` so the
 * sprite strip leads with the roster's most-played Pokemon. Ties on
 * count are broken by the team key for a stable order. Returns [] when
 * the corpus index is empty/unavailable. */
export function topTeams(
  teamCounts: TeamCounts | null,
  k: number,
  m: Float64Array,
): TopTeam[] {
  if (teamCounts === null || teamCounts.size === 0) return [];
  const rows: { key: string; team: number[]; count: number }[] = [];
  for (const [key, count] of teamCounts) {
    const team = parseTeamKey(key);
    team.sort((a, b) => m[b] - m[a]);
    rows.push({ key, team, count });
  }
  rows.sort((a, b) => b.count - a.count || (a.key < b.key ? -1 : 1));
  return rows.slice(0, k).map(({ team, count }) => ({ team, count }));
}
