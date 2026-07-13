// At-a-glance summary stats over the PT completion distribution shown on
// /completer: how varied the suggestions are and how novel they are
// relative to the tournament corpus.
//
// Webapp-only (no Streamlit/Python mirror, no parity row) — pure
// presentation math over the sampled distribution + corpus lookups.

/** Mean pairwise count of differing members across the shown teams.
 * Pinned members are identical on every team, so only completer-filled
 * slots contribute. Returns null with fewer than two teams. */
export function meanPairwiseDifference(
  teams: readonly (readonly number[])[],
): number | null {
  if (teams.length < 2) return null;
  let sum = 0;
  let nPairs = 0;
  for (let i = 0; i < teams.length; i++) {
    const setI = new Set(teams[i]);
    for (let j = i + 1; j < teams.length; j++) {
      let shared = 0;
      for (const m of teams[j]) if (setI.has(m)) shared++;
      sum += teams[i].length - shared;
      nPairs++;
    }
  }
  return sum / nPairs;
}

/** Corpus distance at which a suggestion counts as fully novel. */
export const NOVELTY_DELTA_CAP = 3;

/** Sample-weighted novelty (0–100) of the shown completions. Each team
 * contributes min(delta, NOVELTY_DELTA_CAP) / NOVELTY_DELTA_CAP weighted
 * by its sample count: 0 = every suggestion is an exact tournament roster,
 * 100 = every suggestion is >= NOVELTY_DELTA_CAP swaps from anything
 * observed. Returns null when no team has a corpus lookup (no corpus). */
export function noveltyScore(
  counts: readonly number[],
  deltas: readonly (number | null)[],
): number | null {
  let wSum = 0;
  let acc = 0;
  for (let i = 0; i < counts.length; i++) {
    const d = deltas[i];
    if (d === null) continue;
    wSum += counts[i];
    acc += counts[i] * (Math.min(d, NOVELTY_DELTA_CAP) / NOVELTY_DELTA_CAP);
  }
  if (wSum === 0) return null;
  return (100 * acc) / wSum;
}
