"""Diagnostic helpers for rendering team observables.

The three numeric helpers (`team_obs_count`, `min_swaps_to_observed`,
`intra_team_sum_j`) and the markdown-table builders below are pure and
Streamlit-free; they compute per-team values that the webapp surfaces
across multiple pages (completion tables, analysis observables strip,
meta summaries).
"""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def team_obs_count(
    state: NDArray[np.bool_],
    vocab: list[str],
    team_counts: Counter[frozenset[str]] | None,
) -> int | None:
    """How many times this exact 6-Pokemon team appeared in the ingested corpus.

    Returns None when no team-level data is available (Phase 1, which only has
    aggregate Smogon stats and not per-team rosters).
    """
    if team_counts is None:
        return None
    team = frozenset(vocab[i] for i in np.where(state)[0])
    return team_counts[team]


def min_swaps_to_observed(
    state: NDArray[np.bool_],
    vocab: list[str],
    team_counts: Counter[frozenset[str]] | None,
) -> int | None:
    """Minimum number of slot swaps to transform this team into any team in
    the corpus. 0 = the team itself was observed; >=1 measures distance from
    realized teams. Returns None when team_counts is empty / unavailable.

    Useful for distinguishing high-probability completions that are 1-swap
    variants of a real team (defensible: "model picked a near-neighbor of the
    meta") from globally distinct configurations (likely overfit or genuine
    discovery).

    Implementation: `|team - obs|` (members of `team` not in `obs`) equals the
    swap distance for same-size teams. Doesn't require team_size as a param.
    """
    if not team_counts:
        return None
    team = frozenset(vocab[i] for i in np.where(state)[0])
    return min(len(team - obs) for obs in team_counts)


def nearest_observed(
    state: NDArray[np.bool_],
    vocab: list[str],
    team_counts: Counter[frozenset[str]] | None,
) -> tuple[int, int] | None:
    """Joint (min swap distance, count of the nearest observed roster).

    delta == 0 means the team itself was observed; count is its own corpus
    occurrence count. delta >= 1 returns the corpus count of *the* nearest
    observed roster (ties broken by highest count — the most-played variant
    among equally-close observed teams).

    Returns None when team_counts is empty / unavailable.
    """
    if not team_counts:
        return None
    team = frozenset(vocab[i] for i in np.where(state)[0])
    if team in team_counts:
        return 0, int(team_counts[team])
    best_delta = None
    best_count = 0
    for obs, c in team_counts.items():
        d = len(team - obs)
        if best_delta is None or d < best_delta or (d == best_delta and c > best_count):
            best_delta = d
            best_count = int(c)
    if best_delta is None:
        return None
    return best_delta, best_count


def build_cooccurrence(
    team_counts: dict[str, int],
    V: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Co-occurrence matrix C, marginals m, and team total from a team-count
    index. Keys are sorted-index "-"-joined rosters (the serialize_team_counts
    schema); a team of size k contributes `count` to C[i, j] for each of its
    pairs and to m[i] for each member. Diagonal stays zero. Equivalent to
    `C = (X.T @ X)` with the diagonal zeroed and `m = X.mean(axis=0)` on the
    weighted binary team matrix.

    Mirrored 1:1 in web/src/sampler/cooccurrence.ts:buildCooccurrence, gated by
    tests/test_parity.py::test_cooccurrence_cases.
    """
    C = np.zeros((V, V), dtype=np.float64)
    m = np.zeros(V, dtype=np.float64)
    n_teams = 0
    for key, count in team_counts.items():
        team = [int(x) for x in key.split("-")] if key else []
        n_teams += count
        for a in range(len(team)):
            ia = team[a]
            m[ia] += count
            for b in range(a + 1, len(team)):
                ib = team[b]
                C[ia, ib] += count
                C[ib, ia] += count
    if n_teams > 0:
        m /= n_teams
    return C, m, n_teams


def score_cooccurrence(
    C: NDArray[np.float64],
    held_in: list[int],
) -> NDArray[np.float64]:
    """Co-occurrence score of every feature against a held-in (pinned) set:
    score[i] = sum(C[i, j] for j in held_in). Higher = appears with the pinned
    members more often. Mirrors web/src/sampler/cooccurrence.ts:scoreCooccurrence.
    """
    if not held_in:
        return np.zeros(C.shape[0], dtype=np.float64)
    return C[:, held_in].sum(axis=1)


def cooccurrence_greedy(
    C: NDArray[np.float64],
    team_size: int,
    fixed: list[int],
    excluded: list[int],
    species_of: list[str] | None,
    item_of: list[str | None] | None,
) -> list[int]:
    """Greedy team fill by co-occurrence score -- the "naive teambuilder":
    repeatedly add the highest-scoring candidate that keeps the team legal
    (unique species, unique items), rescoring against the growing team each
    step. Ties break to the lowest feature index (stable scan order). Returns
    the completed team as sorted feature indices.

    Mirrors web/src/sampler/cooccurrence.ts:cooccurrenceGreedy.
    """
    V = C.shape[0]
    team = list(fixed)
    ex = set(excluded)
    in_team = set(team)
    used_species: set[str] = set()
    used_items: set[str] = set()

    def add_used(i: int) -> None:
        if species_of is not None:
            used_species.add(species_of[i])
        if item_of is not None:
            it = item_of[i]
            if it is not None:
                used_items.add(it)

    def conflicts(i: int) -> bool:
        if species_of is not None and species_of[i] in used_species:
            return True
        if item_of is not None:
            it = item_of[i]
            if it is not None and it in used_items:
                return True
        return False

    for i in team:
        add_used(i)

    while len(team) < team_size:
        scores = score_cooccurrence(C, team)
        best_idx = -1
        best_score = -np.inf
        for i in range(V):
            if i in in_team or i in ex:
                continue
            if conflicts(i):
                continue
            if scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        if best_idx < 0:
            break
        team.append(best_idx)
        in_team.add(best_idx)
        add_used(best_idx)

    return sorted(team)


def intra_team_sum_j(
    state: NDArray[np.bool_],
    J: NDArray[np.float64],
) -> float:
    """Sum of pairwise couplings over the team's unordered pairs:
    sum_{i<j: both in team} J_ij = 0.5 * s' J s.

    Measures structural coherence under the fitted model -- the pairwise
    contribution to (-raw_E). A team can have low raw_E either from popular
    members (large h.s) or coherent pairs (large Sigma J); decomposing makes
    that visible.
    """
    s = state.astype(np.float64)
    return float(0.5 * s @ J @ s)


@dataclass(frozen=True)
class PairwiseJRow:
    """One row of the per-team pairwise J decomposition table."""
    rank: int
    name_a: str
    name_b: str
    j_value: float
    pct_of_abs_sum: float  # |J_ij| / sum(|J|) over the C(team_size, 2) team pairs


def pairwise_j_rows(
    team_idx: list[int],
    vocab: list[str],
    J: NDArray[np.float64],
) -> list[PairwiseJRow]:
    """Build the pairwise-J decomposition for an observed team, sorted by |J|."""
    pairs = [
        (vocab[i_a], vocab[i_b], float(J[i_a, i_b]))
        for i_a, i_b in itertools.combinations(team_idx, 2)
    ]
    pairs.sort(key=lambda p: -abs(p[2]))
    abs_sum = sum(abs(j) for _, _, j in pairs) or 1.0
    return [
        PairwiseJRow(
            rank=r + 1, name_a=a, name_b=b, j_value=j,
            pct_of_abs_sum=abs(j) / abs_sum,
        )
        for r, (a, b, j) in enumerate(pairs)
    ]


def render_pairwise_j_table(rows: list[PairwiseJRow]) -> str:
    """Markdown table: rank, pair label, J value, percent of |J| total."""
    lines = [
        "| # | pair | J | % of \\|J\\| sum |",
        "| ---: | :--- | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r.rank} | {r.name_a} × {r.name_b} | "
            f"{r.j_value:+.3f} | {r.pct_of_abs_sum:.1%} |"
        )
    return "\n".join(lines)


