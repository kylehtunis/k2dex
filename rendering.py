"""Diagnostic helpers for rendering team observables.

These three pure functions compute per-team values that the webapp surfaces
across multiple modes / pages (completion tables, analysis observables strip,
meta summaries). They don't touch Streamlit; just plain numpy + Counter logic.
"""
from __future__ import annotations

from collections import Counter

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
