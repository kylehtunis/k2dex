"""Pure model builders. Stateless functions that fit (J, h) and the
auxiliary vocab / corpus structures used by both the Streamlit app and
the static-site precompute pipeline.

`app.py` wraps these in `@st.cache_resource` for per-process caching;
`precompute.py` calls them directly. Keep this module Streamlit-free.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import date

import numpy as np
from numpy.typing import NDArray

from . import tournament_ingest
from .constants import (
    CURRENT_REGULATION,
    IN_PERSON_WEIGHT,
    PHASE2_MIN_TEAM_COUNT,
    RECENCY_TAU_DAYS,
    SPECIES_ITEM_LR_LAMBDA,
    SPECIES_LR_LAMBDA,
)
from .models import fit_pl_ising
from .tournament_ingest import TeamObservation


SpeciesModel = tuple[
    list[str],            # vocab
    NDArray[np.float64],  # m
    NDArray[np.float64],  # J
    NDArray[np.float64],  # h
    Counter,              # team_counts (frozenset[str] -> int)
    list[str],            # species_of
    list[str | None],     # item_of
    str,                  # latest_tournament_date (ISO date string)
]


def _parse_day(date_str: str) -> date:
    """Day-resolution parse of a cache date string. Limitless entries carry
    full ISO timestamps ('2026-06-05T21:30:00.000Z'), in-person entries bare
    dates ('2026-06-06'); both start with YYYY-MM-DD."""
    return date.fromisoformat(date_str[:10])


def team_weights(
    observations: Sequence[TeamObservation],
    *,
    reference_date: str,
    recency_tau: float | None = None,
    in_person_multiplier: float = 1.0,
) -> NDArray[np.float64]:
    """Per-team fit weights: w = exp(-age/tau) * multiplier^[in-person],
    normalized so the weights sum to the team count (mean weight 1).

    `age` is days between each observation's tournament date and
    `reference_date` (normally the corpus's latest tournament date).
    `recency_tau=None` disables the decay term entirely (dates are then not
    parsed, so undated observations are tolerated). The mean-1 normalization
    keeps weighted counts on the same scale as raw counts, which the weighted
    vocab cutoff and `fit_pl_ising`'s degenerate-spin skip both rely on.
    """
    n = len(observations)
    w = np.ones(n, dtype=np.float64)
    if recency_tau is not None:
        ref = _parse_day(reference_date)
        age_days = np.array(
            [(ref - _parse_day(o.date)).days for o in observations],
            dtype=np.float64,
        )
        w *= np.exp(-age_days / recency_tau)
    if in_person_multiplier != 1.0:
        in_person = np.array(
            [o.tournament_type == "in-person" for o in observations], dtype=bool,
        )
        w[in_person] *= in_person_multiplier
    return w * (n / w.sum())


def format_pair(species: str, item: str | None) -> str:
    """Display form for Phase 3 vocab strings: bare species when itemless,
    'Species @ Item' otherwise."""
    if item is None:
        return species
    return f"{species} @ {item}"


def build_species_model(
    *,
    regulation: str = CURRENT_REGULATION,
    min_team_count: int = PHASE2_MIN_TEAM_COUNT,
    lam: float = SPECIES_LR_LAMBDA,
    recency_tau: float | None = RECENCY_TAU_DAYS,
    in_person_multiplier: float = IN_PERSON_WEIGHT,
) -> SpeciesModel:
    """Species-only PL inverse Ising over the cached tournament corpus.

    `recency_tau` / `in_person_multiplier` set the per-team fit weights (see
    `team_weights`). The marginals `m` are weighted to match the weighted fit;
    `team_counts` stays a raw count for display. The vocab cutoff requires
    `min_team_count` in BOTH raw and weighted counts: raw support so that no
    single highly-weighted team can push a one-off feature into the vocab,
    weighted support so features whose evidence is all heavily decayed drop
    out instead of hitting the degenerate-spin path.
    """
    tournaments = tournament_ingest.load_cached_tournaments(regulation=regulation)
    latest_date = max(t.meta.date for t in tournaments)
    observations = tournament_ingest.all_team_observations(tournaments)
    w = team_weights(
        observations,
        reference_date=latest_date,
        recency_tau=recency_tau,
        in_person_multiplier=in_person_multiplier,
    )
    teams = tournament_ingest.species_only_teams([o.members for o in observations])
    team_counts: Counter[frozenset[str]] = Counter(teams)

    raw_counts = Counter(name for team in teams for name in team)
    weighted_counts: dict[str, float] = {}
    for ti, team in enumerate(teams):
        for name in team:
            weighted_counts[name] = weighted_counts.get(name, 0.0) + w[ti]
    vocab = sorted(
        name for name, c in weighted_counts.items()
        if c >= min_team_count and raw_counts[name] >= min_team_count
    )
    name_to_i = {name: i for i, name in enumerate(vocab)}
    V = len(vocab)

    X = np.zeros((len(teams), V), dtype=np.int8)
    for ti, team in enumerate(teams):
        for name in team:
            j = name_to_i.get(name)
            if j is not None:
                X[ti, j] = 1
    m = (w @ X) / w.sum()

    J, h = fit_pl_ising(X, C=1.0 / lam, sample_weight=w)
    species_of = list(vocab)
    item_of: list[str | None] = [None] * len(vocab)
    return vocab, m, J, h, team_counts, species_of, item_of, latest_date


def build_species_item_model(
    *,
    regulation: str = CURRENT_REGULATION,
    min_team_count: int = PHASE2_MIN_TEAM_COUNT,
    lam: float = SPECIES_ITEM_LR_LAMBDA,
    recency_tau: float | None = RECENCY_TAU_DAYS,
    in_person_multiplier: float = IN_PERSON_WEIGHT,
) -> SpeciesModel:
    """(species, item)-pair PL inverse Ising over the cached tournament corpus.

    Weighting semantics match `build_species_model`: weighted marginals, vocab
    cutoff on BOTH raw and weighted counts, raw `team_counts` for display.
    """
    tournaments = tournament_ingest.load_cached_tournaments(regulation=regulation)
    latest_date = max(t.meta.date for t in tournaments)
    observations = tournament_ingest.all_team_observations(tournaments)
    w = team_weights(
        observations,
        reference_date=latest_date,
        recency_tau=recency_tau,
        in_person_multiplier=in_person_multiplier,
    )
    teams = [o.members for o in observations]

    raw_counts = Counter(pair for team in teams for pair in team)
    weighted_counts: dict[tuple[str, str | None], float] = {}
    for ti, team in enumerate(teams):
        for pair in team:
            weighted_counts[pair] = weighted_counts.get(pair, 0.0) + w[ti]
    pair_list_above_cutoff = [
        p for p, c in weighted_counts.items()
        if c >= min_team_count and raw_counts[p] >= min_team_count
    ]
    pair_list = sorted(pair_list_above_cutoff, key=lambda p: format_pair(p[0], p[1]))
    vocab = [format_pair(s, i) for s, i in pair_list]
    pair_to_idx = {p: i for i, p in enumerate(pair_list)}
    V = len(vocab)

    species_of = [s for s, _ in pair_list]
    item_of: list[str | None] = [i for _, i in pair_list]

    X = np.zeros((len(teams), V), dtype=np.int8)
    for ti, team in enumerate(teams):
        for pair in team:
            j = pair_to_idx.get(pair)
            if j is not None:
                X[ti, j] = 1
    m = (w @ X) / w.sum()

    team_counts: Counter[frozenset[str]] = Counter()
    for team in teams:
        if all(pair in pair_to_idx for pair in team):
            team_counts[frozenset(format_pair(s, i) for s, i in team)] += 1

    J, h = fit_pl_ising(X, C=1.0 / lam, sample_weight=w)
    return vocab, m, J, h, team_counts, species_of, item_of, latest_date
