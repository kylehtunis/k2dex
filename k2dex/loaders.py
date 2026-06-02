"""Pure model builders. Stateless functions that fit (J, h) and the
auxiliary vocab / corpus structures used by both the Streamlit app and
the static-site precompute pipeline.

`app.py` wraps these in `@st.cache_resource` for per-process caching;
`precompute.py` calls them directly. Keep this module Streamlit-free.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from numpy.typing import NDArray

from . import tournament_ingest
from .constants import (
    PHASE2_MIN_TEAM_COUNT,
    SPECIES_ITEM_LR_C,
    SPECIES_LR_C,
)
from .models import fit_pl_ising


SpeciesModel = tuple[
    list[str],            # vocab
    NDArray[np.float64],  # m
    NDArray[np.float64],  # J
    NDArray[np.float64],  # h
    Counter,              # team_counts (frozenset[str] -> int)
    list[str],            # species_of
    list[str | None],     # item_of
]


def format_pair(species: str, item: str | None) -> str:
    """Display form for Phase 3 vocab strings: bare species when itemless,
    'Species @ Item' otherwise."""
    if item is None:
        return species
    return f"{species} @ {item}"


def build_species_model() -> SpeciesModel:
    """Species-only PL inverse Ising over the cached tournament corpus."""
    tournaments = tournament_ingest.load_cached_tournaments()
    teams_full = tournament_ingest.all_teams(tournaments)
    teams = tournament_ingest.species_only_teams(teams_full)
    team_counts: Counter[frozenset[str]] = Counter(teams)

    counts = Counter(name for team in teams for name in team)
    vocab = sorted(name for name, c in counts.items() if c >= PHASE2_MIN_TEAM_COUNT)
    name_to_i = {name: i for i, name in enumerate(vocab)}
    V = len(vocab)

    X = np.zeros((len(teams), V), dtype=np.int8)
    for ti, team in enumerate(teams):
        for name in team:
            j = name_to_i.get(name)
            if j is not None:
                X[ti, j] = 1
    m = X.mean(axis=0)

    J, h = fit_pl_ising(X, C=SPECIES_LR_C)
    species_of = list(vocab)
    item_of: list[str | None] = [None] * len(vocab)
    return vocab, m, J, h, team_counts, species_of, item_of


def build_species_item_model() -> SpeciesModel:
    """(species, item)-pair PL inverse Ising over the cached tournament corpus."""
    tournaments = tournament_ingest.load_cached_tournaments()
    teams = tournament_ingest.all_teams(tournaments)

    pair_counts = Counter(pair for team in teams for pair in team)
    pair_list_above_cutoff = [p for p, c in pair_counts.items() if c >= PHASE2_MIN_TEAM_COUNT]
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
    m = X.mean(axis=0)

    team_counts: Counter[frozenset[str]] = Counter()
    for team in teams:
        if all(pair in pair_to_idx for pair in team):
            team_counts[frozenset(format_pair(s, i) for s, i in team)] += 1

    J, h = fit_pl_ising(X, C=SPECIES_ITEM_LR_C)
    return vocab, m, J, h, team_counts, species_of, item_of
