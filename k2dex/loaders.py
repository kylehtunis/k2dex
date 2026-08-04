"""Pure model builders. Stateless functions that fit (J, h) and the
auxiliary vocab / corpus structures the static-site precompute pipeline
serializes into the webapp's model artifacts.

Each build refits from the cached corpus and takes minutes, so callers that
need a model more than once are expected to cache it themselves.
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
    VOCAB_MIN_TEAM_COUNT,
    RECENCY_TAU_DAYS,
    SPECIES_ITEM_LR_LAMBDA,
    SPECIES_LR_LAMBDA,
    TEAM_SIZE,
)
from .models import fit_boltzmann_ising, fit_pl_ising
from .tournament_ingest import TeamObservation


def _fit_jh(
    X: NDArray[np.int8],
    *,
    method: str,
    lam: float,
    w: NDArray[np.float64],
    species_of: list[str] | None,
    item_of: list[str | None] | None,
    prior_J: NDArray[np.float64] | None,
    prior_h: NDArray[np.float64] | None,
    intercept_prior_weight: float,
    boltzmann_opts: dict | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit (J, h) by the requested method.

    `pseudo_likelihood` is the per-spin PL fit (the production path).
    `boltzmann` warm-starts from that PL fit and refines it by constrained-
    MaxEnt moment matching (`fit_boltzmann_ising`). `species_of`/`item_of` define
    the uniqueness constraints of the ensemble the model is sampled from (pass
    `None`/`None` for the species model, the real lookups for species+item).
    `boltzmann_opts` is forwarded to `fit_boltzmann_ising`; a `support_min_count`
    key (popped here) builds a co-occurrence support mask so only pairs seen
    together at least that many times in the corpus are fit.
    """
    if method == "pseudo_likelihood":
        return fit_pl_ising(
            X, C=1.0 / lam, sample_weight=w,
            prior_J=prior_J, prior_h=prior_h,
            intercept_prior_weight=intercept_prior_weight,
        )
    if method != "boltzmann":
        raise ValueError(f"unknown fit method {method!r}")

    # PL fit is the warm start; Boltzmann then matches moments under reg→0.
    init_J, init_h = fit_pl_ising(
        X, C=1.0 / lam, sample_weight=w,
        prior_J=prior_J, prior_h=prior_h,
        intercept_prior_weight=intercept_prior_weight,
    )
    opts = dict(boltzmann_opts or {})
    support_min = opts.pop("support_min_count", None)
    support_mask = None
    if support_min is not None:
        cooc = X.astype(np.int64).T @ X.astype(np.int64)
        support_mask = cooc >= int(support_min)
    J, h, _hist = fit_boltzmann_ising(
        X, team_size=TEAM_SIZE, sample_weight=w,
        init_J=init_J, init_h=init_h,
        species_of=species_of, item_of=item_of,
        support_mask=support_mask, **opts,
    )
    return J, h


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


def _align_prior(
    vocab: list[str],
    prior_vocab: Sequence[str],
    prior_J: NDArray[np.float64],
    prior_h: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Scatter a previous model's (J, h) onto `vocab`'s index order.

    Features shared between `vocab` and `prior_vocab` (matched by vocab string)
    carry their prior values; features new to `vocab` get zeros. The result is
    fed to `fit_pl_ising(prior_J=..., prior_h=...)`, which re-centers the L2
    penalty on it. Matching by string means it works identically for the
    species vocab ('Incineroar') and the pair vocab ('Incineroar @ ...').
    """
    V = len(vocab)
    idx = {name: i for i, name in enumerate(vocab)}
    pidx = {name: i for i, name in enumerate(prior_vocab)}
    shared = [name for name in vocab if name in pidx]
    new_pos = np.array([idx[name] for name in shared], dtype=np.intp)
    old_pos = np.array([pidx[name] for name in shared], dtype=np.intp)
    J_aligned = np.zeros((V, V), dtype=np.float64)
    h_aligned = np.zeros(V, dtype=np.float64)
    if len(shared):
        J_aligned[np.ix_(new_pos, new_pos)] = prior_J[np.ix_(old_pos, old_pos)]
        h_aligned[new_pos] = prior_h[old_pos]
    return J_aligned, h_aligned


def format_pair(species: str, item: str | None) -> str:
    """Display form for a (species, item) vocab string: bare species when
    itemless, 'Species @ Item' otherwise."""
    if item is None:
        return species
    return f"{species} @ {item}"


def build_species_model(
    *,
    regulation: str = CURRENT_REGULATION,
    min_team_count: int = VOCAB_MIN_TEAM_COUNT,
    lam: float = SPECIES_LR_LAMBDA,
    recency_tau: float | None = RECENCY_TAU_DAYS,
    in_person_multiplier: float = IN_PERSON_WEIGHT,
    prior_regulation: str | None = None,
    intercept_prior_weight: float = 1.0,
    method: str = "pseudo_likelihood",
    boltzmann_opts: dict | None = None,
) -> SpeciesModel:
    """Species-only PL inverse Ising over the cached tournament corpus.

    `recency_tau` / `in_person_multiplier` set the per-team fit weights (see
    `team_weights`). The marginals `m` are weighted to match the weighted fit;
    `team_counts` stays a raw count for display. The vocab cutoff requires
    `min_team_count` in BOTH raw and weighted counts: raw support so that no
    single highly-weighted team can push a one-off feature into the vocab,
    weighted support so features whose evidence is all heavily decayed drop
    out instead of hitting the degenerate-spin path.

    `prior_regulation` warm-starts the fit from another regulation (assumed a
    legality superset, so every prior feature stays valid here). That prior
    model is fit on its own corpus with default knobs, its full vocab is folded
    into this one unconditionally (a prior feature absent from `regulation` just
    hits the degenerate-spin skip and keeps its prior value), and its (J, h)
    re-centers the L2 penalty so thin features relax toward the prior instead of
    zero. `intercept_prior_weight` tunes how hard the bias is pulled to the
    prior (see `fit_pl_ising`). With `prior_regulation=None` this is an ordinary
    zero-centered fit.
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

    prior: SpeciesModel | None = None
    if prior_regulation is not None:
        prior = build_species_model(regulation=prior_regulation)

    raw_counts = Counter(name for team in teams for name in team)
    weighted_counts: dict[str, float] = {}
    for ti, team in enumerate(teams):
        for name in team:
            weighted_counts[name] = weighted_counts.get(name, 0.0) + w[ti]
    native_vocab = {
        name for name, c in weighted_counts.items()
        if c >= min_team_count and raw_counts[name] >= min_team_count
    }
    if prior is not None:
        native_vocab |= set(prior[0])  # prior vocab folded in unconditionally
    vocab = sorted(native_vocab)
    name_to_i = {name: i for i, name in enumerate(vocab)}
    V = len(vocab)

    X = np.zeros((len(teams), V), dtype=np.int8)
    for ti, team in enumerate(teams):
        for name in team:
            j = name_to_i.get(name)
            if j is not None:
                X[ti, j] = 1
    m = (w @ X) / w.sum()

    prior_J = prior_h = None
    if prior is not None:
        prior_J, prior_h = _align_prior(vocab, prior[0], prior[2], prior[3])

    # Species are distinct per vocab entry, so the species model's ensemble
    # needs no uniqueness lookups -- pass None/None.
    J, h = _fit_jh(
        X, method=method, lam=lam, w=w,
        species_of=None, item_of=None,
        prior_J=prior_J, prior_h=prior_h,
        intercept_prior_weight=intercept_prior_weight,
        boltzmann_opts=boltzmann_opts,
    )
    species_of = list(vocab)
    item_of: list[str | None] = [None] * len(vocab)
    return vocab, m, J, h, team_counts, species_of, item_of, latest_date


def build_species_item_model(
    *,
    regulation: str = CURRENT_REGULATION,
    min_team_count: int = VOCAB_MIN_TEAM_COUNT,
    lam: float = SPECIES_ITEM_LR_LAMBDA,
    recency_tau: float | None = RECENCY_TAU_DAYS,
    in_person_multiplier: float = IN_PERSON_WEIGHT,
    prior_regulation: str | None = None,
    intercept_prior_weight: float = 1.0,
    method: str = "pseudo_likelihood",
    boltzmann_opts: dict | None = None,
) -> SpeciesModel:
    """(species, item)-pair PL inverse Ising over the cached tournament corpus.

    Weighting semantics match `build_species_model`: weighted marginals, vocab
    cutoff on BOTH raw and weighted counts, raw `team_counts` for display.
    `prior_regulation` / `intercept_prior_weight` warm-start the fit exactly as
    in `build_species_model`, here over the (species, item) pair vocabulary.
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
    # Normalize a missing item (Python None) to the string "None" so it becomes
    # an ordinary Potts state (a real item name) rather than a bare-species
    # feature. This merges the corpus's split duplicates -- e.g. a Talonflame
    # observed with no recorded item and a Talonflame recorded with the literal
    # item "None" collapse into one (Talonflame, "None") feature.
    teams = [
        [(species, "None" if item is None else item) for species, item in o.members]
        for o in observations
    ]

    prior: SpeciesModel | None = None
    if prior_regulation is not None:
        prior = build_species_item_model(regulation=prior_regulation)

    raw_counts = Counter(pair for team in teams for pair in team)
    weighted_counts: dict[tuple[str, str], float] = {}
    for ti, team in enumerate(teams):
        for pair in team:
            weighted_counts[pair] = weighted_counts.get(pair, 0.0) + w[ti]
    pair_set = {
        p for p, c in weighted_counts.items()
        if c >= min_team_count and raw_counts[p] >= min_team_count
    }
    if prior is not None:
        pair_set |= set(zip(prior[5], prior[6]))  # prior (species, item) pairs
    pair_list = sorted(pair_set, key=lambda p: format_pair(p[0], p[1]))
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

    prior_J = prior_h = None
    if prior is not None:
        prior_J, prior_h = _align_prior(vocab, prior[0], prior[2], prior[3])

    # species+item has features sharing a species or item, so the ensemble
    # enforces no-duplicate-species and no-duplicate-item -- pass the lookups.
    J, h = _fit_jh(
        X, method=method, lam=lam, w=w,
        species_of=species_of, item_of=item_of,
        prior_J=prior_J, prior_h=prior_h,
        intercept_prior_weight=intercept_prior_weight,
        boltzmann_opts=boltzmann_opts,
    )
    return vocab, m, J, h, team_counts, species_of, item_of, latest_date
