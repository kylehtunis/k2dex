"""Pure model builders. Stateless functions that fit (J, h) and the
auxiliary vocab / corpus structures used by both the Streamlit app and
the static-site precompute pipeline.

`app.py` wraps these in `@st.cache_resource` for per-process caching;
`precompute.py` calls them directly. Keep this module Streamlit-free.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import NamedTuple

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
    track_values: list[list[str | None]] | None,
    track_names: list[str] | None,
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
    `track_values`/`track_names` carry the full attribute-track structure (item,
    ability, ...) the Boltzmann Potts kernel factors the bank over; `None`/`None`
    for the species model (or a single item track when omitted).
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
        track_values=track_values, track_names=track_names,
        support_mask=support_mask, **opts,
    )
    return J, h


@dataclass(frozen=True)
class TrackSpec:
    """One attribute track carried by a model's features.

    `cardinality` is how many values of this track a single team member holds
    (1 for item/ability; a future moves track would be 4). `cross_slot_unique`
    is the no-two-members-share-this-value team rule (item: True; ability:
    False). `within_slot_unique` is the no-duplicate-within-one-member rule
    (relevant only for cardinality > 1, e.g. distinct moves; both current
    tracks: False). The schema is cardinality-general now; the sampler exercises
    cardinality 1 only."""

    name: str
    cardinality: int
    cross_slot_unique: bool
    within_slot_unique: bool


ITEM_TRACK = TrackSpec("item", cardinality=1, cross_slot_unique=True, within_slot_unique=False)
ABILITY_TRACK = TrackSpec("ability", cardinality=1, cross_slot_unique=False, within_slot_unique=False)


def _track_value_view(
    track_specs: Sequence[TrackSpec],
    track_values_of: Sequence[Sequence[str | None]],
    name: str,
    n_features: int,
) -> list[str | None]:
    """Per-feature values of the named track, or all-None when absent."""
    idx = next((i for i, t in enumerate(track_specs) if t.name == name), None)
    if idx is None:
        return [None] * n_features
    return [list(tv)[idx] for tv in track_values_of]


class SpeciesModel(NamedTuple):
    """A fitted model plus its vocab/corpus structures.

    `track_specs` is the ordered attribute-track list ([] for species-only,
    [item, ability] for the species+item+ability build); `track_values_of` is
    the per-feature per-track value list (one entry per track, in `track_specs`
    order). `item_of`/`ability_of` are derived convenience views onto those
    tracks -- callers that only need the item (uniqueness, the current fit) keep
    working unchanged."""

    vocab: list[str]
    m: NDArray[np.float64]
    J: NDArray[np.float64]
    h: NDArray[np.float64]
    team_counts: Counter
    species_of: list[str]
    track_values_of: list[list[str | None]]
    track_specs: list[TrackSpec]
    latest_date: str

    @property
    def item_of(self) -> list[str | None]:
        return _track_value_view(self.track_specs, self.track_values_of, "item", len(self.vocab))

    @property
    def ability_of(self) -> list[str | None]:
        return _track_value_view(self.track_specs, self.track_values_of, "ability", len(self.vocab))


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


def format_triple(species: str, item: str, ability: str) -> str:
    """Display form for a (species, item, ability) vocab string:
    'Species @ Item (Ability)', with the ability always appended (every member
    has an ability). An itemless member -- item the string "None" (or Python
    None) -- drops the '@ Item' part exactly as `format_pair` does, keeping only
    the ability parenthetical: 'Species (Ability)'."""
    base = species if item in (None, "None") else f"{species} @ {item}"
    return f"{base} ({ability})"


def build_species_model(
    *,
    regulation: str = CURRENT_REGULATION,
    min_team_count: int = PHASE2_MIN_TEAM_COUNT,
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
        prior_J, prior_h = _align_prior(vocab, prior.vocab, prior.J, prior.h)

    # Species are distinct per vocab entry, so the species model's ensemble
    # needs no uniqueness lookups -- pass None/None.
    J, h = _fit_jh(
        X, method=method, lam=lam, w=w,
        species_of=None, item_of=None,
        track_values=None, track_names=None,
        prior_J=prior_J, prior_h=prior_h,
        intercept_prior_weight=intercept_prior_weight,
        boltzmann_opts=boltzmann_opts,
    )
    species_of = list(vocab)
    # Species-only: no attribute tracks, so each feature has an empty value list.
    track_values_of: list[list[str | None]] = [[] for _ in vocab]
    return SpeciesModel(
        vocab=vocab, m=m, J=J, h=h, team_counts=team_counts,
        species_of=species_of, track_values_of=track_values_of,
        track_specs=[], latest_date=latest_date,
    )


def build_species_item_model(
    *,
    regulation: str = CURRENT_REGULATION,
    min_team_count: int = PHASE2_MIN_TEAM_COUNT,
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
    # Features are (species, item, ability) triples. Normalize a missing item
    # (Python None) to the string "None" so it becomes an ordinary Potts state (a
    # real item name) rather than a bare-species feature -- merging the corpus's
    # split itemless duplicates. Ability is always a real string (ingest drops
    # abilityless members). The full triple is the vocab-cutoff key, so a rare
    # ability variant is cut while near-degenerate abilities keep most triples at
    # pair-level counts.
    teams = [
        [
            (species, "None" if item is None else item, ability)
            for species, item, ability in o.members
        ]
        for o in observations
    ]

    prior: SpeciesModel | None = None
    if prior_regulation is not None:
        prior = build_species_item_model(regulation=prior_regulation)

    raw_counts = Counter(triple for team in teams for triple in team)
    weighted_counts: dict[tuple[str, str, str], float] = {}
    for ti, team in enumerate(teams):
        for triple in team:
            weighted_counts[triple] = weighted_counts.get(triple, 0.0) + w[ti]
    triple_set: set[tuple[str, str, str]] = {
        t for t, c in weighted_counts.items()
        if c >= min_team_count and raw_counts[t] >= min_team_count
    }
    if prior is not None:
        # prior (species, item, ability) triples, folded in unconditionally. The
        # prior is a fitted species+item model, so its item view is a real string
        # ("None" for itemless, never Python None) and its ability view is always
        # a real string; coerce for the type checker.
        triple_set |= {
            (s, i if i is not None else "None", a)
            for s, i, a in zip(prior.species_of, prior.item_of, prior.ability_of)
            if a is not None
        }
    triple_list = sorted(triple_set, key=lambda t: format_triple(t[0], t[1], t[2]))
    vocab = [format_triple(s, i, a) for s, i, a in triple_list]
    triple_to_idx = {t: i for i, t in enumerate(triple_list)}
    V = len(vocab)

    species_of = [s for s, _, _ in triple_list]
    track_values_of: list[list[str | None]] = [[i, a] for _, i, a in triple_list]
    track_specs = [ITEM_TRACK, ABILITY_TRACK]
    # The item track is the uniqueness dimension the fit consumes; the Potts
    # kernel factors the bank over all tracks (per-track value lists, in
    # `track_specs` order).
    item_of: list[str | None] = [i for _, i, _ in triple_list]
    fit_track_values: list[list[str | None]] = [
        [tv[t] for tv in track_values_of] for t in range(len(track_specs))]
    fit_track_names = [t.name for t in track_specs]

    X = np.zeros((len(teams), V), dtype=np.int8)
    for ti, team in enumerate(teams):
        for triple in team:
            j = triple_to_idx.get(triple)
            if j is not None:
                X[ti, j] = 1
    m = (w @ X) / w.sum()

    team_counts: Counter[frozenset[str]] = Counter()
    for team in teams:
        if all(triple in triple_to_idx for triple in team):
            team_counts[frozenset(format_triple(s, i, a) for s, i, a in team)] += 1

    prior_J = prior_h = None
    if prior is not None:
        prior_J, prior_h = _align_prior(vocab, prior.vocab, prior.J, prior.h)

    # Features sharing a species or item mean the ensemble enforces
    # no-duplicate-species and no-duplicate-item. Ability is not a uniqueness
    # dimension (two members may share an ability), so the fit takes only the
    # item lookup; abilities just make more distinct features.
    J, h = _fit_jh(
        X, method=method, lam=lam, w=w,
        species_of=species_of, item_of=item_of,
        track_values=fit_track_values, track_names=fit_track_names,
        prior_J=prior_J, prior_h=prior_h,
        intercept_prior_weight=intercept_prior_weight,
        boltzmann_opts=boltzmann_opts,
    )
    return SpeciesModel(
        vocab=vocab, m=m, J=J, h=h, team_counts=team_counts,
        species_of=species_of, track_values_of=track_values_of,
        track_specs=track_specs, latest_date=latest_date,
    )
