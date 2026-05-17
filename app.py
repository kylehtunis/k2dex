"""Streamlit webapp for the Ising team auto-completer.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression

import helpers
import limitless_ingest
from constants import (
    PHASE1_MIN_USAGE,
    PHASE1_RIDGE_EPS,
    PHASE2_LR_C,
    PHASE2_MIN_TEAM_COUNT,
    PHASE2_MIN_TEAMS,
    TEAM_SIZE,
)


DATA_PATH_PHASE1 = "gen9championsvgc2026regma-1760.json"

# Log-spaced options for the two sliders. Both T and field_weight operate in
# log space (T governs Boltzmann factors exp(-ΔH/T); field_weight scales h,
# which is itself a log-odds), so linear sliders waste resolution at small
# values. field_weight includes 0.0 as a special-case for pure-pairwise mode.
TEMPERATURE_OPTIONS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
FIELD_WEIGHT_OPTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ANNEAL_T_START_OPTIONS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
ANNEAL_T_END_OPTIONS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
PT_T_MIN_OPTIONS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
PT_T_MAX_OPTIONS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


@st.cache_resource(show_spinner="Loading Phase 1 (Gaussian / Smogon chaos)...")
def load_model_phase1() -> tuple[
    list[str], np.ndarray, np.ndarray, np.ndarray,
    Counter[frozenset[str]] | None,
    list[str], list[str | None],
]:
    """Phase 1: Gaussian / precision-matrix inverse Ising from Smogon chaos JSON.

    Returns (vocab, m, J, h, team_counts, species_of, item_of). h is derived from
    mean-field self-consistency `h = logit(m) - J @ m`. team_counts is None for
    Phase 1 (chaos JSON has aggregate statistics, not per-team rosters).
    species_of is the vocab itself (each forme is a unique entry, no duplicates
    by construction); item_of is all-None (items are baked into Phase 1 forme
    names like 'Charizard-Mega-Y' rather than tracked separately). The sampler's
    uniqueness constraint is thus inert for Phase 1 -- the model continues to
    learn forme exclusion via strong -J as documented.
    """
    chaos = helpers.load_chaos(DATA_PATH_PHASE1)
    vocab = helpers.build_vocab(chaos, min_usage=PHASE1_MIN_USAGE)
    C = helpers.build_cooccurrence(chaos, vocab)
    m, p_joint = helpers.binary_moments(chaos, vocab, C, team_size=TEAM_SIZE)
    Corr = helpers.binary_correlation(m, p_joint)
    J, _ = helpers.ising_gaussian(Corr, eps=PHASE1_RIDGE_EPS)
    h = np.log(m / (1 - m)) - J @ m
    species_of = list(vocab)
    item_of: list[str | None] = [None] * len(vocab)
    return vocab, m, J, h, None, species_of, item_of


@st.cache_resource(show_spinner="Loading Phase 2 (pseudo-likelihood / Limitless teams)...")
def load_model_phase2() -> tuple[
    list[str], np.ndarray, np.ndarray, np.ndarray,
    Counter[frozenset[str]],
    list[str], list[str | None],
]:
    """Phase 2: pseudo-likelihood inverse Ising from Limitless tournament team data,
    projected to species-only vocab.

    Fits V per-spin logistic regressions (one per Pokemon in vocab) on the binary
    species-indicator matrix; intercepts -> h, coefficients -> J rows.
    Post-hoc symmetrization J = (J_asym + J_asym.T) / 2.

    The ingest format is now (species, item) tuples (v2 cache); Phase 2 projects
    these down to species-only via `limitless_ingest.species_only_teams`.

    Returns (vocab, m, J, h, team_counts, species_of, item_of). species_of equals
    vocab (each species is one entry); item_of is all-None (items dropped at
    projection). Sampler's uniqueness constraint is thus inert for Phase 2 --
    species can't duplicate within a single-entry-per-species vocab.
    """
    tournaments = limitless_ingest.ingest(min_teams=PHASE2_MIN_TEAMS)
    teams_full = limitless_ingest.all_teams(tournaments)
    teams = limitless_ingest.species_only_teams(teams_full)
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

    J_asym = np.zeros((V, V), dtype=np.float64)
    h = np.zeros(V, dtype=np.float64)
    for i in range(V):
        y = X[:, i]
        if y.sum() < 2 or (1 - y).sum() < 2:
            continue
        mask = np.ones(V, dtype=bool)
        mask[i] = False
        lr = LogisticRegression(penalty="l2", C=PHASE2_LR_C, solver="lbfgs", max_iter=1000)
        lr.fit(X[:, mask], y)
        h[i] = lr.intercept_[0]
        J_asym[i, mask] = lr.coef_[0]
    J = 0.5 * (J_asym + J_asym.T)
    np.fill_diagonal(J, 0.0)
    species_of = list(vocab)
    item_of: list[str | None] = [None] * len(vocab)
    return vocab, m, J, h, team_counts, species_of, item_of


@st.cache_resource(show_spinner="Loading Phase 3 (pseudo-likelihood / item-pair vocab)...")
def load_model_phase3() -> tuple[
    list[str], np.ndarray, np.ndarray, np.ndarray,
    Counter[frozenset[str]],
    list[str], list[str | None],
]:
    """Phase 3: pseudo-likelihood inverse Ising over (species, item) pairs.

    Same fit machinery as Phase 2 but with each Pokemon-item pair as its own
    feature. Restores held-item forme distinctions that Phase 1 had natively
    (Smogon vocab) but Phase 2 collapsed; lets the model see role specialization
    (one Charizard is a Mega sweeper, another is a Choice Scarf attacker).
    Vocab cutoff is the same `PHASE2_MIN_TEAM_COUNT` (pair must appear in at
    least that many teams).

    Display strings: 'Species @ Item' for items, bare species for itemless mons
    (see `format_pair`). species_of and item_of are populated so the sampler's
    uniqueness constraint can reject duplicate-species and duplicate-item swaps.

    team_counts in this phase counts each exact (species, item)-roster's
    occurrences in the corpus, keyed on frozensets of display strings.
    """
    tournaments = limitless_ingest.ingest(min_teams=PHASE2_MIN_TEAMS)
    teams = limitless_ingest.all_teams(tournaments)

    pair_counts = Counter(pair for team in teams for pair in team)
    pair_list_above_cutoff = [p for p, c in pair_counts.items() if c >= PHASE2_MIN_TEAM_COUNT]
    # Sort by display string for stable vocab ordering across reruns
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

    # team_counts at exact (species, item) granularity, keyed on display strings.
    # Skip teams with any out-of-vocab pair -- they can never be reached by
    # completion under this model and shouldn't appear in obs lookups.
    team_counts: Counter[frozenset[str]] = Counter()
    for team in teams:
        if all(pair in pair_to_idx for pair in team):
            team_counts[frozenset(format_pair(s, i) for s, i in team)] += 1

    J_asym = np.zeros((V, V), dtype=np.float64)
    h = np.zeros(V, dtype=np.float64)
    for i in range(V):
        y = X[:, i]
        if y.sum() < 2 or (1 - y).sum() < 2:
            continue
        mask = np.ones(V, dtype=bool)
        mask[i] = False
        lr = LogisticRegression(penalty="l2", C=PHASE2_LR_C, solver="lbfgs", max_iter=1000)
        lr.fit(X[:, mask], y)
        h[i] = lr.intercept_[0]
        J_asym[i, mask] = lr.coef_[0]
    J = 0.5 * (J_asym + J_asym.T)
    np.fill_diagonal(J, 0.0)
    return vocab, m, J, h, team_counts, species_of, item_of


def team_obs_count(
    state: np.ndarray,
    vocab: list[str],
    team_counts: Counter[frozenset[str]] | None,
) -> int | None:
    """Lookup of how many times this exact 6-Pokemon team appeared in the ingested
    tournament corpus. None when no team-level data is available (Phase 1)."""
    if team_counts is None:
        return None
    team = frozenset(vocab[i] for i in np.where(state)[0])
    return team_counts[team]


def min_swaps_to_observed(
    state: np.ndarray,
    vocab: list[str],
    team_counts: Counter[frozenset[str]] | None,
) -> int | None:
    """Minimum number of slot swaps to transform this team into any team in
    the corpus. 0 means the team itself was observed; values >= 1 measure
    how far the model's suggestion sits from realized teams. Useful to spot
    when a high-probability completion is a Hamming-1 variant of a known team
    (defensible "model picked a near-neighbor of the meta") vs a globally
    distinct configuration (likely overfit or genuine discovery)."""
    if not team_counts:
        return None
    team = frozenset(vocab[i] for i in np.where(state)[0])
    return min(TEAM_SIZE - len(team & obs) for obs in team_counts)


def intra_team_sum_j(state: np.ndarray, J: np.ndarray) -> float:
    """Sum of pairwise couplings over the team's unordered pairs:
    sum_{i<j: both in team} J_ij = 0.5 * s' J s. Measures structural coherence
    under the fitted model -- the pairwise contribution to (-raw_E). A team
    can have low raw_E either from popular members (large h.s) or coherent
    pairs (large Sigma J); decomposing makes that visible."""
    s = state.astype(np.float64)
    return float(0.5 * s @ J @ s)


def format_pair(species: str, item: str | None) -> str:
    """Display-friendly form of a (species, item) pair used as Phase 3 vocab
    strings. Bare species for itemless mons; otherwise 'Species @ Item'."""
    if item is None:
        return species
    return f"{species} @ {item}"


def _swap_violates_uniqueness(
    i_in: int,
    out_k: int,
    on_nf: list[int],
    fixed_species: set[str],
    fixed_items: set[str],
    species_of: list[str] | None,
    item_of: list[str | None] | None,
) -> bool:
    """Return True if swapping i_in into the team -- replacing position out_k
    of on_nf -- would create a duplicate species or duplicate non-None item.

    Phase 2 (species-only vocab) and Phase 1 (forme-tagged vocab) pass
    species_of and item_of arrays that never trigger this check (each vocab
    entry is unique by species and no items are tracked). Phase 3 passes
    arrays where multiple vocab entries can share a species (one Charizard
    per item) -- this function rejects swaps that would put two Charizards
    or two Choice-Scarf holders on the same team."""
    if species_of is not None:
        target_species = species_of[i_in]
        if target_species in fixed_species:
            return True
        for k_idx, idx in enumerate(on_nf):
            if k_idx == out_k:
                continue
            if species_of[idx] == target_species:
                return True
    if item_of is not None:
        target_item = item_of[i_in]
        if target_item is not None:
            if target_item in fixed_items:
                return True
            for k_idx, idx in enumerate(on_nf):
                if k_idx == out_k:
                    continue
                if item_of[idx] == target_item:
                    return True
    return False


def _initialize_state(
    available: np.ndarray,
    n_to_fill: int,
    fixed_species: set[str],
    fixed_items: set[str],
    species_of: list[str] | None,
    item_of: list[str | None] | None,
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> np.ndarray | None:
    """Sample n_to_fill indices from `available` respecting species and item
    uniqueness against the fixed set. Greedy: shuffle `available`, walk it
    picking non-conflicting indices until full or exhausted. Retries up to
    max_attempts on failure. Returns None if no valid init exists -- e.g.,
    the user has fixed too many same-item mons.

    Falls back to uniform sampling when both lookup arrays are None (the
    Phase 1/2 case where no uniqueness check is meaningful)."""
    if species_of is None and item_of is None:
        if len(available) < n_to_fill:
            return None
        return rng.choice(available, size=n_to_fill, replace=False)
    for _ in range(max_attempts):
        chosen: list[int] = []
        used_species = set(fixed_species)
        used_items = set(fixed_items)
        for idx in rng.permutation(available):
            if len(chosen) == n_to_fill:
                break
            idx = int(idx)
            if species_of is not None and species_of[idx] in used_species:
                continue
            if item_of is not None:
                it = item_of[idx]
                if it is not None and it in used_items:
                    continue
            chosen.append(idx)
            if species_of is not None:
                used_species.add(species_of[idx])
            if item_of is not None and item_of[idx] is not None:
                used_items.add(item_of[idx])
        if len(chosen) == n_to_fill:
            return np.array(chosen)
    return None


def _build_constraint_sets(
    fixed_set: set[int],
    species_of: list[str] | None,
    item_of: list[str | None] | None,
) -> tuple[set[str], set[str]]:
    """Precompute the set of species and items occupied by the fixed mons,
    against which all swap proposals must check for collisions."""
    fixed_species: set[str] = (
        {species_of[fi] for fi in fixed_set} if species_of is not None else set()
    )
    fixed_items: set[str] = (
        {item_of[fi] for fi in fixed_set if item_of[fi] is not None}
        if item_of is not None else set()
    )
    return fixed_species, fixed_items


def swap_mcmc(
    J: np.ndarray,
    h: np.ndarray,
    team_size: int,
    fixed: list[int],
    excluded: list[int],
    field_weight: float,
    n_steps: int,
    temperature: float,
    seed: int,
    *,
    species_of: list[str] | None = None,
    item_of: list[str | None] | None = None,
) -> tuple[np.ndarray, float] | None:
    """Swap-move MCMC. Returns (samples, acceptance_rate) or None if over-constrained.

    `species_of` and `item_of` (Phase 3) enforce no-duplicate-species and
    no-duplicate-item via proposal rejection -- a swap is silently dropped if
    it would put two Charizards or two Choice-Scarf holders on the same team."""
    rng = np.random.default_rng(seed)
    n = len(h)
    fixed_set = set(fixed)
    excluded_set = set(excluded)
    fixed_species, fixed_items = _build_constraint_sets(fixed_set, species_of, item_of)

    state = np.zeros(n, dtype=bool)
    for i in fixed_set:
        state[i] = True
    n_to_fill = team_size - len(fixed_set)
    available = np.array([i for i in range(n) if i not in fixed_set and i not in excluded_set])
    if len(available) < n_to_fill:
        return None

    h_eff = field_weight * h
    samples = np.zeros((n_steps, n), dtype=bool)

    if n_to_fill == 0:
        # Team fully determined by `fixed`; no swaps possible.
        samples[:] = state
        return samples, 0.0

    init = _initialize_state(
        available, n_to_fill, fixed_species, fixed_items, species_of, item_of, rng,
    )
    if init is None:
        return None  # constraints make a valid init impossible
    state[init] = True
    on_nf = list(int(x) for x in init)
    off_nf = list(set(available.tolist()) - set(int(x) for x in init))
    state_f = state.astype(np.float64)
    accepted = 0
    proposed = 0

    for step in range(n_steps):
        if not off_nf:
            samples[step] = state
            continue
        out_k = rng.integers(len(on_nf))
        in_k = rng.integers(len(off_nf))
        i_out = on_nf[out_k]
        i_in = off_nf[in_k]
        if _swap_violates_uniqueness(
            i_in, out_k, on_nf, fixed_species, fixed_items, species_of, item_of,
        ):
            samples[step] = state
            continue  # constraint-rejected: no proposal counted, state unchanged
        delta_H = (h_eff[i_out] - h_eff[i_in]
                   + np.dot(J[i_out] - J[i_in], state_f)
                   + J[i_in, i_out])
        proposed += 1
        if delta_H <= 0 or rng.random() < np.exp(-delta_H / temperature):
            state[i_out] = False; state[i_in] = True
            state_f[i_out] = 0.0; state_f[i_in] = 1.0
            on_nf[out_k] = i_in
            off_nf[in_k] = i_out
            accepted += 1
        samples[step] = state
    return samples, (accepted / proposed if proposed else 0.0)


def team_energy(state_bool: np.ndarray, J: np.ndarray, h: np.ndarray) -> float:
    """Raw Ising energy H(s) = -h.s - 0.5 s'Js. Lower energy = more probable under the model."""
    s = state_bool.astype(np.float64)
    return float(-np.dot(h, s) - 0.5 * s @ J @ s)


def anneal_mcmc(
    J: np.ndarray,
    h: np.ndarray,
    team_size: int,
    fixed: list[int],
    excluded: list[int],
    field_weight: float,
    n_steps: int,
    t_start: float,
    t_end: float,
    seed: int,
    *,
    species_of: list[str] | None = None,
    item_of: list[str | None] | None = None,
) -> tuple[np.ndarray, float] | None:
    """Single simulated-annealing run with exponential cooling from t_start -> t_end.
    Returns (final_state, acceptance_rate) or None if over-constrained.

    `species_of` and `item_of` (Phase 3) reject swaps that would create
    duplicate species or duplicate non-None items; see `swap_mcmc`."""
    rng = np.random.default_rng(seed)
    n = len(h)
    fixed_set = set(fixed)
    excluded_set = set(excluded)
    fixed_species, fixed_items = _build_constraint_sets(fixed_set, species_of, item_of)

    state = np.zeros(n, dtype=bool)
    for i in fixed_set:
        state[i] = True
    n_to_fill = team_size - len(fixed_set)
    available = np.array([i for i in range(n) if i not in fixed_set and i not in excluded_set])
    if len(available) < n_to_fill:
        return None

    h_eff = field_weight * h
    if n_to_fill == 0:
        return state, 0.0

    init = _initialize_state(
        available, n_to_fill, fixed_species, fixed_items, species_of, item_of, rng,
    )
    if init is None:
        return None
    state[init] = True
    on_nf = list(int(x) for x in init)
    off_nf = list(set(available.tolist()) - set(int(x) for x in init))
    state_f = state.astype(np.float64)
    accepted = 0
    proposed = 0

    for step in range(n_steps):
        if not off_nf:
            continue
        T = t_start * (t_end / t_start) ** (step / max(n_steps - 1, 1))
        out_k = rng.integers(len(on_nf))
        in_k = rng.integers(len(off_nf))
        i_out = on_nf[out_k]
        i_in = off_nf[in_k]
        if _swap_violates_uniqueness(
            i_in, out_k, on_nf, fixed_species, fixed_items, species_of, item_of,
        ):
            continue
        delta_H = (h_eff[i_out] - h_eff[i_in]
                   + np.dot(J[i_out] - J[i_in], state_f)
                   + J[i_in, i_out])
        proposed += 1
        if delta_H <= 0 or rng.random() < np.exp(-delta_H / T):
            state[i_out] = False; state[i_in] = True
            state_f[i_out] = 0.0; state_f[i_in] = 1.0
            on_nf[out_k] = i_in
            off_nf[in_k] = i_out
            accepted += 1
    return state, (accepted / proposed if proposed else 0.0)


def parallel_tempered_mcmc(
    J: np.ndarray,
    h: np.ndarray,
    team_size: int,
    fixed: list[int],
    excluded: list[int],
    field_weight: float,
    t_ladder: np.ndarray,
    n_steps: int,
    burn_in: int,
    swap_interval: int,
    seed: int,
    *,
    species_of: list[str] | None = None,
    item_of: list[str | None] | None = None,
) -> tuple[np.ndarray, float, float] | None:
    """Parallel-tempered MCMC. K chains run in parallel at temperatures `t_ladder`
    (sorted ascending, so index 0 is the target/cold chain). Every `swap_interval`
    sweeps, propose adjacent-chain state swaps with the standard replica-exchange
    acceptance `min(1, exp((1/T_lo - 1/T_hi)(E_lo - E_hi)))`. Samples are collected
    from the cold chain only after burn-in.

    `species_of` and `item_of` (Phase 3) reject within-chain swaps that would
    create duplicate species or items; see `swap_mcmc`. Replica-exchange
    swaps (between chains) don't need a uniqueness check -- both chains are
    individually valid, so swapping whole states preserves validity.

    Returns (cold_chain_samples, mean_local_accept, mean_swap_accept) or None.
    """
    rng = np.random.default_rng(seed)
    n = len(h)
    K = len(t_ladder)
    fixed_set = set(fixed)
    excluded_set = set(excluded)
    fixed_species, fixed_items = _build_constraint_sets(fixed_set, species_of, item_of)

    available = np.array([i for i in range(n) if i not in fixed_set and i not in excluded_set])
    n_to_fill = team_size - len(fixed_set)
    if len(available) < n_to_fill:
        return None

    h_eff = field_weight * h

    # Per-chain state: K independent initializations
    states: list[np.ndarray] = []
    on_nfs: list[list[int]] = []
    off_nfs: list[list[int]] = []
    state_fs: list[np.ndarray] = []
    energies: list[float] = []

    for _ in range(K):
        st_k = np.zeros(n, dtype=bool)
        for i in fixed_set:
            st_k[i] = True
        if n_to_fill > 0:
            init = _initialize_state(
                available, n_to_fill, fixed_species, fixed_items, species_of, item_of, rng,
            )
            if init is None:
                return None
            st_k[init] = True
            on_nfs.append([int(x) for x in init])
            off_nfs.append(list(set(available.tolist()) - set(int(x) for x in init)))
        else:
            on_nfs.append([])
            off_nfs.append([])
        states.append(st_k)
        state_fs.append(st_k.astype(np.float64))
        energies.append(team_energy(st_k, J, h_eff))

    samples = np.zeros((n_steps, n), dtype=bool)
    local_accept = 0; local_propose = 0
    swap_accept = 0; swap_propose = 0

    for step in range(n_steps):
        # One local MH move in each chain at its own temperature
        for k in range(K):
            if not on_nfs[k] or not off_nfs[k]:
                continue
            T_k = t_ladder[k]
            out_idx = rng.integers(len(on_nfs[k]))
            in_idx = rng.integers(len(off_nfs[k]))
            i_out = on_nfs[k][out_idx]
            i_in = off_nfs[k][in_idx]
            if _swap_violates_uniqueness(
                i_in, out_idx, on_nfs[k], fixed_species, fixed_items, species_of, item_of,
            ):
                continue
            delta_H = (h_eff[i_out] - h_eff[i_in]
                       + np.dot(J[i_out] - J[i_in], state_fs[k])
                       + J[i_in, i_out])
            local_propose += 1
            if delta_H <= 0 or rng.random() < np.exp(-delta_H / T_k):
                states[k][i_out] = False; states[k][i_in] = True
                state_fs[k][i_out] = 0.0; state_fs[k][i_in] = 1.0
                on_nfs[k][out_idx] = i_in
                off_nfs[k][in_idx] = i_out
                energies[k] += delta_H
                local_accept += 1

        # Periodically propose replica-exchange swaps between adjacent T levels
        if step > 0 and step % swap_interval == 0:
            for k in range(K - 1):
                T_lo, T_hi = t_ladder[k], t_ladder[k + 1]
                beta_diff = 1.0 / T_lo - 1.0 / T_hi  # > 0 (cold has higher beta)
                delta = beta_diff * (energies[k] - energies[k + 1])
                swap_propose += 1
                if delta >= 0 or rng.random() < np.exp(delta):
                    states[k], states[k + 1] = states[k + 1], states[k]
                    on_nfs[k], on_nfs[k + 1] = on_nfs[k + 1], on_nfs[k]
                    off_nfs[k], off_nfs[k + 1] = off_nfs[k + 1], off_nfs[k]
                    state_fs[k], state_fs[k + 1] = state_fs[k + 1], state_fs[k]
                    energies[k], energies[k + 1] = energies[k + 1], energies[k]
                    swap_accept += 1

        samples[step] = states[0]

    return (
        samples[burn_in:],
        local_accept / max(local_propose, 1),
        swap_accept / max(swap_propose, 1),
    )


def meanfield_marginals(
    J: np.ndarray,
    h: np.ndarray,
    team_size: int,
    fixed: list[int],
    excluded: list[int],
    field_weight: float,
    *,
    species_of: list[str] | None = None,
    item_of: list[str | None] | None = None,
    n_iters: int = 200,
    tol: float = 1e-5,
    damp: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Damped mean-field iteration on the Ising-pair model with `fixed` clamped
    to 1 and `excluded` clamped to 0. Returns per-candidate marginals plus a
    `valid_mask` over candidates eligible to fill the remaining team slots
    (respecting Phase 3 uniqueness against fixed mons), and the number of
    iterations actually run.

    This is the same approximation `validation.ipynb:score_ising_meanfield`
    uses; that comparison validated MF as a ranking-faithful and ~100x cheaper
    proxy for swap-MCMC at field_weight=1, T=1 (median Spearman 0.93-0.95
    between MF and MCMC pair ranks, <=1.5 pp hit-rate delta). At very low
    field_weight or as a team-size-constrained sampler, MF has not been
    validated and PT remains the right tool.

    Returns None when fewer than (team_size - len(fixed)) valid candidates
    exist -- mirrors the failure mode of `_initialize_state`.
    """
    V = len(h)
    h_eff = field_weight * h
    fixed_mask = np.zeros(V, dtype=bool)
    fixed_mask[fixed] = True
    excluded_mask = np.zeros(V, dtype=bool)
    excluded_mask[excluded] = True

    # Uniqueness against fixed mons (Phase 3). Adds same-species and same-item
    # vocab entries to the "not a candidate" set, on top of fixed/excluded.
    fixed_species, fixed_items = _build_constraint_sets(set(fixed), species_of, item_of)
    uniq_invalid = np.zeros(V, dtype=bool)
    if species_of is not None and fixed_species:
        for i in range(V):
            if species_of[i] in fixed_species:
                uniq_invalid[i] = True
    if item_of is not None and fixed_items:
        for i in range(V):
            if item_of[i] is not None and item_of[i] in fixed_items:
                uniq_invalid[i] = True

    valid_mask = ~(fixed_mask | excluded_mask | uniq_invalid)
    if int(valid_mask.sum()) < team_size - len(fixed):
        return None

    m = 1.0 / (1.0 + np.exp(-h_eff))
    m[fixed_mask] = 1.0
    m[excluded_mask] = 0.0

    iters_used = n_iters
    for it in range(n_iters):
        m_new = 1.0 / (1.0 + np.exp(-(h_eff + J @ m)))
        m_new[fixed_mask] = 1.0
        m_new[excluded_mask] = 0.0
        free_mask = ~(fixed_mask | excluded_mask)
        delta = float(np.max(np.abs(m_new[free_mask] - m[free_mask]))) if free_mask.any() else 0.0
        m = damp * m_new + (1.0 - damp) * m
        m[fixed_mask] = 1.0
        m[excluded_mask] = 0.0
        if delta < tol:
            iters_used = it + 1
            break

    return m, valid_mask, iters_used


def greedy_optimize(
    J: np.ndarray,
    h: np.ndarray,
    team_size: int,
    starting_team: list[int],
    pinned: list[int],
    excluded: list[int],
    field_weight: float,
    *,
    species_of: list[str] | None = None,
    item_of: list[str | None] | None = None,
    max_swaps: int = 20,
) -> tuple[list[int], list[dict]]:
    """Greedy steepest-descent over single-swap moves on the field-weighted
    Ising energy `E_adj(s) = -(fw*h)·s - 0.5 s'Js`.

    At each step, evaluates every (non-pinned out-slot, valid in-candidate) swap
    in vectorized form and accepts the one with the most negative `ΔE_adj`.
    Stops at the first local minimum (no improving swap exists) or after
    `max_swaps` steps. Deterministic given (model, starting_team, pinned).

    The first swap is the model's largest single complaint about the starting
    team; the second is the next-largest given the first was applied; etc.
    The final team is the local optimum reachable by single-swap moves, NOT
    necessarily the global MAP -- multi-swap rearrangements may go lower.

    Returns (final_team_idx_list, chain), where chain entries have keys
    `step`, `out_idx`, `in_idx`, `delta_E_adj`, `energy_adj_after`,
    `energy_raw_after`, `sum_j_after`, `team_after` (sorted tuple of 6 idxs).
    """
    V = len(h)
    fw = field_weight
    current = set(starting_team)
    pinned_set = set(pinned)
    excluded_set = set(excluded)

    def team_e_adj(team_set: set[int]) -> float:
        arr = np.fromiter(team_set, dtype=np.int64, count=len(team_set))
        h_sum = float((fw * h[arr]).sum())
        j_sum = float(J[np.ix_(arr, arr)].sum() * 0.5)
        return -h_sum - j_sum

    def team_e_raw(team_set: set[int]) -> float:
        arr = np.fromiter(team_set, dtype=np.int64, count=len(team_set))
        h_sum = float(h[arr].sum())
        j_sum = float(J[np.ix_(arr, arr)].sum() * 0.5)
        return -h_sum - j_sum

    energy_adj = team_e_adj(current)
    chain: list[dict] = []

    for step in range(1, max_swaps + 1):
        current_arr = np.fromiter(current, dtype=np.int64, count=len(current))
        current_mask = np.zeros(V, dtype=bool)
        current_mask[current_arr] = True

        best_delta = 0.0
        best_out: int | None = None
        best_in: int | None = None

        for out_idx in list(current - pinned_set):
            others = current - {out_idx}
            others_arr = np.fromiter(others, dtype=np.int64, count=len(others))

            # Vectorized ΔE_adj across all candidate in_idxs (V-length array).
            # delta = -fw*(h[in] - h[out]) - (J[in, others].sum() - J[out, others].sum())
            h_part = -fw * (h - h[out_idx])
            J_part = -(J[:, others_arr].sum(axis=1) - J[out_idx, others_arr].sum())
            delta_E_all = h_part + J_part

            # Validity mask for in_idx: not in current team, not excluded,
            # and (Phase 3) doesn't duplicate a species or item already on the team.
            valid = ~current_mask.copy()
            for ex in excluded_set:
                valid[ex] = False
            if species_of is not None:
                others_species = {species_of[j] for j in others}
                if others_species:
                    for i in range(V):
                        if valid[i] and species_of[i] in others_species:
                            valid[i] = False
            if item_of is not None:
                others_items = {item_of[j] for j in others if item_of[j] is not None}
                if others_items:
                    for i in range(V):
                        if valid[i] and item_of[i] is not None and item_of[i] in others_items:
                            valid[i] = False
            delta_E_all = np.where(valid, delta_E_all, np.inf)

            cand_best_in = int(np.argmin(delta_E_all))
            cand_best_delta = float(delta_E_all[cand_best_in])
            if cand_best_delta < best_delta:
                best_delta = cand_best_delta
                best_out = int(out_idx)
                best_in = cand_best_in

        if best_out is None:  # local minimum reached
            break

        current.discard(best_out)
        current.add(best_in)
        energy_adj += best_delta
        new_arr = np.fromiter(current, dtype=np.int64, count=len(current))
        sum_j_after = float(J[np.ix_(new_arr, new_arr)].sum() * 0.5)
        chain.append({
            "step": step,
            "out_idx": best_out,
            "in_idx": best_in,
            "delta_E_adj": best_delta,
            "energy_adj_after": energy_adj,
            "energy_raw_after": team_e_raw(current),
            "sum_j_after": sum_j_after,
            "team_after": tuple(sorted(int(i) for i in current)),
        })

    return sorted(current), chain


@st.cache_data(show_spinner=False)
def parallel_tempered_distribution(
    phase: str,  # in cache key only; differentiates models when index values collide across phases
    fixed_idx_tuple: tuple[int, ...],
    excluded_idx_tuple: tuple[int, ...],
    field_weight: float,
    t_min: float,
    t_max: float,
    K: int,
    n_runs: int,
    n_steps: int,
    burn_in: int,
    swap_interval: int,
    _J: np.ndarray,
    _h: np.ndarray,
    _species_of: list[str] | None = None,
    _item_of: list[str | None] | None = None,
) -> tuple[list[tuple[tuple[int, ...], int]] | None, int, float, float, list[float]]:
    """Multiple PT runs, aggregated. Each run runs K chains in parallel at the
    same geometrically-spaced temperature ladder; only cold-chain samples are
    kept. `_species_of` and `_item_of` (Phase 3) propagate uniqueness constraints
    to the sampler. Returns (distribution, n_kept_total, mean_local_accept,
    mean_swap_accept, ladder)."""
    fixed_idx = list(fixed_idx_tuple)
    excluded_idx = list(excluded_idx_tuple)
    fixed_set = set(fixed_idx)
    t_ladder = np.geomspace(t_min, t_max, K)

    counts: dict[tuple[int, ...], int] = {}
    n_kept = 0
    local_rates: list[float] = []
    swap_rates: list[float] = []
    rng_master = np.random.default_rng(0)

    for _ in range(n_runs):
        run_seed = int(rng_master.integers(2**31))
        result = parallel_tempered_mcmc(
            _J, _h, TEAM_SIZE, fixed_idx, excluded_idx, field_weight,
            t_ladder, n_steps, burn_in, swap_interval, run_seed,
            species_of=_species_of, item_of=_item_of,
        )
        if result is None:
            return None, 0, 0.0, 0.0, list(t_ladder)
        cold_samples, local_rate, swap_rate = result
        local_rates.append(local_rate)
        swap_rates.append(swap_rate)
        for state in cold_samples:
            comp = tuple(sorted(int(i) for i in np.where(state)[0] if i not in fixed_set))
            counts[comp] = counts.get(comp, 0) + 1
            n_kept += 1

    dist = sorted(counts.items(), key=lambda x: -x[1])
    return dist, n_kept, float(np.mean(local_rates)), float(np.mean(swap_rates)), list(t_ladder)


@st.cache_data(show_spinner=False)
def run_anneals(
    phase: str,  # in cache key only; see parallel_tempered_distribution
    fixed_idx_tuple: tuple[int, ...],
    excluded_idx_tuple: tuple[int, ...],
    field_weight: float,
    n_runs: int,
    n_steps: int,
    t_start: float,
    t_end: float,
    _J: np.ndarray,
    _h: np.ndarray,
    _species_of: list[str] | None = None,
    _item_of: list[str | None] | None = None,
) -> tuple[list[tuple[tuple[int, ...], int]] | None, float]:
    """Run `n_runs` independent annealing chains. `_species_of` and `_item_of`
    (Phase 3) propagate uniqueness constraints to the annealer.
    Returns (sorted list of (completion, count), mean_acceptance_rate)."""
    counts: dict[tuple[int, ...], int] = {}
    accept_rates: list[float] = []
    fixed_set = set(fixed_idx_tuple)
    rng_master = np.random.default_rng(0)

    for _ in range(n_runs):
        run_seed = int(rng_master.integers(2**31))
        result = anneal_mcmc(
            _J, _h, TEAM_SIZE,
            list(fixed_idx_tuple), list(excluded_idx_tuple),
            field_weight, n_steps, t_start, t_end, run_seed,
            species_of=_species_of, item_of=_item_of,
        )
        if result is None:
            return None, 0.0
        state, accept_rate = result
        accept_rates.append(accept_rate)
        comp = tuple(sorted(int(i) for i in np.where(state)[0] if i not in fixed_set))
        counts[comp] = counts.get(comp, 0) + 1

    return sorted(counts.items(), key=lambda x: -x[1]), float(np.mean(accept_rates))


@st.cache_data(show_spinner=False)
def sample_distribution(
    phase: str,  # in cache key only; see parallel_tempered_distribution
    fixed_idx_tuple: tuple[int, ...],
    excluded_idx_tuple: tuple[int, ...],
    field_weight: float,
    temperature: float,
    n_chains: int,
    n_steps: int,
    burn_in: int,
    _J: np.ndarray,
    _h: np.ndarray,
    _species_of: list[str] | None = None,
    _item_of: list[str | None] | None = None,
) -> tuple[list[tuple[tuple[int, ...], int]] | None, int, float]:
    """Multi-chain swap sampling. `_species_of` and `_item_of` (Phase 3)
    propagate uniqueness constraints to the sampler.
    Returns (distribution, n_kept_samples, mean_accept_rate)."""
    counts: dict[tuple[int, ...], int] = {}
    n_kept = 0
    accept_rates: list[float] = []
    fixed_idx = list(fixed_idx_tuple)
    excluded_idx = list(excluded_idx_tuple)
    fixed_set = set(fixed_idx)
    rng_master = np.random.default_rng(0)

    for _ in range(n_chains):
        chain_seed = int(rng_master.integers(2**31))
        result = swap_mcmc(_J, _h, TEAM_SIZE, fixed_idx, excluded_idx,
                           field_weight, n_steps, temperature, chain_seed,
                           species_of=_species_of, item_of=_item_of)
        if result is None:
            return None, 0, 0.0
        samples, accept_rate = result
        accept_rates.append(accept_rate)
        for state in samples[burn_in:]:
            comp = tuple(sorted(int(i) for i in np.where(state)[0] if i not in fixed_set))
            counts[comp] = counts.get(comp, 0) + 1
            n_kept += 1

    dist = sorted(counts.items(), key=lambda x: -x[1])
    return dist, n_kept, float(np.mean(accept_rates))


@st.cache_data(show_spinner=False)
def meanfield_distribution(
    phase: str,  # in cache key only; see parallel_tempered_distribution
    fixed_idx_tuple: tuple[int, ...],
    excluded_idx_tuple: tuple[int, ...],
    field_weight: float,
    n_iters: int,
    tol: float,
    _J: np.ndarray,
    _h: np.ndarray,
    _species_of: list[str] | None = None,
    _item_of: list[str | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Thin cached wrapper around `meanfield_marginals`. Cheap to recompute,
    but cached for consistency with the MCMC modes (and so the same fixed/
    field_weight combo hits the cache on re-render)."""
    return meanfield_marginals(
        _J, _h, TEAM_SIZE,
        list(fixed_idx_tuple), list(excluded_idx_tuple),
        field_weight,
        species_of=_species_of, item_of=_item_of,
        n_iters=n_iters, tol=tol,
    )


@st.cache_data(show_spinner=False)
def greedy_optimize_chain(
    phase: str,  # in cache key only; see parallel_tempered_distribution
    starting_team_tuple: tuple[int, ...],
    pinned_tuple: tuple[int, ...],
    excluded_idx_tuple: tuple[int, ...],
    field_weight: float,
    max_swaps: int,
    _J: np.ndarray,
    _h: np.ndarray,
    _species_of: list[str] | None = None,
    _item_of: list[str | None] | None = None,
) -> tuple[list[int], list[dict]]:
    """Cached wrapper around `greedy_optimize`. Cheap to recompute (deterministic
    greedy descent over a single starting team), but cached for re-render
    stability."""
    return greedy_optimize(
        _J, _h, TEAM_SIZE,
        list(starting_team_tuple), list(pinned_tuple),
        list(excluded_idx_tuple), field_weight,
        species_of=_species_of, item_of=_item_of,
        max_swaps=max_swaps,
    )


def main() -> None:
    st.set_page_config(page_title="VGC team auto-completer", layout="wide")
    st.title("VGC team auto-completer")

    with st.sidebar:
        st.subheader("Model")
        phase = st.radio(
            "Inverse Ising fit",
            ["Phase 1 (Gaussian)", "Phase 2 (PL, species)", "Phase 3 (PL, item-pair)"],
            label_visibility="collapsed",
            help=(
                "**Phase 1** — Gaussian / precision-matrix approximation on Smogon "
                "chaos stats. Fast to fit but systematically under-magnitudes strong "
                "negative couplings (mutually-exclusive pairs).\n\n"
                "**Phase 2** — pseudo-likelihood on Limitless tournament team rosters, "
                "species-only vocab (items collapsed). Wider dynamic range on −J than "
                "Phase 1. Held-item formes (Mega Y vs Mega X) are invisible here.\n\n"
                "**Phase 3** — pseudo-likelihood on (species, item) pairs from the "
                "same Limitless data. Restores held-item forme distinctions and lets "
                "the model see role specialization (Choice Scarf vs Charizardite-Y "
                "Charizards as distinct features). Sampler enforces no-duplicate-"
                "species and no-duplicate-item by proposal rejection. First-time "
                "selection re-ingests if cache is at v1 (species-only) format."
            ),
        )

    if phase.startswith("Phase 1"):
        phase_key = "phase1"
        vocab, m, J, h, team_counts, species_of, item_of = load_model_phase1()
        st.caption(
            "Phase 1: Gaussian / precision-matrix inverse Ising fit on Smogon chaos "
            "stats. Sample teams of 6 from the conditional distribution given fixed "
            "(must appear) and excluded (must not appear) members."
        )
    elif phase.startswith("Phase 2"):
        phase_key = "phase2"
        vocab, m, J, h, team_counts, species_of, item_of = load_model_phase2()
        st.caption(
            "Phase 2: pseudo-likelihood inverse Ising fit on Limitless tournament "
            "team data, species-only vocab. Same sampling controls as Phase 1; "
            "(J, h) come from direct per-spin logistic regressions on real team "
            "rosters. The **obs** column shows how many times each suggested team "
            "appeared verbatim in the ingested corpus — a sanity check that the "
            "model's high-probability completions correspond to teams real players "
            "actually brought."
        )
    else:
        phase_key = "phase3"
        vocab, m, J, h, team_counts, species_of, item_of = load_model_phase3()
        st.caption(
            "Phase 3: pseudo-likelihood inverse Ising fit on (species, item) pairs "
            "from Limitless tournament data. Vocab entries are formatted "
            "'Species @ Item' (or bare species for itemless mons). The sampler "
            "enforces no-duplicate-species and no-duplicate-item constraints by "
            "rejecting illegal swap proposals. The **obs** column counts exact "
            "(species, item)-roster appearances in the corpus."
        )

    name_to_idx = {name: i for i, name in enumerate(vocab)}
    sorted_vocab = sorted(vocab, key=lambda v: -m[name_to_idx[v]])

    with st.sidebar:
        st.subheader("Team constraints")
        fixed_names = st.multiselect(
            "Fix (must appear, max 6)",
            sorted_vocab,
            default=[],
            max_selections=TEAM_SIZE,
            placeholder="Choose Pokemon to pin at s=1",
            key=f"fixed_{phase_key}",
        )
        excluded_names = st.multiselect(
            "Exclude (must NOT appear)",
            sorted_vocab,
            default=[],
            placeholder="Choose Pokemon to pin at s=0",
            key=f"excluded_{phase_key}",
        )

        overlap = set(fixed_names) & set(excluded_names)
        if overlap:
            st.error(f"Cannot be both fixed and excluded: {', '.join(overlap)}")
            st.stop()

        st.subheader("Mode")
        mode = st.radio(
            "Run mode",
            ["Sample distribution", "Anneal to MAP", "Parallel-tempered sample", "Mean-field (deterministic)", "Greedy team optimizer"],
            label_visibility="collapsed",
            help=(
                "**Sample distribution** — independent-chain MCMC at one temperature. "
                "Fast but at low T each chain gets stuck in one basin (frequencies "
                "reflect basin discovery, not Boltzmann weight). "
                "**Anneal to MAP** — multiple cooling-schedule runs, returns the "
                "MAP teams each run converged to. "
                "**Parallel-tempered sample** — replica-exchange MCMC across a "
                "temperature ladder. Hot chains explore broadly, swap moves transmit "
                "good states down to the cold chain. The cold-chain samples are "
                "true Boltzmann draws at the target T. "
                "**Mean-field (deterministic)** — analytic per-candidate marginals "
                "via damped MF iteration. Instant, no sampling noise; validated as "
                "a ranking-faithful proxy for PT at fw=1, T=1. Treats free slots as "
                "independent, so weaker than PT at low T or when free slots correlate. "
                "**Greedy team optimizer** — input a full starting team of 6, descend "
                "the energy landscape one steepest single-swap at a time, return the "
                "chain of swaps + per-team metrics. Deterministic local minimum from "
                "the starting team; the swap order is a priority-ranked critique of "
                "the starting team."
            ),
        )

        st.subheader("Sampler")
        field_weight = st.select_slider(
            "Field weight (h scale)",
            options=FIELD_WEIGHT_OPTIONS,
            value=0.2,
            help=(
                "Scales the field h before sampling (log-spaced). "
                "1.0 = data-calibrated posterior (meta-biased — popular mons dominate). "
                "0.0 = pure pairwise model (no popularity prior — archetype-coherent "
                "completions driven only by J)."
            ),
        )

        if mode == "Sample distribution":
            temperature = st.select_slider(
                "Temperature",
                options=TEMPERATURE_OPTIONS,
                value=0.05,
                help=(
                    "Sampling temperature (log-spaced). Lower = sharper distribution. "
                    "Pair low T (0.02-0.05) with low field_weight for sharply peaked "
                    "archetype completions."
                ),
            )
            with st.expander("Sampling details", expanded=False):
                n_chains = st.slider("Chains", 5, 50, 20)
                n_steps = st.slider("Steps per chain", 2000, 20000, 8000, step=1000)
                burn_in = st.slider("Burn-in", 0, 5000, 2000, step=500)
                top_k = st.slider("Top-K teams shown", 5, 50, 20)
            run = st.button("Sample teams", type="primary", use_container_width=True)
        elif mode == "Anneal to MAP":
            t_start = st.select_slider(
                "Start temperature",
                options=ANNEAL_T_START_OPTIONS,
                value=3.0,
                help="Hot temperature — high enough to explore broadly at the start.",
            )
            t_end = st.select_slider(
                "End temperature",
                options=ANNEAL_T_END_OPTIONS,
                value=0.02,
                help="Cold final temperature — sharp enough to lock into a local minimum.",
            )
            with st.expander("Annealing details", expanded=False):
                n_runs = st.slider("Independent anneal runs", 5, 50, 20)
                anneal_steps = st.slider("Steps per run", 5000, 50000, 20000, step=5000)
            run = st.button("Anneal teams", type="primary", use_container_width=True)
        elif mode == "Mean-field (deterministic)":
            with st.expander("MF details", expanded=False):
                mf_n_iters = st.slider(
                    "Max iterations", 50, 500, 200, step=50,
                    help="Cap on damped fixed-point iterations. Typical convergence "
                         "is <50 iterations; this is just a safety bound.",
                )
                mf_tol_log = st.select_slider(
                    "Convergence tolerance (log10)", options=[-3, -4, -5, -6], value=-5,
                    help="Stop when max marginal change between iterations falls below "
                         "10^tol. -5 is plenty for ranking; tighter rarely changes the top-K.",
                )
                top_k = st.slider("Top-K candidates shown", 5, 50, 20)
            mf_tol = 10.0 ** mf_tol_log
            run = st.button("Compute MF marginals", type="primary", use_container_width=True)
        elif mode == "Greedy team optimizer":
            starting_team_names = st.multiselect(
                "Your team (exactly 6)",
                sorted_vocab,
                default=[],
                max_selections=TEAM_SIZE,
                placeholder="Choose your team of 6",
                key=f"opt_team_{phase_key}",
            )
            if len(starting_team_names) != TEAM_SIZE:
                st.warning(f"Need exactly 6 mons (have {len(starting_team_names)}).")
            with st.expander("Optimizer details", expanded=False):
                max_swaps = st.slider(
                    "Max swaps", 1, 30, 15,
                    help="Hard cap on swap chain length. Greedy descent usually "
                         "converges in 3-10 swaps; this is just a safety bound.",
                )
            run = st.button(
                "Optimize team", type="primary", use_container_width=True,
                disabled=(len(starting_team_names) != TEAM_SIZE),
            )
        else:  # Parallel-tempered sample
            pt_t_min = st.select_slider(
                "Target temperature (cold chain)",
                options=PT_T_MIN_OPTIONS,
                value=0.1,
                help="Samples are collected from a chain at this T. Lower = sharper "
                     "Boltzmann distribution at the target.",
            )
            pt_t_max = st.select_slider(
                "Max temperature (hot chain)",
                options=PT_T_MAX_OPTIONS,
                value=2.0,
                help="Top of the replica ladder. Hot enough that the chain can cross "
                     "basin barriers freely. Should be well above the energy scale.",
            )
            pt_K = st.slider(
                "Ladder levels (K)", 3, 12, 7,
                help="Number of replicas. More levels = better swap acceptance between "
                     "adjacent T but slower per sweep. ~7 is a good default.",
            )
            with st.expander("PT details", expanded=False):
                pt_n_runs = st.slider("Independent PT runs", 1, 10, 3)
                pt_n_steps = st.slider("Sweeps per run", 2000, 30000, 10000, step=1000)
                pt_burn_in = st.slider("Burn-in", 0, 10000, 3000, step=500)
                pt_swap_interval = st.slider("Swap proposal interval", 1, 50, 10)
                top_k = st.slider("Top-K teams shown", 5, 50, 20)
            run = st.button("PT sample", type="primary", use_container_width=True)

    if not run:
        if mode == "Sample distribution":
            st.info(
                "Choose constraints in the sidebar and click **Sample teams**.\n\n"
                "Defaults `field_weight=0.2, T=0.05` lean archetype-coherent. "
                "Try `field_weight=0.0, T=0.02` for pure pairwise mode, or "
                "`field_weight=1.0, T=0.1+` for the data-calibrated posterior."
            )
        elif mode == "Anneal to MAP":
            st.info(
                "Choose constraints in the sidebar and click **Anneal teams**.\n\n"
                "Annealing runs simulated cooling from `t_start` to `t_end` "
                "and returns whatever team each independent run converges to. "
                "If multiple runs converge to the same team, that's the model's "
                "robust MAP under the chosen `field_weight`. Multiple distinct "
                "results indicate a shallow / multimodal energy landscape."
            )
        elif mode == "Mean-field (deterministic)":
            st.info(
                "Choose constraints in the sidebar and click **Compute MF marginals**.\n\n"
                "Mean-field iteration computes per-candidate marginal probabilities "
                "P(mon in team | fixed mons) directly — no sampling, no temperature, "
                "no chains. Output is (a) the greedy MF-MAP completion (top free-slot "
                "candidates by marginal, respecting uniqueness) and (b) a ranked list "
                "of the top-K single-slot candidates with their marginals. "
                "Free slots are treated as independent given fixed: fast and ~99% "
                "agreement with PT at fw=1, T=1 in `validation.ipynb`, but blind to "
                "joint structure between free slots. Use PT when you specifically "
                "want low-T archetype coherence or strongly-correlated free slots."
            )
        elif mode == "Greedy team optimizer":
            st.info(
                "Enter your team of 6 in the sidebar and click **Optimize team**.\n\n"
                "Greedy steepest-descent: at each step, the model evaluates every "
                "possible single-mon swap and applies the one that drops `adj E` the "
                "most. Stops at the first local minimum. The output is a chain of "
                "swaps with per-team metrics — **the order is diagnostic**: swap #1 "
                "is the model's biggest complaint about your starting team, swap #2 "
                "is the next-biggest given #1 was applied, etc. Final team is the "
                "local optimum reachable by single-swap moves from your starting team. "
                "`field_weight=1.0` optimizes for popularity (meta-fit); `field_weight=0.0` "
                "optimizes for pure pairwise structure (archetype coherence)."
            )
        else:
            st.info(
                "Choose constraints in the sidebar and click **PT sample**.\n\n"
                "Parallel tempering runs a ladder of `K` chains at log-spaced "
                "temperatures from `t_min` (cold target) to `t_max` (hot exploration). "
                "Hot chains traverse basins freely; replica-exchange swaps every "
                "`swap_interval` sweeps propagate good states down the ladder to "
                "the cold chain. **Cold-chain samples are true Boltzmann draws at "
                "the target T**, with proper basin-mixing — the fix for the "
                "frequency-vs-energy non-monotonicity you see in single-chain mode."
            )
        with st.expander("How to read the output"):
            st.markdown(
                "**Sample distribution mode**\n"
                "- **%** — empirical fraction of post-burn-in MCMC samples producing this completion.\n"
                "- **raw E** / **adj E** — Ising energy with full / rescaled `h` (see below).\n\n"
                "**Annealing mode**\n"
                "- **runs** — how many of `n_runs` independent annealings converged to this team. "
                "More = more robust MAP estimate; fewer distinct teams = sharper landscape.\n"
                "- **raw E** / **adj E** — Ising energy with full / rescaled `h`. Annealing minimizes "
                "**adj E** (the energy the sampler sees), so the top result has the lowest adj E.\n\n"
                "**Both modes**\n"
                "- **raw E** is `H(s) = -h·s - 0.5 s'Js` with the full data-calibrated field. "
                "Independent of sampler knobs; comparable across runs.\n"
                "- **adj E** is `H_adj(s) = -(field_weight·h)·s - 0.5 s'Js`. What the sampler/annealer "
                "actually optimizes. At `field_weight=1.0` raw and adj are identical; at `0.0`, adj "
                "drops the field term entirely."
            )
        return

    fixed_idx = sorted({name_to_idx[n] for n in fixed_names})
    excluded_idx = sorted({name_to_idx[n] for n in excluded_names})

    if mode == "Sample distribution":
        with st.spinner(f"Running {n_chains} chains × {n_steps} steps..."):
            dist, n_kept, accept_rate = sample_distribution(
                phase_key,
                tuple(fixed_idx), tuple(excluded_idx), field_weight, temperature,
                n_chains, n_steps, burn_in, J, h,
                _species_of=species_of, _item_of=item_of,
            )
        if dist is None:
            st.error("Not enough available Pokemon to fill the team after applying constraints.")
            return

        n_distinct = len(dist)
        top5_mass = sum(c for _, c in dist[:5]) / n_kept * 100
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total samples", f"{n_kept:,}")
        col2.metric("Distinct completions", f"{n_distinct:,}")
        col3.metric("Top-5 mass", f"{top5_mass:.2f}%",
                    help="Sum of probabilities for the five most-frequent completions. "
                         "High = sharply peaked posterior; low = diffuse.")
        col4.metric("MH accept %", f"{accept_rate * 100:.1f}%",
                    help="Fraction of swap proposals accepted, averaged across chains. "
                         "Healthy range 20-50%. Very low = chain stuck (rejecting most moves); "
                         "very high = proposals too trivial.")

        diag_header = " obs | Δ | Σ J |" if team_counts is not None else ""
        diag_sep = " ---: | ---: | ---: |" if team_counts is not None else ""
        md_lines = [
            f"| # | % | raw E | adj E |{diag_header} completion |",
            f"| ---: | ---: | ---: | ---: |{diag_sep} :--- |",
        ]
        for rank, (comp, count) in enumerate(dist[:top_k], 1):
            state = np.zeros(len(vocab), dtype=bool)
            for i in fixed_idx:
                state[i] = True
            for i in comp:
                state[i] = True
            prob_pct = (count / n_kept) * 100
            raw_E = team_energy(state, J, h)
            adj_E = team_energy(state, J, field_weight * h)
            names = ", ".join(vocab[i] for i in comp)
            if team_counts is not None:
                obs = team_obs_count(state, vocab, team_counts)
                delta = min_swaps_to_observed(state, vocab, team_counts)
                sum_j = intra_team_sum_j(state, J)
                diag_cells = f" {obs} | {delta} | {sum_j:+.3f} |"
            else:
                diag_cells = ""
            md_lines.append(
                f"| {rank} | {prob_pct:.2f}% | {raw_E:+.3f} | {adj_E:+.3f} |{diag_cells} {names} |"
            )

        st.markdown("\n".join(md_lines))
        obs_caption = (
            "  **obs** — exact-match count of this team in the corpus. 0 means no "
            "player brought this exact roster. "
            "**Δ** — minimum slot-swaps to the nearest observed team (0 ↔ exact "
            "match). A high-probability completion with Δ=1 is a 1-swap variant of "
            "a real team (defensible discovery / fine-tuning); Δ≥3 is a globally "
            "distinct configuration (more likely overfit or genuine novelty). "
            "**Σ J** — sum of intra-team pairwise couplings, i.e. the J-contribution "
            "to −raw E. Large Σ J = structurally coherent (archetype-driven); small "
            "Σ J = low-energy mainly via popular individual members (h-driven)."
            if team_counts is not None else ""
        )
        st.caption(
            "**%** — empirical fraction of post-burn-in samples producing this completion.  "
            "**raw E** — Ising energy `H(s) = -h·s - 0.5 s'Js` with the full data-calibrated "
            "field h. Independent of sampler settings; reflects the team's intrinsic "
            "likelihood under the calibrated model.  "
            "**adj E** — adjusted energy with the field rescaled by `field_weight`: "
            "`H_adj(s) = -(field_weight·h)·s - 0.5 s'Js`. This is what the sampler uses; "
            "ranking by `adj E` matches the sampled probability ordering. "
            "At `field_weight=1.0` they're identical; at `field_weight=0.0`, adj E is "
            "the pure pairwise term `-0.5 s'Js`."
            + obs_caption
        )

    elif mode == "Mean-field (deterministic)":
        with st.spinner("Computing mean-field marginals..."):
            result = meanfield_distribution(
                phase_key,
                tuple(fixed_idx), tuple(excluded_idx), field_weight,
                mf_n_iters, mf_tol,
                J, h,
                _species_of=species_of, _item_of=item_of,
            )
        if result is None:
            st.error("Not enough available Pokemon to fill the team after applying constraints.")
            return
        marginals, valid_mask, iters_used = result

        k_free = TEAM_SIZE - len(fixed_idx)
        valid_idxs = list(np.where(valid_mask)[0])
        sorted_candidates = sorted(valid_idxs, key=lambda i: -marginals[i])

        # Greedy MF-MAP completion: top candidates by marginal, enforcing
        # uniqueness incrementally as we add to the team.
        greedy_completion: list[int] = []
        greedy_species: set[str] = (
            {species_of[i] for i in fixed_idx} if species_of is not None else set()
        )
        greedy_items: set[str] = (
            {item_of[i] for i in fixed_idx if item_of[i] is not None}
            if item_of is not None else set()
        )
        for cand in sorted_candidates:
            if len(greedy_completion) == k_free:
                break
            if species_of is not None and species_of[cand] in greedy_species:
                continue
            if item_of is not None and item_of[cand] is not None and item_of[cand] in greedy_items:
                continue
            greedy_completion.append(int(cand))
            if species_of is not None:
                greedy_species.add(species_of[cand])
            if item_of is not None and item_of[cand] is not None:
                greedy_items.add(item_of[cand])

        if len(greedy_completion) < k_free:
            st.error("Could not greedy-fill the team — insufficient non-conflicting candidates.")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Free slots", f"{k_free}")
        col2.metric("Valid candidates", f"{len(valid_idxs)}")
        col3.metric("MF iterations", f"{iters_used}")

        # Greedy completion as the headline result
        greedy_state = np.zeros(len(vocab), dtype=bool)
        for i in fixed_idx:
            greedy_state[i] = True
        for i in greedy_completion:
            greedy_state[i] = True
        raw_E = team_energy(greedy_state, J, h)
        adj_E = team_energy(greedy_state, J, field_weight * h)
        names = ", ".join(vocab[i] for i in greedy_completion)

        st.markdown("**Greedy MF-MAP completion** (top free-slot candidates by marginal)")
        diag_header = " obs | Δ | Σ J |" if team_counts is not None else ""
        diag_sep = " ---: | ---: | ---: |" if team_counts is not None else ""
        if team_counts is not None:
            obs = team_obs_count(greedy_state, vocab, team_counts)
            delta = min_swaps_to_observed(greedy_state, vocab, team_counts)
            sum_j = intra_team_sum_j(greedy_state, J)
            diag_cells = f" {obs} | {delta} | {sum_j:+.3f} |"
        else:
            diag_cells = ""
        st.markdown(
            f"| raw E | adj E |{diag_header} completion |\n"
            f"| ---: | ---: |{diag_sep} :--- |\n"
            f"| {raw_E:+.3f} | {adj_E:+.3f} |{diag_cells} {names} |"
        )

        st.markdown(f"**Top-{top_k} candidates by MF marginal** (single-slot ranking)")
        md_lines = [
            "| # | MF prob | h_eff |  in greedy? | candidate |",
            "| ---: | ---: | ---: | :---: | :--- |",
        ]
        greedy_set = set(greedy_completion)
        for rank, i in enumerate(sorted_candidates[:top_k], 1):
            prob_pct = float(marginals[i]) * 100
            h_eff_i = field_weight * float(h[i])
            in_greedy = "✓" if int(i) in greedy_set else ""
            md_lines.append(
                f"| {rank} | {prob_pct:.2f}% | {h_eff_i:+.3f} | {in_greedy} | {vocab[i]} |"
            )
        st.markdown("\n".join(md_lines))

        st.caption(
            "**MF prob** — mean-field marginal P(candidate in team | fixed) under "
            "the field-weighted Ising. Free slots are treated as independent of each "
            "other given fixed mons; the joint MAP under that approximation is the "
            "greedy completion above. **h_eff** = `field_weight · h[candidate]` is "
            "the candidate's own popularity contribution (independent of fixed mons); "
            "marginal − sigmoid(h_eff) ≈ the net pull from `J @ m` (the interactions). "
            "**in greedy?** flags whether this candidate was picked into the headline "
            "completion (uniqueness can knock out high-marginal candidates that share "
            "a species/item with a greedier pick). "
            "Validated as a ranking-faithful proxy for PT at fw=1, T=1 (Spearman ρ "
            "0.93-0.95, ≤1.5 pp hit-rate delta in `validation.ipynb`); at low fw or "
            "when free slots correlate strongly through J, prefer PT."
        )

    elif mode == "Greedy team optimizer":
        starting_team_idx = sorted({name_to_idx[n] for n in starting_team_names})

        # Validate Phase 3 uniqueness of the starting team (Phase 1/2 vocabs
        # are unique-by-construction so the loop is inert there).
        if species_of is not None:
            seen_sp: dict[str, str] = {}
            for i in starting_team_idx:
                sp = species_of[i]
                if sp in seen_sp:
                    st.error(f"Starting team has two entries for species **{sp}** "
                             f"({seen_sp[sp]}, {vocab[i]}). Pick one variant.")
                    return
                seen_sp[sp] = vocab[i]
        if item_of is not None:
            seen_it: dict[str, str] = {}
            for i in starting_team_idx:
                it = item_of[i]
                if it is None:
                    continue
                if it in seen_it:
                    st.error(f"Starting team has two mons holding **{it}** "
                             f"({seen_it[it]}, {vocab[i]}). Items must be unique.")
                    return
                seen_it[it] = vocab[i]

        with st.spinner("Computing greedy descent..."):
            final_team, chain = greedy_optimize_chain(
                phase_key,
                tuple(starting_team_idx), (),
                (), field_weight, max_swaps,
                J, h,
                _species_of=species_of, _item_of=item_of,
            )

        def team_state(team_idx: list[int] | tuple[int, ...]) -> np.ndarray:
            s = np.zeros(len(vocab), dtype=bool)
            for i in team_idx:
                s[i] = True
            return s

        start_state = team_state(starting_team_idx)
        start_raw_E = team_energy(start_state, J, h)
        start_adj_E = team_energy(start_state, J, field_weight * h)
        start_sum_j = intra_team_sum_j(start_state, J)
        final_state = team_state(final_team)
        final_raw_E = team_energy(final_state, J, h)
        final_adj_E = team_energy(final_state, J, field_weight * h)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Swaps taken", f"{len(chain)} / {max_swaps}")
        col2.metric("ΔE_adj (total)", f"{final_adj_E - start_adj_E:+.3f}",
                    help="Cumulative decrease in adj E from starting to final team. "
                         "Negative = team improved under the field-weighted model.")
        col3.metric("ΔE_raw (total)", f"{final_raw_E - start_raw_E:+.3f}",
                    help="Cumulative decrease in raw (un-rescaled) E. Comparable across "
                         "different field_weight settings.")
        col4.metric("Σ J change",
                    f"{intra_team_sum_j(final_state, J) - start_sum_j:+.3f}",
                    help="Change in intra-team pairwise-coupling sum. Positive = more "
                         "structurally coherent (archetype-driven); useful at low fw.")

        if not chain:
            st.info(
                "No improving single-swap exists — your starting team is already a "
                "local minimum under the current `field_weight`. Try a different "
                "`field_weight` (e.g., 0.0 for pure pairwise structure) to see if "
                "the model would re-rank under a different objective."
            )

        # Unified chain table: starting team as row 0, then one row per swap.
        diag_header = " obs | Δ |" if team_counts is not None else ""
        diag_sep = " ---: | ---: |" if team_counts is not None else ""
        st.markdown("**Swap chain** (greedy steepest descent — each row is the team *after* that swap)")
        md_lines = [
            f"| # | swap | adj E | raw E | Σ J |{diag_header} team |",
            f"| ---: | :--- | ---: | ---: | ---: |{diag_sep} :--- |",
        ]

        def diag_cells(state: np.ndarray) -> str:
            if team_counts is None:
                return ""
            obs = team_obs_count(state, vocab, team_counts)
            delta_obs = min_swaps_to_observed(state, vocab, team_counts)
            return f" {obs} | {delta_obs} |"

        # Row 0: starting team
        md_lines.append(
            f"| 0 | _starting team_ | {start_adj_E:+.3f} | {start_raw_E:+.3f} | "
            f"{start_sum_j:+.3f} |{diag_cells(start_state)} "
            f"{', '.join(vocab[i] for i in starting_team_idx)} |"
        )

        # Rows 1..N: one per swap
        for ev in chain:
            swap_label = f"{vocab[ev['out_idx']]} → {vocab[ev['in_idx']]}"
            after_state = team_state(ev["team_after"])
            md_lines.append(
                f"| {ev['step']} | {swap_label} | "
                f"{ev['energy_adj_after']:+.3f} | {ev['energy_raw_after']:+.3f} | "
                f"{ev['sum_j_after']:+.3f} |{diag_cells(after_state)} "
                f"{', '.join(vocab[i] for i in ev['team_after'])} |"
            )

        st.markdown("\n".join(md_lines))

        obs_caption = (
            "  **obs** — exact-match count of the resulting team in the corpus. "
            "**Δ** — minimum slot-swaps to the nearest observed team (0 ↔ exact match). "
            if team_counts is not None else ""
        )
        st.caption(
            "**swap** — Pokemon swapped OUT → IN at this step. "
            "**adj E** — field-weighted energy `H_adj(s) = -(field_weight·h)·s - 0.5 s'Js` "
            "of the team *after* this swap. Greedy descent minimizes this. "
            "**raw E** — un-rescaled energy `H(s) = -h·s - 0.5 s'Js`, comparable across "
            "field_weight settings. "
            "**Σ J** — intra-team pairwise-coupling sum (the J-contribution to −raw E)."
            + obs_caption
            + " Greedy stops at the first local minimum reachable by single-swap moves; "
            "the final team is NOT necessarily the global MAP — multi-swap rearrangements "
            "may go lower. Run with `field_weight=0.0` to see what the pure pairwise "
            "model would do; run with `field_weight=1.0` to see what the popularity-driven "
            "posterior would do."
        )

    elif mode == "Anneal to MAP":
        with st.spinner(f"Annealing {n_runs} runs × {anneal_steps} steps each..."):
            results, accept_rate = run_anneals(
                phase_key,
                tuple(fixed_idx), tuple(excluded_idx), field_weight,
                n_runs, anneal_steps, t_start, t_end, J, h,
                _species_of=species_of, _item_of=item_of,
            )
        if results is None:
            st.error("Not enough available Pokemon to fill the team after applying constraints.")
            return

        n_distinct = len(results)
        top_count = results[0][1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total runs", f"{n_runs}")
        col2.metric("Distinct outcomes", f"{n_distinct}")
        col3.metric("MAP convergence", f"{top_count}/{n_runs}")
        col4.metric("MH accept %", f"{accept_rate * 100:.1f}%",
                    help="Mean fraction of swap proposals accepted across the cooling schedule, "
                         "averaged over all annealing runs. Hot phase accepts most proposals; "
                         "cold phase rejects almost all. So aggregate rate depends on schedule.")

        diag_header = " obs | Δ | Σ J |" if team_counts is not None else ""
        diag_sep = " ---: | ---: | ---: |" if team_counts is not None else ""
        md_lines = [
            f"| # | runs | raw E | adj E |{diag_header} completion |",
            f"| ---: | ---: | ---: | ---: |{diag_sep} :--- |",
        ]
        for rank, (comp, count) in enumerate(results, 1):
            state = np.zeros(len(vocab), dtype=bool)
            for i in fixed_idx:
                state[i] = True
            for i in comp:
                state[i] = True
            raw_E = team_energy(state, J, h)
            adj_E = team_energy(state, J, field_weight * h)
            names = ", ".join(vocab[i] for i in comp)
            if team_counts is not None:
                obs = team_obs_count(state, vocab, team_counts)
                delta = min_swaps_to_observed(state, vocab, team_counts)
                sum_j = intra_team_sum_j(state, J)
                diag_cells = f" {obs} | {delta} | {sum_j:+.3f} |"
            else:
                diag_cells = ""
            md_lines.append(
                f"| {rank} | {count}/{n_runs} | {raw_E:+.3f} | {adj_E:+.3f} |{diag_cells} {names} |"
            )

        st.markdown("\n".join(md_lines))
        obs_caption = (
            "  **obs** — exact-match count in the corpus. "
            "**Δ** — minimum slot-swaps to the nearest observed team (0 ↔ exact "
            "match). High obs *or* low Δ = the MAP basin is near realized teams. "
            "**Σ J** — sum of intra-team pairwise couplings (J-contribution to −raw E). "
            "Large Σ J = coherent archetype; small Σ J = the basin is held together "
            "by popular individual mons rather than pairwise structure."
            if team_counts is not None else ""
        )
        st.caption(
            "**runs** — how many independent annealings converged to this team. "
            "A single dominant team indicates a sharp / unimodal landscape; multiple "
            "distinct teams indicate a shallow / multimodal one. "
            "**raw E** is the data-calibrated energy `-h·s - 0.5 s'Js`; "
            "**adj E** is `-(field_weight·h)·s - 0.5 s'Js`, which is what the annealer "
            "actually minimizes. Lower adj E = the energy basin the annealer settled into."
            + obs_caption
        )

    else:  # Parallel-tempered sample
        with st.spinner(f"PT: {pt_n_runs} runs × {pt_K} chains × {pt_n_steps} sweeps..."):
            dist, n_kept, mh_rate, swap_rate, ladder = parallel_tempered_distribution(
                phase_key,
                tuple(fixed_idx), tuple(excluded_idx), field_weight,
                pt_t_min, pt_t_max, pt_K, pt_n_runs, pt_n_steps,
                pt_burn_in, pt_swap_interval, J, h,
                _species_of=species_of, _item_of=item_of,
            )
        if dist is None:
            st.error("Not enough available Pokemon to fill the team after applying constraints.")
            return

        n_distinct = len(dist)
        top5_mass = sum(c for _, c in dist[:5]) / n_kept * 100
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Cold samples", f"{n_kept:,}")
        col2.metric("Distinct completions", f"{n_distinct:,}")
        col3.metric("Top-5 mass", f"{top5_mass:.2f}%",
                    help="Sum of probabilities for the five most-frequent completions. "
                         "Under PT this is a real Boltzmann statistic (basin mixing solved).")
        col4.metric("Local accept %", f"{mh_rate * 100:.1f}%",
                    help="Mean fraction of local swap-MH proposals accepted, averaged across "
                         "chains and runs. Low at cold chain, high at hot chain — this is the "
                         "averaged rate across the whole ladder.")
        col5.metric("Replica swap %", f"{swap_rate * 100:.1f}%",
                    help="Mean acceptance rate of replica-exchange swaps between adjacent T "
                         "levels. Healthy range 20-50%. <10% means ladder is too sparse (raise K "
                         "or lower t_max). >60% means ladder is overly dense (waste of chains).")

        st.caption(f"Temperature ladder: " + " → ".join(f"{t:.3f}" for t in ladder))

        diag_header = " obs | Δ | Σ J |" if team_counts is not None else ""
        diag_sep = " ---: | ---: | ---: |" if team_counts is not None else ""
        md_lines = [
            f"| # | % | raw E | adj E |{diag_header} completion |",
            f"| ---: | ---: | ---: | ---: |{diag_sep} :--- |",
        ]
        for rank, (comp, count) in enumerate(dist[:top_k], 1):
            state = np.zeros(len(vocab), dtype=bool)
            for i in fixed_idx:
                state[i] = True
            for i in comp:
                state[i] = True
            prob_pct = (count / n_kept) * 100
            raw_E = team_energy(state, J, h)
            adj_E = team_energy(state, J, field_weight * h)
            names = ", ".join(vocab[i] for i in comp)
            if team_counts is not None:
                obs = team_obs_count(state, vocab, team_counts)
                delta = min_swaps_to_observed(state, vocab, team_counts)
                sum_j = intra_team_sum_j(state, J)
                diag_cells = f" {obs} | {delta} | {sum_j:+.3f} |"
            else:
                diag_cells = ""
            md_lines.append(
                f"| {rank} | {prob_pct:.2f}% | {raw_E:+.3f} | {adj_E:+.3f} |{diag_cells} {names} |"
            )

        st.markdown("\n".join(md_lines))
        obs_caption = (
            "  **obs** — exact-match count in the corpus. "
            "**Δ** — minimum slot-swaps to nearest observed team. Δ=0 ↔ obs≥1; "
            "Δ=1 = 1-swap variant of a real team (defensible \"fine-tuning of meta\"); "
            "Δ≥3 = globally distinct configuration. "
            "**Σ J** — sum of intra-team pairwise couplings. Sweeping `field_weight` "
            "from 1.0 → 0.0 makes top-K reorder by increasing Σ J: lower fw selects "
            "for coherence, higher fw selects for popular individual members."
            if team_counts is not None else ""
        )
        st.caption(
            "**%** — Boltzmann probability at the target T, estimated from cold-chain samples. "
            "Unlike single-chain sampling at low T, this **should be monotonically ordered with "
            "adj E** (modulo Monte Carlo noise) — that's the whole point of parallel tempering. "
            "If you see non-monotonicity, either chains haven't equilibrated (run more sweeps) "
            "or the swap rate is too low (raise K or lower t_max). "
            "**raw E** / **adj E** are the energies with full / rescaled h, same definitions as "
            "the other modes."
            + obs_caption
        )


if __name__ == "__main__":
    main()
