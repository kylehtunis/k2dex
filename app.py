"""k2dex science — Streamlit webapp around the inverse-Ising team model.

Three pages, all sharing one phase picker at the top:

- **Team completer** — sample completions from the conditional Ising posterior
  (Mean-field, Parallel-tempered, Sample, Anneal, Greedy descent).
- **Team analysis** — per-team observables, pairwise J decomposition,
  J-row partner inspector, greedy swap-chain critique.
- **Meta data** — fitted-model summary, top features by h, top ±J pairs,
  J / h distribution plots.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

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
from models import fit_pl_ising
from rendering import (
    intra_team_sum_j,
    min_swaps_to_observed,
    pairwise_j_rows,
    render_j_row_inspector,
    render_pairwise_j_table,
    team_obs_count,
)
from sampling import (
    anneal_mcmc,
    greedy_optimize,
    meanfield_marginals,
    parallel_tempered_mcmc,
    swap_mcmc,
    team_energy,
)


@dataclass
class PhaseModel:
    """Resolved bundle of everything a page needs to render under one phase.

    `species_of` and `item_of` are None for Phase 1/2 (no uniqueness constraint)
    and populated for Phase 3 (where the sampler must reject duplicate-species
    and duplicate-item swap proposals).
    """
    vocab: list[str]
    m: np.ndarray
    J: np.ndarray
    h: np.ndarray
    team_counts: Counter | None
    species_of: list[str] | None
    item_of: list[str | None] | None


def _load_phase(phase: str) -> tuple[str, PhaseModel]:
    if phase.startswith("Phase 1"):
        return "phase1", PhaseModel(*load_model_phase1())
    if phase.startswith("Phase 2"):
        return "phase2", PhaseModel(*load_model_phase2())
    return "phase3", PhaseModel(*load_model_phase3())


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

    J, h = fit_pl_ising(X, C=PHASE2_LR_C)
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

    J, h = fit_pl_ising(X, C=PHASE2_LR_C)
    return vocab, m, J, h, team_counts, species_of, item_of


def format_pair(species: str, item: str | None) -> str:
    """Display-friendly form of a (species, item) pair used as Phase 3 vocab
    strings. Bare species for itemless mons; otherwise 'Species @ Item'."""
    if item is None:
        return species
    return f"{species} @ {item}"


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
    but cached for consistency with the MCMC techniques (and so the same
    fixed/field_weight combo hits the cache on re-render)."""
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
    st.set_page_config(page_title="k2dex · science", layout="wide")

    # Persistent header: wordmark + phase picker, shown above the tab bar.
    # Streamlit re-runs the script on every interaction so the radio's value
    # is naturally page-global without needing st.session_state plumbing.
    header_col1, header_col2 = st.columns([3, 2])
    with header_col1:
        st.markdown("### k2dex · science")
    with header_col2:
        phase = st.radio(
            "Phase",
            ["Phase 1 (Gaussian)", "Phase 2 (PL, species)", "Phase 3 (PL, item-pair)"],
            horizontal=True,
            label_visibility="collapsed",
            key="phase_picker",
            help=(
                "**Phase 1** — Gaussian / precision-matrix Ising fit on Smogon "
                "chaos stats.  "
                "**Phase 2** — pseudo-likelihood fit on Limitless team rosters, "
                "species-only vocab.  "
                "**Phase 3** — pseudo-likelihood fit on (species, item) pairs."
            ),
        )

    phase_key, model = _load_phase(phase)

    completer_tab, analysis_tab, meta_tab = st.tabs(
        ["Team completer", "Team analysis", "Meta data"],
    )
    with completer_tab:
        _render_completer(phase_key, model)
    with analysis_tab:
        _render_analysis(phase_key, model)
    with meta_tab:
        _render_meta(phase_key, model)


def _render_completer(phase_key: str, model: PhaseModel) -> None:
    """Team completer — sample completions from the conditional Ising posterior.

    Five techniques share the same constraint widgets and field-weight slider:
    Mean-field (deterministic), Parallel-tempered, Sample distribution,
    Anneal → MAP, Greedy descent. Each renders its own headline metrics +
    completion table.
    """
    st.subheader("Team completer")
    st.caption("Sample team completions from the conditional Ising posterior.")

    vocab = model.vocab
    J, h = model.J, model.h
    species_of, item_of = model.species_of, model.item_of
    team_counts = model.team_counts

    name_to_idx = {name: i for i, name in enumerate(vocab)}
    sorted_vocab = sorted(vocab, key=lambda v: -model.m[name_to_idx[v]])

    constraint_col1, constraint_col2 = st.columns(2)
    with constraint_col1:
        fixed_names = st.multiselect(
            "Pin (must appear)",
            sorted_vocab,
            max_selections=TEAM_SIZE,
            placeholder="Choose Pokemon to pin at s=1",
            key=f"completer_fixed_{phase_key}",
        )
    with constraint_col2:
        excluded_names = st.multiselect(
            "Exclude (must NOT appear)",
            sorted_vocab,
            placeholder="Choose Pokemon to pin at s=0",
            key=f"completer_excluded_{phase_key}",
        )

    overlap = set(fixed_names) & set(excluded_names)
    if overlap:
        st.error(f"Cannot be both pinned and excluded: {', '.join(overlap)}")
        return

    sampler_col, technique_col = st.columns([2, 1])
    with sampler_col:
        st.markdown("##### Sampler")
        field_weight = st.select_slider(
            "Field weight (h scale)",
            options=FIELD_WEIGHT_OPTIONS,
            value=0.5,
            key=f"completer_fw_{phase_key}",
            help=(
                "Scales h before sampling. 1.0 = data-calibrated posterior "
                "(popular mons dominate). 0.0 = pure pairwise (archetype-coherent, "
                "no popularity prior). Useful operating range 0.3–0.6 with "
                "current regularization."
            ),
        )
        temperature = st.select_slider(
            "Temperature",
            options=TEMPERATURE_OPTIONS,
            value=0.05,
            key=f"completer_temp_{phase_key}",
            help=(
                "Sampling temperature. Lower = sharper Boltzmann distribution. "
                "Used by Sample / Anneal / PT; ignored by Mean-field and Greedy."
            ),
        )
    with technique_col:
        st.markdown("##### Technique")
        technique = st.radio(
            "Technique",
            [
                "Mean-field",
                "Parallel-tempered",
                "Sample distribution",
                "Anneal → MAP",
                "Greedy descent",
            ],
            index=0,
            label_visibility="collapsed",
            key=f"completer_technique_{phase_key}",
            help=(
                "**Mean-field** — deterministic per-candidate marginals via "
                "damped MF iteration. Instant; ranking-faithful proxy for PT "
                "at fw=1, T=1.  "
                "**Parallel-tempered** — replica-exchange MCMC. True Boltzmann "
                "draws at the cold-chain target T.  "
                "**Sample distribution** — independent-chain MCMC. Fast but "
                "at low T each chain gets stuck in one basin.  "
                "**Anneal → MAP** — multiple cooling-schedule runs, returns "
                "the MAP each run converged to.  "
                "**Greedy descent** — greedy-fill free slots respecting "
                "uniqueness, then steepest-descent with pinned mons frozen."
            ),
        )

    with st.expander("Run options", expanded=False):
        if technique == "Sample distribution":
            n_chains = st.slider("Chains", 5, 50, 20,
                                 key=f"completer_nchains_{phase_key}")
            n_steps = st.slider("Steps per chain", 2000, 20000, 8000, step=1000,
                                key=f"completer_nsteps_{phase_key}")
            burn_in = st.slider("Burn-in", 0, 5000, 2000, step=500,
                                key=f"completer_burnin_{phase_key}")
        elif technique == "Anneal → MAP":
            n_runs = st.slider("Independent anneal runs", 5, 50, 20,
                               key=f"completer_anneal_runs_{phase_key}")
            anneal_steps = st.slider("Steps per run", 5000, 50000, 20000, step=5000,
                                     key=f"completer_anneal_steps_{phase_key}")
            t_start = st.select_slider("Start T", options=ANNEAL_T_START_OPTIONS, value=3.0,
                                       key=f"completer_anneal_tstart_{phase_key}")
            t_end = st.select_slider("End T", options=ANNEAL_T_END_OPTIONS, value=0.02,
                                     key=f"completer_anneal_tend_{phase_key}")
        elif technique == "Parallel-tempered":
            pt_t_min = st.select_slider("Cold (target) T", options=PT_T_MIN_OPTIONS, value=0.1,
                                        key=f"completer_pt_tmin_{phase_key}")
            pt_t_max = st.select_slider("Hot T", options=PT_T_MAX_OPTIONS, value=2.0,
                                        key=f"completer_pt_tmax_{phase_key}")
            pt_K = st.slider("Ladder levels (K)", 3, 12, 7,
                             key=f"completer_pt_K_{phase_key}")
            pt_n_runs = st.slider("Independent PT runs", 1, 10, 3,
                                  key=f"completer_pt_nruns_{phase_key}")
            pt_n_steps = st.slider("Sweeps per run", 2000, 30000, 10000, step=1000,
                                   key=f"completer_pt_nsteps_{phase_key}")
            pt_burn_in = st.slider("Burn-in", 0, 10000, 3000, step=500,
                                   key=f"completer_pt_burnin_{phase_key}")
            pt_swap_interval = st.slider("Swap proposal interval", 1, 50, 10,
                                         key=f"completer_pt_swap_{phase_key}")
        elif technique == "Mean-field":
            mf_n_iters = st.slider("Max iterations", 50, 500, 200, step=50,
                                   key=f"completer_mf_iters_{phase_key}",
                                   help="Cap on damped fixed-point iterations.")
            mf_tol_log = st.select_slider("Convergence tolerance (log10)",
                                          options=[-3, -4, -5, -6], value=-5,
                                          key=f"completer_mf_tol_{phase_key}")
            mf_tol = 10.0 ** mf_tol_log
        else:  # Greedy descent
            max_swaps = st.slider("Max swaps", 1, 30, 15,
                                  key=f"completer_greedy_swaps_{phase_key}",
                                  help="Hard cap on swap chain length.")
        top_k = st.slider("Top-K shown", 5, 50, 20, key=f"completer_topk_{phase_key}")

    run_label = {
        "Sample distribution": "Sample teams",
        "Anneal → MAP": "Anneal teams",
        "Parallel-tempered": "PT sample",
        "Mean-field": "Compute MF marginals",
        "Greedy descent": "Greedy completion",
    }[technique]
    run = st.button(run_label, type="primary", use_container_width=True,
                    key=f"completer_run_{phase_key}")
    if not run:
        return

    fixed_idx = sorted({name_to_idx[n] for n in fixed_names})
    excluded_idx = sorted({name_to_idx[n] for n in excluded_names})

    if technique == "Sample distribution":
        with st.spinner(f"Running {n_chains} chains × {n_steps} steps..."):
            dist, n_kept, accept_rate = sample_distribution(
                phase_key, tuple(fixed_idx), tuple(excluded_idx),
                field_weight, temperature, n_chains, n_steps, burn_in, J, h,
                _species_of=species_of, _item_of=item_of,
            )
        if dist is None:
            st.error("Not enough available Pokemon to fill the team after applying constraints.")
            return
        n_distinct = len(dist)
        top5_mass = sum(c for _, c in dist[:5]) / n_kept * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total samples", f"{n_kept:,}")
        c2.metric("Distinct completions", f"{n_distinct:,}")
        c3.metric("Top-5 mass", f"{top5_mass:.2f}%")
        c4.metric("MH accept %", f"{accept_rate * 100:.1f}%")
        _render_completion_table(
            dist[:top_k], n_kept, fixed_idx, vocab, J, h, field_weight,
            team_counts, count_col="%",
        )

    elif technique == "Anneal → MAP":
        with st.spinner(f"Annealing {n_runs} runs × {anneal_steps} steps each..."):
            results, accept_rate = run_anneals(
                phase_key, tuple(fixed_idx), tuple(excluded_idx), field_weight,
                n_runs, anneal_steps, t_start, t_end, J, h,
                _species_of=species_of, _item_of=item_of,
            )
        if results is None:
            st.error("Not enough available Pokemon to fill the team after applying constraints.")
            return
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total runs", f"{n_runs}")
        c2.metric("Distinct outcomes", f"{len(results)}")
        c3.metric("MAP convergence", f"{results[0][1]}/{n_runs}")
        c4.metric("MH accept %", f"{accept_rate * 100:.1f}%")
        _render_completion_table(
            results, n_runs, fixed_idx, vocab, J, h, field_weight,
            team_counts, count_col="runs",
        )

    elif technique == "Parallel-tempered":
        with st.spinner(f"PT: {pt_n_runs} runs × {pt_K} chains × {pt_n_steps} sweeps..."):
            dist, n_kept, mh_rate, swap_rate, ladder = parallel_tempered_distribution(
                phase_key, tuple(fixed_idx), tuple(excluded_idx), field_weight,
                pt_t_min, pt_t_max, pt_K, pt_n_runs, pt_n_steps,
                pt_burn_in, pt_swap_interval, J, h,
                _species_of=species_of, _item_of=item_of,
            )
        if dist is None:
            st.error("Not enough available Pokemon to fill the team after applying constraints.")
            return
        top5_mass = sum(c for _, c in dist[:5]) / n_kept * 100
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Cold samples", f"{n_kept:,}")
        c2.metric("Distinct completions", f"{len(dist):,}")
        c3.metric("Top-5 mass", f"{top5_mass:.2f}%")
        c4.metric("Local accept %", f"{mh_rate * 100:.1f}%")
        c5.metric("Replica swap %", f"{swap_rate * 100:.1f}%")
        st.caption("Temperature ladder: " + " → ".join(f"{t:.3f}" for t in ladder))
        _render_completion_table(
            dist[:top_k], n_kept, fixed_idx, vocab, J, h, field_weight,
            team_counts, count_col="%",
        )

    elif technique == "Mean-field":
        with st.spinner("Computing mean-field marginals..."):
            result = meanfield_distribution(
                phase_key, tuple(fixed_idx), tuple(excluded_idx), field_weight,
                mf_n_iters, mf_tol, J, h,
                _species_of=species_of, _item_of=item_of,
            )
        if result is None:
            st.error("Not enough available Pokemon to fill the team after applying constraints.")
            return
        marginals, valid_mask, iters_used = result
        k_free = TEAM_SIZE - len(fixed_idx)
        valid_idxs = list(np.where(valid_mask)[0])
        sorted_candidates = sorted(valid_idxs, key=lambda i: -marginals[i])

        # Greedy MF-MAP completion: top candidates by marginal, respecting
        # Phase 3 uniqueness incrementally as we accumulate the team.
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

        c1, c2, c3 = st.columns(3)
        c1.metric("Free slots", f"{k_free}")
        c2.metric("Valid candidates", f"{len(valid_idxs)}")
        c3.metric("MF iterations", f"{iters_used}")

        st.markdown("**Greedy MF-MAP completion** (top free-slot candidates by marginal)")
        _render_completion_table(
            [(tuple(greedy_completion), 1)], 1, fixed_idx, vocab, J, h, field_weight,
            team_counts, count_col=None,
        )

        st.markdown(f"**Top-{top_k} candidates by MF marginal** (single-slot ranking)")
        greedy_set = set(greedy_completion)
        md = [
            "| # | MF prob | h_eff | in greedy? | candidate |",
            "| ---: | ---: | ---: | :---: | :--- |",
        ]
        for rank, i in enumerate(sorted_candidates[:top_k], 1):
            prob_pct = float(marginals[i]) * 100
            h_eff_i = field_weight * float(h[i])
            in_greedy = "✓" if int(i) in greedy_set else ""
            md.append(f"| {rank} | {prob_pct:.2f}% | {h_eff_i:+.3f} | {in_greedy} | {vocab[i]} |")
        st.markdown("\n".join(md))

    else:  # Greedy descent — greedy-fill, then descend with pinned frozen
        k_free = TEAM_SIZE - len(fixed_idx)
        if k_free < 0:
            st.error(f"Pinned more than {TEAM_SIZE} mons.")
            return
        excluded_set = set(excluded_idx)
        used_sp = (
            {species_of[i] for i in fixed_idx} if species_of is not None else set()
        )
        used_it = (
            {item_of[i] for i in fixed_idx if item_of[i] is not None}
            if item_of is not None else set()
        )
        order_by_m = sorted(range(len(vocab)), key=lambda i: -model.m[i])
        init_free: list[int] = []
        for cand in order_by_m:
            if len(init_free) == k_free:
                break
            if cand in excluded_set or cand in set(fixed_idx):
                continue
            if species_of is not None and species_of[cand] in used_sp:
                continue
            if item_of is not None and item_of[cand] is not None and item_of[cand] in used_it:
                continue
            init_free.append(cand)
            if species_of is not None:
                used_sp.add(species_of[cand])
            if item_of is not None and item_of[cand] is not None:
                used_it.add(item_of[cand])
        if len(init_free) < k_free:
            st.error("Could not greedy-fill the team — insufficient non-conflicting candidates.")
            return
        start_team = sorted(fixed_idx + init_free)
        with st.spinner("Greedy descent..."):
            final_team, chain = greedy_optimize_chain(
                phase_key,
                tuple(start_team), tuple(fixed_idx), tuple(excluded_idx),
                field_weight, max_swaps, J, h,
                _species_of=species_of, _item_of=item_of,
            )
        c1, c2 = st.columns(2)
        c1.metric("Swaps taken", f"{len(chain)} / {max_swaps}")
        final_state = np.zeros(len(vocab), dtype=bool)
        for i in final_team:
            final_state[i] = True
        c2.metric("Σ J (final)", f"{intra_team_sum_j(final_state, J):+.3f}")
        if not chain:
            st.info(
                "No improving swap from the greedy-fill initialization — already a "
                "local minimum at this field_weight."
            )
        free_final = [i for i in final_team if i not in set(fixed_idx)]
        _render_completion_table(
            [(tuple(free_final), 1)], 1, fixed_idx, vocab, J, h, field_weight,
            team_counts, count_col=None,
        )


def _render_completion_table(
    rows: list[tuple[tuple[int, ...], int]],
    total: int,
    fixed_idx: list[int],
    vocab: list[str],
    J: np.ndarray,
    h: np.ndarray,
    field_weight: float,
    team_counts: Counter | None,
    *,
    count_col: str | None,
) -> None:
    """Render the standard completion table used by every technique.

    `count_col` is the header label for the count column ("%", "runs"), or
    None to omit it (single-row tables: MF-MAP, greedy descent).
    """
    diag_header = " obs | Δ | Σ J |" if team_counts is not None else ""
    diag_sep = " ---: | ---: | ---: |" if team_counts is not None else ""
    if count_col is not None:
        head = f"| # | {count_col} | raw E | adj E |{diag_header} completion |"
        sep = f"| ---: | ---: | ---: | ---: |{diag_sep} :--- |"
    else:
        head = f"| raw E | adj E |{diag_header} completion |"
        sep = f"| ---: | ---: |{diag_sep} :--- |"
    lines = [head, sep]
    for rank, (comp, count) in enumerate(rows, 1):
        state = np.zeros(len(vocab), dtype=bool)
        for i in fixed_idx:
            state[i] = True
        for i in comp:
            state[i] = True
        raw_E = team_energy(state, J, h)
        adj_E = team_energy(state, J, field_weight * h)
        if team_counts is not None:
            diag = (
                f" {team_obs_count(state, vocab, team_counts)} | "
                f"{min_swaps_to_observed(state, vocab, team_counts)} | "
                f"{intra_team_sum_j(state, J):+.3f} |"
            )
        else:
            diag = ""
        names = ", ".join(vocab[i] for i in comp)
        if count_col == "%":
            lines.append(f"| {rank} | {(count / total) * 100:.2f}% | {raw_E:+.3f} | {adj_E:+.3f} |{diag} {names} |")
        elif count_col == "runs":
            lines.append(f"| {rank} | {count}/{total} | {raw_E:+.3f} | {adj_E:+.3f} |{diag} {names} |")
        else:
            lines.append(f"| {raw_E:+.3f} | {adj_E:+.3f} |{diag} {names} |")
    st.markdown("\n".join(lines))


def _render_analysis(phase_key: str, model: PhaseModel) -> None:
    """Per-team analysis under the fitted (J, h): observables, pairwise J
    decomposition, J-row partner inspector, and greedy swap-chain critique.
    The team must be exactly 6 mons; no slot-filling or sampling.
    """
    st.subheader("Team analysis")
    st.caption("Per-team observables under the fitted (J, h).")

    vocab = model.vocab
    J, h = model.J, model.h
    species_of, item_of = model.species_of, model.item_of
    team_counts = model.team_counts

    name_to_idx = {n: i for i, n in enumerate(vocab)}
    sorted_vocab = sorted(vocab, key=lambda v: -model.m[name_to_idx[v]])

    team_names = st.multiselect(
        f"Your team (exactly {TEAM_SIZE})",
        sorted_vocab,
        max_selections=TEAM_SIZE,
        placeholder="Choose your team of 6",
        key=f"analysis_team_{phase_key}",
    )
    if len(team_names) != TEAM_SIZE:
        st.info(f"Pick {TEAM_SIZE} Pokemon to analyze (have {len(team_names)}).")
        return

    team_idx = sorted({name_to_idx[n] for n in team_names})

    # Phase 3 uniqueness check (inert under Phase 1/2 vocabs).
    if species_of is not None:
        seen_sp: dict[str, str] = {}
        for i in team_idx:
            sp = species_of[i]
            if sp in seen_sp:
                st.error(f"Two entries for species **{sp}** ({seen_sp[sp]}, {vocab[i]}). "
                         "Pick one variant.")
                return
            seen_sp[sp] = vocab[i]
    if item_of is not None:
        seen_it: dict[str, str] = {}
        for i in team_idx:
            it = item_of[i]
            if it is None:
                continue
            if it in seen_it:
                st.error(f"Two mons holding **{it}** ({seen_it[it]}, {vocab[i]}). "
                         "Items must be unique.")
                return
            seen_it[it] = vocab[i]

    state = np.zeros(len(vocab), dtype=bool)
    state[team_idx] = True

    field_weight = st.select_slider(
        "Field weight (for E_adj and greedy critique)",
        options=FIELD_WEIGHT_OPTIONS,
        value=0.5,
        key=f"analysis_fw_{phase_key}",
        help=(
            "Rescales the field h before computing E_adj and choosing greedy "
            "swaps. E_raw always uses fw=1. Lower fw weights pairwise structure "
            "more heavily relative to popularity."
        ),
    )

    # Observables strip
    raw_E = team_energy(state, J, h)
    adj_E = team_energy(state, J, field_weight * h)
    sum_j = intra_team_sum_j(state, J)
    cols = st.columns(5)
    cols[0].metric("E_adj", f"{adj_E:+.3f}", help=f"fw = {field_weight}")
    cols[1].metric("E_raw", f"{raw_E:+.3f}", help="fw = 1.0")
    cols[2].metric("Σ J", f"{sum_j:+.3f}",
                   help="Intra-team pair sum (J-contribution to −E).")
    if team_counts is not None:
        obs = team_obs_count(state, vocab, team_counts)
        delta = min_swaps_to_observed(state, vocab, team_counts)
        cols[3].metric("Corpus obs", f"{obs}",
                       help="Exact-roster appearances in the ingested corpus.")
        cols[4].metric("Δ to obs", f"{delta}",
                       help="Min single-swap distance to the nearest observed team.")
    else:
        cols[3].metric("Corpus obs", "—",
                       help="Phase 1 has no per-team counts (Smogon aggregates).")
        cols[4].metric("Δ to obs", "—")

    # Pairwise J decomposition
    st.markdown("##### Pairwise J decomposition")
    rows = pairwise_j_rows(team_idx, vocab, J)
    st.markdown(render_pairwise_j_table(rows))
    st.caption(
        "C(6, 2) = 15 unordered pairs. **% of |J| sum** shows which pair drives "
        "the team's coherence (or anti-coherence): one row at ≥30% means a "
        "single pair is doing most of the structural work; a flat 6-8% across "
        "pairs is a diffuse, balanced team."
    )

    # J-row inspector — partner couplings for one selected team member
    st.markdown("##### J-row · partners")
    selected_mon = st.selectbox(
        "Inspect partner couplings for",
        team_names,
        key=f"analysis_jrow_{phase_key}",
    )
    sel_idx = name_to_idx[selected_mon]
    st.markdown(render_j_row_inspector(sel_idx, set(team_idx), vocab, J))
    st.caption(
        "Top |J| partners for the selected mon, across the whole vocab — rows "
        "with ✓ are already on the team (and appear above); the others are "
        "off-team alternatives the model rates as strong (positive J) or "
        "incompatible (negative J) with this mon."
    )

    # Greedy swap-chain critique (no pinning — full team is fair game)
    st.markdown("##### Greedy critique · single-swap chain")
    max_swaps = st.slider(
        "Max swaps", 1, 30, 15,
        key=f"analysis_max_swaps_{phase_key}",
        help="Hard cap on swap chain length. Greedy descent usually converges "
             "in 3-10 swaps; this is a safety bound.",
    )
    with st.spinner("Computing greedy descent..."):
        final_team, chain = greedy_optimize_chain(
            phase_key,
            tuple(team_idx), (),           # no pinning — full team is fair game
            (), field_weight, max_swaps,
            J, h,
            _species_of=species_of, _item_of=item_of,
        )

    def team_state(idx_iter) -> np.ndarray:
        s = np.zeros(len(vocab), dtype=bool)
        for i in idx_iter:
            s[i] = True
        return s

    start_state = team_state(team_idx)
    start_raw_E = team_energy(start_state, J, h)
    start_adj_E = team_energy(start_state, J, field_weight * h)
    start_sum_j = intra_team_sum_j(start_state, J)
    final_state = team_state(final_team)
    final_raw_E = team_energy(final_state, J, h)
    final_adj_E = team_energy(final_state, J, field_weight * h)
    final_sum_j = intra_team_sum_j(final_state, J)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Swaps taken", f"{len(chain)} / {max_swaps}")
    sc2.metric("ΔE_adj", f"{final_adj_E - start_adj_E:+.3f}")
    sc3.metric("ΔE_raw", f"{final_raw_E - start_raw_E:+.3f}")
    sc4.metric("ΔΣ J", f"{final_sum_j - start_sum_j:+.3f}")

    if not chain:
        st.info(
            "No improving single-swap exists — this team is a local minimum "
            "under the current field_weight. Try a different field_weight "
            "(e.g. 0.0 for pure pairwise) to see if the model re-ranks under "
            "a different objective."
        )
        return

    diag_header = " obs | Δ |" if team_counts is not None else ""
    diag_sep = " ---: | ---: |" if team_counts is not None else ""

    def diag_cells(state: np.ndarray) -> str:
        if team_counts is None:
            return ""
        return (
            f" {team_obs_count(state, vocab, team_counts)} | "
            f"{min_swaps_to_observed(state, vocab, team_counts)} |"
        )

    md = [
        f"| # | swap | adj E | raw E | Σ J |{diag_header} team |",
        f"| ---: | :--- | ---: | ---: | ---: |{diag_sep} :--- |",
        f"| 0 | _starting team_ | {start_adj_E:+.3f} | {start_raw_E:+.3f} | "
        f"{start_sum_j:+.3f} |{diag_cells(start_state)} "
        f"{', '.join(vocab[i] for i in team_idx)} |",
    ]
    for ev in chain:
        swap_label = f"{vocab[ev['out_idx']]} → {vocab[ev['in_idx']]}"
        after_state = team_state(ev["team_after"])
        md.append(
            f"| {ev['step']} | {swap_label} | "
            f"{ev['energy_adj_after']:+.3f} | {ev['energy_raw_after']:+.3f} | "
            f"{ev['sum_j_after']:+.3f} |{diag_cells(after_state)} "
            f"{', '.join(vocab[i] for i in ev['team_after'])} |"
        )
    st.markdown("\n".join(md))
    st.caption(
        "Each row is the team **after** that swap. The chain is diagnostic: "
        "swap #1 is the model's biggest complaint about the starting team, "
        "swap #2 is the next-biggest after #1 is applied, etc. The final team "
        "is the local minimum reachable by single-swap moves — not "
        "necessarily the global MAP."
    )


def _render_meta(phase_key: str, model: PhaseModel) -> None:
    """Format-wide statistics derived from the fitted (J, h): summary strip,
    top features by h, top ±J pairs (synergies / exclusions), distribution
    plots for J and h. Cross-phase comparison and the J-graph figure are
    deferred to v1.1 (both want real React work to land cleanly).
    """
    st.subheader("Meta data")
    st.caption("Format-wide statistics derived from the fitted (J, h).")

    vocab = model.vocab
    J, h = model.J, model.h
    team_counts = model.team_counts
    V = len(vocab)

    # Summary strip
    n_corpus_teams = sum(team_counts.values()) if team_counts is not None else None
    fit_label = (
        "sklearn LogReg · L2 · C=0.1" if phase_key != "phase1"
        else "Gaussian · ridge ε=0.01"
    )
    summary_cols = st.columns(6)
    summary_cols[0].metric(
        "Phase",
        {"phase1": "Phase 1", "phase2": "Phase 2", "phase3": "Phase 3"}[phase_key],
    )
    summary_cols[1].metric("Vocab size", f"{V:,}")
    summary_cols[2].metric(
        "Corpus teams",
        f"{n_corpus_teams:,}" if n_corpus_teams is not None else "—",
        help="Phase 1 uses aggregate Smogon chaos stats (no per-team rosters).",
    )
    headline = _load_baseline_headline(phase_key)
    if headline is not None:
        top1, mrr = headline
        summary_cols[3].metric("Top-1 @ k=1", f"{top1:.1%}",
                               help="Held-out leave-1-out accuracy from validation.ipynb.")
        summary_cols[4].metric("MRR @ k=1", f"{mrr:.3f}")
    else:
        summary_cols[3].metric("Top-1 @ k=1", "—",
                               help="No locked validation baseline for this phase.")
        summary_cols[4].metric("MRR @ k=1", "—")
    summary_cols[5].metric("Fit", fit_label)

    # Top features by h
    st.markdown("##### Top features by h")
    n_show = st.slider(
        "Show top N features", 10, min(150, V), 30,
        key=f"meta_topn_{phase_key}",
        help="Sorted by h (log-odds of inclusion in a random team under the model).",
    )
    order = np.argsort(-h)
    feature_rows = [
        {
            "rank": rank,
            "feature": vocab[i],
            "h": float(h[i]),
            "m̂": float(model.m[i]),
        }
        for rank, i in enumerate(order[:n_show], 1)
    ]
    st.dataframe(feature_rows, hide_index=True, use_container_width=True)
    st.caption(
        "**h** is the per-feature log-odds — `m̂ ≈ sigmoid(h + J·m)` under MF, "
        "so for popular features h ≈ logit(m̂). The order here is the model's "
        "popularity ranking with the pairwise correction folded in."
    )

    # Top ±J pairs
    st.markdown("##### Extreme couplings")
    n_pairs = st.slider(
        "Show top N pairs (each direction)", 10, 50, 25,
        key=f"meta_npairs_{phase_key}",
        help="Top +J pairs are synergies; top −J pairs are exclusions / "
             "mutually-exclusive picks.",
    )
    iu, ju = np.triu_indices(V, k=1)
    j_flat = J[iu, ju]
    pos_col, neg_col = st.columns(2)
    with pos_col:
        st.markdown("**Top +J (synergies)**")
        pos_order = np.argsort(-j_flat)[:n_pairs]
        st.markdown(_format_extreme_pairs(iu, ju, j_flat, pos_order, vocab))
    with neg_col:
        st.markdown("**Top −J (exclusions)**")
        neg_order = np.argsort(j_flat)[:n_pairs]
        st.markdown(_format_extreme_pairs(iu, ju, j_flat, neg_order, vocab))

    # Distribution plots
    st.markdown("##### Distributional diagnostics")
    fig, (ax_j, ax_h) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax_j.hist(j_flat, bins=80, log=True, color="#3a7d44", edgecolor="white", linewidth=0.4)
    ax_j.set_title("J off-diagonal distribution")
    ax_j.set_xlabel("J")
    ax_j.set_ylabel("count (log)")
    ax_j.axvline(0, color="black", lw=0.5)

    h_sorted = np.sort(h)[::-1]
    ax_h.plot(h_sorted, color="#3a7d44")
    ax_h.set_title("h field (sorted descending)")
    ax_h.set_xlabel("feature rank")
    ax_h.set_ylabel("h")
    ax_h.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _format_extreme_pairs(
    iu: np.ndarray,
    ju: np.ndarray,
    j_flat: np.ndarray,
    order: np.ndarray,
    vocab: list[str],
) -> str:
    lines = ["| # | pair | J |", "| ---: | :--- | ---: |"]
    for r, k in enumerate(order, 1):
        lines.append(f"| {r} | {vocab[int(iu[k])]} × {vocab[int(ju[k])]} | {j_flat[k]:+.3f} |")
    return "\n".join(lines)


def _load_baseline_headline(phase_key: str) -> tuple[float, float] | None:
    """Read the locked post-Phase-3 validation baseline (k=1, top-1 / MRR)
    for the model section that matches this phase. Returns None when the
    baseline file is missing or the phase has no validation entry.
    """
    baseline_path = Path("tests/validation_baseline_post_phase3.json")
    if not baseline_path.exists():
        return None
    try:
        baseline = json.loads(baseline_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    section_for_phase = {
        "phase2": "Ising species",
        "phase3": "Ising item-pair",
    }
    section_name = section_for_phase.get(phase_key)
    if section_name is None:
        return None
    section = baseline.get("cross_model_species_granularity", {}).get(section_name)
    if section is None or "k=1" not in section:
        return None
    return float(section["k=1"]["top_1"]), float(section["k=1"]["mrr"])


if __name__ == "__main__":
    main()
