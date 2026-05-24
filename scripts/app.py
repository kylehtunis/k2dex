"""k2dex science — Streamlit webapp, scientific-dashboard edition.

All model knobs exposed: three phases, five completion techniques with full
parameter control, unlocked sliders in analysis and meta.

Three pages, all sharing one phase picker at the top:

- **Team completer** — conditional Ising posterior via Mean-field,
  Parallel-tempered, Sample distribution, Anneal → MAP, or Greedy descent.
  All technique parameters live in a "Run options" expander.
- **Team analysis** — per-team observables (E_adj, E_raw, ΣJ, corpus),
  pairwise J decomposition, top single swaps, greedy swap-chain critique.
  All display limits are slider-controlled.
- **Meta data** — fitted-model summary with validation headlines, extreme
  features by h, extreme ±J pairs, J / h distribution plots. Slider-controlled
  depth for all tables.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from k2dex import helpers, styles
from k2dex.constants import (
    PHASE1_MIN_USAGE,
    PHASE1_RIDGE_EPS,
    PHASE2_LR_C,
    TEAM_SIZE,
)
from k2dex.loaders import (
    build_species_item_model,
    build_species_model,
)
from k2dex.rendering import (
    intra_team_sum_j,
    nearest_observed,
    pairwise_j_rows,
    render_pairwise_j_table,
)
from k2dex.sampling import (
    anneal_mcmc,
    greedy_optimize,
    meanfield_marginals,
    parallel_tempered_mcmc,
    rank_single_swaps,
    swap_mcmc,
    team_energy,
)


@dataclass
class PhaseModel:
    """Resolved bundle of everything a page needs to render under one phase.

    `species_of` and `item_of` propagate uniqueness constraints to the sampler.
    Both are always set; for Phase 1/2, item_of is all-None and the uniqueness
    constraint is inert. `team_counts` is None for Phase 1 (Smogon chaos JSON
    has only aggregate statistics, no per-team rosters).
    """
    vocab: list[str]
    m: np.ndarray
    J: np.ndarray
    h: np.ndarray
    team_counts: Counter | None
    species_of: list[str]
    item_of: list[str | None]


def _load_phase(phase: str) -> tuple[str, PhaseModel]:
    if phase.startswith("Phase 1"):
        return "phase1", PhaseModel(*load_model_phase1())
    if phase.startswith("Phase 2"):
        return "species", PhaseModel(*load_model_species())
    return "species_item", PhaseModel(*load_model_species_item())


DATA_PATH_PHASE1 = "gen9championsvgc2026regma-1760.json"

# Log-spaced option lists for sliders. T and field_weight operate in log space
# (T governs Boltzmann factors exp(−ΔH/T); field_weight scales h, itself a
# log-odds), so linear sliders waste resolution at small values.
# field_weight includes 0.0 as a special case for pure-pairwise mode.
TEMPERATURE_OPTIONS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0]
FIELD_WEIGHT_OPTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ANNEAL_T_START_OPTIONS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
ANNEAL_T_END_OPTIONS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
PT_T_MAX_OPTIONS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


@st.cache_resource(show_spinner="Loading Phase 1 (Gaussian / Smogon chaos)...")
def load_model_phase1() -> tuple[
    list[str], np.ndarray, np.ndarray, np.ndarray,
    None, list[str], list[str | None],
]:
    """Phase 1: Gaussian / precision-matrix inverse Ising from Smogon chaos JSON.

    h is derived from mean-field self-consistency: h = logit(m) − J @ m.
    team_counts is None (chaos JSON has aggregate statistics, not per-team
    rosters). species_of is the vocab itself (each forme is a unique entry;
    forme mutual-exclusion is learned via strong −J). item_of is all-None
    (items baked into forme names like 'Charizard-Mega-Y' rather than tracked
    separately); the sampler's item-uniqueness constraint is thus inert.
    """
    chaos = helpers.load_chaos(DATA_PATH_PHASE1)
    vocab = helpers.build_vocab(chaos, min_usage=PHASE1_MIN_USAGE)
    C = helpers.build_cooccurrence(chaos, vocab)
    m, p_joint = helpers.binary_moments(chaos, vocab, C, team_size=TEAM_SIZE)
    Corr = helpers.binary_correlation(m, p_joint)
    J, _ = helpers.ising_gaussian(Corr, eps=PHASE1_RIDGE_EPS)
    h = np.log(m / (1 - m)) - J @ m
    species_of: list[str] = list(vocab)
    item_of: list[str | None] = [None] * len(vocab)
    return vocab, m, J, h, None, species_of, item_of


@st.cache_resource(show_spinner="Loading Phase 2 (pseudo-likelihood / Limitless species)...")
def load_model_species():
    """Streamlit-cached wrapper around `loaders.build_species_model`.
    See that function's docstring for fit details."""
    return build_species_model()


@st.cache_resource(show_spinner="Loading Phase 3 (pseudo-likelihood / item-pair vocab)...")
def load_model_species_item():
    """Streamlit-cached wrapper around `loaders.build_species_item_model`.
    See that function's docstring for fit details."""
    return build_species_item_model()


@st.cache_data(show_spinner=False)
def parallel_tempered_distribution(
    phase: str,  # in cache key only; differentiates models when index values collide
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
    """Multiple PT runs, aggregated into a completion distribution.

    Returns (distribution, n_kept, mean_local_accept, mean_swap_accept, ladder).
    """
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
    phase: str,
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
    """Run `n_runs` independent annealing chains; collect MAP outputs.

    Returns (sorted list of (completion, count), mean_accept_rate).
    """
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
    phase: str,
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
    """Multi-chain swap MCMC at fixed temperature.

    Returns (distribution, n_kept_samples, mean_accept_rate).
    """
    counts: dict[tuple[int, ...], int] = {}
    n_kept = 0
    accept_rates: list[float] = []
    fixed_idx = list(fixed_idx_tuple)
    excluded_idx = list(excluded_idx_tuple)
    fixed_set = set(fixed_idx)
    rng_master = np.random.default_rng(0)

    for _ in range(n_chains):
        chain_seed = int(rng_master.integers(2**31))
        result = swap_mcmc(
            _J, _h, TEAM_SIZE, fixed_idx, excluded_idx,
            field_weight, n_steps, temperature, chain_seed,
            species_of=_species_of, item_of=_item_of,
        )
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
    phase: str,
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
    """Cached wrapper around `meanfield_marginals`."""
    return meanfield_marginals(
        _J, _h, TEAM_SIZE,
        list(fixed_idx_tuple), list(excluded_idx_tuple),
        field_weight,
        species_of=_species_of, item_of=_item_of,
        n_iters=n_iters, tol=tol,
    )


@st.cache_data(show_spinner=False)
def greedy_optimize_chain(
    phase: str,
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
    """Cached wrapper around `greedy_optimize`."""
    return greedy_optimize(
        _J, _h, TEAM_SIZE,
        list(starting_team_tuple), list(pinned_tuple),
        list(excluded_idx_tuple), field_weight,
        species_of=_species_of, item_of=_item_of,
        max_swaps=max_swaps,
    )


def main() -> None:
    st.set_page_config(page_title="k2dex · science", layout="wide")
    styles.inject()

    header_col1, header_col2 = st.columns([3, 2])
    with header_col1:
        st.markdown("### k2dex · science")
    with header_col2:
        st.markdown('<div class="lab-phase-picker-marker"></div>', unsafe_allow_html=True)
        phase = st.radio(
            "Phase",
            ["Phase 1 (Gaussian)", "Phase 2 (PL, species)", "Phase 3 (PL, item-pair)"],
            horizontal=True,
            label_visibility="collapsed",
            key="phase_picker",
            help=(
                "**Phase 1** — Gaussian / precision-matrix Ising fit on Smogon "
                "chaos stats. Fast to fit; systematically under-magnitudes strong "
                "negative couplings.  "
                "**Phase 2** — pseudo-likelihood fit on Limitless tournament team "
                "rosters, species-only vocab.  "
                "**Phase 3** — pseudo-likelihood fit on (species, item) pairs. "
                "Restores held-item forme distinctions that Phase 2 collapses."
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
            "field_weight (h scale)",
            options=FIELD_WEIGHT_OPTIONS,
            value=0.5,
            key=f"completer_fw_{phase_key}",
            help=(
                "Scales h before sampling. 1.0 = data-calibrated posterior "
                "(popular mons dominate). 0.0 = pure pairwise (archetype-coherent, "
                "no popularity prior). Useful operating range 0.3–0.6 with "
                "current regularization (C=0.1)."
            ),
        )
        temperature = st.select_slider(
            "Temperature",
            options=TEMPERATURE_OPTIONS,
            value=0.1,
            key=f"completer_temp_{phase_key}",
            help=(
                "Sampling temperature T in exp(−ΔH/T). Lower = sharper Boltzmann "
                "distribution (fewer, more probable completions). Used by Sample "
                "distribution and as default seed for Anneal; ignored by Mean-field "
                "and Greedy descent."
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
                "damped fixed-point iteration. Instant; faithful ranking proxy "
                "for the true Boltzmann distribution at fw=1, T=1.  "
                "**Parallel-tempered** — replica-exchange MCMC. True Boltzmann "
                "draws at the cold-chain target T. Slow but unbiased.  "
                "**Sample distribution** — independent-chain swap MCMC at fixed T. "
                "Fast but each chain can get stuck in one basin at low T.  "
                "**Anneal → MAP** — simulated annealing (T_start → T_end) across "
                "multiple independent runs; returns the MAP each converged to.  "
                "**Greedy descent** — popularity-seeded greedy fill, then "
                "steepest-descent single-swap chain with pinned mons frozen."
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
            pt_t_max = st.select_slider("Hot T", options=PT_T_MAX_OPTIONS, value=2.0,
                                        key=f"completer_pt_tmax_{phase_key}")
            pt_K = st.slider("Ladder levels (K)", 3, 12, 7,
                             key=f"completer_pt_K_{phase_key}",
                             help="Replica chains from cold T to hot T. More levels → "
                                  "smaller gaps between rungs → higher swap acceptance.")
            pt_n_runs = st.slider("Independent PT runs", 1, 10, 3,
                                  key=f"completer_pt_nruns_{phase_key}")
            pt_n_steps = st.slider("Sweeps per run", 2000, 30000, 10000, step=1000,
                                   key=f"completer_pt_nsteps_{phase_key}")
            pt_burn_in = st.slider("Burn-in", 0, 10000, 3000, step=500,
                                   key=f"completer_pt_burnin_{phase_key}")
            pt_swap_interval = st.slider("Swap proposal interval", 1, 50, 10,
                                         key=f"completer_pt_swap_{phase_key}",
                                         help="Steps between replica-exchange proposals.")
        elif technique == "Mean-field":
            mf_n_iters = st.slider("Max iterations", 50, 500, 200, step=50,
                                   key=f"completer_mf_iters_{phase_key}",
                                   help="Cap on damped fixed-point iterations.")
            mf_tol_log = st.select_slider("Convergence tolerance (log10)",
                                          options=[-3, -4, -5, -6], value=-5,
                                          key=f"completer_mf_tol_{phase_key}")
            mf_tol = 10.0 ** mf_tol_log
            max_swaps = st.slider("Greedy descent cap", 1, 30, 10,
                                  key=f"completer_mf_swaps_{phase_key}",
                                  help="MF marginals seed the fill; descent then runs "
                                       "up to this many additional single-swap improvements.")
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
        if temperature >= pt_t_max:
            st.error(
                f"Temperature ({temperature}) must be strictly less than hot T ({pt_t_max}). "
                "Lower the Temperature slider or raise hot T."
            )
            return
        with st.spinner(f"PT: {pt_n_runs} runs × {pt_K} chains × {pt_n_steps} sweeps..."):
            dist, n_kept, mh_rate, swap_rate, ladder = parallel_tempered_distribution(
                phase_key, tuple(fixed_idx), tuple(excluded_idx), field_weight,
                temperature, pt_t_max, pt_K, pt_n_runs, pt_n_steps,
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

        # Greedy MF-MAP fill: top candidates by marginal, respecting uniqueness.
        used_sp: set[str] = {species_of[i] for i in fixed_idx}
        used_it: set[str] = {item_of[i] for i in fixed_idx if item_of[i] is not None}
        mf_fill: list[int] = []
        for cand in sorted_candidates:
            if len(mf_fill) == k_free:
                break
            if species_of[cand] in used_sp:
                continue
            if item_of[cand] is not None and item_of[cand] in used_it:
                continue
            mf_fill.append(int(cand))
            used_sp.add(species_of[cand])
            if item_of[cand] is not None:
                used_it.add(item_of[cand])
        if len(mf_fill) < k_free:
            st.error("Could not fill the team — insufficient non-conflicting candidates.")
            return

        start_team = sorted(fixed_idx + mf_fill)
        with st.spinner("Greedy descent from MF fill..."):
            final_team, chain = greedy_optimize_chain(
                phase_key,
                tuple(start_team), tuple(fixed_idx), tuple(excluded_idx),
                field_weight, max_swaps, J, h,
                _species_of=species_of, _item_of=item_of,
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Free slots", f"{k_free}")
        c2.metric("Valid candidates", f"{len(valid_idxs)}")
        c3.metric("MF iterations", f"{iters_used}")
        c4.metric("Descent swaps", f"{len(chain)} / {max_swaps}")

        st.markdown("**Greedy MF-MAP completion** (MF-ranked fill → greedy descent)")
        free_final = [i for i in final_team if i not in set(fixed_idx)]
        _render_completion_table(
            [(tuple(free_final), 1)], 1, fixed_idx, vocab, J, h, field_weight,
            team_counts, count_col=None,
        )

        st.markdown(f"**Top-{top_k} candidates by MF marginal** (single-slot ranking)")
        greedy_set = set(free_final)
        md = [
            "| # | MF prob | h_eff | in fill? | candidate |",
            "| ---: | ---: | ---: | :---: | :--- |",
        ]
        for rank, i in enumerate(sorted_candidates[:top_k], 1):
            prob_pct = float(marginals[i]) * 100
            h_eff_i = field_weight * float(h[i])
            in_fill = "✓" if int(i) in greedy_set else ""
            md.append(f"| {rank} | {prob_pct:.2f}% | {h_eff_i:+.3f} | {in_fill} | {vocab[i]} |")
        st.markdown("\n".join(md))

    else:  # Greedy descent — popularity-seeded fill, then descent with pinned frozen
        k_free = TEAM_SIZE - len(fixed_idx)
        if k_free < 0:
            st.error(f"Pinned more than {TEAM_SIZE} mons.")
            return
        excluded_set = set(excluded_idx)
        used_sp = {species_of[i] for i in fixed_idx}
        used_it = {item_of[i] for i in fixed_idx if item_of[i] is not None}
        order_by_m = sorted(range(len(vocab)), key=lambda i: -model.m[i])
        init_free: list[int] = []
        for cand in order_by_m:
            if len(init_free) == k_free:
                break
            if cand in excluded_set or cand in set(fixed_idx):
                continue
            if species_of[cand] in used_sp:
                continue
            if item_of[cand] is not None and item_of[cand] in used_it:
                continue
            init_free.append(cand)
            used_sp.add(species_of[cand])
            if item_of[cand] is not None:
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
        c2.metric("ΣJ (final)", f"{intra_team_sum_j(final_state, J):+.3f}")
        if not chain:
            st.info(
                "No improving swap from the popularity-seeded fill — already a "
                "local minimum at this field_weight. Try a lower field_weight "
                "(e.g. 0.3) to weight pairwise structure more heavily."
            )
        free_final = [i for i in final_team if i not in set(fixed_idx)]
        _render_completion_table(
            [(tuple(free_final), 1)], 1, fixed_idx, vocab, J, h, field_weight,
            team_counts, count_col=None,
        )


def _corpus_str(state: np.ndarray, vocab: list[str], team_counts: Counter | None) -> str:
    """Single merged corpus cell: 'N×' if seen exactly, 'Δk (N)' if not."""
    if team_counts is None:
        return "—"
    result = nearest_observed(state, vocab, team_counts)
    if result is None:
        return "—"
    d, c = result
    return f"{c}×" if d == 0 else f"Δ{d} ({c})"


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
    """Markdown completion table used by every technique.

    Columns: rank · [count] · raw E · adj E · ΣJ · [corpus] · completion
    count_col is the header label ("%" or "runs"), or None to omit (single-row tables).
    """
    has_obs = team_counts is not None
    corpus_header = " corpus |" if has_obs else ""
    corpus_sep = " :---: |" if has_obs else ""
    if count_col is not None:
        head = f"| # | {count_col} | raw E | adj E | ΣJ |{corpus_header} completion |"
        sep = f"| ---: | ---: | ---: | ---: | ---: |{corpus_sep} :--- |"
    else:
        head = f"| raw E | adj E | ΣJ |{corpus_header} completion |"
        sep = f"| ---: | ---: | ---: |{corpus_sep} :--- |"
    lines = [head, sep]
    for rank, (comp, count) in enumerate(rows, 1):
        state = np.zeros(len(vocab), dtype=bool)
        for i in fixed_idx:
            state[i] = True
        for i in comp:
            state[i] = True
        raw_E = team_energy(state, J, h)
        adj_E = team_energy(state, J, field_weight * h)
        sum_j = intra_team_sum_j(state, J)
        corpus_cell = f" {_corpus_str(state, vocab, team_counts)} |" if has_obs else ""
        names = ", ".join(vocab[i] for i in comp)
        if count_col == "%":
            lines.append(
                f"| {rank} | {(count / total) * 100:.2f}% "
                f"| {raw_E:+.3f} | {adj_E:+.3f} | {sum_j:+.3f} |{corpus_cell} {names} |"
            )
        elif count_col == "runs":
            lines.append(
                f"| {rank} | {count}/{total} "
                f"| {raw_E:+.3f} | {adj_E:+.3f} | {sum_j:+.3f} |{corpus_cell} {names} |"
            )
        else:
            lines.append(
                f"| {raw_E:+.3f} | {adj_E:+.3f} | {sum_j:+.3f} |{corpus_cell} {names} |"
            )
    st.markdown("\n".join(lines))


def _render_analysis(phase_key: str, model: PhaseModel) -> None:
    """Per-team analysis under the fitted (J, h): observables, pairwise J
    decomposition, top single swaps, and greedy swap-chain critique.
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

    # Species/item uniqueness check (always runs; inert under Phase 1/2 vocabs).
    seen_sp: dict[str, str] = {}
    for i in team_idx:
        sp = species_of[i]
        if sp in seen_sp:
            st.error(f"Two entries for species **{sp}** ({seen_sp[sp]}, {vocab[i]}). "
                     "Pick one variant.")
            return
        seen_sp[sp] = vocab[i]
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
        "field_weight (for E_adj and greedy critique)",
        options=FIELD_WEIGHT_OPTIONS,
        value=0.5,
        key=f"analysis_fw_{phase_key}",
        help=(
            "Rescales h before computing E_adj and choosing greedy swaps. "
            "E_raw always uses fw=1. Lower fw weights pairwise coupling "
            "structure more heavily relative to popularity."
        ),
    )

    # Observables strip
    raw_E = team_energy(state, J, h)
    adj_E = team_energy(state, J, field_weight * h)
    sum_j = intra_team_sum_j(state, J)
    cols = st.columns(5)
    cols[0].metric("E_adj", f"{adj_E:+.3f}", help=f"fw = {field_weight}")
    cols[1].metric("E_raw", f"{raw_E:+.3f}", help="fw = 1.0")
    cols[2].metric("ΣJ", f"{sum_j:+.3f}",
                   help="Intra-team coupling sum. Positive = archetype signature; "
                        "negative = balance-team signature (model can't see what holds it together).")
    if team_counts is not None:
        corpus = nearest_observed(state, vocab, team_counts)
        if corpus is None:
            cols[3].metric("Corpus obs", "—")
            cols[4].metric("Δ to obs", "—")
        else:
            d, c = corpus
            cols[3].metric("Corpus obs", f"{c}" if d == 0 else "0",
                           help="Exact-roster appearances in the ingested corpus.")
            cols[4].metric("Δ to obs", f"{d}",
                           help="Min single-swap distance to the nearest observed team.")
    else:
        cols[3].metric("Corpus obs", "—",
                       help="Phase 1 has no per-team counts (Smogon aggregates only).")
        cols[4].metric("Δ to obs", "—")

    # Pairwise J decomposition
    st.markdown("##### Pairwise J decomposition")
    rows = pairwise_j_rows(team_idx, vocab, J)
    st.markdown(render_pairwise_j_table(rows))
    st.caption(
        "C(6, 2) = 15 unordered pairs sorted by |J|. "
        "**% of |J| sum** shows which pair drives the team's coherence: "
        "one row at ≥30% = single pair doing most structural work; "
        "flat 6–8% across all pairs = diffuse balance team."
    )

    # Top single-swap suggestions (independent, not chained)
    st.markdown("##### Top single swaps from this team")
    top_n_swaps = st.slider(
        "Show top N swaps", 5, 50, 15,
        key=f"analysis_topn_swaps_{phase_key}",
        help="Every legal (out ∈ team, in ∉ team) swap is scored from the "
             "starting team and ranked by ΔE_adj. Unlike the greedy chain "
             "below, no swap is applied — each row is an independent one-step alternative.",
    )
    ranked_swaps = rank_single_swaps(
        J, h, team_idx, field_weight,
        species_of=species_of, item_of=item_of, top_n=top_n_swaps,
    )
    if not ranked_swaps:
        st.info("No legal single swap exists from this team.")
    else:
        swap_md = [
            "| # | swap | ΔE_adj | ΔE_raw | ΔΣJ |",
            "| ---: | :--- | ---: | ---: | ---: |",
        ]
        for r, sw in enumerate(ranked_swaps, 1):
            label = f"{vocab[sw['out_idx']]} → {vocab[sw['in_idx']]}"
            swap_md.append(
                f"| {r} | {label} | {sw['delta_E_adj']:+.3f} | "
                f"{sw['delta_E_raw']:+.3f} | {sw['delta_sum_j']:+.3f} |"
            )
        st.markdown("\n".join(swap_md))
        st.caption(
            "Each row is evaluated **from the starting team** (not chained). "
            "Negative ΔE_adj = improving swap under fw. **ΔΣJ** isolates the "
            "pairwise contribution: a swap with negative ΔE but near-zero ΔΣJ "
            "is a popularity (h) move, not a structural one."
        )

    # Greedy swap-chain critique (no pinning — full team is fair game)
    st.markdown("##### Greedy critique · single-swap chain")
    max_swaps = st.slider(
        "Max swaps", 1, 30, 15,
        key=f"analysis_max_swaps_{phase_key}",
        help="Hard cap on swap chain length. Greedy descent usually converges "
             "in 3–10 swaps; this is a safety bound.",
    )
    with st.spinner("Computing greedy descent..."):
        final_team, chain = greedy_optimize_chain(
            phase_key,
            tuple(team_idx), (),
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
    sc4.metric("ΔΣJ", f"{final_sum_j - start_sum_j:+.3f}")

    if not chain:
        st.info(
            "No improving single-swap exists — this team is a local minimum "
            "under the current field_weight. Try a different field_weight "
            "(e.g. 0.0 for pure pairwise) to re-rank under a different objective."
        )
        return

    has_obs = team_counts is not None
    corpus_header = " corpus |" if has_obs else ""
    corpus_sep = " :---: |" if has_obs else ""

    def corpus_cell(s: np.ndarray) -> str:
        return f" {_corpus_str(s, vocab, team_counts)} |" if has_obs else ""

    md = [
        f"| # | swap | adj E | raw E | ΣJ |{corpus_header} team |",
        f"| ---: | :--- | ---: | ---: | ---: |{corpus_sep} :--- |",
        f"| 0 | _starting team_ | {start_adj_E:+.3f} | {start_raw_E:+.3f} | "
        f"{start_sum_j:+.3f} |{corpus_cell(start_state)} "
        f"{', '.join(vocab[i] for i in team_idx)} |",
    ]
    for ev in chain:
        swap_label = f"{vocab[ev['out_idx']]} → {vocab[ev['in_idx']]}"
        after_state = team_state(ev["team_after"])
        md.append(
            f"| {ev['step']} | {swap_label} | "
            f"{ev['energy_adj_after']:+.3f} | {ev['energy_raw_after']:+.3f} | "
            f"{ev['sum_j_after']:+.3f} |{corpus_cell(after_state)} "
            f"{', '.join(vocab[i] for i in ev['team_after'])} |"
        )
    st.markdown("\n".join(md))
    st.caption(
        "Each row is the team **after** that swap. Swap #1 is the model's biggest "
        "complaint about the starting team; subsequent swaps are chained. The final "
        "team is the local minimum reachable by single-swap moves — not necessarily "
        "the global MAP."
    )


def _render_meta(phase_key: str, model: PhaseModel) -> None:
    """Format-wide statistics: model summary, extreme h features, extreme ±J pairs,
    J / h distribution plots.
    """
    st.subheader("Meta data")
    st.caption("Format-wide statistics derived from the fitted (J, h).")

    vocab = model.vocab
    J, h = model.J, model.h
    team_counts = model.team_counts
    V = len(vocab)

    # Summary strip
    n_corpus_teams = sum(team_counts.values()) if team_counts is not None else None
    fit_label = {
        "phase1": f"Gaussian · ridge ε={PHASE1_RIDGE_EPS}",
        "species": f"sklearn LogReg · L2 · C={PHASE2_LR_C}",
        "species_item": f"sklearn LogReg · L2 · C={PHASE2_LR_C}",
    }[phase_key]
    phase_label = {
        "phase1": "Phase 1", "species": "Phase 2", "species_item": "Phase 3",
    }[phase_key]
    summary_cols = st.columns(6)
    summary_cols[0].metric("Phase", phase_label)
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
                               help="Held-out leave-1-out accuracy (team-level 90/10 split).")
        summary_cols[4].metric("MRR @ k=1", f"{mrr:.3f}")
    else:
        summary_cols[3].metric("Top-1 @ k=1", "—",
                               help="No locked validation baseline for this phase.")
        summary_cols[4].metric("MRR @ k=1", "—")
    summary_cols[5].metric("Fit", fit_label)

    # Extreme features by h
    st.markdown("##### Extreme features by h")
    n_show = st.slider(
        "Show top N features (each direction)", 10, min(150, V // 2), 30,
        key=f"meta_topn_{phase_key}",
        help="Sorted by h (log-odds of inclusion in a random team under the model). "
             "Top +h = most popular features; top −h = features the model considers "
             "unlikely to appear on any team.",
    )
    order_desc = np.argsort(-h)
    order_asc = np.argsort(h)
    top_pos_col, top_neg_col = st.columns(2)
    with top_pos_col:
        st.markdown("**Top +h (most popular)**")
        st.markdown(_format_feature_h_table(order_desc[:n_show], vocab, h, model.m))
    with top_neg_col:
        st.markdown("**Top −h (most unlikely)**")
        st.markdown(_format_feature_h_table(order_asc[:n_show], vocab, h, model.m))
    st.caption(
        "**h** is the per-feature log-odds field. Under mean-field: "
        "m̂ ≈ σ(h + J·m), so h ≈ logit(m̂) for popular features. "
        "The −h side highlights features the model is confident won't "
        "appear — often more informative than the +h side for niche picks."
    )

    # Extreme ±J pairs
    st.markdown("##### Extreme couplings")
    n_pairs = st.slider(
        "Show top N pairs (each direction)", 10, 50, 25,
        key=f"meta_npairs_{phase_key}",
        help="Top +J pairs are synergies; top −J pairs are exclusions / "
             "mutually-exclusive picks.",
    )
    iu, ju = np.triu_indices(V, k=1)
    j_flat = J[iu, ju]

    # Phase 3: filter same-species AND same-item pairs from the extreme lists.
    # Both are trivial mutual-exclusion artifacts; they dominate the −J side and
    # crowd out real cross-species structure. Phase 1/2 masks are no-ops.
    species_arr = np.array(model.species_of)
    item_arr = np.array([it if it is not None else "" for it in model.item_of])
    has_item = item_arr != ""
    cross_species = species_arr[iu] != species_arr[ju]
    cross_item = ~(has_item[iu] & has_item[ju] & (item_arr[iu] == item_arr[ju]))
    keep = cross_species & cross_item
    iu_v, ju_v, j_v = iu[keep], ju[keep], j_flat[keep]

    pos_col, neg_col = st.columns(2)
    with pos_col:
        st.markdown("**Top +J (synergies)**")
        pos_order = np.argsort(-j_v)[:n_pairs]
        st.markdown(_format_extreme_pairs(iu_v, ju_v, j_v, pos_order, vocab))
    with neg_col:
        st.markdown("**Top −J (exclusions)**")
        neg_order = np.argsort(j_v)[:n_pairs]
        st.markdown(_format_extreme_pairs(iu_v, ju_v, j_v, neg_order, vocab))
    if model.species_of is not None:
        st.caption(
            "Same-species and same-item pairs are filtered out — they dominate "
            "the −J side as trivial mutual-exclusion artifacts."
        )

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


def _format_feature_h_table(
    order: np.ndarray,
    vocab: list[str],
    h: np.ndarray,
    m: np.ndarray,
) -> str:
    lines = ["| # | feature | h | m̂ |", "| ---: | :--- | ---: | ---: |"]
    for rank, i in enumerate(order, 1):
        lines.append(
            f"| {rank} | {vocab[int(i)]} | {float(h[int(i)]):+.3f} | "
            f"{float(m[int(i)]):.4f} |"
        )
    return "\n".join(lines)


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
    """Read the locked post-Phase-4 validation baseline (k=1 top-1 / MRR)
    for the model matching this phase. Returns None when missing or unavailable.
    """
    baseline_path = Path("tests/validation_baseline_post_phase4.json")
    if not baseline_path.exists():
        return None
    try:
        baseline = json.loads(baseline_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    section_for_phase = {
        "species": "Ising species",
        "species_item": "Ising item-pair",
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
