"""k2dex science — Streamlit webapp around the inverse-Ising team model.

Three pages, all sharing one model picker at the top:

- **Team completer** — completions from the conditional Ising posterior.
  Default: mean-field marginals seed a greedy descent (fast). Toggle on
  "Full statistical sampler (slow)" for parallel-tempered MCMC.
- **Team analysis** — per-team observables, pairwise coupling decomposition,
  top single swaps, greedy swap-chain critique.
- **Meta data** — fitted-model summary, top features by bias, top ±coupling
  pairs, coupling / bias distribution plots.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import limitless_ingest
import rendering_html as rh
import styles
from constants import (
    PHASE2_LR_C,
    PHASE2_MIN_TEAM_COUNT,
    PHASE2_MIN_TEAMS,
    TEAM_SIZE,
)
from models import fit_pl_ising
from rendering import (
    intra_team_sum_j,
    nearest_observed,
    pairwise_j_rows,
)
from sampling import (
    greedy_optimize,
    meanfield_marginals,
    parallel_tempered_mcmc,
    rank_single_swaps,
    team_energy,
)


@dataclass
class PhaseModel:
    """Resolved bundle of everything a page needs to render under one model.

    `species_of` and `item_of` propagate the species-uniqueness and
    item-uniqueness constraints to the sampler. They are populated under
    both Species and Species @ Item vocabs; the constraint is inert when
    duplication is structurally impossible (species vocab is unique by
    species; item dimension is None on Species).
    """
    vocab: list[str]
    m: np.ndarray
    J: np.ndarray
    h: np.ndarray
    team_counts: Counter
    species_of: list[str]
    item_of: list[str | None]


def _load_phase(phase: str) -> tuple[str, PhaseModel]:
    if phase.startswith("Species @ Item"):
        return "species_item", PhaseModel(*load_model_species_item())
    return "species", PhaseModel(*load_model_species())


# Log-spaced options for the two completer sliders. Both T and Bias Adjustment
# operate in log space (T governs Boltzmann factors exp(-ΔH/T); Bias Adjustment
# scales h, itself a log-odds), so linear sliders waste resolution at small
# values. Bias Adjustment includes 0.0 as a special-case for pure-pairwise mode.
TEMPERATURE_OPTIONS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0]
FIELD_WEIGHT_OPTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Locked completer/analysis/meta knobs (sliders removed in the v1 simplification).
TOP_COMPLETIONS = 10
TOP_SINGLE_SWAPS = 10
GREEDY_MAX_SWAPS = 10
META_TOP_FEATURES = 25
META_TOP_PAIRS = 25
PT_HOT_T = 3.0
PT_LADDER_LEVELS = 7
PT_RUNS = 3
PT_SWEEPS = 10000
PT_BURN_IN = 3000
PT_SWAP_INTERVAL = 10
MF_MAX_ITERS = 200
MF_TOL = 1e-5


@st.cache_resource(show_spinner="Loading Species model (pseudo-likelihood / Limitless teams)...")
def load_model_species() -> tuple[
    list[str], np.ndarray, np.ndarray, np.ndarray,
    Counter[frozenset[str]],
    list[str], list[str | None],
]:
    """Species model: pseudo-likelihood inverse Ising from Limitless tournament
    team data, projected to species-only vocab.

    Fits V per-spin logistic regressions (one per Pokemon in vocab) on the binary
    species-indicator matrix; intercepts -> h, coefficients -> J rows.
    Post-hoc symmetrization J = (J_asym + J_asym.T) / 2.

    The ingest format is now (species, item) tuples (v2 cache); this model
    projects down to species-only via `limitless_ingest.species_only_teams`.

    Returns (vocab, m, J, h, team_counts, species_of, item_of). species_of equals
    vocab (each species is one entry); item_of is all-None (items dropped at
    projection). Sampler's uniqueness constraint is thus inert -- species can't
    duplicate within a single-entry-per-species vocab.
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


@st.cache_resource(show_spinner="Loading Species @ Item model (pseudo-likelihood / item-pair vocab)...")
def load_model_species_item() -> tuple[
    list[str], np.ndarray, np.ndarray, np.ndarray,
    Counter[frozenset[str]],
    list[str], list[str | None],
]:
    """Species @ Item model: pseudo-likelihood inverse Ising over (species,
    item) pairs.

    Same fit machinery as the Species model but with each Pokemon-item pair as
    its own feature. Restores held-item forme distinctions that the species
    projection collapses; lets the model see role specialization (one Charizard
    is a Mega sweeper, another is a Choice Scarf attacker). Vocab cutoff is
    the same `PHASE2_MIN_TEAM_COUNT` (pair must appear in at least that many
    teams).

    Display strings: 'Species @ Item' for items, bare species for itemless mons
    (see `format_pair`). species_of and item_of are populated so the sampler's
    uniqueness constraint can reject duplicate-species and duplicate-item swaps.

    team_counts here counts each exact (species, item)-roster's occurrences in
    the corpus, keyed on frozensets of display strings.
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
    kept. `_species_of` and `_item_of` propagate uniqueness constraints to the
    sampler. Returns (distribution, n_kept_total, mean_local_accept,
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
    but cached so the same fixed/Bias-Adjustment combo hits the cache on
    re-render."""
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
    styles.inject()

    # Persistent header: wordmark + model picker, shown above the tab bar.
    # Streamlit re-runs the script on every interaction so the radio's value
    # is naturally page-global without needing st.session_state plumbing.
    # The `.lab-phase-picker-marker` div is read by styles.py via `:has()`
    # to scope segmented-control styling to this radio only.
    header_col1, header_col2 = st.columns([3, 2])
    with header_col1:
        st.markdown(
            '<div class="lab-wordmark">k2dex'
            '<span class="lab-wordmark-mono">v0.4</span></div>',
            unsafe_allow_html=True,
        )
    with header_col2:
        st.markdown(
            '<div class="lab-phase-picker-marker"></div>',
            unsafe_allow_html=True,
        )
        phase = st.radio(
            "Model",
            ["Species @ Item", "Species"],
            horizontal=True,
            label_visibility="collapsed",
            key="phase_picker",
            help=(
                "**Species** — pseudo-likelihood inverse-Ising fit on Limitless "
                "team rosters, species-only vocab.  "
                "**Species @ Item** — same fit on (species, item) pair features. "
                "Restores held-item forme distinctions and role specialization."
            ),
        )

    phase_key, model = _load_phase(phase)
    corpus_caption = _corpus_caption(model)

    completer_tab, analysis_tab, meta_tab = st.tabs(
        ["Team completer", "Team analysis", "Meta data"],
    )
    with completer_tab:
        _render_completer(phase_key, model, corpus_caption)
    with analysis_tab:
        _render_analysis(phase_key, model, corpus_caption)
    with meta_tab:
        _render_meta(phase_key, model, corpus_caption)


def _corpus_caption(model: PhaseModel) -> str:
    """Right-aligned mono caption shown next to each page's H1. Both models
    are fit on the same Limitless ingest; tournament count isn't tracked
    through the loaded ``PhaseModel`` yet so we surface the team count.
    """
    n_teams = sum(model.team_counts.values()) if model.team_counts else 0
    return f"Limitless 2026 Reg M-A · {n_teams:,} teams"


def _render_completer(phase_key: str, model: PhaseModel, corpus_caption: str) -> None:
    """Team completer — sample completions from the conditional Ising posterior.

    Five techniques share the same constraint widgets and field-weight slider:
    Mean-field (deterministic), Parallel-tempered, Sample distribution,
    Anneal → MAP, Greedy descent. Each renders its own headline metrics +
    completion table.
    """
    st.markdown(
        rh.page_title(
            eyebrow="notebook · /completer",
            h1="Team completer",
            right_caption=corpus_caption,
        ),
        unsafe_allow_html=True,
    )

    vocab = model.vocab
    J, h = model.J, model.h
    species_of, item_of = model.species_of, model.item_of
    team_counts = model.team_counts

    name_to_idx = {name: i for i, name in enumerate(vocab)}
    sorted_vocab = sorted(vocab, key=lambda v: -model.m[name_to_idx[v]])

    # Excludes are species-level (not (species, item)-level): in Phase 3 it
    # would be odd to exclude one specific item-variant but allow others of
    # the same species. Build a species → [vocab idx] map and a popularity-
    # sorted unique species list to feed the excluded multiselect.
    species_to_idxs: dict[str, list[int]] = {}
    for i, sp in enumerate(species_of):
        species_to_idxs.setdefault(sp, []).append(i)
    sorted_species = sorted(
        species_to_idxs.keys(),
        key=lambda s: -sum(model.m[i] for i in species_to_idxs[s]),
    )

    # §01 Starting roster — slot-strip visualization sits above the actual
    # multiselect inputs. We reserve placeholders here, render the inputs
    # below, then write into the placeholders once their values are known.
    section01_slot = st.empty()
    strip_slot = st.empty()
    excluded_slot = st.empty()

    constraint_col1, constraint_col2 = st.columns(2)
    with constraint_col1:
        fixed_names = st.multiselect(
            "Starting Roster",
            sorted_vocab,
            max_selections=TEAM_SIZE,
            placeholder="Choose Pokemon to include",
            key=f"completer_fixed_{phase_key}",
        )
    with constraint_col2:
        excluded_species = st.multiselect(
            "Exclude (must NOT appear)",
            sorted_species,
            placeholder="Choose species to exclude",
            key=f"completer_excluded_{phase_key}",
            help=(
                "Excludes are species-level: selecting a species "
                "rules out every (species, item) variant."
            ),
        )

    excluded_species_set = set(excluded_species)
    overlap = {n for n in fixed_names if species_of[name_to_idx[n]] in excluded_species_set}
    if overlap:
        st.error(f"Cannot be both pinned and excluded: {', '.join(sorted(overlap))}")
        return

    section01_slot.markdown(
        rh.section_label("01", f"Starting roster · {len(fixed_names)} of {TEAM_SIZE} set"),
        unsafe_allow_html=True,
    )
    strip_slot.markdown(
        rh.slot_strip(fixed_names, team_size=TEAM_SIZE),
        unsafe_allow_html=True,
    )
    excluded_slot.markdown(
        rh.excluded_row(excluded_species, note=None),
        unsafe_allow_html=True,
    )

    # §02 Sampler — two sliders + a "Full statistical sampler (slow)" toggle.
    # Toggle off (default): MF-fill seeds greedy descent (fast, deterministic).
    # Toggle on: parallel-tempered MCMC at the chosen cold-target temperature.
    st.markdown(rh.section_label("02", "Sampler"), unsafe_allow_html=True)
    fw_col, temp_col = st.columns(2)
    with fw_col:
        field_weight = st.select_slider(
            "Bias Adjustment",
            options=FIELD_WEIGHT_OPTIONS,
            value=0.5,
            key=f"completer_fw_{phase_key}",
            help=(
                "Scales the Bias before sampling. 1.0 = data-calibrated "
                "(popular mons dominate). 0.0 = pure coupling structure, no "
                "popularity prior. Useful operating range 0.3–0.6."
            ),
        )
    with temp_col:
        temperature = st.select_slider(
            "Temperature",
            options=TEMPERATURE_OPTIONS,
            value=0.5,
            key=f"completer_temp_{phase_key}",
            help=(
                "Cold-chain target temperature for the statistical sampler. "
                "Lower = sharper Boltzmann distribution (fewer, more probable completions). "
                "Ignored when the statistical sampler is off. "
                f"Hot chain is fixed at T={PT_HOT_T}; cold target must be below that."
            ),
        )

    use_pt = st.toggle(
        "Full statistical sampler (slow)",
        value=False,
        key=f"completer_use_pt_{phase_key}",
        help=(
            "Off (default): mean-field marginals seed a greedy descent. "
            "Fast and deterministic. On: parallel-tempered MCMC at the "
            "chosen Temperature. Slow but yields a full Boltzmann distribution "
            "over completions instead of a single top result."
        ),
    )
    if use_pt and temperature >= PT_HOT_T:
        st.error(
            f"Temperature ({temperature}) must be strictly less than hot-T ({PT_HOT_T}) "
            "for the parallel-tempered sampler. Lower the Temperature slider."
        )
        return

    # PT parameter expander — only visible when PT is on and temperature is valid.
    # Defaults are the module-level constants; local variables shadow them for the call.
    pt_runs = PT_RUNS
    pt_ladder = PT_LADDER_LEVELS
    pt_sweeps = PT_SWEEPS
    pt_burn_in = PT_BURN_IN

    if use_pt:
        with st.expander("Sampler parameters", expanded=False):
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                pt_runs = st.slider(
                    "Runs", 1, 10, PT_RUNS,
                    key=f"pt_runs_{phase_key}",
                    help="Independent PT runs. More runs → more samples and better "
                         "distribution coverage. Runtime scales linearly.",
                )
                pt_ladder = st.slider(
                    "Ladder levels (K)", 3, 15, PT_LADDER_LEVELS,
                    key=f"pt_ladder_{phase_key}",
                    help="Replica chains from cold T to hot T=3.0. More levels → "
                         "smaller gaps between rungs → higher replica swap acceptance. "
                         "Runtime scales linearly.",
                )
            with p_col2:
                pt_sweeps = st.slider(
                    "Sweeps per run", 1000, 50000, PT_SWEEPS, step=1000,
                    key=f"pt_sweeps_{phase_key}",
                    help="Swap-move steps per chain per run. More sweeps → more "
                         "samples and better convergence. Runtime scales linearly.",
                )
                pt_burn_in = st.slider(
                    "Burn-in", 0, pt_sweeps - 1, min(PT_BURN_IN, pt_sweeps - 1),
                    step=500,
                    key=f"pt_burn_in_{phase_key}",
                    help="Steps discarded at the start of each run before recording. "
                         "Aim for ~20–30% of sweeps. Too short risks non-equilibrium "
                         "samples.",
                )

    run = st.button(
        "Sample" if use_pt else "Suggest team",
        type="primary",
        use_container_width=True,
        key=f"completer_run_{phase_key}",
    )
    if not run:
        return

    fixed_idx = sorted({name_to_idx[n] for n in fixed_names})
    excluded_idx = sorted({
        i for sp in excluded_species for i in species_to_idxs.get(sp, [])
    })

    if use_pt:
        with st.spinner(
            f"PT: {pt_runs} runs × {pt_ladder} chains × {pt_sweeps} sweeps..."
        ):
            dist, n_kept, mh_rate, swap_rate, ladder = parallel_tempered_distribution(
                phase_key, tuple(fixed_idx), tuple(excluded_idx), field_weight,
                temperature, PT_HOT_T, pt_ladder, pt_runs, pt_sweeps,
                pt_burn_in, PT_SWAP_INTERVAL, J, h,
                _species_of=species_of, _item_of=item_of,
            )
        if dist is None:
            st.error("Not enough available Pokemon to fill the team after applying constraints.")
            return
        top5_mass = sum(c for _, c in dist[:5]) / n_kept * 100
        st.markdown(rh.section_label("03", "Observables · last run"), unsafe_allow_html=True)
        st.markdown(
            rh.stat_strip([
                ("Cold samples", f"{n_kept:,}", "kept post burn-in"),
                ("Distinct", f"{len(dist):,}", "completions"),
                ("Top-5 mass", f"{top5_mass:.2f}%", "concentration",
                 "Fraction of all samples in the top-5 completions. "
                 "Healthy: 5%+, upper bound depends on number of completions."
                 "Too high → Distribution is too steep; lower Temperature. "
                 "Below 5% → Distribution is too flat; raise Temperature."),
                ("Local accept", f"{mh_rate * 100:.1f}%", "within-chain",
                 "Within-chain Metropolis-Hastings acceptance rate, averaged across all replica chains. "
                 "Healthy: 20–60%. "
                 "Below 15% → chains too cold; raise target Temperature."
                 "Above 80% → chains too hot; lower target Temperature."),
                ("Replica swap", f"{swap_rate * 100:.1f}%", "between chains",
                 "Acceptance rate for replica exchange between adjacent temperature rungs. "
                 "Healthy: 20-80%. "
                 "Below 15% → rungs are spaced too far apart; increase ladder levels. "
                 "Above 80% → rungs are too close; decrease ladder levels."),
            ]),
            unsafe_allow_html=True,
        )
        st.markdown(
            rh.section_label(
                "04", "Top completions",
                right=(
                    f"ordered by sample frequency · "
                    f"{min(TOP_COMPLETIONS, len(dist))} of {len(dist):,} shown"
                ),
            ),
            unsafe_allow_html=True,
        )
        _render_completion_table(
            dist[:TOP_COMPLETIONS], n_kept, fixed_idx, vocab, J, h, field_weight,
            team_counts, count_col="%",
        )
        return

    # Fast path: MF-fill + greedy descent. The MF marginals are computed at
    # the chosen Bias Adjustment so the fill respects the slider end-to-end
    # (this was the popularity-only-fill bug in the prior greedy-descent
    # implementation).
    with st.spinner("Mean-field fill + greedy descent..."):
        mf_result = meanfield_distribution(
            phase_key, tuple(fixed_idx), tuple(excluded_idx), field_weight,
            MF_MAX_ITERS, MF_TOL, J, h,
            _species_of=species_of, _item_of=item_of,
        )
        if mf_result is None:
            st.error("Not enough available Pokemon to fill the team after applying constraints.")
            return
        marginals, valid_mask, _ = mf_result
        k_free = TEAM_SIZE - len(fixed_idx)
        if k_free < 0:
            st.error(f"Pinned more than {TEAM_SIZE} mons.")
            return
        valid_idxs = list(np.where(valid_mask)[0])
        sorted_candidates = sorted(valid_idxs, key=lambda i: -marginals[i])
        used_sp = {species_of[i] for i in fixed_idx}
        used_it = {item_of[i] for i in fixed_idx if item_of[i] is not None}
        init_free: list[int] = []
        for cand in sorted_candidates:
            if len(init_free) == k_free:
                break
            if species_of[cand] in used_sp:
                continue
            if item_of[cand] is not None and item_of[cand] in used_it:
                continue
            init_free.append(int(cand))
            used_sp.add(species_of[cand])
            if item_of[cand] is not None:
                used_it.add(item_of[cand])
        if len(init_free) < k_free:
            st.error("Could not fill the team — insufficient non-conflicting candidates.")
            return
        start_team = sorted(fixed_idx + init_free)
        final_team, chain = greedy_optimize_chain(
            phase_key,
            tuple(start_team), tuple(fixed_idx), tuple(excluded_idx),
            field_weight, GREEDY_MAX_SWAPS, J, h,
            _species_of=species_of, _item_of=item_of,
        )
    final_state = np.zeros(len(vocab), dtype=bool)
    for i in final_team:
        final_state[i] = True
    free_final = [i for i in final_team if i not in set(fixed_idx)]

    st.markdown(rh.section_label("03", "Observables"), unsafe_allow_html=True)
    st.markdown(
        rh.stat_strip([
            ("Score (adj)",
             f"{-team_energy(final_state, J, field_weight * h):+.3f}",
             f"Bias Adj. = {field_weight}"),
            ("Coherence",
             f"{intra_team_sum_j(final_state, J):+.3f}",
             "intra-team coupling"),
            ("Swaps taken", f"{len(chain)} / {GREEDY_MAX_SWAPS}",
             "MF fill → local min"),
        ]),
        unsafe_allow_html=True,
    )
    st.markdown(
        rh.section_label(
            "04", "Suggested completion",
            right=(
                "mean-field fill → greedy descent · "
                f"pinned: {len(fixed_idx)}"
            ),
        ),
        unsafe_allow_html=True,
    )
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
    """Render the §04 top-completions table as a rich HTML table.

    Columns (left → right):
        rank · completion (sprite cards) · freq · Score (adj) · Score (raw) · Coherence · corpus

    ``count_col`` controls the freq column:
        - ``"%"``    — frequency as ``28.4%`` + a unipolar mini-bar
        - ``None``   — column omitted entirely (single-row tables for the
          fast-path MF-fill + greedy result, where "frequency" isn't
          meaningful)

    Score is rendered with ``higher = better`` (sign-flipped energy:
    ``-team_energy(state, J, field_weight * h)``). The top row gets a faint
    accent tint (``tr.top-row``). The ``corpus`` column merges the old
    ``obs`` and ``Δswaps`` columns: a green ``N×`` chip when the exact
    roster was seen in the corpus, an amber/red ``Δk (N)`` chip otherwise
    (where N is the nearest observed roster's count).
    """
    has_obs = team_counts is not None
    show_freq = count_col is not None

    head_cells: list[str] = ['<th class="num">#</th>', "<th>completion</th>"]
    if show_freq:
        head_cells.append('<th class="num">freq</th>')
    head_cells.append('<th class="num">Score (adj)</th>')
    head_cells.append('<th class="num">Score (raw)</th>')
    head_cells.append('<th class="num">Coherence</th>')
    if has_obs:
        head_cells.append('<th class="num">corpus</th>')

    # Cap mini-bar at the top row's frequency so the visual scales relative
    # to the most-frequent completion (clearest read at a glance).
    max_freq = max((count / total for _, count in rows), default=1.0) if show_freq else 1.0

    body_rows: list[str] = []
    for rank, (comp, count) in enumerate(rows, 1):
        state = np.zeros(len(vocab), dtype=bool)
        for i in fixed_idx:
            state[i] = True
        for i in comp:
            state[i] = True
        score_adj = -team_energy(state, J, field_weight * h)
        score_raw = -team_energy(state, J, h)
        sum_j = intra_team_sum_j(state, J)

        pair_html = '<div class="lab-comp-pair">' + "".join(
            rh.comp_mon_cell(vocab[i]) for i in comp
        ) + "</div>"

        cells = [
            f'<td class="rank">{rank:02d}</td>',
            f'<td>{pair_html}</td>',
        ]
        if show_freq:
            freq = count / total
            bar = rh.mini_bar(freq, max_freq, width=60)
            cells.append(
                f'<td class="num"><div class="lab-comp-freq">'
                f'<span class="lab-comp-freq-pct">{freq * 100:.1f}%</span>{bar}'
                f'</div></td>'
            )
        cells.append(f'<td class="num">{score_adj:+.3f}</td>')
        cells.append(f'<td class="num">{score_raw:+.3f}</td>')
        cells.append(f'<td class="num">{rh.score_chip(sum_j, "signed")}</td>')
        if has_obs:
            corpus = nearest_observed(state, vocab, team_counts)
            if corpus is None:
                cells.append('<td class="num"></td>')
            else:
                d, c = corpus
                cells.append(f'<td class="num">{rh.corpus_cell(d, c)}</td>')

        row_class = ' class="top-row"' if rank == 1 else ""
        body_rows.append(f'<tr{row_class}>{"".join(cells)}</tr>')

    html = (
        '<table class="lab-comp-table">'
        f'<thead><tr>{"".join(head_cells)}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_analysis(phase_key: str, model: PhaseModel, corpus_caption: str) -> None:
    """Per-team analysis under the fitted model: observables, pairwise
    coupling decomposition, top single swaps, and greedy swap-chain critique.
    The team must be exactly 6 mons; no slot-filling or sampling.
    """
    st.markdown(
        rh.page_title(
            eyebrow="notebook · /analysis",
            h1="Team analysis",
            right_caption=corpus_caption,
        ),
        unsafe_allow_html=True,
    )

    vocab = model.vocab
    J, h = model.J, model.h
    species_of, item_of = model.species_of, model.item_of
    team_counts = model.team_counts

    name_to_idx = {n: i for i, n in enumerate(vocab)}
    sorted_vocab = sorted(vocab, key=lambda v: -model.m[name_to_idx[v]])

    # §01 Team — slot-strip visualization above the multiselect input.
    # Same st.empty placeholder pattern as /completer so the strip
    # appears above the picker even though Streamlit renders top-down.
    section01_slot = st.empty()
    team_strip_slot = st.empty()

    team_names = st.multiselect(
        f"Your team (exactly {TEAM_SIZE})",
        sorted_vocab,
        max_selections=TEAM_SIZE,
        placeholder="Choose your team of 6",
        label_visibility="collapsed",
        key=f"analysis_team_{phase_key}",
    )

    section01_slot.markdown(
        rh.section_label("01", f"Team · {len(team_names)} of {TEAM_SIZE} set"),
        unsafe_allow_html=True,
    )
    team_strip_slot.markdown(
        rh.slot_strip(team_names, team_size=TEAM_SIZE),
        unsafe_allow_html=True,
    )

    if len(team_names) != TEAM_SIZE:
        st.info(f"Pick {TEAM_SIZE} Pokemon to analyze (have {len(team_names)}).")
        return

    team_idx = sorted({name_to_idx[n] for n in team_names})

    # Species @ Item uniqueness check (inert under Species vocab — that vocab
    # is unique by species and item_of is all-None).
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
        "Bias Adjustment",
        options=FIELD_WEIGHT_OPTIONS,
        value=0.5,
        key=f"analysis_fw_{phase_key}",
        help=(
            "Rescales the Bias before computing Score (adj) and choosing "
            "greedy swaps. Score (raw) always uses Bias Adj. = 1. Lower "
            "values weight coupling structure more heavily relative to "
            "popularity."
        ),
    )

    # §02 Observables strip
    score_raw = -team_energy(state, J, h)
    score_adj = -team_energy(state, J, field_weight * h)
    sum_j = intra_team_sum_j(state, J)
    obs_cells: list[tuple[str, str, str | None]] = [
        ("Score (adj)", f"{score_adj:+.3f}", f"Bias Adj. = {field_weight}"),
        ("Score (raw)", f"{score_raw:+.3f}", "Bias Adj. = 1.0"),
        ("Coherence", f"{sum_j:+.3f}", "intra-team coupling"),
    ]
    corpus = nearest_observed(state, vocab, team_counts)
    if corpus is not None:
        d, c = corpus
        if d == 0:
            obs_cells.append(("Corpus", f"{c}×", "exact roster seen"))
        else:
            obs_cells.append(("Corpus", f"Δ{d} ({c})", "to nearest observed"))
    st.markdown(rh.section_label("02", "Observables"), unsafe_allow_html=True)
    st.markdown(rh.stat_strip(obs_cells), unsafe_allow_html=True)

    # §03 Pairwise coupling decomposition
    st.markdown(
        rh.section_label("03", "Pairwise coupling decomposition",
                         right=f"C({TEAM_SIZE}, 2) = 15 unordered pairs · sorted by |Coupling|"),
        unsafe_allow_html=True,
    )
    pj_rows = pairwise_j_rows(team_idx, vocab, J)
    _render_pairwise_j_html(pj_rows)

    # §04 Top single-swap suggestions from the starting team (no chaining)
    ranked_swaps = rank_single_swaps(
        J, h, team_idx, field_weight,
        species_of=species_of, item_of=item_of, top_n=TOP_SINGLE_SWAPS,
    )
    st.markdown(
        rh.section_label("04", "Top single swaps from this team",
                         right="independent one-step alternatives · ranked by ΔScore (adj)"),
        unsafe_allow_html=True,
    )
    if not ranked_swaps:
        st.info("No legal single swap exists from this team.")
    else:
        _render_swap_table_html(ranked_swaps, vocab, team_idx, team_counts)

    # §05 Greedy swap-chain critique (no pinning — full team is fair game)
    with st.spinner("Computing greedy descent..."):
        final_team, chain = greedy_optimize_chain(
            phase_key,
            tuple(team_idx), (),           # no pinning — full team is fair game
            (), field_weight, GREEDY_MAX_SWAPS,
            J, h,
            _species_of=species_of, _item_of=item_of,
        )

    def team_state(idx_iter) -> np.ndarray:
        s = np.zeros(len(vocab), dtype=bool)
        for i in idx_iter:
            s[i] = True
        return s

    start_state = team_state(team_idx)
    start_score_raw = -team_energy(start_state, J, h)
    start_score_adj = -team_energy(start_state, J, field_weight * h)
    start_sum_j = intra_team_sum_j(start_state, J)
    final_state = team_state(final_team)
    final_score_raw = -team_energy(final_state, J, h)
    final_score_adj = -team_energy(final_state, J, field_weight * h)
    final_sum_j = intra_team_sum_j(final_state, J)

    st.markdown(
        rh.section_label("05", "Greedy critique · single-swap chain",
                         right=f"converged in {len(chain)} of max {GREEDY_MAX_SWAPS} swaps"),
        unsafe_allow_html=True,
    )
    st.markdown(
        rh.stat_strip([
            ("Swaps taken", f"{len(chain)} / {GREEDY_MAX_SWAPS}", "to local min"),
            ("Δ Score (adj)", f"{final_score_adj - start_score_adj:+.3f}", "vs starting"),
            ("Δ Score (raw)", f"{final_score_raw - start_score_raw:+.3f}", "vs starting"),
            ("Δ Coherence", f"{final_sum_j - start_sum_j:+.3f}", "vs starting"),
        ]),
        unsafe_allow_html=True,
    )

    if not chain:
        st.info(
            "No improving single-swap exists — this team is a local maximum "
            "of Score (adj). Try a different Bias Adjustment to see if the "
            "model re-ranks under a different objective."
        )
        return

    _render_chain_table_html(
        team_idx, chain, vocab, J, h, field_weight, team_counts,
        start_score_adj, start_score_raw, start_sum_j,
    )


def _render_pairwise_j_html(rows: list) -> None:
    """Rich HTML pairwise-coupling decomposition table (§03 of /analysis).

    Each row: rank · pair (sprite × sprite) · |Coupling| signed-bar ·
    Coupling chip · % of total. SignedBar max is fixed at 2.5 to match
    typical coupling magnitudes; over-range values clamp visually but the
    chip still shows the true number.
    """
    head = (
        '<tr><th class="num">#</th><th>pair</th>'
        '<th class="num">Coupling</th>'
        '<th class="num">% of total</th></tr>'
    )
    body_rows: list[str] = []
    for r in rows:
        coupling_cell = (
            f'<div style="display:flex;align-items:center;gap:8px;justify-content:flex-end;">'
            f'{rh.signed_bar(r.j_value, max_value=2.5, width=80)}'
            f'{rh.score_chip(r.j_value, "signed")}'
            f'</div>'
        )
        body_rows.append(
            '<tr>'
            f'<td class="rank">{r.rank:02d}</td>'
            f'<td>{rh.pair_cell(r.name_a, r.name_b)}</td>'
            f'<td class="num">{coupling_cell}</td>'
            f'<td class="num">{r.pct_of_abs_sum * 100:.1f}%</td>'
            '</tr>'
        )
    st.markdown(
        '<table class="lab-comp-table">'
        f'<thead>{head}</thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table>',
        unsafe_allow_html=True,
    )


def _render_swap_table_html(
    ranked_swaps: list[dict], vocab: list[str], team_idx: list[int],
    team_counts: Counter,
) -> None:
    """Rich HTML top-single-swaps table (§04 of /analysis).

    Each row: rank · swap (out → in sprites) · ΔScore (adj) · ΔScore (raw) ·
    ΔCoherence chip · corpus · team-after mini-strip. The after-team is the
    starting team with this single swap applied (not chained — each row is
    evaluated independently).

    Δ columns use sign-flipped values (positive = improvement, matching the
    new Score convention) and render as signed chips with green/red color.
    """
    head_cells = [
        '<th class="num">#</th>', "<th>swap</th>",
        '<th class="num">Δ Score (adj)</th>', '<th class="num">Δ Score (raw)</th>',
        '<th class="num">Δ Coherence</th>',
        '<th class="num">corpus</th>',
        "<th>team after</th>",
    ]

    team_set = set(team_idx)
    body_rows: list[str] = []
    for rank, sw in enumerate(ranked_swaps, 1):
        # Hypothetical after-team for this single swap (not chained — each
        # row is evaluated independently from the starting team).
        after_idx = sorted((team_set - {sw["out_idx"]}) | {sw["in_idx"]})
        # ΔScore = −ΔE (sign-flipped at display so positive = improvement).
        cells = [
            f'<td class="rank">{rank:02d}</td>',
            f'<td>{rh.swap_cell(vocab[sw["out_idx"]], vocab[sw["in_idx"]])}</td>',
            f'<td class="num">{rh.score_chip(-sw["delta_E_adj"], "signed")}</td>',
            f'<td class="num">{rh.score_chip(-sw["delta_E_raw"], "signed")}</td>',
            f'<td class="num">{rh.score_chip(sw["delta_sum_j"], "signed")}</td>',
        ]
        after_state = np.zeros(len(vocab), dtype=bool)
        for i in after_idx:
            after_state[i] = True
        corpus = nearest_observed(after_state, vocab, team_counts)
        if corpus is None:
            cells.append('<td class="num"></td>')
        else:
            d, c = corpus
            cells.append(f'<td class="num">{rh.corpus_cell(d, c)}</td>')
        cells.append(f'<td>{rh.team_mini_strip([vocab[i] for i in after_idx])}</td>')
        body_rows.append(f'<tr>{"".join(cells)}</tr>')
    st.markdown(
        '<table class="lab-comp-table">'
        f'<thead><tr>{"".join(head_cells)}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table>',
        unsafe_allow_html=True,
    )


def _render_chain_table_html(
    team_idx: list[int],
    chain: list[dict],
    vocab: list[str],
    J: np.ndarray,
    h: np.ndarray,
    field_weight: float,
    team_counts: Counter,
    start_score_adj: float,
    start_score_raw: float,
    start_sum_j: float,
) -> None:
    """Rich HTML greedy-chain table (§05 of /analysis).

    Each row is the team's state **after** that swap step. Row 0 is the
    starting team; subsequent rows show ``out → in`` sprite cells plus
    after-state observables. Score columns are sign-flipped (higher = better).
    """
    def _state(idx_iter) -> np.ndarray:
        s = np.zeros(len(vocab), dtype=bool)
        for i in idx_iter:
            s[i] = True
        return s

    head_cells = ['<th class="num">#</th>', "<th>swap</th>",
                  '<th class="num">Score (adj)</th>', '<th class="num">Score (raw)</th>',
                  '<th class="num">Coherence</th>',
                  '<th class="num">corpus</th>',
                  "<th>team after</th>"]

    body_rows: list[str] = []

    # Row 0 — starting team. No swap; show italic "starting team" + the team strip.
    start_state = _state(team_idx)
    start_cells = [
        '<td class="rank">00</td>',
        '<td><span style="font-family:var(--lab-font-serif);font-style:italic;'
        'color:var(--lab-ink-muted);">starting team</span></td>',
        f'<td class="num">{start_score_adj:+.3f}</td>',
        f'<td class="num">{start_score_raw:+.3f}</td>',
        f'<td class="num">{rh.score_chip(start_sum_j, "signed")}</td>',
    ]
    s_corpus = nearest_observed(start_state, vocab, team_counts)
    if s_corpus is None:
        start_cells.append('<td class="num"></td>')
    else:
        d, c = s_corpus
        start_cells.append(f'<td class="num">{rh.corpus_cell(d, c)}</td>')
    start_cells.append(f'<td>{rh.team_mini_strip([vocab[i] for i in team_idx])}</td>')
    body_rows.append(f'<tr>{"".join(start_cells)}</tr>')

    # Subsequent rows — one per applied swap. Sign-flip energies → Scores.
    for ev in chain:
        after_state = _state(ev["team_after"])
        cells = [
            f'<td class="rank">{ev["step"]:02d}</td>',
            f'<td>{rh.swap_cell(vocab[ev["out_idx"]], vocab[ev["in_idx"]])}</td>',
            f'<td class="num">{-ev["energy_adj_after"]:+.3f}</td>',
            f'<td class="num">{-ev["energy_raw_after"]:+.3f}</td>',
            f'<td class="num">{rh.score_chip(ev["sum_j_after"], "signed")}</td>',
        ]
        a_corpus = nearest_observed(after_state, vocab, team_counts)
        if a_corpus is None:
            cells.append('<td class="num"></td>')
        else:
            d, c = a_corpus
            cells.append(f'<td class="num">{rh.corpus_cell(d, c)}</td>')
        cells.append(
            f'<td>{rh.team_mini_strip([vocab[i] for i in ev["team_after"]])}</td>'
        )
        body_rows.append(f'<tr>{"".join(cells)}</tr>')

    st.markdown(
        '<table class="lab-comp-table">'
        f'<thead><tr>{"".join(head_cells)}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table>',
        unsafe_allow_html=True,
    )


def _render_meta(phase_key: str, model: PhaseModel, corpus_caption: str) -> None:
    """Format-wide statistics derived from the fitted (J, h): summary strip,
    top features by h, top ±J pairs (synergies / exclusions), distribution
    plots for J and h. Cross-phase comparison and the J-graph figure are
    deferred to v1.1 (both want real React work to land cleanly).
    """
    st.markdown(
        rh.page_title(
            eyebrow="notebook · /meta",
            h1="Meta data",
            right_caption=corpus_caption,
        ),
        unsafe_allow_html=True,
    )

    vocab = model.vocab
    J, h = model.J, model.h
    team_counts = model.team_counts
    V = len(vocab)

    # §01 Fitted model — stat strip.
    n_corpus_teams = sum(team_counts.values()) if team_counts is not None else None
    model_label = "Species" if phase_key == "species" else "Species @ Item"
    model_sub = "PL · species" if phase_key == "species" else "PL · item-pair"

    st.markdown(rh.section_label("01", "Fitted model"), unsafe_allow_html=True)
    st.markdown(
        rh.stat_strip([
            ("Model", model_label, model_sub),
            ("Vocab", f"{V:,}", "features ≥ cutoff"),
            ("Corpus",
             f"{n_corpus_teams:,}" if n_corpus_teams is not None else "—",
             "teams"),
        ]),
        unsafe_allow_html=True,
    )

    # §02 Extreme features by Bias — +Bias on left, −Bias on right.
    st.markdown(
        rh.section_label("02", "Extreme features by Bias",
                         right=f"top {META_TOP_FEATURES} each direction · ranked by Bias"),
        unsafe_allow_html=True,
    )
    order_desc = np.argsort(-h)
    order_asc = np.argsort(h)
    # m̂ scale: cap mini-bar at the max observed marginal so all rows compare
    # cleanly across both +Bias / −Bias columns.
    max_m = float(model.m.max())
    top_pos_col, top_neg_col = st.columns(2)
    with top_pos_col:
        st.markdown('<div class="lab-subheading lab-subheading-pos">Top +Bias · most popular</div>',
                    unsafe_allow_html=True)
        _render_feature_h_table_html(order_desc[:META_TOP_FEATURES], vocab, h, model.m, max_m)
    with top_neg_col:
        st.markdown('<div class="lab-subheading lab-subheading-neg">Top −Bias · most unlikely</div>',
                    unsafe_allow_html=True)
        _render_feature_h_table_html(order_asc[:META_TOP_FEATURES], vocab, h, model.m, max_m)

    # §03 Extreme couplings — +Coupling synergies on left, −Coupling on right.
    st.markdown(
        rh.section_label("03", "Extreme couplings",
                         right=f"top {META_TOP_PAIRS} each direction · ranked by Coupling"),
        unsafe_allow_html=True,
    )
    iu, ju = np.triu_indices(V, k=1)
    j_flat = J[iu, ju]

    # On the Species @ Item vocab, two artifacts dominate the extreme
    # −Coupling list as trivial mutual exclusions: same-species pairs (one
    # mon can't hold two items) and same-item pairs (each item is unique
    # per team under VGC rules). Filter both so the table surfaces real
    # cross-species cross-item structure. Species vocab is unique by
    # species and has item_of all-None, so both masks are no-ops there.
    species_arr = np.array(model.species_of)
    item_arr = np.array([it if it is not None else "" for it in model.item_of])
    has_item = item_arr != ""
    cross_species = species_arr[iu] != species_arr[ju]
    cross_item = ~(has_item[iu] & has_item[ju] & (item_arr[iu] == item_arr[ju]))
    keep = cross_species & cross_item
    iu_v, ju_v, j_v = iu[keep], ju[keep], j_flat[keep]

    # Bar scale: max(|Coupling|) over the filtered set, so positive and
    # negative tables share a calibration.
    max_j = float(np.max(np.abs(j_v))) if j_v.size else 1.0

    pos_col, neg_col = st.columns(2)
    with pos_col:
        st.markdown('<div class="lab-subheading lab-subheading-pos">Top +Coupling · synergies</div>',
                    unsafe_allow_html=True)
        pos_order = np.argsort(-j_v)[:META_TOP_PAIRS]
        _render_extreme_pairs_html(iu_v, ju_v, j_v, pos_order, vocab, max_j)
    with neg_col:
        st.markdown('<div class="lab-subheading lab-subheading-neg">Top −Coupling · exclusions</div>',
                    unsafe_allow_html=True)
        neg_order = np.argsort(j_v)[:META_TOP_PAIRS]
        _render_extreme_pairs_html(iu_v, ju_v, j_v, neg_order, vocab, max_j)

    # §04 — distributional diagnostics. Side-by-side matplotlib panels
    # restyled with the lab palette. Bar / line color is the forest-green
    # accent; axes and zero lines use the ink-soft for low-contrast guides.
    st.markdown(rh.section_label("04", "Distributional diagnostics"), unsafe_allow_html=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": "#5b554c",
        "axes.labelcolor": "#1a1815",
        "axes.titlecolor": "#1a1815",
        "xtick.color": "#5b554c",
        "ytick.color": "#5b554c",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (ax_j, ax_h) = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.patch.set_facecolor("#faf7f0")
    for ax in (ax_j, ax_h):
        ax.set_facecolor("#faf7f0")
    ax_j.hist(j_flat, bins=80, log=True, color="#3a7d44", edgecolor="#faf7f0", linewidth=0.4)
    ax_j.set_title("Coupling Strength · off-diagonal distribution", fontsize=10, loc="left")
    ax_j.set_xlabel("Coupling Strength")
    ax_j.set_ylabel("count (log)")
    ax_j.axvline(0, color="#8b857a", lw=0.6, linestyle="--")

    h_sorted = np.sort(h)[::-1]
    ax_h.plot(h_sorted, color="#3a7d44", linewidth=1.4)
    ax_h.set_title("Bias · sorted descending", fontsize=10, loc="left")
    ax_h.set_xlabel("feature rank")
    ax_h.set_ylabel("Bias")
    ax_h.axhline(0, color="#8b857a", lw=0.6, linestyle="--")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_feature_h_table_html(
    order: np.ndarray,
    vocab: list[str],
    h: np.ndarray,
    m: np.ndarray,
    max_m: float,
) -> None:
    """Rich HTML feature-by-Bias table (§02 of /meta).

    Each row: # · feature (sprite + name + item) · Bias (signed chip) · m̂
    (pct + unipolar mini-bar). The mini-bar is scaled to ``max_m`` so the
    +Bias and −Bias tables share a horizontal calibration.
    """
    head = (
        '<tr><th class="num">#</th><th>feature</th>'
        '<th class="num">Bias</th><th class="num">m̂</th></tr>'
    )
    body_rows: list[str] = []
    for rank, i in enumerate(order, 1):
        i = int(i)
        m_i = float(m[i])
        bar = rh.mini_bar(m_i, max_m, width=80)
        body_rows.append(
            '<tr>'
            f'<td class="rank">{rank:02d}</td>'
            f'<td>{rh.inline_mon(vocab[i])}</td>'
            f'<td class="num">{rh.score_chip(float(h[i]), "signed")}</td>'
            f'<td class="num"><div class="lab-comp-freq">'
            f'<span class="lab-comp-freq-pct">{m_i * 100:.1f}%</span>{bar}'
            f'</div></td>'
            '</tr>'
        )
    st.markdown(
        '<table class="lab-comp-table">'
        f'<thead>{head}</thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table>',
        unsafe_allow_html=True,
    )


def _render_extreme_pairs_html(
    iu: np.ndarray,
    ju: np.ndarray,
    j_flat: np.ndarray,
    order: np.ndarray,
    vocab: list[str],
    max_j: float,
) -> None:
    """Rich HTML extreme-couplings table (§03 of /meta).

    Each row: # · pair (sprite × sprite) · |Coupling| signed bar · Coupling chip.
    ``max_j`` is shared between the +Coupling and −Coupling tables so the
    bars are visually comparable.
    """
    head = (
        '<tr><th class="num">#</th><th>pair</th>'
        '<th class="num">Coupling</th></tr>'
    )
    body_rows: list[str] = []
    for r, k in enumerate(order, 1):
        k = int(k)
        j_val = float(j_flat[k])
        coupling_cell = (
            f'<div style="display:flex;align-items:center;gap:8px;justify-content:flex-end;">'
            f'{rh.signed_bar(j_val, max_value=max_j, width=80)}'
            f'{rh.score_chip(j_val, "signed")}'
            f'</div>'
        )
        body_rows.append(
            '<tr>'
            f'<td class="rank">{r:02d}</td>'
            f'<td>{rh.pair_cell(vocab[int(iu[k])], vocab[int(ju[k])])}</td>'
            f'<td class="num">{coupling_cell}</td>'
            '</tr>'
        )
    st.markdown(
        '<table class="lab-comp-table">'
        f'<thead>{head}</thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
