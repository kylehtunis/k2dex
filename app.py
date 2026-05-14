"""Streamlit webapp for the Ising team auto-completer.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st

import helpers


DATA_PATH = "gen9championsvgc2026regma-1760.json"
TEAM_SIZE = 6
EPS = 0.01

# Log-spaced options for the two sliders. Both T and field_weight operate in
# log space (T governs Boltzmann factors exp(-ΔH/T); field_weight scales h,
# which is itself a log-odds), so linear sliders waste resolution at small
# values. field_weight includes 0.0 as a special-case for pure-pairwise mode.
TEMPERATURE_OPTIONS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
FIELD_WEIGHT_OPTIONS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
ANNEAL_T_START_OPTIONS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
ANNEAL_T_END_OPTIONS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
PT_T_MIN_OPTIONS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
PT_T_MAX_OPTIONS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


@st.cache_resource
def load_model() -> tuple[helpers.ChaosData, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load chaos data and fit Ising parameters once per session."""
    chaos = helpers.load_chaos(DATA_PATH)
    vocab = helpers.build_vocab(chaos, min_usage=0.002)
    C = helpers.build_cooccurrence(chaos, vocab)
    m, p_joint = helpers.binary_moments(chaos, vocab, C, team_size=TEAM_SIZE)
    Corr = helpers.binary_correlation(m, p_joint)
    J, _ = helpers.ising_gaussian(Corr, eps=EPS)
    h = np.log(m / (1 - m)) - J @ m
    return chaos, vocab, m, J, h


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
) -> tuple[np.ndarray, float] | None:
    """Swap-move MCMC. Returns (samples, acceptance_rate) or None if over-constrained."""
    rng = np.random.default_rng(seed)
    n = len(h)
    fixed_set = set(fixed)
    excluded_set = set(excluded)

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

    init = rng.choice(available, size=n_to_fill, replace=False)
    state[init] = True
    on_nf = list(init)
    off_nf = list(set(available.tolist()) - set(init.tolist()))
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
) -> tuple[np.ndarray, float] | None:
    """Single simulated-annealing run with exponential cooling from t_start -> t_end.
    Returns (final_state, acceptance_rate) or None if over-constrained."""
    rng = np.random.default_rng(seed)
    n = len(h)
    fixed_set = set(fixed)
    excluded_set = set(excluded)

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

    init = rng.choice(available, size=n_to_fill, replace=False)
    state[init] = True
    on_nf = list(init)
    off_nf = list(set(available.tolist()) - set(init.tolist()))
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
) -> tuple[np.ndarray, float, float] | None:
    """Parallel-tempered MCMC. K chains run in parallel at temperatures `t_ladder`
    (sorted ascending, so index 0 is the target/cold chain). Every `swap_interval`
    sweeps, propose adjacent-chain state swaps with the standard replica-exchange
    acceptance `min(1, exp((1/T_lo - 1/T_hi)(E_lo - E_hi)))`. Samples are collected
    from the cold chain only after burn-in.

    Returns (cold_chain_samples, mean_local_accept, mean_swap_accept) or None.
    """
    rng = np.random.default_rng(seed)
    n = len(h)
    K = len(t_ladder)
    fixed_set = set(fixed)
    excluded_set = set(excluded)

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
            init = rng.choice(available, size=n_to_fill, replace=False)
            st_k[init] = True
            on_nfs.append(list(init))
            off_nfs.append(list(set(available.tolist()) - set(init.tolist())))
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


@st.cache_data(show_spinner=False)
def parallel_tempered_distribution(
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
) -> tuple[list[tuple[tuple[int, ...], int]] | None, int, float, float, list[float]]:
    """Multiple PT runs, aggregated. Each run runs K chains in parallel at the
    same geometrically-spaced temperature ladder; only cold-chain samples are
    kept. Returns (distribution, n_kept_total, mean_local_accept, mean_swap_accept, ladder)."""
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
    fixed_idx_tuple: tuple[int, ...],
    excluded_idx_tuple: tuple[int, ...],
    field_weight: float,
    n_runs: int,
    n_steps: int,
    t_start: float,
    t_end: float,
    _J: np.ndarray,
    _h: np.ndarray,
) -> tuple[list[tuple[tuple[int, ...], int]] | None, float]:
    """Run `n_runs` independent annealing chains.
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
    fixed_idx_tuple: tuple[int, ...],
    excluded_idx_tuple: tuple[int, ...],
    field_weight: float,
    temperature: float,
    n_chains: int,
    n_steps: int,
    burn_in: int,
    _J: np.ndarray,
    _h: np.ndarray,
) -> tuple[list[tuple[tuple[int, ...], int]] | None, int, float]:
    """Multi-chain swap sampling. Returns (distribution, n_kept_samples, mean_accept_rate)."""
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
                           field_weight, n_steps, temperature, chain_seed)
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


def main() -> None:
    st.set_page_config(page_title="VGC team auto-completer", layout="wide")
    st.title("VGC team auto-completer")
    st.caption(
        "Inverse Ising model fit to Smogon chaos stats. Sample teams of 6 from the "
        "conditional distribution given fixed members (must appear) and excluded "
        "members (must not appear)."
    )

    chaos, vocab, m, J, h = load_model()
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
        )
        excluded_names = st.multiselect(
            "Exclude (must NOT appear)",
            sorted_vocab,
            default=[],
            placeholder="Choose Pokemon to pin at s=0",
        )

        overlap = set(fixed_names) & set(excluded_names)
        if overlap:
            st.error(f"Cannot be both fixed and excluded: {', '.join(overlap)}")
            st.stop()

        st.subheader("Mode")
        mode = st.radio(
            "Run mode",
            ["Sample distribution", "Anneal to MAP", "Parallel-tempered sample"],
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
                "true Boltzmann draws at the target T."
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
                tuple(fixed_idx), tuple(excluded_idx), field_weight, temperature,
                n_chains, n_steps, burn_in, J, h,
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

        md_lines = ["| # | % | raw E | adj E | completion |",
                    "| ---: | ---: | ---: | ---: | :--- |"]
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
            md_lines.append(
                f"| {rank} | {prob_pct:.2f}% | {raw_E:+.3f} | {adj_E:+.3f} | {names} |"
            )

        st.markdown("\n".join(md_lines))
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
        )

    elif mode == "Anneal to MAP":
        with st.spinner(f"Annealing {n_runs} runs × {anneal_steps} steps each..."):
            results, accept_rate = run_anneals(
                tuple(fixed_idx), tuple(excluded_idx), field_weight,
                n_runs, anneal_steps, t_start, t_end, J, h,
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

        md_lines = ["| # | runs | raw E | adj E | completion |",
                    "| ---: | ---: | ---: | ---: | :--- |"]
        for rank, (comp, count) in enumerate(results, 1):
            state = np.zeros(len(vocab), dtype=bool)
            for i in fixed_idx:
                state[i] = True
            for i in comp:
                state[i] = True
            raw_E = team_energy(state, J, h)
            adj_E = team_energy(state, J, field_weight * h)
            names = ", ".join(vocab[i] for i in comp)
            md_lines.append(
                f"| {rank} | {count}/{n_runs} | {raw_E:+.3f} | {adj_E:+.3f} | {names} |"
            )

        st.markdown("\n".join(md_lines))
        st.caption(
            "**runs** — how many independent annealings converged to this team. "
            "A single dominant team indicates a sharp / unimodal landscape; multiple "
            "distinct teams indicate a shallow / multimodal one. "
            "**raw E** is the data-calibrated energy `-h·s - 0.5 s'Js`; "
            "**adj E** is `-(field_weight·h)·s - 0.5 s'Js`, which is what the annealer "
            "actually minimizes. Lower adj E = the energy basin the annealer settled into."
        )

    else:  # Parallel-tempered sample
        with st.spinner(f"PT: {pt_n_runs} runs × {pt_K} chains × {pt_n_steps} sweeps..."):
            dist, n_kept, mh_rate, swap_rate, ladder = parallel_tempered_distribution(
                tuple(fixed_idx), tuple(excluded_idx), field_weight,
                pt_t_min, pt_t_max, pt_K, pt_n_runs, pt_n_steps,
                pt_burn_in, pt_swap_interval, J, h,
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

        md_lines = ["| # | % | raw E | adj E | completion |",
                    "| ---: | ---: | ---: | ---: | :--- |"]
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
            md_lines.append(
                f"| {rank} | {prob_pct:.2f}% | {raw_E:+.3f} | {adj_E:+.3f} | {names} |"
            )

        st.markdown("\n".join(md_lines))
        st.caption(
            "**%** — Boltzmann probability at the target T, estimated from cold-chain samples. "
            "Unlike single-chain sampling at low T, this **should be monotonically ordered with "
            "adj E** (modulo Monte Carlo noise) — that's the whole point of parallel tempering. "
            "If you see non-monotonicity, either chains haven't equilibrated (run more sweeps) "
            "or the swap rate is too low (raise K or lower t_max). "
            "**raw E** / **adj E** are the energies with full / rescaled h, same definitions as "
            "the other modes."
        )


if __name__ == "__main__":
    main()
