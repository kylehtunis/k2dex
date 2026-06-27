"""Swap-move MCMC and deterministic samplers for the inverse Ising model.

Three swap-MCMC variants share one `_local_swap_step` inner loop:
    - swap_mcmc           constant temperature, single chain
    - anneal_mcmc         cooling schedule, returns final state
    - parallel_tempered_mcmc  K chains, replica-exchange between adjacent T

Plus deterministic alternatives:
    - meanfield_marginals damped MF iteration
    - greedy_optimize     steepest-descent over single-swap moves

Uniqueness machinery (no-duplicate-species, no-duplicate-item) is shared
across all of them; pass `species_of=None, item_of=None` to disable (the
Phase 1 / Phase 2 case where each vocab entry is already unique by species
and items aren't tracked).
"""
from __future__ import annotations

from dataclasses import dataclass
from tqdm import tqdm

import numpy as np


# ---------- Uniqueness helpers ----------

def build_constraint_sets(
    fixed_set: set[int],
    species_of: list[str] | None,
    item_of: list[str | None] | None,
) -> tuple[set[str], set[str]]:
    """Precompute species + item sets occupied by the fixed mons.

    All swap proposals check against these to reject duplicate-species or
    duplicate-item moves. Phase 1 / Phase 2 pass `species_of=None, item_of=None`
    and both returned sets are empty.
    """
    fixed_species: set[str] = (
        {species_of[i] for i in fixed_set} if species_of is not None else set()
    )
    fixed_items: set[str] = (
        {item_of[i] for i in fixed_set if item_of[i] is not None}
        if item_of is not None else set()
    )
    return fixed_species, fixed_items


def swap_violates_uniqueness(
    i_in: int,
    out_k: int,
    on_nf: list[int],
    fixed_species: set[str],
    fixed_items: set[str],
    species_of: list[str] | None,
    item_of: list[str | None] | None,
) -> bool:
    """True iff swapping i_in into position out_k of on_nf creates a duplicate
    species or duplicate non-None item against the team (fixed + free)."""
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


def initialize_state(
    available: np.ndarray,
    n_to_fill: int,
    fixed_species: set[str],
    fixed_items: set[str],
    species_of: list[str] | None,
    item_of: list[str | None] | None,
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> np.ndarray | None:
    """Sample n_to_fill indices from `available` respecting species + item
    uniqueness against the fixed set. Greedy: shuffle `available`, walk it
    picking non-conflicting indices until full or exhausted. Retries up to
    max_attempts on failure. Returns None if no valid init exists -- e.g.,
    the user has fixed too many same-item mons.

    Falls back to uniform sampling when both lookup arrays are None (the
    Phase 1/2 case where no uniqueness check is meaningful).
    """
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


# ---------- Energy ----------

def team_energy(state_bool: np.ndarray, J: np.ndarray, h: np.ndarray) -> float:
    """Raw Ising energy H(s) = -h.s - 0.5 s'Js. Lower energy = more probable."""
    s = state_bool.astype(np.float64)
    return float(-np.dot(h, s) - 0.5 * s @ J @ s)


# ---------- Local-swap inner loop (shared core) ----------

@dataclass
class _SwapChainState:
    """Mutable per-chain state for the swap-move MCMC family.

    Used internally by swap_mcmc / anneal_mcmc / parallel_tempered_mcmc.
    """
    state: np.ndarray            # bool, length V
    state_f: np.ndarray          # float64 view of `state` for delta-H math
    on_nf: list[int]             # non-fixed mons currently on the team
    off_nf: list[int]            # non-fixed mons currently off the team
    energy: float                # H(state) under h_eff (tracked so PT can
                                 # compute beta_diff * (E_lo - E_hi) without
                                 # recomputing the energy at swap time)


def _local_swap_step(
    chain: _SwapChainState,
    *,
    J: np.ndarray,
    h_eff: np.ndarray,
    T: float,
    fixed_species: set[str],
    fixed_items: set[str],
    species_of: list[str] | None,
    item_of: list[str | None] | None,
    rng: np.random.Generator,
) -> tuple[bool, bool]:
    """One Metropolis swap proposal. Mutates `chain` in place on accept.

    Returns (proposed, accepted). `proposed=False` only when the move was
    pre-rejected by the uniqueness check (no MH proposal was actually drawn).
    Callers decide whether to count uniqueness rejections toward acceptance
    statistics; the v0 convention used in the three callers is to NOT count
    them (preserves "acceptance rate" as a measure of MH-step quality).
    """
    if not chain.off_nf or not chain.on_nf:
        return False, False
    out_k = rng.integers(len(chain.on_nf))
    in_k = rng.integers(len(chain.off_nf))
    i_out = chain.on_nf[out_k]
    i_in = chain.off_nf[in_k]
    if swap_violates_uniqueness(
        i_in, out_k, chain.on_nf, fixed_species, fixed_items, species_of, item_of,
    ):
        return False, False
    delta_H = (h_eff[i_out] - h_eff[i_in]
               + np.dot(J[i_out] - J[i_in], chain.state_f)
               + J[i_in, i_out])
    accept = delta_H <= 0 or rng.random() < np.exp(-delta_H / T)
    if accept:
        chain.state[i_out] = False
        chain.state[i_in] = True
        chain.state_f[i_out] = 0.0
        chain.state_f[i_in] = 1.0
        chain.on_nf[out_k] = i_in
        chain.off_nf[in_k] = i_out
        chain.energy += delta_H
    return True, accept


def _init_chain(
    n: int,
    fixed_set: set[int],
    available: np.ndarray,
    n_to_fill: int,
    fixed_species: set[str],
    fixed_items: set[str],
    species_of: list[str] | None,
    item_of: list[str | None] | None,
    J: np.ndarray,
    h_eff: np.ndarray,
    rng: np.random.Generator,
) -> _SwapChainState | None:
    """Build a _SwapChainState with `fixed` clamped on and `n_to_fill` free
    slots filled via `initialize_state` (uniqueness-respecting greedy retry).
    Returns None if a valid initial state can't be constructed.
    """
    state = np.zeros(n, dtype=bool)
    for i in fixed_set:
        state[i] = True
    if n_to_fill > 0:
        init = initialize_state(
            available, n_to_fill, fixed_species, fixed_items,
            species_of, item_of, rng,
        )
        if init is None:
            return None
        state[init] = True
        on_nf = [int(x) for x in init]
        off_nf = list(set(available.tolist()) - set(on_nf))
    else:
        on_nf, off_nf = [], []
    state_f = state.astype(np.float64)
    return _SwapChainState(
        state=state, state_f=state_f, on_nf=on_nf, off_nf=off_nf,
        energy=team_energy(state, J, h_eff),
    )


# ---------- The three swap-move samplers ----------

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
    """Constant-T swap-move MCMC. Returns (samples, acceptance_rate) or None
    if a valid initial state can't be constructed (over-constrained).

    `species_of` and `item_of` (Phase 3) enforce no-duplicate-species and
    no-duplicate-item via proposal rejection.
    """
    rng = np.random.default_rng(seed)
    n = len(h)
    fixed_set = set(fixed)
    excluded_set = set(excluded)
    fixed_species, fixed_items = build_constraint_sets(fixed_set, species_of, item_of)
    available = np.array(
        [i for i in range(n) if i not in fixed_set and i not in excluded_set]
    )
    n_to_fill = team_size - len(fixed_set)
    if len(available) < n_to_fill:
        return None

    h_eff = field_weight * h
    samples = np.zeros((n_steps, n), dtype=bool)

    if n_to_fill == 0:
        # Team fully determined by `fixed`; no swaps possible.
        state = np.zeros(n, dtype=bool)
        for i in fixed_set:
            state[i] = True
        samples[:] = state
        return samples, 0.0

    chain = _init_chain(
        n, fixed_set, available, n_to_fill, fixed_species, fixed_items,
        species_of, item_of, J, h_eff, rng,
    )
    if chain is None:
        return None

    accepted = 0
    proposed = 0
    for step in range(n_steps):
        prop, acc = _local_swap_step(
            chain, J=J, h_eff=h_eff, T=temperature,
            fixed_species=fixed_species, fixed_items=fixed_items,
            species_of=species_of, item_of=item_of, rng=rng,
        )
        proposed += int(prop)
        accepted += int(acc)
        samples[step] = chain.state
    return samples, (accepted / proposed if proposed else 0.0)


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
    """Single simulated-annealing run with exponential cooling from t_start
    to t_end. Returns (final_state, acceptance_rate) or None if over-constrained.

    `species_of` and `item_of` (Phase 3) reject swaps that would create
    duplicate species or duplicate non-None items; see `swap_mcmc`.
    """
    rng = np.random.default_rng(seed)
    n = len(h)
    fixed_set = set(fixed)
    excluded_set = set(excluded)
    fixed_species, fixed_items = build_constraint_sets(fixed_set, species_of, item_of)
    available = np.array(
        [i for i in range(n) if i not in fixed_set and i not in excluded_set]
    )
    n_to_fill = team_size - len(fixed_set)
    if len(available) < n_to_fill:
        return None

    h_eff = field_weight * h
    if n_to_fill == 0:
        state = np.zeros(n, dtype=bool)
        for i in fixed_set:
            state[i] = True
        return state, 0.0

    chain = _init_chain(
        n, fixed_set, available, n_to_fill, fixed_species, fixed_items,
        species_of, item_of, J, h_eff, rng,
    )
    if chain is None:
        return None

    accepted = 0
    proposed = 0
    for step in range(n_steps):
        T = t_start * (t_end / t_start) ** (step / max(n_steps - 1, 1))
        prop, acc = _local_swap_step(
            chain, J=J, h_eff=h_eff, T=T,
            fixed_species=fixed_species, fixed_items=fixed_items,
            species_of=species_of, item_of=item_of, rng=rng,
        )
        proposed += int(prop)
        accepted += int(acc)
    return chain.state, (accepted / proposed if proposed else 0.0)


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
    """Parallel-tempered MCMC. K chains run in parallel at temperatures
    `t_ladder` (sorted ascending; index 0 is the target / cold chain). Every
    `swap_interval` sweeps, propose adjacent-chain state swaps with the
    standard replica-exchange acceptance
    `min(1, exp((1/T_lo - 1/T_hi) * (E_lo - E_hi)))`. Samples are collected
    from the cold chain only, after burn-in.

    Replica-exchange swaps (between chains) don't need a uniqueness check --
    both chains are individually valid, so swapping whole states preserves
    validity.

    Returns (cold_chain_samples, mean_local_accept, mean_swap_accept) or None.
    """
    rng = np.random.default_rng(seed)
    n = len(h)
    K = len(t_ladder)
    fixed_set = set(fixed)
    excluded_set = set(excluded)
    fixed_species, fixed_items = build_constraint_sets(fixed_set, species_of, item_of)
    available = np.array(
        [i for i in range(n) if i not in fixed_set and i not in excluded_set]
    )
    n_to_fill = team_size - len(fixed_set)
    if len(available) < n_to_fill:
        return None

    h_eff = field_weight * h

    chains: list[_SwapChainState] = []
    for _ in range(K):
        c = _init_chain(
            n, fixed_set, available, n_to_fill, fixed_species, fixed_items,
            species_of, item_of, J, h_eff, rng,
        )
        if c is None:
            return None
        chains.append(c)

    samples = np.zeros((n_steps, n), dtype=bool)
    local_accept = local_propose = swap_accept = swap_propose = 0

    for step in range(n_steps):
        # One local MH move in each chain at its own temperature.
        for k in range(K):
            prop, acc = _local_swap_step(
                chains[k], J=J, h_eff=h_eff, T=t_ladder[k],
                fixed_species=fixed_species, fixed_items=fixed_items,
                species_of=species_of, item_of=item_of, rng=rng,
            )
            local_propose += int(prop)
            local_accept += int(acc)

        # Periodically propose replica-exchange swaps between adjacent T levels.
        if step > 0 and step % swap_interval == 0:
            for k in range(K - 1):
                T_lo, T_hi = t_ladder[k], t_ladder[k + 1]
                beta_diff = 1.0 / T_lo - 1.0 / T_hi  # > 0 (cold has higher beta)
                delta = beta_diff * (chains[k].energy - chains[k + 1].energy)
                swap_propose += 1
                if delta >= 0 or rng.random() < np.exp(delta):
                    chains[k], chains[k + 1] = chains[k + 1], chains[k]
                    swap_accept += 1

        samples[step] = chains[0].state

    return (
        samples[burn_in:],
        local_accept / max(local_propose, 1),
        swap_accept / max(swap_propose, 1),
    )


# ---------- Mean-field marginals ----------

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
    to 1 and `excluded` clamped to 0. Returns (marginals, valid_mask, iters)
    or None when fewer than (team_size - len(fixed)) valid candidates exist.

    `valid_mask` is True for candidates eligible to fill the remaining team
    slots, respecting Phase 3 uniqueness against the fixed mons (no candidate
    sharing a species or item with anything already pinned).

    Validated (see CLAUDE.md MF-vs-MCMC bullet) as a ranking-faithful, ~100x
    cheaper proxy for swap-MCMC at field_weight=1, T=1. At very low
    field_weight or as a team-size-constrained sampler, MF has not been
    validated and PT remains the right tool.
    """
    V = len(h)
    h_eff = field_weight * h
    fixed_mask = np.zeros(V, dtype=bool)
    fixed_mask[fixed] = True
    excluded_mask = np.zeros(V, dtype=bool)
    excluded_mask[excluded] = True

    # Uniqueness against fixed mons (Phase 3). Adds same-species and same-item
    # vocab entries to the "not a candidate" set, on top of fixed/excluded.
    fixed_species, fixed_items = build_constraint_sets(set(fixed), species_of, item_of)
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


# ---------- Greedy single-swap descent ----------

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
    Ising energy `E_adj(s) = -(fw*h).s - 0.5 s'Js`.

    At each step, evaluates every (non-pinned out-slot, valid in-candidate) swap
    in vectorized form and accepts the one with the most negative `dE_adj`.
    Stops at the first local minimum (no improving swap exists) or after
    `max_swaps` steps. Deterministic given (model, starting_team, pinned).

    Returns (final_team_idx_list, chain), where each chain entry has keys
    `step`, `out_idx`, `in_idx`, `delta_E_adj`, `energy_adj_after`,
    `energy_raw_after`, `sum_j_after`, `team_after`.
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

            # Vectorized dE_adj across all candidate in_idxs (V-length array).
            # delta = -fw*(h[in] - h[out]) - (J[in, others].sum() - J[out, others].sum())
            h_part = -fw * (h - h[out_idx])
            J_part = -(J[:, others_arr].sum(axis=1) - J[out_idx, others_arr].sum())
            delta_E_all = h_part + J_part

            # Validity mask for in_idx: not in current team, not excluded, and
            # (Phase 3) doesn't duplicate a species or item already on the team.
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


def rank_single_swaps(
    J: np.ndarray,
    h: np.ndarray,
    team: list[int],
    field_weight: float,
    *,
    species_of: list[str] | None = None,
    item_of: list[str | None] | None = None,
    top_n: int = 20,
) -> list[dict]:
    """Score every legal (out ∈ team, in ∉ team) single-swap from `team` and
    return the top `top_n` by `ΔE_adj` (ascending — most improving first).

    Unlike `greedy_optimize`, no swap is applied: every entry is evaluated
    from the same starting team. This makes the output a *menu* of independent
    one-step suggestions rather than a chain, useful for surfacing "the model's
    N biggest critiques" of an observed team.

    Each entry has keys: `out_idx`, `in_idx`, `delta_E_adj`, `delta_E_raw`,
    `delta_sum_j`.
    """
    V = len(h)
    fw = field_weight
    team_arr = np.asarray(team, dtype=np.int64)
    team_set = set(int(i) for i in team_arr)
    team_mask = np.zeros(V, dtype=bool)
    team_mask[team_arr] = True

    results: list[dict] = []

    for out_idx in team_arr:
        out_idx = int(out_idx)
        others = team_set - {out_idx}
        others_arr = np.fromiter(others, dtype=np.int64, count=len(others))

        # ΔE_adj = -fw*(h[in] - h[out]) - (J[in, others].sum() - J[out, others].sum())
        # ΔE_raw is the same formula with fw=1.
        # ΔΣ J  =  J[in, others].sum() - J[out, others].sum()  (the J-only piece)
        j_in_others = J[:, others_arr].sum(axis=1)
        j_out_others = float(J[out_idx, others_arr].sum())
        delta_sum_j_all = j_in_others - j_out_others
        delta_E_raw_all = -(h - h[out_idx]) - delta_sum_j_all
        delta_E_adj_all = -fw * (h - h[out_idx]) - delta_sum_j_all

        valid = ~team_mask.copy()
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

        for in_idx in np.where(valid)[0]:
            results.append({
                "out_idx": out_idx,
                "in_idx": int(in_idx),
                "delta_E_adj": float(delta_E_adj_all[in_idx]),
                "delta_E_raw": float(delta_E_raw_all[in_idx]),
                "delta_sum_j": float(delta_sum_j_all[in_idx]),
            })

    results.sort(key=lambda r: r["delta_E_adj"])
    return results[:top_n]


# ---------- Model-moment estimation (for Boltzmann learning) ----------

def estimate_moments(
    J: np.ndarray,
    h: np.ndarray,
    team_size: int,
    *,
    species_of: list[str] | None = None,
    item_of: list[str | None] | None = None,
    field_weight: float = 1.0,
    t_ladder: np.ndarray | None = None,
    n_steps: int = 100_000,
    burn_in: int = 10_000,
    swap_interval: int = 10,
    thin: int = 25,
    n_runs: int = 25,
    seed: int = 0,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Estimate the model's first and second moments under free generation.

    Runs `n_runs` independent parallel-tempered chains over the *unconstrained*
    team ensemble (`fixed=[]`, `excluded=[]`) at `field_weight`, pools the
    thinned post-burn-in cold-chain samples, and returns the sample marginals
    `<s_i>` and raw pair moments `<s_i s_j>`. These are the model side of the
    Boltzmann moment-matching gradient; compare against `models.empirical_moments`.

    The swap sampler's stationary distribution is exactly the constrained Gibbs
    measure `P(s) ∝ exp(-H(s))` restricted to valid teams (fixed size +
    species/item uniqueness), so these estimates and the empirical moments live
    on the same support. Generalizes the `sample_model` prototype in
    `notebooks/three_body_check.ipynb` (same default sampler budget).

    `t_ladder` defaults to `geomspace(1.0, 3.0, 8)` -- the cold chain (index 0,
    the one sampled) is at T=1 so it samples the model `P ∝ exp(-H)`; hotter
    rungs only assist mixing via replica exchange. Returns `(m, C, diag)` where
    `diag` carries `n_samples`, `local_accept`, `swap_accept`, or `None` if the
    ensemble is over-constrained (`team_size` exceeds the available features).
    """
    if t_ladder is None:
        # Cold chain MUST be at T=1 to sample the model distribution
        # P ∝ exp(-H); hotter rungs only aid mixing via replica exchange.
        t_ladder = np.geomspace(1.0, 3.0, 8)
    rng = np.random.default_rng(seed)
    pooled: list[np.ndarray] = []
    local_rates: list[float] = []
    swap_rates: list[float] = []
    for _ in tqdm(range(n_runs), desc="PT runs", disable=not progress, leave=False):
        result = parallel_tempered_mcmc(
            J, h, team_size, [], [], field_weight,
            t_ladder, n_steps, burn_in, swap_interval,
            int(rng.integers(2**31)),
            species_of=species_of, item_of=item_of,
        )
        if result is None:
            return None
        samples, local_rate, swap_rate = result
        pooled.append(samples[::thin])
        local_rates.append(local_rate)
        swap_rates.append(swap_rate)

    S = np.concatenate(pooled).astype(np.float64)
    n_samples = S.shape[0]
    m = S.mean(axis=0)
    C = (S.T @ S) / n_samples
    diag = {
        "n_samples": int(n_samples),
        "local_accept": float(np.mean(local_rates)),
        "swap_accept": float(np.mean(swap_rates)),
    }
    return m, C, diag
