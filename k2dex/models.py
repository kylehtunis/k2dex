"""Inverse Ising fits.

Pseudo-likelihood fit: V per-spin L2-regularized logistic regressions.
Symmetrize by averaging J with its transpose. Used by the species /
species+item loaders and the validation notebooks.

The per-spin regression is solved directly with `scipy.optimize.minimize`
(L-BFGS-B) on the penalized logistic objective. This is what lets the penalty
be re-centered on an informative prior (`prior_J` / `prior_h`) for
cross-regulation warm-starting; an off-the-shelf logistic regressor only
shrinks toward zero. With no prior the objective is identical to the standard
L2 logistic fit (penalty centered at zero, intercept unpenalized).

Gaussian / precision-matrix fit lives in helpers.py.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from .sampling import initialize_state

try:  # progress bar is optional; the fit runs fine without it
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


def _fit_one_spin(
    X_design: sp.csr_matrix,
    y: NDArray[np.float64],
    w: NDArray[np.float64],
    *,
    C: float,
    max_iter: int,
    w_prior: NDArray[np.float64],
    c_prior: float,
    pen_c: float,
) -> tuple[NDArray[np.float64], float]:
    """One per-spin L2 logistic regression with a prior-centered penalty.

    Minimizes ``0.5*||coef - w_prior||^2 + 0.5*pen_c*(intercept - c_prior)^2 +
    C * sum_t w_t * logloss_t`` over (coef, intercept). With ``w_prior = 0``,
    ``c_prior = 0`` and ``pen_c = 0`` this is exactly the objective a standard
    L2 logistic regressor (lbfgs solver) optimizes: penalty centered at zero,
    intercept unpenalized, loss scaled by ``C = 1/lambda``.

    Returns (coef, intercept).
    """
    p = w_prior.shape[0]
    Xt = X_design.T.tocsr()

    def objective(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        coef = theta[:p]
        intercept = float(theta[p])
        z = X_design @ coef + intercept
        # log(1 + exp(z)) - y*z, stable via logaddexp(0, z).
        loss = C * float(np.dot(w, np.logaddexp(0.0, z) - y * z))
        d_coef = coef - w_prior
        d_int = pen_c * (intercept - c_prior)
        reg = 0.5 * float(np.dot(d_coef, d_coef)) + 0.5 * pen_c * (intercept - c_prior) ** 2
        resid = w * (expit(z) - y)
        grad = np.empty(p + 1, dtype=np.float64)
        grad[:p] = d_coef + C * (Xt @ resid)
        grad[p] = d_int + C * float(resid.sum())
        return loss + reg, grad

    theta0 = np.empty(p + 1, dtype=np.float64)
    theta0[:p] = w_prior
    theta0[p] = c_prior
    res = minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": 1e-6, "ftol": 1e-12},
    )
    return res.x[:p], float(res.x[p])


def fit_pl_ising(
    X: NDArray[np.integer],
    *,
    C: float = 0.1,
    max_iter: int = 1000,
    sample_weight: NDArray[np.floating] | None = None,
    prior_J: NDArray[np.floating] | None = None,
    prior_h: NDArray[np.floating] | None = None,
    intercept_prior_weight: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit inverse Ising (J, h) via per-spin pseudo-likelihood on a binary
    team-indicator matrix.

    Input:
        X: (n_teams, V) integer matrix with X[t, i] = 1 iff feature i appears
           in team t. Any integer dtype is accepted; cast to int32 internally.
        C: L2 inverse-strength (C = 1/lambda). Lower = stronger regularization.
        sample_weight: optional (n_teams,) nonnegative per-team weights applied
           to each per-spin regression. The degenerate-spin skip then also
           requires >= 2 units of weighted mass in each class, so weights
           should be normalized to mean ~1 (`loaders.team_weights` guarantees
           this) to keep that threshold on the same scale as raw counts.
        prior_J, prior_h: optional (V, V) symmetric prior couplings and (V,)
           prior fields, on the SAME vocab/index order as X. When given, each
           spin's L2 penalty is re-centered on the prior instead of zero (and
           the intercept is shrunk toward prior_h, see intercept_prior_weight),
           so a feature with thin or no evidence in X retains its prior value
           rather than collapsing to zero. The caller is responsible for
           aligning a previous model's (J, h) onto X's vocab (zeros for features
           the prior did not contain). Must be supplied together.
        intercept_prior_weight: relative strength with which the intercept is
           shrunk toward prior_h, as a multiple of the coupling-penalty strength
           (1.0 = shrink the bias as hard as each coupling toward its prior;
           0.0 = leave the bias free). Ignored when no prior is given.

    Output:
        J: (V, V) symmetric float64 with zero diagonal.
        h: (V,) float64.

    For each spin i: drop column i, fit y = X[:, i] against X[:, ~i] with
    L2 logistic regression. The intercept becomes h[i]; coefficients become row
    i of an asymmetric J_raw. Post-fit symmetrize: J = (J_raw + J_raw.T) / 2.

    Degenerate spins (all-on / all-off, or < 2 units of weighted mass in either
    class) skip the regression. Without a prior such a spin gets h[i] = 0 and a
    zero J row (after symmetrization J[i, :] may still carry small ~1e-4 entries
    from other spins' regressions assigning the constant column a coefficient).
    With a prior, a degenerate spin falls back to the prior: h[i] = prior_h[i]
    and J_raw[i, :] = prior_J[i, :]. On a real corpus no feature is always-on,
    so the degenerate path effectively fires only for empty-evidence features.
    """
    X = X.astype(np.int32, copy=False)
    n, V = X.shape
    w: NDArray[np.float64]
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"sample_weight shape {w.shape} != ({n},)")
        if np.any(w < 0):
            raise ValueError("sample_weight must be nonnegative")
    else:
        w = np.ones(n, dtype=np.float64)

    has_prior = prior_J is not None or prior_h is not None
    if has_prior:
        if prior_J is None or prior_h is None:
            raise ValueError("prior_J and prior_h must be supplied together")
        prior_J = np.asarray(prior_J, dtype=np.float64)
        prior_h = np.asarray(prior_h, dtype=np.float64)
        if prior_J.shape != (V, V):
            raise ValueError(f"prior_J shape {prior_J.shape} != ({V}, {V})")
        if prior_h.shape != (V,):
            raise ValueError(f"prior_h shape {prior_h.shape} != ({V},)")
    pen_c = intercept_prior_weight if has_prior else 0.0

    # X has exactly TEAM_SIZE ones per row, so the per-spin gradients run via
    # sparse matvec -- far cheaper than dense at corpus scale. CSC makes the
    # per-spin column slice cheap; each spin's design is then taken to CSR.
    X_cols = sp.csc_matrix(X)
    J_asym = np.zeros((V, V), dtype=np.float64)
    h = np.zeros(V, dtype=np.float64)
    for i in range(V):
        mask = np.ones(V, dtype=bool)
        mask[i] = False
        if prior_J is not None and prior_h is not None:
            w_prior_i = prior_J[i, mask]
            c_prior_i = float(prior_h[i])
        else:
            w_prior_i = np.zeros(V - 1)
            c_prior_i = 0.0

        y = X[:, i].astype(np.float64)
        degenerate = (
            y.sum() < 2
            or (1.0 - y).sum() < 2
            or w[y == 1].sum() < 2.0
            or w[y == 0].sum() < 2.0
        )
        if degenerate:
            # No (usable) evidence for this spin: keep the prior, or zero.
            if has_prior:
                h[i] = c_prior_i
                J_asym[i, mask] = w_prior_i
            continue

        X_design = X_cols[:, mask].tocsr().astype(np.float64)
        coef, intercept = _fit_one_spin(
            X_design,
            y,
            w,
            C=C,
            max_iter=max_iter,
            w_prior=w_prior_i,
            c_prior=c_prior_i,
            pen_c=pen_c,
        )
        h[i] = intercept
        J_asym[i, mask] = coef
    J = 0.5 * (J_asym + J_asym.T)
    np.fill_diagonal(J, 0.0)
    return J, h


def empirical_moments(
    X: NDArray[np.integer],
    sample_weight: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Weighted first and (raw) second moments of a binary team-indicator matrix.

    These are the fit targets for Boltzmann (moment-matching) learning: the
    constrained-MaxEnt fit drives the model's ``<s_i>`` and ``<s_i s_j>`` to
    equal the data's.

    Input:
        X: (n_teams, V) binary matrix, X[t, i] = 1 iff feature i is on team t.
        sample_weight: optional (n_teams,) nonnegative weights. Normalized
           internally (only relative weights matter for moments), so the scale
           is irrelevant here -- unlike `fit_pl_ising`, where mean-1 matters.

    Output:
        m: (V,) weighted marginals ``<s_i> = sum_t w_t X[t,i] / sum_t w_t``.
        C: (V, V) raw weighted second moments ``<s_i s_j> = sum_t w_t X[t,i]
           X[t,j] / sum_t w_t``. The diagonal is ``C[i,i] = m[i]`` since X is
           binary. NOT the connected covariance (subtract ``outer(m, m)`` for
           that); moment matching uses the raw form.
    """
    Xf = X.astype(np.float64, copy=False)
    n = Xf.shape[0]
    if sample_weight is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"sample_weight shape {w.shape} != ({n},)")
        if np.any(w < 0):
            raise ValueError("sample_weight must be nonnegative")
    w_sum = w.sum()
    m = (w @ Xf) / w_sum
    C = (Xf.T * w) @ Xf / w_sum
    return m, C


def _group_ids(
    labels: list | None, V: int, *, none_sentinel: bool,
) -> NDArray[np.int64] | None:
    """Map a per-feature label list (species or item) to contiguous int ids for
    vectorized uniqueness checks. With `none_sentinel`, a `None` label maps to
    -1 (an item-less feature, exempt from the item-uniqueness check)."""
    if labels is None:
        return None
    uniq: dict[object, int] = {}
    ids = np.empty(V, dtype=np.int64)
    for i, lab in enumerate(labels):
        if none_sentinel and lab is None:
            ids[i] = -1
            continue
        if lab not in uniq:
            uniq[lab] = len(uniq)
        ids[i] = uniq[lab]
    return ids


def _candidate_invalid(
    cand: NDArray[np.int64],
    team: NDArray[np.int64],
    pos_mask: NDArray[np.bool_],
    species_id: NDArray[np.int64] | None,
    item_id: NDArray[np.int64] | None,
) -> NDArray[np.bool_]:
    """Per-chain mask: True where `cand[c]` cannot legally enter team `c`
    (already on the team, or duplicates a species/item of a *retained* member).
    `pos_mask[c]` is False at the out-slot so the leaving member is ignored."""
    invalid = (team == cand[:, None]).any(axis=1)
    if species_id is not None:
        cs = species_id[cand]
        team_sp = species_id[team]
        invalid = invalid | ((team_sp == cs[:, None]) & pos_mask).any(axis=1)
    if item_id is not None:
        ci = item_id[cand]
        team_it = item_id[team]
        conf = ((team_it == ci[:, None]) & (team_it >= 0) & pos_mask).any(axis=1)
        invalid = invalid | (conf & (ci >= 0))
    return invalid


def _batched_swap_sweep(
    team: NDArray[np.int64],
    J: NDArray[np.float64],
    h_eff: NDArray[np.float64],
    species_id: NDArray[np.int64] | None,
    item_id: NDArray[np.int64] | None,
    rng: np.random.Generator,
    *,
    temp: NDArray[np.float64] | float = 1.0,
    max_tries: int = 16,
) -> float:
    """One Metropolis swap proposal per chain, vectorized across the whole bank.

    `team` is (n_chains, team_size) of on-feature indices and is mutated in
    place on accept. Mirrors `sampling._local_swap_step`: the leaving member is
    `team[c, p]`, an entering candidate is drawn by vectorized rejection
    (uniqueness-respecting), and the energy delta uses the same
    `ΔH = h_eff[out] - h_eff[in] + Σ_{j∈team, j≠out}(J[out,j] - J[in,j])`,
    accepted with `exp(-ΔH / temp)`. `temp` is a scalar or a per-chain array
    (the latter lets one bank hold replicas at several temperatures). Returns
    the acceptance fraction over chains.
    """
    n_chains, k = team.shape
    V = h_eff.shape[0]
    rows = np.arange(n_chains)
    p = rng.integers(k, size=n_chains)
    i_out = team[rows, p]

    pos_mask = np.ones((n_chains, k), dtype=bool)
    pos_mask[rows, p] = False

    cand = rng.integers(V, size=n_chains)
    unresolved = np.ones(n_chains, dtype=bool)
    for _ in range(max_tries):
        invalid = _candidate_invalid(cand, team, pos_mask, species_id, item_id)
        unresolved &= invalid
        if not unresolved.any():
            break
        cand[unresolved] = rng.integers(V, size=int(unresolved.sum()))
    valid = ~unresolved

    diff = J[i_out[:, None], team] - J[cand[:, None], team]
    diff[rows, p] = 0.0  # exclude the leaving member (clean swap form)
    delta_H = h_eff[i_out] - h_eff[cand] + diff.sum(axis=1)
    accept = valid & (
        (delta_H <= 0.0)
        | (rng.random(n_chains) < np.exp(-np.clip(delta_H / temp, 0.0, 700.0)))
    )
    team[rows[accept], p[accept]] = cand[accept]
    return float(accept.mean())


def _bank_energies(
    team: NDArray[np.int64], J: NDArray[np.float64], h_eff: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Raw energy `H = -h_eff·s - 0.5 s'Js` of every replica in a tempered bank
    `team` of shape (n_temps, n_chains, team_size). Sparse over on-bits."""
    n_temps, n_chains, k = team.shape
    flat = team.reshape(n_temps * n_chains, k)
    h_term = -h_eff[flat].sum(axis=1)
    Jg = J[flat[:, :, None], flat[:, None, :]]  # (N, k, k); J diagonal is 0
    pair = -0.5 * Jg.sum(axis=(1, 2))
    return (h_term + pair).reshape(n_temps, n_chains)


def _replica_exchange(
    team: NDArray[np.int64],
    energies: NDArray[np.float64],
    temps: NDArray[np.float64],
    rng: np.random.Generator,
) -> float:
    """Propose replica-exchange swaps between adjacent temperature levels of a
    tempered bank `team` (n_temps, n_chains, team_size), per the standard
    `min(1, exp((β_lo - β_hi)(E_lo - E_hi)))`. Mutates `team` in place; returns
    the exchange acceptance fraction."""
    n_temps, n_chains, _ = team.shape
    acc = prop = 0
    for kk in range(n_temps - 1):
        beta_diff = 1.0 / temps[kk] - 1.0 / temps[kk + 1]  # > 0 (cold higher β)
        delta = beta_diff * (energies[kk] - energies[kk + 1])
        swap = (delta >= 0.0) | (rng.random(n_chains) < np.exp(np.minimum(delta, 0.0)))
        idx = np.where(swap)[0]
        if idx.size:
            tmp = team[kk, idx].copy()
            team[kk, idx] = team[kk + 1, idx]
            team[kk + 1, idx] = tmp
            energies[kk, idx], energies[kk + 1, idx] = (
                energies[kk + 1, idx].copy(), energies[kk, idx].copy())
        acc += int(idx.size)
        prop += n_chains
    return acc / prop if prop else 0.0


def _advance_bank(
    team: NDArray[np.int64],
    J: NDArray[np.float64],
    h_eff: NDArray[np.float64],
    temps: NDArray[np.float64],
    species_id: NDArray[np.int64] | None,
    item_id: NDArray[np.int64] | None,
    rng: np.random.Generator,
    *,
    n_sweeps: int,
    swap_interval: int,
) -> tuple[float, float]:
    """Advance a tempered persistent bank `n_sweeps` swap steps (each level at
    its own temperature), interleaving replica exchange every `swap_interval`
    sweeps. `team` is (n_temps, n_chains, team_size); returns (mean local-swap
    acceptance, mean replica-exchange acceptance). With n_temps==1 it degenerates
    to a single-temperature PCD bank (no exchange)."""
    n_temps, n_chains, k = team.shape
    flat = team.reshape(n_temps * n_chains, k)  # view: swaps write through
    temp_row = np.repeat(temps, n_chains)
    local_sum = 0.0
    swap_sum = 0.0
    swaps_done = 0
    for sw in range(n_sweeps):
        local_sum += _batched_swap_sweep(
            flat, J, h_eff, species_id, item_id, rng, temp=temp_row)
        if n_temps > 1 and swap_interval and (sw + 1) % swap_interval == 0:
            energies = _bank_energies(team, J, h_eff)
            swap_sum += _replica_exchange(team, energies, temps, rng)
            swaps_done += 1
    return local_sum / n_sweeps, (swap_sum / swaps_done if swaps_done else 0.0)


def _batched_moments(
    team: NDArray[np.int64], V: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate `<s_i>` and raw `<s_i s_j>` from the chain bank `team`
    ((n_chains, team_size) on-indices). Sparse: only the `team_size²`
    co-occurring pairs per chain are touched, so cost is O(n_chains·team_size²),
    independent of V. The diagonal of C equals m (binary features)."""
    n_chains, k = team.shape
    m = np.bincount(team.ravel(), minlength=V).astype(np.float64) / n_chains
    a = np.broadcast_to(team[:, :, None], (n_chains, k, k)).reshape(-1)
    b = np.broadcast_to(team[:, None, :], (n_chains, k, k)).reshape(-1)
    C = np.zeros((V, V), dtype=np.float64)
    np.add.at(C, (a, b), 1.0)
    C /= n_chains
    return m, C


def fit_boltzmann_ising(
    X: NDArray[np.integer],
    *,
    team_size: int,
    sample_weight: NDArray[np.floating] | None = None,
    init_J: NDArray[np.floating] | None = None,
    init_h: NDArray[np.floating] | None = None,
    species_of: list[str] | None = None,
    item_of: list[str | None] | None = None,
    field_weight: float = 1.0,
    reg: str = "l2",
    reg_lambda: float = 1e-3,
    support_mask: NDArray[np.bool_] | None = None,
    n_iters: int = 500,
    lr: float = 0.05,
    lr_final: float | None = None,
    avg_last: int = 0,
    n_chains: int = 200,
    n_sweeps: int = 50,
    n_burn: int = 200,
    n_temps: int = 1,
    t_max: float = 3.0,
    swap_interval: int = 10,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    seed: int = 0,
    progress: bool = True,
    callback: Callable[[int, dict], None] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict]:
    """Constrained-MaxEnt (Boltzmann) fit of `(J, h)` by moment matching.

    Unlike `fit_pl_ising` (which matches per-spin *conditionals*), this drives
    the model's first and second *moments* to equal the data's by stochastic
    gradient ascent on the regularized log-likelihood, with the model moments
    estimated from a persistent bank of swap-move MCMC chains (persistent
    contrastive divergence). The chains sample exactly the constrained ensemble
    `P(s) ∝ exp(-H(s))` over valid teams (fixed size + species/item uniqueness),
    so a converged fit reproduces `<s_i>` and `<s_i s_j>` of the corpus -- the
    precondition a Schneidman-style three-body test requires.

    Gradient (ascent): `g_h = <s>_data - <s>_model`, `g_J = <ssᵀ>_data -
    <ssᵀ>_model`. The regularizer is applied toward zero -- `l2` adds `-λθ` to
    the gradient, `l1` adds `-λ·sign(θ)` (subgradient). On the fixed-size
    manifold `(J,h)` is gauge-degenerate (`h→h+c·1`, `J_ij→J_ij+a_i+a_j` leave
    `P` unchanged); the non-gauge-invariant regularizer selects the minimum-norm
    representative, so reg also fixes the gauge. Keep `reg_lambda > 0`.

    Input:
        X: (n_teams, V) binary team-indicator matrix (as for `fit_pl_ising` /
           `empirical_moments`).
        team_size: number of features per team (the fixed-size constraint).
        sample_weight: optional (n_teams,) weights for the empirical moments.
        init_J, init_h: warm-start (e.g. the PL fit). Default zeros. `init_J` is
           also the frozen value for any coupling masked out by `support_mask`.
        species_of, item_of: uniqueness lookups; passed to the sampler. `None`
           disables the corresponding uniqueness constraint.
        field_weight: scales `h` inside the sampler's energy (`h_eff = fw*h`).
           Leave at 1.0 to match moments at full field strength.
        reg: "l2" or "l1". reg_lambda: regularization strength (toward zero).
        support_mask: optional (V, V) symmetric bool. Only couplings where True
           are fit; the rest stay at `init_J` (typically used to fit only pairs
           with enough corpus co-occurrence). Default: all off-diagonal pairs.
        n_iters, lr: gradient steps and Adam step size.
        lr_final: if set, cosine-anneal the step size from `lr` to `lr_final`
           over the run. Shrinks the stochastic-gradient "noise ball" near the
           optimum -- the dominant source of residual error here is that ball
           (gradient noise scales ~1/n_chains; a fixed lr orbits the optimum
           rather than settling), so decay tightens the fit. None = constant lr.
        avg_last: if > 0, return the **iterate average** of `(J, h)` over the
           last `avg_last` gradient steps instead of the final snapshot. Polyak
           averaging cancels the zero-mean orbit directly and is the cheapest
           large win against gradient noise; combine with `lr_final` and a large
           `n_chains`. 0 = return the final iterate.
        n_chains, n_sweeps: cold-level bank size (= cold samples per snapshot)
           and local swaps per chain per step. The bank is stepped fully
           vectorized (batched NumPy) and moments are accumulated sparsely over
           each team's on-bits, so per-step cost is independent of V. Note the
           per-iteration `max_resid_*` is a single `n_chains`-sample estimate and
           is mostly Monte-Carlo noise (max over V features); `mean_resid_*` is
           the reliable in-loop signal. Verify true convergence by re-estimating
           the fitted model's moments with many samples (`sampling.estimate_moments`)
           -- on the M-A corpus that drops the worst-feature marginal error from
           ~0.34 (PL) to ~0.05.
        n_burn: swaps per chain to mix the bank under the warm-start params
           before the first gradient step.
        n_temps, t_max, swap_interval: optional parallel tempering for the
           persistent bank -- `n_temps` replicas per chain on a geometric ladder
           from T=1 (cold, the level sampled for moments; it MUST be 1 to sample
           `P ∝ exp(-H)`) to `t_max` (hot, mixing only), replica exchange every
           `swap_interval` sweeps. **Defaults off (`n_temps=1`):** on this corpus
           single-temperature PCD already converges and tempering didn't improve
           it (the distribution isn't mode-trapped at this fit quality), at
           several times the cost. Reach for `n_temps>1` only if you hit a
           genuinely multimodal model where the cold chain gets stuck.
        progress: show a tqdm progress bar (falls back to silent if tqdm is
           unavailable).

    Output:
        J: (V, V) symmetric float64, zero diagonal.
        h: (V,) float64.
        history: dict with per-iteration `max_resid_m`, `max_resid_C`,
           `mean_resid_m`, `mean_resid_C` (residuals over fit entries) plus
           sampler `local_accept` / `swap_accept`. Use it to confirm residuals
           fall to the MCMC noise floor.
    """
    if reg not in ("l1", "l2"):
        raise ValueError(f"reg must be 'l1' or 'l2', got {reg!r}")
    X = X.astype(np.int32, copy=False)
    V = X.shape[1]

    m_data, C_data = empirical_moments(X, sample_weight)

    J = (np.zeros((V, V)) if init_J is None
         else np.array(init_J, dtype=np.float64, copy=True))
    h = (np.zeros(V) if init_h is None
         else np.array(init_h, dtype=np.float64, copy=True))
    np.fill_diagonal(J, 0.0)

    if support_mask is None:
        mask = ~np.eye(V, dtype=bool)
    else:
        mask = np.asarray(support_mask, dtype=bool) & ~np.eye(V, dtype=bool)
        mask |= mask.T  # enforce symmetry

    # Persistent tempered bank over the free ensemble: (n_temps, n_chains,
    # team_size) of on-feature indices. Level 0 (T=1) is the one sampled for
    # moments; hotter levels exist only to mix it via replica exchange. Built
    # once (greedy, uniqueness-respecting), then stepped vectorized.
    rng = np.random.default_rng(seed)
    available = np.arange(V)
    temps = np.geomspace(1.0, t_max, n_temps)
    team = np.empty((n_temps, n_chains, team_size), dtype=np.int64)
    for t in range(n_temps):
        for c in range(n_chains):
            init = initialize_state(
                available, team_size, set(), set(), species_of, item_of, rng,
            )
            if init is None:
                raise ValueError("over-constrained: cannot initialize a valid team")
            team[t, c] = init
    species_id = _group_ids(species_of, V, none_sentinel=False)
    item_id = _group_ids(item_of, V, none_sentinel=True)

    # Mix the bank under the warm-start params before the first gradient step.
    h_eff = field_weight * h
    _advance_bank(team, J, h_eff, temps, species_id, item_id, rng,
                  n_sweeps=n_burn, swap_interval=swap_interval)

    # Adam state.
    m_h = np.zeros(V); v_h = np.zeros(V)
    m_J = np.zeros((V, V)); v_J = np.zeros((V, V))
    b1, b2 = adam_betas
    # Polyak iterate-average accumulators (the returned estimate when avg_last>0).
    J_acc = np.zeros((V, V)); h_acc = np.zeros(V); n_acc = 0
    history: dict[str, list[float]] = {
        "max_resid_m": [], "max_resid_C": [],
        "mean_resid_m": [], "mean_resid_C": [],
        "local_accept": [], "swap_accept": [],
    }

    iters = range(1, n_iters + 1)
    bar = _tqdm(iters, desc="boltzmann fit") if (progress and _tqdm is not None) else None
    for it in (bar if bar is not None else iters):
        lr_t = lr if lr_final is None else (
            lr_final + 0.5 * (lr - lr_final) * (1.0 + np.cos(np.pi * (it - 1) / max(n_iters - 1, 1))))
        h_eff = field_weight * h
        acc, swap_acc = _advance_bank(
            team, J, h_eff, temps, species_id, item_id, rng,
            n_sweeps=n_sweeps, swap_interval=swap_interval)
        m_model, C_model = _batched_moments(team[0], V)  # cold level (T=1)

        # Ascent gradients on the regularized log-likelihood.
        g_h = m_data - m_model
        g_J = C_data - C_model
        if reg == "l2":
            g_h = g_h - reg_lambda * h
            g_J = g_J - reg_lambda * J
        else:  # l1 subgradient
            g_h = g_h - reg_lambda * np.sign(h)
            g_J = g_J - reg_lambda * np.sign(J)
        np.fill_diagonal(g_J, 0.0)
        g_J = np.where(mask, g_J, 0.0)
        g_J = 0.5 * (g_J + g_J.T)  # keep the J update symmetric

        # Adam step (ascent).
        m_h = b1 * m_h + (1 - b1) * g_h
        v_h = b2 * v_h + (1 - b2) * g_h**2
        m_J = b1 * m_J + (1 - b1) * g_J
        v_J = b2 * v_J + (1 - b2) * g_J**2
        bc1 = 1 - b1**it
        bc2 = 1 - b2**it
        h = h + lr_t * (m_h / bc1) / (np.sqrt(v_h / bc2) + adam_eps)
        J = J + lr_t * (m_J / bc1) / (np.sqrt(v_J / bc2) + adam_eps)
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)

        if avg_last and it > n_iters - avg_last:
            J_acc += J; h_acc += h; n_acc += 1

        # Residual diagnostics over the fit entries only.
        rm = m_data - m_model
        rC = (C_data - C_model)[mask]
        history["max_resid_m"].append(float(np.max(np.abs(rm))))
        history["max_resid_C"].append(float(np.max(np.abs(rC))) if rC.size else 0.0)
        history["mean_resid_m"].append(float(np.mean(np.abs(rm))))
        history["mean_resid_C"].append(float(np.mean(np.abs(rC))) if rC.size else 0.0)
        history["local_accept"].append(acc)
        history["swap_accept"].append(swap_acc)
        if bar is not None:
            bar.set_postfix(
                resid_m=f"{history['max_resid_m'][-1]:.4f}",
                resid_C=f"{history['max_resid_C'][-1]:.4f}",
                acc=f"{acc:.2f}", xswap=f"{swap_acc:.2f}",
            )
        if callback is not None:
            callback(it, {key: val[-1] for key, val in history.items()})

    if n_acc:
        J = J_acc / n_acc
        h = h_acc / n_acc
    return J, h, history
