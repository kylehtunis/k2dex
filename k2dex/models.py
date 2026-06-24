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

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit


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
