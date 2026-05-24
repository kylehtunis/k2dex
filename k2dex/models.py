"""Inverse Ising fits.

Pseudo-likelihood fit (Phase 2 / 3): V per-spin L2-regularized logistic
regressions. Symmetrize by averaging J with its transpose. Used by
app.py:load_model_phase2, app.py:load_model_phase3, validation.ipynb's
cross-model section, and validation.ipynb's chronological split.

Gaussian / precision-matrix fit (Phase 1) lives in helpers.py since v0;
left there to avoid churn.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from .constants import PHASE2_LR_C


def fit_pl_ising(
    X: NDArray[np.integer],
    *,
    C: float = PHASE2_LR_C,
    max_iter: int = 1000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit inverse Ising (J, h) via per-spin pseudo-likelihood on a binary
    team-indicator matrix.

    Input:
        X: (n_teams, V) integer matrix with X[t, i] = 1 iff feature i appears
           in team t. Any integer dtype is accepted; cast to int32 internally.
        C: sklearn L2 inverse-strength. Lower = stronger regularization.

    Output:
        J: (V, V) symmetric float64 with zero diagonal.
        h: (V,) float64.

    For each spin i: drop column i, fit y = X[:, i] against X[:, ~i] with
    logistic regression. The intercept becomes h[i]; coefficients become row i
    of an asymmetric J_raw. Post-fit symmetrize: J = (J_raw + J_raw.T) / 2.

    Skips spins that are all-on or all-off in X (avoids degenerate logreg).
    For a skipped spin i: `h[i] = 0` exactly, and `J_asym[i, :] = 0` exactly,
    but after symmetrization `J[i, :]` may carry small (~1e-4) nonzero values
    that come from other spins' regressions assigning the constant column
    X[:, i] a small coefficient (redundancy between intercept and constant
    predictor under L2). In practice no real Pokemon is always-on across an
    entire training corpus, so the degenerate-spin path effectively never
    fires; this is just documented to avoid surprise.
    """
    X = X.astype(np.int32, copy=False)
    n, V = X.shape
    J_asym = np.zeros((V, V), dtype=np.float64)
    h = np.zeros(V, dtype=np.float64)
    for i in range(V):
        y = X[:, i]
        if y.sum() < 2 or (1 - y).sum() < 2:
            continue
        mask = np.ones(V, dtype=bool)
        mask[i] = False
        lr = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=max_iter)
        lr.fit(X[:, mask], y)
        h[i] = lr.intercept_[0]
        J_asym[i, mask] = lr.coef_[0]
    J = 0.5 * (J_asym + J_asym.T)
    np.fill_diagonal(J, 0.0)
    return J, h
