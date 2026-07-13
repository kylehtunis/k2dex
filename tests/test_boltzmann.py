"""Validation of the Boltzmann (moment-matching) learning pipeline.

The MCMC pieces are stochastic, so these tests pin seeds and check against the
*exact* moments of a tiny constrained ensemble computed by brute-force
enumeration. Tolerances are loose enough to absorb sampling noise but tight
enough that a real regression (wrong temperature, broken uniqueness, sign error
in the gradient) trips them.
"""
import unittest
from itertools import combinations

import numpy as np

from k2dex.models import (
    _batched_swap_sweep,
    _group_ids,
    empirical_moments,
    fit_boltzmann_ising,
)
from k2dex.sampling import estimate_moments


def _valid_teams(V, k, species_of=None, item_of=None):
    """All k-subsets of range(V) that respect species/item uniqueness."""
    teams = []
    for t in combinations(range(V), k):
        if species_of is not None:
            sp = [species_of[i] for i in t]
            if len(set(sp)) != k:
                continue
        if item_of is not None:
            it = [item_of[i] for i in t if item_of[i] is not None]
            if len(set(it)) != len(it):
                continue
        teams.append(t)
    return teams


def _exact_moments(J, h, k, species_of=None, item_of=None):
    """Exact <s_i> and <s_i s_j> of P(s) ∝ exp(-H) over the valid k-team set."""
    V = len(h)
    teams = _valid_teams(V, k, species_of, item_of)
    e = np.empty(len(teams))
    for r, t in enumerate(teams):
        s = np.zeros(V)
        s[list(t)] = 1.0
        e[r] = -h @ s - 0.5 * s @ J @ s
    logp = -e
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    m = np.zeros(V)
    C = np.zeros((V, V))
    for pt, t in zip(p, teams):
        s = np.zeros(V)
        s[list(t)] = 1.0
        m += pt * s
        C += pt * np.outer(s, s)
    return m, C


def _onehot(t, V):
    s = np.zeros(V)
    s[list(t)] = 1.0
    return s


def _sample_corpus(J, h, k, n, seed, species_of=None, item_of=None):
    """Draw a corpus of n teams from the exact distribution as fit data."""
    rng = np.random.default_rng(seed)
    V = len(h)
    teams = _valid_teams(V, k, species_of, item_of)
    e = np.array([-h @ _onehot(t, V) - 0.5 * _onehot(t, V) @ J @ _onehot(t, V)
                  for t in teams])
    logp = -e
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    idx = rng.choice(len(teams), size=n, p=p)
    X = np.zeros((n, V), dtype=np.int8)
    for r, ti in enumerate(idx):
        X[r, list(teams[ti])] = 1
    return X


class TestEmpiricalMoments(unittest.TestCase):
    def test_matches_hand_computation(self) -> None:
        X = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.int8)
        m, C = empirical_moments(X)
        np.testing.assert_allclose(m, [3 / 4, 3 / 4, 3 / 4])
        # diagonal is the marginal; off-diagonals are co-occurrence fractions
        np.testing.assert_allclose(np.diag(C), m)
        self.assertAlmostEqual(C[0, 1], 2 / 4)  # teams 0 and 3
        self.assertAlmostEqual(C[0, 2], 2 / 4)  # teams 1 and 3

    def test_weighting(self) -> None:
        X = np.array([[1, 0], [0, 1]], dtype=np.int8)
        m, _ = empirical_moments(X, np.array([3.0, 1.0]))
        np.testing.assert_allclose(m, [0.75, 0.25])


class TestEstimateMoments(unittest.TestCase):
    def test_matches_exact_no_uniqueness(self) -> None:
        rng = np.random.default_rng(0)
        V, k = 7, 3
        J = rng.normal(0, 0.7, (V, V))
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)
        h = rng.normal(0, 0.5, V)
        m_ex, C_ex = _exact_moments(J, h, k)
        res = estimate_moments(
            J, h, k, n_runs=5, n_steps=15_000, burn_in=4_000, thin=20, seed=3
        )
        assert res is not None
        m, C, _ = res
        self.assertLess(np.abs(m - m_ex).max(), 0.06)
        self.assertLess(np.abs(C - C_ex).max(), 0.06)

    def test_uniqueness_zeroes_forbidden_pairs(self) -> None:
        # Features 0 and 1 share a species: they can never co-occur, so the
        # model second moment C[0,1] must be ~0.
        V, k = 6, 3
        species_of = ["A", "A", "B", "C", "D", "E"]
        rng = np.random.default_rng(1)
        J = rng.normal(0, 0.5, (V, V))
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)
        h = rng.normal(0, 0.5, V)
        res = estimate_moments(
            J, h, k, species_of=species_of,
            n_runs=4, n_steps=12_000, burn_in=3_000, thin=20, seed=5,
        )
        assert res is not None
        m, C, _ = res
        self.assertAlmostEqual(C[0, 1], 0.0, places=7)
        m_ex, _ = _exact_moments(J, h, k, species_of)
        self.assertLess(np.abs(m - m_ex).max(), 0.06)


class TestBatchedSwapUniqueness(unittest.TestCase):
    def test_never_produces_duplicates(self) -> None:
        V, k = 8, 4
        species_of = ["A", "A", "B", "B", "C", "D", "E", "F"]
        item_of = ["x", "y", "x", None, "y", None, "z", "z"]
        rng = np.random.default_rng(2)
        species_id = _group_ids(species_of, V, none_sentinel=False)
        item_id = _group_ids(item_of, V, none_sentinel=True)
        J = rng.normal(0, 0.5, (V, V))
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)
        h = rng.normal(0, 0.5, V)
        # seed the bank from valid teams
        teams = _valid_teams(V, k, species_of, item_of)
        team = np.array([teams[i] for i in rng.integers(len(teams), size=32)])
        for _ in range(200):
            _batched_swap_sweep(team, J, h, species_id, item_id, rng)
        for row in team:
            sp = [species_of[i] for i in row]
            self.assertEqual(len(set(sp)), k, "duplicate species produced")
            it = [item_of[i] for i in row if item_of[i] is not None]
            self.assertEqual(len(set(it)), len(it), "duplicate item produced")


class TestBoltzmannFit(unittest.TestCase):
    def test_recovers_data_moments(self) -> None:
        rng = np.random.default_rng(0)
        V, k = 7, 3
        J_true = rng.normal(0, 0.6, (V, V))
        J_true = 0.5 * (J_true + J_true.T)
        np.fill_diagonal(J_true, 0.0)
        h_true = rng.normal(0, 0.5, V)
        X = _sample_corpus(J_true, h_true, k, n=8000, seed=1)
        m_data, C_data = empirical_moments(X)

        # Negligible reg so the converged model matches data moments tightly
        # (larger reg deliberately shrinks (J,h), biasing moments -- that's the
        # gauge/stability trade, not a recovery failure).
        # Lean, fully-pinned fit (2-temp PT, explicit decay): a fast full-pipeline
        # check, not a rigorous convergence gate. Budgets are small and the
        # statistical tolerances loosened to match; assertions independent of the
        # BOLTZMANN_* production defaults so tuning them can't perturb the test.
        J, h, hist = fit_boltzmann_ising(
            X, team_size=k, n_iters=300, lr=0.05, lr_final=0.0025, n_chains=200,
            n_sweeps=12, n_temps=2, t_max=3.0, swap_interval=10,
            reg="l2", reg_lambda=1e-4, seed=2, progress=False,
        )
        # Tail-averaged residuals (single-snapshot per-iter values are MC-noisy)
        # settle near the sampling floor.
        self.assertLess(np.mean(hist["mean_resid_m"][-50:]), 0.07)
        self.assertLess(np.mean(hist["mean_resid_C"][-50:]), 0.07)
        # A fresh T=1 estimate of the fitted model tracks the data moments.
        res = estimate_moments(
            J, h, k, n_runs=5, n_steps=15_000, burn_in=4_000, thin=20, seed=9
        )
        assert res is not None
        m, C, _ = res
        self.assertLess(np.abs(m - m_data).max(), 0.07)
        self.assertLess(np.abs(C - C_data).max(), 0.07)

    def test_support_mask_freezes_couplings(self) -> None:
        rng = np.random.default_rng(3)
        V, k = 6, 3
        X = np.zeros((2000, V), dtype=np.int8)
        for r in range(2000):
            X[r, rng.choice(V, k, replace=False)] = 1
        # Freeze the (0,1) coupling at a sentinel value; it must not move.
        init_J = np.zeros((V, V))
        init_J[0, 1] = init_J[1, 0] = 0.123
        mask = np.ones((V, V), dtype=bool)
        mask[0, 1] = mask[1, 0] = False
        J, _, _ = fit_boltzmann_ising(
            X, team_size=k, init_J=init_J, support_mask=mask,
            n_iters=30, lr=0.05, lr_final=0.0025, n_chains=64, n_sweeps=10,
            n_temps=2, t_max=3.0, swap_interval=10, seed=4, progress=False,
        )
        self.assertAlmostEqual(J[0, 1], 0.123, places=12)


if __name__ == "__main__":
    unittest.main()
