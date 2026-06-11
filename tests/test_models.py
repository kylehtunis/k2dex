import unittest

import numpy as np

from k2dex.models import fit_pl_ising


class TestFitPLIsing(unittest.TestCase):
    def test_symmetric_zero_diagonal(self) -> None:
        rng = np.random.default_rng(0)
        X = (rng.random((200, 12)) < 0.3).astype(np.int8)
        J, h = fit_pl_ising(X)
        np.testing.assert_allclose(J, J.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(J), 0.0)
        self.assertEqual(h.shape, (12,))

    def test_recovers_positive_coupling(self) -> None:
        # Two perfectly correlated features should get J[0, 1] > 0.
        rng = np.random.default_rng(1)
        n = 1000
        co = (rng.random(n) < 0.4).astype(np.int8)
        independent = (rng.random((n, 3)) < 0.3).astype(np.int8)
        X = np.column_stack([co, co, independent]).astype(np.int8)
        J, _ = fit_pl_ising(X)
        self.assertGreater(J[0, 1], 0.5)

    def test_skips_degenerate_column(self) -> None:
        # Spin 0 always on (degenerate) -> skipped without crashing; h[0] = 0.
        # J[0, :] / J[:, 0] may have small nonzero entries from other regressions
        # treating the constant column as a (near-redundant) feature — that's v0
        # behavior and we preserve it here.
        X = np.zeros((100, 5), dtype=np.int8)
        X[:, 0] = 1
        rng = np.random.default_rng(2)
        X[:, 1:] = (rng.random((100, 4)) < 0.3).astype(np.int8)
        J, h = fit_pl_ising(X)
        self.assertEqual(h[0], 0.0)
        # Row 0 of J should still be small (the LogReg for spins 1..4 with
        # constant feature 0 absorbs it into the intercept).
        self.assertLess(float(np.max(np.abs(J[0, :]))), 1e-2)


class TestFitPLIsingSampleWeight(unittest.TestCase):
    def test_unit_weights_match_unweighted(self) -> None:
        rng = np.random.default_rng(3)
        X = (rng.random((200, 8)) < 0.3).astype(np.int8)
        J0, h0 = fit_pl_ising(X)
        J1, h1 = fit_pl_ising(X, sample_weight=np.ones(200))
        np.testing.assert_allclose(J1, J0, atol=1e-8)
        np.testing.assert_allclose(h1, h0, atol=1e-8)

    def test_weight_two_matches_row_duplication(self) -> None:
        # Weighting a row 2.0 gives the same objective as duplicating it,
        # so the fits should agree up to solver tolerance.
        rng = np.random.default_rng(4)
        X = (rng.random((150, 6)) < 0.35).astype(np.int8)
        dup = np.arange(150) < 50
        w = np.where(dup, 2.0, 1.0)
        X_dup = np.vstack([X, X[dup]])
        J_w, h_w = fit_pl_ising(X, sample_weight=w)
        J_d, h_d = fit_pl_ising(X_dup)
        np.testing.assert_allclose(J_w, J_d, atol=1e-4)
        np.testing.assert_allclose(h_w, h_d, atol=1e-4)

    def test_weighted_degenerate_skip(self) -> None:
        # Spin 0 is on in 3 rows, but those rows carry < 2 units of weighted
        # mass -> the weight-aware skip fires and h[0] stays exactly 0.
        rng = np.random.default_rng(5)
        X = (rng.random((100, 5)) < 0.3).astype(np.int8)
        X[:, 0] = 0
        X[:3, 0] = 1
        w = np.ones(100)
        w[:3] = 0.1
        _, h = fit_pl_ising(X, sample_weight=w)
        self.assertEqual(h[0], 0.0)

    def test_sample_weight_validation(self) -> None:
        X = np.zeros((10, 3), dtype=np.int8)
        with self.assertRaises(ValueError):
            fit_pl_ising(X, sample_weight=np.ones(5))
        with self.assertRaises(ValueError):
            fit_pl_ising(X, sample_weight=-np.ones(10))


if __name__ == "__main__":
    unittest.main()
