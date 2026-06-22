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


class TestFitPLIsingPrior(unittest.TestCase):
    @staticmethod
    def _random_prior(V: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        A = rng.normal(scale=0.5, size=(V, V))
        prior_J = 0.5 * (A + A.T)
        np.fill_diagonal(prior_J, 0.0)
        prior_h = rng.normal(scale=0.5, size=V)
        return prior_J, prior_h

    def test_requires_both_prior_arrays(self) -> None:
        X = (np.random.default_rng(0).random((50, 5)) < 0.3).astype(np.int8)
        with self.assertRaises(ValueError):
            fit_pl_ising(X, prior_J=np.zeros((5, 5)))
        with self.assertRaises(ValueError):
            fit_pl_ising(X, prior_h=np.zeros(5))

    def test_prior_shape_validation(self) -> None:
        X = (np.random.default_rng(1).random((50, 5)) < 0.3).astype(np.int8)
        with self.assertRaises(ValueError):
            fit_pl_ising(X, prior_J=np.zeros((4, 4)), prior_h=np.zeros(5))
        with self.assertRaises(ValueError):
            fit_pl_ising(X, prior_J=np.zeros((5, 5)), prior_h=np.zeros(4))

    def test_empty_evidence_feature_retains_prior(self) -> None:
        # Feature 0 never appears in X -> its spin is degenerate (falls back to
        # the prior row) and other spins pull its all-zero column's coefficient
        # to the prior under the re-centered penalty. After symmetrization the
        # whole row, and h[0], equal the prior exactly (to solver tolerance).
        rng = np.random.default_rng(2)
        V = 6
        X = (rng.random((400, V)) < 0.35).astype(np.int8)
        X[:, 0] = 0
        prior_J, prior_h = self._random_prior(V, seed=3)
        J, h = fit_pl_ising(X, C=1.0 / 25.0, prior_J=prior_J, prior_h=prior_h)
        np.testing.assert_allclose(J[0, :], prior_J[0, :], atol=1e-4)
        np.testing.assert_allclose(h[0], prior_h[0], atol=1e-6)

    def test_well_sampled_feature_overrules_prior(self) -> None:
        # Two perfectly correlated features get J > 0 even when the prior says
        # they should be strongly negative: enough data overrules the prior.
        rng = np.random.default_rng(4)
        co = (rng.random(1500) < 0.4).astype(np.int8)
        indep = (rng.random((1500, 3)) < 0.3).astype(np.int8)
        X = np.column_stack([co, co, indep]).astype(np.int8)
        V = X.shape[1]
        prior_J = np.zeros((V, V))
        prior_J[0, 1] = prior_J[1, 0] = -3.0
        J, _ = fit_pl_ising(X, C=1.0 / 1.0, prior_J=prior_J, prior_h=np.zeros(V))
        self.assertGreater(J[0, 1], 0.0)

    def test_zero_prior_matches_no_prior(self) -> None:
        # A zero prior with a free intercept must reproduce the base fit.
        rng = np.random.default_rng(5)
        X = (rng.random((300, 7)) < 0.3).astype(np.int8)
        V = X.shape[1]
        J0, h0 = fit_pl_ising(X, C=1.0 / 25.0)
        J1, h1 = fit_pl_ising(
            X,
            C=1.0 / 25.0,
            prior_J=np.zeros((V, V)),
            prior_h=np.zeros(V),
            intercept_prior_weight=0.0,
        )
        np.testing.assert_allclose(J1, J0, atol=1e-6)
        np.testing.assert_allclose(h1, h0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
