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


if __name__ == "__main__":
    unittest.main()
