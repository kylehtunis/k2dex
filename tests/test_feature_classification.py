import unittest

import numpy as np

from feature_classification import (
    classify_features,
    feature_metrics,
    partner_contributions,
    residual,
)


def _fixture():
    """Hand-built (J, m, h) where every output of every function is verifiable
    by inspection. V=7; feature 6 has m below the floor used in classify."""
    V = 7
    J = np.array([
        [ 0.0,  2.0,  1.0,  0.0,  0.0, -1.5,  0.0],
        [ 2.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0],
        [ 1.0,  1.0,  0.0, -2.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0, -2.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [-1.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    ], dtype=float)
    m = np.array([0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.005])
    h = np.array([-2.0, -2.5, -3.0, -4.0, -4.5, -5.5, -10.0])
    return V, J, m, h


class TestFeatureMetrics(unittest.TestCase):
    def test_returns_three_named_vectors(self) -> None:
        _, J, m, _ = _fixture()
        out = feature_metrics(J, m)
        self.assertEqual(
            sorted(out.keys()), ["abs_j_dot_m", "j_dot_m", "sum_abs_j"]
        )

    def test_j_dot_m_values(self) -> None:
        _, J, m, _ = _fixture()
        out = feature_metrics(J, m)
        expected = np.array([0.725, 1.000, 0.400, -0.400, 0.000, -0.600, 0.000])
        np.testing.assert_allclose(out["j_dot_m"], expected, atol=1e-12)

    def test_abs_j_dot_m_values(self) -> None:
        _, J, m, _ = _fixture()
        out = feature_metrics(J, m)
        expected = np.array([0.875, 1.000, 1.000, 0.400, 0.000, 0.600, 0.000])
        np.testing.assert_allclose(out["abs_j_dot_m"], expected, atol=1e-12)

    def test_sum_abs_j_values(self) -> None:
        _, J, m, _ = _fixture()
        out = feature_metrics(J, m)
        expected = np.array([4.5, 3.0, 4.0, 2.0, 0.0, 1.5, 0.0])
        np.testing.assert_allclose(out["sum_abs_j"], expected, atol=1e-12)


class TestResidual(unittest.TestCase):
    def test_known_value(self) -> None:
        h = np.array([-2.0])
        m = np.array([0.4])
        out = residual(h, m)
        # logit(0.4) = log(0.4/0.6) ≈ -0.4054651
        # residual = -2.0 - (-0.4054651) ≈ -1.5945349
        np.testing.assert_allclose(out, np.array([-1.5945348781]), atol=1e-6)

    def test_no_inf_at_m_zero(self) -> None:
        h = np.array([-5.0])
        m = np.array([0.0])
        out = residual(h, m)
        self.assertTrue(np.isfinite(out).all())

    def test_no_inf_at_m_one(self) -> None:
        h = np.array([5.0])
        m = np.array([1.0])
        out = residual(h, m)
        self.assertTrue(np.isfinite(out).all())


class TestPartnerContributions(unittest.TestCase):
    def test_equals_jrow_times_m(self) -> None:
        _, J, m, _ = _fixture()
        out = partner_contributions(J, m, i=0)
        expected = np.array([0.0, 0.6, 0.2, 0.0, 0.0, -0.075, 0.0])
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_sum_equals_j_dot_m(self) -> None:
        _, J, m, _ = _fixture()
        # For every feature, the partner contributions must sum to J·m_i.
        j_dot_m = feature_metrics(J, m)["j_dot_m"]
        for i in range(J.shape[0]):
            with self.subTest(i=i):
                self.assertAlmostEqual(
                    float(partner_contributions(J, m, i=i).sum()),
                    float(j_dot_m[i]),
                    places=12,
                )


class TestClassifyFeatures(unittest.TestCase):
    def test_basic_quadrants_top_k_1(self) -> None:
        _, J, m, _ = _fixture()
        metrics = feature_metrics(J, m)
        out = classify_features(
            j_dot_m=metrics["j_dot_m"],
            abs_j_dot_m=metrics["abs_j_dot_m"],
            m=m,
            m_floor=0.01,
            top_k=1,
        )
        self.assertEqual(out["glue"], [1])
        self.assertEqual(out["outcast"], [5])
        self.assertEqual(out["specialist"], [2])
        self.assertEqual(out["flex"], [4])

    def test_keys(self) -> None:
        _, J, m, _ = _fixture()
        metrics = feature_metrics(J, m)
        out = classify_features(
            j_dot_m=metrics["j_dot_m"],
            abs_j_dot_m=metrics["abs_j_dot_m"],
            m=m,
            m_floor=0.01,
            top_k=1,
        )
        self.assertEqual(
            sorted(out.keys()), ["flex", "glue", "outcast", "specialist"]
        )

    def test_floor_excludes_tail(self) -> None:
        _, J, m, _ = _fixture()
        metrics = feature_metrics(J, m)
        # With a higher floor, feature 4 (m=0.10) also drops out, so the
        # central band shrinks.
        out = classify_features(
            j_dot_m=metrics["j_dot_m"],
            abs_j_dot_m=metrics["abs_j_dot_m"],
            m=m,
            m_floor=0.15,
            top_k=1,
        )
        # Feature 6 (m=0.005), 5 (m=0.05), 4 (m=0.10) all excluded.
        # Surviving: {0,1,2,3}. j_dot_m = [0.725, 1.000, 0.400, -0.400].
        # glue=top by j_dot_m=[1], outcast=bottom=[3].
        # |j_dot_m| = [0.725, 1.000, 0.400, 0.400], median = 0.5625.
        # Central band = {2, 3}. |J|·m for {2,3} = {2:1.000, 3:0.400}.
        # specialist (top 1) = [2]; flex (bottom 1) = [3].
        # Note: feature 3 is both outcast AND flex under this floor — this
        # is acceptable when the central band is small relative to top_k.
        self.assertEqual(out["glue"], [1])
        self.assertEqual(out["outcast"], [3])
        self.assertEqual(out["specialist"], [2])
        self.assertEqual(out["flex"], [3])


class TestBuildScatterFigure(unittest.TestCase):
    def test_returns_figure_without_crash(self) -> None:
        _, J, m, _ = _fixture()
        metrics = feature_metrics(J, m)
        classification = classify_features(
            j_dot_m=metrics["j_dot_m"],
            abs_j_dot_m=metrics["abs_j_dot_m"],
            m=m,
            m_floor=0.01,
            top_k=1,
        )
        vocab = ["a", "b", "c", "d", "e", "f", "g"]
        from feature_classification import build_scatter_figure
        fig = build_scatter_figure(
            j_dot_m=metrics["j_dot_m"],
            abs_j_dot_m=metrics["abs_j_dot_m"],
            m=m,
            classification=classification,
            vocab=vocab,
            label_top_k=3,
        )
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]
        self.assertGreater(len(ax.collections), 0)
        import matplotlib.pyplot as plt
        plt.close(fig)
