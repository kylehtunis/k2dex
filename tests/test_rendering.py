import unittest

import numpy as np

from k2dex.rendering import (
    intra_team_sum_j,
    pairwise_j_rows,
)


class TestRenderingHelpers(unittest.TestCase):
    def _state(self, V: int, on: list[int]) -> np.ndarray:
        s = np.zeros(V, dtype=bool)
        s[on] = True
        return s

    def test_intra_team_sum_j(self) -> None:
        # 4x4 J with only J[0, 1] = J[1, 0] = 2.0, J[2, 3] = J[3, 2] = -1.0
        J = np.zeros((4, 4))
        J[0, 1] = J[1, 0] = 2.0
        J[2, 3] = J[3, 2] = -1.0
        # Team = {0, 1}: only pair is (0, 1) -> sum_J = 2.0
        self.assertAlmostEqual(
            intra_team_sum_j(self._state(4, [0, 1]), J),
            2.0,
        )
        # Team = {0, 1, 2, 3}: pairs (0,1) -> +2, (2,3) -> -1, rest 0 -> sum 1.0
        self.assertAlmostEqual(
            intra_team_sum_j(self._state(4, [0, 1, 2, 3]), J),
            1.0,
        )


class TestPairwiseJDecomposition(unittest.TestCase):
    def test_sorted_by_abs_descending(self) -> None:
        vocab = ["a", "b", "c", "d"]
        J = np.zeros((4, 4))
        J[0, 1] = J[1, 0] = -2.0
        J[0, 2] = J[2, 0] = 0.5
        J[1, 2] = J[2, 1] = 1.5
        J[0, 3] = J[3, 0] = 0.0  # tied zero -- order undefined among zeros
        rows = pairwise_j_rows([0, 1, 2, 3], vocab, J)
        self.assertEqual(rows[0].name_a, "a")
        self.assertEqual(rows[0].name_b, "b")
        self.assertAlmostEqual(rows[0].j_value, -2.0)
        self.assertEqual(rows[1].name_a, "b")
        self.assertEqual(rows[1].name_b, "c")
        self.assertAlmostEqual(rows[1].j_value, 1.5)
        # Percentages sum to 100% (modulo zero-J rows).
        total_pct = sum(r.pct_of_abs_sum for r in rows)
        self.assertAlmostEqual(total_pct, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
