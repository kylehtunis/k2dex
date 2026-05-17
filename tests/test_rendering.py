import unittest
from collections import Counter

import numpy as np

from rendering import intra_team_sum_j, min_swaps_to_observed, team_obs_count


class TestRenderingHelpers(unittest.TestCase):
    def _state(self, V: int, on: list[int]) -> np.ndarray:
        s = np.zeros(V, dtype=bool)
        s[on] = True
        return s

    def test_team_obs_count_none_when_no_team_counts(self) -> None:
        state = self._state(5, [0, 1, 2])
        vocab = ["a", "b", "c", "d", "e"]
        self.assertIsNone(team_obs_count(state, vocab, None))

    def test_team_obs_count_hits_corpus(self) -> None:
        vocab = ["a", "b", "c", "d", "e", "f"]
        team_counts = Counter({frozenset(["a", "b", "c", "d", "e", "f"]): 7})
        state = self._state(6, [0, 1, 2, 3, 4, 5])
        self.assertEqual(team_obs_count(state, vocab, team_counts), 7)

    def test_team_obs_count_zero_for_missing(self) -> None:
        vocab = ["a", "b", "c", "d", "e", "f"]
        team_counts = Counter({frozenset(["a", "b", "c", "d", "e", "f"]): 7})
        # Different team; should return 0 (Counter default).
        state = self._state(6, [0, 1, 2, 3, 4, 5])
        # Swap one out:
        state[5] = False
        state[4] = True  # already True; doesn't matter
        # Build the state from scratch to be sure:
        state2 = self._state(6, [0, 1, 2, 3, 4])  # only 5 mons; not realistic but fine for test
        self.assertEqual(team_obs_count(state2, vocab, team_counts), 0)

    def test_min_swaps_to_observed(self) -> None:
        vocab = ["a", "b", "c", "d", "e", "f"]
        team_counts = Counter({frozenset(["a", "b", "c", "d", "e", "f"]): 1})
        self.assertEqual(
            min_swaps_to_observed(self._state(6, [0, 1, 2, 3, 4, 5]), vocab, team_counts),
            0,
        )

    def test_min_swaps_to_observed_one_swap(self) -> None:
        vocab = ["a", "b", "c", "d", "e", "f", "g"]
        team_counts = Counter({frozenset(["a", "b", "c", "d", "e", "f"]): 1})
        # Swap 'f' for 'g' -> one-swap variant
        self.assertEqual(
            min_swaps_to_observed(self._state(7, [0, 1, 2, 3, 4, 6]), vocab, team_counts),
            1,
        )

    def test_min_swaps_none_when_empty(self) -> None:
        vocab = ["a", "b", "c"]
        self.assertIsNone(
            min_swaps_to_observed(self._state(3, [0, 1, 2]), vocab, Counter()),
        )
        self.assertIsNone(
            min_swaps_to_observed(self._state(3, [0, 1, 2]), vocab, None),
        )

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


if __name__ == "__main__":
    unittest.main()
