import unittest

import numpy as np

from sampling import (
    anneal_mcmc,
    build_constraint_sets,
    greedy_optimize,
    meanfield_marginals,
    parallel_tempered_mcmc,
    rank_single_swaps,
    swap_mcmc,
    swap_violates_uniqueness,
    team_energy,
)


def _toy_model(V: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    J = rng.normal(0, 0.3, size=(V, V))
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = rng.normal(0, 0.5, size=V)
    return J, h


class TestUniqueness(unittest.TestCase):
    def test_no_constraint_arrays_means_no_violation(self) -> None:
        self.assertFalse(
            swap_violates_uniqueness(0, 0, [1, 2], set(), set(), None, None)
        )

    def test_duplicate_species_blocked(self) -> None:
        species_of = ["a", "b", "a", "c"]
        # on_nf = [1, 0]; out_k=1 means we're removing idx 0 (species "a").
        # Swapping in idx 2 (species "a") is OK: the only "a" already on team is being removed.
        self.assertFalse(swap_violates_uniqueness(
            2, 1, [1, 0], set(), set(), species_of, None,
        ))
        # out_k=0 means we're removing idx 1 (species "b"); idx 0 (species "a") stays.
        # Swapping in idx 2 (species "a") -> conflicts with the remaining "a".
        self.assertTrue(swap_violates_uniqueness(
            2, 0, [1, 0], set(), set(), species_of, None,
        ))

    def test_fixed_species_blocked(self) -> None:
        species_of = ["a", "b", "c"]
        # Trying to swap idx 0 (species "a") into the team when "a" is in fixed set.
        self.assertTrue(swap_violates_uniqueness(
            0, 0, [1, 2], {"a"}, set(), species_of, None,
        ))

    def test_duplicate_item_blocked(self) -> None:
        item_of = ["X", "Y", "X", None]
        # out_k=0 removes idx 1 (item "Y"); idx 2 (item "X") stays. Swapping in idx 0
        # (also "X") collides.
        self.assertTrue(swap_violates_uniqueness(
            0, 0, [1, 2], set(), set(), None, item_of,
        ))
        # None items don't collide with anything.
        self.assertFalse(swap_violates_uniqueness(
            3, 0, [1, 2], set(), set(), None, item_of,
        ))

    def test_build_constraint_sets(self) -> None:
        species_of = ["a", "b", "a", "c"]
        item_of = ["X", "Y", None, "Z"]
        sp, it = build_constraint_sets({0, 2}, species_of, item_of)
        self.assertEqual(sp, {"a"})
        self.assertEqual(it, {"X"})  # idx 2's item is None, excluded


class TestSamplers(unittest.TestCase):
    def test_swap_mcmc_basic_shape(self) -> None:
        J, h = _toy_model()
        V = len(h)
        result = swap_mcmc(J, h, 6, [], [], 1.0, 100, 1.0, seed=0)
        self.assertIsNotNone(result)
        samples, _ = result
        self.assertEqual(samples.shape, (100, V))
        # Every step has exactly 6 mons on.
        self.assertTrue((samples.sum(axis=1) == 6).all())

    def test_swap_mcmc_respects_fixed(self) -> None:
        J, h = _toy_model()
        result = swap_mcmc(J, h, 6, [0, 5], [], 1.0, 50, 1.0, seed=0)
        self.assertIsNotNone(result)
        samples, _ = result
        self.assertTrue(samples[:, 0].all())
        self.assertTrue(samples[:, 5].all())

    def test_swap_mcmc_respects_excluded(self) -> None:
        J, h = _toy_model()
        result = swap_mcmc(J, h, 6, [], [3, 7, 11], 1.0, 50, 1.0, seed=0)
        self.assertIsNotNone(result)
        samples, _ = result
        self.assertFalse(samples[:, 3].any())
        self.assertFalse(samples[:, 7].any())
        self.assertFalse(samples[:, 11].any())

    def test_swap_mcmc_overconstrained_returns_none(self) -> None:
        J, h = _toy_model(V=8)
        # Need 6 free slots but only 5 candidates available (8 - 2 fixed - 1 excluded = 5).
        result = swap_mcmc(J, h, 8, [0, 1], [2], 1.0, 10, 1.0, seed=0)
        self.assertIsNone(result)

    def test_anneal_mcmc_returns_state_of_team_size(self) -> None:
        J, h = _toy_model()
        result = anneal_mcmc(J, h, 6, [], [], 1.0, 500, 3.0, 0.05, seed=0)
        self.assertIsNotNone(result)
        state, _ = result
        self.assertEqual(int(state.sum()), 6)

    def test_pt_cold_samples_after_burn_in(self) -> None:
        J, h = _toy_model()
        result = parallel_tempered_mcmc(
            J, h, 6, [], [], 1.0,
            np.array([0.5, 1.0, 2.0]), 200, 50, 5, seed=0,
        )
        self.assertIsNotNone(result)
        cold_samples, _, _ = result
        self.assertEqual(cold_samples.shape[0], 200 - 50)
        self.assertTrue((cold_samples.sum(axis=1) == 6).all())

    def test_meanfield_marginals_in_unit_interval(self) -> None:
        # MF iterates m = sigmoid(h + Jm); marginals are probabilities in [0,1].
        # The team-size constraint is NOT enforced here -- it's applied downstream
        # by greedy-picking top-K candidates. So we just verify shape + range +
        # convergence here, not the sum.
        J, h = _toy_model()
        result = meanfield_marginals(J, h, 6, [], [], 1.0)
        self.assertIsNotNone(result)
        marginals, valid_mask, iters = result
        self.assertEqual(marginals.shape, h.shape)
        self.assertEqual(valid_mask.shape, h.shape)
        self.assertTrue(np.all(marginals >= 0.0))
        self.assertTrue(np.all(marginals <= 1.0))
        # Should converge well within the default 200-iter cap.
        self.assertLess(iters, 200)

    def test_meanfield_respects_fixed(self) -> None:
        J, h = _toy_model()
        result = meanfield_marginals(J, h, 6, [0, 3], [7], 1.0)
        self.assertIsNotNone(result)
        marginals, valid_mask, _ = result
        self.assertEqual(marginals[0], 1.0)
        self.assertEqual(marginals[3], 1.0)
        self.assertEqual(marginals[7], 0.0)
        # Valid candidates exclude the fixed + excluded + uniqueness conflicts.
        self.assertFalse(valid_mask[0])
        self.assertFalse(valid_mask[3])
        self.assertFalse(valid_mask[7])

    def test_greedy_optimize_monotone_decreasing(self) -> None:
        J, h = _toy_model()
        start = list(range(6))
        _, chain = greedy_optimize(J, h, 6, start, [], [], 1.0)
        for step in chain:
            self.assertLess(step["delta_E_adj"], 0.0)

    def test_greedy_optimize_respects_pinned(self) -> None:
        J, h = _toy_model()
        start = list(range(6))
        final, chain = greedy_optimize(J, h, 6, start, pinned=[0, 1], excluded=[], field_weight=1.0)
        # Pinned mons must still be in the final team.
        self.assertIn(0, final)
        self.assertIn(1, final)
        # No chain step ever swaps out a pinned index.
        for step in chain:
            self.assertNotIn(step["out_idx"], {0, 1})


class TestRankSingleSwaps(unittest.TestCase):
    def test_sorted_ascending_by_delta_E_adj(self) -> None:
        J, h = _toy_model()
        team = list(range(6))
        results = rank_single_swaps(J, h, team, field_weight=1.0, top_n=10)
        deltas = [r["delta_E_adj"] for r in results]
        self.assertEqual(deltas, sorted(deltas))

    def test_top_swap_matches_greedy_first_step(self) -> None:
        # The most-improving single swap from the starting team should be the
        # same swap greedy_optimize takes on step 1.
        J, h = _toy_model()
        team = list(range(6))
        ranked = rank_single_swaps(J, h, team, field_weight=1.0, top_n=1)
        if not ranked:
            return  # no improving swap exists -- nothing to compare
        _, chain = greedy_optimize(J, h, 6, team, [], [], 1.0, max_swaps=1)
        if not chain:
            return
        self.assertEqual(ranked[0]["out_idx"], chain[0]["out_idx"])
        self.assertEqual(ranked[0]["in_idx"], chain[0]["in_idx"])
        self.assertAlmostEqual(ranked[0]["delta_E_adj"], chain[0]["delta_E_adj"])

    def test_in_idx_never_already_on_team(self) -> None:
        J, h = _toy_model()
        team = list(range(6))
        results = rank_single_swaps(J, h, team, field_weight=1.0, top_n=50)
        team_set = set(team)
        for r in results:
            self.assertNotIn(r["in_idx"], team_set)
            self.assertIn(r["out_idx"], team_set)


class TestTeamEnergy(unittest.TestCase):
    def test_team_energy_zero_state(self) -> None:
        J, h = _toy_model()
        state = np.zeros(len(h), dtype=bool)
        self.assertEqual(team_energy(state, J, h), 0.0)


if __name__ == "__main__":
    unittest.main()
