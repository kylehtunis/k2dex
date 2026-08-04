import unittest
from itertools import combinations

import numpy as np

from k2dex.sampling import (
    anchor_boost,
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
        _, chain = greedy_optimize(J, h, 6, team, [], [], 1.0, max_swaps=1)
        # Assert rather than skip: on the fixed toy model an improving swap
        # always exists, so an empty result is a regression, not a no-op case.
        self.assertTrue(ranked, "no improving swap found on the toy model")
        self.assertTrue(chain, "greedy took no step on the toy model")
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


# A tiny species+item ensemble (shared item x -> item-exclusion in play).
#   A: {x, y}   B: {x, None}   C: {z}   D: {None}   E: {w}   F: {None}
_TILT_SPECIES = ["A", "A", "B", "B", "C", "D", "E", "F"]
_TILT_ITEM = ["x", "y", "x", None, "z", None, "w", None]


def _tilt_toy(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    V = len(_TILT_SPECIES)
    rng = np.random.default_rng(seed)
    J = rng.normal(0, 0.6, (V, V))
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0.0)
    # Same-species couplings never co-occur; zero them to match the fit
    # convention so the exact enumeration is unambiguous.
    for a in range(V):
        for b in range(V):
            if _TILT_SPECIES[a] == _TILT_SPECIES[b]:
                J[a, b] = 0.0
    h = rng.normal(0, 0.5, V)
    return J, h


def _exact_tilted_conditional(
    J: np.ndarray, h: np.ndarray, k: int, pin: int, alpha: float,
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Enumerate P_alpha(team | pin on team) over uniqueness-valid teams:
    P_alpha ∝ exp(-H + (alpha-1) * sum_{j in team, j != pin} J[pin, j])."""
    V = len(h)
    teams = []
    log_w = []
    for rest in combinations([i for i in range(V) if i != pin], k - 1):
        t = (pin, *rest)
        sp = [_TILT_SPECIES[i] for i in t]
        if len(set(sp)) != k:
            continue
        it = [_TILT_ITEM[i] for i in t if _TILT_ITEM[i] is not None]
        if len(set(it)) != len(it):
            continue
        s = np.zeros(V)
        s[list(t)] = 1.0
        neg_h = h @ s + 0.5 * s @ J @ s
        tilt = (alpha - 1.0) * sum(J[pin, j] for j in rest)
        teams.append(t)
        log_w.append(neg_h + tilt)
    lw = np.array(log_w)
    p = np.exp(lw - lw.max())
    p /= p.sum()
    return teams, p


class TestAnchorFieldTilt(unittest.TestCase):
    def test_anchor_boost_values_and_pinned_species_zeroed(self) -> None:
        J, _h = _tilt_toy()
        alpha = 2.0
        boost = anchor_boost(J, [0], alpha, _TILT_SPECIES)
        # Feature 1 shares species A with the pin -> zeroed; free features get
        # (alpha-1) * J[pin, j]; the pin itself is zeroed.
        self.assertEqual(boost[0], 0.0)
        self.assertEqual(boost[1], 0.0)
        for j in range(2, len(boost)):
            self.assertAlmostEqual(boost[j], (alpha - 1.0) * J[0, j], places=12)

    def test_anchor_boost_alpha_one_is_zero(self) -> None:
        J, _h = _tilt_toy()
        self.assertEqual(np.abs(anchor_boost(J, [0, 3], 1.0, _TILT_SPECIES)).max(), 0.0)

    def test_pt_matches_exact_tilted_conditional(self) -> None:
        # The strong gate: PT with a pinned feature and alpha != 1 must sample
        # the exponentially tilted conditional exactly (up to MC error).
        J, h = _tilt_toy()
        k, pin, alpha = 3, 0, 2.0
        teams, p = _exact_tilted_conditional(J, h, k, pin, alpha)
        V = len(h)
        m_ex = np.zeros(V)
        for pt, t in zip(p, teams):
            for j in t:
                m_ex[j] += pt

        result = parallel_tempered_mcmc(
            J, h, k, [pin], [], 1.0,
            np.array([1.0, 2.0]), 60_000, 10_000, 10, seed=11,
            species_of=_TILT_SPECIES, item_of=_TILT_ITEM,
            anchor_strength=alpha,
        )
        self.assertIsNotNone(result)
        assert result is not None
        samples, _, _ = result
        m_hat = samples.mean(axis=0)
        self.assertLess(np.abs(m_hat - m_ex).max(), 0.03)

    def test_pin_integration_monotone_in_alpha_exact(self) -> None:
        # d/dalpha E[T] = Var(T) >= 0 on the exact tilted distribution.
        J, h = _tilt_toy()
        k, pin = 3, 0
        means = []
        for alpha in (1.0, 1.5, 2.0, 3.0):
            teams, p = _exact_tilted_conditional(J, h, k, pin, alpha)
            T = np.array([sum(J[pin, j] for j in t if j != pin) for t in teams])
            means.append(float(p @ T))
        for lo, hi in zip(means, means[1:]):
            self.assertLessEqual(lo, hi + 1e-12)

    def test_pt_alpha_one_reproduces_default(self) -> None:
        J, h = _tilt_toy()
        kwargs: dict = dict(
            species_of=_TILT_SPECIES, item_of=_TILT_ITEM,
        )
        a = parallel_tempered_mcmc(
            J, h, 3, [0], [], 1.0, np.array([1.0, 2.0]), 500, 100, 10,
            seed=5, **kwargs,
        )
        b = parallel_tempered_mcmc(
            J, h, 3, [0], [], 1.0, np.array([1.0, 2.0]), 500, 100, 10,
            seed=5, anchor_strength=1.0, **kwargs,
        )
        assert a is not None and b is not None
        np.testing.assert_array_equal(a[0], b[0])

    def test_greedy_tilt_equals_greedy_on_boosted_field(self) -> None:
        # At fw=1, greedy with anchor_strength=alpha must walk the same chain
        # as untilted greedy on h' = h + anchor_boost (same H_alpha).
        J, h = _tilt_toy(seed=3)
        pin, alpha = 4, 2.5
        start = [4, 0, 2]
        tilted_final, tilted_chain = greedy_optimize(
            J, h, 3, start, [pin], [], 1.0,
            species_of=_TILT_SPECIES, item_of=_TILT_ITEM,
            anchor_strength=alpha,
        )
        h_boosted = h + anchor_boost(J, [pin], alpha, _TILT_SPECIES)
        manual_final, manual_chain = greedy_optimize(
            J, h_boosted, 3, start, [pin], [], 1.0,
            species_of=_TILT_SPECIES, item_of=_TILT_ITEM,
        )
        self.assertEqual(tilted_final, manual_final)
        self.assertEqual(
            [(c["out_idx"], c["in_idx"]) for c in tilted_chain],
            [(c["out_idx"], c["in_idx"]) for c in manual_chain],
        )

    def test_meanfield_tilt_equals_meanfield_on_boosted_field(self) -> None:
        J, h = _tilt_toy(seed=3)
        pin, alpha = 4, 2.0
        a = meanfield_marginals(
            J, h, 3, [pin], [], 1.0,
            species_of=_TILT_SPECIES, item_of=_TILT_ITEM,
            anchor_strength=alpha,
        )
        h_boosted = h + anchor_boost(J, [pin], alpha, _TILT_SPECIES)
        b = meanfield_marginals(
            J, h_boosted, 3, [pin], [], 1.0,
            species_of=_TILT_SPECIES, item_of=_TILT_ITEM,
        )
        assert a is not None and b is not None
        np.testing.assert_allclose(a[0], b[0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
