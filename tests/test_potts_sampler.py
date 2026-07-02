"""Validation of the Potts (separate species / item) training sampler.

The Potts move kernel (Metropolized-Gibbs species swap + Gibbs item reroll) must
target the same constrained Gibbs measure as the atomic swap sampler. These
tests pin that down three ways: it reproduces the *exact* moments of a tiny
enumerable species+item ensemble; its species-swap acceptance degenerates
exactly to the atomic sampler's MH rule when every species has one item (q=2);
and it never violates species / item uniqueness.
"""
import unittest
from itertools import combinations

import numpy as np

from k2dex.models import (
    _advance_bank_potts,
    _build_site_tables,
    _group_ids,
    _site_conditional,
    fit_boltzmann_ising,
)
from k2dex.sampling import estimate_moments, initialize_state


def _valid_teams(V, k, species_of, item_of):
    teams = []
    for t in combinations(range(V), k):
        sp = [species_of[i] for i in t]
        if len(set(sp)) != k:
            continue
        it = [item_of[i] for i in t if item_of[i] is not None]
        if len(set(it)) != len(it):
            continue
        teams.append(t)
    return teams


def _onehot(t, V):
    s = np.zeros(V)
    s[list(t)] = 1.0
    return s


def _exact_moments(J, h, k, species_of, item_of):
    V = len(h)
    teams = _valid_teams(V, k, species_of, item_of)
    e = np.array([-h @ _onehot(t, V) - 0.5 * _onehot(t, V) @ J @ _onehot(t, V)
                  for t in teams])
    logp = -e
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    m = np.zeros(V)
    C = np.zeros((V, V))
    for pt, t in zip(p, teams):
        s = _onehot(t, V)
        m += pt * s
        C += pt * np.outer(s, s)
    return m, C


def _potts_moments(J, h, k, species_of, item_of, *, seed,
                   n_chains=600, burn=3000, snapshots=500, stride=4):
    """Pooled moments from a single persistent Potts-kernel bank at T=1."""
    V = len(h)
    species_id = _group_ids(species_of, V, none_sentinel=False)
    item_id = _group_ids(item_of, V, none_sentinel=True)
    assert species_id is not None and item_id is not None
    site = _build_site_tables(species_id, item_id, V)
    rng = np.random.default_rng(seed)
    available = np.arange(V)
    temps = np.array([1.0])
    team = np.empty((1, n_chains, k), dtype=np.int64)
    for c in range(n_chains):
        team[0, c] = initialize_state(
            available, k, set(), set(), species_of, item_of, rng)
    _advance_bank_potts(team, J, h, temps, species_id, item_id, site, rng,
                        n_sweeps=burn, swap_interval=10, p_reroll=0.5)
    m_acc = np.zeros(V)
    C_acc = np.zeros((V, V))
    for _ in range(snapshots):
        _advance_bank_potts(team, J, h, temps, species_id, item_id, site, rng,
                            n_sweeps=stride, swap_interval=10, p_reroll=0.5)
        flat = team[0]
        m_acc += np.bincount(flat.ravel(), minlength=V) / n_chains
        # raw second moments over on-bits
        for row in flat:
            C_acc[np.ix_(row, row)] += 1.0
    m_acc /= snapshots
    C_acc /= snapshots * n_chains
    return m_acc, C_acc


# A tiny species+item ensemble with a shared item (item-exclusion in play).
#   A: {x, y}   B: {x, None}   C: {z}   D: {None}
_SPECIES = ["A", "A", "B", "B", "C", "D"]
_ITEM = ["x", "y", "x", None, "z", None]


class TestPottsSamplerTargetsExact(unittest.TestCase):
    def test_matches_exact_moments(self) -> None:
        rng = np.random.default_rng(0)
        V, k = 6, 3
        J = rng.normal(0, 0.6, (V, V))
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)
        # Same-species couplings are irrelevant (never co-occur); zero them so J
        # matches the fit convention and the exact enumeration is unambiguous.
        sid = _group_ids(_SPECIES, V, none_sentinel=False)
        assert sid is not None
        J[sid[:, None] == sid[None, :]] = 0.0
        h = rng.normal(0, 0.5, V)

        m_ex, C_ex = _exact_moments(J, h, k, _SPECIES, _ITEM)
        m, C = _potts_moments(J, h, k, _SPECIES, _ITEM, seed=7)
        self.assertLess(np.abs(m - m_ex).max(), 0.03)
        self.assertLess(np.abs(C - C_ex).max(), 0.03)

    def test_forbidden_pairs_stay_zero(self) -> None:
        # A@x (0) and B@x (2) share item x; A@x and A@y (0,1) share species A.
        rng = np.random.default_rng(1)
        V, k = 6, 3
        J = rng.normal(0, 0.5, (V, V))
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)
        h = rng.normal(0, 0.5, V)
        _, C = _potts_moments(J, h, k, _SPECIES, _ITEM, seed=3,
                              n_chains=300, burn=1500, snapshots=200)
        self.assertAlmostEqual(C[0, 1], 0.0, places=7)  # same species
        self.assertAlmostEqual(C[0, 2], 0.0, places=7)  # same item


class TestQ2Degeneracy(unittest.TestCase):
    """One item per species -> the species-swap acceptance must reduce exactly to
    the atomic sampler's ΔH rule, and the item conditional is trivial."""

    def test_logratio_equals_negative_delta_h(self) -> None:
        V = 5
        species_of = ["A", "B", "C", "D", "E"]  # one item each
        item_of = [None, None, None, None, None]
        rng = np.random.default_rng(2)
        J = rng.normal(0, 0.6, (V, V))
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)
        h = rng.normal(0, 0.5, V)
        species_id = _group_ids(species_of, V, none_sentinel=False)
        item_id = _group_ids(item_of, V, none_sentinel=True)
        assert species_id is not None and item_id is not None
        site = _build_site_tables(species_id, item_id, V)

        # Held members R = {2, 3}; swap species A (feat 0) for B (feat 1).
        R_feat = np.array([[2, 3]])
        R_iid = item_id[R_feat]
        inv_temp = np.array([1.0])
        A = np.array([species_id[0]])
        B = np.array([species_id[1]])
        logZ_A, negE_A, valid_A, _ = _site_conditional(
            A, R_feat, R_iid, J, h, inv_temp, *site[1:])
        logZ_B, _, _, _ = _site_conditional(
            B, R_feat, R_iid, J, h, inv_temp, *site[1:])
        # Single item -> logZ is just that item's -E_slot, conditional is trivial.
        self.assertEqual(int(valid_A.sum()), 1)
        self.assertAlmostEqual(float(logZ_A[0]), float(negE_A[valid_A][0]), places=12)

        log_ratio = float(logZ_B[0] - logZ_A[0])
        # Atomic sampler's ΔH for out=0, in=1 over the retained members {2,3}.
        delta_h = (h[0] - h[1]
                   + (J[0, 2] - J[1, 2]) + (J[0, 3] - J[1, 3]))
        self.assertAlmostEqual(log_ratio, -delta_h, places=12)

    def test_single_item_ensemble_matches_exact(self) -> None:
        V, k = 5, 3
        species_of = ["A", "B", "C", "D", "E"]
        item_of = [None] * 5
        rng = np.random.default_rng(4)
        J = rng.normal(0, 0.6, (V, V))
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)
        h = rng.normal(0, 0.5, V)
        m_ex, C_ex = _exact_moments(J, h, k, species_of, item_of)
        m, C = _potts_moments(J, h, k, species_of, item_of, seed=8,
                              n_chains=400, burn=2000, snapshots=300)
        self.assertLess(np.abs(m - m_ex).max(), 0.03)
        self.assertLess(np.abs(C - C_ex).max(), 0.03)


class TestConstraintInvariants(unittest.TestCase):
    def test_never_violates_uniqueness(self) -> None:
        V, k = 6, 3
        rng = np.random.default_rng(5)
        J = rng.normal(0, 0.5, (V, V))
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)
        h = rng.normal(0, 0.5, V)
        species_id = _group_ids(_SPECIES, V, none_sentinel=False)
        item_id = _group_ids(_ITEM, V, none_sentinel=True)
        assert species_id is not None and item_id is not None
        site = _build_site_tables(species_id, item_id, V)
        available = np.arange(V)
        team = np.empty((1, 64, k), dtype=np.int64)
        for c in range(64):
            team[0, c] = initialize_state(
                available, k, set(), set(), _SPECIES, _ITEM, rng)
        _advance_bank_potts(team, J, h, np.array([1.0]), species_id, item_id,
                            site, rng, n_sweeps=800, swap_interval=10, p_reroll=0.5)
        for row in team[0]:
            sp = [_SPECIES[i] for i in row]
            self.assertEqual(len(set(sp)), k, "duplicate species produced")
            it = [_ITEM[i] for i in row if _ITEM[i] is not None]
            self.assertEqual(len(set(it)), len(it), "duplicate item produced")

    def test_item_reroll_keeps_species_roster(self) -> None:
        # p_reroll=1.0 -> only item rerolls; the multiset of species per chain
        # must be invariant.
        V, k = 6, 3
        rng = np.random.default_rng(6)
        J = rng.normal(0, 0.5, (V, V))
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)
        h = rng.normal(0, 0.5, V)
        species_id = _group_ids(_SPECIES, V, none_sentinel=False)
        item_id = _group_ids(_ITEM, V, none_sentinel=True)
        assert species_id is not None and item_id is not None
        site = _build_site_tables(species_id, item_id, V)
        available = np.arange(V)
        team = np.empty((1, 32, k), dtype=np.int64)
        for c in range(32):
            team[0, c] = initialize_state(
                available, k, set(), set(), _SPECIES, _ITEM, rng)
        before = np.sort(species_id[team[0]], axis=1)
        _advance_bank_potts(team, J, h, np.array([1.0]), species_id, item_id,
                            site, rng, n_sweeps=300, swap_interval=10, p_reroll=1.0)
        after = np.sort(species_id[team[0]], axis=1)
        np.testing.assert_array_equal(before, after)


class TestPottsFitPath(unittest.TestCase):
    def test_fit_recovers_moments_with_potts_kernel(self) -> None:
        # End-to-end: the default species+item fit path (potts_moves=True) still
        # drives the model moments to the data moments.
        rng = np.random.default_rng(0)
        V, k = 6, 3
        J_true = rng.normal(0, 0.5, (V, V))
        J_true = 0.5 * (J_true + J_true.T)
        np.fill_diagonal(J_true, 0.0)
        sid = _group_ids(_SPECIES, V, none_sentinel=False)
        assert sid is not None
        J_true[sid[:, None] == sid[None, :]] = 0.0
        h_true = rng.normal(0, 0.4, V)

        teams = _valid_teams(V, k, _SPECIES, _ITEM)
        e = np.array([-h_true @ _onehot(t, V)
                      - 0.5 * _onehot(t, V) @ J_true @ _onehot(t, V) for t in teams])
        p = np.exp(-(e - e.max())); p /= p.sum()
        draws = rng.choice(len(teams), size=8000, p=p)
        X = np.zeros((8000, V), dtype=np.int8)
        for r, ti in enumerate(draws):
            X[r, list(teams[ti])] = 1

        J, h, hist = fit_boltzmann_ising(
            X, team_size=k, species_of=_SPECIES, item_of=_ITEM,
            n_iters=600, lr=0.05, n_chains=500, n_sweeps=20,
            reg="l2", reg_lambda=1e-4, seed=2, progress=False,
        )
        self.assertLess(np.mean(hist["mean_resid_m"][-50:]), 0.03)
        self.assertLess(np.mean(hist["mean_resid_C"][-50:]), 0.03)
        # Same-species couplings must remain frozen at zero.
        same = sid[:, None] == sid[None, :]
        np.testing.assert_allclose(J[same], 0.0, atol=1e-12)
        # Independent check against the atomic sampler used by estimate_moments.
        res = estimate_moments(
            J, h, k, species_of=_SPECIES, item_of=_ITEM,
            n_runs=12, n_steps=40_000, burn_in=8_000, thin=20, seed=9)
        assert res is not None
        m_hat, _, _ = res
        # A cross-sampler sanity check, not the tight gate: it stacks the atomic
        # sampler's MC noise on top of the finite-corpus mean, so it is looser
        # than the Potts-vs-exact gate (test_matches_exact_moments, tol 0.03).
        m_data = X.mean(axis=0)
        self.assertLess(np.abs(m_hat - m_data).max(), 0.05)


if __name__ == "__main__":
    unittest.main()
