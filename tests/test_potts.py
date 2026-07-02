"""Tests for the Potts / item-modulation decomposition (`k2dex.potts`).

Exercises the block decomposition invariants (zero-sum gauge, exact
reconstruction), the single-item degeneracy that mirrors the q=2 Ising special
case, the APC formula, and end-to-end scoring on a hand-built model -- all on
synthetic (J, h) so no artifact files are needed.
"""
from __future__ import annotations

import unittest

import numpy as np

from k2dex import potts


def _toy_model(J: np.ndarray) -> potts.FittedModel:
    """A 3-site model: species A has 2 items, B has 2 items, C has 1 item."""
    vocab = ["A @ i1", "A @ i2", "B @ j1", "B @ j2", "C @ k1"]
    species_of = ["A", "A", "B", "B", "C"]
    item_of: list[str | None] = ["i1", "i2", "j1", "j2", "k1"]
    V = len(vocab)
    m = np.full(V, 0.1)
    h = np.zeros(V)
    return potts.FittedModel(
        vocab=vocab, species_of=species_of, item_of=item_of,
        J=J, h=h, m=m, team_size=4, n_corpus_teams=1000,
    )


def _random_symmetric(V: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((V, V))
    J = 0.5 * (A + A.T)
    np.fill_diagonal(J, 0.0)
    return J


class TestDecomposeBlock(unittest.TestCase):
    def test_reconstruction_and_zero_sum(self) -> None:
        rng = np.random.default_rng(1)
        B = rng.standard_normal((4, 3))
        d = potts.decompose_block(B)
        recon = (d.synergy + d.row_effects[:, None]
                 + d.col_effects[None, :] + d.interaction)
        np.testing.assert_allclose(recon, B, atol=1e-12)
        # zero-sum gauge: effects sum to zero, interaction sums to zero on both axes.
        self.assertAlmostEqual(float(d.row_effects.sum()), 0.0, places=12)
        self.assertAlmostEqual(float(d.col_effects.sum()), 0.0, places=12)
        np.testing.assert_allclose(d.interaction.sum(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(d.interaction.sum(axis=1), 0.0, atol=1e-12)

    def test_synergy_is_grand_mean(self) -> None:
        B = np.array([[1.0, 3.0], [5.0, 7.0]])
        self.assertAlmostEqual(potts.decompose_block(B).synergy, 4.0, places=12)

    def test_row_modulation_is_column_centered(self) -> None:
        rng = np.random.default_rng(2)
        B = rng.standard_normal((5, 4))
        mod = potts.row_modulation(B)
        np.testing.assert_allclose(mod.mean(axis=0), 0.0, atol=1e-12)
        # row_modulation == row_effects + interaction.
        d = potts.decompose_block(B)
        np.testing.assert_allclose(mod, d.row_effects[:, None] + d.interaction, atol=1e-12)


class TestSingleItemDegeneracy(unittest.TestCase):
    """A species with one item state has no item variation: its modulation
    residual is identically zero (the q=2 Ising special case)."""

    def test_single_row_block_has_zero_modulation(self) -> None:
        B = np.array([[1.0, 2.0, 3.0]])  # k_P = 1
        d = potts.decompose_block(B)
        np.testing.assert_allclose(d.row_effects, 0.0, atol=1e-12)
        np.testing.assert_allclose(potts.row_modulation(B), 0.0, atol=1e-12)

    def test_single_item_species_scores_zero(self) -> None:
        model = _toy_model(_random_symmetric(5, seed=3))
        rows = {r.species: r for r in potts.modulation_scores(model)}
        self.assertEqual(rows["C"].n_items, 1)
        # C's own item choice cannot modulate anything (one state only).
        self.assertAlmostEqual(rows["C"].mod_frob, 0.0, places=12)
        self.assertAlmostEqual(rows["C"].mod_frac, 0.0, places=12)


class TestModulationScores(unittest.TestCase):
    def test_frac_in_unit_interval_and_sorted(self) -> None:
        model = _toy_model(_random_symmetric(5, seed=4))
        rows = potts.modulation_scores(model)
        fracs = [r.mod_frac for r in rows]
        for f in fracs:
            self.assertGreaterEqual(f, -1e-12)
            self.assertLessEqual(f, 1.0 + 1e-12)
        self.assertEqual(fracs, sorted(fracs, reverse=True))

    def test_appearances_matches_marginal_times_corpus(self) -> None:
        model = _toy_model(_random_symmetric(5, seed=5))
        # species A occupies two features, each m=0.1, corpus=1000 -> ~200.
        self.assertAlmostEqual(model.appearances("A"), 200.0, places=6)


class TestItemModulation(unittest.TestCase):
    def test_partner_shift_signs(self) -> None:
        # Build a block where A's item i1 couples strongly to B, i2 not at all.
        J = np.zeros((5, 5))
        # A@i1 (idx 0) <-> B@j1 (idx 2): strong positive.
        J[0, 2] = J[2, 0] = 2.0
        model = _toy_model(J)
        detail = {d.item: d for d in potts.item_modulation(model, "A")}
        # i1 should pull toward B (positive), i2 away from B (negative), since the
        # item-agnostic mean over {i1,i2} is split symmetrically.
        self.assertGreater(detail["i1"].pulls_toward[0][1], 0.0)
        self.assertEqual(detail["i1"].pulls_toward[0][0], "B")
        self.assertLess(detail["i2"].pulls_away[0][1], 0.0)


class TestAPC(unittest.TestCase):
    def test_apc_formula_matches_manual(self) -> None:
        F = np.array([
            [0.0, 2.0, 4.0],
            [2.0, 0.0, 6.0],
            [4.0, 6.0, 0.0],
        ])
        apc = potts._apc(F)
        row_mean = F.sum(axis=1) / 2.0                 # (6, 8, 10) / 2
        grand = F.sum() / (3 * 2)                       # 24 / 6 = 4
        expected = np.outer(row_mean, row_mean) / grand
        np.testing.assert_allclose(apc, expected, atol=1e-12)

    def test_graph_symmetry_and_zero_diagonal(self) -> None:
        model = _toy_model(_random_symmetric(5, seed=6))
        g = potts.species_apc_graph(model)
        np.testing.assert_allclose(g.frob, g.frob.T, atol=1e-12)
        np.testing.assert_allclose(g.synergy, g.synergy.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(g.corrected), 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
