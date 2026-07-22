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


def _toy_model_ability(J: np.ndarray) -> potts.FittedModel:
    """A 3-site model carrying an ability track. Species A's item i1 is split
    across two abilities (item-modulated AND ability-modulated); B has two items
    each with one ability (item-modulated, ability-degenerate); C is a single
    state. Layout (V=6): A -> [i1/p, i1/q, i2/p], B -> [j1/p, j2/p], C -> [k1/p].
    """
    vocab = [
        "A @ i1 (p)", "A @ i1 (q)", "A @ i2 (p)",
        "B @ j1 (p)", "B @ j2 (p)", "C @ k1 (p)",
    ]
    species_of = ["A", "A", "A", "B", "B", "C"]
    item_of: list[str | None] = ["i1", "i1", "i2", "j1", "j2", "k1"]
    track_values_of: list[list[str | None]] = [
        ["i1", "p"], ["i1", "q"], ["i2", "p"],
        ["j1", "p"], ["j2", "p"], ["k1", "p"],
    ]
    V = len(vocab)
    m = np.full(V, 0.1)
    h = np.zeros(V)
    return potts.FittedModel(
        vocab=vocab, species_of=species_of, item_of=item_of,
        J=J, h=h, m=m, team_size=4, n_corpus_teams=1000,
        track_values_of=track_values_of,
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


class TestHierarchicalDecomposition(unittest.TestCase):
    def test_reconstructs_full_block(self) -> None:
        model = _toy_model_ability(_random_symmetric(6, seed=10))
        for P in model.sites:
            for Q in model.sites:
                if P == Q:
                    continue
                hd = potts.decompose_block_hierarchical(model, P, Q)
                np.testing.assert_allclose(
                    hd.reconstruct(), potts.species_block(model, P, Q), atol=1e-12,
                    err_msg=f"reconstruction mismatch for {P}->{Q}",
                )

    def test_item_level_zero_sum(self) -> None:
        # The item-level ANOVA is the zero-sum gauge of the item block.
        model = _toy_model_ability(_random_symmetric(6, seed=11))
        hd = potts.decompose_block_hierarchical(model, "A", "B")
        self.assertAlmostEqual(float(hd.item_row.sum()), 0.0, places=12)
        self.assertAlmostEqual(float(hd.item_col.sum()), 0.0, places=12)
        np.testing.assert_allclose(hd.item_interaction.sum(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(hd.item_interaction.sum(axis=1), 0.0, atol=1e-12)

    def test_ability_residual_usage_weighted_zero(self) -> None:
        # Within each (item_P, item_Q) cell the residual's usage-weighted mean
        # over abilities is zero: collapsing it with both weight matrices -> 0.
        model = _toy_model_ability(_random_symmetric(6, seed=12))
        hd = potts.decompose_block_hierarchical(model, "A", "B")
        _, _, cA = potts._item_groups(model, "A")
        _, _, cB = potts._item_groups(model, "B")
        collapsed = cA @ hd.ability_residual @ cB.T
        np.testing.assert_allclose(collapsed, 0.0, atol=1e-12)

    def test_synergy_is_usage_weighted_full_alphabet_mean(self) -> None:
        model = _toy_model_ability(_random_symmetric(6, seed=13))
        model.m[0], model.m[1] = 0.25, 0.05  # skew A's i1 abilities
        hd = potts.decompose_block_hierarchical(model, "A", "B")
        wA = model.item_weights("A")
        wB = model.item_weights("B")
        B = potts.species_block(model, "A", "B")
        self.assertAlmostEqual(hd.synergy, float(wA @ B @ wB), places=12)

    def test_degenerate_reduces_to_flat_decompose(self) -> None:
        # A model with no ability track (each item one state) must give an
        # identically-zero ability residual and an item ANOVA equal to the flat
        # decompose_block of the full block.
        model = _toy_model(_random_symmetric(5, seed=14))
        B = potts.species_block(model, "A", "B")
        flat = potts.decompose_block(B)
        hd = potts.decompose_block_hierarchical(model, "A", "B")
        np.testing.assert_allclose(hd.ability_residual, 0.0, atol=1e-12)
        self.assertAlmostEqual(hd.item_synergy, flat.synergy, places=12)
        np.testing.assert_allclose(hd.item_row, flat.row_effects, atol=1e-12)
        np.testing.assert_allclose(hd.item_col, flat.col_effects, atol=1e-12)
        np.testing.assert_allclose(hd.item_interaction, flat.interaction, atol=1e-12)


class TestAbilityModulation(unittest.TestCase):
    def test_ability_degenerate_species_score_zero(self) -> None:
        model = _toy_model_ability(_random_symmetric(6, seed=15))
        rows = {r.species: r for r in potts.modulation_scores(model)}
        # B: two items, each a single ability -> ability-degenerate.
        self.assertEqual(rows["B"].n_states, rows["B"].n_items)
        self.assertAlmostEqual(rows["B"].ability_mod_frob, 0.0, places=12)
        self.assertAlmostEqual(rows["B"].ability_mod_frac, 0.0, places=12)
        # C: single state -> both item and ability modulation zero.
        self.assertAlmostEqual(rows["C"].mod_frob, 0.0, places=12)
        self.assertAlmostEqual(rows["C"].ability_mod_frob, 0.0, places=12)

    def test_ability_split_species_scores_nonzero(self) -> None:
        # Give A's i1/p a coupling to B that i1/q lacks, so A's ability genuinely
        # modulates its coupling. A (item i1 split p/q) must score > 0.
        J = np.zeros((6, 6))
        J[0, 3] = J[3, 0] = 2.0  # A@i1(p) <-> B@j1(p)
        model = _toy_model_ability(J)
        rows = {r.species: r for r in potts.modulation_scores(model)}
        self.assertGreater(rows["A"].n_states, rows["A"].n_items)
        self.assertGreater(rows["A"].ability_mod_frob, 0.0)

    def test_flat_model_ability_mod_is_zero(self) -> None:
        # The pre-ability (single-track) model has no residual: item mod matches
        # the old flat behaviour and ability mod is identically zero.
        model = _toy_model(_random_symmetric(5, seed=16))
        for r in potts.modulation_scores(model):
            self.assertAlmostEqual(r.ability_mod_frob, 0.0, places=12)


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

    def test_partners_exclude_the_species_itself(self) -> None:
        # A single-signed deviation row (i1 couples only positively, to B) must
        # not fall back on the self-block's structurally-zero columns for its
        # opposite-sign extremes: every partner listed is a *different* species.
        J = np.zeros((5, 5))
        J[0, 2] = J[2, 0] = 2.0
        model = _toy_model(J)
        for d in potts.item_modulation(model, "A"):
            partners = [sp for sp, _ in d.pulls_toward + d.pulls_away]
            self.assertNotIn("A", partners)


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

    def test_synergy_is_usage_weighted_mean(self) -> None:
        # A has 2 items, B has 2 items. Skew A's marginals so item i1
        # dominates; the graph synergy must equal the m-weighted block mean,
        # not the flat grand mean.
        J = _random_symmetric(5, seed=3)
        model = _toy_model(J)
        model.m[0], model.m[1] = 0.30, 0.02   # A: i1 dominant
        model.m[2], model.m[3] = 0.05, 0.15   # B: j2 dominant
        g = potts.species_apc_graph(model)
        wA = model.item_weights("A")
        wB = model.item_weights("B")
        B = J[np.ix_([0, 1], [2, 3])]
        expected = float(wA @ B @ wB)
        ai, bi = g.species.index("A"), g.species.index("B")
        self.assertAlmostEqual(g.synergy[ai, bi], expected, places=12)
        # It should differ from the flat grand mean under skewed usage.
        self.assertNotAlmostEqual(g.synergy[ai, bi], float(B.mean()), places=4)

    def test_single_item_species_synergy_unchanged(self) -> None:
        # C has one item, so its weight is 1 and its synergy equals the plain
        # per-partner-item mean regardless of usage weighting.
        J = _random_symmetric(5, seed=7)
        model = _toy_model(J)
        g = potts.species_apc_graph(model)
        wA = model.item_weights("A")
        block = J[np.ix_([4], [0, 1])]  # C x A
        expected = float(block[0] @ wA)
        ci, ai = g.species.index("C"), g.species.index("A")
        self.assertAlmostEqual(g.synergy[ci, ai], expected, places=12)


if __name__ == "__main__":
    unittest.main()
