"""Round-trip tests for the precompute serialization layer.

These exercise the binary packing and JSON shape without running the
full Limitless builder (slow). Synthetic (J, h, m, team_counts) are
fabricated, serialized via the precompute helpers, then loaded back
manually to assert equality within float32 tolerance.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np

from k2dex import potts
from scripts import precompute


class TestPackLowerTriangle(unittest.TestCase):
    def test_round_trip_symmetric_matrix(self) -> None:
        rng = np.random.default_rng(0)
        V = 8
        J_asym = rng.standard_normal((V, V))
        J = 0.5 * (J_asym + J_asym.T)
        np.fill_diagonal(J, 0.0)

        flat = precompute.pack_lower_triangle(J)
        self.assertEqual(flat.dtype, np.float32)
        self.assertEqual(flat.shape, (V * (V - 1) // 2,))

        # Reconstruct: same scheme JS will use
        J_reco = np.zeros((V, V), dtype=np.float32)
        rows, cols = np.tril_indices(V, k=-1)
        J_reco[rows, cols] = flat
        J_reco = J_reco + J_reco.T

        # Float32 quantization tolerance
        np.testing.assert_allclose(J_reco, J.astype(np.float32), atol=1e-6)

    def test_ordering_is_row_major_lower(self) -> None:
        # Make J[i, j] = 100*i + j for i > j so we can read off the order
        # in the flat output by eye.
        V = 4
        J = np.zeros((V, V), dtype=np.float64)
        for i in range(V):
            for j in range(i):
                J[i, j] = 100 * i + j
                J[j, i] = 100 * i + j
        flat = precompute.pack_lower_triangle(J)
        # Expected order: (1,0), (2,0), (2,1), (3,0), (3,1), (3,2)
        expected = np.array([100, 200, 201, 300, 301, 302], dtype=np.float32)
        np.testing.assert_array_equal(flat, expected)


class TestSerializeTeamCounts(unittest.TestCase):
    def test_index_keys_sorted_ascending(self) -> None:
        vocab = ["A", "B", "C", "D", "E", "F"]
        team_counts = Counter({
            frozenset(["E", "A", "C", "B", "D", "F"]): 7,
        })
        out = precompute.serialize_team_counts(team_counts, vocab)
        # Indices are 0..5, sorted; "-"-joined
        self.assertEqual(out, {"0-1-2-3-4-5": 7})

    def test_skips_out_of_vocab_rosters(self) -> None:
        vocab = ["A", "B", "C"]
        team_counts = Counter({
            frozenset(["A", "B", "C"]): 5,
            frozenset(["A", "B", "Z"]): 99,  # Z not in vocab -> dropped
        })
        with contextlib.redirect_stdout(io.StringIO()):
            out = precompute.serialize_team_counts(team_counts, vocab)
        self.assertEqual(out, {"0-1-2": 5})


class TestWriteModelRoundTrip(unittest.TestCase):
    def test_round_trip_via_write_model(self) -> None:
        """End-to-end: pass a synthetic model through `write_model` by
        mocking the builder, then read back the on-disk files and
        verify they reconstruct the inputs."""
        rng = np.random.default_rng(42)
        V = 12
        vocab = [f"Pkmn{i}" for i in range(V)]
        m = rng.uniform(0.01, 0.4, size=V).astype(np.float64)
        h = rng.standard_normal(V).astype(np.float64)
        J_asym = rng.standard_normal((V, V))
        J = 0.5 * (J_asym + J_asym.T)
        np.fill_diagonal(J, 0.0)
        team_counts = Counter({
            frozenset([vocab[0], vocab[1], vocab[2], vocab[3], vocab[4], vocab[5]]): 10,
            frozenset([vocab[6], vocab[7], vocab[8], vocab[9], vocab[10], vocab[11]]): 4,
        })
        species_of = list(vocab)
        item_of = [None] * V
        latest_date = "2026-01-15T00:00:00.000Z"

        fake_builder = mock.Mock(return_value=(
            vocab, m, J, h, team_counts, species_of, item_of, latest_date,
        ))

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(precompute.MODEL_BUILDERS, {"species": fake_builder}):
                with contextlib.redirect_stdout(io.StringIO()):
                    precompute.write_model(
                        slug="synthetic",
                        display_name="Synthetic Test",
                        regulation="test",
                        builder_type="species",
                        lam=10.0,
                        out_dir=Path(tmp),
                        force=True,
                    )

            model_dir = Path(tmp) / "synthetic"
            self.assertTrue((model_dir / "meta.json").exists())
            self.assertTrue((model_dir / "J.bin").exists())
            self.assertTrue((model_dir / "h.bin").exists())
            self.assertTrue((model_dir / "m.bin").exists())
            self.assertTrue((model_dir / "team_counts.json").exists())

            with open(model_dir / "meta.json") as f:
                meta = json.load(f)
            self.assertEqual(meta["V"], V)
            self.assertEqual(meta["id"], "synthetic")
            self.assertEqual(meta["display_name"], "Synthetic Test")
            self.assertEqual(meta["regulation"], "test")
            self.assertEqual(meta["feature_dimensions"], 1)
            self.assertEqual(meta["latest_tournament_date"], latest_date)
            self.assertEqual(meta["vocab"], vocab)
            # v3 factored schema: species-only model has one site per feature,
            # no tracks, and empty per-feature track_values.
            self.assertNotIn("species_of", meta)
            self.assertNotIn("item_of", meta)
            self.assertEqual(meta["sites"], species_of)
            self.assertEqual(meta["site_of"], list(range(V)))
            self.assertEqual(meta["tracks"], [])
            self.assertEqual(meta["track_values"], [[] for _ in range(V)])
            self.assertEqual(meta["n_corpus_teams"], 14)
            self.assertEqual(meta["schema_version"], 3)
            self.assertEqual(meta["fit"]["lambda"], 10.0)

            h_read = np.fromfile(model_dir / "h.bin", dtype=np.float32)
            m_read = np.fromfile(model_dir / "m.bin", dtype=np.float32)
            np.testing.assert_allclose(h_read, h.astype(np.float32), atol=1e-6)
            np.testing.assert_allclose(m_read, m.astype(np.float32), atol=1e-6)

            j_flat = np.fromfile(model_dir / "J.bin", dtype=np.float32)
            self.assertEqual(j_flat.shape, (V * (V - 1) // 2,))
            J_reco = np.zeros((V, V), dtype=np.float32)
            rows, cols = np.tril_indices(V, k=-1)
            J_reco[rows, cols] = j_flat
            J_reco = J_reco + J_reco.T
            np.testing.assert_allclose(J_reco, J.astype(np.float32), atol=1e-6)

            with open(model_dir / "team_counts.json") as f:
                tc = json.load(f)
            self.assertEqual(tc, {"0-1-2-3-4-5": 10, "6-7-8-9-10-11": 4})

    def test_species_item_factored_schema(self) -> None:
        """A species+item model derives one 'item' track and per-feature
        track_values, grouping features onto their shared species site."""
        rng = np.random.default_rng(7)
        # Three species (A, B, C) with 2 / 1 / 2 item states -> V = 5.
        species_of = ["A", "A", "B", "C", "C"]
        item_of = ["Life Orb", "None", "Focus Sash", "Leftovers", "None"]
        vocab = [precompute_format(s, i) for s, i in zip(species_of, item_of)]
        V = len(vocab)
        m = rng.uniform(0.01, 0.4, size=V).astype(np.float64)
        h = rng.standard_normal(V).astype(np.float64)
        J_asym = rng.standard_normal((V, V))
        J = 0.5 * (J_asym + J_asym.T)
        np.fill_diagonal(J, 0.0)
        team_counts = Counter({frozenset(vocab[:5]): 3})
        latest_date = "2026-01-15T00:00:00.000Z"

        fake_builder = mock.Mock(return_value=(
            vocab, m, J, h, team_counts, species_of, item_of, latest_date,
        ))

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(precompute.MODEL_BUILDERS, {"species_item": fake_builder}):
                with contextlib.redirect_stdout(io.StringIO()):
                    precompute.write_model(
                        slug="synthetic-si",
                        display_name="Synthetic SI",
                        regulation="test",
                        builder_type="species_item",
                        lam=4.5,
                        out_dir=Path(tmp),
                        force=True,
                    )
            with open(Path(tmp) / "synthetic-si" / "meta.json") as f:
                meta = json.load(f)
        self.assertEqual(meta["feature_dimensions"], 2)
        self.assertEqual(meta["schema_version"], 3)
        self.assertEqual(meta["sites"], ["A", "B", "C"])
        self.assertEqual(meta["site_of"], [0, 0, 1, 2, 2])
        self.assertEqual(meta["tracks"], [{"name": "item", "unique": True}])
        self.assertEqual(
            meta["track_values"],
            [["Life Orb"], ["None"], ["Focus Sash"], ["Leftovers"], ["None"]],
        )

    def test_artifact_reloads_through_potts_loader(self) -> None:
        """The factored schema round-trips back to the (species, item) view,
        and species_graph.json's flattened upper triangle matches a direct
        species_apc_graph call. Covers the write -> load_fitted_model path the
        Potts analysis and the webapp's species graph both depend on."""
        rng = np.random.default_rng(11)
        species_of = ["A", "A", "B", "C", "C"]
        item_of = ["Life Orb", "None", "Focus Sash", "Leftovers", "None"]
        vocab = [precompute_format(s, i) for s, i in zip(species_of, item_of)]
        V = len(vocab)
        m = rng.uniform(0.01, 0.4, size=V).astype(np.float64)
        h = rng.standard_normal(V).astype(np.float64)
        J_asym = rng.standard_normal((V, V))
        J = 0.5 * (J_asym + J_asym.T)
        np.fill_diagonal(J, 0.0)
        team_counts = Counter({frozenset(vocab[:5]): 3})

        fake_builder = mock.Mock(return_value=(
            vocab, m, J, h, team_counts, species_of, item_of,
            "2026-01-15T00:00:00.000Z",
        ))

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(precompute.MODEL_BUILDERS, {"species_item": fake_builder}):
                with contextlib.redirect_stdout(io.StringIO()):
                    precompute.write_model(
                        slug="synthetic-si", display_name="Synthetic SI",
                        regulation="test", builder_type="species_item",
                        lam=4.5, out_dir=Path(tmp), force=True,
                    )
            model_dir = Path(tmp) / "synthetic-si"
            loaded = potts.load_fitted_model(model_dir)
            with open(model_dir / "species_graph.json") as f:
                sg_data = json.load(f)

        # The (species, item) view is rebuilt from sites + site_of + track_values.
        self.assertEqual(loaded.vocab, vocab)
        self.assertEqual(loaded.species_of, species_of)
        self.assertEqual(loaded.item_of, item_of)
        # J.bin is float32, so compare at float32 resolution.
        np.testing.assert_allclose(loaded.J, J, atol=1e-6)
        np.testing.assert_allclose(loaded.h, h, atol=1e-6)
        np.testing.assert_allclose(loaded.m, m, atol=1e-6)

        # species_graph.json is the upper triangle in row-major order; catches a
        # transpose or an off-by-one in the flattening loop.
        sg = potts.species_apc_graph(loaded)
        self.assertEqual(sg_data["species"], list(sg.species))
        S = len(sg.species)
        expect_syn = [sg.synergy[i, j] for i in range(S) for j in range(i + 1, S)]
        self.assertEqual(len(sg_data["synergy_ut"]), S * (S - 1) // 2)
        np.testing.assert_allclose(sg_data["synergy_ut"], expect_syn, atol=1e-5)
        # Only the signed synergy ships; `corrected` stays a Python analysis
        # output (see species_apc_graph) and must not reappear in the artifact.
        self.assertNotIn("corrected_ut", sg_data)

    def test_load_fitted_model_rejects_other_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "stale"
            model_dir.mkdir()
            with open(model_dir / "meta.json", "w") as f:
                json.dump({"schema_version": 2, "V": 1}, f)
            with self.assertRaisesRegex(ValueError, "schema_version"):
                potts.load_fitted_model(model_dir)


def precompute_format(species: str, item: str | None) -> str:
    return species if item is None else f"{species} @ {item}"


if __name__ == "__main__":
    unittest.main()
