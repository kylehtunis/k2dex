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

from k2dex.loaders import ABILITY_TRACK, ITEM_TRACK, SpeciesModel
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
        latest_date = "2026-01-15T00:00:00.000Z"

        fake_builder = mock.Mock(return_value=SpeciesModel(
            vocab=vocab, m=m, J=J, h=h, team_counts=team_counts,
            species_of=species_of, track_values_of=[[] for _ in vocab],
            track_specs=[], latest_date=latest_date,
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
            self.assertNotIn("feature_dimensions", meta)
            self.assertEqual(meta["latest_tournament_date"], latest_date)
            self.assertEqual(meta["vocab"], vocab)
            # v4 factored schema: species-only model has one site per feature,
            # no tracks, and empty per-feature track_values.
            self.assertNotIn("species_of", meta)
            self.assertNotIn("item_of", meta)
            self.assertEqual(meta["sites"], species_of)
            self.assertEqual(meta["site_of"], list(range(V)))
            self.assertEqual(meta["tracks"], [])
            self.assertEqual(meta["track_values"], [[] for _ in range(V)])
            self.assertEqual(meta["n_corpus_teams"], 14)
            self.assertEqual(meta["schema_version"], 4)
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
        """A species+item+ability model derives two tracks (item, ability) and
        per-feature track_values, grouping features onto their shared species
        site. Skips team_counts (species_graph fit is out of this round-trip's
        scope)."""
        rng = np.random.default_rng(7)
        # Three species (A, B, C) with 2 / 1 / 2 (item, ability) states -> V = 5.
        species_of = ["A", "A", "B", "C", "C"]
        item_of = ["Life Orb", "None", "Focus Sash", "Leftovers", "None"]
        ability_of = ["Overgrow", "Chlorophyll", "Blaze", "Torrent", "Rain Dish"]
        track_values_of: list[list[str | None]] = [[i, a] for i, a in zip(item_of, ability_of)]
        vocab = [
            precompute_format(s, i, a)
            for s, i, a in zip(species_of, item_of, ability_of)
        ]
        V = len(vocab)
        m = rng.uniform(0.01, 0.4, size=V).astype(np.float64)
        h = rng.standard_normal(V).astype(np.float64)
        J_asym = rng.standard_normal((V, V))
        J = 0.5 * (J_asym + J_asym.T)
        np.fill_diagonal(J, 0.0)
        team_counts = Counter({frozenset(vocab[:5]): 3})
        latest_date = "2026-01-15T00:00:00.000Z"

        fake_builder = mock.Mock(return_value=SpeciesModel(
            vocab=vocab, m=m, J=J, h=h, team_counts=team_counts,
            species_of=species_of, track_values_of=track_values_of,
            track_specs=[ITEM_TRACK, ABILITY_TRACK], latest_date=latest_date,
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
                        skip_team_counts=True,
                        force=True,
                    )
            with open(Path(tmp) / "synthetic-si" / "meta.json") as f:
                meta = json.load(f)
        self.assertNotIn("feature_dimensions", meta)
        self.assertEqual(meta["schema_version"], 4)
        self.assertEqual(meta["sites"], ["A", "B", "C"])
        self.assertEqual(meta["site_of"], [0, 0, 1, 2, 2])
        self.assertEqual(meta["tracks"], [
            {"name": "item", "cardinality": 1,
             "crossSlotUnique": True, "withinSlotUnique": False},
            {"name": "ability", "cardinality": 1,
             "crossSlotUnique": False, "withinSlotUnique": False},
        ])
        self.assertEqual(
            meta["track_values"],
            [["Life Orb", "Overgrow"], ["None", "Chlorophyll"],
             ["Focus Sash", "Blaze"], ["Leftovers", "Torrent"], ["None", "Rain Dish"]],
        )


def precompute_format(species: str, item: str | None, ability: str) -> str:
    base = species if item in (None, "None") else f"{species} @ {item}"
    return f"{base} ({ability})"


if __name__ == "__main__":
    unittest.main()
