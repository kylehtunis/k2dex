"""Parity test: Python sampling.py must reproduce the JS sampler's
outputs on the synthetic baseline emitted by
`web/scripts/emit-parity-baseline.ts`.

Baseline is committed at `tests/parity_baseline.json`. To regenerate
(after modifying either side of the port):

    cd web && npm run emit-baseline

Then re-run this test. If it fails, the JS port has drifted from
sampling.py (or vice versa) and needs reconciling before deploy.

Tolerances:
- MF marginals: 1e-9 (deterministic fixed-point, only FP-order
  discrepancies can show up here).
- Greedy descent: identical final team + chain; deltas within 1e-9.
- rank_single_swaps: identical (outIdx, inIdx) tuples in identical
  order; numeric fields within 1e-9.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from sampling import (
    greedy_optimize,
    meanfield_marginals,
    rank_single_swaps,
)
from collections import Counter

from rendering import intra_team_sum_j, nearest_observed, pairwise_j_rows
from rendering_html import species_to_slug


BASELINE_PATH = Path(__file__).parent / "parity_baseline.json"

# Tolerance for floating-point comparisons. Both implementations use
# Float64 in the hot path so we should be at FP-order; allow a little
# slack for differences in summation order.
ATOL = 1e-9


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        raise unittest.SkipTest(
            f"{BASELINE_PATH} missing. Regenerate with `cd web && npm run emit-baseline`."
        )
    with open(BASELINE_PATH) as f:
        return json.load(f)


class TestParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = _load_baseline()
        m = cls.baseline["model"]
        cls.V = m["V"]
        cls.team_size = m["teamSize"]
        cls.species_of = m["speciesOf"]
        cls.item_of = m["itemOf"]
        cls.J = np.array(m["J"], dtype=np.float64).reshape(cls.V, cls.V)
        cls.h = np.array(m["h"], dtype=np.float64)

    def test_meanfield_cases(self) -> None:
        for case in self.baseline["mf"]:
            with self.subTest(case=case["name"]):
                inp = case["input"]
                result = meanfield_marginals(
                    self.J, self.h, self.team_size,
                    fixed=inp["fixed"], excluded=inp["excluded"],
                    field_weight=inp["fieldWeight"],
                    species_of=self.species_of, item_of=self.item_of,
                    n_iters=inp["nIters"], tol=inp["tol"], damp=inp["damp"],
                )
                expected = case["expected"]
                if expected is None:
                    self.assertIsNone(result)
                    continue
                self.assertIsNotNone(result)
                marginals, valid_mask, iters = result
                np.testing.assert_allclose(
                    marginals, np.array(expected["marginals"]), atol=ATOL,
                    err_msg=f"marginals mismatch for {case['name']}",
                )
                np.testing.assert_array_equal(
                    valid_mask.astype(np.uint8),
                    np.array(expected["validMask"], dtype=np.uint8),
                    err_msg=f"validMask mismatch for {case['name']}",
                )
                self.assertEqual(iters, expected["iters"],
                                 f"iters mismatch for {case['name']}")

    def test_greedy_cases(self) -> None:
        for case in self.baseline["greedy"]:
            with self.subTest(case=case["name"]):
                inp = case["input"]
                final_team, chain = greedy_optimize(
                    self.J, self.h, self.team_size,
                    starting_team=inp["startingTeam"],
                    pinned=inp["pinned"],
                    excluded=inp["excluded"],
                    field_weight=inp["fieldWeight"],
                    species_of=self.species_of, item_of=self.item_of,
                    max_swaps=inp["maxSwaps"],
                )
                expected = case["expected"]
                self.assertEqual(sorted(final_team), expected["finalTeam"],
                                 f"finalTeam mismatch for {case['name']}")
                self.assertEqual(len(chain), len(expected["chain"]),
                                 f"chain length mismatch for {case['name']}")
                for i, (py_entry, js_entry) in enumerate(zip(chain, expected["chain"])):
                    self.assertEqual(py_entry["step"], js_entry["step"])
                    self.assertEqual(py_entry["out_idx"], js_entry["outIdx"],
                                     f"out_idx mismatch step {i} of {case['name']}")
                    self.assertEqual(py_entry["in_idx"], js_entry["inIdx"],
                                     f"in_idx mismatch step {i} of {case['name']}")
                    self.assertAlmostEqual(
                        py_entry["delta_E_adj"], js_entry["deltaEAdj"], delta=ATOL,
                        msg=f"delta_E_adj mismatch step {i} of {case['name']}",
                    )
                    self.assertAlmostEqual(
                        py_entry["energy_adj_after"], js_entry["energyAdjAfter"], delta=ATOL,
                    )
                    self.assertAlmostEqual(
                        py_entry["energy_raw_after"], js_entry["energyRawAfter"], delta=ATOL,
                    )
                    self.assertAlmostEqual(
                        py_entry["sum_j_after"], js_entry["sumJAfter"], delta=ATOL,
                    )
                    self.assertEqual(
                        list(py_entry["team_after"]), js_entry["teamAfter"],
                        f"team_after mismatch step {i} of {case['name']}",
                    )

    def test_corpus_cases(self) -> None:
        """nearest_observed parity. The TS side encodes rosters as sorted-
        index keys; here we reconstruct them as frozenset[str] using the
        synthetic vocab (which is what rendering.nearest_observed expects).
        """
        corpus = self.baseline["corpus"]
        vocab = self.baseline["model"]["vocab"]
        team_counts: Counter[frozenset[str]] = Counter()
        for entry in corpus["rosters"]:
            roster = frozenset(vocab[i] for i in entry["team"])
            team_counts[roster] = entry["count"]

        # Python's nearest_observed takes a boolean state array and a vocab.
        V = self.V
        for case in corpus["cases"]:
            with self.subTest(case=case["name"]):
                state = np.zeros(V, dtype=bool)
                for i in case["team"]:
                    state[i] = True
                result = nearest_observed(state, vocab, team_counts)
                expected = case["expected"]
                if expected is None:
                    self.assertIsNone(result)
                    continue
                self.assertIsNotNone(result)
                delta, count = result
                self.assertEqual(delta, expected["delta"],
                                 f"delta mismatch for {case['name']}")
                self.assertEqual(count, expected["count"],
                                 f"count mismatch for {case['name']}")

    def test_obs_cases(self) -> None:
        """intra_team_sum_j + pairwise_j_rows parity. Used by /analysis
        for the per-team observables strip and pairwise-J decomposition
        table — drift here would silently corrupt the diagnostics."""
        V = self.V
        vocab = self.baseline["model"]["vocab"]
        for case in self.baseline["obs"]:
            with self.subTest(case=case["name"]):
                team = case["team"]
                state = np.zeros(V, dtype=bool)
                for i in team:
                    state[i] = True
                self.assertAlmostEqual(
                    intra_team_sum_j(state, self.J),
                    case["expected"]["intraTeamSumJ"],
                    delta=ATOL,
                    msg=f"intra_team_sum_j mismatch for {case['name']}",
                )
                py_rows = pairwise_j_rows(team, vocab, self.J)
                js_rows = case["expected"]["pairwise"]
                self.assertEqual(len(py_rows), len(js_rows))
                for py_row, js_row in zip(py_rows, js_rows):
                    self.assertEqual(py_row.rank, js_row["rank"])
                    js_a = vocab[js_row["idxA"]]
                    js_b = vocab[js_row["idxB"]]
                    # Pair order within an unordered pair isn't promised
                    # to match; check the set instead.
                    self.assertEqual({py_row.name_a, py_row.name_b}, {js_a, js_b})
                    self.assertAlmostEqual(py_row.j_value, js_row["jValue"], delta=ATOL)
                    self.assertAlmostEqual(
                        py_row.pct_of_abs_sum, js_row["pctOfAbsSum"], delta=ATOL,
                    )

    def test_species_to_slug_cases(self) -> None:
        """Sprite-slug rules duplicated in web/src/render/sprite.ts. Any
        rule change must touch both sides + this gate."""
        for case in self.baseline["slugs"]:
            with self.subTest(input=case["input"]):
                self.assertEqual(species_to_slug(case["input"]), case["expected"])

    def test_rank_cases(self) -> None:
        for case in self.baseline["rank"]:
            with self.subTest(case=case["name"]):
                inp = case["input"]
                results = rank_single_swaps(
                    self.J, self.h,
                    team=inp["team"],
                    field_weight=inp["fieldWeight"],
                    species_of=self.species_of, item_of=self.item_of,
                    top_n=inp["topN"],
                )
                expected = case["expected"]
                self.assertEqual(len(results), len(expected),
                                 f"result count mismatch for {case['name']}")
                for i, (py_entry, js_entry) in enumerate(zip(results, expected)):
                    self.assertEqual(py_entry["out_idx"], js_entry["outIdx"],
                                     f"out_idx mismatch entry {i} of {case['name']}")
                    self.assertEqual(py_entry["in_idx"], js_entry["inIdx"],
                                     f"in_idx mismatch entry {i} of {case['name']}")
                    self.assertAlmostEqual(
                        py_entry["delta_E_adj"], js_entry["deltaEAdj"], delta=ATOL,
                        msg=f"delta_E_adj mismatch entry {i} of {case['name']}",
                    )
                    self.assertAlmostEqual(
                        py_entry["delta_E_raw"], js_entry["deltaERaw"], delta=ATOL,
                    )
                    self.assertAlmostEqual(
                        py_entry["delta_sum_j"], js_entry["deltaSumJ"], delta=ATOL,
                    )


if __name__ == "__main__":
    unittest.main()
