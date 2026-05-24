"""Smoke tests for scotus_precompute.py.

Verifies output shape and finiteness. Does not pin specific J values —
PL convergence depends on regularization and is exercised separately by
test_models.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "web" / "public" / "scotus"
EXPECTED_CHECKPOINTS = [10, 50, 100, 500, "all"]
N_JUSTICES = 9


class TestScotusPrecompute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "scotus_precompute.py")],
            check=True,
            cwd=REPO,
        )

    def test_votes_json_shape(self):
        with (OUT_DIR / "votes.json").open() as f:
            data = json.load(f)
        self.assertIn("justices", data)
        self.assertIn("votes", data)
        self.assertEqual(len(data["justices"]), N_JUSTICES)
        self.assertGreater(len(data["votes"]), 100)
        for row in data["votes"]:
            self.assertEqual(len(row), N_JUSTICES)
            for v in row:
                self.assertIn(v, (0, 1))

    def test_fits_json_checkpoints(self):
        with (OUT_DIR / "fits.json").open() as f:
            data = json.load(f)
        self.assertEqual(set(data.keys()), {str(c) for c in EXPECTED_CHECKPOINTS})
        for key, fit in data.items():
            J = fit["J"]
            h = fit["h"]
            n_used = fit["n_used"]
            self.assertEqual(len(J), N_JUSTICES)
            for row in J:
                self.assertEqual(len(row), N_JUSTICES)
                for v in row:
                    self.assertTrue(v == v)  # not NaN
            self.assertEqual(len(h), N_JUSTICES)
            for v in h:
                self.assertTrue(v == v)
            if key == "all":
                self.assertGreater(n_used, 500)
            else:
                self.assertEqual(n_used, int(key))


if __name__ == "__main__":
    unittest.main()
