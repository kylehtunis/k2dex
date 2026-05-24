"""Precompute PL inverse-Ising fits on Rehnquist court votes at several
N checkpoints. Sibling to precompute.py; outputs static JSON for the
/science page's SCOTUS section.

Output layout (under web/public/scotus/):

    votes.json   {justices: [9 names], votes: [[0|1]*9, ...]}
    fits.json    {"10": {n_used, J, h}, "50": {...}, ..., "all": {...}}

Justices appear in the fixed column order of scotus_votes.txt. Names
follow the Rehnquist court composition Lee et al. studied.

Run locally; inspect; commit. Not invoked from CI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from k2dex.models import fit_pl_ising
INPUT = REPO / "scotus_votes.txt"
OUT_DIR = REPO / "web" / "public" / "scotus"
CHECKPOINTS: list[int | str] = [10, 50, 100, 500, "all"]

JUSTICES = [
    "Rehnquist",
    "Stevens",
    "O'Connor",
    "Scalia",
    "Kennedy",
    "Souter",
    "Thomas",
    "Ginsburg",
    "Breyer",
]


def load_votes(path: Path) -> np.ndarray:
    rows: list[list[int]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(line) != len(JUSTICES):
                raise ValueError(
                    f"unexpected row width {len(line)} (expected {len(JUSTICES)}): {line!r}"
                )
            rows.append([int(c) for c in line])
    return np.array(rows, dtype=np.int32)


def fit_at(X: np.ndarray, n: int | str) -> dict:
    n_used = X.shape[0] if n == "all" else int(n)
    if n_used > X.shape[0]:
        raise ValueError(f"checkpoint {n} exceeds row count {X.shape[0]}")
    J, h = fit_pl_ising(X[:n_used])
    return {
        "n_used": n_used,
        "J": J.tolist(),
        "h": h.tolist(),
    }


def main() -> None:
    X = load_votes(INPUT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "votes.json").write_text(
        json.dumps(
            {"justices": JUSTICES, "votes": X.tolist()},
            indent=None,
            separators=(",", ":"),
        )
    )
    fits = {str(c): fit_at(X, c) for c in CHECKPOINTS}
    (OUT_DIR / "fits.json").write_text(json.dumps(fits, indent=2))
    print(f"wrote {OUT_DIR}/votes.json and {OUT_DIR}/fits.json (N rows = {X.shape[0]})")


if __name__ == "__main__":
    main()
