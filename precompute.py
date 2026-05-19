"""Offline precompute pipeline for the static web build.

Fits both PL inverse-Ising models (Species, Species @ Item) on the live
Limitless corpus and serializes the artifacts the JS client needs into
`web/public/models/{species,species_item}/`:

    meta.json         — vocab, species_of, item_of, scalars, fit hyperparams
    J.bin             — float32 lower triangle, V*(V-1)/2 entries
                        ordering: [J[i,j] for i in range(1,V) for j in range(i)]
    h.bin             — float32, V entries
    m.bin             — float32, V entries
    team_counts.json  — sorted-index "-"-joined roster keys -> int count

Run locally; inspect artifacts; commit. Not invoked from CI per the
static-deploy plan: humans should eyeball every model refresh before it
goes live.

Float32 is sufficient for J/h/m at our scale (values O(1e-3..5),
regularized, no accumulating round-off). Lower-triangle packing halves
J's bytes; the JS loader reconstructs the symmetric matrix on init.

Usage:
    python precompute.py                # build both models -> default dir
    python precompute.py --out custom/  # alternate output root
    python precompute.py --model species
    python precompute.py --skip-team-counts  # debug: skip the corpus index
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from constants import (
    PHASE2_LR_C,
    PHASE2_MIN_TEAM_COUNT,
    PHASE2_MIN_TEAMS,
    TEAM_SIZE,
)
from loaders import build_species_item_model, build_species_model


DEFAULT_OUT_DIR = Path(__file__).parent / "web" / "public" / "models"

MODEL_BUILDERS = {
    "species": build_species_model,
    "species_item": build_species_item_model,
}


def pack_lower_triangle(J: NDArray[np.floating]) -> NDArray[np.float32]:
    """Extract J's strict lower triangle in row-major order as a flat
    float32 array of length V*(V-1)/2.

    Ordering: [J[1,0], J[2,0], J[2,1], J[3,0], J[3,1], J[3,2], ...].
    Equivalent to `J[np.tril_indices(V, k=-1)]`, which `np.tril_indices`
    returns in exactly this order.
    """
    V = J.shape[0]
    rows, cols = np.tril_indices(V, k=-1)
    return J[rows, cols].astype(np.float32)


def serialize_team_counts(
    team_counts: Counter,
    vocab: list[str],
) -> dict[str, int]:
    """Project frozenset[str] keys onto sorted-index lists, joined by "-".

    e.g. team {vocab[5], vocab[12], vocab[33], vocab[87], vocab[102], vocab[134]}
    -> "5-12-33-87-102-134".

    Skips rosters with any out-of-vocab member (defensive; the Phase 3
    builder already filters these, the Phase 2 builder doesn't filter
    explicitly but every roster member is by construction in vocab when
    the corpus indices are right).
    """
    name_to_idx = {name: i for i, name in enumerate(vocab)}
    out: dict[str, int] = {}
    skipped = 0
    for roster, count in team_counts.items():
        try:
            idxs = sorted(name_to_idx[name] for name in roster)
        except KeyError:
            skipped += 1
            continue
        out["-".join(str(i) for i in idxs)] = int(count)
    if skipped:
        print(f"  warn: skipped {skipped} rosters with out-of-vocab members")
    return out


def write_model(
    name: str,
    out_dir: Path,
    *,
    skip_team_counts: bool = False,
) -> None:
    print(f"\n=== Building model: {name} ===")
    builder = MODEL_BUILDERS[name]
    vocab, m, J, h, team_counts, species_of, item_of = builder()
    V = len(vocab)
    n_teams = int(sum(team_counts.values()))
    print(f"  V = {V}, corpus teams = {n_teams:,}")

    model_dir = out_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    j_flat = pack_lower_triangle(J)
    j_flat.tofile(model_dir / "J.bin")
    h.astype(np.float32).tofile(model_dir / "h.bin")
    m.astype(np.float32).tofile(model_dir / "m.bin")
    print(f"  J.bin: {j_flat.nbytes:,} bytes ({V}*{V-1}/2 = {len(j_flat):,} float32)")
    print(f"  h.bin: {V * 4:,} bytes")
    print(f"  m.bin: {V * 4:,} bytes")

    meta = {
        "name": name,
        "V": V,
        "team_size": TEAM_SIZE,
        "n_corpus_teams": n_teams,
        "vocab": vocab,
        "species_of": species_of,
        "item_of": item_of,
        "fit": {
            "method": "pseudo_likelihood",
            "C": PHASE2_LR_C,
            "min_team_count": PHASE2_MIN_TEAM_COUNT,
            "min_teams": PHASE2_MIN_TEAMS,
        },
        "schema_version": 1,
    }
    with open(model_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=None, separators=(",", ":"))
    print(f"  meta.json: {(model_dir / 'meta.json').stat().st_size:,} bytes")

    if skip_team_counts:
        print("  (skip-team-counts: not writing team_counts.json)")
        return

    tc_serialized = serialize_team_counts(team_counts, vocab)
    with open(model_dir / "team_counts.json", "w") as f:
        json.dump(tc_serialized, f, indent=None, separators=(",", ":"))
    tc_path = model_dir / "team_counts.json"
    print(f"  team_counts.json: {tc_path.stat().st_size:,} bytes "
          f"({len(tc_serialized):,} unique rosters)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output root directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_BUILDERS),
        action="append",
        help="Which model(s) to build; repeatable. Default: all.",
    )
    parser.add_argument(
        "--skip-team-counts",
        action="store_true",
        help="Skip team_counts.json (faster iteration during dev).",
    )
    args = parser.parse_args()

    models = args.model or list(MODEL_BUILDERS)
    out_dir = args.out.resolve()
    print(f"Output root: {out_dir}")

    for name in models:
        write_model(name, out_dir, skip_team_counts=args.skip_team_counts)

    print("\nDone. Inspect artifacts before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
