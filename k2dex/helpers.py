"""Helpers for VGC complexity analysis (Phase 1).

Smogon chaos JSON -> co-occurrence matrix -> PPMI.
SVD and downstream analyses live in notebooks until patterns stabilize.
See vgc_complexity_phase1_plan.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ChaosData:
    """Parsed Smogon chaos JSON for a single format-month."""

    pokemon: dict[str, dict]
    metagame: str
    cutoff: int
    n_battles: int


def load_chaos(path: str | Path) -> ChaosData:
    """Load a Smogon chaos JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    info = raw["info"]
    return ChaosData(
        pokemon=raw["data"],
        metagame=info["metagame"],
        cutoff=int(info["cutoff"]),
        n_battles=int(info["number of battles"]),
    )


def build_vocab(chaos: ChaosData, min_usage: float = 0.0) -> list[str]:
    """Pokemon names with usage >= min_usage, sorted by descending usage."""
    entries = [
        (name, entry["usage"])
        for name, entry in chaos.pokemon.items()
        if entry["usage"] >= min_usage
    ]
    entries.sort(key=lambda x: -x[1])
    return [name for name, _ in entries]


def build_cooccurrence(chaos: ChaosData, vocab: list[str]) -> NDArray[np.float64]:
    """Build a symmetric V x V co-occurrence matrix from Teammates entries.

    Teammate references not in `vocab` are dropped. Chaos JSON Teammates
    values are skill-weighted floats (not raw integer counts), and the source
    matrix is already exactly symmetric with zero diagonal -- we use the
    values as-is rather than re-symmetrizing.
    """
    idx = {name: i for i, name in enumerate(vocab)}
    n = len(vocab)
    C = np.zeros((n, n), dtype=np.float64)
    for name, entry in chaos.pokemon.items():
        i = idx.get(name)
        if i is None:
            continue
        for partner, weight in entry["Teammates"].items():
            j = idx.get(partner)
            if j is None:
                continue
            C[i, j] = weight
    return C


def build_ppmi(C: NDArray[np.float64], shift: float = 1.0) -> NDArray[np.float64]:
    """Compute shifted PPMI from a symmetric co-occurrence matrix.

    Works in probability space (sidesteps the absolute-N choice; PMI is
    invariant under any rescaling of the weights):

        p(i,j) = C[i,j] / sum(C)
        p(i)   = sum_j C[i,j] / sum(C)
        PMI    = log( p(i,j) / (p(i) p(j)) )
        PPMI   = max( PMI - log(shift), 0 )

    shift=1.0 is vanilla PPMI; plan calls for sweeping {1, 2, 5}.

    Smoothing (add-one, or context-distribution per Levy/Goldberg/Dagan 2015)
    is not applied here. Add when rare-pair noise becomes a problem in the
    long tail.
    """
    if (C < 0).any():
        raise ValueError("co-occurrence matrix has negative entries")
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    total = float(C.sum())
    if total <= 0:
        raise ValueError("co-occurrence matrix is all-zero")

    row = C.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (C * total) / np.outer(row, row)
        pmi = np.where(ratio > 0, np.log(ratio), -np.inf)
    ppmi = np.maximum(pmi - np.log(shift), 0.0)
    np.fill_diagonal(ppmi, 0.0)
    return ppmi


def binary_moments(
    chaos: ChaosData,
    vocab: list[str],
    cooccurrence: NDArray[np.float64],
    team_size: int = 6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute binary marginals m and joints p_joint for the Ising fit.

    Treats each team as a binary vector over Pokemon. Each team contributes
    team_size * (team_size - 1) ordered Teammate entries, so the total weight
    in `cooccurrence` divides into that many "pair events" per team.

        m[i]         = chaos.pokemon[name]["usage"]                 (marginal)
        p_joint[i,j] = cooccurrence[i,j] / weighted_teams           (joint)
        weighted_teams = sum(cooccurrence) / (team_size * (team_size - 1))

    Returns (m, p_joint). Sanity: sum_j p_joint[i,j] should be close to
    (team_size - 1) * m[i] -- one row of teammates per team-appearance of i.
    """
    m = np.array([chaos.pokemon[name]["usage"] for name in vocab])
    weighted_teams = cooccurrence.sum() / (team_size * (team_size - 1))
    p_joint = cooccurrence / weighted_teams
    return m, p_joint


def binary_correlation(
    m: NDArray[np.float64],
    p_joint: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Pearson correlation matrix for binary indicator variables.

    For binary X_i with marginal m[i] and joint p_joint[i,j]:
        Cov(X_i, X_j) = p_joint[i,j] - m[i] * m[j]
        Var(X_i)      = m[i] * (1 - m[i])
        Corr[i,j]     = Cov / sqrt(Var(X_i) * Var(X_j))

    Diagonal is set to 1 by definition.
    """
    denom = np.sqrt(np.outer(m * (1 - m), m * (1 - m)))
    corr = (p_joint - np.outer(m, m)) / denom
    np.fill_diagonal(corr, 1.0)
    return corr


def ising_gaussian(
    corr: NDArray[np.float64],
    eps: float = 0.01,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gaussian / precision-matrix inverse Ising approximation.

    At maximum entropy with given 1st and 2nd moments, treating spins as
    Gaussian gives the leading-order coupling matrix:

        Theta = inv(corr + eps * I)        (precision matrix)
        J     = -Theta                     (off-diagonal coupling)

    The diagonal of J is zeroed -- self-coupling is absorbed into the field h
    (which can be recovered via mean-field self-consistency from J and m).
    `eps` is a small ridge term that stabilizes the inversion when corr has
    near-zero eigenvalues; values in [1e-3, 1e-1] are typical.

    Returns (J, Theta).
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    n = corr.shape[0]
    theta = np.linalg.inv(corr + eps * np.eye(n))
    J = -theta.copy()
    np.fill_diagonal(J, 0.0)
    return J, theta
