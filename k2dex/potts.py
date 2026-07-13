"""Potts-model analysis of the fitted species+item model.

The species+item Ising model is a Potts model in lattice-gas (absent-reference)
encoding: each team **site** is a species, and a site's **state** is one of
``{absent, item_1, ..., item_k}``. The flat ``(species, item)`` spins encode
only the *present* states; the absent state is the all-zero reference, so the
cross-species block ``J[items_of_P, items_of_Q]`` is exactly the Potts coupling
between sites P and Q, measured relative to both being absent.

This module is **post-fit analysis only** -- it does not touch the fitting
objective or the sampler. It decomposes each species-pair coupling block into

    B(a, b) = synergy + row_effect(a) + col_effect(b) + interaction(a, b)

a 2-way ANOVA that is exactly the transform to the zero-sum (Ising/lattice-gas)
gauge of the block: ``synergy`` is the state-independent species-level coupling
(the average over item states), and everything else is the zero-sum
item-modulation residual -- how each specific item shifts the coupling relative
to the species average. From that it builds

  * the **item-modulation table** (how much each species' role is defined by
    which item it holds), reported alongside per-species support so a near-zero
    residual from *no effect* can be told from one from *no data*; and
  * an **average-product-corrected (APC) species interaction graph**, the
    standard plmDCA contact score (Frobenius norm of the zero-sum block, then
    APC to strip the popularity/degree background that ``h`` also carries).

See ``potts-reframe-plan.md`` R2 and the DCA references (Morcos 2011, Ekeberg
2013) in the research knowledge base.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


# ---------- Loading a fitted artifact ----------

@dataclass
class FittedModel:
    """A fitted species+item model loaded from a precompute artifact.

    ``J`` is the full symmetric (V, V) coupling matrix (zero diagonal),
    reconstructed from the packed lower triangle. ``sites`` is the sorted list
    of distinct species (the Potts sites); ``site_index[P]`` gives the flat
    feature indices of species P's item-states, in vocab order.
    """
    vocab: list[str]
    species_of: list[str]
    item_of: list[str | None]
    J: NDArray[np.float64]
    h: NDArray[np.float64]
    m: NDArray[np.float64]
    team_size: int
    n_corpus_teams: int

    @property
    def V(self) -> int:
        return len(self.vocab)

    @cached_property
    def sites(self) -> list[str]:
        return sorted(self.site_index)

    @cached_property
    def site_index(self) -> dict[str, list[int]]:
        idx: dict[str, list[int]] = defaultdict(list)
        for i, sp in enumerate(self.species_of):
            idx[sp].append(i)
        return dict(idx)

    def item_alphabet(self, species: str) -> list[str | None]:
        """The item states observed for `species`, in flat-index order."""
        return [self.item_of[i] for i in self.site_index[species]]

    def appearances(self, species: str) -> float:
        """Approximate weighted corpus appearances of `species` (all its
        item-states), from the marginal `m` scaled by the corpus size. This is
        the support figure the modulation table must be read against."""
        return float(self.m[self.site_index[species]].sum() * self.n_corpus_teams)

    def item_weights(self, species: str) -> NDArray[np.float64]:
        """Empirical item-usage distribution for `species`, conditional on the
        species being present: the weighted corpus marginals ``m`` of its
        item-states, renormalized over the species' item alphabet to sum to 1
        (flat-index order, matching :meth:`item_alphabet`).

        ``m`` is the sample-weighted empirical appearance rate per feature (from
        the data, not the fitted ``(J, h)``), so these weights answer "when this
        species is on a team, how often does it hold each item?" Falls back to
        uniform if the species has no marginal mass."""
        idx = self.site_index[species]
        w = self.m[idx].astype(np.float64)
        total = w.sum()
        if total <= 1e-12:
            return np.full(len(idx), 1.0 / len(idx))
        return w / total


def load_fitted_model(model_dir: str | Path) -> FittedModel:
    """Load a precompute artifact (``meta.json`` + ``J.bin``/``h.bin``/``m.bin``)
    into a :class:`FittedModel`. ``J.bin`` is the float32 strict lower triangle
    in row-major order (see ``precompute.pack_lower_triangle``); this
    reconstructs the full symmetric matrix."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    V = int(meta["V"])
    tri = np.fromfile(model_dir / "J.bin", dtype=np.float32).astype(np.float64)
    J = np.zeros((V, V), dtype=np.float64)
    rows, cols = np.tril_indices(V, k=-1)
    J[rows, cols] = tri
    J[cols, rows] = tri
    h = np.fromfile(model_dir / "h.bin", dtype=np.float32).astype(np.float64)
    m = np.fromfile(model_dir / "m.bin", dtype=np.float32).astype(np.float64)
    # Reconstruct the per-feature (species, item) view this module's analysis
    # API is built on from the factored v3 schema: species_of[i] = sites[site_of[i]],
    # item_of[i] = the first track's value (None when the model has no tracks).
    sites = list(meta["sites"])
    site_of = list(meta["site_of"])
    track_values = list(meta["track_values"])
    species_of = [sites[s] for s in site_of]
    item_of: list[str | None] = [
        (vals[0] if vals else None) for vals in track_values
    ]
    return FittedModel(
        vocab=list(meta["vocab"]),
        species_of=species_of,
        item_of=item_of,
        J=J, h=h, m=m,
        team_size=int(meta["team_size"]),
        n_corpus_teams=int(meta["n_corpus_teams"]),
    )


# ---------- Block decomposition (zero-sum gauge / 2-way ANOVA) ----------

def species_block(model: FittedModel, P: str, Q: str) -> NDArray[np.float64]:
    """The (k_P, k_Q) Potts coupling block between species P and Q: rows are
    P's item-states, columns are Q's, in flat-index order. Relative to the
    absent reference (both absent -> zero coupling)."""
    return model.J[np.ix_(model.site_index[P], model.site_index[Q])]


@dataclass
class BlockDecomposition:
    """2-way ANOVA / zero-sum-gauge split of one species-pair block.

    ``synergy`` is the grand mean (state-independent species-level coupling).
    ``row_effects[a]`` is how P's item a shifts the coupling on average over Q's
    items; ``col_effects[b]`` is the same for Q's item b. ``interaction`` is the
    remaining item-item-specific term, zero-sum along both axes. The four
    reconstruct the block exactly:
    ``B = synergy + row_effects[:,None] + col_effects[None,:] + interaction``.
    """
    synergy: float
    row_effects: NDArray[np.float64]
    col_effects: NDArray[np.float64]
    interaction: NDArray[np.float64]


def decompose_block(B: NDArray[np.float64]) -> BlockDecomposition:
    """Split a coupling block into species-level synergy + zero-sum modulation.

    Equivalent to transforming the block to the zero-sum gauge: after the split,
    ``row_effects`` and ``col_effects`` each sum to zero and ``interaction`` sums
    to zero along both axes. See :class:`BlockDecomposition`.
    """
    B = np.asarray(B, dtype=np.float64)
    synergy = float(B.mean())
    row_mean = B.mean(axis=1)
    col_mean = B.mean(axis=0)
    row_effects = row_mean - synergy
    col_effects = col_mean - synergy
    interaction = B - row_mean[:, None] - col_mean[None, :] + synergy
    return BlockDecomposition(synergy, row_effects, col_effects, interaction)


def row_modulation(B: NDArray[np.float64]) -> NDArray[np.float64]:
    """The part of block ``B`` explained by *which item the row species holds*:
    the deviation of each row from the row-averaged (row-species-item-agnostic)
    block. Equals ``row_effects[:,None] + interaction``; its column-wise mean
    over rows is zero. This is the quantity the item-modulation score aggregates.
    """
    B = np.asarray(B, dtype=np.float64)
    return B - B.mean(axis=0, keepdims=True)


# ---------- Item-modulation table (the headline artifact) ----------

@dataclass
class ModulationRow:
    """One row of the item-modulation table for a species.

    All three magnitudes measure the same thing (how much the coupling depends
    on which item the species holds) but normalize differently, and they can
    disagree -- report them together with support:

      * ``mod_frob`` -- raw Frobenius norm of the modulation part, aggregated
        over partner species. The literal "modulation residual" magnitude, but
        strongly confounded by popularity and alphabet size (a busy species with
        many items scores high regardless). Read only against support.
      * ``mod_rms`` -- per-coupling RMS shift (``mod_frob`` divided by the number
        of contributing coupling entries). Removes the alphabet-size confound;
        still tracks overall coupling strength (popularity).
      * ``mod_frac`` -- share of the species' total pairwise coupling energy
        explained by its item choice (``||modulation||^2 / ||block||^2``).
        Scale-free; can read low for a species whose item flips its role yet
        whose item-agnostic baseline coupling is itself large.

    ``synergy_frob`` is the complementary state-independent (species-level)
    coupling magnitude. ``n_items`` and ``appearances`` are the support columns.
    """
    species: str
    n_items: int
    appearances: float
    mod_frob: float
    mod_rms: float
    mod_frac: float
    synergy_frob: float


def modulation_scores(model: FittedModel) -> list[ModulationRow]:
    """Item-modulation table over all species, most item-defined first (by the
    scale-free ``mod_frac``). Aggregates each species' cross-species blocks
    (self-block excluded: a species never shares a team with itself). A species
    with a single observed item has zero modulation by construction (no item
    variation) -- correct behaviour, flagged by ``n_items == 1``.
    """
    sites = model.sites
    idx = model.site_index
    rows: list[ModulationRow] = []
    for P in sites:
        iP = idx[P]
        mod_sq = 0.0      # sum of squared modulation entries
        full_sq = 0.0     # sum of squared block entries (denominator for frac)
        syn_sq = 0.0      # sum of squared synergies (species-level magnitude)
        n_entries = 0
        for Q in sites:
            if Q == P:
                continue
            B = species_block(model, P, Q)
            mod = row_modulation(B)
            mod_sq += float((mod ** 2).sum())
            full_sq += float((B ** 2).sum())
            syn_sq += float(B.mean()) ** 2
            n_entries += B.size
        rows.append(ModulationRow(
            species=P,
            n_items=len(iP),
            appearances=model.appearances(P),
            mod_frob=mod_sq ** 0.5,
            mod_rms=(mod_sq / n_entries) ** 0.5 if n_entries else 0.0,
            mod_frac=(mod_sq / full_sq) if full_sq > 1e-12 else 0.0,
            synergy_frob=syn_sq ** 0.5,
        ))
    rows.sort(key=lambda r: -r.mod_frac)
    return rows


@dataclass
class ItemModulation:
    """Per-item modulation detail for a species (drill-down under a table row).

    ``magnitude`` is the Frobenius norm of item ``item``'s row across all partner
    blocks (its contribution to the species' ``mod_frob``). ``pulls_toward`` /
    ``pulls_away`` list the partner *species* whose coupling this item shifts most
    positively / negatively relative to the species' item-agnostic average, each
    as ``(species, delta)``.
    """
    item: str | None
    magnitude: float
    pulls_toward: list[tuple[str, float]]
    pulls_away: list[tuple[str, float]]


def item_modulation(
    model: FittedModel, species: str, *, top_partners: int = 3,
) -> list[ItemModulation]:
    """Break a species' item-modulation down by item, strongest item first.

    For each item state, its role-shift magnitude and the partner species it
    most pulls the team toward / away from, relative to the species' item-
    agnostic coupling profile. This is the interpretability read: an item that
    genuinely defines a role shows sensible, distinct partner shifts.
    """
    sites = model.sites
    idx = model.site_index
    iP = idx[species]
    items = model.item_of
    # deviation of each of P's item-rows from the item-agnostic profile, laid
    # out as (k_P, V); only cross-species partner columns are populated. The
    # self-block columns stay zero and must be excluded from the ranking below,
    # or a single-signed row would surface them as its opposite-sign extremes.
    dev = np.zeros((len(iP), model.V), dtype=np.float64)
    partner_cols: list[int] = []
    for Q in sites:
        if Q == species:
            continue
        jQ = idx[Q]
        dev[:, jQ] = row_modulation(species_block(model, species, Q))
        partner_cols.extend(jQ)
    cols = np.array(sorted(partner_cols), dtype=int)
    out: list[ItemModulation] = []
    for a, feat in enumerate(iP):
        drow = dev[a]
        mag = float(np.linalg.norm(drow))
        order = cols[np.argsort(-drow[cols])] if cols.size else cols
        toward = [(model.species_of[k], float(drow[k]))
                  for k in order[:top_partners]]
        away = [(model.species_of[k], float(drow[k]))
                for k in order[::-1][:top_partners]]
        out.append(ItemModulation(items[feat], mag, toward, away))
    out.sort(key=lambda r: -r.magnitude)
    return out


# ---------- APC-corrected species interaction graph ----------

@dataclass
class SpeciesGraph:
    """Reduced species x species coupling graph.

    ``species`` indexes the (S, S) matrices. ``frob`` is the Frobenius norm of
    each zero-sum species-pair block (the plmDCA contact score, all-positive).
    ``apc`` is the average-product correction ``F_i. * F_.j / F_..`` and
    ``corrected = frob - apc`` strips the popularity/degree background. ``synergy``
    is the *signed* species-level coupling (usage-weighted mean per block) -- use
    it when the sign of the interaction (synergy vs anti-synergy) matters.
    """
    species: list[str]
    frob: NDArray[np.float64]
    apc: NDArray[np.float64]
    corrected: NDArray[np.float64]
    synergy: NDArray[np.float64]


def species_apc_graph(model: FittedModel) -> SpeciesGraph:
    """Collapse the flat coupling matrix to a species x species graph and apply
    the average-product correction.

    Each species-pair block is reduced to a scalar by the Frobenius norm of its
    zero-sum-gauged form (the item-modulation residual plus effects; the grand
    mean is the separate signed ``synergy``). APC then removes the background in
    which busy/popular species couple to everything -- the same confound the
    field ``h`` carries -- giving a cleaner graph than thresholding raw
    magnitudes. The diagonal is zero (no self-interaction).

    The signed ``synergy`` is the **usage-weighted** mean of the block: each
    item-state is weighted by its empirical appearance rate (:meth:`item_weights`)
    rather than treated as equally likely. A flat grand mean dilutes a species
    with many rarely-run items (its block is mostly near-zero entries) while
    concentrating one with a single dominant item, so it partly ranks by
    ``1/alphabet_size``. Weighting by how players actually pick items makes
    ``synergy`` the expected pairwise coupling the pair contributes to a team --
    a species with one item is unchanged, a many-item species is no longer
    penalized for its long tail. ``frob`` keeps the unweighted zero-sum gauge:
    it is the standard plmDCA contact norm (spread around the block center), not
    a signed strength, and is not what the meta ranking sorts on.
    """
    sites = model.sites
    S = len(sites)
    weights = {P: model.item_weights(P) for P in sites}
    frob = np.zeros((S, S), dtype=np.float64)
    synergy = np.zeros((S, S), dtype=np.float64)
    for a, P in enumerate(sites):
        for b, Q in enumerate(sites):
            if b <= a:
                continue
            B = species_block(model, P, Q)
            dec = decompose_block(B)
            # zero-sum block = effects + interaction = B - grand mean.
            zs = B - dec.synergy
            f = float(np.linalg.norm(zs))
            frob[a, b] = frob[b, a] = f
            syn = float(weights[P] @ B @ weights[Q])
            synergy[a, b] = synergy[b, a] = syn
    apc = _apc(frob)
    corrected = frob - apc
    np.fill_diagonal(corrected, 0.0)
    return SpeciesGraph(sites, frob, apc, corrected, synergy)


def _apc(F: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average-product correction ``APC_ij = F_i. * F_.j / F_..`` for a symmetric
    score matrix with zero diagonal. Row/column means and the grand mean are
    taken over the off-diagonal entries (each row excludes its own diagonal)."""
    S = F.shape[0]
    if S < 2:
        return np.zeros_like(F)
    off_sum = F.sum(axis=1)  # diagonal is zero, so this is the off-diagonal sum
    row_mean = off_sum / (S - 1)
    grand_mean = F.sum() / (S * (S - 1))
    if grand_mean <= 1e-12:
        return np.zeros_like(F)
    return np.outer(row_mean, row_mean) / grand_mean
