# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code applying complexity-science / distributional-semantics methods to competitive Pokémon (VGC) team-composition data. **Read `vgc_complexity_phase1_plan.md` first** — it's the authoritative source on methodology, scope, and rationale. Phase 1 uses only Smogon chaos JSON (marginal usage + pairwise teammate co-occurrence); Phase 2 (team-level data via the Limitless API) is explicitly out of scope for the current repo.

The repo is a sequence of notebooks that build on each other, plus a Streamlit webapp wrapping the most useful artifact (the inverse-Ising team auto-completer).

## Run commands

```bash
streamlit run app.py        # team auto-completer webapp
jupyter lab                 # notebooks
```

No package manifest, build, lint, or test runner. Dependencies live in the user's global Python env; confirm before adding new ones (per `~/.claude/CLAUDE.md`). The work is exploratory — verify changes by re-running affected notebooks and inspecting outputs.

## Pipeline order

The notebooks form a dependency chain. Run in order, or re-derive intermediate state inside any standalone notebook.

1. **`spatial_embedding.ipynb`** — Stages 0–4 from the plan. PPMI → truncated SVD → role-residual NN ranking → NMF role decomposition. **Key finding (load-bearing for everything downstream): the SVD/NMF spectrum is smooth with no elbow.** Role structure is not low-rank, so dimensionality reduction is information-lossy. This motivated the pivot to inverse Ising.
2. **`inverse_ising.ipynb`** — fit `(J, h)` via the Gaussian / precision-matrix approximation; visualize J (sign-split network panels, ego networks, hierarchically-reordered heatmap); cross-reference top `-J` pairs to the role-residual ranking from notebook 1.
3. **`j_communities.ipynb`** — Louvain community detection on the `+J` subgraph (archetype clusters) and `-J` subgraph (role pools, with caveats about diffuse competition).
4. **`forward_ising.ipynb`** — forward sampling from the fitted model: swap-move MCMC with the `team_size=6` constraint, calibration against empirical marginals, team auto-completer (`complete_distribution` + `complete_anneal`).
5. **`app.py`** — Streamlit wrapper around the forward-sampling work. Three modes: single-chain sampling, annealing-to-MAP, parallel-tempered sampling.

## helpers.py

Stable primitives promoted from the notebooks. Type-first (frozen dataclass `ChaosData`, type hints, docstrings explaining the math). New functions get added here once they stabilize in a notebook — code lives in notebooks first.

- `load_chaos`, `build_vocab`, `build_cooccurrence`, `build_ppmi` — Stage 0–1
- `binary_moments`, `binary_correlation`, `ising_gaussian` — Stage 4 (inverse Ising)

The swap-move MCMC sampler and the team-completer (`complete_*` / `anneal_*` / parallel-tempered) currently live in `forward_ising.ipynb` and `app.py` as inline code, not in `helpers.py`. Promote once stable.

## Conventions threaded through multiple files

Easy to get wrong if you only look at one file:

- **Vocab cutoff is `min_usage=0.002`** (170 Pokémon). Consistent across all notebooks and the app so rankings are directly comparable.
- **PPMI is computed in probability space** (`p_ij / (p_i p_j)`) so absolute counts cancel. Smogon's `Teammates` values are **skill-weighted floats, not raw integer counts**; we treat them as proportional to team-appearances throughout.
- **Binary-moment normalization**: each team contributes `team_size × (team_size − 1)` ordered Teammate entries, so `weighted_teams = sum(C) / 30` for VGC. Sanity check: row-sums of `p_joint` should equal `(team_size − 1) × m[i]`.
- **Field h** is derived from mean-field self-consistency: `h = logit(m) − J @ m`. Two terms: marginal log-odds (popularity) + interaction correction.
- **Energy formulas**: raw `H(s) = -h·s - 0.5 s'Js`. Adjusted `H_adj(s) = -(field_weight·h)·s - 0.5 s'Js`. At `field_weight=0` only the pairwise term remains, dropping the popularity prior — this is the knob for surfacing archetype-coherent completions when standard sampling is too meta-biased.
- **Swap-move MH** for fixed team size: simultaneously turn one currently-on mon off and one currently-off mon on (single-spin flips break the size constraint). Energy diff `ΔH = h_eff[i_out] − h_eff[i_in] + (J[i_out] − J[i_in]) · s + J[i_in, i_out]`. The last term corrects for the swapped pair's mutual coupling — easy to forget.

## Important findings to know up front

- **Gaussian inverse Ising under-magnitudes strong negative couplings.** A known limitation: J for mutual-exclusion pairs (mega/base, weather setters in the same archetype) is much less negative than it would be in a true binary Ising fit. At low `field_weight` this can over-predict mutually-exclusive teams (e.g., dual megas). The empirical data shows dual-megas are common at the 1500 elo cutoff (~40% of teams), so the model isn't necessarily wrong — but at very low field_weight it produces them more often than the data warrants.
- **Single-chain MCMC at low T is basin discovery, not Boltzmann sampling.** Frequencies reflect which local minima random inits land in, not Boltzmann weights. So at low T in single-chain mode, frequency-vs-adj-E ordering decouples (non-monotonic). The **Parallel-tempered sample** mode in `app.py` is the fix: replica exchange between hot chains (broad exploration) and cold chains (concentrated sampling) gives true Boltzmann draws at the target T. Adj E *should* be monotonic with probability there, modulo Monte Carlo noise.

## Data file

`gen9championsvgc2026regma-1500.json` (VGC Reg M-A Champions, 1500 elo cutoff, 3.1M battles, 263 Pokémon) is the current target metagame. An earlier BSS variant existed briefly; do not reintroduce it.

## External reference

`~/Projects/k2dex-calculator/scripts/update_meta.py` — sibling project's Smogon chaos-JSON downloader/parser. Reuse if we ever need to re-fetch or move to a different format-month.
