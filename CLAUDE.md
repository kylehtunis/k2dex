# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code applying complexity-science / distributional-semantics methods to competitive Pokémon (VGC) team-composition data. **Read `vgc_complexity_phase1_plan.md` first** — it's the authoritative source on Phase 1 methodology, scope, and rationale.

Three modeling phases, all using the inverse Ising pairwise-MaxEnt model with `(J, h)`:

- **Phase 1** — Gaussian / precision-matrix approximation on Smogon chaos JSON (marginal usage + pairwise teammate co-occurrence). Fast to fit but systematically under-magnitudes strong negative couplings.
- **Phase 2** — pseudo-likelihood (PL) fit on Limitless tournament team rosters, species-only vocab.
- **Phase 3** — PL fit on `(species, item)` pair features from the same Limitless data. Restores held-item forme distinctions Phase 2 collapses.

The repo is a sequence of notebooks plus a Streamlit webapp. The webapp has **three pages** under one phase picker: `/completer` (sample completions, five techniques), `/analysis` (per-team diagnostics under the fitted `(J, h)`), `/meta` (format-wide statistics).

## Run commands

```bash
streamlit run app.py            # three-page webapp
jupyter lab                     # notebooks
python -m unittest discover tests   # regression gate
python -m limitless_ingest       # ingest tournaments (manual; cache is in tournaments_cache/)
```

No build system; dependencies live in the user's global Python env. **`pandas` and `pyarrow` are intentionally absent** — use markdown tables for tabular display in Streamlit, not `st.dataframe`. Confirm before adding any new dependency (per `~/.claude/CLAUDE.md`).

## Repo layout

```
app.py                  Streamlit webapp (rendering only; samplers imported)
constants.py            Shared numeric constants (vocab cutoffs, seeds, etc.)
helpers.py              Phase 1 Gaussian inverse Ising primitives
models.py               fit_pl_ising — Phase 2/3 PL fit
sampling.py             MCMC family (swap / anneal / PT) + mean-field + greedy
                        + rank_single_swaps (top-N independent swap suggestions)
rendering.py            Diagnostic helpers + markdown-table builders
limitless_ingest.py     Limitless API ingest with normalize_name + strip_mega_prefix
tests/                  unittest tests + locked validation baselines
```

The "promote when stable" convention still applies: new functions live in their notebook until they stabilize, then move to the relevant module. Phase 1 primitives are in `helpers.py`; the PL fit is in `models.py`; the MCMC family is in `sampling.py`. Each module's docstring lists what's already promoted.

## Pipeline order

The notebooks form a dependency chain. Run in order, or re-derive intermediate state inside any standalone notebook.

1. **`spatial_embedding.ipynb`** — Stages 0–4 from the plan. PPMI → truncated SVD → role-residual NN ranking → NMF role decomposition. **Key finding (load-bearing for everything downstream): the SVD/NMF spectrum is smooth with no elbow.** Role structure is not low-rank, so dimensionality reduction is information-lossy. This motivated the pivot to inverse Ising. Uses `helpers.py` primitives.
2. **`inverse_ising.ipynb`** — fit `(J, h)` via the Gaussian / precision-matrix approximation; visualize J (sign-split network panels, ego networks, hierarchically-reordered heatmap). Uses `helpers.ising_gaussian`.
3. **`j_communities.ipynb`** — Louvain community detection on the `+J` subgraph (archetype clusters) and `-J` subgraph (role pools).
4. **`forward_ising.ipynb`** — forward sampling from the fitted model: swap-move MCMC with the `team_size=6` constraint, calibration against empirical marginals, team auto-completer. Uses `sampling.swap_mcmc`.
5. **`inverse_ising_phase2.ipynb`** — Phase 2 PL fit; mirrors notebook 2's analysis cells. Uses `models.fit_pl_ising`.
6. **`validation.ipynb`** — leave-k-out evaluation harness. Three sections: cross-model species-granularity comparison (team-level random 90/10 split), pair-prediction metric (Phase 3 only), temporal-drift study (chronological appendix). Uses `models.fit_pl_ising`.
7. **`app.py`** — three-page Streamlit webapp:
   - **`/completer`** — five techniques (Mean-field / Parallel-tempered / Sample distribution / Anneal → MAP / Greedy descent) sharing one constraints + field-weight UI.
   - **`/analysis`** — observables strip (E_adj, E_raw, ΣJ, corpus obs, Δswaps), pairwise J decomposition (15 pairs sorted by |J|), top single swaps from the starting team (one-shot menu, not chained), greedy swap-chain critique.
   - **`/meta`** — fitted-model summary with locked headline validation numbers, side-by-side ±h feature tables, side-by-side ±J pair tables (same-species pairs filtered out on Phase 3), J/h distribution plots.
   - Persistent phase picker at the top selects between Phase 1 / 2 / 3. Each page keeps its own multiselect state via `key=` suffixes.

## Module structure

- **`constants.py`** — vocab cutoffs, corpus size, regularization strength, validation seed / fraction, ingest filters. Change any knob here, not in the notebook or app.
- **`helpers.py`** — Phase 1 only: `load_chaos`, `build_vocab`, `build_cooccurrence`, `build_ppmi`, `binary_moments`, `binary_correlation`, `ising_gaussian`. Frozen-dataclass return types throughout.
- **`models.py`** — `fit_pl_ising(X, *, C, max_iter)` runs V per-spin L2 logistic regressions, skips degenerate spins, symmetrizes via `J = 0.5 * (J_asym + J_asym.T)`, zeros diagonal. Used by both notebooks 5/6 and `app.py:load_model_phase{2,3}`.
- **`sampling.py`** — `swap_mcmc` / `anneal_mcmc` / `parallel_tempered_mcmc` / `meanfield_marginals` / `greedy_optimize` / `rank_single_swaps`. All accept keyword-only `species_of` / `item_of` for Phase 3 uniqueness (no-duplicate-species, no-duplicate-item). Shared inner loop in `_local_swap_step` so the three MCMC variants stay in sync.
- **`rendering.py`** — pure-function helpers: `team_obs_count`, `min_swaps_to_observed`, `intra_team_sum_j`, `pairwise_j_rows`, `render_pairwise_j_table`. No Streamlit imports.
- **`limitless_ingest.py`** — fetches recent VGC Reg M-A tournaments from `play.limitlesstcg.com/api` (paginated, with name + Protect-ratio singles filters), extracts parsed team rosters and caches one small JSON per tournament under `tournaments_cache/`. Walks newest-first until `>= min_teams` (default `PHASE2_MIN_TEAMS = 10000`) teams accumulate. Cache schema is versioned (`CACHE_VERSION`); bump when parsing logic changes. `normalize_name` collapses case/whitespace variants in the raw API; `strip_mega_prefix` collapses `Mega <X>` species names (with optional X/Y/Z forme suffix) down to the base species — the held Mega Stone is the source of truth for which mega forme, and players were inconsistent about whether they prefixed the species name.

## Conventions threaded through multiple files

Easy to get wrong if you only look at one file:

- **Phase 1 vocab cutoff is `min_usage=0.002`** (~170 Pokémon at Reg M-A 1760). Phase 2/3 vocab cutoff is `PHASE2_MIN_TEAM_COUNT = 5` (feature must appear in ≥5 teams).
- **`MIN_TEAMS_PER_TOURNAMENT = 64`** filters out small/quirky tournaments at ingest time. Below this they tended to be majority-Bo1 or unusual local metas.
- **`PHASE2_MIN_TEAMS = 10000`** is the current corpus-size cutoff for ingest. The ingestor walks newest-first until this many teams accumulate.
- **PPMI is computed in probability space** (`p_ij / (p_i p_j)`) so absolute counts cancel. Smogon's `Teammates` values are **skill-weighted floats, not raw integer counts**; we treat them as proportional to team-appearances throughout.
- **Binary-moment normalization**: each team contributes `team_size × (team_size − 1)` ordered Teammate entries, so `weighted_teams = sum(C) / 30` for VGC. Sanity check: row-sums of `p_joint` should equal `(team_size − 1) × m[i]`.
- **Field h** for Phase 1 is derived from mean-field self-consistency: `h = logit(m) − J @ m`. Phase 2/3 PL fit gives `h` directly as the per-spin logreg intercept.
- **Energy formulas**: raw `H(s) = -h·s - 0.5 s'Js`. Adjusted `H_adj(s) = -(field_weight·h)·s - 0.5 s'Js`. At `field_weight=0` only the pairwise term remains, dropping the popularity prior — this is the knob for surfacing archetype-coherent completions when standard sampling is too meta-biased.
- **Swap-move MH** for fixed team size: simultaneously turn one currently-on mon off and one currently-off mon on (single-spin flips break the size constraint). Energy diff `ΔH = h_eff[i_out] − h_eff[i_in] + (J[i_out] − J[i_in]) · s + J[i_in, i_out]`. The last term corrects for the swapped pair's mutual coupling — easy to forget.
- **Validation split is team-level random 90/10** (`VALIDATION_TEAM_FRAC_TEST = 0.10`, `VALIDATION_SEED = 42`). Tournament-strided was the v0 approach — it concentrated within-tournament sibling correlation on one side of the split and made the held-out size depend on a few large tournaments' position in the stride. The team-level shuffle decorrelates and gives an exact 90/10 every time. The *temporal-drift* section of `validation.ipynb` deliberately stays chronological — different question (does the model decay as the meta moves forward).
- **Vocab normalization** at ingest time collapses case/whitespace variants (`'Sitrus Berry'` / `'sitrus berry'` → one bucket) AND collapses `Mega <X>` species names down to base species. The cache is invalidated when either normalizer changes (bump `CACHE_VERSION`).

## Important findings to know up front

- **Gaussian inverse Ising under-magnitudes strong negative couplings.** A known limitation: J for mutual-exclusion pairs (mega/base, weather setters in the same archetype) is much less negative than it would be in a true binary Ising fit. At low `field_weight` this can over-predict mutually-exclusive teams (e.g., dual megas). The empirical data shows dual-megas are common at the 1760 elo cutoff (~40% of teams), so the model isn't necessarily wrong — but at very low field_weight it produces them more often than the data warrants. Phase 2/3 PL doesn't have this issue (PL's J distribution is wider on both sides).
- **Single-chain MCMC at low T is basin discovery, not Boltzmann sampling.** Frequencies reflect which local minima random inits land in, not Boltzmann weights. So at low T in single-chain mode, frequency-vs-adj-E ordering decouples (non-monotonic). The **Parallel-tempered** technique in `/completer` is the fix: replica exchange between hot chains (broad exploration) and cold chains (concentrated sampling) gives true Boltzmann draws at the target T.
- **Phase 1 and Phase 2/3 vocabs are not directly comparable.** Smogon chaos data splits Pokémon by held-item-induced forme (`Charizard-Mega-Y`, `Charizard-Mega-X`, base `Charizard` are three entries). Limitless standings give only the species name, and `strip_mega_prefix` further collapses any inconsistent `Mega <X>` labelings. Within each phase the vocab is self-consistent — but cross-phase comparisons of J need an explicit collapse step on the Phase 1 side before indices can be aligned. Phase 3 restores the forme information via the item dimension (e.g. `(Charizard, Charizardite Y)` is a distinct feature).
- **The most-played teams in the Phase 2 corpus have negative intra-team ΣJ.** Flexibility/balance teams (Aerodactyl/Basculegion/Charizard/Garchomp/Kingambit cores with various flex slots) have `sum_{i<j} J_ij < 0` under the PL fit, despite being the global energy minima at `fw=1` (most negative `raw_E` in the entire model). They're held together by individual mon quality, matchup coverage, and in-game flexibility — properties the Ising model fundamentally cannot represent. This is a structural ceiling on what any pairwise MaxEnt fit can do, not a regularization issue. **Heuristic**: teams with negative ΣJ are "balance team" signatures (model can't see what makes them work); teams with positive ΣJ are "archetype" signatures (model can see the synergy).
- **Model rankings are recency-biased because ingest walks newest-first.** Lesson learned the hard way: an earlier observation that the model "predicted the Sylveon-variant rise" turned out to be an artifact of `PHASE2_MIN_TEAMS` being stale at 1500 — the model was being fit on a recent slice where Sylveon over-represents. With the full corpus the older Floette variant correctly outranks Sylveon. The model is a summary statistic of whatever subset it's fit on, not a predictor of meta evolution. A real test of "model predicts meta shift" would fit on an older time window and check whether held-out recent tournaments are enriched for the model's top-K.
- **`field_weight` sweep regime map (Phase 2 app)** — useful operational mental model when interpreting completions:

  | `fw` | regime | top-K characteristic |
  | ---: | :--- | :--- |
  | 1.0 | popularity-driven | 1-swap variant of meta mode (empirical mode + h gradient) |
  | 0.7 | balanced | empirical mode wins; variants nearby |
  | 0.5 | archetype-coherent | real high-ΣJ archetypes (sand+balance, Perish Trap) at Δ=0; empirical mode drops mid-rank because it's negative-ΣJ |
  | 0.3 | exploration | archetypes stable; Δ=1–2 structured variants emerge below |
  | 0.0 | pure-J / overfit | repeated-specific-team basins dominate |

  **Useful operating range with `PHASE2_LR_C = 0.1`: `fw ≈ 0.3–0.6`.** At sklearn's default `C = 1.0` the floor was tighter (~0.3–0.5); dropping to `C = 0.1` shrinks rare-co-occurrence couplings enough to extend the useful range without compromising strong-archetype detection at moderate fw. **More data alone does not fix repeated-roster basins** — the persistent attractors trace to specific roster repetitions in the corpus where the PL fit reinforces all 15 pairwise couplings every time. **Regularization is the lever, not corpus size**, and the `C = 0.1` default reflects this finding.

- **Quantitative validation: Phase 3 (item-pair) > Phase 2 (species) > co-occurrence > marginal > PPMI.** From the locked post-Phase-4 baseline (`tests/validation_baseline_post_phase4.json`, team-level 90/10 split):

  | model | k=1 top-1 | k=1 MRR | k=2 top-1 |
  | :--- | ---: | ---: | ---: |
  | Ising item-pair | 44.7% | 0.565 | 26.2% |
  | Ising species | 35.0% | 0.494 | 22.6% |
  | co-occurrence | 17.3% | 0.328 | 16.0% |
  | marginal | 11.6% | 0.258 | 12.0% |
  | PPMI | 11.7% | 0.196 | 8.6% |

  Phase 3's edge over Phase 2 (+9.7 pp at k=1) is an asymmetry the species model can't exploit: held-in mons' actual items are observed at eval time, while held-out items are marginalized. The item-pair model uses the specific `(held-in-species, held-in-item)` pair's J-row to score candidates rather than averaging over all item variants. This is "same data, better-disambiguated features," not "more data." Phase 3's vocab is ~3× Phase 2's (~574 pair features vs ~205 species), so each feature has fewer positive examples and the fit is a strictly harder problem — the win comes *despite* that handicap.

- **PPMI is the wrong baseline for completion ranking.** It significantly underperforms even the trivial marginal baseline. PPMI measures *surprise* of co-occurrence — which is the right quantity for distributional similarity (and is correctly load-bearing in the Phase 1 spatial-embedding work) but the wrong quantity for "which mon completes this team": it's biased toward *rare–rare* pairs because the denominator `p(i)·p(j)` vanishes when both mons are unpopular, inflating the ratio. So PPMI ranks candidates with strong-but-rare association above candidates with high conditional probability. **The right baseline is raw co-occurrence sum** (`sum_{j in held-in} C_ij`, an unnormalized proxy for `p(i, held-in)`).

- **Mean-field is a faithful fast proxy for the true sampler.** Direct head-to-head on 100 in-vocab test teams (matched held-out positions across methods, post-Phase-4 baseline): MF and single-chain swap MCMC (at fw=1, T=1, 20000 post-burn-in samples — the target distribution PT samples from at default settings) agree on top-1 picks in 85.0% / 75.0% / 73.7% of teams at k = 1 / 2 / 3, with Spearman rank correlation of held-out-pair ranks at 0.94–0.95 across all k. Hit-rate / MRR deltas are ≤1.5 pp at every (k, K). Practical takeaways: (a) the validator's MF-based headline numbers are trustworthy estimates of true Ising performance; (b) MF is the default `/completer` technique for the same reason (instant, no MCMC tuning, same ranking quality up to head-permutation noise); (c) at very low T or very low `field_weight` the MF approximation is *not* validated — those regimes weren't tested. For low-T pure-J archetype-coherence work, stick with Parallel-tempered.

## Data file

`gen9championsvgc2026regma-1760.json` (VGC Reg M-A Champions, 1760 elo cutoff) is the current target metagame. An earlier 1500-cutoff file was tried, found equivalent, and dropped. An earlier BSS variant existed briefly; do not reintroduce it.

## External reference

`~/Projects/k2dex-calculator/scripts/update_meta.py` — sibling project's Smogon chaos-JSON downloader/parser. Reuse if we ever need to re-fetch or move to a different format-month.

## v0 history

The pre-refactor state is tagged `v0`. Anything in the git history before that tag was structurally different in three ways relevant to anyone reading this doc:

1. **Sampler / MF / greedy code lived inline in `app.py`** (single ~1750-line file). It's now in `sampling.py`; `app.py` is purely rendering + Streamlit plumbing.
2. **Validation used a tournament-strided split.** The numbers in older commit messages (e.g. "Phase 2 top-1: 37.8%", "Phase 3 top-1: 46.3%") were measured under that split. The current post-Phase-3 baseline above uses the team-level split.
3. **Streamlit app was one page with five modes in the sidebar.** Now three pages with a top-of-page phase picker; the "Greedy team optimizer" mode became the centerpiece of the `/analysis` page.

If you need an old number, check `tests/validation_baseline_v0.json` for the canonical v0 figures.
