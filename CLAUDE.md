# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code applying complexity-science / distributional-semantics methods to competitive Pokémon (VGC) team-composition data. **Read `vgc_complexity_phase1_plan.md` first** — it's the authoritative source on Phase 1 methodology, scope, and rationale.

Three modeling phases, all using the inverse Ising pairwise-MaxEnt model with `(J, h)`:

- **Phase 1** — Gaussian / precision-matrix approximation on Smogon chaos JSON (marginal usage + pairwise teammate co-occurrence). Fast to fit but systematically under-magnitudes strong negative couplings.
- **Phase 2** — pseudo-likelihood (PL) fit on Limitless tournament team rosters, species-only vocab.
- **Phase 3** — PL fit on `(species, item)` pair features from the same Limitless data. Restores held-item forme distinctions Phase 2 collapses.

The repo is a sequence of notebooks plus a Streamlit webapp. The webapp exposes only Phase 2 (Species) and Phase 3 (Species @ Item) — Phase 1 lives on in the notebooks but was cut from the user-facing surface in the v1 simplification. The webapp has **three pages** under one model picker: `/completer` (suggested completions or full PT sampling), `/analysis` (per-team diagnostics), `/meta` (format-wide statistics). Webapp user-facing terminology is decoupled from the underlying math: `J` → "Coupling Strength", `h` → "Bias", `ΣJ` → "Coherence", `E` → "Score" (sign-flipped so higher = better), `field_weight` → "Bias Adjustment". The webapp is one deliverable of the larger project — not the single end goal; deeper scientific content lives in the notebooks and a future explanatory page.

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
rendering.py            Diagnostic helpers + markdown-table builders (pure, no Streamlit)
rendering_html.py       Lab-notebook HTML helpers (section labels, stat strips,
                        score chips, signed/mini bars, sprite cells, slot cards,
                        rich-row tables); paired with styles.py classes
styles.py               Design tokens (LAB.* CSS vars), font import, widget
                        overrides (st.tabs, phase picker, segmented control,
                        primary button, multiselect chips)
assets/                 Static assets bundled into the page as data URIs
                        (currently just missingno.svg, the sprite fallback)
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
   - **`/completer`** — Bias Adjustment + Temperature sliders, pin/exclude multiselects, plus a single `Full statistical sampler (slow)` toggle. Toggle off (default): mean-field marginals at the chosen Bias Adjustment seed a greedy descent — fast, deterministic, returns one suggested team. Toggle on: parallel-tempered MCMC with the Temperature slider as the cold-chain target T and a fixed hot T = 2.0 — slow but yields a full Boltzmann distribution over the top-K completions. All other PT/MF/greedy knobs are locked to defaults (`PT_LADDER_LEVELS=7`, `PT_RUNS=3`, `PT_SWEEPS=10000`, `PT_BURN_IN=3000`, `PT_SWAP_INTERVAL=10`, `MF_MAX_ITERS=200`, `MF_TOL=1e-5`, `GREEDY_MAX_SWAPS=10`, `TOP_COMPLETIONS=10`). Error if cold > hot.
   - **`/analysis`** — observables strip (Score (adj), Score (raw), Coherence, corpus), pairwise coupling decomposition (15 pairs sorted by |Coupling|), top single swaps from the starting team (`TOP_SINGLE_SWAPS=10`, independent — not chained), greedy swap-chain critique (`GREEDY_MAX_SWAPS=10`).
   - **`/meta`** — fitted-model summary with locked headline validation numbers, side-by-side ±Bias feature tables (`META_TOP_FEATURES=25` each), side-by-side ±Coupling pair tables with same-species filter on Species @ Item (`META_TOP_PAIRS=25` each), Coupling/Bias distribution plots.
   - Persistent model picker at the top selects between Species / Species @ Item. Each page keeps its own multiselect state via `key=` suffixes; internal `phase_key` values are `"species"` / `"species_item"`.
   - **Sign convention:** the webapp displays `Score = -E` so higher = better team. The underlying `team_energy` in `sampling.py` is unchanged (Hamiltonian space, lower = more probable). All sign-flipping happens at the render layer only.
   - **The merged `corpus` column** replaces the prior `obs` + `Δswaps` pair: a green `N×` badge when the exact roster was seen `N` times in the corpus; an amber/red `Δk (N)` badge when the nearest observed roster is `k` swaps away and was itself seen `N` times. Helpers: `rendering.nearest_observed` returns `(delta, count)`; `rendering_html.corpus_cell` renders the chip.

## Module structure

- **`constants.py`** — vocab cutoffs, corpus size, regularization strength, validation seed / fraction, ingest filters. Change any knob here, not in the notebook or app.
- **`helpers.py`** — Phase 1 only: `load_chaos`, `build_vocab`, `build_cooccurrence`, `build_ppmi`, `binary_moments`, `binary_correlation`, `ising_gaussian`. Frozen-dataclass return types throughout.
- **`models.py`** — `fit_pl_ising(X, *, C, max_iter)` runs V per-spin L2 logistic regressions, skips degenerate spins, symmetrizes via `J = 0.5 * (J_asym + J_asym.T)`, zeros diagonal. Used by both notebooks 5/6 and `app.py:load_model_{species,species_item}`.
- **`sampling.py`** — `swap_mcmc` / `anneal_mcmc` / `parallel_tempered_mcmc` / `meanfield_marginals` / `greedy_optimize` / `rank_single_swaps`. All accept keyword-only `species_of` / `item_of` for Phase 3 uniqueness (no-duplicate-species, no-duplicate-item). Shared inner loop in `_local_swap_step` so the three MCMC variants stay in sync.
- **`rendering.py`** — pure-function helpers: `team_obs_count`, `min_swaps_to_observed`, `nearest_observed` (joint delta + nearest count for the merged corpus column), `intra_team_sum_j`, `pairwise_j_rows`, `render_pairwise_j_table`. No Streamlit imports. `team_obs_count` and `min_swaps_to_observed` stay for notebook use; the webapp uses `nearest_observed` exclusively.
- **`rendering_html.py`** — HTML-string builders for the lab-notebook visual language. Atoms: `section_label`, `stat`, `stat_strip`, `score_chip`, `signed_bar`, `mini_bar`, `corpus_cell` (merged obs / Δswaps badge — replaces the old `delta_swaps_badge`). Sprite primitives: `sprite_img` / `_sprite_box` (return `<object>` elements with native HTML5 fallback — see "Webapp visual language" gotchas below for *why*). Vocab parsing: `extract_species` / `extract_item` to split `"Species @ Item"` strings. Composed cells: `slot_card` / `slot_card_empty` / `slot_strip`, `comp_mon_cell`, `pair_cell`, `swap_cell`, `inline_mon`, `team_mini_strip`. Class names are paired with `styles.py` — renaming one means renaming both.
- **`styles.py`** — design tokens (LAB.* CSS variables, Google Fonts import for Source Serif 4 / Inter / IBM Plex Mono) and Streamlit widget overrides. `inject()` is called once at the top of `main()`. Includes overrides for Streamlit's dark-mode auto-detection (`--primary-color` etc. forced to the lab palette so widget internals stay in light mode even when the OS reports `prefers-color-scheme: dark`).
- **`limitless_ingest.py`** — fetches recent VGC Reg M-A tournaments from `play.limitlesstcg.com/api` (paginated, with name + Protect-ratio singles filters), extracts parsed team rosters and caches one small JSON per tournament under `tournaments_cache/`. Walks newest-first until `>= min_teams` (default `PHASE2_MIN_TEAMS = 10000`) teams accumulate. Cache schema is versioned (`CACHE_VERSION`); bump when parsing logic changes. `normalize_name` collapses case/whitespace variants in the raw API; `strip_mega_prefix` collapses `Mega <X>` species names (with optional X/Y/Z forme suffix) down to the base species — the held Mega Stone is the source of truth for which mega forme, and players were inconsistent about whether they prefixed the species name.

## Webapp visual language

Post-v0.4 the app uses a "lab notebook" palette + type system delivered via CSS injection. Four non-obvious things to know before touching it:

- **Streamlit strips inline event handlers and won't execute `<script>` tags injected via `st.markdown`.** This is why `sprite_img` returns an HTML5 `<object>` with an inner `<span>` fallback — when the `data=` URL fails, the browser natively renders the inner content. We tried `onerror`-swap (handler stripped), layered `background-image` (transparent sprites let the missingno bleed through), and inline `<script>` rebinding (React's `dangerouslySetInnerHTML` refuses to execute it). `<object>` was the only path that worked.
- **Sprite slug rules** (`rendering_html.species_to_slug`): first hyphen is treated as a forme separator and preserved (`Calyrex-Shadow` → `calyrex-shadow`, `Blastoise-Mega` → `blastoise-mega`); subsequent hyphens collapse (`Urshifu-Rapid-Strike` → `urshifu-rapidstrike`). Species whose canonical name *contains* a hyphen as part of the base (Ho-Oh, Chien-Pao, Porygon-Z, …) are in a curated `_HYPHEN_BASE_SPECIES` set and have all hyphens stripped. Missing sprites fall through to `assets/missingno.svg`, inlined as a data URI at import time.
- **Widget-scoping via marker divs + `:has()`.** Streamlit doesn't let you wrap a widget in a custom-class div. To style only specific widgets, the pattern is: emit an invisible marker element (`<div class="lab-phase-picker-marker"></div>`) right before/inside the widget's column, then target via `[data-testid="stColumn"]:has(.lab-phase-picker-marker) [role="radiogroup"] { … }`. If Streamlit ever changes its `data-testid` hierarchy, these selectors are the first thing that breaks — search styles.py for `:has(`.
- **OS dark mode bleeds into widget internals** unless `--primary-color` / `--background-color` / `--text-color` are forced on `:root` AND on `html[data-theme="dark"]`. `.streamlit/config.toml` would be the idiomatic fix but is gitignored.

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

- **In the webapp `/completer`, MF marginals seed the greedy descent — bias adjustment is honored at every step.** Prior versions seeded greedy descent from a popularity-only initial fill (`sorted(range(V), key=-m[i])`), so the Bias Adjustment slider was effectively ignored at init and only entered at descent time; descent usually terminated in a handful of swaps before escaping the popularity basin. The post-v1 fast path computes MF marginals at the chosen Bias Adjustment first, greedy-fills from that ranking (which already accounts for popularity ± coupling structure under the bias-adjusted field), then runs greedy descent. Net: the slider has visible end-to-end effect now. Chain length of the descent step is itself a diagnostic — short chain means MF approximation is tight for this query; longer chain means the one-shot MF-fill missed an internally-inconsistent pairing.

- **Phase 3 (Species @ Item) qualitatively mitigates the negative-ΣJ "balance team" artifact observed under Phase 2.** Under Phase 2 (species-only) the most-played teams had ΣJ < 0 — the model couldn't represent what made them work because the pairwise-MaxEnt fit on species labels alone is blind to role and item synergy. Adding the item dimension restores most of that information, so high-popularity rosters under Phase 3 are typically positive-ΣJ. Qualitative assessment only; worth a more rigorous comparison later — would need to project Phase 3's `(species, item)` rosters down to species, compute ΣJ on Phase 2's J for matched teams, and contrast.

## Data file

`gen9championsvgc2026regma-1760.json` (VGC Reg M-A Champions, 1760 elo cutoff) is the current target metagame. An earlier 1500-cutoff file was tried, found equivalent, and dropped. An earlier BSS variant existed briefly; do not reintroduce it.

## External reference

`~/Projects/k2dex-calculator/scripts/update_meta.py` — sibling project's Smogon chaos-JSON downloader/parser. Reuse if we ever need to re-fetch or move to a different format-month.

## v0 history

The pre-refactor state is tagged `v0`. Anything in the git history before that tag was structurally different in three ways relevant to anyone reading this doc:

1. **Sampler / MF / greedy code lived inline in `app.py`** (single ~1750-line file). It's now in `sampling.py`; `app.py` is purely rendering + Streamlit plumbing.
2. **Validation used a tournament-strided split.** The numbers in older commit messages (e.g. "Phase 2 top-1: 37.8%", "Phase 3 top-1: 46.3%") were measured under that split. The current post-Phase-3 baseline above uses the team-level split.
3. **Streamlit app was one page with five modes in the sidebar.** Now three pages with a top-of-page model picker; the "Greedy team optimizer" mode became the centerpiece of the `/analysis` page.

If you need an old number, check `tests/validation_baseline_v0.json` for the canonical v0 figures.

## v1 webapp simplification

The pre-v1 webapp exposed Phase 1/2/3 in a phase picker, five techniques in `/completer` (Mean-field / Parallel-tempered / Sample distribution / Anneal → MAP / Greedy descent), per-technique slider expanders, and shared the underlying mathematical terminology (`E_adj`, `E_raw`, `ΣJ`, `J`, `h`, `field_weight`) in the UI. The v1 simplification cut all of that:

1. **Phase 1 removed from the picker.** `helpers.py` and `load_model_phase1` in `app.py` (renamed `load_model_species` / `load_model_species_item`) survive for notebook use. The picker now has two options: `Species` and `Species @ Item`.
2. **Five techniques collapsed to a single `Full statistical sampler (slow)` toggle.** Off (default) runs MF-marginals → greedy-fill → greedy-descent. On runs PT. `Sample distribution` and `Anneal → MAP` were dropped from the UI entirely (still in `sampling.py`).
3. **Per-technique sliders locked to constants** (see the `/completer` description above for values).
4. **User-facing terms decoupled from the math.** `E` → Score (sign-flipped), `E_adj` → Score (adj), `field_weight` → Bias Adjustment, `h` → Bias, `J` → Coupling Strength, `ΣJ` → Coherence. Internal code, notebook outputs, and `sampling.py` / `models.py` API keep the original names. Sign-flipping happens only at the render layer.
5. **`obs` + `Δswaps` columns merged into one `corpus` column** rendered by `rendering_html.corpus_cell` (green `N×` if seen, amber/red `Δk (N)` if not). `rendering.nearest_observed` returns `(delta, count)`.
6. **Inline captions stripped from `/analysis` and `/meta`** — column names + the planned future explanatory page carry the explanatory load.
