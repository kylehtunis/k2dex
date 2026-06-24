# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code applying complexity-science / distributional-semantics methods to competitive Pokémon (VGC) team-composition data.

Three inverse Ising pairwise-MaxEnt models, each fitting `(J, h)` to a different data source and vocabulary:

- **Gaussian model** — precision-matrix approximation on Smogon chaos JSON (marginal usage + pairwise teammate co-occurrence). Fast to fit but systematically under-magnitudes strong negative couplings. Lives in the notebooks; not on the user-facing webapp.
- **Species model** — pseudo-likelihood (PL) fit on Limitless tournament team rosters, species-only vocab.
- **Species+Item model** — PL fit on `(species, item)` pair features from the same Limitless data. Restores held-item forme distinctions the species model collapses.

The repo is a sequence of notebooks plus **two coexisting webapp surfaces**: the original Streamlit app (`scripts/app.py`) and a static React/TypeScript port (`web/`) deployed to GitHub Pages. Both surface all precomputed models (across multiple regulations) via a manifest-driven model picker — the Gaussian model lives only in the notebooks. Both have **three pages** under one model picker: `/completer` (full PT sampling by default, or an opt-in fast greedy completion), `/analysis` (per-team diagnostics), `/meta` (Metagame Model, format-wide statistics). User-facing terminology is decoupled from the underlying math: `J` → "Coupling Strength", `h` → "Bias", `ΣJ` → "Coherence", `E` → "Score" (sign-flipped so higher = better), `field_weight` → "Bias Adjustment". The webapp is one deliverable of the larger project — not the single end goal; deeper scientific content lives in the notebooks and the `/science` page.

The static webapp is the primary public surface. The Streamlit app remains as a development/research tool. The TS port mirrors every piece of math; any drift is caught by `tests/test_parity.py` (see "Code duplicated across Python and TypeScript" below).

## Run commands

```bash
# Python side
streamlit run scripts/app.py        # Streamlit webapp
jupyter lab                          # notebooks (in notebooks/)
python -m unittest discover tests    # regression gate (includes parity tests)
python -m k2dex.tournament_ingest     # fetch from Limitless API + import in-person data into cache
python -m k2dex.tournament_ingest --limitless-only   # Limitless API only
python -m k2dex.tournament_ingest --in-person-only   # import tournament_json/ only
python scripts/precompute.py \        # build one model (see precompute CLI below)
  --display-name "Reg M-A Species @ Item" \
  --regulation M-A --type species_item
python scripts/precompute.py --generate-manifest  # rebuild manifest.json after all models built
python scripts/precompute.py --recompute  # rebuild ALL models from their stored meta.json params + refresh manifest
python scripts/scotus_precompute.py  # rebuild web/public/scotus/ artifacts for the /science SCOTUS section (manual)

# Static webapp (all run from web/)
(cd web && npm install)         # one-time
(cd web && npm run dev)         # local dev server at http://localhost:5173
(cd web && npm test)            # JS smoke tests
(cd web && npm run build)       # production build → web/dist/
(cd web && npm run emit-baseline)  # regenerate tests/parity_baseline.json
```

No build system on the Python side; dependencies live in the user's global Python env. **`pandas` and `pyarrow` are intentionally absent** — use markdown tables for tabular display in Streamlit, not `st.dataframe`. Confirm before adding any new dependency (per `~/.claude/CLAUDE.md`).

Static webapp has a `package.json` in `web/` (Vite + React 18 + TypeScript + Vitest + react-select + react-router-dom). No other JS deps. Same confirm-before-adding rule applies.

## Repo layout

```
k2dex/                  Python package — core model library
  constants.py          Shared numeric constants (vocab cutoffs, seeds, etc.)
  helpers.py            Gaussian inverse Ising primitives (Smogon chaos path)
  loaders.py            Pure (J, h, m, vocab) model builders; used by scripts/app.py
                        via @st.cache_resource wrappers and by scripts/precompute.py
                        for direct invocation
  models.py             fit_pl_ising — PL fit for species and species+item models
  sampling.py           MCMC family (swap / anneal / PT) + mean-field + greedy
                        + rank_single_swaps (top-N independent swap suggestions)
  rendering.py          Diagnostic helpers + markdown-table builders (pure, no Streamlit)
  rendering_html.py     Lab-notebook HTML helpers (section labels, stat strips,
                        score chips, signed/mini bars, sprite cells, slot cards,
                        rich-row tables); paired with styles.py classes
  styles.py             Design tokens (LAB.* CSS vars), font import, widget
                        overrides (st.tabs, phase picker, segmented control,
                        primary button, multiselect chips)
  assets/               Static assets bundled into the page as data URIs
                        (currently just missingno.svg, the sprite fallback)
  tournament_ingest.py   Tournament data ingest and cache; see "Data pipeline" below
tournament_json/        In-person tournament raw JSON source (committed dir, JSON gitignored)
  M-A/                  Reg M-A format; files named [id]_[YYYY-MM-DD].json or [id]_[YYYY_MM_DD].json
tournaments_cache/      Unified cache (gitignored); both Limitless + in-person entries
scripts/                CLI entry points (not part of the k2dex package)
  app.py                Streamlit webapp (rendering only; samplers imported)
  precompute.py         Offline pipeline: fits one model per invocation via
                        k2dex.loaders; serializes to web/public/models/<slug>/
                        for the static webapp. Also generates manifest.json.
  scotus_precompute.py  Fits PL inverse Ising on scotus_votes.txt at several
                        N checkpoints; writes web/public/scotus/{votes,fits}.json
                        for the /science page's SCOTUS section
notebooks/              Jupyter notebooks — analysis pipeline (see Pipeline order)
scotus_votes.txt        895 second-Rehnquist-court votes (9 bits per row,
                        ~44% unanimous; column order matches
                        scotus_precompute.JUSTICES)
tests/                  unittest tests + locked validation baselines
                        + parity_baseline.json (JS-side outputs gated by
                          tests/test_parity.py)
web/                    Static React/TypeScript webapp — see "Static web build"
                        below for layout details
.github/workflows/      GitHub Actions deploy workflow (push to main → GH Pages)
```

The "promote when stable" convention still applies: new functions live in their notebook until they stabilize, then move to the relevant `k2dex/` module. Gaussian model primitives are in `k2dex/helpers.py`; the PL fit is in `k2dex/models.py`; the MCMC family is in `k2dex/sampling.py`. Each module's docstring lists what's already promoted.

## Pipeline order

The notebooks form a dependency chain. Run in order, or re-derive intermediate state inside any standalone notebook.

1. **`notebooks/spatial_embedding.ipynb`** — Stages 0–4 from the plan. PPMI → truncated SVD → role-residual NN ranking → NMF role decomposition. Uses `k2dex.helpers` primitives.
2. **`notebooks/inverse_ising.ipynb`** — fit `(J, h)` via the Gaussian / precision-matrix approximation; visualize J. Uses `k2dex.helpers.ising_gaussian`.
3. **`notebooks/j_communities.ipynb`** — standalone (no longer depends on notebook 2). Louvain community detection on the `+J` subgraph (archetype clusters) and `-J` subgraph (role pools) of the **weighted species@item model** (`loaders.build_species_item_model`). The `-J` analysis and the J-graph visualization mask **structural exclusions** first: same-species and same-item feature pairs can never co-occur by game rule (species clause / item clause), so their large negative couplings are rule-driven, not meta-driven, and would otherwise dominate the top of the `-J` distribution.
4. **`notebooks/forward_ising.ipynb`** — forward sampling from the fitted model: swap-move MCMC with the `team_size=6` constraint, calibration against empirical marginals. Uses `k2dex.sampling.swap_mcmc`.
5. **`notebooks/inverse_ising_phase2.ipynb`** — PL species fit; mirrors notebook 2's analysis cells. Uses `k2dex.models.fit_pl_ising`.
6. **`notebooks/validation.ipynb`** — leave-k-out evaluation harness: cross-model comparison (chronological 90/10 split), pair-prediction metric (species+item model), MF-vs-MCMC agreement check. Uses `k2dex.models.fit_pl_ising`.
7. **`notebooks/temporal_drift.ipynb`** — standalone. How quickly does model accuracy degrade as the meta evolves? Chronological 25/75 split; 8 test windows.
8. **`notebooks/energy_discrimination.ipynb`** — standalone. Real-vs-null team scoring, Score decomposition (popularity vs. coherence), and `field_weight` AUC sweep. Chronological 90/10 split. Fits use the v1.1 sample weighting (matching the shipped weighted artifacts); AUC peak is ~0.2 for both models (was ~0.25-0.30 on the unweighted λ=10/λ=1 fits — λ=3 shrinks J ~3x, so the h/J balance shifts and the optimal bias dial moves down with it).
9. **`notebooks/outcome_validation.ipynb`** — standalone. Tests whether Score/coherence predicts tournament placement. Fits on the full corpus (placement is never an input to `(J,h)`, so in-sample scoring isn't leakage). Requires cache schema >= v2.
10. **`notebooks/weighting_sweep.ipynb`** — standalone. Tunes the sample-weighting knobs (`RECENCY_TAU_DAYS`, `IN_PERSON_WEIGHT`): grid over `(τ, multiplier)` scored by leave-one-out completion on held-out *future in-person* events, plus a lambda re-sweep at the winning weights (ESS shrinks under decay). Uses the production `loaders.team_weights` / `fit_pl_ising(sample_weight=...)` path.
11. **`notebooks/three_body_check.ipynb`** — standalone. Pairwise-assumption diagnostic: model-sampled (PT, cold chain at T=1) vs. weighted-empirical connected three-point correlations for both production weighted models, on the top-K features; per-triplet z-scores and worst-violating-triplet tables. In-sample by design (the question is whether the model *family* reproduces third moments of the data it was fit to).
12. **`notebooks/outcome_ceiling.ipynb`** — standalone; the gate for the goodness-model work. Uncontrolled composition-level ICC of within-event placement percentile (exact roster / `+J`-archetype signature / `-J`-role signature / dominant archetype granularities) with within-event permutation nulls, time-windowed against meta drift, plus the in-person match-outcome ICC (winner-flip null; Swiss pairing makes it a conservative bound) and a ΔJ pair-support count for the match corpus. Both signature partitions come from Louvain on the weighted species@item fit with the same graph recipe and seed as `j_communities.ipynb` (deliberately identical partitions — one canonical community structure), structural exclusions masked on the `-J` side. Requires cache schema v3 (match lists).

## Module structure

All modules below live under `k2dex/` and use relative imports within the package.

- **`constants.py`** — vocab cutoffs, corpus size, regularization strength (as lambda, converted to the logistic inverse-strength `C = 1/lambda` internally), validation seed / fraction, ingest filters. Change any knob here, not in the notebook or app. (Static webapp mirrors a subset of these in `web/src/constants.ts` — see duplication register below.)
- **`helpers.py`** — Gaussian model only (Smogon chaos path): `load_chaos`, `build_vocab`, `build_cooccurrence`, `build_ppmi`, `binary_moments`, `binary_correlation`, `ising_gaussian`. Frozen-dataclass return types throughout.
- **`loaders.py`** — pure `build_species_model()` / `build_species_item_model()` that fit `(J, h, m, vocab, team_counts, species_of, item_of, latest_tournament_date)` end-to-end from the cached corpus (via `load_cached_tournaments`). Both accept keyword args `regulation`, `min_team_count`, `lam` (L2 lambda; converted to the logistic inverse-strength `C = 1/lam` internally), the weighting knobs `recency_tau` / `in_person_multiplier`, and the cross-regulation warm-start knobs `prior_regulation` / `intercept_prior_weight` (see "Cross-regulation warm-start" below). `_align_prior(vocab, prior_vocab, prior_J, prior_h)` scatters a donor model's `(J, h)` onto the current vocab order (matched by vocab string, zeros for new features) to feed `fit_pl_ising`'s prior. No Streamlit, no API calls. `scripts/app.py` wraps them in `@st.cache_resource`; `scripts/precompute.py` calls them directly. Also exports `format_pair(species, item)` and `team_weights(observations, *, reference_date, recency_tau, in_person_multiplier)` — the per-team fit weights `w = exp(-Δt/τ)·m^[in-person]`, normalized to `Σw = N` (mean 1).
- **`models.py`** — `fit_pl_ising(X, *, C, max_iter, sample_weight, prior_J, prior_h)` runs V per-spin L2 logistic regressions, skips degenerate spins (weight-aware: a class also needs ≥2 units of weighted mass, which assumes mean-1 weights), symmetrizes via `J = 0.5 * (J_asym + J_asym.T)`, zeros diagonal. Each per-spin fit is solved directly with `scipy.optimize.minimize` (L-BFGS-B) on the penalized logistic objective `½‖w−w_prior‖² + ½·pen_c·(c−c_prior)² + C·Σ wₜ·logloss` — not sklearn. With no prior (`prior_J=prior_h=None`) this is identical to a standard zero-centered L2 logistic fit (intercept unpenalized), which is what every shipped model uses today. Supplying `prior_J`/`prior_h` (a previous model's `(J,h)` aligned onto X's vocab, zeros for new features) re-centers the penalty on the prior and shrinks the intercept toward `prior_h`, so a feature with thin/no evidence in X retains its prior value instead of collapsing to zero — the mechanism for cross-regulation warm-starting. Degenerate spins fall back to the prior (or to zero when no prior). Used by notebooks 5/6 and `loaders.py` (which in turn feeds `scripts/app.py:load_model_{species,species_item}` and `scripts/precompute.py`).
- **`sampling.py`** — `swap_mcmc` / `anneal_mcmc` / `parallel_tempered_mcmc` / `meanfield_marginals` / `greedy_optimize` / `rank_single_swaps`. All accept keyword-only `species_of` / `item_of` for species+item uniqueness constraints (no-duplicate-species, no-duplicate-item). Shared inner loop in `_local_swap_step` so the three MCMC variants stay in sync. **Mirrored 1:1 in `web/src/sampler/`** — every public symbol has a TypeScript counterpart gated by `tests/test_parity.py`.
- **`rendering.py`** — pure-function helpers: `team_obs_count`, `min_swaps_to_observed`, `nearest_observed`, `intra_team_sum_j`, `pairwise_j_rows`, `render_pairwise_j_table`. No Streamlit imports.
- **`rendering_html.py`** — HTML-string builders for the lab-notebook visual language. Atoms (`section_label`, `stat_strip`, `score_chip`, `signed_bar`, etc.), sprite primitives (`sprite_img` via `<object>` fallback, see Webapp visual language gotchas), vocab parsing (`extract_species`/`extract_item`), and composed cells (`slot_card`, `comp_mon_cell`, `pair_cell`, `swap_cell`, `team_mini_strip`, etc.). Class names are paired with `styles.py`.
- **`styles.py`** — design tokens (LAB.* CSS vars, Google Fonts) and Streamlit widget overrides. `inject()` called once at top of `main()`. Forces light-mode palette to override OS dark-mode detection.
- **`tournament_ingest.py`** — tournament data ingest and unified cache. Two data sources feed `tournaments_cache/`: the Limitless API (`fetch_limitless_tournaments`) and in-person tournament exports (`import_in_person_tournaments` from `tournament_json/[format]/[id]_[date].json`). Downstream consumers call only `load_cached_tournaments()`, which is a pure offline cache reader. Each cache entry carries a `"type"` field (`"limitless"` or `"in-person"`) for provenance, surfaced as `TournamentTeams.tournament_type`. A `Team` record holds the roster frozenset (`members`) plus outcome (`placing`, `record`). In-person entries additionally carry per-round **match results** (`TournamentTeams.matches`: tuple of `Match(round, winner, loser)`, winner/loser indexing into `teams`), parsed by `extract_matches` from each standings entry's `rounds` dict. Only decisive reciprocal W/L pairs are kept — byes/`LATE`, ties, double-losses, and matches touching duplicated player names are dropped; player names are used for opponent resolution at parse time only, then discarded. The Limitless API exposes no match-level data, so limitless entries have empty matches. Other key exports: `all_teams`, `all_teams_with_outcomes`, `all_team_observations` (keeps `(roster, date, type)` per team — the input to `loaders.team_weights`), `all_match_observations` (matches resolved to roster frozensets + round/date/tournament id — the input for Bradley-Terry fitting and match-level outcome analysis), `chronological_split`, `species_only_teams`. Cache schema is versioned (`CACHE_VERSION`, currently 3) with a **per-type minimum** (`_MIN_CACHE_VERSION`): v3 only added in-person match lists, so v2 limitless entries stay valid (avoids a multi-hour API re-walk) while v2 in-person entries are auto re-imported on the next `--in-person-only` run. Bump when parsing logic changes. Limitless cache dates are full ISO timestamps, in-person dates bare `YYYY-MM-DD`; anything consuming dates must parse at day resolution (`loaders._parse_day`).

Scripts (under `scripts/`):

- **`precompute.py`** — CLI that fits **one model per invocation** via `k2dex.loaders` and serializes it as `meta.json` (schema v2) + `J.bin` (float32 lower triangle) + `h.bin` + `m.bin` + `team_counts.json` under `web/public/models/<slug>/`. The slug is auto-generated from `--display-name`. Also generates `manifest.json` (via `--generate-manifest`) for webapp model discovery. Lower-triangle packing halves J's bytes; JS-side loader reconstructs symmetric J. `--prior-regulation` / `--intercept-prior-weight` expose the cross-regulation warm-start (recorded in `meta.json:fit` only when a prior is used). `--recompute` rebuilds **every** model from its own stored `meta.json` parameters (lambda, weighting, min-team-count, warm-start prior, description, new-badge) and refreshes the manifest — the one-shot refresh after a fit change, no per-model flags needed. Artifacts are committed to git after manual inspection; CI does **not** regenerate them.
- **`scotus_precompute.py`** — fits PL inverse Ising on `scotus_votes.txt` at several N checkpoints; writes `web/public/scotus/{votes,fits}.json`.
- **`app.py`** — Streamlit webapp. Rendering only; samplers imported from `k2dex.sampling`.

## Webapp visual language

The app uses a "lab notebook" palette + type system delivered via CSS injection. **Brand accent color is forest green: `--lab-accent: oklch(0.42 0.10 145)` in `tokens.css` / `LAB_ACCENT` in `styles.py`.** It must appear on every page: active tab underline, active model card border, focused select outline, primary button fill, minibar fills, and `/science` act-heading dividers. Use `var(--lab-accent)` in new CSS rather than hardcoding a hex. On the `/science` page, interactive simulation controls use `.lab-science-btn` from `widgets.css`. Do NOT write bare `<button>` elements in the science sections. Non-obvious gotchas:

- **Streamlit strips inline event handlers and `<script>` tags.** `sprite_img` (Streamlit/`rendering_html.py`) uses an HTML5 `<object>` with inner `<span>` fallback (`<object>` is the only approach that works). React `Sprite.tsx` uses `<img onError>` instead, with a bottom-right item-icon overlay for Species @ Item features.
- **Sprite slug rules** (`rendering_html.species_to_slug`): first hyphen preserved as forme separator; subsequent hyphens collapse. Species with hyphens in the base name (Ho-Oh, Chien-Pao, Porygon-Z, ...) are in `_HYPHEN_BASE_SPECIES` and have all hyphens stripped. Missing sprites fall through to `assets/missingno.svg`.
- **Widget-scoping via marker divs + `:has()`.** Emit an invisible marker div before a widget's column, then target via `[data-testid="stColumn"]:has(.marker-class) ...`. Search styles.py for `:has(` to find all instances. Breaks if Streamlit changes its `data-testid` hierarchy.
- **OS dark mode bleeds into widget internals** unless `--primary-color` / `--background-color` / `--text-color` are forced on `:root` AND on `html[data-theme="dark"]`. `.streamlit/config.toml` would be the idiomatic fix but is gitignored.
- **Math typesetting**: `/science` uses KaTeX via `react-katex` (lazy-imported in `SciencePage`; wrapped by `science/widgets/Math.tsx`). Streamlit uses `st.latex`; no shared dependency.
- **Responsive / mobile layout.** Desktop-first, two width breakpoints (`@media (max-width: 900px)` "narrow", `@media (max-width: 640px)` "phone") plus `@media (pointer: coarse)` for touch-target sizing. Convention is documented in a comment block at the top of `tokens.css`; breakpoint values are consistent literals across all four CSS files (media queries can't consume CSS vars). All responsive rules are additive overrides at the bottom of each stylesheet; **never rename existing `lab-*` classes** (they're mirrored in `rendering_html.py`), only add new ones. Key patterns: `.lab-form-grid` / `.lab-split-pair` utility classes replace inline grid styles so media queries can collapse them; `.lab-table-cards` modifier on the three wide interactive tables (Completion / Swaps / Chain) triggers a card-row restructure at phone width; `ScrollX` wrapper provides scroll-shadow affordance for simpler tables; all SVG widgets use `viewBox` so they scale down to container width via `max-width: 100%; height: auto`; the `CouplingGraph` wrapper is fluid-width with `aspect-ratio: 1/1` and `maxWidth` capped at the coordinate-space `viewSize` (no ResizeObserver relayout). The `Stat` help "?" is a `<button>` with a tap-to-toggle popover (`lab-stat-pop`), not a hover `title=` attribute.

## Conventions threaded through multiple files

Easy to get wrong if you only look at one file:

- **`CURRENT_REGULATION`** (`constants.py` + `constants.ts`) determines the active regulation for the webapp model picker (models from other regulations are collapsed under a "Legacy" toggle) and serves as the default for `--regulation` in `precompute.py` and both loaders. Update this constant when a new regulation begins.
- **Regulation relabeling (Limitless lag).** When Limitless has not yet tagged a new regulation, its catalog keeps serving the superseded label (or `CUSTOM`) for events that are really the new format. `tournament_ingest._REGULATION_RELABELS` recovers them: a `RegulationRelabel` rule promotes a tournament to its `target` when the catalog date is on/after `start_date`, the catalog label is in `from_labels`, **and** a roster runs any `marker_species`/`marker_items` (picks legal only in the target — illegal in the stale format — so a single appearance is necessary and sufficient). The current rule covers **M-B** (start `2026-06-17`, from `{M-A, CUSTOM}`, markers Gholdengo/Metagross/Grimmsnarl + Life Orb/Light Clay). `could_match_regulation` is the cheap standings-free pre-filter (date+label) that bounds which events get fetched; `resolve_regulation` is the post-parse marker confirmation. Resolution is symmetric: a post-cutoff `M-A`-labeled event with markers is excluded from an M-A ingest and re-cached as M-B, and a stale entry cached under the wrong label is re-fetched so item-stripping runs under the true regulation. Add the marker species/item to `_REGULATION_RELABELS` when starting a new regulation Limitless hasn't tagged; remove the rule once Limitless tags it directly. The relabel re-extracts under the resolved regulation, so `_ILLEGAL_ITEMS_BY_REGULATION[target]` governs item-stripping for promoted events — M-B mirrors M-A's bans (Covert Cloak, Assault Vest, Safety Goggles).
- **Cross-regulation warm-start.** A sparse new regulation can borrow a previous one's structure via `prior_regulation` (loaders) / `--prior-regulation` (precompute). This assumes the prior is a **legality superset** (everything legal in the donor stays legal here — true for M-A → M-B): the donor model is fit on its own corpus with default knobs, its **entire** vocab is folded into the new model unconditionally, and `_align_prior` scatters its `(J, h)` onto the new vocab so `fit_pl_ising` re-centers each per-spin L2 penalty on the prior instead of zero. A donor feature with no teams in the new regulation hits the degenerate-spin skip and keeps its prior `(J row, h)` exactly; a feature with data relaxes off the prior as evidence accumulates, so the warm-start self-extinguishes per-feature without a λ schedule. `intercept_prior_weight` (default 1.0) sets how hard `h` is pulled to `prior_h` relative to the coupling penalty (0.0 = bias free); it's the most arguable knob (biases *should* track the new meta, but anchoring tames thin-feature ranking noise) and the first thing to sweep on the boundary-validation harness. No TypeScript changes — warm-started models are ordinary manifest entries.
- **Gaussian model vocab cutoff is `min_usage=0.002`** (~170 Pokémon at Reg M-A 1760). PL model vocab cutoff is `PHASE2_MIN_TEAM_COUNT = 5` (feature must appear in ≥5 teams).
- **Regularization differs by model.** `SPECIES_LR_LAMBDA = 25.0`; `SPECIES_ITEM_LR_LAMBDA = 4.5` (sparser pair matrix still wants weaker regularization than species). Both were raised from their original values (10.0 / 1.0) after `regularization_sweep.ipynb` was re-run on the grown corpus: the unweighted random-split optima had drifted up (species PLL-optimal ~21, MRR-optimal ~46; pair MRR-optimal ~7.5 on a flat 5-10 plateau), a genuine data shift rather than a weighting or eval-split artifact. Lambda is the canonical L2 penalty; converted to the logistic inverse-strength `C = 1/lambda` internally. Each `loaders` builder passes its own default; `precompute.py --lambda` overrides it. `meta.json` records `fit.lambda`. `notebooks/regularization_sweep.ipynb` is the sweep harness behind these values.
- **Internal research default model.** New research/notebook work should use the **weighted species@item fit** — `loaders.build_species_item_model()` with its constant defaults, which after the post-NAIC re-sweep *are* the weighted knobs (`RECENCY_TAU_DAYS`, `IN_PERSON_WEIGHT`, `SPECIES_ITEM_LR_LAMBDA`) and reproduce the shipped `reg-m-a-species-item-weighted` fit family — unless there is a stated reason not to (species-only vocab when its statistical power is specifically needed, via `build_species_model()`; or explicit unweighted knobs for comparison baselines). Pull weighting/λ from `k2dex.constants` rather than hardcoding literals, so notebooks track the constants and don't drift when they're re-swept. Notebooks written before this convention may still hardcode old knobs; update them when touched.

- **Sample weighting (weighted base model).** Per-team fit weight `w = exp(-Δt/τ)·m^[in-person]`, normalized to mean 1 (`loaders.team_weights`). Knobs: `RECENCY_TAU_DAYS` / `IN_PERSON_WEIGHT` in `constants.py` (neutral — `None` / `1.0` — until `weighting_sweep.ipynb` picks values), overridable via `precompute.py --tau / --in-person-weight`; both recorded in `meta.json:fit` (`recency_tau_days`, `in_person_weight`). Under weighting, `m` is *weighted* (it's what the fit sees) and the vocab cutoff requires `min_team_count` in **both** raw and weighted counts — raw so a single highly-weighted recent team can't mint one-off vocab entries (which land on the degenerate-spin path and get h=0 exactly, outranking every real feature at full Bias Adjustment), weighted so features whose support has fully decayed drop out. `team_counts` stays a raw count for display. No TypeScript changes — weighted models are ordinary manifest entries.
- **`MIN_TEAMS_PER_TOURNAMENT = 64`** filters out small/quirky tournaments at ingest time. Below this they tended to be majority-Bo1 or unusual local metas.
- **`LIMITLESS_MAX_TEAMS = 25000`** is the Limitless API fetch limit (stop walking the API after this many teams). It does not cap the model corpus; model fitting uses all cached data from both Limitless and in-person sources. The CLI `--start-page` flag lets callers skip past recent catalog pages to reach older formats.
- **Smogon's `Teammates` values are skill-weighted floats, not raw integer counts**; we treat them as proportional to team-appearances. PPMI is computed in probability space so absolute counts cancel.
- **Field h**: the Gaussian model derives it from mean-field self-consistency; the PL models give h directly as the per-spin logreg intercept.
- **Energy formulas**: raw `H(s) = -h·s - 0.5 s'Js`. Adjusted `H_adj(s) = -(field_weight·h)·s - 0.5 s'Js`. At `field_weight=0` only the pairwise term remains, dropping the popularity prior — this is the knob for surfacing archetype-coherent completions when standard sampling is too meta-biased.
- **Completer mode + Bias Adjustment defaults.** Web page-load default is `0.5` (`web/src/state/PageStateContext.tsx`, completer and analysis). The Greedy checkbox sets bias in both directions: checking it flips to `0.8` (greedy wanders into incoherent basins at low weight with few pins), unchecking returns to `0.3` — so a greedy round-trip lands at `0.3`, not the page-load `0.5` (`CompleterPage.tsx`). Streamlit sliders default `0.3` (`scripts/app.py`). The three values have drifted; `energy_discrimination.ipynb` (now fit on the weighted models) is the harness for re-deciding them.
- **Clipboard import + shareable URLs (static webapp only).** Both `/completer` and `/analysis` parse pokepastes via `render/vocab-match.ts` (slug-based matching bridges Showdown forme names to corpus names, with mega-normalization mirroring `tournament_ingest.strip_mega_prefix` and a small `IMPORT_SPECIES_ALIASES` map for edge cases). OOV species/items surface as `.lab-form-note` warnings. Both pages **live-sync state into the URL** (`render/shareLink.ts`), encoded by slug (not index) so links survive vocab reorders. Token format: `<modelSlug>.<fwIndex>.<mons>`, where model slug is the auto-generated id from manifest. Legacy codes `"s"` / `"si"` decode to `"reg-m-a-species"` / `"reg-m-a-species-item"` for backward compatibility. The completer token carries the RNG seed so shared links reproduce exact PT results. No Python twin, so no parity rows.
- **Swap-move MH** for fixed team size: simultaneously turn one currently-on mon off and one currently-off mon on (single-spin flips break the size constraint). Energy diff `ΔH = h_eff[i_out] − h_eff[i_in] + (J[i_out] − J[i_in]) · s + J[i_in, i_out]`. The last term corrects for the swapped pair's mutual coupling — easy to forget.
- **Validation split is chronological 90/10** (`VALIDATION_TEAM_FRAC_TEST = 0.10`), out-of-time (fit on past, test on future). Vocab is built from train only; OOV test teams are dropped. `VALIDATION_SEED = 42` seeds the within-test-set held-out position RNG, not the split itself. Temporal drift uses a separate 25/75 split (`DRIFT_TRAIN_FRAC = 0.25`).
- **Vocab normalization** at ingest time collapses case/whitespace variants (`'Sitrus Berry'` / `'sitrus berry'` → one bucket) AND collapses `Mega <X>` species names down to base species. The cache is invalidated when either normalizer changes (bump `CACHE_VERSION`).

## Static web build

The `web/` tree is a self-contained Vite + React 18 + TypeScript app that ports the Streamlit UI to a static site for GitHub Pages deployment. Its sampler math is a faithful port of `k2dex/sampling.py`; rendering helpers mirror `k2dex/rendering.py` + `k2dex/rendering_html.py`; styles port `k2dex/styles.py` to plain CSS.

```
web/
  index.html
  package.json              Vite + React 18 + TS + Vitest + react-select + react-router-dom
  tsconfig.json
  vite.config.ts            base path from VITE_BASE_PATH (production: / — custom domain root)
  scripts/
    emit-parity-baseline.ts  Node script that emits tests/parity_baseline.json
    prerender-routes.ts      Post-build: writes per-route dist/<route>/index.html
                             (200 + unique title/canonical) + dist/sitemap.xml
                             from src/siteMeta.ts. Run as last step of npm build.
  public/
    assets/missingno.svg     also copied into web/src/assets/ for ?url import
    404.html                 GH Pages SPA fallback (humans only — see SEO note)
    robots.txt               Allow all + Sitemap: line (sitemap itself generated)
    CNAME                    custom domain (k2dex.kyletunis.com) — see deploy note
    models/manifest.json    ← model discovery index (precompute.py --generate-manifest)
    models/<slug>/           ← per-model artifacts (precompute.py), committed to git
    scotus/{votes,fits}.json ← scotus_precompute.py output, committed to git
  src/
    main.tsx, App.tsx        bootstrap + router (includes /science route)
    siteMeta.ts              Single source of truth for per-route title/desc/
                             canonical; consumed by usePageMeta + prerender script
    usePageMeta.ts           Hook: syncs <head> (title/canonical/OG) on SPA nav
    constants.ts             mirrors a subset of constants.py
    assets/                  Vite-handled imports (missingno fallback)
    state/manifest.ts       ModelSummary/Manifest types + loadManifest()
    state/ModelContext.tsx   manifest-driven model loader; caches (J, h, m, team_counts) per slug
    components/              Layout, ModelPicker, VocabSelect (react-select wrapper),
                             CouplingGraph (shared force-directed coupling graph),
                             ScrollX (horizontal-scroll wrapper with scroll-shadow)
    pages/{Completer,Analysis,Meta,Science}Page.tsx
    sampler/                 1:1 port of k2dex/sampling.py — see duplication register
      types.ts, rng.ts, model.ts, energy.ts,
      meanfield.ts, greedy.ts, swap.ts, pt.ts, rank.ts
    render/                  1:1 port of k2dex/rendering.py + k2dex/rendering_html.py
      format.ts, corpus.ts, sprite-url.ts, Sprite.tsx, atoms.tsx,
      cells.tsx, observables.ts
      vocab-match.ts         web-only: slug-bridge matching of external
                             names (pasted pokepastes, shared-link slugs)
                             to vocab indices; parsePokepaste + matchPaste
      shareLink.ts           web-only: shareable-URL token codec (no Python
                             twin, no parity row)
    completer/               Page-local logic: fastPath, ptWorker, ptDriver,
                             CompletionRow
    analysis/                Page-local tables: PairwiseJTable, SwapsTable, ChainTable
    meta/                    Page-local tables: FeatureBiasTable,
                             ExtremeCouplingsTable, couplings,
                             CouplingNetwork (unused; retained for potential reuse)
    science/                 /science page — see dedicated section below
      primitives/            Toy lattice/graph/mcmc/mf/pt/landscape
      widgets/               Math, LinePlot, SpinGrid, GraphView, ChainStrip, Landscape3D
      sections/              Magnets, Lattice, Graph, Metropolis, ParallelTempering,
                             MeanField, SCOTUS, Pokemon, References
                             Three act headers (h2.lab-science-act); section titles h3;
                             subheads h4.lab-science-subhead
      data/scotusLayout.ts   Fixed 2D positions for the 9 justices
      __tests__/             vitest smoke tests for the toy primitives
    styles/                  tokens.css, layout.css, components.css, widgets.css
  __tests__/                 vitest smoke tests for stochastic samplers
```

**Build artifacts pipeline:**
0. Run `python -m k2dex.tournament_ingest` to populate the cache (Limitless API fetch + in-person import). This is the only step that makes network calls.
1. Run `python scripts/precompute.py --display-name "..." --regulation ... --type ...` once per model. This reads from the cache only (offline). Each invocation writes artifacts to `web/public/models/<slug>/`.
2. Run `python scripts/precompute.py --generate-manifest` to rebuild `manifest.json` from all model directories.
3. Inspect `web/public/models/<slug>/{meta.json,J.bin,...}` and `manifest.json` for sanity (vocab size, n_corpus_teams, file sizes).
4. Commit the artifacts. CI is not allowed to regenerate them (the user wants manual eyeballing of every refresh).
5. `npm run build` reads the artifacts as static files (Vite copies `public/` into `dist/`).

**Deployment domain.** The site is served at the root of the custom subdomain **`k2dex.kyletunis.com`** (a CNAME of `kylehtunis.github.io`), not the old `kylehtunis.github.io/k2dex` subdirectory. Three things keep this consistent and must move together if the domain ever changes: `web/public/CNAME` (the custom-domain marker GitHub Pages reads from the built artifact), `VITE_BASE_PATH` in `.github/workflows/deploy.yml` (set to `/` so assets resolve at the domain root — local/dev also default to `/`), and `SITE_URL` in `src/siteMeta.ts` (drives every canonical, OG URL, and the sitemap). The github.io subdirectory URLs 301-redirect to the custom domain automatically once the domain is set in repo Settings → Pages. The subdomain (vs. a subdirectory) is what lets Google honor the declared site name (`og:site_name` + `WebSite` JSON-LD in `index.html`); Google only attributes site names at the domain/subdomain level. The `404.html` SPA-fallback `base` is `""` (root) to match. GA (`gtag.js`, measurement id in `index.html`) is domain-agnostic and unaffected by the move.

**SEO / indexability (per-route static HTML).** `scripts/prerender-routes.ts` runs as the last step of `npm run build` and writes a real `dist/<route>/index.html` per route (HTTP 200, unique title/description, self-referencing canonical) plus `dist/sitemap.xml`, all driven from `src/siteMeta.ts`. **Adding a route means adding one entry to `src/siteMeta.ts`** — it feeds the prerendered HTML, the runtime head, and the sitemap together. `public/sitemap.xml` does **not** exist; it's generated (don't reintroduce a static one).

**Parallel-tempered sampler** runs inside a Web Worker (`web/src/completer/ptWorker.ts`) so the UI stays responsive during the 10–30s sample. Structured-clone passes the model (~2.5 MB J for the species+item model) into the worker; not a performance issue at this scale.

**Sprite rendering** (`web/src/render/Sprite.tsx`): `<img onError>` → missingno fallback; Species @ Item features get a bottom-right item-icon overlay via `itemSpriteUrl()`. Slug rules in `sprite-url.ts` mirror `rendering_html.species_to_slug`.

## /science page

Interactive explainer for the math behind the project. Lives under `web/src/science/`; rendered at `/science`. Structure is in the directory tree above (three act headers grouping eight sections). Non-obvious facts:

- **Toy primitives are independent of the production sampler.** `science/primitives/` are small re-implementations for the pedagogical case (single-spin flips, no team-size constraint). They never import from `web/src/sampler/`. No parity-test obligation.
- **PT swap-acceptance sign**: the correct exponent is `(beta_cold - beta_hot)(H_cold - H_hot)`. The simulation appears to "work" under the wrong sign due to two-well symmetry. Double-check if you touch `primitives/pt.ts`.
- **Two spin conventions on one page.** Magnets/Lattice use +/-1; Graph and downstream use {0, 1}. The Graph section's prose calls out the switch.
- **Synthetic 2D energy landscape** (`primitives/landscape.ts`): a two-well surface rendered as an SVG isometric mesh by `Landscape3D.tsx`. Drives both the PT section (walkers) and MeanField section (gradient descent). A caricature for basin-trapping intuition, not a projection of any real Ising graph.
- **Animation rates** are per-section `STEPS_PER_FRAME` constants. Tune if the page feels too fast/slow.
- **SCOTUS section**: 3-state pin chips, exact enumeration (2^(9-|pinned|) <= 512). Color convention: **conservative = red, liberal = blue** (political-map standard). Vote data encodes `1 = conservative, 0 = liberal`. Graph-edge colors are unrelated (coupling sign: positive = blue, negative = red).
- **`GraphView` node render modes**: default circle, `sprite` (raw URL, e.g. SCOTUS justices), or `feature` (Pokemon vocab string, renders via `SpriteImg` with item overlay). Prefer `feature` for Pokemon nodes.
- **Pokemon section** uses the user's active model (from `useModel()`) for both prose and the coupling graph figure. No separate model load. Reuses `CouplingGraph.tsx` (shared with Metagame Model page). Prose counts and regulation name are read live from the model, not hardcoded.
- **Citations**: `References.tsx` is the single source of truth. Inline citations are `<a href="#ref-...">` anchors. Add new citations there; don't scatter free-floating references.

## Code duplicated across Python and TypeScript

Every duplicated symbol below must stay in sync. Editing one side without the other is allowed only if a parity test fails first and the fix updates both sides plus the gate.

The invariant: **no Python↔TypeScript code duplication is allowed without a gate test.** New duplications added in future work must add a row to this table and a corresponding subtest in `tests/test_parity.py`.

| Python (`k2dex/`) | TypeScript | Gate test |
| :--- | :--- | :--- |
| `k2dex.sampling.team_energy` | `sampler/energy.ts:teamEnergy` | parity (indirect, via MF/greedy) |
| `k2dex.sampling.build_constraint_sets` | `sampler/energy.ts:buildConstraintSets` | parity (indirect) |
| `k2dex.sampling.swap_violates_uniqueness` | `sampler/energy.ts:swapViolatesUniqueness` | parity (indirect) |
| `k2dex.sampling.initialize_state` | `sampler/energy.ts:initializeState` | JS smoke + parity (indirect) |
| `k2dex.sampling.meanfield_marginals` | `sampler/meanfield.ts:meanfieldMarginals` | `test_parity.py::test_meanfield_cases` |
| `k2dex.sampling.greedy_optimize` | `sampler/greedy.ts:greedyOptimize` | `test_parity.py::test_greedy_cases` |
| `k2dex.sampling._local_swap_step` | `sampler/swap.ts:localSwapStep` | JS smoke + parity (indirect) |
| `k2dex.sampling._init_chain` | `sampler/swap.ts:initChain` | JS smoke |
| `k2dex.sampling.swap_mcmc` | `sampler/swap.ts:swapMcmc` | JS smoke (stochastic — no parity baseline) |
| `k2dex.sampling.parallel_tempered_mcmc` | `sampler/pt.ts:parallelTemperedMcmc` | JS smoke (stochastic) |
| `k2dex.sampling.rank_single_swaps` | `sampler/rank.ts:rankSingleSwaps` | `test_parity.py::test_rank_cases` |
| `k2dex.rendering.intra_team_sum_j` | `render/observables.ts:intraTeamSumJ` | `test_parity.py::test_obs_cases` |
| `k2dex.rendering.pairwise_j_rows` | `render/observables.ts:pairwiseJRows` | `test_parity.py::test_obs_cases` |
| `k2dex.rendering.nearest_observed` | `render/corpus.ts:nearestObserved` | `test_parity.py::test_corpus_cases` |
| `k2dex.rendering_html.species_to_slug` + `_HYPHEN_BASE_SPECIES` | `render/sprite-url.ts:speciesToSlug` + `HYPHEN_BASE_SPECIES` | `test_parity.py::test_species_to_slug_cases` |
| `k2dex.rendering_html.extract_species` / `extract_item` | `render/format.ts:extractSpecies` / `extractItem` | covered indirectly by parity baseline construction |
| `k2dex.loaders.format_pair` | `render/format.ts:formatPair` | covered indirectly (precompute writes the joined string into `meta.json:vocab`) |
| `scripts.precompute.pack_lower_triangle` (writer) | `sampler/model.ts:unpackLowerTriangle` (reader) | JS smoke + `tests/test_precompute.py` round-trip |
| `scripts.precompute.serialize_team_counts` schema | `sampler/model.ts:loadTeamCounts` + `render/corpus.ts:teamKey` | `tests/test_precompute.py` + `test_parity.py::test_corpus_cases` |
| `k2dex.constants.{TEAM_SIZE, CURRENT_REGULATION, FIELD_WEIGHT_OPTIONS, ...}` | `web/src/constants.ts` | manual (rarely changes; values pinned by v1 simplification) |
| `k2dex.rendering_html` HTML class names (`lab-slot`, `lab-comp-table`, …) | `web/src/render/*.tsx` className strings + `web/src/styles/components.css` | visual — class renames must touch both sides |
| `k2dex.styles` `LAB_*` constants + CSS rules | `web/src/styles/{tokens,layout,components,widgets}.css` | visual — palette/typography changes touch both |

Parity baseline regeneration:

```bash
cd web && npm run emit-baseline   # writes tests/parity_baseline.json (JS-side outputs)
cd .. && python -m unittest tests.test_parity   # Python rerun + compare
```

The synthetic model used in the baseline is hand-built (V=12, team_size=4) so both stacks compute it byte-identically. Tolerance is `1e-9` for deterministic cases (MF, greedy, rank, obs); stochastic cases (swap MCMC, PT) are smoke-tested for well-formed output only.

## Data file

`gen9championsvgc2026regma-1760.json` (VGC Reg M-A Champions, 1760 elo cutoff) is the current target metagame. Do not reintroduce BSS variants.

## External reference

`~/Projects/k2dex-calculator/scripts/update_meta.py` — sibling project's Smogon chaos-JSON downloader/parser. Reuse if we ever need to re-fetch or move to a different format-month.
