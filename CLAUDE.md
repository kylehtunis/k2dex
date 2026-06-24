# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code applying complexity-science / distributional-semantics methods to competitive Pokemon (VGC) team-composition data. Three inverse Ising pairwise-MaxEnt models fit `(J, h)` to different data:

- **Gaussian model** — precision-matrix approximation on Smogon chaos JSON. Lives in notebooks only.
- **Species model** — pseudo-likelihood (PL) fit on Limitless tournament rosters, species-only vocab.
- **Species+Item model** — PL fit on `(species, item)` pair features from the same Limitless data.

Two coexisting webapp surfaces: Streamlit (`scripts/app.py`, dev/research tool) and a static React/TS port (`web/`, primary public surface on GitHub Pages). Both surface precomputed models via a manifest-driven picker with three pages: `/completer`, `/analysis`, `/meta`, plus a `/science` explainer page (static webapp only). User-facing terminology: `J` = "Coupling Strength", `h` = "Bias", `ΣJ` = "Coherence", `E` = "Score" (sign-flipped), `field_weight` = "Bias Adjustment".

## Run commands

```bash
# Python side
streamlit run scripts/app.py
jupyter lab                              # notebooks in notebooks/
python -m unittest discover tests        # regression gate (includes parity tests)
python -m k2dex.tournament_ingest        # fetch Limitless API + import in-person data
python scripts/precompute.py \
  --display-name "Reg M-A Species @ Item" --regulation M-A --type species_item
python scripts/precompute.py --generate-manifest   # rebuild manifest.json
python scripts/precompute.py --recompute           # rebuild ALL models from stored meta.json
python scripts/scotus_precompute.py                # rebuild scotus artifacts

# Static webapp (from web/)
(cd web && npm install)          # one-time
(cd web && npm run dev)          # localhost:5173
(cd web && npm test)             # JS smoke tests
(cd web && npm run build)        # production build -> web/dist/
(cd web && npm run emit-baseline)  # regenerate tests/parity_baseline.json
```

No build system on the Python side; global env. **`pandas` and `pyarrow` are intentionally absent.** Confirm before adding any new dependency.

## Repo layout

```
k2dex/                  Python package (constants, helpers, loaders, models, sampling, rendering, rendering_html, styles, tournament_ingest)
tournament_json/        In-person tournament raw JSON (committed dir, JSON gitignored)
tournaments_cache/      Unified cache (gitignored)
scripts/                app.py (Streamlit), precompute.py, scotus_precompute.py
notebooks/              Analysis pipeline (numbered chain 1-10 + standalone diagnostics: pl_three_body_check, outcome_ceiling, boltzmann_learning)
tests/                  unittest + parity_baseline.json
web/                    Static React/TS webapp (Vite + React 18)
  src/sampler/          1:1 port of k2dex/sampling.py
  src/render/           1:1 port of k2dex/rendering.py + rendering_html.py
  src/science/          /science page (toy primitives independent of production sampler)
  public/models/        Precomputed model artifacts (committed)
```

Promote-when-stable convention: new functions live in notebooks until stable, then move to `k2dex/`.

## Key module notes

Read the code for full APIs. These are the non-obvious parts:

- **`models.py`** — `fit_pl_ising` uses `scipy.optimize.minimize` (L-BFGS-B), not sklearn. Degenerate spins (< 2 units weighted mass) are skipped and fall back to prior or zero. Prior support enables cross-regulation warm-starting. `fit_boltzmann_ising` is the alternative **moment-matching** fit (constrained-MaxEnt / Boltzmann learning): warm-starts from the PL fit, then ascends the regularized log-likelihood with model moments from a persistent batched swap-chain bank (PCD). `empirical_moments(X, w)` returns the weighted 1-pt/2-pt targets.
- **`sampling.py`** — All MCMC variants share `_local_swap_step`. All accept `species_of`/`item_of` for uniqueness constraints. **Mirrored 1:1 in `web/src/sampler/`**, gated by `tests/test_parity.py`. `estimate_moments` (free PT generation → 1-pt/2-pt model moments) is **training-only Python — NOT ported to TS, no parity row** (the webapp consumes the resulting `(J,h)`, never the training loop). Same for `fit_boltzmann_ising` and the `_batched_*` swap/moment helpers in `models.py`.
- **`loaders.py`** — `build_species_model()` / `build_species_item_model()` accept `prior_regulation`/`intercept_prior_weight` for warm-start. `team_weights` computes `w = exp(-dt/tau) * m^[in-person]`, normalized to mean 1.
- **`tournament_ingest.py`** — Cache schema versioned (`CACHE_VERSION`); bump when parsing changes. Limitless dates are ISO timestamps, in-person dates bare `YYYY-MM-DD`; consumers must parse at day resolution.

## Conventions that span multiple files

- **`CURRENT_REGULATION`** (`constants.py` + `constants.ts`) — determines active regulation for webapp model picker; default for `--regulation` in precompute and loaders. Update when a new regulation begins.
- **Regulation relabeling** — `tournament_ingest._REGULATION_RELABELS` promotes mis-labeled Limitless events using marker species/items. Current rule covers **M-B** (start `2026-06-17`, from `{M-A, CUSTOM}`, markers Gholdengo/Metagross/Grimmsnarl + Life Orb/Light Clay). `_ILLEGAL_ITEMS_BY_REGULATION[target]` governs item-stripping for promoted events. Add marker rules when starting a new regulation Limitless hasn't tagged; remove once tagged.
- **Cross-regulation warm-start** — `prior_regulation` (loaders) / `--prior-regulation` (precompute). Assumes prior is a legality superset. Donor vocab folded in unconditionally; `_align_prior` scatters `(J, h)` onto new vocab. Features with no new data keep prior exactly; features with data relax off it. `intercept_prior_weight` (default 1.0) controls bias anchoring.
- **Regularization** — `SPECIES_LR_LAMBDA = 25.0`; `SPECIES_ITEM_LR_LAMBDA = 4.5`. Lambda is canonical L2 penalty, converted to `C = 1/lambda` internally. `meta.json` records `fit.lambda`.
- **Boltzmann learning** (`fit_boltzmann_ising`, `notebooks/boltzmann_learning.ipynb`) — the moment-matching alternative to PL, motivated by the `pl_three_body_check` finding (the renamed, historical `three_body_check`) that PL matches conditionals not moments. On the M-A corpus it cuts the worst-feature marginal error from ~0.34 (PL over-concentration) to ~0.05. Gotchas: (1) **the model is sampled at T=1** — `estimate_moments`' cold chain (ladder index 0) MUST be at T=1 to sample `P ∝ exp(-H)`; a cold rung below 1 samples a sharper distribution (note: `pl_three_body_check`'s ladder starts at 0.5, a latent bug there — the new notebook fixes it). (2) **Reg fixes the gauge** — on the fixed-size manifold `(J,h)` is gauge-degenerate (`h→h+c·1`, `J_ij→J_ij+a_i+a_j`); the non-gauge-invariant L1/L2 penalty toward zero picks the min-norm representative, so keep `reg_lambda>0` (small for tightest moment match — larger deliberately shrinks `(J,h)`). (3) **The dominant fit error is the stochastic-gradient noise ball, killed by `lr_final`/`avg_last`** — a fixed lr orbits the optimum rather than settling, so a single final iterate is displaced most in the strong-coupling directions; the moment-reconstruction scatter then fans into a *cone* (worst-feature pair bias ~0.04, marginal ~0.11). Cosine step-decay (`lr_final≈lr/20`) **or** Polyak iterate averaging (`avg_last`) cuts that **~7×** to ~0.006, for free (same budget) — precompute enables lr decay by default. This was diagnosed *not* to be regularization (a 100× λ sweep didn't move it) or mixing (more sweeps/tempering didn't fix it; 2× `n_chains` helped only via lower gradient noise). (3b) **Read `mean_resid_*`, not `max_resid_*`** in the live history — the per-iteration max is a single `n_chains`-sample snapshot, mostly Monte-Carlo noise; confirm true convergence by re-estimating the fitted model with many samples via `estimate_moments`. The PCD bank is batched across chains in pure NumPy with sparse on-bit moment accumulation (cost independent of V). Parallel tempering (`n_temps>1`) is implemented but **defaults off** — single-T PCD sufficed (not mode-trapped); reserve it for genuinely multimodal models. (4) **Species ensemble passes `species_of=None`** (species distinct per vocab entry → no uniqueness needed); species+item passes the real lookups + a co-occurrence `support_mask`.
- **Sample weighting** — `RECENCY_TAU_DAYS` / `IN_PERSON_WEIGHT` in `constants.py`. Vocab cutoff requires `min_team_count` in **both** raw and weighted counts.
- **Energy formulas** — raw `H(s) = -h*s - 0.5 s'Js`. Adjusted uses `field_weight*h`. At `field_weight=0` only pairwise term remains.
- **Swap-move MH** — energy diff `dH = h_eff[out] - h_eff[in] + (J[out] - J[in]) * s + J[in, out]`. Last term corrects for swapped pair's mutual coupling.
- **Bias Adjustment defaults** — Web page-load: `0.5`. Greedy on: `0.8`, greedy off: `0.3`. Streamlit: `0.3`. Values have drifted; `energy_discrimination.ipynb` is the harness.
- **Shareable URLs** — `render/shareLink.ts` encodes by slug (not index). Token format: `<modelSlug>.<fwIndex>.<mons>`. Legacy codes `"s"`/`"si"` decode to `"reg-m-a-species"`/`"reg-m-a-species-item"`.
- **Validation split** — chronological 90/10 (`VALIDATION_TEAM_FRAC_TEST = 0.10`). Vocab from train only; OOV test teams dropped.
- **Vocab normalization** — ingest collapses case/whitespace and `Mega <X>` to base species. Bump `CACHE_VERSION` if normalizers change.

## Webapp visual language

**Brand accent: `--lab-accent: oklch(0.42 0.10 145)`** in `tokens.css` / `LAB_ACCENT` in `styles.py`. Use `var(--lab-accent)` in new CSS. Key gotchas:

- **Streamlit strips `<script>` and inline handlers.** `sprite_img` uses `<object>` with inner `<span>` fallback. React `Sprite.tsx` uses `<img onError>`.
- **Sprite slug rules** (`species_to_slug`): first hyphen preserved as forme separator; subsequent collapse. `_HYPHEN_BASE_SPECIES` has all hyphens stripped. Missing -> `missingno.svg`.
- **Widget-scoping** via marker divs + `:has()` in `styles.py`. Breaks if Streamlit changes `data-testid` hierarchy.
- **Responsive**: desktop-first, breakpoints `900px` (narrow) and `640px` (phone) + `pointer: coarse`. **Never rename existing `lab-*` classes** (mirrored in `rendering_html.py`).
- **`/science` controls** use `.lab-science-btn` from `widgets.css`. No bare `<button>` elements.

## /science page gotchas

- **Toy primitives** (`science/primitives/`) are independent of production sampler. No parity obligation.
- **PT swap-acceptance sign**: correct exponent is `(beta_cold - beta_hot)(H_cold - H_hot)`. Wrong sign appears to work due to two-well symmetry.
- **Two spin conventions**: Magnets/Lattice use +/-1; Graph and downstream use {0, 1}.
- **SCOTUS section**: conservative = red, liberal = blue. Vote data `1 = conservative`. Graph-edge colors are coupling sign (positive = blue, negative = red), unrelated.
- **Pokemon section** uses active model from `useModel()`. Prose counts are live, not hardcoded.
- **Citations**: `References.tsx` is the single source of truth.

## Build and deploy

**Build artifacts pipeline**: ingest -> precompute per model -> generate manifest -> inspect -> commit. CI does **not** regenerate artifacts. `precompute.py --method boltzmann` (with `--bz-*` knobs) builds a Boltzmann-fit artifact instead of PL — **same artifact format, no TS/parity changes** (only the fitted numbers + `meta.fit.method`/`meta.fit.boltzmann` differ); `--recompute` reads the boltzmann block back. This is an experimentation convenience, not a production path; if you ever make one the webapp default, recheck `field_weight` (moment-matched `h` wants `field_weight ≈ 1`, unlike PL's down-weighted `h`).

**Domain**: `k2dex.kyletunis.com` (CNAME of `kylehtunis.github.io`). Three things must move together: `web/public/CNAME`, `VITE_BASE_PATH` in deploy workflow, `SITE_URL` in `src/siteMeta.ts`.

**SEO**: `scripts/prerender-routes.ts` generates per-route HTML + sitemap from `src/siteMeta.ts`. Adding a route = adding one entry to `siteMeta.ts`. No static `public/sitemap.xml`.

**PT sampler** runs in a Web Worker (`completer/ptWorker.ts`).

## Code duplicated across Python and TypeScript

**Invariant: no duplication without a gate test.** New duplications must add a parity table row + subtest.

| Python (`k2dex/`) | TypeScript | Gate test |
| :--- | :--- | :--- |
| `sampling.team_energy` | `sampler/energy.ts:teamEnergy` | parity (indirect) |
| `sampling.build_constraint_sets` | `sampler/energy.ts:buildConstraintSets` | parity (indirect) |
| `sampling.swap_violates_uniqueness` | `sampler/energy.ts:swapViolatesUniqueness` | parity (indirect) |
| `sampling.initialize_state` | `sampler/energy.ts:initializeState` | JS smoke + parity (indirect) |
| `sampling.meanfield_marginals` | `sampler/meanfield.ts:meanfieldMarginals` | `test_parity.py::test_meanfield_cases` |
| `sampling.greedy_optimize` | `sampler/greedy.ts:greedyOptimize` | `test_parity.py::test_greedy_cases` |
| `sampling._local_swap_step` | `sampler/swap.ts:localSwapStep` | JS smoke + parity (indirect) |
| `sampling._init_chain` | `sampler/swap.ts:initChain` | JS smoke |
| `sampling.swap_mcmc` | `sampler/swap.ts:swapMcmc` | JS smoke (stochastic) |
| `sampling.parallel_tempered_mcmc` | `sampler/pt.ts:parallelTemperedMcmc` | JS smoke (stochastic) |
| `sampling.rank_single_swaps` | `sampler/rank.ts:rankSingleSwaps` | `test_parity.py::test_rank_cases` |
| `rendering.intra_team_sum_j` | `render/observables.ts:intraTeamSumJ` | `test_parity.py::test_obs_cases` |
| `rendering.pairwise_j_rows` | `render/observables.ts:pairwiseJRows` | `test_parity.py::test_obs_cases` |
| `rendering.nearest_observed` | `render/corpus.ts:nearestObserved` | `test_parity.py::test_corpus_cases` |
| `rendering_html.species_to_slug` | `render/sprite-url.ts:speciesToSlug` | `test_parity.py::test_species_to_slug_cases` |
| `rendering_html.extract_species/item` | `render/format.ts:extractSpecies/Item` | indirect via baseline |
| `loaders.format_pair` | `render/format.ts:formatPair` | indirect (precompute writes vocab) |
| `precompute.pack_lower_triangle` | `sampler/model.ts:unpackLowerTriangle` | JS smoke + round-trip test |
| `precompute.serialize_team_counts` | `sampler/model.ts:loadTeamCounts` | `test_precompute.py` + parity |
| `constants.*` | `web/src/constants.ts` | manual |
| `rendering_html` class names | `render/*.tsx` + `styles/components.css` | visual |
| `styles` LAB_* constants | `styles/{tokens,layout,components,widgets}.css` | visual |

Parity baseline regeneration:

```bash
cd web && npm run emit-baseline   # writes tests/parity_baseline.json
cd .. && python -m unittest tests.test_parity   # compare
```

Synthetic model: V=12, team_size=4. Tolerance `1e-9` for deterministic; stochastic smoke-tested only.

## Data file

`gen9championsvgc2026regma-1760.json` — VGC Reg M-A Champions, 1760 elo cutoff. Do not reintroduce BSS variants.

## External reference

`~/Projects/k2dex-calculator/scripts/update_meta.py` — sibling project's Smogon chaos-JSON downloader/parser.
