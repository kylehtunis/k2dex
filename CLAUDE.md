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
python scripts/precompute.py --build --regulation M-A   # one (regulation, type) model; standard tier
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
k2dex/                  Python package (constants, helpers, loaders, models, sampling, potts, rendering, rendering_html, styles, tournament_ingest)
tournament_json/        In-person tournament raw JSON (committed dir, JSON gitignored)
tournaments_cache/      Unified cache (gitignored)
scripts/                app.py (Streamlit), precompute.py, scotus_precompute.py
notebooks/              Analysis pipeline (numbered chain 1-10 + standalone diagnostics: pl_three_body_check, outcome_ceiling, boltzmann_learning, pl_vs_boltzmann_eval, higher_order_emergence, potts_modulation)
tests/                  unittest + parity_baseline.json
web/                    Static React/TS webapp (Vite + React 18)
  src/sampler/          1:1 port of k2dex/sampling.py
  src/render/           1:1 port of k2dex/rendering.py + rendering_html.py
  src/science/          /science page (toy primitives independent of production sampler)
  src/components/       Shared UI incl. FeatureModal (global feature-detail modal)
  public/models/        Precomputed model artifacts (committed)
```

Promote-when-stable convention: new functions live in notebooks until stable, then move to `k2dex/`.

## Key module notes

Read the code for full APIs. These are the non-obvious parts:

- **`models.py`** — `fit_pl_ising` uses `scipy.optimize.minimize` (L-BFGS-B), not sklearn. Degenerate spins (< 2 units weighted mass) are skipped and fall back to prior or zero. Prior support enables cross-regulation warm-starting. `fit_boltzmann_ising` is the alternative **moment-matching** fit (constrained-MaxEnt / Boltzmann learning): warm-starts from the PL fit, then ascends the regularized log-likelihood with model moments from a persistent batched swap-chain bank (PCD). `empirical_moments(X, w)` returns the weighted 1-pt/2-pt targets. For the species+item ensemble it samples the PCD bank with the **Potts move kernel** (`potts_moves=True`, the default when `species_of` is given): a random-scan mixture of a Metropolized-Gibbs **species swap** (accept on the local free-energy ratio `Z_B/Z_A`, then draw the item from the exact conditional) and a **Gibbs value reroll** (`_potts_species_swap_sweep` / `_potts_item_reroll_sweep` / `_advance_bank_potts`, built on `_build_site_tables` + `_site_conditional`). This treats species selection and item assignment as separate moves; the species-only ensemble keeps the atomic `_batched_swap_sweep` (the q=2 reference the Potts kernel degenerates to). Both target the same constrained Gibbs measure. Within-species couplings are frozen at 0 and dropped from the fit mask (same-species item-states never co-occur, so their coupling is unconstrained). The Potts PCD bank is **factored**: `team_sites` (n_temps, n_chains, team_size) holds site (species) ids and `team_values` (…, n_tracks) the per-track value indices; `_bank_flat_indices(team_sites, team_values, feat_lookup)` rebuilds flat feature indices for energy/moment/replica-exchange code (`feat_lookup` = the `sp_item_feat` site table at n_tracks=1). The Gumbel `choice` a conditional draws IS the item-column value index, so it writes straight into `team_values` with no flat round-trip. This factoring is the multi-track substrate (adding a track = one more `team_values` column + `feat_lookup` axis), currently exercised at n_tracks=1.
- **`sampling.py`** — `swap_mcmc` / `anneal_mcmc` use the atomic `_local_swap_step`, but `parallel_tempered_mcmc` now runs the **universal Potts move kernel** (per-chain `build_site_tables` / `site_conditional` / `_potts_species_swap_step` / `_potts_track_reroll_step`): each sweep is a species-swap, or — when the model has an item track — with probability `p_reroll` (default 0.5) an item reroll. Species-only (no item track) degenerates to the atomic swap. This is the per-chain mirror of `models.py`'s batched training kernel and the 1:1 counterpart of `web/src/sampler/potts.ts`. `build_constraint_sets`/`swap_violates_uniqueness`/`initialize_state` still gate the atomic samplers. **Mirrored 1:1 in `web/src/sampler/`**, gated by `tests/test_parity.py`. `estimate_moments` (free PT generation → 1-pt/2-pt model moments) is **training-only Python — NOT ported to TS, no parity row** (the webapp consumes the resulting `(J,h)`, never the training loop). Same for `fit_boltzmann_ising` and the `_batched_*` / `_potts_*_sweep` / `_advance_bank_potts` batched training helpers in `models.py` (no TS/parity obligation). The two **deterministic** per-chain Potts helpers `sampling.build_site_tables` / `sampling.site_conditional` DO have parity rows (see the duplicated-code table); the stochastic per-chain steps are JS-smoke-tested only.
- **`loaders.py`** — `build_species_model()` / `build_species_item_model()` accept `prior_regulation`/`intercept_prior_weight` for warm-start. `team_weights` computes `w = exp(-dt/tau) * m^[in-person]`, normalized to mean 1. `build_species_item_model` normalizes a missing item (Python `None`) to the string `"None"` when forming `(species, item)` pairs, so an itemless build is an ordinary Potts state (subject to item uniqueness) and the corpus's split duplicates (`Talonflame` / `Talonflame @ None`) merge into one feature. The return signature is unchanged; the factored-schema derivation happens in precompute only.
- **Artifact schema v3 (factored `sites` + `tracks`)** — `meta.json` stores `sites: string[]` (distinct species, vocab order), `site_of: number[]` (per-feature site index), `tracks: {name, unique}[]` (attribute definitions; `[]` for species-only, `[{name:"item",unique:true}]` for species+item), and `track_values: (string|null)[][]` (per-feature per-track values). `species_of`/`item_of` are **removed** from the schema and reconstructed as convenience arrays by both loaders (`speciesOf[i]=sites[site_of[i]]`, `itemOf[i]=track_values[i][0] ?? null`). `site_features` (per-site feature grouping) is **not stored** — it is a pure projection of `site_of` both loaders derive at load. `vocab` stays (display string; drift is visible). `meta.json` also stores `type` (product tier, e.g. `"standard"`). `manifest.json` is **schema_version 3**: a `(regulation, type)` grid, one entry per model, each tagged `regulation` + `type`, carrying per-model `tracks` and derived provenance (`V`, `n_corpus_teams`, `latest_tournament_date`). `description` lives once per type in a shared top-level `types` block (not per model); `manifest.ts` resolves it back onto each `ModelSummary`. `potts.py:load_fitted_model` and `web/src/sampler/model.ts:loadModel` both read v3 and rebuild the `(species, item)` view. `SUPPORTED_SCHEMA_VERSIONS = [3]` (no back-compat; `--recompute` rebuilds all artifacts).
- **`potts.py`** — **post-fit Potts analysis only** (no fitting/sampler change); the Potts reframe R2. Treats the fitted species+item `J` as a Potts coupling tensor: site = species, state = {absent, item…}, so the cross-species block `J[items_of_P, items_of_Q]` is that pair's Potts block relative to the absent (all-zero) reference. `load_fitted_model(model_dir)` reads a precompute artifact back into a `FittedModel`. `decompose_block` is the 2-way ANOVA / zero-sum-gauge split (species-level **synergy** = grand mean + zero-sum **item-modulation** residual). `modulation_scores` is the item-modulation table — reports **three** magnitudes (`mod_frob` raw Frobenius, `mod_rms` alphabet-normalized, `mod_frac` scale-free share) because they disagree and each keeps one confound: `mod_frob` ≈ pure popularity proxy (~0.9 corr), so "read against support" (`n_items`/`appearances`) is mandatory, not optional. `species_apc_graph` gives the plmDCA-style APC-corrected species graph (`corrected = frob − apc`) plus the signed `synergy` matrix. Gated by `tests/test_potts.py`; explored in `notebooks/potts_modulation.ipynb`.
- **`tournament_ingest.py`** — Cache schema versioned (`CACHE_VERSION`); bump when parsing changes. Limitless dates are ISO timestamps, in-person dates bare `YYYY-MM-DD`; consumers must parse at day resolution.

## Conventions that span multiple files

- **`CURRENT_REGULATION`** (`constants.py` + `constants.ts`) — determines active regulation for webapp model picker; default for `--regulation` in precompute and loaders. Update when a new regulation begins.
- **Regulation relabeling** — `tournament_ingest._REGULATION_RELABELS` promotes mis-labeled Limitless events using marker species/items. Current rule covers **M-B** (start `2026-06-17`, from `{M-A, CUSTOM}`, markers Gholdengo/Metagross/Grimmsnarl + Life Orb/Light Clay). `_ILLEGAL_ITEMS_BY_REGULATION[target]` governs item-stripping for promoted events. Add marker rules when starting a new regulation Limitless hasn't tagged; remove once tagged.
- **Cross-regulation warm-start** — `prior_regulation` (loaders) / `--prior-regulation` (precompute). Assumes prior is a legality superset. Donor vocab folded in unconditionally; `_align_prior` scatters `(J, h)` onto new vocab. Features with no new data keep prior exactly; features with data relax off it. `intercept_prior_weight` (default 1.0) controls bias anchoring.
- **Regularization** — `SPECIES_LR_LAMBDA = 25.0`; `SPECIES_ITEM_LR_LAMBDA = 4.5`. Lambda is canonical L2 penalty, converted to `C = 1/lambda` internally. `meta.json` records `fit.lambda`.
- **Boltzmann fit defaults** — the `BOLTZMANN_*` constants in `constants.py` are the single source of truth for the moment-matching build recipe. Both `fit_boltzmann_ising`'s signature defaults AND precompute's `--bz-*` argparse defaults read from them, so precompute builds, notebooks, and tests that don't override a knob fit identically — tune here, not at call sites. `lr_final` defaults to `BOLTZMANN_LR_FINAL` (cosine decay **on** by default; `lr_final == lr` disables it). The heavy `fit_boltzmann_ising`-based tests in `test_boltzmann.py` / `test_potts_sampler.py` deliberately **pin lean values** (small budgets, `n_temps=2`, explicit `lr_final`) so production tuning of these constants can't perturb them; the rigorous convergence work lives in the notebooks.
- **Boltzmann learning** (`fit_boltzmann_ising`, `notebooks/boltzmann_learning.ipynb`, downstream eval in `notebooks/pl_vs_boltzmann_eval.ipynb`) — the moment-matching alternative to PL, motivated by the `pl_three_body_check` finding (the renamed, historical `three_body_check`) that PL matches conditionals not moments. On the M-A corpus it cuts the worst-feature marginal error from ~0.34 (PL over-concentration) to ~0.05. Gotchas: (1) **the model is sampled at T=1** — `estimate_moments`' cold chain (ladder index 0) MUST be at T=1 to sample `P ∝ exp(-H)`; a cold rung below 1 samples a sharper distribution (note: `pl_three_body_check`'s ladder starts at 0.5, a latent bug there — the new notebook fixes it). (2) **Reg fixes the gauge** — on the fixed-size manifold `(J,h)` is gauge-degenerate (`h→h+c·1`, `J_ij→J_ij+a_i+a_j`); the non-gauge-invariant L1/L2 penalty toward zero picks the min-norm representative, so keep `reg_lambda>0` (small for tightest moment match — larger deliberately shrinks `(J,h)`). (3) **The dominant fit error is the stochastic-gradient noise ball, killed by `lr_final`/`avg_last`** — a fixed lr orbits the optimum rather than settling, so a single final iterate is displaced most in the strong-coupling directions; the moment-reconstruction scatter then fans into a *cone* (worst-feature pair bias ~0.04, marginal ~0.11). Cosine step-decay (`lr_final≈lr/20`) **or** Polyak iterate averaging (`avg_last`) cuts that **~7×** to ~0.006, for free (same budget) — lr decay is on by default via `BOLTZMANN_LR_FINAL` (both `fit_boltzmann_ising` and precompute). This was diagnosed *not* to be regularization (a 100× λ sweep didn't move it) or mixing (more sweeps/tempering didn't fix it; 2× `n_chains` helped only via lower gradient noise). (3b) **Read `mean_resid_*`, not `max_resid_*`** in the live history — the per-iteration max is a single `n_chains`-sample snapshot, mostly Monte-Carlo noise; confirm true convergence by re-estimating the fitted model with many samples via `estimate_moments`. The PCD bank is batched across chains in pure NumPy with sparse on-bit moment accumulation (cost independent of V). Parallel tempering (`n_temps>1`) is implemented and its default lives in `BOLTZMANN_N_TEMPS` — single-T PCD sufficed on the earlier M-A corpus (not mode-trapped), so it's mainly for genuinely multimodal models. (4) **Species ensemble passes `species_of=None`** (species distinct per vocab entry → no uniqueness needed); species+item passes the real lookups + a co-occurrence `support_mask`. (5) **Species+item is sampled with the Potts move kernel by default** (`potts_moves=True`; see the `models.py` note) — separate species-swap / item-reroll moves accepting on `Z_B/Z_A`, gated by `tests/test_potts_sampler.py` (exact-moment match vs brute-force enumeration; q=2 degeneracy `log Z_B − log Z_A == −ΔH`; constraint invariants). ~40% slower per sweep than atomic swaps, matches moments identically. Pass `potts_moves=False` to force the old atomic sampler for A/B comparison. Same-species `J` is frozen at 0. This changes a species+item Boltzmann artifact's fitted numbers vs a pre-Potts atomic-swap fit (PL production path is unaffected).
- **Sample weighting** — `RECENCY_TAU_DAYS` / `IN_PERSON_WEIGHT` in `constants.py`. Vocab cutoff requires `min_team_count` in **both** raw and weighted counts.
- **Energy formulas** — raw `H(s) = -h*s - 0.5 s'Js`. Adjusted uses `field_weight*h`. At `field_weight=0` only pairwise term remains.
- **Swap-move MH** — energy diff `dH = h_eff[out] - h_eff[in] + (J[out] - J[in]) * s + J[in, out]`. Last term corrects for swapped pair's mutual coupling.
- **Bias Adjustment defaults** — Web page-load: `0.5`. Greedy on: `0.8`, greedy off: `0.3`. Streamlit: `0.3`. Values have drifted; `energy_discrimination.ipynb` is the harness.
- **Shareable URLs** — `render/shareLink.ts` encodes by slug (not index). Token format: `<modelSlug>.<fwIndex>.<mons>`; `<modelSlug>` is now a per-regulation slug (`reg-m-a`). Each mon is `speciesSlug~itemSlug` (feature pin, itemless = `~none`) or bare `speciesSlug` (site pin). Feature mons come first in ascending index order (pre-site-pin tokens stay byte-identical); site mons are appended. `LEGACY_CODE_TO_SLUG` remaps `s`/`si`/`reg-m-a-species`/`reg-m-a-species-item`/`-weighted`/the M-B olds → the unified per-regulation slug.
- **Completer roster + attribute toggle** (`/completer`, webapp-only — no Streamlit/Python mirror, no parity rows) — the completer state is an ordered `roster: RosterSlot[]` (`{site, feature}`); `components/RosterEditor.tsx` is the factored 6-slot editor (per slot: a species picker + an *optional*, clearable item picker). A slot with a chosen item is a **feature pin**, a slot with the item left unset is a **site pin** (species locked, item filled by the completer), an empty slot is fully free. `fixedIdxs`/`fixedSites` are **derived** from `roster` in `CompleterPage` for the sampler + share links (there is no "any item" option — unspecified is the default). `RosterEditor` is parametrized (`itemPlaceholder`, `emptyHint`) and **shared with `/analysis`**, which stores its own `roster: RosterSlot[]` (mirror of the completer) and derives the feature-level `teamIdxs` from the item-pinned slots (`emptyHint={null}`, no site pins — an unset item just means an incomplete team). In PT, `sampling.energy:resolveSitePins` seeds each site pin to its best placeable feature at the front of `onNf`, and `PottsContext.lockedSlots` (slot indices `0..L-1`) bars species-swaps on those slots while `pottsTrackReroll` still rerolls their item (`sampler/swap.ts:initChain` gained a `preOn` param); the greedy fast path resolves a site pin to that feature up front (no reroll). **Completion results** render as stacked cards (`completer/CompletionCard.tsx`: `CompletionCard` + `CompletionList`) showing the **full six-mon team** (pinned + filled; pinned tiles marked) in a 3-col/2-col grid with observables + actions underneath — not just the completed members. The **attribute toggle** lives in an "Advanced options" expander as an **"Excluded attributes"** section (all attributes active by default; checking one *excludes* it): `sampler/model.ts:withInactiveTracks` clears the track's `unique` flag (degenerate — no uniqueness), `pReroll=0` (no reroll), the worker aggregates completions by species set (`ptWorker.ts:aggregateBySite`, representative = most-frequent real feature-team) and the UI hides the item (`CompletionCard hideItems`). The species-swap's `Z_B/Z_A` still marginalizes the degenerate track, so the species-set distribution is exact. `/analysis` and `/meta` stay feature-level regardless of the toggle. **User-facing copy says "Pokémon", not "Species"** (the species-vs-species+item model distinction is gone).
- **Itemless "None" item** — a Pokémon with no held item (mainly Talonflame, for Acrobatics) is an **ordinary item value** `"None"` in the vocab, with **zero special handling** anywhere (sampler, share-links `~none`, roster editor item picker). It appears only in the item picker of species that actually carry it. The corpus's un-collapsed itemless spellings (`None`/`No item`/`Nothing`) inflate this in sparse regulations (reg-m-b) — a **deferred ingest-normalization** issue, not special-cased in the app.
- **Validation split** — chronological 90/10 (`VALIDATION_TEAM_FRAC_TEST = 0.10`). Vocab from train only; OOV test teams dropped.
- **Vocab normalization** — ingest collapses case/whitespace and `Mega <X>` to base species. Bump `CACHE_VERSION` if normalizers change.

## Webapp visual language

**Brand accent: `--lab-accent: oklch(0.42 0.10 145)`** in `tokens.css` / `LAB_ACCENT` in `styles.py`. Use `var(--lab-accent)` in new CSS. Key gotchas:

- **Streamlit strips `<script>` and inline handlers.** `sprite_img` uses `<object>` with inner `<span>` fallback. React `Sprite.tsx` uses `<img onError>`.
- **Sprite slug rules** (`species_to_slug`): first hyphen preserved as forme separator; subsequent collapse. `_HYPHEN_BASE_SPECIES` has all hyphens stripped. Missing -> `missingno.svg`.
- **Widget-scoping** via marker divs + `:has()` in `styles.py`. Breaks if Streamlit changes `data-testid` hierarchy.
- **Responsive**: desktop-first, breakpoints `900px` (narrow) and `640px` (phone) + `pointer: coarse`. **Never rename existing `lab-*` classes** (mirrored in `rendering_html.py`).
- **`/science` controls** use `.lab-science-btn` from `widgets.css`. No bare `<button>` elements.
- **Feature detail modal** (webapp-only, no Streamlit/Python counterpart, no parity row): `components/FeatureModal.tsx` (provider + body) + `components/FeatureModalContext.ts` (`useFeatureModal` hook, split out to avoid a cycle with `render/cells.tsx`) + generic `components/Modal.tsx` (portal/backdrop/focus-trap). `openFeature(name)` resolves a vocab string via `model.indexOf`. The shared cells `InlineMon`/`CompMonCell`/`SlotCard` are clickable wherever a provider is mounted; `TeamMiniStrip` is inert unless passed `interactive` (the `/meta` §03 TopTeamsTable and the modal's own corpus-appearance rows opt in; inside the panel those clicks drill via the nested context). Per-feature data: `render/featureDetail.ts` (`featureCouplings`/`featureCorpusAppearances`/`featureRanks`), reusing `meta/couplings.ts:isStructuralPair` and `render/corpus.ts:parseTeamKey` (the now-canonical roster-key parser, also used by `meta/topTeams.ts`). Modal actions are route-gated (completer pin/exclude, analysis add/swap, meta none). New `lab-feature-*` / `lab-modal*` / `lab-feature-dock` classes join the "never rename" set. `SwapsTable` takes an optional `onAcceptSwap(out,in)` to apply a swap to the analysis team. **Two presentation variants** (`Modal.tsx` `variant` prop, chosen by `useMediaQuery(DOCK_QUERY=min-width:1200px)`): centered `modal` (backdrop + scroll-lock + focus-trap) on narrow screens, and a non-blocking right-hand `dock` inspector on wide screens (page stays interactive; body gets `feature-docked` → `padding-right` reserves the gutter and the margin:auto `.lab-container` slides left). The panel re-provides a nested `FeatureModalContext` whose `openFeature` *pushes* (drill-through + Back), while the outer page context *resets* the stack (fresh select) — same `InlineMon`, behavior decided by render location.

## /science page gotchas

- **Toy primitives** (`science/primitives/`) are independent of production sampler. No parity obligation.
- **PT swap-acceptance sign**: correct exponent is `(beta_cold - beta_hot)(H_cold - H_hot)`. Wrong sign appears to work due to two-well symmetry.
- **Two spin conventions**: Magnets/Lattice use +/-1; Graph and downstream use {0, 1}.
- **SCOTUS section**: conservative = red, liberal = blue. Vote data `1 = conservative`. Graph-edge colors are coupling sign (positive = blue, negative = red), unrelated.
- **Pokemon section** uses active model from `useModel()`. Prose counts are live, not hardcoded.
- **Citations**: `References.tsx` is the single source of truth.

## Build and deploy

**Build artifacts pipeline**: ingest -> precompute per model -> generate manifest -> inspect -> commit. CI does **not** regenerate artifacts. Production identity is a `(regulation, type)` grid: one artifact per cell, slug `reg-<regulation>` for the standard tier. `type` is a **product-tier build recipe** (closed enum, today only `standard`), applied uniformly across regulations — distinct from precompute's **retired** `--type species|species_item` (feature-dimensions) meaning, which the factored schema + attribute toggle made obsolete. `PRODUCT_TYPES` / `TYPE_DESCRIPTIONS` / `TYPE_BUILDER` / `TYPE_METHOD` in `precompute.py` define each type; `model_slug(regulation, type)` derives the id. Build one model with `--build --regulation <R>` (standard implied); `--recompute` rebuilds every committed model from its stored `meta.fit` block. The `standard` recipe is a **Boltzmann** species+item fit with the `BOLTZMANN_*` constants (incl. `BOLTZMANN_SUPPORT_MIN_COUNT`) — moment-matching is what makes the attribute toggle's on-the-fly track marginalization faithful. **`field_weight` caveat**: moment-matched `h` wants `field_weight ≈ 1`, unlike PL's down-weighted `h`; the webapp's PL-tuned `field_weight` defaults are still in place and slated for re-derivation (harness: `energy_discrimination.ipynb`) — possibly removed entirely — once the Boltzmann standard models are the default.

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
| `sampling.build_site_tables` | `sampler/potts.ts:buildSiteTables` | `test_parity.py::test_site_tables_cases` |
| `sampling.site_conditional` | `sampler/potts.ts:siteConditional` | `test_parity.py::test_site_conditional_cases` |
| `sampling._potts_species_swap_step` | `sampler/potts.ts:pottsSpeciesSwap` | JS smoke (stochastic) |
| `sampling._potts_track_reroll_step` | `sampler/potts.ts:pottsTrackReroll` | JS smoke (stochastic) |
| `sampling.parallel_tempered_mcmc` | `sampler/pt.ts:parallelTemperedMcmc` | JS smoke (stochastic; Potts inner move) |
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
