# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code applying complexity-science / distributional-semantics methods to competitive Pokémon (VGC) team-composition data. **Read `vgc_complexity_phase1_plan.md` first** — it's the authoritative source on Phase 1 methodology, scope, and rationale.

Three modeling phases, all using the inverse Ising pairwise-MaxEnt model with `(J, h)`:

- **Phase 1** — Gaussian / precision-matrix approximation on Smogon chaos JSON (marginal usage + pairwise teammate co-occurrence). Fast to fit but systematically under-magnitudes strong negative couplings.
- **Phase 2** — pseudo-likelihood (PL) fit on Limitless tournament team rosters, species-only vocab.
- **Phase 3** — PL fit on `(species, item)` pair features from the same Limitless data. Restores held-item forme distinctions Phase 2 collapses.

The repo is a sequence of notebooks plus **two coexisting webapp surfaces**: the original Streamlit app (`app.py`) and a static React/TypeScript port (`web/`) intended for future GitHub Pages deploy. Both surface only Phase 2 (Species) and Phase 3 (Species @ Item) — Phase 1 lives on in the notebooks but was cut from the user-facing surface in the v1 simplification. Both have **three pages** under one model picker: `/completer` (suggested completions or full PT sampling), `/analysis` (per-team diagnostics), `/meta` (format-wide statistics). User-facing terminology is decoupled from the underlying math: `J` → "Coupling Strength", `h` → "Bias", `ΣJ` → "Coherence", `E` → "Score" (sign-flipped so higher = better), `field_weight` → "Bias Adjustment". The webapp is one deliverable of the larger project — not the single end goal; deeper scientific content lives in the notebooks and a future explanatory page.

Until the static webapp ships, `app.py` is the canonical reference implementation. The TS port mirrors every piece of math; any drift is caught by `tests/test_parity.py` (see "Code duplicated across Python and TypeScript" below).

## Run commands

```bash
# Python side
streamlit run app.py            # canonical Streamlit webapp
jupyter lab                     # notebooks
python -m unittest discover tests   # regression gate (includes parity tests)
python -m limitless_ingest       # ingest tournaments (manual; cache is in tournaments_cache/)
python precompute.py            # rebuild web/public/models/ artifacts (manual; inspect before commit)
python scotus_precompute.py     # rebuild web/public/scotus/ artifacts for the /science SCOTUS section (manual)

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
app.py                  Streamlit webapp (rendering only; samplers imported)
constants.py            Shared numeric constants (vocab cutoffs, seeds, etc.)
helpers.py              Phase 1 Gaussian inverse Ising primitives
loaders.py              Pure (J, h, m, vocab) model builders; used by app.py
                        via @st.cache_resource wrappers and by precompute.py
                        for direct invocation
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
precompute.py           Offline pipeline: fits both models via loaders.py and
                        serializes them to web/public/models/{species,species_item}/
                        for the static webapp to load at runtime
scotus_votes.txt        895 second-Rehnquist-court votes (9 bits per row,
                        ~44% unanimous; column order matches
                        scotus_precompute.JUSTICES)
scotus_precompute.py    Fits PL inverse Ising on scotus_votes.txt at several
                        N checkpoints; writes web/public/scotus/{votes,fits}.json
                        for the /science page's SCOTUS section
tests/                  unittest tests + locked validation baselines
                        + parity_baseline.json (JS-side outputs gated by
                          tests/test_parity.py)
web/                    Static React/TypeScript webapp — see "Static web build"
                        below for layout details
```

The "promote when stable" convention still applies: new functions live in their notebook until they stabilize, then move to the relevant module. Phase 1 primitives are in `helpers.py`; the PL fit is in `models.py`; the MCMC family is in `sampling.py`. Each module's docstring lists what's already promoted.

## Pipeline order

The notebooks form a dependency chain. Run in order, or re-derive intermediate state inside any standalone notebook.

1. **`spatial_embedding.ipynb`** — Stages 0–4 from the plan. PPMI → truncated SVD → role-residual NN ranking → NMF role decomposition. Uses `helpers.py` primitives.
2. **`inverse_ising.ipynb`** — fit `(J, h)` via the Gaussian / precision-matrix approximation; visualize J. Uses `helpers.ising_gaussian`.
3. **`j_communities.ipynb`** — Louvain community detection on the `+J` subgraph (archetype clusters) and `-J` subgraph (role pools).
4. **`forward_ising.ipynb`** — forward sampling from the fitted model: swap-move MCMC with the `team_size=6` constraint, calibration against empirical marginals. Uses `sampling.swap_mcmc`.
5. **`inverse_ising_phase2.ipynb`** — Phase 2 PL fit; mirrors notebook 2's analysis cells. Uses `models.fit_pl_ising`.
6. **`validation.ipynb`** — leave-k-out evaluation harness: cross-model comparison (team-level 90/10 split), pair-prediction metric (Phase 3), temporal-drift study (chronological). Uses `models.fit_pl_ising`.

## Module structure

- **`constants.py`** — vocab cutoffs, corpus size, regularization strength, validation seed / fraction, ingest filters. Change any knob here, not in the notebook or app. (Static webapp mirrors a subset of these in `web/src/constants.ts` — see duplication register below.)
- **`helpers.py`** — Phase 1 only: `load_chaos`, `build_vocab`, `build_cooccurrence`, `build_ppmi`, `binary_moments`, `binary_correlation`, `ising_gaussian`. Frozen-dataclass return types throughout.
- **`loaders.py`** — pure `build_species_model()` / `build_species_item_model()` that fit `(J, h, m, vocab, team_counts, species_of, item_of)` end-to-end. No Streamlit. `app.py` wraps them in `@st.cache_resource`; `precompute.py` calls them directly. Also exports `format_pair(species, item)`.
- **`models.py`** — `fit_pl_ising(X, *, C, max_iter)` runs V per-spin L2 logistic regressions, skips degenerate spins, symmetrizes via `J = 0.5 * (J_asym + J_asym.T)`, zeros diagonal. Used by both notebooks 5/6 and `loaders.py` (which in turn feeds `app.py:load_model_{species,species_item}` and `precompute.py`).
- **`precompute.py`** — CLI that fits both models via `loaders.py` and serializes each as `meta.json` + `J.bin` (float32 lower triangle) + `h.bin` + `m.bin` + `team_counts.json` under `web/public/models/<name>/`. Lower-triangle packing halves J's bytes; loader on the JS side reconstructs symmetric J. `--emit-parity-baseline` is unused (parity baseline emitted from JS — see `web/scripts/emit-parity-baseline.ts`); `--skip-team-counts` is a dev-iteration shortcut. The artifacts are committed to git after manual inspection — CI does **not** regenerate them.
- **`sampling.py`** — `swap_mcmc` / `anneal_mcmc` / `parallel_tempered_mcmc` / `meanfield_marginals` / `greedy_optimize` / `rank_single_swaps`. All accept keyword-only `species_of` / `item_of` for Phase 3 uniqueness (no-duplicate-species, no-duplicate-item). Shared inner loop in `_local_swap_step` so the three MCMC variants stay in sync. **Mirrored 1:1 in `web/src/sampler/`** — every public symbol has a TypeScript counterpart gated by `tests/test_parity.py`.
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
- **Math typesetting (static webapp only)**: `/science` uses KaTeX via `react-katex`. The CSS bundle is lazy-imported once in `SciencePage` (`import("katex/dist/katex.min.css")`) so the ~45 KB stylesheet only loads when a user lands on that route. `web/src/science/widgets/Math.tsx` wraps `<InlineMath>` / `<BlockMath>` with the project's typography. Streamlit-side equations are still rendered via `st.latex` (server-side MathJax); no shared dependency.

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

## Static web build

The `web/` tree is a self-contained Vite + React 18 + TypeScript app that ports the Streamlit UI to a static site for GitHub Pages deployment. Its sampler math is a faithful port of `sampling.py`; rendering helpers mirror `rendering.py` + `rendering_html.py`; styles port `styles.py` to plain CSS.

```
web/
  index.html
  package.json              Vite + React 18 + TS + Vitest + react-select + react-router-dom
  tsconfig.json
  vite.config.ts            base path from VITE_BASE_PATH (production: /k2dex-science/)
  scripts/
    emit-parity-baseline.ts  Node script that emits tests/parity_baseline.json
  public/
    assets/missingno.svg     also copied into web/src/assets/ for ?url import
    models/{species,species_item}/  ← precompute.py output, committed to git
    scotus/{votes,fits}.json ← scotus_precompute.py output, committed to git
  src/
    main.tsx, App.tsx        bootstrap + router (includes /science route)
    constants.ts             mirrors a subset of constants.py
    assets/                  Vite-handled imports (missingno fallback)
    state/ModelContext.tsx   load + cache (J, h, m, team_counts) per model
    components/              Layout, ModelPicker, VocabSelect (react-select wrapper)
    pages/{Completer,Analysis,Meta,Science}Page.tsx
    sampler/                 1:1 port of sampling.py — see duplication register
      types.ts, rng.ts, model.ts, energy.ts,
      meanfield.ts, greedy.ts, swap.ts, pt.ts, rank.ts
    render/                  1:1 port of rendering.py + rendering_html.py
      format.ts, corpus.ts, sprite-url.ts, Sprite.tsx, atoms.tsx,
      cells.tsx, observables.ts
    completer/               Page-local logic: fastPath, ptWorker, ptDriver,
                             CompletionRow
    analysis/                Page-local tables: PairwiseJTable, SwapsTable, ChainTable
    meta/                    Page-local tables: FeatureBiasTable,
                             ExtremeCouplingsTable, couplings
    science/                 /science page — pedagogical Ising explainer
      primitives/            Toy implementations: lattice, graph (+ ER + spring
                             layout), mcmc, mf, pt, landscape (synthetic 2D
                             energy surface with Metropolis + gradient descent)
      widgets/               Math (KaTeX), LinePlot, SpinGrid, GraphView (now
                             sprite-capable), ChainStrip, Landscape3D (SVG
                             isometric mesh + walker overlays)
      sections/              Magnets, Lattice, Graph, Metropolis,
                             ParallelTempering (with energy-landscape
                             intermission), MeanField, SCOTUS, Pokemon,
                             References (citations + data/asset acks footer).
                             Page order is the source of section identity —
                             files have descriptive names, not S1/S2
                             prefixes. SciencePage groups them under three
                             act headers (h2.lab-science-act): "The model"
                             (Magnets/Lattice/Graph), "Sampling" (Metropolis/
                             ParallelTempering/MeanField), "The inverse
                             problem" (SCOTUS/Pokemon); References renders
                             last under its own act header. Section titles are
                             h3; in-section subheads are h4.lab-science-subhead.
      data/scotusLayout.ts   Fixed 2D positions for the 9 justices
      __tests__/             vitest smoke tests for the toy primitives
    styles/                  tokens.css, layout.css, components.css, widgets.css
  __tests__/                 vitest smoke tests for stochastic samplers
```

**Build artifacts pipeline:**
1. Run `python precompute.py` locally after any change to corpus / fit hyperparams.
2. Inspect `web/public/models/{species,species_item}/{meta.json,J.bin,...}` for sanity (vocab size, n_corpus_teams, file sizes).
3. Commit the artifacts. CI is not allowed to regenerate them (the user wants manual eyeballing of every refresh).
4. `npm run build` reads the artifacts as static files (Vite copies `public/` into `dist/`).

**Parallel-tempered sampler** runs inside a Web Worker (`web/src/completer/ptWorker.ts`) so the UI stays responsive during the 10–30s sample. Structured-clone passes the model (~2.5 MB J for Species @ Item) into the worker; not a performance issue at this scale.

**Sprite rendering** uses HTML5 `<object>` with a missingno fallback child — the same approach Streamlit-side took, ported as a React component (`web/src/render/Sprite.tsx`). The Showdown CDN URL builder + slug rules live in `web/src/render/sprite-url.ts` and mirror `rendering_html.species_to_slug`.

**GH Pages deploy** is postponed — `STATIC_DEPLOY_PLAN.md` has the Phase E workflow draft when we're ready. Until then, `web/dist/` is local-only.

## /science page

An interactive explainer for the math behind the project. Lives under `web/src/science/`; rendered at `/science`. The narrative arc is **Magnets → Lattice → Graph → Metropolis → Parallel Tempering (with energy-landscape intermission) → MeanField → SCOTUS → Pokemon**, each section a self-contained interactive widget plus prose. `SciencePage` groups the sections under three act headers — **The model** (Magnets/Lattice/Graph), **Sampling** (Metropolis/ParallelTempering/MeanField), **The inverse problem** (SCOTUS/Pokemon). The act headers are `h2.lab-science-act`; section titles are `h3`; in-section subheads are `h4.lab-science-subhead`. (Section *files* `Metropolis.tsx` / `ParallelTempering.tsx` were formerly `MCMC.tsx` / `PT.tsx`; the toy primitives `primitives/mcmc.ts` / `primitives/pt.ts` kept their abbreviated names.)

Key design decisions and non-obvious facts:

- **Toy primitives are independent of the production sampler.** `web/src/science/primitives/{lattice,graph,mcmc,mf,pt,landscape}.ts` are deliberately small re-implementations. They never import from `web/src/sampler/`, and they cover only the pedagogical case (single-spin flips on small spin systems, no team-size constraint). The production swap-move sampler is mentioned in the Pokemon section but not reused.
- **PT swap-acceptance sign** (subtle): the correct exponent under detailed balance is `(β_cold − β_hot)(H_cold − H_hot)`. The toy `primitives/pt.ts` and the section's inline loop both had the inverted form during initial development; that was caught during the doc-review pass and fixed. `sampling.py:parallel_tempered_mcmc` was always correct. If you touch either of these files, double-check the sign — the simulation appears to "work" visually under the wrong sign because of two-well symmetry.
- **Two spin conventions on one page.** The Magnets and Lattice sections use ±1 (physical magnet convention). Graph and everything downstream use {0, 1} (the convention `models.fit_pl_ising` and the Pokemon completer use). The Graph section's prose explicitly calls out the switch. The math is equivalent up to relabeling, but the energy formulas look different — `H = -Σ s_i s_j` vs. `H = -Σ h_i s_i - Σ_{i<j} J_ij s_i s_j` on the {0,1} side.
- **Synthetic 2D energy landscape** (`primitives/landscape.ts`): a two-well surface, `E(x, y) = -3·exp(-((x±1)² + y²)/0.6) + 0.25·(x² + y²)`, rendered as an SVG isometric mesh by `widgets/Landscape3D.tsx`. Drives both the PT section (animated Metropolis walkers) and the MeanField section (deterministic gradient descent via the analytic `gradAt`). This is a caricature — not a projection of any actual Ising graph — but the basin-trapping/escape story it tells is the load-bearing intuition for both methods.
- **Animation rates.** All animated sections use `requestAnimationFrame` with a `STEPS_PER_FRAME` constant; vsync (~60fps) caps the rate. PT runs 6 steps/frame, MF 1 step/frame, the Ising sections (Lattice/Graph/Metropolis) run one sweep batch every 100 ms. The Ising sections also expose a manual Step button. Tune the per-section constants if the page feels too fast/slow.
- **Random RNG seeds by default.** This is a playground page, not a determinism showcase — every reset / "New graph" pull from `Math.random()`. The toy `randomGraph()` still accepts a seed for tests and the Graph section's "New graph" button.
- **SCOTUS section** is a mini team-completer: 3-state pin chips (unset → conservative → liberal → unset), top-N configurations panel by energy with corpus observation counts, and per-justice conditional marginal bars. Exact enumeration over 2^(9 − |pinned|) ≤ 512 configurations — no MF or PT needed at V=9. The color convention is **conservative = red, liberal = blue** (US political-map standard) across all components (pin chips, slot tiles, marginal bars, prose swatches). Note the vote data itself encodes `1 = conservative, 0 = liberal`; only the *display* colors follow the red/blue political standard. The graph-edge colors are unrelated — those encode coupling *sign* (positive = blue, negative = red) via the shared `GraphView`. `JUSTICE_SPRITES: (string | null)[]` is the drop-in slot for justice sprite URLs; null falls back to the two-letter abbreviation.
- **`GraphView` is sprite-capable.** Optional `sprite?: string` on each `GraphNode`; when set, renders `<image>` (with `onError` → bundled `missingno.svg` fallback in `widgets/GraphView.tsx`) instead of the default circle. Also takes `showLabels` (default true) and `spriteOpacity` (default 1) — the Pokemon figure sets `showLabels={false}` + `spriteOpacity={0.8}` to declutter. The Pokemon section uses this with `spriteUrl()` from `render/sprite-url.ts`; SCOTUS will once justice sprites land.
- **Pokemon section figure.** A force-directed graph of the top ~32 species (one representative `(species, item)` node each — the highest-marginal build), drawn from the live Phase 3 `J` via `ModelContext`. A `|J|` threshold slider hides weaker couplings and **recomputes the spring layout** (`primitives/graph.ts:springLayout`) over only the still-connected nodes; a small `relaxOverlaps` pass declumps sprites afterward. No sampling/animation — pure layout. The prose's spin/parameter/team counts are read live from `model.V` / `model.nCorpusTeams`, not hardcoded.
- **No code duplication with Python.** The page reuses the live model artifacts (`model.J`, `model.h`, `model.speciesOf`) via the existing `ModelContext`, but every section either uses toy primitives or operates directly on the loaded model. Nothing in `web/src/science/` needs to mirror a Python counterpart, so no parity-test obligation is added to `tests/test_parity.py`.
- **Citations.** `sections/References.tsx` is the single source of truth for the page's bibliography (Ising 1925, Onsager 1944, Metropolis et al. 1953, Hukushima & Nemoto 1996, Besag 1975, Schneidman et al. 2006, Lee/Broedersz/Bialek 2015) plus data/asset acknowledgments (Limitless VGC, Pokémon Showdown). Each entry has an `id` (`ref-onsager`, `ref-lee`, …); inline `(Author, year)` citations in the section prose are plain `<a href="#ref-…">` anchor links. Add a new method's citation here and link to it inline where the method is introduced — don't scatter free-floating references.

## Code duplicated across Python and TypeScript

Every duplicated symbol below must stay in sync. Editing one side without the other is allowed only if a parity test fails first and the fix updates both sides plus the gate.

The invariant: **no Python↔TypeScript code duplication is allowed without a gate test.** New duplications added in future work must add a row to this table and a corresponding subtest in `tests/test_parity.py`.

| Python | TypeScript | Gate test |
| :--- | :--- | :--- |
| `sampling.team_energy` | `sampler/energy.ts:teamEnergy` | parity (indirect, via MF/greedy) |
| `sampling.build_constraint_sets` | `sampler/energy.ts:buildConstraintSets` | parity (indirect) |
| `sampling.swap_violates_uniqueness` | `sampler/energy.ts:swapViolatesUniqueness` | parity (indirect) |
| `sampling.initialize_state` | `sampler/energy.ts:initializeState` | JS smoke + parity (indirect) |
| `sampling.meanfield_marginals` | `sampler/meanfield.ts:meanfieldMarginals` | `test_parity.py::test_meanfield_cases` |
| `sampling.greedy_optimize` | `sampler/greedy.ts:greedyOptimize` | `test_parity.py::test_greedy_cases` |
| `sampling._local_swap_step` | `sampler/swap.ts:localSwapStep` | JS smoke + parity (indirect) |
| `sampling._init_chain` | `sampler/swap.ts:initChain` | JS smoke |
| `sampling.swap_mcmc` | `sampler/swap.ts:swapMcmc` | JS smoke (stochastic — no parity baseline) |
| `sampling.parallel_tempered_mcmc` | `sampler/pt.ts:parallelTemperedMcmc` | JS smoke (stochastic) |
| `sampling.rank_single_swaps` | `sampler/rank.ts:rankSingleSwaps` | `test_parity.py::test_rank_cases` |
| `rendering.intra_team_sum_j` | `render/observables.ts:intraTeamSumJ` | `test_parity.py::test_obs_cases` |
| `rendering.pairwise_j_rows` | `render/observables.ts:pairwiseJRows` | `test_parity.py::test_obs_cases` |
| `rendering.nearest_observed` | `render/corpus.ts:nearestObserved` | `test_parity.py::test_corpus_cases` |
| `rendering_html.species_to_slug` + `_HYPHEN_BASE_SPECIES` | `render/sprite-url.ts:speciesToSlug` + `HYPHEN_BASE_SPECIES` | `test_parity.py::test_species_to_slug_cases` |
| `rendering_html.extract_species` / `extract_item` | `render/format.ts:extractSpecies` / `extractItem` | covered indirectly by parity baseline construction |
| `loaders.format_pair` | `render/format.ts:formatPair` | covered indirectly (precompute writes the joined string into `meta.json:vocab`) |
| `precompute.pack_lower_triangle` (writer) | `sampler/model.ts:unpackLowerTriangle` (reader) | JS smoke + `tests/test_precompute.py` round-trip |
| `precompute.serialize_team_counts` schema | `sampler/model.ts:loadTeamCounts` + `render/corpus.ts:teamKey` | `tests/test_precompute.py` + `test_parity.py::test_corpus_cases` |
| `constants.{TEAM_SIZE, FIELD_WEIGHT_OPTIONS, TEMPERATURE_OPTIONS, TOP_COMPLETIONS, TOP_SINGLE_SWAPS, GREEDY_MAX_SWAPS, META_TOP_FEATURES, META_TOP_PAIRS, PT_*, MF_*}` | `web/src/constants.ts` | manual (rarely changes; values pinned by v1 simplification) |
| `rendering_html` HTML class names (`lab-slot`, `lab-comp-table`, …) | `web/src/render/*.tsx` className strings + `web/src/styles/components.css` | visual — class renames must touch both sides |
| `styles.py` `LAB_*` constants + CSS rules | `web/src/styles/{tokens,layout,components,widgets}.css` | visual — palette/typography changes touch both |

Parity baseline regeneration:

```bash
cd web && npm run emit-baseline   # writes tests/parity_baseline.json (JS-side outputs)
cd .. && python -m unittest tests.test_parity   # Python rerun + compare
```

The synthetic model used in the baseline is hand-built (V=12, team_size=4) so both stacks compute it byte-identically. Tolerance is `1e-9` for deterministic cases (MF, greedy, rank, obs); stochastic cases (swap MCMC, PT) are smoke-tested for well-formed output only.

## Data file

`gen9championsvgc2026regma-1760.json` (VGC Reg M-A Champions, 1760 elo cutoff) is the current target metagame. An earlier 1500-cutoff file was tried, found equivalent, and dropped. An earlier BSS variant existed briefly; do not reintroduce it.

## External reference

`~/Projects/k2dex-calculator/scripts/update_meta.py` — sibling project's Smogon chaos-JSON downloader/parser. Reuse if we ever need to re-fetch or move to a different format-month.

