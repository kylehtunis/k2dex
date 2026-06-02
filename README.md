# k2dex

Interactive tools and explainers applying complexity-science methods to competitive Pokemon (VGC) team composition data.

The core model is a **pairwise maximum-entropy (inverse Ising)** fit on tournament team rosters from [Limitless VGC](https://play.limitlesstcg.com/) and in-person tournament data. The fitted couplings *J* and biases *h* capture which Pokemon (and held items) tend to appear together on teams, and which ones compete for the same slot. Two model phases are user-facing:

- **Species** (Phase 2): pseudo-likelihood fit on species-only features
- **Species @ Item** (Phase 3): pseudo-likelihood fit on `(species, held item)` pair features, preserving held-item forme distinctions that Phase 2 collapses

The webapp surfaces these as a **team completer** (parallel-tempered MCMC or greedy sampling), **per-team diagnostics** (pairwise couplings, swap suggestions, nearest observed teams), and **format-wide statistics** (bias rankings, extreme couplings). An interactive [Science](https://kylehtunis.github.io/k2dex/science) page explains the math from first principles with toy simulations.

A local Streamlit webapp is also available, which surfaces **all** available models, parameters and options. Intended as a  scientific dashboard and playground, rather than as a tool for competitive Pokemon.

**[Live site →](https://kylehtunis.github.io/k2dex/)**

## Repository structure

```
k2dex/                  Python package: model fitting, sampling, rendering
  constants.py          Shared numeric constants (vocab cutoffs, regularization, seeds)
  helpers.py            Phase 1 Gaussian inverse Ising primitives
  loaders.py            Model builders: build_species_model / build_species_item_model
  models.py             fit_pl_ising: per-spin L2 logistic regression + symmetrize
  sampling.py           MCMC family (swap / anneal / PT) + mean-field + greedy + rank_single_swaps
  rendering.py          Diagnostic helpers + markdown-table builders
  rendering_html.py     Lab-notebook HTML helpers (sprites, slot cards, tables)
  styles.py             Design tokens + Streamlit widget overrides
  tournament_ingest.py   Tournament data ingest, in-person import, and unified cache

scripts/                CLI entry points
  app.py                Streamlit webapp
  precompute.py         Offline pipeline: fits models → web/public/models/
  scotus_precompute.py  SCOTUS inverse Ising fits → web/public/scotus/

tournament_json/        In-person tournament data (committed dir, JSON files gitignored)
  M-A/                  Reg M-A format; files named [id]_[YYYY-MM-DD].json
tournaments_cache/      Unified cache (gitignored); Limitless + in-person entries
notebooks/              Analysis pipeline (see Notebooks below)
tests/                  Unit tests + parity baseline (Python ↔ TypeScript)
web/                    Static React/TypeScript webapp (see below)
```

## Tech stacks

### Python (research + Streamlit app)

Dependencies are listed in `requirements.txt`: numpy, scipy, scikit-learn, streamlit, matplotlib, and jupyterlab.

### Static webapp (React/TypeScript)

A Vite + React 18 + TypeScript app under `web/`: a static site deployed to GitHub Pages, polishing and expanding upon the Streamlit app. All sampler math is a faithful 1:1 port of `k2dex/sampling.py`, gated by cross-stack parity tests (`tests/test_parity.py`).

Dependencies: `react-router-dom` (routing), `react-select` (vocab dropdowns), `react-katex` + `katex` (math typesetting on /science). Dev tooling: Vite, TypeScript, Vitest.

## Getting started

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Ingest tournament data

Populate the tournament cache from both data sources. Results are cached as one JSON per tournament under `tournaments_cache/`.

```bash
python -m k2dex.tournament_ingest              # fetch Limitless API + import in-person data
python -m k2dex.tournament_ingest --limitless-only    # Limitless API only
python -m k2dex.tournament_ingest --in-person-only    # import tournament_json/ only
```

The Limitless path walks tournaments newest-first, applying size/format filters, until it reaches the fetch limit (default 25,000 teams). The in-person path reads raw standings exports from `tournament_json/[format]/[id]_[date].json` and normalizes them into cache entries. Both paths are idempotent; re-running only processes new data.

### 3. Precompute model artifacts

Fit both models and serialize them for the static webapp:

```bash
python scripts/precompute.py
```

This writes `meta.json`, `J.bin`, `h.bin`, `m.bin`, and `team_counts.json` for each model under `web/public/models/{species,species_item}/`. Inspect the output (vocab size, corpus team count, file sizes) before committing. CI does not regenerate these; they are committed to git after manual review.

To rebuild the SCOTUS science-page artifacts:

```bash
python scripts/scotus_precompute.py
```

### 4. Run the webapp locally 

```bash
cd web
npm install        # one-time
npm run dev        # local dev server at http://localhost:5173
```

The precomputed model artifacts in `web/public/models/` are already committed to git, so you can skip steps 2-3 if you just want to run the webapp as-is.

### 5. Run the Streamlit dashboard

```bash
streamlit run scripts/app.py
```

The Streamlit app is a development/research tool. It reads the same models but fits them live from the cache (via `@st.cache_resource`) rather than loading precomputed binaries. Requires the cache to be populated first (step 2).

### 6. Run tests

```bash
# Python tests (includes parity checks against TypeScript outputs)
python -m unittest discover tests

# TypeScript tests
cd web && npm test

# Regenerate the parity baseline after changing sampler math on the JS side
cd web && npm run emit-baseline
```

## Notebooks

The notebooks form a dependency chain. Explore in order, or jump into any standalone notebook.

| # | Notebook | Description |
|---|---------|-------------|
| 1 | `spatial_embedding.ipynb` | PPMI, truncated SVD, role-residual NN ranking, NMF role decomposition |
| 2 | `inverse_ising.ipynb` | Phase 1: Gaussian / precision-matrix `(J, h)` fit and visualization |
| 3 | `j_communities.ipynb` | Louvain community detection on +J (archetypes) and -J (role pools) subgraphs |
| 4 | `forward_ising.ipynb` | Forward sampling: swap-move MCMC, calibration against empirical marginals |
| 5 | `inverse_ising_phase2.ipynb` | Phase 2: pseudo-likelihood fit on Limitless rosters |
| 6 | `validation.ipynb` | Leave-k-out evaluation, cross-model comparison (chronological 90/10 split) |
| 7 | `temporal_drift.ipynb` | Standalone: how quickly model accuracy degrades as the meta evolves (25/75 split) |
| 8 | `energy_discrimination.ipynb` | Standalone: real-vs-null scoring, Bias Adjustment sweep, AUC analysis |
| 9 | `regularization_sweep.ipynb` | Standalone: L2 regularization sweep behind the per-phase C values |
| 10 | `outcome_validation.ipynb` | Standalone: does Score/coherence predict tournament placement? (confirms the model captures meta typicality, not team quality) |

## Key concepts

- **Inverse Ising model**: given binary team-composition vectors (each Pokemon/item is a spin), find the pairwise couplings *J* and biases *h* that make the observed teams maximally likely under a Boltzmann distribution
- **Parallel-tempered MCMC**: runs multiple chains at different temperatures with replica-exchange swaps, helping escape local minima. Runs in a Web Worker to keep the UI responsive

## License

[GPL-3.0](LICENSE)
