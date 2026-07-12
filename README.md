# k2dex

Interactive tools and explainers applying complexity-science methods to competitive Pokemon (VGC) team composition data.

The core model is a **pairwise maximum-entropy (inverse Ising)** fit on tournament team rosters from [Limitless VGC](https://play.limitlesstcg.com/) and in-person tournament data. The fitted couplings *J* and biases *h* capture which Pokemon (and held items) tend to appear together on teams, and which ones compete for the same slot. The pipeline supports **N arbitrary models** across different regulations, each built independently via a CLI:

- **Species models**: pseudo-likelihood fit on species-only features
- **Species @ Item models**: pseudo-likelihood fit on `(species, held item)` pair features, preserving held-item forme distinctions that the species model collapses

Models are organized by regulation (e.g. M-A, M-B) and discoverable via a `manifest.json` that the webapp fetches on boot. A collapsible model picker groups available models by regulation.

The webapp surfaces these as a **team completer** (parallel-tempered MCMC or greedy sampling), **per-team diagnostics** (pairwise couplings, swap suggestions, nearest observed teams), and **format-wide statistics** (bias rankings, extreme couplings). An **Articles** section holds interactive explainers: "The Science of k2dex" walks through the math from first principles with toy simulations, and "Why Not Just Count?" compares the model head-to-head against the raw co-occurrence baseline every other teambuilder uses.

A local Streamlit webapp is also available, which surfaces all available models, parameters and options. Intended as a scientific dashboard and playground, rather than as a tool for competitive Pokemon.

**[Live site →](https://kylehtunis.github.io/k2dex/)**

## Repository structure

```
k2dex/                  Python package: model fitting, sampling, rendering
  constants.py          Shared numeric constants (vocab cutoffs, regularization, seeds)
  helpers.py            Gaussian inverse Ising primitives (Smogon chaos path)
  loaders.py            Model builders (parameterized by regulation, lambda, vocab cutoff)
  models.py             fit_pl_ising: per-spin L2 logistic regression + symmetrize
  sampling.py           MCMC family (swap / anneal / PT) + mean-field + greedy + rank_single_swaps
  rendering.py          Diagnostic helpers + markdown-table builders
  rendering_html.py     Lab-notebook HTML helpers (sprites, slot cards, tables)
  styles.py             Design tokens + Streamlit widget overrides
  tournament_ingest.py   Tournament data ingest, in-person import, and unified cache

scripts/                CLI entry points
  app.py                Streamlit webapp
  precompute.py         Offline pipeline: fits one model per invocation → web/public/models/<slug>/
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
python -m k2dex.tournament_ingest --regulation G --start-page 20  # older format, skip recent pages
```

The Limitless path walks tournaments newest-first, applying size/format filters, until it reaches the fetch limit (`--max-teams`, default 25,000). `--start-page` skips past recent catalog pages to reach tournaments from older formats. The in-person path reads raw standings exports from `tournament_json/[format]/[id]_[date].json` and normalizes them into cache entries. Both paths are idempotent; re-running only processes new data.

### 3. Precompute model artifacts

Build one model per invocation:

```bash
# Species @ Item model for Regulation M-A
python scripts/precompute.py \
  --display-name "Reg M-A Species @ Item" \
  --regulation M-A \
  --type species_item

# Species-only model for Regulation M-A
python scripts/precompute.py \
  --display-name "Reg M-A Species" \
  --regulation M-A \
  --type species

# Override regularization (lambda = L2 penalty strength; default 10.0 for species, 1.0 for species_item)
python scripts/precompute.py \
  --display-name "Reg M-B Species @ Item" \
  --regulation M-B \
  --type species_item \
  --lambda 2.0

# Weighted fit: exponential recency decay (--tau, in days) and an in-person
# upweight (--in-person-weight). Defaults come from k2dex/constants.py
# (RECENCY_TAU_DAYS / IN_PERSON_WEIGHT, tuned by notebooks/weighting_sweep.ipynb);
# both values are recorded in meta.json under "fit".
python scripts/precompute.py \
  --display-name "Reg M-A Species @ Item" \
  --type species_item \
  --tau 90 \
  --in-person-weight 2.0

# After building all models, generate the manifest for webapp discovery
python scripts/precompute.py --generate-manifest --default-model reg-m-a-species-item
```

Each invocation writes `meta.json`, `J.bin`, `h.bin`, `m.bin`, and `team_counts.json` under `web/public/models/<slug>/`, where the slug is auto-generated from the display name. The manifest (`manifest.json`) lists all models with summary metadata so the webapp can populate its model picker without loading model data. Inspect the output before committing. CI does not regenerate these; they are committed to git after manual review.

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

The Streamlit app is a development/research tool with a model selector dropdown driven by the same `manifest.json`. It fits models live from the cache (via `@st.cache_resource`) rather than loading precomputed binaries. Requires the cache to be populated first (step 2).

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
| 2 | `inverse_ising.ipynb` | Gaussian / precision-matrix `(J, h)` fit and visualization |
| 3 | `j_communities.ipynb` | Louvain community detection on +J (archetypes) and -J (role pools) subgraphs |
| 4 | `forward_ising.ipynb` | Forward sampling: swap-move MCMC, calibration against empirical marginals |
| 5 | `inverse_ising_phase2.ipynb` | Pseudo-likelihood species fit on Limitless rosters |
| 6 | `validation.ipynb` | Leave-k-out evaluation, cross-model comparison (chronological 90/10 split) |
| 7 | `temporal_drift.ipynb` | Standalone: how quickly model accuracy degrades as the meta evolves (25/75 split) |
| 8 | `energy_discrimination.ipynb` | Standalone: real-vs-null scoring, Bias Adjustment sweep, AUC analysis |
| 9 | `regularization_sweep.ipynb` | Standalone: L2 regularization sweep behind the per-model lambda values |
| 10 | `outcome_validation.ipynb` | Standalone: does Score/coherence predict tournament placement? (confirms the model captures meta typicality, not team quality) |
| 11 | `weighting_sweep.ipynb` | Standalone: tunes the sample-weighting knobs (recency decay tau, in-person multiplier) on held-out future in-person events |
| 12 | `anchor_field_tilt.ipynb` | Standalone: validates the completer's Anchor Strength tilt (pin-integration monotonicity, seed stability, score/variety cost, popular-pin control) |

## Key concepts

- **Inverse Ising model**: given binary team-composition vectors (each Pokemon/item is a spin), find the pairwise couplings *J* and biases *h* that make the observed teams maximally likely under a Boltzmann distribution
- **Parallel-tempered MCMC**: runs multiple chains at different temperatures with replica-exchange swaps, helping escape local minima. Runs in a Web Worker to keep the UI responsive
- **Anchor Strength**: an exponential tilt on the completer's sampling distribution that amplifies couplings between the pinned Pokemon and every candidate teammate, so completions build around your picks instead of attaching them to a generic strong core. 1.0 is neutral (the fitted distribution)

## License

[GPL-3.0](LICENSE)
