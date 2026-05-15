# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code applying complexity-science / distributional-semantics methods to competitive Pokémon (VGC) team-composition data. **Read `vgc_complexity_phase1_plan.md` first** — it's the authoritative source on Phase 1 methodology, scope, and rationale. Phase 1 uses only Smogon chaos JSON (marginal usage + pairwise teammate co-occurrence) and fits inverse Ising via the Gaussian / precision-matrix approximation. Phase 2 (in progress) adds team-level data from the Limitless TCG API for direct pseudo-likelihood fitting; ingestor is `limitless_ingest.py`.

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
5. **`app.py`** — Streamlit wrapper around the forward-sampling work. Three modes: single-chain sampling, annealing-to-MAP, parallel-tempered sampling. Top of the sidebar toggles between the Phase 1 (Gaussian, Smogon) and Phase 2 (PL, Limitless teams) fits; each phase keeps its own multiselect state via a `key=` suffix.

**Phase 2 additions (parallel track to the numbered chain above):**

- **`limitless_ingest.py`** — fetches recent VGC Reg M-A tournaments from `play.limitlesstcg.com/api`, extracts parsed team rosters (six Pokémon names per team), caches one small JSON per tournament under `tournaments_cache/`. Walks newest-first until `>= min_teams` (default 1500) teams accumulate. Item/ability/move/tera/player metadata is dropped at parse time; only team rosters and minimal identifying metadata are persisted.
- **`inverse_ising_phase2.ipynb`** — direct pseudo-likelihood fit of `(J, h)` via V per-spin logistic regressions (sklearn `LogisticRegression`, L2 with `C=1.0`, lbfgs solver). Mirrors `inverse_ising.ipynb`'s analysis cells (top ±J pairs, network panels, per-query synergies/substitutes). +J network panel uses `q=0.90` (vs Phase 1's `q=0.75`) because PL's wider J distribution would otherwise over-saturate the force-directed layout.

## helpers.py

Stable primitives promoted from the notebooks. Type-first (frozen dataclass `ChaosData`, type hints, docstrings explaining the math). New functions get added here once they stabilize in a notebook — code lives in notebooks first.

- `load_chaos`, `build_vocab`, `build_cooccurrence`, `build_ppmi` — Stage 0–1
- `binary_moments`, `binary_correlation`, `ising_gaussian` — Stage 4 (inverse Ising)

The swap-move MCMC sampler and the team-completer (`complete_*` / `anneal_*` / parallel-tempered) currently live in `forward_ising.ipynb` and `app.py` as inline code, not in `helpers.py`. Promote once stable. The Phase 2 PL fit (per-spin logistic regression loop) similarly lives inline in `inverse_ising_phase2.ipynb` and `app.py:load_model_phase2` — same "promote when stable" convention.

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
- **Phase 1 and Phase 2 vocabs are not directly comparable.** Smogon chaos data splits Pokémon by held-item-induced forme (`Charizard-Mega-Y`, `Charizard-Mega-X`, base `Charizard` are three entries). Limitless standings give only the species name (one `Charizard` entry, items dropped at parse time in `limitless_ingest.py`). Within each phase the vocab is self-consistent — but cross-phase comparisons of J need an explicit collapse step on the Phase 1 side (sum/merge mega variants into the base mon) before indices can be aligned. Mutually-exclusive forme pairs are also invisible in Phase 2 by construction (you can't run two Charizards).
- **The most-played teams in the Phase 2 corpus have negative intra-team ΣJ.** The empirical mode of Reg M-A at 5000 teams is the **Eternal Flower Floette variant** of the Aerodactyl/Basculegion/Charizard/Garchomp/Kingambit core (144 player-team-appearances, ~2.9% of corpus); the Sylveon variant (54x) is the 1-swap competitor. Both are flexibility/balance teams with $\sum_{i<j \in \text{team}} J_{ij} < 0$ under the PL fit, despite being the global energy minima at `fw=1` (most negative `raw_E` in the entire model). These teams are held together by individual mon quality, matchup coverage, and in-game flexibility — properties the Ising model fundamentally cannot represent. This is a structural ceiling on what any pairwise MaxEnt fit can do, not a regularization issue. **Heuristic**: teams with negative ΣJ are "balance team" signatures (model can't see what makes them work); teams with positive ΣJ are "archetype" signatures (model can see the synergy).
- **Model rankings are recency-biased because ingest walks newest-first.** A short cautionary tale: an earlier observation that "the model ranks the Sylveon variant above the Floette variant, mirroring the community's emerging shift" turned out to be an artifact of `PHASE2_MIN_TEAMS` being stale at 1500 in the app — the model was being fit on the most-recent slice of the corpus, where Sylveon teams over-represent. With `PHASE2_MIN_TEAMS = 5000` (the full ingested corpus), Floette correctly outranks Sylveon, matching the broader corpus's frequency ordering. Lesson: the model is a summary statistic of whatever subset it's fit on, not a predictor of meta evolution. **A real test of "model predicts meta shift"** would fit on an older time window and check whether held-out recent tournaments are enriched for the model's top-K — an experiment worth running explicitly with date-filtered tournaments.
- **`field_weight` sweep regime map (Phase 2 app)** — useful operational mental model when interpreting completions:

  | `fw` | regime | top-K characteristic |
  | ---: | :--- | :--- |
  | 1.0 | popularity-driven | 1-swap variant of meta mode (empirical mode + h gradient) |
  | 0.7 | balanced | empirical mode wins; variants nearby |
  | 0.5 | archetype-coherent | real high-ΣJ archetypes (sand+balance, Perish Trap) at Δ=0; empirical mode drops mid-rank because it's negative-ΣJ |
  | 0.3 | exploration | archetypes stable; Δ=1–2 structured variants emerge below |
  | 0.0 | pure-J / overfit | repeated-specific-team basins dominate. The persistent Alakazam/Flapple/Machamp cluster occupies 7+ of top 10 at both the 1500-team and 5000-team corpora |

  **Useful operating range with current `PHASE2_LR_C = 0.1`: `fw ≈ 0.3–0.6`.** At sklearn's default `C = 1.0` the floor was tighter and the useful range narrower (~0.3–0.5); dropping to `C = 0.1` shrinks rare-co-occurrence couplings enough to extend the useful range modestly without compromising strong-archetype detection at moderate fw. **More data alone does not fix repeated-roster basins** — the persistent Alakazam basin at 5000 teams traces to exactly one specific roster (Aerodactyl/Alakazam/Flapple/Gyarados/Machamp/Sableye) repeated 5× in the corpus, where the PL fit reinforces its 15 pairwise couplings 5 times each, carving an attractor whose near-neighbors fill the surrounding rank slots regardless of `min_teams`. **Regularization is the lever, not corpus size**, and the `C = 0.1` default reflects this finding.

## Data file

`gen9championsvgc2026regma-1500.json` (VGC Reg M-A Champions, 1500 elo cutoff, 3.1M battles, 263 Pokémon) is the current target metagame. An earlier BSS variant existed briefly; do not reintroduce it.

## External reference

`~/Projects/k2dex-calculator/scripts/update_meta.py` — sibling project's Smogon chaos-JSON downloader/parser. Reuse if we ever need to re-fetch or move to a different format-month.
