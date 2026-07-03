"""Shared numeric constants for the k2dex project.

Centralizes values that previously drifted between app.py, tournament_ingest.py,
and the notebooks. Anything that's a meaningful knob (corpus size, vocab
cutoff, regularization strength) lives here.
"""
from __future__ import annotations

from pathlib import Path

# --- Regulation ---
CURRENT_REGULATION = "M-B"

# --- Team / sampling ---
TEAM_SIZE = 6

# --- Phase 1 (Gaussian inverse Ising on Smogon chaos) ---
PHASE1_MIN_USAGE = 0.002       # 170 Pokemon at Reg M-A 1760
PHASE1_RIDGE_EPS = 0.01        # ridge for precision-matrix inversion

# --- Phase 2 / 3 (PL inverse Ising on Limitless teams) ---
LIMITLESS_MAX_TEAMS = 25000    # Limitless API fetch limit: stop walking after this many teams
PHASE2_MIN_TEAM_COUNT = 3      # vocab cutoff: feature must appear in >=5 teams
SPECIES_LR_LAMBDA = 25.0       # L2 regularization strength for the species model
SPECIES_ITEM_LR_LAMBDA = 4.5   # L2 regularization strength for the species+item model

# --- Boltzmann (moment-matching) fit defaults ---
# Single source of truth for the `fit_boltzmann_ising` build recipe: the function's
# signature defaults AND precompute.py's --bz-* defaults both read from here, so
# precompute artifacts, notebooks, and tests that don't override a knob all fit with
# identical hyperparameters. See notebooks/boltzmann_learning.ipynb and CLAUDE.md's
# "Boltzmann learning" notes for the convergence gotchas each knob addresses.
BOLTZMANN_N_ITERS = 1000                # gradient steps
BOLTZMANN_LR = 0.01                     # Adam learning rate
BOLTZMANN_LR_FINAL = BOLTZMANN_LR / 20.0  # cosine-decay target: shrinks the SG noise
                                        # ball (the dominant fit error), ~7x lower moment
                                        # bias. Set == BOLTZMANN_LR to disable decay.
BOLTZMANN_AVG_LAST = 100                # Polyak iterate-averaging window (0 = off;
                                        # alternative to lr decay, comparable effect)
BOLTZMANN_N_CHAINS = 500                # PCD chain-bank size
BOLTZMANN_N_SWEEPS = 100                # sweeps per chain per gradient step
BOLTZMANN_N_BURN = 200                  # initial bank-mixing sweeps per chain
BOLTZMANN_N_TEMPS = 3                   # PT replicas per chain (1 = off; single-T PCD
                                        # sufficed. >1 only for multimodal models)
BOLTZMANN_T_MAX = 3.0                   # hot temperature for tempering (unused if N_TEMPS == 1)
BOLTZMANN_SWAP_INTERVAL = 10            # sweeps between replica-exchange attempts
BOLTZMANN_REG = "l2"                    # regularizer toward zero ("l1" or "l2")
BOLTZMANN_REG_LAMBDA = 1e-3             # regularization strength (small: tightest moment
                                        # match; larger deliberately shrinks (J,h))
BOLTZMANN_SEED = 0                      # PCD sampler RNG seed

# --- Sample weighting (weighted base model) ---
# Per-team fit weight: w = exp(-age_days / RECENCY_TAU_DAYS) * IN_PERSON_WEIGHT^[in-person],
# normalized so the weights average to 1. Tuned by notebooks/weighting_sweep.ipynb.
RECENCY_TAU_DAYS = 30.0                # decay timescale in days; None = no recency decay
IN_PERSON_WEIGHT = 1.0                 # multiplier on in-person teams; 1.0 = no upweight

# --- Limitless ingest filter ---
MIN_TEAMS_PER_TOURNAMENT = 16  # was 16; bump spreads corpus more temporally
                               # and indirectly filters small-tournament quirks

# --- In-person tournament data ---
DEFAULT_LOCAL_DIR = Path("tournament_json")

# --- Validation ---
VALIDATION_TEAM_FRAC_TEST = 0.10   # chronological: newest ~10% of teams -> test
VALIDATION_SEED = 42

# --- Temporal drift ---
DRIFT_TRAIN_FRAC = 0.25            # oldest 25% of teams -> train; 75% -> windowed test
