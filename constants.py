"""Shared numeric constants for the k2dex-science project.

Centralizes values that previously drifted between app.py, limitless_ingest.py,
and the notebooks. Anything that's a meaningful knob (corpus size, vocab
cutoff, regularization strength) lives here.
"""
from __future__ import annotations

# --- Team / sampling ---
TEAM_SIZE = 6

# --- Phase 1 (Gaussian inverse Ising on Smogon chaos) ---
PHASE1_MIN_USAGE = 0.002       # 170 Pokemon at Reg M-A 1760
PHASE1_RIDGE_EPS = 0.01        # ridge for precision-matrix inversion

# --- Phase 2 / 3 (PL inverse Ising on Limitless teams) ---
PHASE2_MIN_TEAMS = 10000       # corpus size cutoff for ingest
PHASE2_MIN_TEAM_COUNT = 5      # vocab cutoff: feature must appear in >=5 teams
PHASE2_LR_C = 0.1              # L2 inverse-strength for the per-spin logreg

# --- Limitless ingest filter ---
MIN_TEAMS_PER_TOURNAMENT = 64  # was 16; bump spreads corpus more temporally
                               # and indirectly filters small-tournament quirks

# --- Validation ---
VALIDATION_TEAM_FRAC_TEST = 0.10   # 10% of teams (not tournaments) -> test
VALIDATION_SEED = 42
