"""Tests for the pure helpers in k2dex.loaders (sample weighting)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from k2dex.loaders import build_species_item_model, team_weights
from k2dex.tournament_ingest import (
    Team,
    TeamObservation,
    TournamentMeta,
    TournamentTeams,
    _save_tournament,
)


def _obs(date: str, tournament_type: str = "limitless") -> TeamObservation:
    return TeamObservation(
        members=frozenset({("Incineroar", "Sitrus Berry", "Intimidate")}),
        date=date,
        tournament_type=tournament_type,
    )


class TestTeamWeights(unittest.TestCase):
    def test_neutral_settings_give_unit_weights(self) -> None:
        obs = [_obs("2026-01-01"), _obs("2026-03-01", "in-person"), _obs("2026-05-01")]
        w = team_weights(obs, reference_date="2026-05-01")
        np.testing.assert_allclose(w, np.ones(3))

    def test_neutral_settings_tolerate_undated_observations(self) -> None:
        # With recency_tau=None dates are never parsed, so an empty date
        # (possible for in-person files without a date suffix) is fine.
        obs = [_obs(""), _obs("2026-05-01")]
        w = team_weights(obs, reference_date="2026-05-01", in_person_multiplier=2.0)
        np.testing.assert_allclose(w, np.ones(2))

    def test_recency_decay_ratio(self) -> None:
        # One tau of age -> weight ratio of exactly e.
        obs = [_obs("2026-04-21"), _obs("2026-05-01")]  # ages 10, 0 days
        w = team_weights(obs, reference_date="2026-05-01", recency_tau=10.0)
        self.assertAlmostEqual(w[1] / w[0], np.e, places=12)

    def test_decay_parses_limitless_timestamps(self) -> None:
        # Limitless cache dates are full ISO timestamps; in-person dates are
        # bare YYYY-MM-DD. Both must parse at day resolution.
        obs = [_obs("2026-04-21T21:30:00.000Z"), _obs("2026-05-01", "in-person")]
        w = team_weights(
            obs, reference_date="2026-05-01T10:00:00.000Z", recency_tau=10.0,
        )
        self.assertAlmostEqual(w[1] / w[0], np.e, places=12)

    def test_in_person_multiplier_ratio(self) -> None:
        obs = [_obs("2026-05-01"), _obs("2026-05-01", "in-person")]
        w = team_weights(obs, reference_date="2026-05-01", in_person_multiplier=3.0)
        self.assertAlmostEqual(w[1] / w[0], 3.0, places=12)

    def test_normalized_to_team_count(self) -> None:
        obs = [
            _obs("2026-01-01"),
            _obs("2026-02-15", "in-person"),
            _obs("2026-04-01"),
            _obs("2026-05-01", "in-person"),
        ]
        w = team_weights(
            obs, reference_date="2026-05-01",
            recency_tau=30.0, in_person_multiplier=2.5,
        )
        self.assertAlmostEqual(float(w.sum()), len(obs), places=12)
        self.assertTrue(np.all(w > 0))
        # Newer in-person teams must outweigh older online teams.
        self.assertGreater(w[3], w[0])


class TestWeightedVocabCutoff(unittest.TestCase):
    """The vocab cutoff requires min_team_count in BOTH raw and weighted
    counts: a single highly-weighted recent team must not push a one-off
    feature into the vocab, and a feature whose support is entirely decayed
    away must drop out."""

    BASE = [(f"Mon{k}", f"Item{k}", f"Ability{k}") for k in range(12)]

    def _team(self, members) -> Team:
        return Team(members=frozenset(members), placing=None, wins=0, losses=0, ties=0)

    def _rotating(self, t: int, n: int) -> list[tuple[str, str, str]]:
        # n distinct members per team, rotating through BASE so no column is
        # all-on / all-off (which would trip the degenerate-spin skip).
        return [self.BASE[(t + j) % len(self.BASE)] for j in range(n)]

    def _tournament(self, tid, date, ttype, teams) -> TournamentTeams:
        meta = TournamentMeta(id=tid, name=tid, date=date, regulation="M-A",
                              players=len(teams))
        return TournamentTeams(meta=meta, teams=teams, tournament_type=ttype)

    def test_single_heavy_team_cannot_mint_vocab(self) -> None:
        # 64 old online teams (age 60d) + 32 recent in-person teams. At
        # tau=20 / mult=4 each recent team carries weight ~2.9, above the
        # cutoff of 2. One recent team carries a unique (ghost) pair; two old
        # teams carry a (stale) pair whose weighted mass decays below 2.
        old_teams = [self._team(self._rotating(t, 6)) for t in range(62)]
        old_teams += [self._team(self._rotating(t, 5) + [("StaleMon", "Stale Item", "Stale Ability")])
                      for t in range(2)]
        new_teams = [self._team(self._rotating(t, 6)) for t in range(31)]
        new_teams += [self._team(self._rotating(7, 5) + [("GhostMon", "Ghost Item", "Ghost Ability")])]

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            try:
                os.chdir(td)
                cache = Path("tournaments_cache")
                _save_tournament(cache, self._tournament(
                    "old", "2026-01-01", "limitless", old_teams))
                _save_tournament(cache, self._tournament(
                    "new", "2026-03-02", "in-person", new_teams))

                vocab, m, J, h, *_ = build_species_item_model(
                    regulation="M-A", min_team_count=2,
                    recency_tau=20.0, in_person_multiplier=4.0,
                )
            finally:
                os.chdir(cwd)

        self.assertNotIn("GhostMon @ Ghost Item (Ghost Ability)", vocab)  # raw=1, weighted ~2.9
        self.assertNotIn("StaleMon @ Stale Item (Stale Ability)", vocab)  # raw=2, weighted ~0.1
        self.assertIn("Mon0 @ Item0 (Ability0)", vocab)                   # staple: both counts high
        # No degenerate-skip artifacts: every vocab feature has a fitted h.
        self.assertTrue(np.all(h != 0.0))


if __name__ == "__main__":
    unittest.main()
