"""Tests for the pure helpers in k2dex.loaders (sample weighting)."""

from __future__ import annotations

import unittest

import numpy as np

from k2dex.loaders import team_weights
from k2dex.tournament_ingest import TeamObservation


def _obs(date: str, tournament_type: str = "limitless") -> TeamObservation:
    return TeamObservation(
        members=frozenset({("Incineroar", "Sitrus Berry")}),
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


if __name__ == "__main__":
    unittest.main()
