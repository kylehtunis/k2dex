"""Tests for the decoupled ingest pipeline: cache loading, in-person import,
and provenance tracking."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from k2dex.tournament_ingest import (
    CACHE_VERSION,
    TournamentMeta,
    TournamentTeams,
    Team,
    _save_tournament,
    all_team_observations,
    load_cached_tournaments,
    import_in_person_tournaments,
    normalize_bracket_forme,
)


def _make_team(species_items: list[tuple[str, str | None]], placing=None) -> Team:
    return Team(
        members=frozenset(species_items),
        placing=placing,
        wins=3, losses=1, ties=0,
    )


def _make_tournament(
    tid: str, name: str, date: str, regulation: str, teams: list[Team],
    tournament_type: str = "limitless",
) -> TournamentTeams:
    return TournamentTeams(
        meta=TournamentMeta(
            id=tid, name=name, date=date, regulation=regulation, players=len(teams),
        ),
        teams=teams,
        tournament_type=tournament_type,
    )


SAMPLE_TEAMS = [
    _make_team([
        ("Venusaur", "Focus Sash"), ("Torkoal", "Charcoal"),
        ("Rillaboom", "Miracle Seed"), ("Incineroar", "Safety Goggles"),
        ("Urshifu", "Choice Band"), ("Farigiraf", "Leftovers"),
    ], placing=i + 1)
    for i in range(40)
]


class TestSaveTournamentType(unittest.TestCase):
    def test_type_field_written(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            tt = _make_tournament(
                "t1", "Test Event", "2026-01-01", "M-A", SAMPLE_TEAMS,
                tournament_type="in-person",
            )
            _save_tournament(cache_dir, tt)

            with (cache_dir / "t1.json").open() as f:
                payload = json.load(f)
            self.assertEqual(payload["type"], "in-person")
            self.assertEqual(payload["version"], CACHE_VERSION)

    def test_default_type_is_limitless(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            tt = _make_tournament("t2", "Online Cup", "2026-02-01", "M-A", SAMPLE_TEAMS)
            _save_tournament(cache_dir, tt)

            with (cache_dir / "t2.json").open() as f:
                payload = json.load(f)
            self.assertEqual(payload["type"], "limitless")


class TestLoadCachedTournaments(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.tmpdir.name)
        t1 = _make_tournament("a", "Alpha", "2026-01-10", "M-A", SAMPLE_TEAMS)
        t2 = _make_tournament("b", "Beta", "2026-01-05", "M-A", SAMPLE_TEAMS,
                              tournament_type="in-person")
        t3 = _make_tournament("c", "Gamma", "2026-02-01", "G", SAMPLE_TEAMS)
        t4 = _make_tournament("d", "Delta Singles", "2026-01-15", "M-A", SAMPLE_TEAMS)
        for t in (t1, t2, t3, t4):
            _save_tournament(self.cache_dir, t)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_filters_by_regulation(self):
        result = load_cached_tournaments(cache_dir=self.cache_dir, regulation="M-A")
        ids = {t.meta.id for t in result}
        self.assertIn("a", ids)
        self.assertIn("b", ids)
        self.assertNotIn("c", ids)

    def test_filters_singles_by_name(self):
        result = load_cached_tournaments(cache_dir=self.cache_dir, regulation="M-A")
        ids = {t.meta.id for t in result}
        self.assertNotIn("d", ids)

    def test_filters_by_type(self):
        result = load_cached_tournaments(
            cache_dir=self.cache_dir, regulation="M-A", tournament_type="in-person",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].meta.id, "b")

    def test_tournament_type_round_trips(self):
        result = load_cached_tournaments(cache_dir=self.cache_dir, regulation="M-A")
        types = {t.meta.id: t.tournament_type for t in result}
        self.assertEqual(types, {"a": "limitless", "b": "in-person"})

    def test_all_team_observations_keeps_provenance(self):
        result = load_cached_tournaments(cache_dir=self.cache_dir, regulation="M-A")
        obs = all_team_observations(result)
        self.assertEqual(len(obs), sum(len(t.teams) for t in result))
        # Same ordering as the tournament list: result is date-sorted, so the
        # in-person event ("b", 2026-01-05) contributes the first block.
        self.assertEqual(obs[0].date, "2026-01-05")
        self.assertEqual(obs[0].tournament_type, "in-person")
        self.assertEqual(obs[-1].date, "2026-01-10")
        self.assertEqual(obs[-1].tournament_type, "limitless")
        self.assertEqual(obs[0].members, SAMPLE_TEAMS[0].members)

    def test_sorted_by_date(self):
        result = load_cached_tournaments(cache_dir=self.cache_dir, regulation="M-A")
        dates = [t.meta.date for t in result]
        self.assertEqual(dates, sorted(dates))

    def test_old_cache_defaults_to_limitless(self):
        payload = {
            "version": CACHE_VERSION,
            "meta": {"id": "old", "name": "Old Event", "date": "2025-12-01",
                     "regulation": "M-A", "players": 40},
            "teams": [
                {"members": [list(m) for m in sorted(t.members, key=lambda m: m[0])],
                 "placing": t.placing, "record": [t.wins, t.losses, t.ties]}
                for t in SAMPLE_TEAMS
            ],
        }
        with (self.cache_dir / "old.json").open("w") as f:
            json.dump(payload, f)

        result = load_cached_tournaments(
            cache_dir=self.cache_dir, regulation="M-A", tournament_type="limitless",
        )
        ids = {t.meta.id for t in result}
        self.assertIn("old", ids)

    def test_size_filter(self):
        small_teams = SAMPLE_TEAMS[:5]
        t = _make_tournament("small", "Tiny Cup", "2026-01-20", "M-A", small_teams)
        _save_tournament(self.cache_dir, t)
        result = load_cached_tournaments(
            cache_dir=self.cache_dir, regulation="M-A", min_teams_per_tournament=32,
        )
        ids = {t.meta.id for t in result}
        self.assertNotIn("small", ids)

    def test_missing_cache_dir(self):
        result = load_cached_tournaments(cache_dir=Path("/nonexistent"))
        self.assertEqual(result, [])


class TestImportInPerson(unittest.TestCase):
    def test_import_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            local_dir = base / "local"
            cache_dir = base / "cache"
            cache_dir.mkdir()
            reg_dir = local_dir / "M-A"
            reg_dir.mkdir(parents=True)

            standings = [
                {
                    "name": f"Player {i}",
                    "placing": i + 1,
                    "record": {"wins": 3, "losses": 1, "ties": 0},
                    "decklist": [
                        {"name": f"Mon{j}", "item": f"Item{j}"}
                        for j in range(6)
                    ],
                }
                for i in range(40)
            ]
            with (reg_dir / "ev001_2026-05-30.json").open("w") as f:
                json.dump(standings, f)

            result1 = import_in_person_tournaments(local_dir=local_dir, cache_dir=cache_dir)
            self.assertEqual(len(result1), 1)
            self.assertEqual(result1[0].meta.regulation, "M-A")
            self.assertEqual(result1[0].meta.date, "2026-05-30")
            self.assertTrue(result1[0].meta.id.startswith("local_"))

            cache_file = cache_dir / f"{result1[0].meta.id}.json"
            self.assertTrue(cache_file.exists())
            with cache_file.open() as f:
                payload = json.load(f)
            self.assertEqual(payload["type"], "in-person")

            result2 = import_in_person_tournaments(local_dir=local_dir, cache_dir=cache_dir)
            self.assertEqual(len(result2), 0)

    def test_date_parsing_underscore_format(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            local_dir = base / "local"
            cache_dir = base / "cache"
            cache_dir.mkdir()
            reg_dir = local_dir / "M-A"
            reg_dir.mkdir(parents=True)

            standings = [
                {
                    "name": f"Player {i}",
                    "placing": i + 1,
                    "record": {"wins": 3, "losses": 1, "ties": 0},
                    "decklist": [
                        {"name": f"Mon{j}", "item": f"Item{j}"}
                        for j in range(6)
                    ],
                }
                for i in range(40)
            ]
            with (reg_dir / "0000187-Masters_2026_05_30.json").open("w") as f:
                json.dump(standings, f)

            result = import_in_person_tournaments(local_dir=local_dir, cache_dir=cache_dir)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].meta.date, "2026-05-30")


class TestNormalizeBracketForme(unittest.TestCase):
    def test_regional_hisuian(self):
        self.assertEqual(normalize_bracket_forme("Arcanine [Hisuian Form]"), "Hisuian Arcanine")
        self.assertEqual(normalize_bracket_forme("Typhlosion [Hisuian Form]"), "Hisuian Typhlosion")

    def test_regional_alolan(self):
        self.assertEqual(normalize_bracket_forme("Ninetales [Alolan Form]"), "Alolan Ninetales")
        self.assertEqual(normalize_bracket_forme("Raichu [Alolan Form]"), "Alolan Raichu")

    def test_regional_galarian(self):
        self.assertEqual(normalize_bracket_forme("Slowbro [Galarian Form]"), "Galarian Slowbro")
        self.assertEqual(normalize_bracket_forme("Slowking [Galarian Form]"), "Galarian Slowking")

    def test_regional_paldean_with_breed(self):
        self.assertEqual(
            normalize_bracket_forme("Tauros [Paldean Form - Aqua Breed]"),
            "Paldean Tauros Aqua Breed",
        )
        self.assertEqual(
            normalize_bracket_forme("Tauros [Paldean Form - Combat Breed]"),
            "Paldean Tauros Combat Breed",
        )

    def test_rotom_formes(self):
        self.assertEqual(normalize_bracket_forme("Rotom [Wash Rotom]"), "Wash Rotom")
        self.assertEqual(normalize_bracket_forme("Rotom [Heat Rotom]"), "Heat Rotom")
        self.assertEqual(normalize_bracket_forme("Rotom [Mow Rotom]"), "Mow Rotom")

    def test_cosmetic_stripped(self):
        self.assertEqual(normalize_bracket_forme("Maushold [Family of Four]"), "Maushold")
        self.assertEqual(normalize_bracket_forme("Maushold [Family of Three]"), "Maushold")
        self.assertEqual(normalize_bracket_forme("Sinistcha [Unremarkable Form]"), "Sinistcha")
        self.assertEqual(normalize_bracket_forme("Sinistcha [Masterpiece Form]"), "Sinistcha")
        self.assertEqual(normalize_bracket_forme("Polteageist [Phony Form]"), "Polteageist")

    def test_gender_male_stripped(self):
        self.assertEqual(normalize_bracket_forme("Meowstic [Male]"), "Meowstic")
        self.assertEqual(normalize_bracket_forme("Basculegion [Male]"), "Basculegion")

    def test_gender_female(self):
        self.assertEqual(normalize_bracket_forme("Basculegion [Female]"), "Basculegion ♀")
        self.assertEqual(normalize_bracket_forme("Meowstic [Female]"), "Meowstic ♀")

    def test_named_forme_suffix(self):
        self.assertEqual(normalize_bracket_forme("Lycanroc [Dusk Form]"), "Lycanroc Dusk")
        self.assertEqual(normalize_bracket_forme("Lycanroc [Midday Form]"), "Lycanroc Midday")
        self.assertEqual(normalize_bracket_forme("Lycanroc [Midnight Form]"), "Lycanroc Midnight")

    def test_eternal_flower(self):
        self.assertEqual(normalize_bracket_forme("Floette [Eternal Flower]"), "Eternal Flower Floette")

    def test_no_brackets_unchanged(self):
        self.assertEqual(normalize_bracket_forme("Incineroar"), "Incineroar")
        self.assertEqual(normalize_bracket_forme("Wash Rotom"), "Wash Rotom")

    def test_empty_and_none(self):
        self.assertEqual(normalize_bracket_forme(""), "")


if __name__ == "__main__":
    unittest.main()
