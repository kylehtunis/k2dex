"""Tests for the decoupled ingest pipeline: cache loading, in-person import,
and provenance tracking."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from k2dex.tournament_ingest import (
    CACHE_VERSION,
    Match,
    TournamentMeta,
    TournamentTeams,
    Team,
    _save_tournament,
    all_match_observations,
    all_team_observations,
    extract_matches,
    extract_teams,
    load_cached_tournaments,
    import_in_person_tournaments,
    normalize_bracket_forme,
)


def _make_team(members: list[tuple[str, str | None, str]], placing=None) -> Team:
    return Team(
        members=frozenset(members),
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
        ("Venusaur", "Focus Sash", "Chlorophyll"), ("Torkoal", "Charcoal", "Drought"),
        ("Rillaboom", "Miracle Seed", "Grassy Surge"),
        ("Incineroar", "Safety Goggles", "Intimidate"),
        ("Urshifu", "Choice Band", "Unseen Fist"), ("Farigiraf", "Leftovers", "Armor Tail"),
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
                        {"name": f"Mon{j}", "item": f"Item{j}", "ability": f"Ability{j}"}
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
                        {"name": f"Mon{j}", "item": f"Item{j}", "ability": f"Ability{j}"}
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


def _standings_entry(
    name: str,
    rounds: dict[str, dict] | None = None,
    *,
    n_mons: int = 6,
    placing: int | None = None,
) -> dict:
    """One raw standings entry with a valid (or deliberately short) decklist."""
    return {
        "name": name,
        "placing": placing,
        "record": {"wins": 0, "losses": 0, "ties": 0},
        "rounds": rounds or {},
        "decklist": [
            {"name": f"Mon{j}", "item": f"Item{j}", "ability": f"Ability{j}"} for j in range(n_mons)
        ],
    }


class TestExtractMatches(unittest.TestCase):
    def test_basic_reciprocal_match(self):
        standings = [
            _standings_entry("Alice", {"1": {"name": "Bob", "result": "W"}}),
            _standings_entry("Bob", {"1": {"name": "Alice", "result": "L"}}),
        ]
        matches = extract_matches(standings)
        self.assertEqual(matches, [Match(round=1, winner=0, loser=1)])

    def test_match_emitted_once_despite_two_views(self):
        standings = [
            _standings_entry("Alice", {"1": {"name": "Bob", "result": "L"}}),
            _standings_entry("Bob", {"1": {"name": "Alice", "result": "W"}}),
        ]
        matches = extract_matches(standings)
        self.assertEqual(matches, [Match(round=1, winner=1, loser=0)])

    def test_bye_and_late_skipped(self):
        standings = [
            _standings_entry("Alice", {
                "1": {"name": "BYE", "result": "W"},
                "2": {"name": "LATE", "result": "W"},
                "3": {"name": "Bob", "result": "W"},
            }),
            _standings_entry("Bob", {"3": {"name": "Alice", "result": "L"}}),
        ]
        matches = extract_matches(standings)
        self.assertEqual(matches, [Match(round=3, winner=0, loser=1)])

    def test_duplicate_player_name_drops_their_matches(self):
        standings = [
            _standings_entry("Noah", {"1": {"name": "Alice", "result": "W"}}),
            _standings_entry("Noah", {"1": {"name": "Bob", "result": "W"}}),
            _standings_entry("Alice", {
                "1": {"name": "Noah", "result": "L"},
                "2": {"name": "Bob", "result": "W"},
            }),
            _standings_entry("Bob", {
                "1": {"name": "Noah", "result": "L"},
                "2": {"name": "Alice", "result": "L"},
            }),
        ]
        matches = extract_matches(standings)
        # Only the Alice-vs-Bob round 2 game survives: every match touching
        # the duplicated "Noah" is ambiguous and dropped.
        self.assertEqual(matches, [Match(round=2, winner=2, loser=3)])

    def test_double_loss_skipped(self):
        standings = [
            _standings_entry("Alice", {"1": {"name": "Bob", "result": "L"}}),
            _standings_entry("Bob", {"1": {"name": "Alice", "result": "L"}}),
        ]
        self.assertEqual(extract_matches(standings), [])

    def test_tie_result_skipped(self):
        standings = [
            _standings_entry("Alice", {"1": {"name": "Bob", "result": "T"}}),
            _standings_entry("Bob", {"1": {"name": "Alice", "result": "T"}}),
        ]
        self.assertEqual(extract_matches(standings), [])

    def test_non_reciprocal_report_skipped(self):
        standings = [
            _standings_entry("Alice", {"1": {"name": "Bob", "result": "W"}}),
            _standings_entry("Bob", {}),  # Bob never reports round 1
        ]
        self.assertEqual(extract_matches(standings), [])

    def test_invalid_roster_drops_match_and_shifts_indices(self):
        standings = [
            _standings_entry("Carol", {"1": {"name": "Dave", "result": "W"}},
                             n_mons=5),  # invalid decklist -> no team
            _standings_entry("Dave", {
                "1": {"name": "Carol", "result": "L"},
                "2": {"name": "Erin", "result": "W"},
            }),
            _standings_entry("Erin", {"2": {"name": "Dave", "result": "L"}}),
        ]
        matches = extract_matches(standings)
        # Carol's match vanishes (no roster); Dave/Erin get indices 0/1 in
        # the post-filter team list, matching extract_teams output.
        self.assertEqual(matches, [Match(round=2, winner=0, loser=1)])
        teams = extract_teams(standings)
        self.assertEqual(len(teams), 2)

    def test_limitless_standings_without_rounds(self):
        standings = [
            {k: v for k, v in _standings_entry("Alice").items() if k != "rounds"},
            {k: v for k, v in _standings_entry("Bob").items() if k != "rounds"},
        ]
        self.assertEqual(extract_matches(standings), [])


class TestMatchRoundTrip(unittest.TestCase):
    def test_matches_survive_save_and_load(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            matches = (Match(round=1, winner=0, loser=1),
                       Match(round=2, winner=1, loser=2))
            tt = TournamentTeams(
                meta=TournamentMeta(id="m1", name="Match Event", date="2026-05-30",
                                    regulation="M-A", players=len(SAMPLE_TEAMS)),
                teams=SAMPLE_TEAMS,
                tournament_type="in-person",
                matches=matches,
            )
            _save_tournament(cache_dir, tt)
            loaded = load_cached_tournaments(cache_dir=cache_dir, regulation="M-A")
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].matches, matches)

    def test_all_match_observations_resolves_rosters(self):
        tt = TournamentTeams(
            meta=TournamentMeta(id="m2", name="Match Event", date="2026-05-30",
                                regulation="M-A", players=len(SAMPLE_TEAMS)),
            teams=SAMPLE_TEAMS,
            tournament_type="in-person",
            matches=(Match(round=3, winner=4, loser=7),),
        )
        obs = all_match_observations([tt])
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].winner, SAMPLE_TEAMS[4].members)
        self.assertEqual(obs[0].loser, SAMPLE_TEAMS[7].members)
        self.assertEqual(obs[0].round, 3)
        self.assertEqual(obs[0].date, "2026-05-30")
        self.assertEqual(obs[0].tournament_id, "m2")

    def test_limitless_entries_have_no_matches(self):
        obs = all_match_observations([
            _make_tournament("t", "Online Cup", "2026-02-01", "M-A", SAMPLE_TEAMS),
        ])
        self.assertEqual(obs, [])


class TestCacheVersionGate(unittest.TestCase):
    """v4 widened the member schema (the ability track) for BOTH sources, so any
    pre-v4 cache entry -- regardless of type -- is stale and must be re-fetched /
    re-imported."""

    def _old_payload(self, entry_type: str, version: int = 3) -> dict:
        return {
            "version": version,
            "type": entry_type,
            "meta": {"id": f"old-{entry_type}", "name": "Old Event",
                     "date": "2026-01-01", "regulation": "M-A", "players": 40},
            "teams": [
                {"members": [list(m) for m in sorted(t.members, key=lambda m: m[0])],
                 "placing": t.placing, "record": [t.wins, t.losses, t.ties]}
                for t in SAMPLE_TEAMS
            ],
        }

    def test_pre_v4_limitless_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            with (cache_dir / "old.json").open("w") as f:
                json.dump(self._old_payload("limitless"), f)
            loaded = load_cached_tournaments(cache_dir=cache_dir, regulation="M-A")
            self.assertEqual(loaded, [])

    def test_pre_v4_in_person_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            with (cache_dir / "old.json").open("w") as f:
                json.dump(self._old_payload("in-person"), f)
            loaded = load_cached_tournaments(cache_dir=cache_dir, regulation="M-A")
            self.assertEqual(loaded, [])

    def test_outdated_in_person_reimported(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            local_dir = base / "local"
            cache_dir = base / "cache"
            cache_dir.mkdir()
            reg_dir = local_dir / "M-A"
            reg_dir.mkdir(parents=True)

            standings = [
                _standings_entry("Alice", {"1": {"name": "Bob", "result": "W"}}),
                _standings_entry("Bob", {"1": {"name": "Alice", "result": "L"}}),
            ]
            with (reg_dir / "ev9_2026-05-30.json").open("w") as f:
                json.dump(standings, f)

            stale = self._old_payload("in-person")
            stale["meta"]["id"] = "local_ev9_2026-05-30"
            with (cache_dir / "local_ev9_2026-05-30.json").open("w") as f:
                json.dump(stale, f)

            result = import_in_person_tournaments(local_dir=local_dir, cache_dir=cache_dir)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].matches,
                             (Match(round=1, winner=0, loser=1),))


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
