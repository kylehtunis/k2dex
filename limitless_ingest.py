"""Limitless TCG API ingestor for VGC tournament team data.

Phase 2 data source. Walks the Limitless tournaments endpoint newest-first
(across all catalog pages as needed), fetches standings for tournaments
matching the target regulation, applies singles-tournament filters (the API
labels both doubles VGC and singles VGC events as `game=VGC` with identical
regulation tags), and extracts team rosters (six Pokemon per team) for
downstream inverse Ising fitting via pseudo-likelihood.

Singles filter (two-stage, cheap-first):

1. **Name pre-check**: skip tournaments whose name contains "singles"
   (case-insensitive). Avoids a standings fetch for the obvious cases.
2. **Protect ratio post-check**: after fetching standings, count occurrences
   of "Protect" across all decklisted moves and require >= MIN_PROTECT_PER_PLAYER
   per player. Protect is a doubles staple (2-4 per team typical) but rare
   in singles (0-1 per team), so the per-player rate cleanly separates them
   without depending on the tournament name convention.

Cache strategy: per-tournament parsed team list only -- items, abilities,
moves, tera, player names, records, drops are all discarded at parse time.
Tournaments are immutable once finished, so the cache is write-once /
read-many. Rejected tournaments are not cached; re-runs will re-fetch and
re-reject. Cached entries are still subject to the name pre-check on load,
but cannot be re-validated against the Protect heuristic (we don't keep
decklists). If MIN_PROTECT_PER_PLAYER is tightened, clear `tournaments_cache/`
to force a clean re-validation.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("limitless_ingest")

API_BASE = "https://play.limitlesstcg.com/api"
TEAM_SIZE = 6
DEFAULT_REGULATION = "M-A"
DEFAULT_GAME = "VGC"
DEFAULT_CACHE_DIR = Path("tournaments_cache")
DEFAULT_MIN_TEAMS = 5000
MIN_PROTECT_PER_PLAYER = 1.0
POLITE_SLEEP_SEC = 2.0


@dataclass(frozen=True)
class TournamentMeta:
    """Tournament identifiers preserved alongside the cached team list."""

    id: str
    name: str
    date: str
    regulation: str
    players: int


@dataclass(frozen=True)
class TournamentTeams:
    """A tournament's parsed team rosters and minimal identifying metadata."""

    meta: TournamentMeta
    teams: list[frozenset[str]]


def _http_get_json(url: str) -> object:
    """Fetch a URL and parse the response body as JSON."""
    req = urllib.request.Request(url, headers={"User-Agent": "k2dex-science/0.1"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def list_tournaments_page(game: str = DEFAULT_GAME, page: int = 1) -> list[TournamentMeta]:
    """Fetch one page of the Limitless tournaments catalog (50 entries max).
    Returns an empty list when the page is past the end of the catalog."""
    try:
        raw = _http_get_json(f"{API_BASE}/tournaments?game={game}&page={page}")
    except urllib.error.HTTPError as err:
        logger.error("HTTP error on page %d: %s", page, err)
        logger.error("aborting catalog walk -- retry later or investigate API changes")
        raise
    if not isinstance(raw, list):
        raise ValueError(f"expected list from /tournaments, got {type(raw).__name__}")
    return [
        TournamentMeta(
            id=entry["id"],
            name=entry["name"],
            date=entry["date"],
            regulation=entry["format"],
            players=int(entry["players"]),
        )
        for entry in raw
    ]


def iter_catalog_pages(game: str = DEFAULT_GAME) -> Iterator[list[TournamentMeta]]:
    """Lazily yield catalog pages (50 entries each), newest first, until the
    API returns an empty page. Caller is responsible for ID-deduping across
    pages and for deciding when the relevant region of the catalog is
    exhausted (see `ingest` for the format-aware termination logic)."""
    page = 1
    while True:
        batch = list_tournaments_page(game=game, page=page)
        if not batch:
            return
        yield batch
        page += 1


def fetch_standings(tournament_id: str) -> list[dict]:
    """Fetch raw standings JSON for a tournament."""
    raw = _http_get_json(f"{API_BASE}/tournaments/{tournament_id}/standings")
    if not isinstance(raw, list):
        raise ValueError(f"expected list from /standings, got {type(raw).__name__}")
    return raw


def is_likely_singles_by_name(name: str) -> bool:
    """Cheap pre-check: the tournament name explicitly says 'singles'."""
    return "singles" in name.lower()


def passes_doubles_protect_check(
    standings: list[dict],
    min_ratio: float = MIN_PROTECT_PER_PLAYER,
) -> bool:
    """Heuristic: count Protect occurrences across all decklists and require
    at least `min_ratio` per player. Doubles teams average 2-4 Protect users
    per roster; singles teams average 0-1. Substring match is case-insensitive
    in case the API's capitalization ever drifts.
    """
    if not standings:
        return False
    protect_count = 0
    for entry in standings:
        for mon in entry.get("decklist") or []:
            for move in mon.get("attacks") or []:
                if move and "protect" in move.lower():
                    protect_count += 1
    return protect_count / len(standings) >= min_ratio


def extract_teams(standings: list[dict]) -> list[frozenset[str]]:
    """Parse standings into a list of teams (frozensets of Pokemon names).

    Drops players whose decklist isn't exactly TEAM_SIZE distinct Pokemon
    (incomplete submissions, drops before lock-in, or rare data anomalies).
    Names come from the `name` field of each decklist entry, preserving the
    Limitless capitalization convention -- vocab will be built directly from
    these strings downstream, so no cross-source normalization is needed.
    """
    teams: list[frozenset[str]] = []
    for entry in standings:
        decklist = entry.get("decklist") or []
        names = [mon["name"] for mon in decklist if mon.get("name")]
        if len(names) != TEAM_SIZE:
            continue
        team = frozenset(names)
        if len(team) != TEAM_SIZE:
            continue
        teams.append(team)
    return teams


def _cache_path(cache_dir: Path, tournament_id: str) -> Path:
    return cache_dir / f"{tournament_id}.json"


def _save_tournament(cache_dir: Path, t: TournamentTeams) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "id": t.meta.id,
            "name": t.meta.name,
            "date": t.meta.date,
            "regulation": t.meta.regulation,
            "players": t.meta.players,
        },
        "teams": [sorted(team) for team in t.teams],
    }
    with _cache_path(cache_dir, t.meta.id).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _load_tournament(cache_dir: Path, tournament_id: str) -> TournamentTeams | None:
    p = _cache_path(cache_dir, tournament_id)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    meta = TournamentMeta(**payload["meta"])
    teams = [frozenset(t) for t in payload["teams"]]
    return TournamentTeams(meta=meta, teams=teams)


def ingest(
    *,
    min_teams: int = DEFAULT_MIN_TEAMS,
    regulation: str = DEFAULT_REGULATION,
    game: str = DEFAULT_GAME,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[TournamentTeams]:
    """Ingest recent tournaments newest-first until >= min_teams teams accumulated.

    Walks `iter_catalog_pages` lazily across catalog pages; filters to the
    target regulation; skips tournaments flagged as singles by name (cache
    hit or miss) or by Protect-ratio heuristic (fresh fetches only); serves
    cached tournaments from disk if present, hits the API otherwise. Returns
    once the running team total meets the cutoff.

    Termination conditions (whichever fires first):
    - `total_teams >= min_teams` (target hit -- normal stop)
    - The API returns an empty page (catalog truly exhausted)
    - A whole page yields zero new in-regulation tournament IDs. Limitless
      is date-sorted newest-first, so once we walk past the start of the
      target regulation we'll never see another instance of it -- no point
      paging back through years of older formats.
    """
    cache_dir = Path(cache_dir)
    out: list[TournamentTeams] = []
    total_teams = 0
    seen_ids: set[str] = set()

    for batch in iter_catalog_pages(game=game):
        new_in_regulation = 0
        for tm in batch:
            if tm.id in seen_ids:
                continue
            seen_ids.add(tm.id)
            if tm.regulation != regulation:
                continue
            new_in_regulation += 1
            if is_likely_singles_by_name(tm.name):
                logger.info("[skip name]    %s '%s'", tm.id, tm.name)
                continue
            cached = _load_tournament(cache_dir, tm.id)
            if cached is not None:
                ttd = cached
                source = "cached "
            else:
                try:
                    standings = fetch_standings(tm.id)
                except urllib.error.HTTPError as err:
                    logger.warning("HTTP error on %s: %s -- skipping", tm.id, err)
                    continue
                if not passes_doubles_protect_check(standings):
                    logger.info("[skip protect] %s '%s'", tm.id, tm.name)
                    time.sleep(POLITE_SLEEP_SEC)
                    continue
                teams = extract_teams(standings)
                ttd = TournamentTeams(meta=tm, teams=teams)
                _save_tournament(cache_dir, ttd)
                source = "fetched"
                time.sleep(POLITE_SLEEP_SEC)
            out.append(ttd)
            total_teams += len(ttd.teams)
            logger.info(
                "[%5d / %d] %s %s '%s' (+%d teams)",
                total_teams, min_teams, source, tm.id, tm.name, len(ttd.teams),
            )
            if total_teams >= min_teams:
                logger.info("ingested %d tournaments, %d teams total", len(out), total_teams)
                return out
        if new_in_regulation == 0:
            logger.info(
                "page yielded no new %s tournaments -- catalog region exhausted",
                regulation,
            )
            break

    logger.info("ingested %d tournaments, %d teams total", len(out), total_teams)
    return out


def all_teams(tournaments: list[TournamentTeams]) -> list[frozenset[str]]:
    """Flatten ingested tournaments into a single team list."""
    return [team for t in tournaments for team in t.teams]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tournaments = ingest()
    teams = all_teams(tournaments)
    print(f"Tournaments: {len(tournaments)}")
    print(f"Total teams: {len(teams)}")
    if teams:
        sample = next(iter(teams))
        print(f"Sample team: {sorted(sample)}")
