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

Cache strategy: per-tournament parsed team list only -- abilities, moves,
tera, player names, and drops are discarded at parse time. Each team's
final `placing` and swiss/bracket `record` (wins/losses/ties) ARE kept, to
support outcome validation (does coupling coherence predict placement?).
Each team member is preserved as a `(species, item)` tuple; items are kept
because they (a) re-introduce held-item forme distinctions invisible in the
species-only namespace, and (b) enable the Phase 3 item-pair Ising fit in
`app.py:load_model_phase3`. Tournaments are immutable once finished, so the
cache is write-once / read-many. Rejected tournaments are not cached;
re-runs will re-fetch and re-reject. Cached entries are still subject to
the name pre-check on load, but cannot be re-validated against the Protect
heuristic (we don't keep decklists). If MIN_PROTECT_PER_PLAYER is tightened,
clear `tournaments_cache/` to force a clean re-validation.

Cache format versioning: each payload includes a `version` field. Loading a
cached file at version < CACHE_VERSION returns None, forcing a re-fetch; old
formats are not upgraded in place. Bump CACHE_VERSION whenever the parsing
logic changes in a way that would yield different team rosters than what's
in cached files (item normalization, species canonicalization, etc.) so old
caches are invalidated automatically.
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

from .constants import (
    MIN_TEAMS_PER_TOURNAMENT,
    PHASE2_MIN_TEAMS as DEFAULT_MIN_TEAMS,
    TEAM_SIZE,
)

API_BASE = "https://play.limitlesstcg.com/api"
DEFAULT_REGULATION = "M-A"
DEFAULT_GAME = "VGC"
DEFAULT_CACHE_DIR = Path("tournaments_cache")
MIN_PROTECT_PER_PLAYER = 1.0
POLITE_SLEEP_SEC = 2.0
CACHE_VERSION = 2  # bumped when team payload schema changes; see module docstring
                   # v2: each team carries placing + win/loss/tie record


@dataclass(frozen=True)
class TournamentMeta:
    """Tournament identifiers preserved alongside the cached team list."""

    id: str
    name: str
    date: str
    regulation: str
    players: int


@dataclass(frozen=True)
class Team:
    """A single player's roster plus tournament outcome.

    `members` is the model feature: a frozenset of `(species, item)` tuples
    (`item` is None for itemless mons). The frozenset enforces that no two
    members share the exact same (species, item) pair -- which the upstream
    data should never produce anyway, since such a team would be illegal.

    `placing` is the final standing (1 = winner) when the API reports it,
    else None -- some events report it for top cut only, or not at all.
    `wins`/`losses`/`ties` are the swiss/bracket record, present for every
    standings entry observed, so they are the robust outcome signal to fall
    back on when `placing` is missing."""

    members: frozenset[tuple[str, str | None]]
    placing: int | None
    wins: int
    losses: int
    ties: int


@dataclass(frozen=True)
class TournamentTeams:
    """A tournament's parsed teams (rosters + outcomes) and identifying metadata."""

    meta: TournamentMeta
    teams: list[Team]


_MEGA_FORME_SUFFIXES = (" X", " Y", " Z")

# Species whose Limitless API name appears inconsistently and must be
# collapsed to a single canonical form before entering the corpus vocab.
_SPECIES_ALIASES: dict[str, str] = {
    # The base Floette form is unobtainable; the Eternal Flower variant is the
    # only one that sees tournament play. Both names appear in the raw API.
    "Floette": "Eternal Flower Floette",
}


def strip_mega_prefix(species: str) -> str:
    """Collapse 'Mega <Base>' / 'Mega <Base> X/Y/Z' species names down to
    '<Base>'. Players are inconsistent about whether they prefix mega-evolved
    species in the Limitless decklist; the held Mega Stone is the source of
    truth for which mega forme. By stripping the prefix here we bucket
    'Mega Blastoise @ Blastoisite' with 'Blastoise @ Blastoisite' (same
    forme, same model feature), and 'Mega Charizard Y @ Charizardite Y' with
    'Charizard @ Charizardite Y'. The trailing forme letter is stripped only
    when the 'Mega ' prefix is present, to avoid mangling any non-mega species
    name that happens to end in a stray letter. ' Z' is included for
    future-proofing (e.g. 'Mega Lucario Z' -> 'Lucario').
    """
    if not species.startswith("Mega "):
        return species
    stripped = species[len("Mega "):]
    for suffix in _MEGA_FORME_SUFFIXES:
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
            break
    return stripped


def normalize_name(s: str | None) -> str | None:
    """Canonical form for a Limitless-API species or item string.

    Collapses case/whitespace variants seen in the raw API ('Sitrus Berry',
    'Sitrus berry', 'sitrus berry') into a single bucket. Goal is internal
    consistency within the corpus, not matching any external canonical
    source. Returns None for None or empty inputs.

    Lowercases, then title-cases each whitespace-separated word, then each
    hyphen/apostrophe-separated subpart, so multi-word names and hyphenated
    formes both survive ('Tapu Koko', 'Charizard-Mega-Y', "Farfetch'd").
    """
    if not s:
        return None
    s = s.strip().lower()
    if not s:
        return None
    # Word boundaries are whitespace and hyphens; apostrophes stay literal
    # within a part so "Farfetch'd" / "Sirfetch'd" round-trip correctly.
    return " ".join(
        "-".join(sub.capitalize() for sub in part.split("-"))
        for part in s.split()
    )


def _http_get_json(url: str) -> object:
    """Fetch a URL and parse the response body as JSON."""
    req = urllib.request.Request(url, headers={"User-Agent": "k2dex/0.1"})
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


def extract_teams(standings: list[dict]) -> list[Team]:
    """Parse standings into a list of `Team` records (roster + outcome).

    Drops players whose decklist isn't exactly TEAM_SIZE entries with distinct
    species (incomplete submissions, drops before lock-in, or the game's
    no-duplicate-species rule violated by malformed data). Names come from
    the `name` field of each decklist entry; items from `item` (None when
    missing/null). Both species and item are passed through `normalize_name`
    so case/whitespace variants of the same name collapse to one vocab entry
    -- the Limitless API has been observed to return e.g. 'Sitrus Berry',
    'Sitrus berry', and 'sitrus berry' for the same item across tournaments.

    `placing` and the win/loss/tie `record` are read straight from the
    standings entry; placing is None when the API doesn't report it, and a
    missing record defaults to 0-0-0 (no signal, filterable downstream).
    """
    teams: list[Team] = []
    for entry in standings:
        decklist = entry.get("decklist") or []
        members: list[tuple[str, str | None]] = []
        for mon in decklist:
            name = normalize_name(mon.get("name"))
            if not name:
                continue
            name = strip_mega_prefix(name)
            name = _SPECIES_ALIASES.get(name, name)
            item = normalize_name(mon.get("item"))
            members.append((name, item))
        if len(members) != TEAM_SIZE:
            continue
        # Reject teams with duplicate species (game rule)
        if len({species for species, _ in members}) != TEAM_SIZE:
            continue
        placing = entry.get("placing")
        record = entry.get("record") or {}
        teams.append(Team(
            members=frozenset(members),
            placing=placing if isinstance(placing, int) else None,
            wins=int(record.get("wins", 0)),
            losses=int(record.get("losses", 0)),
            ties=int(record.get("ties", 0)),
        ))
    return teams


def _cache_path(cache_dir: Path, tournament_id: str) -> Path:
    return cache_dir / f"{tournament_id}.json"


def _save_tournament(cache_dir: Path, t: TournamentTeams) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "meta": {
            "id": t.meta.id,
            "name": t.meta.name,
            "date": t.meta.date,
            "regulation": t.meta.regulation,
            "players": t.meta.players,
        },
        # Each team: members (list of [species, item_or_null], sorted by
        # species for reproducible diffs across re-saves) plus outcome.
        "teams": [
            {
                "members": [
                    list(member)
                    for member in sorted(team.members, key=lambda m: m[0])
                ],
                "placing": team.placing,
                "record": [team.wins, team.losses, team.ties],
            }
            for team in t.teams
        ],
    }
    with _cache_path(cache_dir, t.meta.id).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _load_tournament(cache_dir: Path, tournament_id: str) -> TournamentTeams | None:
    """Load a cached tournament, or return None if missing / outdated schema.

    Returning None signals the caller to re-fetch from the API. Older cached
    files (v1: no placing/record; pre-v1: no items) are incompatible with the
    current schema, so we treat them as cache misses.
    """
    p = _cache_path(cache_dir, tournament_id)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("version", 1) < CACHE_VERSION:
        return None  # outdated schema; force re-fetch
    meta = TournamentMeta(**payload["meta"])
    teams = [
        Team(
            members=frozenset((m[0], m[1]) for m in team["members"]),
            placing=team["placing"],
            wins=team["record"][0],
            losses=team["record"][1],
            ties=team["record"][2],
        )
        for team in payload["teams"]
    ]
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
    hit or miss) or by Protect-ratio heuristic (fresh fetches only); skips
    tournaments smaller than MIN_TEAMS_PER_TOURNAMENT (pre-fetch on registered
    player count and post-extract on actual team count -- the second catches
    high-dropout small events and re-filters cached tournaments if the
    threshold is tightened, without needing a cache wipe); serves cached
    tournaments from disk if present, hits the API otherwise. Returns once
    the running team total meets the cutoff.

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
            # Pre-fetch size check: registered players bound the team count
            # from above (teams <= players always), so skipping here avoids
            # API calls for events too small to clear the threshold.
            if tm.players < MIN_TEAMS_PER_TOURNAMENT:
                logger.info(
                    "[skip small]   %s '%s' (players=%d < %d)",
                    tm.id, tm.name, tm.players, MIN_TEAMS_PER_TOURNAMENT,
                )
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
            # Post-extract size check: catches cached tournaments under a
            # newly-tightened threshold and high-dropout events where
            # extract_teams trimmed below the limit.
            if len(ttd.teams) < MIN_TEAMS_PER_TOURNAMENT:
                logger.info(
                    "[skip small]   %s '%s' (teams=%d < %d after extract)",
                    tm.id, tm.name, len(ttd.teams), MIN_TEAMS_PER_TOURNAMENT,
                )
                continue
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


def all_teams(tournaments: list[TournamentTeams]) -> list[frozenset[tuple[str, str | None]]]:
    """Flatten ingested tournaments into a single list of roster frozensets.

    Projects away the outcome (placing/record); this is the model-feature view
    consumed by the fitting path. Use `all_teams_with_outcomes` when you need
    placement for outcome validation.
    """
    return [team.members for t in tournaments for team in t.teams]


def all_teams_with_outcomes(tournaments: list[TournamentTeams]) -> list[Team]:
    """Flatten ingested tournaments into a single list of `Team` records.

    Same flattening as `all_teams`, but keeps each team's placing and
    win/loss/tie record alongside its roster -- the input for outcome
    validation (coherence/Score vs. placement).
    """
    return [team for t in tournaments for team in t.teams]


def chronological_split(
    tournaments: list[TournamentTeams],
    train_frac: float,
) -> tuple[list[TournamentTeams], list[TournamentTeams]]:
    """Split tournaments chronologically: oldest *train_frac* of teams → train.

    Tournaments are sorted oldest-first by date. The split boundary falls at
    the tournament edge nearest to *train_frac* of total teams.  Returns
    ``(train_tournaments, test_tournaments)`` both in oldest-first order.
    """
    chrono = sorted(tournaments, key=lambda t: t.meta.date)
    total_teams = sum(len(t.teams) for t in chrono)
    target = int(round(total_teams * train_frac))

    cumulative = 0
    split_idx = len(chrono)
    for i, t in enumerate(chrono):
        cumulative += len(t.teams)
        if cumulative >= target:
            split_idx = i + 1
            break

    return chrono[:split_idx], chrono[split_idx:]


def species_only_teams(
    teams: list[frozenset[tuple[str, str | None]]],
) -> list[frozenset[str]]:
    """Project (species, item) teams down to species-only frozensets.

    Used by `app.py:load_model_phase2` and the validation harness to consume
    the Phase 2 species-only namespace from the new v2 cache format.
    """
    return [frozenset(species for species, _ in team) for team in teams]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tournaments = ingest()
    teams = all_teams(tournaments)
    print(f"Tournaments: {len(tournaments)}")
    print(f"Total teams: {len(teams)}")
    if teams:
        sample = next(iter(teams))
        print(f"Sample team:")
        for species, item in sorted(sample, key=lambda m: m[0]):
            print(f"  {species:<25} item: {item}")
