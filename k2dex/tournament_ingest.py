"""Tournament team data ingestor and cache for VGC inverse Ising fitting.

Two data sources feed a unified cache (``tournaments_cache/``):

1. **Limitless API** -- online fetch via ``fetch_limitless_tournaments()``.
   Walks the Limitless tournaments endpoint newest-first, fetches standings
   for tournaments matching the target regulation, applies singles-tournament
   filters. Only triggered manually via the CLI.

2. **In-person tournament exports** -- offline import via
   ``import_in_person_tournaments()``. Reads raw Limitless-format standings
   JSON files from ``tournament_json/[format]/[id]_[date].json`` and
   normalizes them into cache entries.

Downstream consumers (loaders, notebooks, precompute) call only
``load_cached_tournaments()``, which is a pure offline cache reader.

Singles filter (Limitless API path only, two-stage, cheap-first):

1. **Name pre-check**: skip tournaments whose name contains "singles"
   (case-insensitive). Avoids a standings fetch for the obvious cases.
2. **Protect ratio post-check**: after fetching standings, count occurrences
   of "Protect" across all decklisted moves and require >= MIN_PROTECT_PER_PLAYER
   per player. Protect is a doubles staple (2-4 per team typical) but rare
   in singles (0-1 per team), so the per-player rate cleanly separates them
   without depending on the tournament name convention.

Cache strategy: per-tournament parsed team list only -- abilities, moves,
tera, player names, and drops are discarded at parse time. Each team's
final ``placing`` and swiss/bracket ``record`` (wins/losses/ties) ARE kept,
to support outcome validation. Each team member is preserved as a
``(species, item)`` tuple. Tournaments are immutable once finished, so the
cache is write-once / read-many. Each cache entry carries a ``type`` field
(``"limitless"`` or ``"in-person"``) for provenance tracking; old cache
files without this field default to ``"limitless"`` on read.

In-person entries additionally carry per-round **match results**: each
standings entry's ``rounds`` dict (opponent name + W/L result per round)
is resolved to ``Match(round, winner, loser)`` records whose winner/loser
are indices into the cached team list. Player names are used only during
parsing for opponent resolution and are still discarded. The Limitless API
exposes no match-level data, so limitless entries have an empty match list.

Cache format versioning: each payload includes a ``version`` field. Loading
a cached file below the minimum version for its type returns None, forcing
a re-fetch / re-import; old formats are not upgraded in place. Bump
CACHE_VERSION whenever the parsing logic changes in a way that would yield
different payloads than what's in cached files. The minimum is per-type
(``_MIN_CACHE_VERSION``) so that a bump which only adds in-person data
(e.g. v3's match lists) does not force a multi-hour re-walk of the
Limitless API for entries whose payload would be byte-identical.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import urllib.error
import urllib.request
import warnings
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

logger = logging.getLogger("tournament_ingest")

from .constants import (
    DEFAULT_LOCAL_DIR,
    LIMITLESS_MAX_TEAMS as DEFAULT_MAX_TEAMS,
    MIN_TEAMS_PER_TOURNAMENT,
    TEAM_SIZE,
    CURRENT_REGULATION as DEFAULT_REGULATION,
)

API_BASE = "https://play.limitlesstcg.com/api"
DEFAULT_GAME = "VGC"
DEFAULT_CACHE_DIR = Path("tournaments_cache")
MIN_PROTECT_PER_PLAYER = 1.0
POLITE_SLEEP_SEC = 2.0
CACHE_VERSION = 3  # bumped when payload schema changes; see module docstring
                   # v2: each team carries placing + win/loss/tie record
                   # v3: in-person entries carry per-round match results

# Minimum acceptable cached version per entry type. v3 only added match
# lists, which the Limitless API cannot provide -- a v2 limitless entry is
# byte-equivalent to what a v3 re-fetch would write, so v2 stays valid
# there. In-person entries must be re-imported to pick up their matches.
_MIN_CACHE_VERSION: dict[str, int] = {"limitless": 2, "in-person": 3}


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
class Match:
    """A decisive game between two teams of the same tournament.

    `winner` / `loser` index into the parent `TournamentTeams.teams` list
    (the post-filter order produced by `extract_teams`). `round` is the
    1-based round number across the whole event -- day-1 swiss, day-2 swiss,
    and top cut share one numbering, so notebooks can filter phases by
    round range. Only decisive results are kept: byes, ties, and irregular
    reports (e.g. double losses) are dropped at parse time."""

    round: int
    winner: int
    loser: int


@dataclass(frozen=True)
class TournamentTeams:
    """A tournament's parsed teams (rosters + outcomes) and identifying metadata.

    `tournament_type` is the provenance of the entry: `"limitless"` (online,
    fetched from the API) or `"in-person"` (imported from tournament_json/).
    `matches` holds per-round match results when the source exposes them
    (in-person exports only; empty for limitless entries).
    """

    meta: TournamentMeta
    teams: list[Team]
    tournament_type: str = "limitless"
    matches: tuple[Match, ...] = ()


@dataclass(frozen=True)
class TeamObservation:
    """A roster plus the per-team provenance used for sample weighting.

    The flattened counterpart of `Team` that keeps the parent tournament's
    date and source type (which `all_teams` projects away). Consumed by
    `loaders.team_weights` to compute recency / in-person fit weights.
    """

    members: frozenset[tuple[str, str | None]]
    date: str             # parent tournament date, ISO YYYY-MM-DD
    tournament_type: str  # "limitless" or "in-person"


@dataclass(frozen=True)
class MatchObservation:
    """A decisive match resolved to its two rosters, plus provenance.

    The flattened counterpart of `Match`: winner/loser indices are resolved
    to the actual roster frozensets, and the parent tournament's date and id
    are kept. The input for Bradley-Terry fitting and match-level outcome
    analysis."""

    winner: frozenset[tuple[str, str | None]]
    loser: frozenset[tuple[str, str | None]]
    round: int
    date: str
    tournament_id: str


_MEGA_FORME_SUFFIXES = (" X", " Y", " Z")

# Species whose Limitless API name appears inconsistently and must be
# collapsed to a single canonical form before entering the corpus vocab.
_SPECIES_ALIASES: dict[str, str] = {
    # The base Floette form is unobtainable; the Eternal Flower variant is the
    # only one that sees tournament play. Both names appear in the raw API.
    "Floette": "Eternal Flower Floette",
}

_ITEM_ALIASES: dict[str, str] = {
    "Dread Plate": "Black Glasses",
    "Glimmorite": "Glimmoranite",
}

# Item strings that mean "no held item". The raw API uses several spellings
# (plus a genuinely absent field); all collapse to Python ``None`` so a single
# itemless bucket forms downstream. Values are the ``normalize_name`` (title-
# cased) forms, since normalization runs before this lookup.
_ITEMLESS_SPELLINGS: frozenset[str] = frozenset({
    "None", "No Item", "Nothing", "Null",
})

_ILLEGAL_ITEMS_BY_REGULATION: dict[str, frozenset[str]] = {
    "M-A": frozenset({
        "Covert Cloak",
        "Assault Vest",
        "Safety Goggles",
    }),
    "M-B": frozenset({
        "Covert Cloak",
        "Assault Vest",
        "Safety Goggles",
    }),
}


@dataclass(frozen=True)
class RegulationRelabel:
    """Recovers a regulation's true label when the Limitless catalog has not
    been updated to tag it yet.

    Limitless has been observed to keep serving a superseded format label (or
    ``CUSTOM``) for events that, by date and roster content, belong to a newer
    regulation. A tournament is relabeled to ``target`` when all three hold:

    - its catalog date is on or after ``start_date`` (day resolution),
    - its catalog label is one of ``from_labels`` (the stale labels), and
    - at least one roster runs a species in ``marker_species`` or item in
      ``marker_items`` -- picks that are legal only in ``target`` and illegal
      in every ``from_labels`` format, so a single appearance rules the stale
      label out.

    Marker presence is necessary and sufficient: a date/label candidate whose
    rosters carry no marker keeps its original label.
    """

    target: str
    start_date: str            # inclusive lower bound, ISO YYYY-MM-DD
    from_labels: frozenset[str]
    marker_species: frozenset[str]
    marker_items: frozenset[str]


# Limitless had not tagged Regulation M-B as of its 2026-06-17 start: events
# that are truly M-B still carry the M-A or CUSTOM label. Recover them by date
# window plus a roster content check. Each marker species/item is legal in M-B
# but illegal in M-A, so any single appearance identifies the event as M-B.
_REGULATION_RELABELS: tuple[RegulationRelabel, ...] = (
    RegulationRelabel(
        target="M-B",
        start_date="2026-06-17",
        from_labels=frozenset({"M-A", "CUSTOM"}),
        marker_species=frozenset({"Gholdengo", "Metagross", "Grimmsnarl"}),
        marker_items=frozenset({"Life Orb", "Light Clay"}),
    ),
)


def _day(date_str: str) -> str:
    """Day-resolution key for a catalog date. Limitless dates are full ISO
    timestamps, in-person dates bare YYYY-MM-DD; both share a sortable
    YYYY-MM-DD prefix, so a string compare orders them at day resolution."""
    return (date_str or "")[:10]


def _roster_has_marker(teams: list[Team], rule: RegulationRelabel) -> bool:
    """True when any team runs one of the rule's marker species or items."""
    return any(
        species in rule.marker_species
        or (item is not None and item in rule.marker_items)
        for team in teams
        for species, item in team.members
    )


def resolve_regulation(meta: TournamentMeta, teams: list[Team]) -> str:
    """The tournament's true regulation, correcting stale Limitless labels.

    Returns ``meta.regulation`` unchanged unless a relabel rule's date window,
    source label, and roster-marker conditions all match, in which case the
    rule's target is returned. The marker scan runs on ``teams``, so this must
    be called only after standings are parsed. Label/date are checked first,
    so non-candidates short-circuit without touching the rosters.
    """
    for rule in _REGULATION_RELABELS:
        if meta.regulation in rule.from_labels and _day(meta.date) >= rule.start_date:
            if _roster_has_marker(teams, rule):
                return rule.target
    return meta.regulation


def could_match_regulation(meta: TournamentMeta, target: str) -> bool:
    """Cheap, standings-free pre-filter: could this tournament resolve to
    ``target``? True when it already carries that label, or when a relabel
    rule could promote it (date window + source label match). The roster-marker
    confirmation happens later, once standings are fetched.
    """
    if meta.regulation == target:
        return True
    return any(
        rule.target == target
        and meta.regulation in rule.from_labels
        and _day(meta.date) >= rule.start_date
        for rule in _REGULATION_RELABELS
    )


def _has_pending_relabel(meta: TournamentMeta) -> bool:
    """Whether any relabel rule's label/date precondition matches, i.e. the
    roster must be inspected before this event's regulation can be trusted."""
    return any(
        meta.regulation in rule.from_labels and _day(meta.date) >= rule.start_date
        for rule in _REGULATION_RELABELS
    )


_BRACKET_RE = re.compile(r"^(.+?)\s*\[(.+)]\s*$")

_COSMETIC_FORMES = frozenset({
    "family of four", "family of three",
    "unremarkable form", "masterpiece form",
    "phony form", "antique form",
})

_REGIONAL_PREFIXES = frozenset({"hisuian", "alolan", "galarian", "paldean"})


def normalize_bracket_forme(name: str) -> str:
    """Convert bracket-notation formes to Limitless prefix-style names.

    In-person tournament data uses ``Species [Forme]`` notation (e.g.
    ``Arcanine [Hisuian Form]``) while the Limitless API uses prefix-style
    names (``Hisuian Arcanine``). This normalizes to the Limitless convention
    so both data sources produce identical vocab entries.

    Must be called BEFORE ``normalize_name``, because ``normalize_name``'s
    ``.capitalize()`` fails on bracket-prefixed tokens (``[hisuian`` stays
    lowercase since ``[`` is not a letter).
    """
    if not name or "[" not in name:
        return name

    match = _BRACKET_RE.match(name)
    if not match:
        return name

    base = match.group(1).strip()
    bracket = match.group(2).strip()
    bracket_lower = bracket.lower()

    if bracket_lower in _COSMETIC_FORMES:
        return base

    if bracket_lower == "male":
        return base
    if bracket_lower == "female":
        return f"{base} ♀"

    if base.lower() == "rotom" and "rotom" in bracket_lower:
        return bracket

    for region in _REGIONAL_PREFIXES:
        if bracket_lower.startswith(f"{region} form"):
            region_title = region.capitalize()
            remainder = bracket[len(f"{region} form"):].strip()
            if remainder.startswith("- "):
                breed = remainder[2:].strip()
                return f"{region_title} {base} {breed}"
            return f"{region_title} {base}"

    if bracket_lower == "eternal flower":
        return f"Eternal Flower {base}"

    if bracket_lower.endswith(" form"):
        forme_word = bracket[:-5].strip()
        return f"{base} {forme_word}"

    return name


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


def iter_catalog_pages(
    game: str = DEFAULT_GAME,
    start_page: int = 1,
) -> Iterator[list[TournamentMeta]]:
    """Lazily yield catalog pages (50 entries each), newest first, until the
    API returns an empty page. ``start_page`` lets callers skip past recent
    pages to reach older tournaments (useful for fetching previous formats).
    Caller is responsible for ID-deduping across pages and for deciding when
    the relevant region of the catalog is exhausted."""
    page = start_page
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


def _parse_members(
    decklist: list[dict],
    illegal_items: frozenset[str],
) -> frozenset[tuple[str, str | None]] | None:
    """Parse one standings entry's decklist into a roster frozenset.

    Returns None when the decklist isn't exactly TEAM_SIZE entries with
    distinct species (incomplete submissions, drops before lock-in, or the
    game's no-duplicate-species rule violated by malformed data). Shared by
    `extract_teams` and `extract_matches` so both produce the same valid-entry
    filtering -- match winner/loser indices stay aligned with the team list.
    """
    members: list[tuple[str, str | None]] = []
    for mon in decklist:
        raw_name = mon.get("name") or ""
        name = normalize_name(normalize_bracket_forme(raw_name))
        if not name:
            continue
        name = strip_mega_prefix(name)
        name = _SPECIES_ALIASES.get(name, name)
        item = normalize_name(mon.get("item"))
        if item in _ITEMLESS_SPELLINGS:
            item = None
        if item:
            item = _ITEM_ALIASES.get(item, item)
            if item in illegal_items:
                item = None
        members.append((name, item))
    if len(members) != TEAM_SIZE:
        return None
    # Reject teams with duplicate species (game rule)
    if len({species for species, _ in members}) != TEAM_SIZE:
        return None
    # Reject teams with no item reported on any member: a complete VGC teamsheet
    # always lists items, so an all-itemless roster is a species-only submission
    # (items simply weren't captured), not six mons genuinely holding nothing.
    # Kept out of the corpus so unreported items don't masquerade as itemless.
    if all(item is None for _, item in members):
        return None
    return frozenset(members)


def extract_teams(
    standings: list[dict],
    *,
    regulation: str | None = None,
) -> list[Team]:
    """Parse standings into a list of `Team` records (roster + outcome).

    Drops players whose decklist fails `_parse_members` validation. Names
    come from the `name` field of each decklist entry; items from `item`
    (None when missing/null). Both species and item are passed through
    `normalize_name` so case/whitespace variants of the same name collapse
    to one vocab entry -- the Limitless API has been observed to return
    e.g. 'Sitrus Berry', 'Sitrus berry', and 'sitrus berry' for the same
    item across tournaments.

    When ``regulation`` is provided, items banned in that regulation are
    stripped from the team member (the mon is kept with ``item=None``).

    `placing` and the win/loss/tie `record` are read straight from the
    standings entry; placing is None when the API doesn't report it, and a
    missing record defaults to 0-0-0 (no signal, filterable downstream).
    """
    illegal_items = _ILLEGAL_ITEMS_BY_REGULATION.get(regulation or "", frozenset())
    teams: list[Team] = []
    for entry in standings:
        members = _parse_members(entry.get("decklist") or [], illegal_items)
        if members is None:
            continue
        placing = entry.get("placing")
        record = entry.get("record") or {}
        teams.append(Team(
            members=members,
            placing=placing if isinstance(placing, int) else None,
            wins=int(record.get("wins", 0)),
            losses=int(record.get("losses", 0)),
            ties=int(record.get("ties", 0)),
        ))
    return teams


def extract_matches(
    standings: list[dict],
    *,
    regulation: str | None = None,
) -> list[Match]:
    """Parse per-round match results from standings into `Match` records.

    Each standings entry may carry a ``rounds`` dict: round number (string)
    -> ``{"name": opponent, "result": "W"/"L", ...}``. In-person tournament
    exports have it; the Limitless API does not (those standings yield an
    empty list). Winner/loser are indices into the team list produced by
    `extract_teams` on the same standings (both share `_parse_members`
    validation, so the indices line up).

    A match is kept only when every check passes; each guard below drops a
    known artifact of the export format rather than a real game:

    - opponent resolves to a unique player name in the standings (drops
      ``BYE`` / ``LATE`` pseudo-opponents and players whose name appears
      more than once -- with a duplicated name the opponent pointer is
      ambiguous, so all matches touching it are dropped);
    - both players' rosters survived `_parse_members`;
    - the two entries report the same pairing reciprocally for that round,
      with complementary results (one W, one L) -- this drops double-loss
      rounds and one-sided reports;
    - each match is emitted once even though the data states it twice.
    """
    illegal_items = _ILLEGAL_ITEMS_BY_REGULATION.get(regulation or "", frozenset())

    team_index: list[int | None] = []
    n_valid = 0
    for entry in standings:
        members = _parse_members(entry.get("decklist") or [], illegal_items)
        if members is None:
            team_index.append(None)
        else:
            team_index.append(n_valid)
            n_valid += 1

    name_counts = Counter((entry.get("name") or "") for entry in standings)
    pos_by_name = {
        (entry.get("name") or ""): pos
        for pos, entry in enumerate(standings)
        if name_counts[entry.get("name") or ""] == 1
    }

    matches: list[Match] = []
    seen: set[tuple[int, int, int]] = set()
    for pos, entry in enumerate(standings):
        my_idx = team_index[pos]
        if my_idx is None:
            continue
        my_name = entry.get("name") or ""
        if name_counts[my_name] != 1:
            continue
        for round_key, played in (entry.get("rounds") or {}).items():
            played = played or {}
            result = played.get("result")
            if result not in ("W", "L"):
                continue
            opp_pos = pos_by_name.get(played.get("name") or "")
            if opp_pos is None or opp_pos == pos:
                continue
            opp_idx = team_index[opp_pos]
            if opp_idx is None:
                continue
            back = (standings[opp_pos].get("rounds") or {}).get(round_key) or {}
            if back.get("name") != my_name:
                continue
            if {result, back.get("result")} != {"W", "L"}:
                continue
            try:
                round_no = int(round_key)
            except (TypeError, ValueError):
                logger.warning("non-numeric round key %r -- skipping match", round_key)
                continue
            key = (round_no, min(pos, opp_pos), max(pos, opp_pos))
            if key in seen:
                continue
            seen.add(key)
            winner, loser = (my_idx, opp_idx) if result == "W" else (opp_idx, my_idx)
            matches.append(Match(round=round_no, winner=winner, loser=loser))
    return matches


def _cache_path(cache_dir: Path, tournament_id: str) -> Path:
    return cache_dir / f"{tournament_id}.json"


def _save_tournament(cache_dir: Path, t: TournamentTeams) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "type": t.tournament_type,
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
        # Each match: [round, winner, loser] with winner/loser indexing
        # into the teams list above. Empty for limitless entries.
        "matches": [[m.round, m.winner, m.loser] for m in t.matches],
    }
    with _cache_path(cache_dir, t.meta.id).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _payload_version_ok(payload: dict) -> bool:
    """True when a cached payload meets the minimum version for its type."""
    entry_type = payload.get("type", "limitless")
    minimum = _MIN_CACHE_VERSION.get(entry_type, CACHE_VERSION)
    return payload.get("version", 1) >= minimum


def _matches_from_payload(payload: dict) -> tuple[Match, ...]:
    return tuple(
        Match(round=r, winner=w, loser=l)
        for r, w, l in payload.get("matches", [])
    )


def _load_tournament(cache_dir: Path, tournament_id: str) -> TournamentTeams | None:
    """Load a cached tournament, or return None if missing / outdated schema.

    Returning None signals the caller to re-fetch / re-import. Older cached
    files (v1: no placing/record; pre-v1: no items; in-person v2: no
    matches) are incompatible with the current schema for their type, so we
    treat them as cache misses.
    """
    p = _cache_path(cache_dir, tournament_id)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not _payload_version_ok(payload):
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
    return TournamentTeams(
        meta=meta, teams=teams, tournament_type=payload.get("type", "limitless"),
        matches=_matches_from_payload(payload),
    )


def fetch_limitless_tournaments(
    *,
    max_teams: int = DEFAULT_MAX_TEAMS,
    regulation: str = DEFAULT_REGULATION,
    game: str = DEFAULT_GAME,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    start_page: int = 1,
    min_teams_per_tournament: int = MIN_TEAMS_PER_TOURNAMENT,
) -> list[TournamentTeams]:
    """Fetch tournaments from the Limitless API, caching as we go.

    Walks the API newest-first until >= ``max_teams`` teams accumulated.
    ``max_teams`` is a fetch limit (stop walking the API), not a corpus cap.
    ``start_page`` lets callers skip past recent catalog pages to reach
    tournaments from older formats. ``min_teams_per_tournament`` filters out
    small events (both pre-fetch by player count and post-extract by team count).

    Termination conditions (whichever fires first):
    - ``total_teams >= max_teams`` (target hit)
    - The API returns an empty page (catalog exhausted)
    - A whole page yields zero new in-regulation tournament IDs
    """
    cache_dir = Path(cache_dir)
    out: list[TournamentTeams] = []
    total_teams = 0
    seen_ids: set[str] = set()

    for batch in iter_catalog_pages(game=game, start_page=start_page):
        new_in_regulation = 0
        for tm in batch:
            if tm.id in seen_ids:
                continue
            seen_ids.add(tm.id)
            if not could_match_regulation(tm, regulation):
                continue
            new_in_regulation += 1
            if is_likely_singles_by_name(tm.name):
                logger.info("[skip name]    %s '%s'", tm.id, tm.name)
                continue
            # Pre-fetch size check: registered players bound the team count
            # from above (teams <= players always), so skipping here avoids
            # API calls for events too small to clear the threshold.
            if tm.players < min_teams_per_tournament:
                logger.info(
                    "[skip small]   %s '%s' (players=%d < %d)",
                    tm.id, tm.name, tm.players, min_teams_per_tournament,
                )
                continue
            cached = _load_tournament(cache_dir, tm.id)
            # Trust the cache only when its stored label still matches what the
            # roster resolves to -- a stale entry cached under an old label
            # (e.g. an M-B event cached as M-A by a prior run) is re-fetched so
            # its items are stripped under the correct regulation.
            if cached is not None and (
                resolve_regulation(cached.meta, cached.teams) == cached.meta.regulation
            ):
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
                # Detect the true regulation on an unstripped parse (markers
                # survive any item filter), then extract under that regulation
                # so its illegal-item filter -- not the stale label's -- applies.
                if _has_pending_relabel(tm):
                    true_reg = resolve_regulation(
                        tm, extract_teams(standings, regulation=None)
                    )
                else:
                    true_reg = tm.regulation
                if true_reg != tm.regulation:
                    logger.info(
                        "[relabel %-5s] %s '%s' (catalog=%s, date=%s)",
                        true_reg, tm.id, tm.name, tm.regulation, _day(tm.date),
                    )
                teams = extract_teams(standings, regulation=true_reg)
                meta = replace(tm, regulation=true_reg)
                ttd = TournamentTeams(meta=meta, teams=teams, tournament_type="limitless")
                _save_tournament(cache_dir, ttd)
                source = "fetched"
                time.sleep(POLITE_SLEEP_SEC)
            # A relabel can resolve a candidate to a regulation other than the
            # one being ingested (e.g. a post-cutoff event that stayed genuinely
            # M-A): it is cached under its true label but excluded from this run.
            if ttd.meta.regulation != regulation:
                logger.info(
                    "[skip reg]     %s '%s' (resolved=%s != %s)",
                    tm.id, tm.name, ttd.meta.regulation, regulation,
                )
                continue
            # Post-extract size check: catches cached tournaments under a
            # newly-tightened threshold and high-dropout events where
            # extract_teams trimmed below the limit.
            if len(ttd.teams) < min_teams_per_tournament:
                logger.info(
                    "[skip small]   %s '%s' (teams=%d < %d after extract)",
                    tm.id, tm.name, len(ttd.teams), min_teams_per_tournament,
                )
                continue
            out.append(ttd)
            total_teams += len(ttd.teams)
            logger.info(
                "[%5d / %d] %s %s '%s' (+%d teams)",
                total_teams, max_teams, source, tm.id, tm.name, len(ttd.teams),
            )
            if total_teams >= max_teams:
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


def all_team_observations(tournaments: list[TournamentTeams]) -> list[TeamObservation]:
    """Flatten ingested tournaments into per-team `TeamObservation` records.

    Same flattening (and ordering) as `all_teams`, but keeps each roster's
    tournament date and provenance type -- the inputs for recency / in-person
    sample weighting in `loaders`.
    """
    return [
        TeamObservation(
            members=team.members,
            date=t.meta.date,
            tournament_type=t.tournament_type,
        )
        for t in tournaments
        for team in t.teams
    ]


def all_match_observations(tournaments: list[TournamentTeams]) -> list[MatchObservation]:
    """Flatten ingested tournaments into per-match `MatchObservation` records.

    Resolves each match's winner/loser team indices to roster frozensets and
    keeps the parent tournament's date and id. Limitless entries contribute
    nothing (no match data), so this is effectively the in-person match
    corpus -- the input for Bradley-Terry fitting and match-level outcome
    ceilings.
    """
    return [
        MatchObservation(
            winner=t.teams[m.winner].members,
            loser=t.teams[m.loser].members,
            round=m.round,
            date=t.meta.date,
            tournament_id=t.meta.id,
        )
        for t in tournaments
        for m in t.matches
    ]


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


def load_cached_tournaments(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    regulation: str | None = DEFAULT_REGULATION,
    tournament_type: str | None = None,
    min_teams_per_tournament: int = MIN_TEAMS_PER_TOURNAMENT,
) -> list[TournamentTeams]:
    """Load all cached tournaments from disk. Pure offline, no API calls.

    Reads every JSON file in ``cache_dir``, filters by regulation, provenance
    type, tournament size, and singles name. Returns all qualifying
    tournaments sorted by date (oldest first).
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        logger.warning("cache directory %s does not exist", cache_dir)
        return []

    out: list[TournamentTeams] = []
    for p in sorted(cache_dir.glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError) as err:
            logger.warning("skipping unreadable cache file %s: %s", p.name, err)
            continue

        if not _payload_version_ok(payload):
            continue

        entry_type = payload.get("type", "limitless")
        if tournament_type is not None and entry_type != tournament_type:
            continue

        meta = TournamentMeta(**payload["meta"])
        if regulation is not None and meta.regulation != regulation:
            continue
        if is_likely_singles_by_name(meta.name):
            continue

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

        if len(teams) < min_teams_per_tournament:
            continue

        out.append(TournamentTeams(
            meta=meta, teams=teams, tournament_type=entry_type,
            matches=_matches_from_payload(payload),
        ))

    out.sort(key=lambda t: t.meta.date)
    logger.info(
        "loaded %d cached tournaments, %d teams total",
        len(out), sum(len(t.teams) for t in out),
    )
    return out


_DATE_SUFFIX_RE = re.compile(r"[_-](\d{4})[_-](\d{2})[_-](\d{2})$")


def import_in_person_tournaments(
    *,
    local_dir: Path = DEFAULT_LOCAL_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[TournamentTeams]:
    """Import in-person tournament JSON files into the cache.

    Scans ``local_dir/[format]/[id]_[date].json`` for raw Limitless-format
    standings exports. Each file is normalized via ``extract_teams()`` +
    ``extract_matches()`` and written to ``cache_dir`` with
    ``type="in-person"``. Cache entries already at the current schema are
    skipped (idempotent); outdated ones are re-imported in place.

    The parent directory name is used as the regulation (e.g. ``M-A``).
    The date is parsed from the filename suffix (``[id]_YYYY-MM-DD.json``).
    """
    local_dir = Path(local_dir)
    if not local_dir.exists():
        logger.info("local tournament directory %s does not exist, skipping", local_dir)
        return []

    cache_dir = Path(cache_dir)
    imported: list[TournamentTeams] = []

    for format_dir in sorted(local_dir.iterdir()):
        if not format_dir.is_dir():
            continue
        regulation = format_dir.name

        for json_file in sorted(format_dir.glob("*.json")):
            stem = json_file.stem
            tournament_id = f"local_{stem}"

            if _load_tournament(cache_dir, tournament_id) is not None:
                logger.info("[skip cached]  %s (already imported)", stem)
                continue

            date_match = _DATE_SUFFIX_RE.search(stem)
            date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}" if date_match else ""

            try:
                with json_file.open("r", encoding="utf-8") as f:
                    standings = json.load(f)
            except (json.JSONDecodeError, OSError) as err:
                logger.warning("skipping unreadable file %s: %s", json_file, err)
                continue

            if not isinstance(standings, list):
                logger.warning("skipping %s: expected list, got %s", json_file, type(standings).__name__)
                continue

            teams = extract_teams(standings, regulation=regulation)
            matches = extract_matches(standings, regulation=regulation)

            meta = TournamentMeta(
                id=tournament_id,
                name=stem,
                date=date,
                regulation=regulation,
                players=len(standings),
            )
            tt = TournamentTeams(
                meta=meta, teams=teams, tournament_type="in-person",
                matches=tuple(matches),
            )
            _save_tournament(cache_dir, tt)
            imported.append(tt)
            logger.info(
                "[imported]     %s -> %s (%d teams, %d matches, regulation=%s, date=%s)",
                json_file.name, tournament_id, len(teams), len(matches),
                regulation, date or "(none)",
            )

    logger.info("imported %d in-person tournaments", len(imported))
    return imported


def ingest(
    *,
    max_teams: int = DEFAULT_MAX_TEAMS,
    regulation: str = DEFAULT_REGULATION,
    game: str = DEFAULT_GAME,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[TournamentTeams]:
    """Deprecated: use ``fetch_limitless_tournaments()`` + ``load_cached_tournaments()``.

    Fetches from the Limitless API, then returns all cached tournaments.
    """
    warnings.warn(
        "ingest() is deprecated. Use fetch_limitless_tournaments() to populate "
        "the cache, then load_cached_tournaments() to read it.",
        DeprecationWarning,
        stacklevel=2,
    )
    fetch_limitless_tournaments(
        max_teams=max_teams, regulation=regulation, game=game, cache_dir=cache_dir,
    )
    return load_cached_tournaments(cache_dir=cache_dir, regulation=regulation)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch/import tournament data into the cache.",
    )
    parser.add_argument(
        "--limitless-only", action="store_true",
        help="Only fetch from the Limitless API (skip in-person import)",
    )
    parser.add_argument(
        "--in-person-only", action="store_true",
        help="Only import local in-person files (skip API fetch)",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--regulation", default=DEFAULT_REGULATION)
    parser.add_argument(
        "--max-teams", type=int, default=DEFAULT_MAX_TEAMS,
        help="Limitless API fetch limit (stop after this many teams)",
    )
    parser.add_argument(
        "--start-page", type=int, default=1,
        help="Catalog page to start from (newest-first; use to skip past recent tournaments)",
    )
    parser.add_argument(
        "--min-teams-per-tournament", type=int, default=MIN_TEAMS_PER_TOURNAMENT,
        help="Skip tournaments with fewer teams than this (default %(default)s)",
    )
    args = parser.parse_args()

    if not args.in_person_only:
        fetch_limitless_tournaments(
            max_teams=args.max_teams,
            regulation=args.regulation,
            cache_dir=args.cache_dir,
            start_page=args.start_page,
            min_teams_per_tournament=args.min_teams_per_tournament,
        )

    if not args.limitless_only:
        import_in_person_tournaments(
            local_dir=args.local_dir,
            cache_dir=args.cache_dir,
        )

    all_t = load_cached_tournaments(
        cache_dir=args.cache_dir,
        regulation=args.regulation,
        min_teams_per_tournament=args.min_teams_per_tournament,
    )
    teams = all_teams(all_t)
    print(f"Tournaments in cache: {len(all_t)}")
    print(f"Total teams: {len(teams)}")
    print(f"Total matches: {sum(len(t.matches) for t in all_t)}")
    if teams:
        sample = next(iter(teams))
        print(f"Sample team:")
        for species, item in sorted(sample, key=lambda m: m[0]):
            print(f"  {species:<25} item: {item}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
