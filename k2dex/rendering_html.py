"""Species-name → Showdown sprite-slug mapping.

The sole live export is :func:`species_to_slug`, which is mirrored 1:1 by
``web/src/render/sprite-url.ts:speciesToSlug`` and gated by
``tests/test_parity.py::test_species_to_slug_cases``. The webapp uses the TS
port to build sprite URLs; this module exists to keep the Python side of that
parity contract.
"""

from __future__ import annotations

import re

# Species whose canonical name *contains* a hyphen as part of the base name
# (not as a forme separator). For these, all hyphens are stripped to match
# Showdown's home-folder convention (``chienpao.png``, ``hooh.png``).
# Everything else preserves the first hyphen (forme separator) and
# collapses any subsequent ones.
_HYPHEN_BASE_SPECIES = frozenset({
    "ho-oh",
    "porygon-z",
    "jangmo-o",
    "hakamo-o",
    "kommo-o",
    "wo-chien",
    "chien-pao",
    "ting-lu",
    "chi-yu",
    "type-null",
    "nidoran-f",
    "nidoran-m",
})

# Species whose canonical corpus name doesn't follow normal slug rules.
# Keys are the lowercased canonical name; values are the Showdown slug.
_SLUG_OVERRIDES: dict[str, str] = {
    "eternal flower floette": "floette",
    "paldean tauros aqua breed": "tauros-paldeaaqua",
    "paldean tauros blaze breed": "tauros-paldeablaze",
    "paldean tauros combat breed": "tauros-paldeacombat",
}

# Limitless stores regional formes as "Adjective Species" (e.g. "Alolan Ninetales").
# Maps the adjective → the Showdown regional suffix.
_REGIONAL_ADJECTIVE: dict[str, str] = {
    "alolan": "alola",
    "galarian": "galar",
    "hisuian": "hisui",
    "paldean": "paldea",
}

# Limitless stores Rotom formes as "Forme Rotom" (e.g. "Wash Rotom").
_ROTOM_FORMES: frozenset[str] = frozenset({"wash", "heat", "frost", "mow", "fan"})

# Limitless stores Lycanroc formes as "Lycanroc Forme" (e.g. "Lycanroc Dusk").
# Showdown's home folder uses base "lycanroc" for Midday and a hyphen suffix
# for the others ("lycanroc-dusk", "lycanroc-midnight").
_LYCANROC_FORMES: frozenset[str] = frozenset({"dusk", "midnight", "midday"})


def species_to_slug(name: str) -> str:
    """Convert a display species name to Showdown's home-sprite slug.

    Rules (empirically verified against the Showdown home-folder URLs):

    1. Lowercase; strip apostrophes / periods / other punctuation.
    2. Collapse internal spaces to nothing (``Iron Hands`` → ``ironhands``).
    3. If the result (with hyphens intact) is a known base-name-contains-hyphen
       species (``Ho-Oh``, ``Chien-Pao``, ``Porygon-Z``, ``Nidoran-F`` …),
       strip all hyphens → ``hooh``, ``chienpao``, ``porygonz``, ``nidoranf``.
    4. Otherwise the first hyphen is a forme separator (``Calyrex-Shadow``,
       ``Blastoise-Mega``, ``Arcanine-Hisui``) — preserve it and collapse
       any subsequent hyphens. So ``Urshifu-Rapid-Strike`` →
       ``urshifu-rapidstrike`` (matches Showdown's folder layout).

    Examples
    --------
    >>> species_to_slug("Calyrex-Shadow")
    'calyrex-shadow'
    >>> species_to_slug("Blastoise-Mega")
    'blastoise-mega'
    >>> species_to_slug("Arcanine-Hisui")
    'arcanine-hisui'
    >>> species_to_slug("Urshifu-Rapid-Strike")
    'urshifu-rapidstrike'
    >>> species_to_slug("Chien-Pao")
    'chienpao'
    >>> species_to_slug("Ho-Oh")
    'hooh'
    >>> species_to_slug("Iron Hands")
    'ironhands'
    >>> species_to_slug("Farfetch'd")
    'farfetchd'

    Misses still fall through to the webapp's ``missingno`` sprite fallback.
    """
    override = _SLUG_OVERRIDES.get(name.lower())
    if override is not None:
        return override
    # Keep only alphanumerics, hyphens, and spaces; lowercase.
    cleaned = re.sub(r"[^a-z0-9\s\-]+", "", name.lower()).strip()
    words = cleaned.split()
    if len(words) >= 2:
        # "Alolan Ninetales" → "ninetales-alola", "Hisuian Arcanine" → "arcanine-hisui"
        if words[0] in _REGIONAL_ADJECTIVE:
            region = _REGIONAL_ADJECTIVE[words[0]]
            base = "".join(words[1:])
            return f"{base}-{region}"
        # "Wash Rotom" → "rotom-wash", "Heat Rotom" → "rotom-heat"
        if len(words) == 2 and words[1] == "rotom" and words[0] in _ROTOM_FORMES:
            return f"rotom-{words[0]}"
        # "Lycanroc Dusk" → "lycanroc-dusk"; "Lycanroc Midday" → "lycanroc" (base).
        if len(words) == 2 and words[0] == "lycanroc" and words[1] in _LYCANROC_FORMES:
            return "lycanroc" if words[1] == "midday" else f"lycanroc-{words[1]}"
    # Spaces collapse to nothing per Showdown convention.
    no_spaces = re.sub(r"\s+", "", cleaned)
    if "-" not in no_spaces:
        return no_spaces
    if no_spaces in _HYPHEN_BASE_SPECIES:
        return no_spaces.replace("-", "")
    head, *rest = no_spaces.split("-")
    return f"{head}-{''.join(rest)}"
