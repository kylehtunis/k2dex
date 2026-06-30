"""HTML helpers for the lab-notebook visual language.

These return raw HTML strings to be passed into ``st.markdown(..., unsafe_allow_html=True)``.
They're paired with the CSS classes defined in :mod:`styles`; renaming a
class here means renaming it there too.

Component vocabulary mirrors DESIGN.md's component patterns:

- :func:`section_label`  — §N eyebrow + title + optional right slot
- :func:`stat`           — label / mono-numeric value / muted sub-text
- :func:`score_chip`     — sign-colored numeric pill
- :func:`signed_bar`     — bipolar horizontal bar centered at 0
- :func:`mini_bar`       — unipolar bar (e.g. m̂ percentage)
- :func:`sprite_url`     — Showdown CDN URL for a species name
- :func:`sprite_img`     — ``<img>`` tag with missingno fallback

All number formatting goes through :func:`_fmt_signed` / :func:`_fmt_pct`
so we get consistent ``+0.81`` / ``28.4%`` rendering across pages.
"""

from __future__ import annotations

import base64
import re
from html import escape
from pathlib import Path

from .styles import LAB_NEG, LAB_POS

# Showdown's home-format sprite CDN. Forme variants live under
# ``<species>-<forme>.png`` (e.g. ``calyrex-shadow``, ``blastoise-mega``,
# ``arcanine-hisui``). See :func:`species_to_slug` for the slug rules.
_SPRITE_CDN = "https://play.pokemonshowdown.com/sprites/home"

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


def _load_fallback_sprite() -> str:
    """Embed ``assets/missingno.{svg,png}`` as a ``data:`` URI used when a sprite 404s.

    Inlining the fallback as a data URI avoids depending on Streamlit's
    static-file router (which isn't enabled in this project) and keeps the
    fallback working offline. SVG is preferred when available — it
    scales crisply at any size and base64-encodes to a smaller payload
    than a PNG of equivalent visual quality.
    """
    assets_dir = Path(__file__).parent / "assets"
    for fname, mime in (("missingno.svg", "image/svg+xml"),
                        ("missingno.png", "image/png")):
        asset = assets_dir / fname
        if asset.exists():
            encoded = base64.b64encode(asset.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
    return "https://play.pokemonshowdown.com/sprites/gen5/missingno.png"


_SPRITE_FALLBACK = _load_fallback_sprite()


def _fmt_signed(value: float, decimals: int = 3) -> str:
    """``+0.810`` / ``-4.123`` — always show sign, fixed decimals, tabular."""
    return f"{value:+.{decimals}f}"


def _fmt_pct(value: float, decimals: int = 1) -> str:
    """``28.4%`` — fraction-in [0,1] formatted as a percentage."""
    return f"{value * 100:.{decimals}f}%"


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

    Misses still fall through to the ``onerror`` fallback in :func:`sprite_img`.
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


def extract_species(feature: str) -> str:
    """Pull the species name out of a Phase 3 vocab string.

    Phase 3 vocab uses ``"Species @ Item"`` (or bare ``"Species"`` for
    itemless mons). Strip the ``@ Item`` suffix to get a species sprite key.
    """
    return feature.split(" @ ", 1)[0]


def extract_item(feature: str) -> str | None:
    """Pull the item out of a Phase 3 vocab string, or None if itemless."""
    if " @ " not in feature:
        return None
    return feature.split(" @ ", 1)[1]


def sprite_url(name: str) -> str:
    """Showdown home-sprite CDN URL for ``name`` (species or "Species @ Item")."""
    species = extract_species(name)
    return f"{_SPRITE_CDN}/{species_to_slug(species)}.png"


def sprite_img(name: str, size: int = 64, *, classes: str = "lab-sprite") -> str:
    """Sprite as an HTML5 ``<object>`` with a missingno fallback child.

    Three approaches we tried before landing here:

    1. ``<img onerror=...>`` — Streamlit's React renderer strips inline
       event handlers, so the swap never fires; sprite 404s show alt text.
    2. ``<script>`` re-binding via ``addEventListener`` — Streamlit uses
       ``dangerouslySetInnerHTML`` for unsafe markdown, which refuses to
       execute script content.
    3. Layered-``background-image`` — the fallback sits *behind* the CDN
       sprite, so transparent pixels in successful sprites let the
       missingno bleed through.

    ``<object>`` is the native HTML5 mechanism for resource fallback: if
    the browser can't load ``data``, it renders the inner content instead.
    When the sprite loads, the inner fallback is hidden entirely (no
    bleed). When the sprite fails, only the fallback is rendered. No JS,
    no event handlers, no transparency tricks.

    ``pointer-events: none`` prevents focus rings / interactive behavior
    that some browsers add to ``<object>``.
    """
    return _sprite_object(name, size, classes)


def _sprite_box(name: str, size: int, classes: str = "") -> str:
    """Internal sprite helper used by slot cards, completion rows, and the
    inline pair / swap cells. Same ``<object>`` + fallback-child pattern
    as :func:`sprite_img`, with ``flex-shrink:0`` so the sprite doesn't
    squish inside flex containers.
    """
    return _sprite_object(name, size, classes, flex_shrink=True)


def _sprite_object(
    name: str, size: int, classes: str = "", *, flex_shrink: bool = False,
) -> str:
    """Shared implementation of the ``<object>`` sprite element."""
    src = sprite_url(name)
    species = extract_species(name)
    cls = f' class="{classes}"' if classes else ""
    flex = "flex-shrink:0;" if flex_shrink else ""
    return (
        f'<object{cls} data="{src}" type="image/png" '
        f'style="display:inline-block;width:{size}px;height:{size}px;'
        f'pointer-events:none;border:0;background:transparent;{flex}">'
        f'<span role="img" aria-label="{escape(species)}" '
        f'style="display:inline-block;width:{size}px;height:{size}px;'
        f"background:url('{_SPRITE_FALLBACK}') center/contain no-repeat;\""
        f"></span>"
        f"</object>"
    )


def section_label(num: str, title: str, right: str | None = None) -> str:
    """Numbered section header with optional right slot.

    ``num`` is rendered as ``§01`` (caller passes ``"01"`` etc.).
    ``right`` is a raw HTML fragment (e.g. a caption, segmented control,
    or count) placed at the right edge of the header row.
    """
    right_html = (
        f'<div class="lab-section-right">{right}</div>' if right else ""
    )
    return (
        f'<div class="lab-section">'
        f'<div><span class="lab-section-num">§{escape(num)}</span>'
        f'<span class="lab-section-title">{escape(title)}</span></div>'
        f'{right_html}'
        f'</div>'
    )


def stat(label: str, value: str, sub: str | None = None, tooltip: str | None = None) -> str:
    """Single foregrounded number: small-caps label, mono value, muted sub.

    ``value`` is passed through as-is (caller is responsible for formatting
    — typically via :func:`_fmt_signed` / :func:`_fmt_pct`). ``sub`` is an
    optional second line in the muted mono style. ``tooltip`` renders as a
    browser-native ``title`` attribute (hover to read).
    """
    sub_html = f'<div class="lab-stat-sub">{escape(sub)}</div>' if sub else ""
    title_attr = f' title="{escape(tooltip)}"' if tooltip else ""
    return (
        f'<div{title_attr}>'
        f'<div class="lab-stat-label">{escape(label)}</div>'
        f'<div class="lab-stat-value">{escape(value)}</div>'
        f'{sub_html}'
        f'</div>'
    )


def score_chip(value: float, fmt: str = "signed", decimals: int = 3) -> str:
    """Sign-colored numeric pill. ``fmt`` ∈ {signed, pct, count}.

    - ``signed``: ``+0.810`` / ``-4.123`` with green / red color
    - ``pct``: ``28.4%`` always inked (no sign coloring; percentages are
      magnitudes, not bipolar)
    - ``count``: bare integer, ``ink-muted`` when zero, otherwise ink
    """
    if fmt == "signed":
        color = LAB_POS if value >= 0 else LAB_NEG
        text = _fmt_signed(value, decimals)
        return f'<span class="lab-chip" style="color:{color};">{text}</span>'
    if fmt == "pct":
        return f'<span class="lab-chip">{_fmt_pct(value, decimals)}</span>'
    if fmt == "count":
        color = "var(--lab-ink-muted)" if int(value) == 0 else "var(--lab-ink)"
        return f'<span class="lab-chip" style="color:{color};">{int(value)}</span>'
    raise ValueError(f"Unknown fmt for score_chip: {fmt!r}")


def signed_bar(value: float, max_value: float, width: int = 80) -> str:
    """Bipolar horizontal bar centered at 0.

    Bar extends right (green) for positive ``value``, left (red) for
    negative. Magnitude is clamped to ``max_value`` so over-range values
    render at full extent rather than overflowing the container.
    """
    half = width // 2
    mag = min(abs(value), max_value) / max_value if max_value > 0 else 0
    pixels = int(round(mag * half))
    if value >= 0:
        fill = (
            f'<div class="lab-bar-fill" '
            f'style="left:{half}px;width:{pixels}px;background:{LAB_POS};">'
            f'</div>'
        )
    else:
        fill = (
            f'<div class="lab-bar-fill" '
            f'style="left:{half - pixels}px;width:{pixels}px;background:{LAB_NEG};">'
            f'</div>'
        )
    return (
        f'<div class="lab-bar-wrap" style="width:{width}px;">'
        f'<div class="lab-bar-mid" style="left:{half}px;"></div>'
        f'{fill}'
        f'</div>'
    )


def mini_bar(value: float, max_value: float, width: int = 80) -> str:
    """Unipolar bar for ``[0, max_value]`` values (marginals, frequencies)."""
    mag = min(value, max_value) / max_value if max_value > 0 else 0
    pixels = int(round(mag * width))
    return (
        f'<div class="lab-minibar-wrap" style="width:{width}px;">'
        f'<div class="lab-minibar-fill" style="width:{pixels}px;"></div>'
        f'</div>'
    )


def slot_card(name: str, indicator: str = "s=1") -> str:
    """Filled slot card: 3px ink top border, sprite, serif name, dotted divider, mono item.

    ``name`` is a vocab string ("Species" or "Species @ Item"). The item
    line is omitted on Phase 1/2 (no item info encoded). The ``indicator``
    is the top-right state badge (``s=1`` for picked, ``s=?`` for free —
    the latter is handled by :func:`slot_card_empty` instead).
    """
    species = extract_species(name)
    item = extract_item(name)
    item_html = (
        f'<hr class="lab-slot-divider"/>'
        f'<div class="lab-slot-item">@ {escape(item)}</div>'
        if item
        else ""
    )
    return (
        f'<div class="lab-slot">'
        f'<div class="lab-slot-indicator">{escape(indicator)}</div>'
        f'{_sprite_box(name, 72, "lab-slot-sprite")}'
        f'<div class="lab-slot-name">{escape(species)}</div>'
        f'{item_html}'
        f'</div>'
    )


def slot_card_empty(ordinal: int) -> str:
    """Empty slot — `FREE SLOT` label + `·N·` mono ordinal."""
    return (
        f'<div class="lab-slot-empty">'
        f'<div class="lab-slot-empty-label">free slot</div>'
        f'<div class="lab-slot-empty-ord">·{ordinal}·</div>'
        f'</div>'
    )


def slot_strip(picked: list[str], team_size: int = 6) -> str:
    """Render the 6-card roster strip.

    ``picked`` is the ordered list of currently-picked vocab strings.
    Slots beyond ``len(picked)`` render as empty slots numbered from
    ``len(picked) + 1``.
    """
    cells = [slot_card(name) for name in picked]
    for i in range(len(picked) + 1, team_size + 1):
        cells.append(slot_card_empty(i))
    return f'<div class="lab-slot-strip">{"".join(cells)}</div>'


def excluded_row(names: list[str], note: str | None = None) -> str:
    """`EXCLUDED` label + red-bordered tag pills + optional italic explainer."""
    if not names and not note:
        return ""
    tags = "".join(
        f'<span class="lab-excluded-tag">{escape(extract_species(n))}</span>'
        for n in names
    )
    note_html = (
        f'<span class="lab-excluded-note">— {escape(note)}</span>'
        if note
        else ""
    )
    label = '<span class="lab-excluded-label">excluded</span>' if names else ""
    return f'<div class="lab-excluded-row">{label}{tags}{note_html}</div>'


def stat_strip(cells: list[tuple[str, ...]], columns: int | None = None) -> str:
    """Render a 1×N grid of ``stat`` cells inside a panel.

    Each cell is ``(label, value, sub)`` or ``(label, value, sub, tooltip)``.
    A 4th element, when present, is passed as the ``tooltip`` (browser-native
    ``title`` attribute, visible on hover). ``columns`` defaults to
    ``len(cells)`` (one row, N equal columns).
    """
    n = columns or len(cells)
    parts = []
    for c in cells:
        tooltip = c[3] if len(c) > 3 else None
        parts.append(stat(c[0], c[1], c[2] if len(c) > 2 else None, tooltip))
    return (
        f'<div class="lab-stat-strip" '
        f'style="grid-template-columns: repeat({n}, 1fr);">'
        f'{"".join(parts)}'
        f'</div>'
    )


def comp_mon_cell(name: str) -> str:
    """One mon as a sprite + serif name + mono item row (used in completion-pair cells)."""
    species = extract_species(name)
    item = extract_item(name)
    item_html = (
        f'<div class="lab-comp-mon-item">@ {escape(item)}</div>'
        if item
        else ""
    )
    return (
        f'<div class="lab-comp-mon">'
        f'{_sprite_box(name, 36)}'
        f'<div><div class="lab-comp-mon-name">{escape(species)}</div>'
        f'{item_html}</div>'
        f'</div>'
    )


def pair_cell(name_a: str, name_b: str) -> str:
    """Two mons side-by-side (sprite + serif name) joined by `×` — used for
    pair-decomposition tables and extreme-couplings tables.
    """
    return (
        '<div style="display:flex;align-items:center;gap:10px;">'
        f'{inline_mon(name_a)}'
        '<span style="font-family:var(--lab-font-mono);color:var(--lab-ink-muted);">×</span>'
        f'{inline_mon(name_b)}'
        '</div>'
    )


def swap_cell(name_out: str, name_in: str) -> str:
    """Swap action cell: out-mon → in-mon, with a forest-green arrow."""
    return (
        '<div style="display:flex;align-items:center;gap:10px;">'
        f'{inline_mon(name_out)}'
        '<span style="font-family:var(--lab-font-mono);font-size:14px;'
        'color:var(--lab-accent);">→</span>'
        f'{inline_mon(name_in)}'
        '</div>'
    )


def inline_mon(name: str, size: int = 28) -> str:
    """Compact sprite + serif name, used inside pair / swap cells."""
    species = extract_species(name)
    item = extract_item(name)
    item_html = (
        f'<span class="lab-comp-mon-item" style="margin-left:6px;">@ {escape(item)}</span>'
        if item
        else ""
    )
    return (
        '<span style="display:inline-flex;align-items:center;gap:6px;">'
        f'{_sprite_box(name, size)}'
        f'<span class="lab-comp-mon-name">{escape(species)}</span>'
        f'{item_html}'
        '</span>'
    )


def team_mini_strip(names: list[str], size: int = 24) -> str:
    """Small inline sprite strip for the 6 (or fewer) mons of a team."""
    sprites = "".join(_sprite_box(n, size) for n in names)
    return (
        '<div style="display:inline-flex;align-items:center;gap:4px;">'
        f'{sprites}'
        '</div>'
    )


def corpus_cell(delta: int, count: int) -> str:
    """Merged corpus-status badge.

    delta == 0 → green ``N×`` showing how many times the exact roster was
    observed in the ingested corpus. delta >= 1 → amber/red ``Δk (N)``
    showing the swap distance ``k`` to the nearest observed roster and that
    nearest roster's own corpus count ``N``. Replaces the old separate
    obs-count + Δswaps-badge pair.
    """
    if delta == 0:
        return f'<span class="lab-badge lab-badge-seen">{count}×</span>'
    cls = "lab-badge-near" if delta == 1 else "lab-badge-far"
    return f'<span class="lab-badge {cls}">Δ{delta} ({count})</span>'


def page_title(
    eyebrow: str,
    h1: str,
    subtitle_html: str = "",
    right_caption: str | None = None,
) -> str:
    """Page title block: eyebrow + H1 + optional italic subtitle + optional right caption.

    ``subtitle_html`` is passed through as raw HTML so callers can embed
    inline ``<code>`` for math notation. ``right_caption`` is plain text
    (escaped); usually the corpus identity line. Subtitle div is omitted
    when ``subtitle_html`` is empty or falsy.
    """
    right_html = (
        f'<div class="lab-corpus-caption">{escape(right_caption)}</div>'
        if right_caption
        else ""
    )
    subtitle_div = (
        f'<div class="lab-subtitle">{subtitle_html}</div>'
        if subtitle_html
        else ""
    )
    return (
        f'<div style="display:flex;align-items:flex-end;justify-content:space-between;'
        f'gap:32px;margin-top:8px;margin-bottom:8px;">'
        f'<div>'
        f'<div class="lab-eyebrow">{escape(eyebrow)}</div>'
        f'<div class="lab-h1">{escape(h1)}</div>'
        f'{subtitle_div}'
        f'</div>'
        f'{right_html}'
        f'</div>'
    )
