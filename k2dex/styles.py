"""Visual-language design tokens for the k2dex webapp.

Injects Google Fonts (Source Serif 4 / Inter / IBM Plex Mono), defines the
``--lab-*`` CSS custom properties for the palette, and applies base
typography to Streamlit's container chrome. Call :func:`inject` once near
the top of ``main()``; downstream HTML helpers reference the variables
via ``var(--lab-ink)`` etc. so the palette is defined in exactly one place.

The palette mirrors the LAB.* tokens documented in DESIGN.md. Three semantic
chromas (pos / neg / warn) share luminance and chroma; only hue varies.
"""

from __future__ import annotations

import streamlit as st

# Python-side mirrors of the most-used tokens. Keep these in sync with the
# CSS variables in :func:`_css` below. Used when a value must appear in an
# inline ``style="..."`` attribute (e.g. computed bar widths) and can't
# pull from a CSS var directly.
LAB_BG = "#faf7f0"
LAB_PANEL = "#ffffff"
LAB_PANEL_ALT = "#f3efe5"
LAB_INK = "#1a1815"
LAB_INK_SOFT = "#5b554c"
LAB_INK_MUTED = "#8b857a"
LAB_RULE = "#d9d2c2"
LAB_RULE_SOFT = "#e8e3d6"
LAB_ACCENT = "oklch(0.42 0.10 145)"  # forest green
LAB_POS = "oklch(0.55 0.13 155)"     # brighter green (synergy)
LAB_NEG = "oklch(0.55 0.13 25)"      # coral red (exclusion)
LAB_WARN = "oklch(0.62 0.13 75)"     # amber (caveat)


def _css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --lab-bg: #faf7f0;
    --lab-panel: #ffffff;
    --lab-panel-alt: #f3efe5;
    --lab-ink: #1a1815;
    --lab-ink-soft: #5b554c;
    --lab-ink-muted: #8b857a;
    --lab-rule: #d9d2c2;
    --lab-rule-soft: #e8e3d6;
    --lab-accent: oklch(0.42 0.10 145);
    --lab-pos: oklch(0.55 0.13 155);
    --lab-neg: oklch(0.55 0.13 25);
    --lab-warn: oklch(0.62 0.13 75);

    --lab-font-serif: 'Source Serif 4', 'Georgia', serif;
    --lab-font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
    --lab-font-mono: 'IBM Plex Mono', 'SF Mono', Menlo, monospace;

    /* Force Streamlit's theme vars to the lab palette. Without these
       overrides Streamlit reads `prefers-color-scheme: dark` from the OS
       and applies its dark theme to widget internals (multiselects render
       navy, labels go pale, slider stays red). `.streamlit/config.toml`
       would be the idiomatic fix but it's gitignored in this project. */
    --primary-color: #2d5e3d;
    --background-color: #faf7f0;
    --secondary-background-color: #f3efe5;
    --text-color: #1a1815;
}

/* Some Streamlit versions key the dark theme off a `data-theme="dark"`
   attribute on `html`. Reset it so widget internals stay in light mode. */
html[data-theme="dark"], [data-theme="dark"] {
    --primary-color: #2d5e3d;
    --background-color: #faf7f0;
    --secondary-background-color: #f3efe5;
    --text-color: #1a1815;
    color-scheme: light;
}

/* Warm off-white page surface. Streamlit's default background is white;
   override at the outermost container and the app shell. */
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--lab-bg);
}

/* Remove Streamlit's top header element entirely. `toolbarMode = "minimal"`
   empties its contents but the element still occupies vertical space and
   blocks clicks on the wordmark below it. */
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }

/* Body type: Inter sans for the chrome, ink color. Streamlit's defaults
   (Source Sans) get superseded here. Specific overrides for headings /
   numbers happen via component-level classes below. */
html, body, [class*="css"] {
    font-family: var(--lab-font-sans);
    color: var(--lab-ink);
}

/* Page padding — design calls for 24px / 56px at 1400px width. Streamlit's
   default block-container caps content width; nudge horizontal padding up
   without removing the cap. */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    padding-left: 3.5rem;
    padding-right: 3.5rem;
    max-width: 1400px;
}

/* ----- Lab component classes ----- */

/* Eyebrow / small-caps section label. Used by `SectionLabel` and the page
   title's NOTEBOOK · /ROUTE eyebrow. */
.lab-eyebrow {
    font-family: var(--lab-font-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--lab-ink-muted);
}

/* H1 — Source Serif 4 page title. */
.lab-h1 {
    font-family: var(--lab-font-serif);
    font-size: 42px;
    font-weight: 400;
    line-height: 1.1;
    color: var(--lab-ink);
    margin: 4px 0 8px 0;
    letter-spacing: -0.01em;
}

/* Italic serif subtitle that follows H1 (may include inline `code` math). */
.lab-subtitle {
    font-family: var(--lab-font-serif);
    font-style: italic;
    font-size: 14px;
    line-height: 1.5;
    color: var(--lab-ink-soft);
    max-width: 72ch;
}
.lab-subtitle code {
    font-family: var(--lab-font-mono);
    font-style: normal;
    font-size: 13px;
    background: transparent;
    color: var(--lab-ink);
    padding: 0;
}

/* Right-aligned mono caption used in the page title row for corpus identity. */
.lab-corpus-caption {
    font-family: var(--lab-font-mono);
    font-size: 11px;
    color: var(--lab-ink-muted);
    text-align: right;
}

/* Section header (§01 STARTING ROSTER · 4 OF 6 SET …). A flex row with the
   §N + title on the left and an optional right slot. Followed by a 1px rule. */
.lab-section {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--lab-rule);
    margin: 28px 0 18px 0;
}
.lab-section-title {
    font-family: var(--lab-font-sans);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--lab-ink);
}
.lab-section-num {
    font-family: var(--lab-font-mono);
    font-size: 11px;
    font-weight: 500;
    color: var(--lab-ink-muted);
    margin-right: 10px;
}
.lab-section-right {
    font-family: var(--lab-font-mono);
    font-size: 11px;
    color: var(--lab-ink-muted);
}

/* Stat cell (single foregrounded number). */
.lab-stat-label {
    font-family: var(--lab-font-sans);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--lab-ink-muted);
    margin-bottom: 4px;
}
.lab-stat-value {
    font-family: var(--lab-font-mono);
    font-size: 22px;
    font-weight: 500;
    color: var(--lab-ink);
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
}
.lab-stat-sub {
    font-family: var(--lab-font-mono);
    font-size: 10px;
    color: var(--lab-ink-muted);
    margin-top: 4px;
}

/* Score chip (signed pill). Color is set inline via `style="color: …"`. */
.lab-chip {
    font-family: var(--lab-font-mono);
    font-size: 12px;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    display: inline-block;
}

/* Signed bar — bipolar around a zero midline. */
.lab-bar-wrap {
    display: inline-block;
    position: relative;
    height: 8px;
    background: var(--lab-rule-soft);
    border-radius: 1px;
    vertical-align: middle;
}
.lab-bar-mid {
    position: absolute;
    top: -2px; bottom: -2px;
    width: 1px;
    background: var(--lab-rule);
}
.lab-bar-fill {
    position: absolute;
    top: 0; bottom: 0;
    border-radius: 1px;
}

/* Unipolar mini bar (e.g. m̂ percentages). */
.lab-minibar-wrap {
    display: inline-block;
    height: 6px;
    background: var(--lab-rule-soft);
    border-radius: 1px;
    vertical-align: middle;
}
.lab-minibar-fill {
    height: 100%;
    background: var(--lab-accent);
    border-radius: 1px;
}

/* Numeric tables — markdown tables get tabular-nums on numeric columns
   via a wrapping `.lab-table` div. The `.stMarkdown table` selector keeps
   it scoped so we don't break other Streamlit markdown tables. */
.lab-table table {
    font-family: var(--lab-font-sans);
    font-size: 13px;
    border-collapse: collapse;
    width: 100%;
}
.lab-table th {
    font-family: var(--lab-font-sans);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--lab-ink-muted);
    text-align: left;
    border-bottom: 1px solid var(--lab-rule);
    padding: 8px 10px;
}
.lab-table td {
    font-variant-numeric: tabular-nums;
    border-bottom: 1px solid var(--lab-rule-soft);
    padding: 10px;
    color: var(--lab-ink);
}
.lab-table td.num,
.lab-table th.num { text-align: right; font-family: var(--lab-font-mono); }
.lab-table td.name { font-family: var(--lab-font-serif); font-size: 14px; }

/* Sprite tile fallback box (when CDN 404s). Per user: missingno fallback. */
.lab-sprite {
    width: 64px;
    height: 64px;
    object-fit: contain;
    image-rendering: -webkit-optimize-contrast;
}

/* ----- Streamlit widget overrides ----- */

/* Header row: wordmark on left, phase picker on right. Tabs sit below. */
.lab-wordmark {
    font-family: var(--lab-font-sans);
    font-size: 20px;
    font-weight: 600;
    color: var(--lab-ink);
    letter-spacing: -0.01em;
    padding-top: 4px;
}
.lab-wordmark .lab-wordmark-mono {
    font-family: var(--lab-font-mono);
    font-size: 13px;
    font-weight: 400;
    color: var(--lab-ink-muted);
    margin-left: 6px;
}

/* Re-skin st.tabs labels: Inter uppercase, forest-green underline on active. */
.stTabs [data-baseweb="tab-list"] {
    gap: 32px;
    border-bottom: 1px solid var(--lab-rule);
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 4px;
    background: transparent;
    border: none;
}
.stTabs [data-baseweb="tab"] p {
    font-family: var(--lab-font-sans);
    font-size: 13px;
    font-weight: 500;
    color: var(--lab-ink-muted);
    letter-spacing: 0.02em;
    margin: 0;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] p {
    color: var(--lab-ink);
    font-weight: 600;
}
.stTabs [data-baseweb="tab-highlight"] {
    background: var(--lab-accent);
    height: 4px;
}

/* Phase picker — Streamlit doesn't let us wrap a widget in a custom-class
   div directly, so we emit a marker element (`.lab-phase-picker-marker`)
   right before the radio and target the ancestor column via `:has()`.
   Within that scope, hide the native radio dot and turn each option into
   a boxed segment with active ink-fill (matches the design's top-right
   tri-box). */

/* Vertically center the picker column relative to the wordmark column. */
[data-testid="stHorizontalBlock"]:has(.lab-phase-picker-marker) {
    align-items: center;
}

/* Right-align the picker within its column. */
[data-testid="stColumn"]:has(.lab-phase-picker-marker)
    [data-testid="stVerticalBlock"] {
    align-items: flex-end;
}

[data-testid="stColumn"]:has(.lab-phase-picker-marker)
    [data-testid="stWidgetLabel"] { display: none; }
[data-testid="stColumn"]:has(.lab-phase-picker-marker)
    [role="radiogroup"] {
    display: flex;
    gap: 0;
}
[data-testid="stColumn"]:has(.lab-phase-picker-marker)
    [role="radiogroup"] label {
    background: var(--lab-panel);
    border: 1px solid var(--lab-rule);
    padding: 6px 14px;
    cursor: pointer;
    margin: 0;
    border-right: none;
    transition: background 80ms ease;
}
[data-testid="stColumn"]:has(.lab-phase-picker-marker)
    [role="radiogroup"] label:first-of-type { border-radius: 3px 0 0 3px; }
[data-testid="stColumn"]:has(.lab-phase-picker-marker)
    [role="radiogroup"] label:last-of-type {
    border-radius: 0 3px 3px 0;
    border-right: 1px solid var(--lab-rule);
}
[data-testid="stColumn"]:has(.lab-phase-picker-marker)
    [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stColumn"]:has(.lab-phase-picker-marker)
    [role="radiogroup"] label p {
    font-family: var(--lab-font-sans);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.04em;
    color: var(--lab-ink-soft);
    margin: 0;
    line-height: 1.3;
}
[data-testid="stColumn"]:has(.lab-phase-picker-marker)
    [role="radiogroup"] label:has(input:checked) {
    background: var(--lab-ink);
}
[data-testid="stColumn"]:has(.lab-phase-picker-marker)
    [role="radiogroup"] label:has(input:checked) p {
    color: var(--lab-bg);
    font-weight: 600;
}

/* Segmented control for the sampler technique picker. Same marker pattern:
   emit `.lab-segmented-marker` before the radio, scope styles to the
   containing block. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .lab-segmented-marker)
    > [data-testid="stElementContainer"]:has(> [data-testid="stRadio"])
    [data-testid="stWidgetLabel"] { display: none; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .lab-segmented-marker)
    > [data-testid="stElementContainer"]:has(> [data-testid="stRadio"])
    [role="radiogroup"] {
    display: flex;
    gap: 0;
    background: var(--lab-panel-alt);
    border-radius: 3px;
    padding: 3px;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .lab-segmented-marker)
    > [data-testid="stElementContainer"]:has(> [data-testid="stRadio"])
    [role="radiogroup"] label {
    flex: 1;
    text-align: center;
    padding: 8px 12px;
    margin: 0;
    cursor: pointer;
    border-radius: 2px;
    transition: background 80ms ease;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .lab-segmented-marker)
    > [data-testid="stElementContainer"]:has(> [data-testid="stRadio"])
    [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .lab-segmented-marker)
    > [data-testid="stElementContainer"]:has(> [data-testid="stRadio"])
    [role="radiogroup"] label p {
    font-family: var(--lab-font-sans);
    font-size: 12px;
    font-weight: 500;
    color: var(--lab-ink-soft);
    margin: 0;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .lab-segmented-marker)
    > [data-testid="stElementContainer"]:has(> [data-testid="stRadio"])
    [role="radiogroup"] label:has(input:checked) {
    background: var(--lab-ink);
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .lab-segmented-marker)
    > [data-testid="stElementContainer"]:has(> [data-testid="stRadio"])
    [role="radiogroup"] label:has(input:checked) p {
    color: var(--lab-bg);
    font-weight: 600;
}

/* Primary button — full-width, forest-green fill, mono uppercase label. */
.stButton button[kind="primary"] {
    background: var(--lab-accent);
    color: var(--lab-bg);
    border: none;
    border-radius: 3px;
    font-family: var(--lab-font-sans);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    padding: 14px;
    transition: filter 80ms ease;
}
.stButton button[kind="primary"]:hover {
    filter: brightness(1.05);
    color: var(--lab-bg);
}

/* ----- Slot-card row (§01 starting roster) ----- */

.lab-slot-strip {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
    margin-bottom: 12px;
}
.lab-slot {
    background: var(--lab-panel);
    border: 1px solid var(--lab-rule);
    border-top: 3px solid var(--lab-ink);
    border-radius: 0 0 3px 3px;
    padding: 12px 10px 10px;
    text-align: center;
    position: relative;
    min-height: 168px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
}
.lab-slot-indicator {
    position: absolute;
    top: 6px;
    right: 8px;
    font-family: var(--lab-font-mono);
    font-size: 9px;
    color: var(--lab-ink-muted);
    letter-spacing: 0.04em;
}
.lab-slot-sprite {
    width: 72px;
    height: 72px;
    object-fit: contain;
}
.lab-slot-name {
    font-family: var(--lab-font-serif);
    font-size: 13px;
    color: var(--lab-ink);
    line-height: 1.2;
    margin-top: 4px;
    text-align: center;
}
.lab-slot-divider {
    border: none;
    border-top: 1px dotted var(--lab-rule);
    width: 100%;
    margin: 8px 0 6px 0;
}
.lab-slot-item {
    font-family: var(--lab-font-mono);
    font-size: 10px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--lab-ink-muted);
    margin-top: auto;
}
/* Empty-slot variant — no top border accent, just centered "FREE SLOT" + ordinal. */
.lab-slot-empty {
    background: transparent;
    border: none;
    border-top: none;
    text-align: center;
    min-height: 168px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 12px 10px;
}
.lab-slot-empty-label {
    font-family: var(--lab-font-sans);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--lab-ink-muted);
    margin-bottom: 12px;
}
.lab-slot-empty-ord {
    font-family: var(--lab-font-mono);
    font-size: 28px;
    color: var(--lab-ink-soft);
    letter-spacing: 0.1em;
}

/* Excluded-tag row. */
.lab-excluded-row {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin: 4px 0 8px 0;
}
.lab-excluded-label {
    font-family: var(--lab-font-sans);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--lab-ink-muted);
}
.lab-excluded-tag {
    font-family: var(--lab-font-sans);
    font-size: 11px;
    color: var(--lab-neg);
    border: 1px solid var(--lab-neg);
    background: var(--lab-panel);
    padding: 3px 10px;
    border-radius: 2px;
}
.lab-excluded-note {
    font-family: var(--lab-font-serif);
    font-style: italic;
    font-size: 12px;
    color: var(--lab-ink-muted);
}

/* ----- Stat strip (observables, fitted-model summary) ----- */

.lab-stat-strip {
    display: grid;
    gap: 28px;
    background: var(--lab-panel);
    border: 1px solid var(--lab-rule);
    border-radius: 3px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

/* ----- Rich-row completion table (§04) ----- */

.lab-comp-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--lab-font-sans);
    font-size: 13px;
}
.lab-comp-table th {
    font-family: var(--lab-font-sans);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--lab-ink-muted);
    border-bottom: 1px solid var(--lab-rule);
    padding: 10px 12px;
    text-align: left;
}
.lab-comp-table th.num { text-align: right; }
.lab-comp-table td {
    border-bottom: 1px solid var(--lab-rule-soft);
    padding: 12px;
    vertical-align: middle;
    font-variant-numeric: tabular-nums;
    color: var(--lab-ink);
}
.lab-comp-table td.num { text-align: right; font-family: var(--lab-font-mono); }
.lab-comp-table td.rank { font-family: var(--lab-font-mono); color: var(--lab-ink-muted); width: 40px; }
.lab-comp-table tr.top-row { background: color-mix(in oklch, var(--lab-accent) 8%, transparent); }
.lab-comp-pair {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.lab-comp-mon {
    display: flex;
    align-items: center;
    gap: 10px;
}
.lab-comp-mon img {
    width: 36px;
    height: 36px;
    object-fit: contain;
}
.lab-comp-mon-name {
    font-family: var(--lab-font-serif);
    font-size: 13px;
    color: var(--lab-ink);
    line-height: 1.2;
}
.lab-comp-mon-item {
    font-family: var(--lab-font-mono);
    font-size: 10px;
    letter-spacing: 0.04em;
    color: var(--lab-ink-muted);
}
.lab-comp-freq {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
}
.lab-comp-freq-pct {
    font-family: var(--lab-font-mono);
    font-size: 13px;
    color: var(--lab-ink);
}

/* Δswaps badge — colored pill. */
.lab-badge {
    display: inline-block;
    font-family: var(--lab-font-mono);
    font-size: 10px;
    letter-spacing: 0.06em;
    text-transform: lowercase;
    padding: 2px 8px;
    border-radius: 2px;
    border: 1px solid currentColor;
}
.lab-badge-seen { color: var(--lab-pos); }
.lab-badge-near { color: var(--lab-warn); }
.lab-badge-far  { color: var(--lab-neg); }

/* Sub-section headings used between sibling tables (e.g. Top +J / Top −J).
   Same letter-spacing as section labels but colored by semantic. */
.lab-subheading {
    font-family: var(--lab-font-sans);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 4px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid currentColor;
}
.lab-subheading-pos { color: var(--lab-pos); }
.lab-subheading-neg { color: var(--lab-neg); }

/* ----- Multiselect chips (Streamlit BaseWeb tags) — light-themed ----- */
[data-baseweb="tag"] {
    background: var(--lab-panel) !important;
    color: var(--lab-ink) !important;
    border: 1px solid var(--lab-rule) !important;
    font-family: var(--lab-font-sans) !important;
    font-size: 11px !important;
    border-radius: 2px !important;
    display: inline-flex !important;
    align-items: center !important;
    line-height: 1 !important;
    height: 24px !important;
    padding: 0 8px !important;
}
/* Inner label span — reset margin/padding that BaseWeb adds by default. */
[data-baseweb="tag"] > span:first-child {
    line-height: 1 !important;
    padding: 0 !important;
    margin: 0 !important;
}
</style>
"""


def inject() -> None:
    """Inject the design-tokens stylesheet into the current Streamlit page.

    Idempotent within a Streamlit script run (the ``<style>`` block is
    written into the DOM each rerun, but multiple identical blocks are
    harmless). Call once at the top of :func:`app.main`.
    """
    st.markdown(_css(), unsafe_allow_html=True)
