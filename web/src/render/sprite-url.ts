// Showdown home-sprite URL builder + slug rules.
//
// Mirrors rendering_html.species_to_slug, _HYPHEN_BASE_SPECIES, sprite_url.
// Slug rules are duplicated across stacks — parity gate in
// tests/test_parity.py::test_species_to_slug_cases.

import { extractSpecies } from "./format";

const SPRITE_CDN = "https://play.pokemonshowdown.com/sprites/home";

/** Species whose canonical name *contains* a hyphen as part of the
 * base (not as a forme separator). All hyphens are stripped for these
 * to match Showdown's home-folder convention. */
const HYPHEN_BASE_SPECIES = new Set([
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
]);

/** Species whose canonical corpus name doesn't follow normal slug rules.
 * Keys are the lowercased canonical name; values are the Showdown slug. */
const SLUG_OVERRIDES: Record<string, string> = {
  "eternal flower floette": "floette",
  "paldean tauros aqua breed": "tauros-paldeaaqua",
  "paldean tauros blaze breed": "tauros-paldeablaze",
  "paldean tauros combat breed": "tauros-paldeacombat",
};

// Limitless stores regional formes as "Adjective Species" (e.g. "Alolan Ninetales").
const REGIONAL_ADJECTIVE: Record<string, string> = {
  alolan: "alola",
  galarian: "galar",
  hisuian: "hisui",
  paldean: "paldea",
};

// Limitless stores Rotom formes as "Forme Rotom" (e.g. "Wash Rotom").
const ROTOM_FORME_NAMES = new Set(["wash", "heat", "frost", "mow", "fan"]);

/** Convert a display species name to Showdown's home-sprite slug.
 *
 * Rules (parity with rendering_html.species_to_slug):
 *  1. Lowercase; strip apostrophes / periods / punctuation.
 *  2. "Adjective Species" regional forms reordered: "Alolan X" → "x-alola".
 *  3. "Forme Rotom" reordered: "Wash Rotom" → "rotom-wash".
 *  4. Collapse internal whitespace to nothing.
 *  5. If the result is in HYPHEN_BASE_SPECIES, strip all hyphens.
 *  6. Otherwise first hyphen is a forme separator (kept); subsequent
 *     hyphens are collapsed.
 */
export function speciesToSlug(name: string): string {
  const override = SLUG_OVERRIDES[name.toLowerCase()];
  if (override !== undefined) return override;
  // Keep alphanumerics, hyphens, and whitespace. Drop everything else.
  const cleaned = name.toLowerCase().replace(/[^a-z0-9\s-]+/g, "").trim();
  const words = cleaned.split(/\s+/).filter(Boolean);
  if (words.length >= 2) {
    // "Alolan Ninetales" → "ninetales-alola", "Hisuian Arcanine" → "arcanine-hisui"
    const region = REGIONAL_ADJECTIVE[words[0]];
    if (region !== undefined) {
      const base = words.slice(1).join("");
      return `${base}-${region}`;
    }
    // "Wash Rotom" → "rotom-wash", "Heat Rotom" → "rotom-heat"
    if (words.length === 2 && words[1] === "rotom" && ROTOM_FORME_NAMES.has(words[0])) {
      return `rotom-${words[0]}`;
    }
  }
  // Whitespace collapses to nothing.
  const noSpaces = cleaned.replace(/\s+/g, "");
  if (!noSpaces.includes("-")) return noSpaces;
  if (HYPHEN_BASE_SPECIES.has(noSpaces)) return noSpaces.replace(/-/g, "");
  const parts = noSpaces.split("-");
  const head = parts[0];
  const rest = parts.slice(1).join("");
  return `${head}-${rest}`;
}

/** Showdown home-sprite URL for a vocab string (species or "Species @ Item"). */
export function spriteUrl(name: string): string {
  return `${SPRITE_CDN}/${speciesToSlug(extractSpecies(name))}.png`;
}
