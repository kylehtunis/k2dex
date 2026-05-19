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

/** Convert a display species name to Showdown's home-sprite slug.
 *
 * Rules (parity with rendering_html.species_to_slug):
 *  1. Lowercase; strip apostrophes / periods / punctuation.
 *  2. Collapse internal whitespace to nothing.
 *  3. If the result is in HYPHEN_BASE_SPECIES, strip all hyphens.
 *  4. Otherwise first hyphen is a forme separator (kept); subsequent
 *     hyphens are collapsed.
 */
export function speciesToSlug(name: string): string {
  // Keep alphanumerics, hyphens, and whitespace. Drop everything else.
  const cleaned = name.toLowerCase().replace(/[^a-z0-9\s-]+/g, "");
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
