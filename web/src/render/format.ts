// Number + vocab-string formatting helpers.
//
// Mirrors the private `_fmt_signed` / `_fmt_pct` and the public
// `extract_species` / `extract_item` from rendering_html.py.

export function formatSigned(value: number, decimals = 3): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(decimals)}`;
}

export function formatPct(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Pull the species name out of a Phase 3 vocab string. Phase 3 uses
 * "Species @ Item"; bare species for itemless mons. */
export function extractSpecies(feature: string): string {
  const i = feature.indexOf(" @ ");
  return i < 0 ? feature : feature.slice(0, i);
}

/** Pull the item out of a Phase 3 vocab string, or null if itemless. */
export function extractItem(feature: string): string | null {
  const i = feature.indexOf(" @ ");
  return i < 0 ? null : feature.slice(i + 3);
}

/** Inverse of extractSpecies / extractItem. Used when client-side code
 * needs to reconstruct a vocab string from its parts (currently rare;
 * mostly the model loader carries vocab through pre-built). */
export function formatPair(species: string, item: string | null): string {
  return item === null ? species : `${species} @ ${item}`;
}

/** Partial pokepaste: one line per mon (species slug, or slug @ Item),
 * blank-line separated. Uses canonical Smogon slugs for species names. */
export function buildPartialPaste(
  teamIdxs: readonly number[],
  vocab: readonly string[],
  slugFn: (species: string) => string,
): string {
  const sorted = [...teamIdxs].sort((a, b) => a - b);
  return sorted
    .map((idx) => {
      const entry = vocab[idx];
      const species = extractSpecies(entry);
      const item = extractItem(entry);
      const slug = slugFn(species);
      return item !== null ? `${slug} @ ${item}` : slug;
    })
    .join("\n\n");
}
