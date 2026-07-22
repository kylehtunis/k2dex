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

// Vocab strings are "Species @ Item" (bare species if itemless), with an
// optional trailing " (Ability)" (schema v4; current committed artifacts are
// still item-only, so the parenthetical may be absent). Species names never
// contain parens, so the ability group — when present — is always the
// trailing "(...)" and is stripped first before the @ split.
function stripAbility(feature: string): string {
  const m = feature.match(/^(.*) \([^()]+\)$/);
  return m ? m[1] : feature;
}

/** Pull the species name out of a vocab string (ability parenthetical, if
 * any, stripped first). */
export function extractSpecies(feature: string): string {
  const base = stripAbility(feature);
  const i = base.indexOf(" @ ");
  return i < 0 ? base : base.slice(0, i);
}

/** Pull the item out of a vocab string, or null if itemless (ability
 * parenthetical, if any, stripped first). */
export function extractItem(feature: string): string | null {
  const base = stripAbility(feature);
  const i = base.indexOf(" @ ");
  return i < 0 ? null : base.slice(i + 3);
}

/** Pull the ability out of a vocab string, or null when the vocab has no
 * ability track (current committed artifacts). */
export function extractAbility(feature: string): string | null {
  const m = feature.match(/ \(([^()]+)\)$/);
  return m ? m[1] : null;
}

/** Inverse of extractSpecies / extractItem. Used when client-side code
 * needs to reconstruct a vocab string from its parts (currently rare;
 * mostly the model loader carries vocab through pre-built). */
export function formatPair(species: string, item: string | null): string {
  return item === null ? species : `${species} @ ${item}`;
}

/** Reconstruct a (species, item, ability) vocab string:
 * "Species @ Item (Ability)", with the ability always appended (every member
 * has one). An itemless member — item the string "None" or null — drops the
 * "@ Item" part exactly as formatPair does, keeping only the ability
 * parenthetical: "Species (Ability)". Parity twin of loaders.format_triple. */
export function formatTriple(
  species: string,
  item: string | null,
  ability: string,
): string {
  const base = item === null || item === "None" ? species : `${species} @ ${item}`;
  return `${base} (${ability})`;
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
