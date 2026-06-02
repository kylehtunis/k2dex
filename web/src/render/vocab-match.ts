// Slug-based vocab matching: turns external names (pasted pokepastes,
// shared-link slugs) into model vocab indices.
//
// The bridge is speciesToSlug / itemToSlug (sprite-url.ts, parity-gated):
// slugifying BOTH the external name and each vocab entry's speciesOf[i]
// collapses Showdown forme naming ("Ninetales-Alola") and corpus naming
// ("Alolan Ninetales") onto the same key ("ninetales-alola").
//
// Out-of-vocab species/items are surfaced gently (warnings, not errors) —
// the corpus vocab is format-filtered, so a valid paste can name mons or
// items this model never saw.

import { speciesToSlug, itemToSlug } from "./sprite-url";
import { TEAM_SIZE } from "../constants";
import type { IsingModel } from "../sampler/types";

/** Lookup from species slug to candidate vocab indices (best marginal first). */
export interface SlugIndex {
  bySpecies: Map<string, number[]>;
}

export function buildSlugIndex(model: IsingModel): SlugIndex {
  const bySpecies = new Map<string, number[]>();
  for (let i = 0; i < model.V; i++) {
    const slug = speciesToSlug(model.speciesOf[i]);
    const arr = bySpecies.get(slug);
    if (arr) arr.push(i);
    else bySpecies.set(slug, [i]);
  }
  for (const arr of bySpecies.values()) {
    arr.sort((a, b) => model.m[b] - model.m[a]);
  }
  return { bySpecies };
}

export interface ResolveResult {
  idx: number | null;
  warning: string | null;
}

/** Resolve a (speciesSlug, itemSlug) pair to a vocab index.
 * Unknown species -> idx:null + warning. Item given but unmatched ->
 * best-marginal candidate for that species + a gentle warning. Itemless
 * models/candidates -> best candidate. `labels` supply friendly display
 * names for the messages (fall back to the slugs). */
export function resolveFeature(
  slugIndex: SlugIndex,
  model: IsingModel,
  speciesSlug: string,
  itemSlug: string | null,
  labels?: { species?: string; item?: string },
): ResolveResult {
  const candidates = slugIndex.bySpecies.get(speciesSlug);
  const speciesLabel = labels?.species ?? speciesSlug;
  if (!candidates || candidates.length === 0) {
    return { idx: null, warning: `${speciesLabel} is not in this model — skipped.` };
  }
  if (itemSlug) {
    for (const i of candidates) {
      const it = model.itemOf[i];
      if (it !== null && itemToSlug(it) === itemSlug) {
        return { idx: i, warning: null };
      }
    }
    const itemLabel = labels?.item ?? itemSlug;
    return {
      idx: candidates[0],
      warning: `${itemLabel} on ${speciesLabel} isn't in this model — matched the species only.`,
    };
  }
  return { idx: candidates[0], warning: null };
}

/** Resolve a species slug to its corpus display name (for excluded lists). */
export function resolveSpeciesSlug(
  slugIndex: SlugIndex,
  model: IsingModel,
  speciesSlug: string,
): string | null {
  const candidates = slugIndex.bySpecies.get(speciesSlug);
  if (!candidates || candidates.length === 0) return null;
  return model.speciesOf[candidates[0]];
}

export interface ParsedMon {
  rawSpecies: string;
  rawItem: string | null;
  speciesSlug: string;
  itemSlug: string | null;
}

// Showdown exports mega-evolved species under their mega forme name
// ("Charizard-Mega-Y", "Blastoise-Mega"), but the corpus buckets them
// under the base species — the held Mega Stone is the source of truth for
// which forme (mirrors limitless_ingest.strip_mega_prefix, which collapses
// the Limitless API's *prefix* form). Stripping the suffix on import lets
// the slug bridge to the base species.
//
// A few formes the corpus also collapses don't reduce by suffix-stripping
// alone and need an explicit slug pre-map: "Eternal Flower Floette" is
// corpus-slugged to "floette" (SLUG_OVERRIDES in sprite-url.ts), but
// Showdown can export that mon as "Floette", "Floette-Mega", or
// "Floette-Eternal" — the first two reduce via the mega strip, the third
// is aliased here.
const IMPORT_SPECIES_ALIASES: Record<string, string> = {
  "floette-eternal": "floette",
};

/** Slugify a pasted Showdown species name, bridging mega/forme naming to
 * the base-species slug the corpus stores. */
function importSpeciesSlug(species: string): string {
  const base = species.replace(/-Mega(-[XYZ])?$/i, "");
  const slug = speciesToSlug(base);
  return IMPORT_SPECIES_ALIASES[slug] ?? slug;
}

/** Parse the species line of a Showdown / pokepaste block. Handles
 * "Species @ Item", "Nickname (Species) @ Item", and a trailing gender
 * marker " (M)" / " (F)". Returns null if no species is present. */
function parseSpeciesLine(line: string): ParsedMon | null {
  let item: string | null = null;
  let namePart = line;
  const at = line.lastIndexOf(" @ ");
  if (at >= 0) {
    namePart = line.slice(0, at);
    const rest = line.slice(at + 3).trim();
    item = rest.length ? rest : null;
  }
  namePart = namePart.trim();
  // Strip a trailing gender marker.
  namePart = namePart.replace(/\s*\((?:M|F)\)\s*$/i, "").trim();
  // Nickname form: "Nickname (Species)" -> species is inside the parens.
  const paren = namePart.match(/^.*\s+\(([^)]+)\)\s*$/);
  const species = (paren ? paren[1] : namePart).trim();
  if (!species) return null;
  return {
    rawSpecies: species,
    rawItem: item,
    speciesSlug: importSpeciesSlug(species),
    itemSlug: item ? itemToSlug(item) : null,
  };
}

/** Parse a full pokepaste into per-mon species/item slugs. Blocks are
 * blank-line separated; only each block's first non-empty line matters
 * (ability/moves/EVs are discarded). */
export function parsePokepaste(text: string): ParsedMon[] {
  const out: ParsedMon[] = [];
  for (const block of text.split(/\n\s*\n/)) {
    const firstLine = block
      .split("\n")
      .map((l) => l.trim())
      .find((l) => l.length > 0);
    if (!firstLine) continue;
    const parsed = parseSpeciesLine(firstLine);
    if (parsed) out.push(parsed);
  }
  return out;
}

export interface MatchResult {
  idxs: number[];
  errors: string[];
  warnings: string[];
}

/** Parse pasted text and resolve it to a team of vocab indices, deduping
 * by species and capping at TEAM_SIZE. Empty/unmatched -> error; partial
 * out-of-vocab -> warnings plus whatever matched. */
export function matchPaste(model: IsingModel, text: string): MatchResult {
  const parsed = parsePokepaste(text);
  if (parsed.length === 0) {
    return { idxs: [], errors: ["No Pokémon found in the clipboard."], warnings: [] };
  }
  const slugIndex = buildSlugIndex(model);
  const idxs: number[] = [];
  const warnings: string[] = [];
  const seenSpecies = new Set<string>();
  for (const p of parsed) {
    if (idxs.length >= TEAM_SIZE) break;
    const r = resolveFeature(slugIndex, model, p.speciesSlug, p.itemSlug, {
      species: p.rawSpecies,
      item: p.rawItem ?? undefined,
    });
    if (r.idx === null) {
      if (r.warning) warnings.push(r.warning);
      continue;
    }
    if (seenSpecies.has(p.speciesSlug)) continue;
    seenSpecies.add(p.speciesSlug);
    idxs.push(r.idx);
    if (r.warning) warnings.push(r.warning);
  }
  const errors =
    idxs.length === 0
      ? ["No Pokémon from the clipboard matched this model."]
      : [];
  return { idxs, errors, warnings };
}
