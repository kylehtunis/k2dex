// Single source of truth for per-route page metadata (title, description,
// canonical URL). Consumed in two places:
//   - the app, via usePageMeta(), to set document.title / description /
//     canonical / OG tags on client-side navigation so the live DOM matches
//     the static file the crawler first received.
//   - scripts/prerender-routes.ts, which at build time writes a real
//     dist/<route>/index.html per route (HTTP 200 with a self-referencing
//     canonical and a unique title) plus dist/sitemap.xml.
//
// Why this file exists: the app is a client-routed SPA on static hosting
// (GitHub Pages). Without per-route static HTML, every subroute returns a 404
// to crawlers (the 404.html SPA-redirect is only for humans) and the single
// index.html declares the homepage as everyone's canonical, so Google
// collapses the whole site into one indexed URL. Adding a route here — and
// only here — keeps the prerendered HTML, the runtime <head>, and the sitemap
// in sync.

/** Production origin + base path, no trailing slash. Fixed regardless of the
 *  build's base path (local builds use "/"); canonicals must always point at
 *  the real public host. */
export const SITE_URL = "https://k2dex.kyletunis.com";
export const SITE_NAME = "k2dex";

import { ARTICLES, articlePath } from "./articles/articles";

export interface RouteMeta {
  /** Path relative to the app base, no slashes. "" is the home page. */
  path: string;
  title: string;
  description: string;
  /** Absolute canonical URL override. Set for routes that redirect elsewhere
   *  (e.g. legacy /science → the article) so crawlers consolidate onto the
   *  target. When present, the route is also omitted from the sitemap. */
  canonical?: string;
}

/** Primary, indexable routes. Article routes are appended below from the shared
 *  ARTICLES list; the legacy /science redirect is appended after that. */
const BASE_ROUTES: readonly RouteMeta[] = [
  {
    path: "",
    title: "k2dex | VGC Team Builder with Autocomplete & Meta Analysis",
    description:
      "k2dex is a VGC team builder that autocompletes partial teams using a statistical physics model learned from thousands of real tournament teams. Analyze team synergy and explore the metagame.",
  },
  {
    path: "completer",
    title: "VGC Team Builder & Autocompleter | k2dex",
    description:
      "Free VGC team builder and autocompleter: pick any partial roster and get optimal completions learned from thousands of real tournament teams. Build around any Pokemon.",
  },
  {
    path: "analysis",
    title: "VGC Team Analysis: Rating, Synergies & Suggestions | k2dex",
    description:
      "Rate my VGC team: score and analyze a competitive Pokemon team, see its synergy and coherence, suggested swaps, and similar real tournament teams.",
  },
  // Fallback for /meta before the model loads on client-side navigation; the
  // prerendered file and the loaded-model runtime head use metaRouteMeta()
  // below, which carries the active regulation for "reg X teams" queries.
  {
    path: "meta",
    title: "VGC Metagame Stats: Top Teams & Synergies | k2dex",
    description:
      "The VGC metagame at a glance: the most common tournament teams and the strongest Pokemon synergies and anti-synergies in the format.",
  },
  {
    path: "pokemon",
    title: "VGC Pokémon Index | k2dex",
    description:
      "Browse every Pokémon in the current VGC regulation by tournament usage. Click through for best teammates, item builds, synergies, and real tournament teams.",
  },
  {
    path: "articles",
    title: "Articles | k2dex",
    description:
      "Write-ups on the ideas, findings, and implementation details behind k2dex.",
  },
];

/** URL segment for a species page under /pokemon/. Deliberately NOT the sprite
 *  slug (render/sprite-url.ts collapses gendered formes and Lycanroc Midday
 *  onto their base species): words hyphenated, gender symbols mapped to f/m,
 *  so every distinct vocab species gets a distinct segment. The prerender
 *  asserts uniqueness across the model's sites at build time. */
export function speciesPageSlug(species: string): string {
  return species
    .toLowerCase()
    .replace(/♀/g, " f")
    .replace(/♂/g, " m")
    .replace(/[^a-z0-9\s-]+/g, "")
    .trim()
    .split(/\s+/)
    .join("-");
}

/** RouteMeta for /meta with the active regulation in the title, capturing the
 *  time-sensitive "reg X teams" queries. The prerender substitutes it for the
 *  static BASE_ROUTES fallback at build time (so a regulation rotation updates
 *  the shipped title on the next deploy automatically), and usePageMeta does
 *  the same at runtime once the model is loaded. */
export function metaRouteMeta(regulation: string): RouteMeta {
  return {
    path: "meta",
    title: `VGC Metagame Stats: Top Teams & Synergies (Reg ${regulation}) | k2dex`,
    description:
      `The Reg ${regulation} VGC metagame at a glance: the most common tournament teams ` +
      `and the strongest Pokemon synergies and anti-synergies in the format.`,
  };
}

/** RouteMeta for one species page. Shared by the prerender (which appends one
 *  per model site) and usePageMeta (client-side navigation onto the page). */
export function speciesRouteMeta(species: string, regulation: string): RouteMeta {
  return {
    path: `pokemon/${speciesPageSlug(species)}`,
    title: `${species} VGC Teammates & Synergies | k2dex`,
    description:
      `${species} in VGC Regulation ${regulation}: best teammates, strongest synergies and anti-synergies, ` +
      `item builds, and real tournament teams featuring ${species}, computed by the k2dex model.`,
  };
}

/** One indexable route per article, derived from the shared ARTICLES list. */
const ARTICLE_ROUTES: readonly RouteMeta[] = ARTICLES.map((a) => ({
  path: articlePath(a.slug),
  title: a.title,
  description: a.description,
}));

/** Legacy /science URL, retained only to redirect (App.tsx) and to canonicalize
 *  onto the article for any crawler that still holds the old URL. */
const SCIENCE_REDIRECT: RouteMeta = {
  path: "science",
  title: "The Science of k2dex",
  description: ARTICLES.find((a) => a.slug === "the-science-of-k2dex")?.description ?? "",
  canonical: canonicalUrl(articlePath("the-science-of-k2dex")),
};

export const ROUTE_META: readonly RouteMeta[] = [
  ...BASE_ROUTES,
  ...ARTICLE_ROUTES,
  SCIENCE_REDIRECT,
];

/** Strip surrounding slashes so a router pathname ("/completer", "/") maps to
 *  a RouteMeta.path ("completer", ""). */
export function normalizePath(pathname: string): string {
  return pathname.replace(/^\/+/, "").replace(/\/+$/, "");
}

/** Canonical URL for a route path. Every URL gets a trailing slash — GitHub
 *  Pages serves directory index.html files at the trailing-slash URL and 301s
 *  the bare path, so canonicals must match the final served URL. */
export function canonicalUrl(path: string): string {
  return path ? `${SITE_URL}/${path}/` : `${SITE_URL}/`;
}

/** RouteMeta for a normalized path, falling back to home for unknown paths
 *  (the app's catch-all route redirects those to home anyway). */
export function metaForPath(path: string): RouteMeta {
  return ROUTE_META.find((m) => m.path === path) ?? ROUTE_META[0];
}
