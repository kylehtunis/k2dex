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
    title: "k2dex | Competitive Pokemon Team Analysis",
    description:
      "k2dex learns teambuilding patterns from thousands of real VGC tournament teams using a statistical physics model. Explore which Pokemon pair well, complete partial teams, and analyze metagame trends.",
  },
  {
    path: "completer",
    title: "Team Completer | k2dex",
    description:
      "Complete a partial competitive Pokemon (VGC) team or generate full teams. Suggestions are driven by a model trained on thousands of real tournament rosters.",
  },
  {
    path: "analysis",
    title: "Team Analysis | k2dex",
    description:
      "Analyze a competitive Pokemon (VGC) team: see pairwise synergy strengths, overall coherence score, and the closest teams from real tournament results.",
  },
  {
    path: "meta",
    title: "Metagame Statistics | k2dex",
    description:
      "Explore the VGC metagame at a glance: Pokemon usage rates, the strongest synergies and anti-synergies between species, and format-wide coupling statistics.",
  },
  {
    path: "articles",
    title: "Articles | k2dex",
    description:
      "Write-ups on the ideas, findings, and implementation details behind k2dex.",
  },
];

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
