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
export const SITE_URL = "https://kylehtunis.github.io/k2dex";

export interface RouteMeta {
  /** Path relative to the app base, no slashes. "" is the home page. */
  path: string;
  title: string;
  description: string;
}

export const ROUTE_META: readonly RouteMeta[] = [
  {
    path: "",
    title: "k2dex — Competitive Pokemon Team Analysis",
    description:
      "Interactive tools and explainers applying inverse Ising models from statistical physics to competitive Pokemon (VGC) team composition data.",
  },
  {
    path: "completer",
    title: "Team Completer — k2dex",
    description:
      "Suggest the strongest completions for a partial competitive Pokemon (VGC) team, or sample full teams, using a pairwise maximum-entropy model fit to tournament data.",
  },
  {
    path: "analysis",
    title: "Team Analysis — k2dex",
    description:
      "Per-team diagnostics for competitive Pokemon (VGC): pairwise coupling strengths, coherence scores, and nearest observed teams from tournament data.",
  },
  {
    path: "meta",
    title: "Metagame Statistics — k2dex",
    description:
      "Format-wide competitive Pokemon (VGC) metagame statistics: usage biases and the strongest positive and negative species couplings from tournament data.",
  },
  {
    path: "science",
    title: "The Science of k2dex",
    description:
      "An interactive explainer for the statistical-physics methods behind k2dex: Ising models, Metropolis sampling, parallel tempering, mean field, and the inverse problem.",
  },
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
