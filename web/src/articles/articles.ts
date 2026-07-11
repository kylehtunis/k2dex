// Article registry — metadata only, deliberately free of React imports so the
// Node prerender script (scripts/prerender-routes.ts, via siteMeta.ts) and the
// SEO layer can consume it without pulling in the component tree. The slug →
// lazy component mapping lives separately in ./components.tsx.
//
// Adding an article = one entry here + one entry in ./components.tsx. That keeps
// the /articles index, the /articles/:slug route, the prerendered static HTML,
// and the sitemap all in sync from this single list.

export interface ArticleMeta {
  /** URL segment under /articles/, no slashes. */
  slug: string;
  /** Display title: the index card heading and the document <title>. */
  title: string;
  /** Index-card blurb and the meta description for the article route. */
  description: string;
  /** Publish date, ISO YYYY-MM-DD. Drives newest-first ordering. */
  date: string;
}

export const ARTICLES: readonly ArticleMeta[] = [
  {
    slug: "the-science-of-k2dex",
    title: "The Science of k2dex",
    description:
      "An interactive explainer on the statistical physics behind k2dex. Walk through Ising models, Metropolis sampling, parallel tempering, and mean field with live simulations.",
    date: "2026-05-20",
  },
];

/** Full route path (relative to the app base, no leading slash) for an article. */
export function articlePath(slug: string): string {
  return `articles/${slug}`;
}

/** ARTICLES sorted newest-first for the index listing. */
export function articlesByDate(): ArticleMeta[] {
  return [...ARTICLES].sort((a, b) => b.date.localeCompare(a.date));
}

/** Human-readable publish date, e.g. "May 20, 2026". */
export function formatArticleDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`);
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  });
}
