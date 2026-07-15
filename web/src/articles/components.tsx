// Slug → lazy article component. Split from ./articles.ts so the metadata list
// stays React-free for the prerender/SEO layer. Each key must match an ARTICLES
// slug; ArticlePage resolves the active slug against this map.

import { lazy, type LazyExoticComponent, type ComponentType } from "react";

// One importer per article body. The same thunk backs the lazy route component
// and main.tsx's pre-mount chunk preload (which keeps the prerendered prose on
// screen instead of a loading fallback during a hard load).
const ARTICLE_IMPORTERS: Record<string, () => Promise<{ default: ComponentType }>> = {
  "the-v2-update": () => import("../pages/PottsUpdatePage"),
  "model-vs-counting": () => import("../pages/ValidationArticlePage"),
  "the-science-of-k2dex": () => import("../pages/SciencePage"),
};

export const ARTICLE_COMPONENTS: Record<string, LazyExoticComponent<ComponentType>> =
  Object.fromEntries(
    Object.entries(ARTICLE_IMPORTERS).map(([slug, importer]) => [slug, lazy(importer)]),
  );

/** Kick off (and await) an article body's chunk download. */
export const ARTICLE_PRELOADS: Record<string, () => Promise<unknown>> = ARTICLE_IMPORTERS;

/** Vite build-manifest keys for each article body's lazy assets (the body
 * module plus any stylesheets it dynamically imports), used by
 * scripts/prerender-routes.tsx to inject preload tags into that article's
 * static HTML. Must mirror ARTICLE_IMPORTERS (and the bodies' own dynamic CSS
 * imports); the prerender fails the build loudly if a key is missing from the
 * manifest. */
export const ARTICLE_SOURCES: Record<string, string[]> = {
  "the-v2-update": ["src/pages/PottsUpdatePage.tsx"],
  "model-vs-counting": ["src/pages/ValidationArticlePage.tsx"],
  "the-science-of-k2dex": [
    "src/pages/SciencePage.tsx",
    "node_modules/katex/dist/katex.min.css",
  ],
};
