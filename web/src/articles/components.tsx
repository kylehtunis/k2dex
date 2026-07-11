// Slug → lazy article component. Split from ./articles.ts so the metadata list
// stays React-free for the prerender/SEO layer. Each key must match an ARTICLES
// slug; ArticlePage resolves the active slug against this map.

import { lazy, type LazyExoticComponent, type ComponentType } from "react";

export const ARTICLE_COMPONENTS: Record<string, LazyExoticComponent<ComponentType>> = {
  "the-science-of-k2dex": lazy(() => import("../pages/SciencePage")),
};
