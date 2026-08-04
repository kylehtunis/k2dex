// The lazily-loaded route modules, in one place.
//
// Three consumers need to agree on exactly this set, and a mismatch between
// them is silent (the page still renders, it just flashes a Suspense fallback
// where a prerendered body used to be):
//   - AppRoutes.tsx      wraps each importer in React.lazy()
//   - main.tsx           awaits the current route's importer before mounting
//   - prerender-routes   stamps <link rel="modulepreload"> from the source path
//
// React-free on purpose so the Node prerender script can import it.
//
// Adding a lazy route means adding an entry here AND teaching each consumer
// which paths map to it; the source paths themselves can no longer drift.

export const ROUTE_CHUNK_IMPORTERS = {
  articlesIndex: () => import("./pages/ArticlesPage"),
  article: () => import("./pages/ArticlePage"),
} as const;

/** Vite-manifest keys (repo-relative source paths) for the same modules. */
export const ROUTE_CHUNK_SOURCES: Record<keyof typeof ROUTE_CHUNK_IMPORTERS, string> = {
  articlesIndex: "src/pages/ArticlesPage.tsx",
  article: "src/pages/ArticlePage.tsx",
};
