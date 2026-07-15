// The route tree, shared by two consumers:
//   - App.tsx wraps it in BrowserRouter + live providers (the real app)
//   - scripts/prerender-routes.ts wraps it in StaticRouter + a preloaded
//     StaticModelProvider to render each route's HTML at build time
// Keeping the tree in one place means the prerendered pages can never drift
// from what the app actually routes.

import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { HomePage } from "./pages/HomePage";
import { CompleterPage } from "./pages/CompleterPage";
import { AnalysisPage } from "./pages/AnalysisPage";
import { MetaPage } from "./pages/MetaPage";
import { PokedexPage } from "./pages/PokedexPage";
import { PokemonPage } from "./pages/PokemonPage";

const ArticlesPage = lazy(() => import("./pages/ArticlesPage"));
const ArticlePage = lazy(() => import("./pages/ArticlePage"));

export function AppRoutes() {
  return (
    <Suspense
      fallback={
        <p style={{ color: "var(--lab-ink-muted)", padding: "32px 24px" }}>Loading…</p>
      }
    >
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="completer" element={<CompleterPage />} />
          <Route path="analysis" element={<AnalysisPage />} />
          <Route path="meta" element={<MetaPage />} />
          <Route path="pokemon" element={<PokedexPage />} />
          <Route path="pokemon/:slug" element={<PokemonPage />} />
          <Route path="articles" element={<ArticlesPage />} />
          <Route path="articles/:slug" element={<ArticlePage />} />
          {/* Legacy /science URL: the explainer now lives as an article. */}
          <Route
            path="science"
            element={<Navigate to="/articles/the-science-of-k2dex/" replace />}
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
