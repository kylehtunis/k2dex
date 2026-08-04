import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { ARTICLE_PRELOADS } from "./articles/components";
import { ROUTE_CHUNK_IMPORTERS } from "./routeChunks";
import "./styles/index.css";

// GitHub Pages SPA fallback: 404.html redirects here with ?route=/original-path.
// Restore the clean URL before React mounts so BrowserRouter sees the right path.
const params = new URLSearchParams(location.search);
const route = params.get("route");
const base = import.meta.env.BASE_URL.replace(/\/$/, "");
if (route) {
  history.replaceState(null, "", base + route + location.hash);
}

/** Lazy chunks the current route needs before it can render its content.
 * Mounting replaces the prerendered static HTML wholesale, so waiting for
 * these keeps a hard-loaded article's prose on screen instead of swapping it
 * for a Suspense loading fallback while the chunk downloads. Routes without
 * lazy chunks resolve immediately. */
function routeChunkPreloads(pathname: string): Array<Promise<unknown>> {
  const path = pathname.startsWith(base) ? pathname.slice(base.length) : pathname;
  const m = path.match(/^\/articles(?:\/([^/]+))?\/?$/);
  if (!m) return [];
  if (!m[1]) return [ROUTE_CHUNK_IMPORTERS.articlesIndex()];
  const preloads: Array<Promise<unknown>> = [ROUTE_CHUNK_IMPORTERS.article()];
  const body = ARTICLE_PRELOADS[m[1]];
  if (body) preloads.push(body());
  return preloads;
}

// allSettled, then mount regardless: a failed chunk fetch falls back to the
// normal in-app Suspense/redirect handling.
Promise.allSettled(routeChunkPreloads(location.pathname)).then(() => {
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
});
