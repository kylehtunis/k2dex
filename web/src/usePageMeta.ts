// Keep the document <head> in sync with the active route on client-side
// navigation. The prerendered static files (scripts/prerender-routes.ts) give
// each route the correct head on first paint / for crawlers that don't run JS;
// this hook reproduces the same values when the user navigates within the SPA,
// so the title, canonical, and social tags never go stale after the first load.
//
// Values come from the shared ROUTE_META so the runtime head, the prerendered
// HTML, and the sitemap can never drift apart.

import { useEffect } from "react";
import { useLocation } from "react-router-dom";

import { SITE_NAME, canonicalUrl, metaForPath, normalizePath } from "./siteMeta";

/** Find a matching <head> element (creating it if absent) and apply `mutate`. */
function upsertHeadEl<T extends HTMLElement>(
  selector: string,
  create: () => T,
  mutate: (el: T) => void,
): void {
  let el = document.head.querySelector<T>(selector);
  if (!el) {
    el = create();
    document.head.appendChild(el);
  }
  mutate(el);
}

function setMeta(attr: "name" | "property", key: string, content: string): void {
  upsertHeadEl<HTMLMetaElement>(
    `meta[${attr}="${key}"]`,
    () => {
      const m = document.createElement("meta");
      m.setAttribute(attr, key);
      return m;
    },
    (m) => m.setAttribute("content", content),
  );
}

export function usePageMeta(): void {
  const { pathname } = useLocation();
  useEffect(() => {
    const meta = metaForPath(normalizePath(pathname));
    const url = canonicalUrl(normalizePath(pathname));

    document.title = meta.title;
    setMeta("name", "description", meta.description);
    setMeta("property", "og:site_name", SITE_NAME);
    setMeta("property", "og:title", meta.title);
    setMeta("property", "og:description", meta.description);
    setMeta("property", "og:url", url);
    setMeta("name", "twitter:title", meta.title);
    setMeta("name", "twitter:description", meta.description);
    upsertHeadEl<HTMLLinkElement>(
      'link[rel="canonical"]',
      () => {
        const l = document.createElement("link");
        l.rel = "canonical";
        return l;
      },
      (l) => {
        l.href = url;
      },
    );
  }, [pathname]);
}
