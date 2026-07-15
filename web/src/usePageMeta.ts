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

import { useModel } from "./state/ModelContext";
import {
  SITE_NAME,
  canonicalUrl,
  metaForPath,
  metaRouteMeta,
  normalizePath,
  speciesPageSlug,
  speciesRouteMeta,
  type RouteMeta,
} from "./siteMeta";

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
  const { model } = useModel();
  useEffect(() => {
    const path = normalizePath(pathname);
    let meta: RouteMeta = metaForPath(path);
    // Species pages aren't in the static ROUTE_META (they're derived from the
    // model); resolve their meta dynamically. Until the model loads, leave the
    // head alone: on a hard load it already carries the prerendered values.
    const species = path.match(/^pokemon\/([^/]+)$/);
    if (species) {
      if (!model) return;
      const name = model.sites.find((s) => speciesPageSlug(s) === species[1]);
      if (name) meta = speciesRouteMeta(name, model.regulation);
    }
    // /meta carries the active regulation in its title once the model is
    // loaded (mirrors the prerendered head).
    if (path === "meta" && model) meta = metaRouteMeta(model.regulation);
    const url = canonicalUrl(path);

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
  }, [pathname, model]);
}
