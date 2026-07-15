// Post-build step: turn the single-page Vite output into per-route static HTML
// so GitHub Pages serves a real 200 (not the 404.html SPA-redirect) for every
// route, each with a unique <title>/description, a self-referencing canonical,
// and — crucially for SEO — the route's actual rendered content in <body>.
// Also (re)writes dist/sitemap.xml from the same route list.
//
// Body prerendering: the default model's committed artifacts are loaded from
// dist/ through the app's own loaders (a fetch shim maps URLs to files), then
// each route renders the real <AppRoutes> tree inside a StaticRouter with a
// preloaded StaticModelProvider. Crawlers and no-JS visitors get the full page;
// the browser app mounts over it via createRoot (replace, not hydrate — the
// client's first render is model-less, so hydration could never match).
//
// Run automatically as the last step of `npm run build` (see package.json).
// Reads the freshly built dist/index.html as the template; asset URLs in it are
// absolute (base-path prefixed) so copies in subfolders resolve correctly.

/// <reference types="vite/client" />

import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Writable } from "node:stream";

import { renderToPipeableStream } from "react-dom/server";
import { StaticRouter } from "react-router-dom/server";

import {
  ROUTE_META,
  canonicalUrl,
  speciesPageSlug,
  speciesRouteMeta,
  type RouteMeta,
} from "../src/siteMeta";
import { ARTICLE_SOURCES } from "../src/articles/components";
import { AppRoutes } from "../src/AppRoutes";
import { PageStateProvider } from "../src/state/PageStateContext";
import { FeatureModalProvider } from "../src/components/FeatureModal";
import {
  StaticModelProvider,
  type ModelContextValue,
} from "../src/state/ModelContext";
import { loadManifest } from "../src/state/manifest";
import {
  loadModel,
  loadSpeciesGraph,
  loadTeamCounts,
} from "../src/sampler/model";
import { buildCorpusScoreIndex } from "../src/render/corpusScore";

const distDir = resolve(dirname(fileURLToPath(import.meta.url)), "../dist");

const BASE_URL: string = import.meta.env.BASE_URL;
if (typeof BASE_URL !== "string" || !BASE_URL.startsWith("/")) {
  throw new Error(`unexpected import.meta.env.BASE_URL: ${BASE_URL}`);
}
const basename = BASE_URL.replace(/\/$/, "");

// ---------------------------------------------------------------------------
// Fetch shim: the app's loaders fetch("<BASE_URL>models/..."); in Node, serve
// those straight from the built dist/ so the prerender sees exactly the
// artifacts that ship. Anything else falls through to real fetch.
// ---------------------------------------------------------------------------

const realFetch = globalThis.fetch;
globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  if (url.startsWith(BASE_URL) && !/^https?:/.test(url)) {
    const file = join(distDir, url.slice(BASE_URL.length));
    if (!existsSync(file)) {
      return Promise.resolve(new Response(null, { status: 404, statusText: "Not Found" }));
    }
    return Promise.resolve(new Response(readFileSync(file)));
  }
  return realFetch(input, init);
}) as typeof fetch;

// ---------------------------------------------------------------------------
// Head stamping (title/description/canonical/OG) — unchanged mechanics.
// ---------------------------------------------------------------------------

function escapeAttr(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function escapeText(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function replaceTitle(html: string, title: string): string {
  const re = /<title>[\s\S]*?<\/title>/i;
  if (!re.test(html)) throw new Error("template is missing <title>");
  return html.replace(re, `<title>${escapeText(title)}</title>`);
}

/** Replace the content="" of a <meta name|property="key"> tag (identifier
 *  attribute precedes content, matching web/index.html). Throws on a miss so
 *  template drift fails the build loudly. */
function replaceMetaContent(
  html: string,
  attr: "name" | "property",
  key: string,
  content: string,
): string {
  const re = new RegExp(`(<meta\\s+${attr}="${key}"\\s+content=")[^"]*(")`, "i");
  if (!re.test(html)) throw new Error(`template is missing <meta ${attr}="${key}">`);
  return html.replace(re, `$1${escapeAttr(content)}$2`);
}

function replaceCanonical(html: string, url: string): string {
  const re = /(<link\s+rel="canonical"\s+href=")[^"]*(")/i;
  if (!re.test(html)) throw new Error('template is missing <link rel="canonical">');
  return html.replace(re, `$1${escapeAttr(url)}$2`);
}

/** Repoint the model-artifact preload hrefs at the manifest's default model,
 *  so the hand-written paths in index.html can never drift from the shipped
 *  default (they had, twice). */
function stampModelPreloads(html: string, defaultModel: string): string {
  const re = /(<link\s+rel="preload"\s+href="[^"]*\/models\/)[^/"]+(\/)/gi;
  if (!re.test(html)) throw new Error("template is missing model preload links");
  return html.replace(re, `$1${defaultModel}$2`);
}

// ---------------------------------------------------------------------------
// Per-route modulepreload injection: the article routes lazy-load their chunks,
// and a hard load would otherwise fetch them only after React mounts (a long
// gap on slow connections). Preloading them from <head> downloads them in
// parallel with the main bundle.
// ---------------------------------------------------------------------------

interface ManifestChunk {
  file: string;
  css?: string[];
  imports?: string[];
}

const viteManifest: Record<string, ManifestChunk> = JSON.parse(
  readFileSync(resolve(distDir, ".vite/manifest.json"), "utf8"),
);

/** Vite-manifest keys for the lazy modules a route needs; [] for routes that
 *  live entirely in the main bundle. */
function routeChunkKeys(path: string): string[] {
  if (path === "articles") return ["src/pages/ArticlesPage.tsx"];
  if (path.startsWith("articles/")) {
    const slug = path.slice("articles/".length);
    const sources = ARTICLE_SOURCES[slug];
    if (!sources) throw new Error(`no ARTICLE_SOURCES entry for article slug "${slug}"`);
    return ["src/pages/ArticlePage.tsx", ...sources];
  }
  return [];
}

/** Preload links for the chunks (and their static imports + CSS) behind the
 *  given manifest keys, skipping assets the template already references. */
function chunkPreloadLinks(template: string, keys: string[]): string {
  const js = new Set<string>();
  const css = new Set<string>();
  const seen = new Set<string>();
  const visit = (key: string) => {
    if (seen.has(key)) return;
    seen.add(key);
    const chunk = viteManifest[key];
    if (!chunk) throw new Error(`vite manifest is missing chunk "${key}"`);
    // A key can name a stylesheet module (a dynamic CSS import, e.g. KaTeX's).
    (chunk.file.endsWith(".css") ? css : js).add(chunk.file);
    for (const c of chunk.css ?? []) css.add(c);
    for (const i of chunk.imports ?? []) visit(i);
  };
  keys.forEach(visit);
  const links: string[] = [];
  for (const f of js) {
    if (template.includes(f)) continue;
    links.push(`    <link rel="modulepreload" crossorigin href="${BASE_URL}${f}">`);
  }
  for (const f of css) {
    if (template.includes(f)) continue;
    // crossorigin matches the <link rel="stylesheet" crossorigin> vite inserts
    // when the chunk loads, so the preload shares its cache entry.
    links.push(`    <link rel="preload" as="style" crossorigin href="${BASE_URL}${f}">`);
  }
  return links.length ? `${links.join("\n")}\n  ` : "";
}

function injectHeadLinks(html: string, links: string): string {
  if (!links) return html;
  if (!html.includes("</head>")) throw new Error("template is missing </head>");
  return html.replace("</head>", `${links}</head>`);
}

function applyMeta(template: string, meta: RouteMeta): string {
  // Redirecting routes (meta.canonical set) point their canonical + og:url at
  // the target so crawlers consolidate onto it; all others self-reference.
  const url = meta.canonical ?? canonicalUrl(meta.path);
  let html = replaceTitle(template, meta.title);
  html = replaceMetaContent(html, "name", "description", meta.description);
  html = replaceMetaContent(html, "property", "og:title", meta.title);
  html = replaceMetaContent(html, "property", "og:description", meta.description);
  html = replaceMetaContent(html, "property", "og:url", url);
  html = replaceMetaContent(html, "name", "twitter:title", meta.title);
  html = replaceMetaContent(html, "name", "twitter:description", meta.description);
  return replaceCanonical(html, url);
}

function injectBody(html: string, appHtml: string): string {
  const re = /<div id="root">\s*<\/div>/;
  if (!re.test(html)) throw new Error('template is missing an empty <div id="root">');
  return html.replace(re, `<div id="root">${appHtml}</div>`);
}

// ---------------------------------------------------------------------------
// Body rendering
// ---------------------------------------------------------------------------

/** renderToPipeableStream, buffered to a string. onAllReady (not onShellReady)
 *  so lazy routes/article bodies are fully resolved — no Suspense fallbacks in
 *  the emitted HTML. */
function renderRouteBody(path: string, modelValue: ModelContextValue): Promise<string> {
  const location = `${basename}/${path ? `${path}/` : ""}`;
  const element = (
    <StaticRouter location={location} basename={basename}>
      <StaticModelProvider value={modelValue}>
        <PageStateProvider>
          <FeatureModalProvider>
            <AppRoutes />
          </FeatureModalProvider>
        </PageStateProvider>
      </StaticModelProvider>
    </StaticRouter>
  );
  return new Promise<string>((resolvePromise, reject) => {
    const chunks: Buffer[] = [];
    const sink = new Writable({
      write(chunk, _enc, cb) {
        chunks.push(Buffer.from(chunk));
        cb();
      },
      final(cb) {
        cb();
        resolvePromise(Buffer.concat(chunks).toString("utf8"));
      },
    });
    const stream = renderToPipeableStream(element, {
      onAllReady() {
        stream.pipe(sink);
      },
      onError(err) {
        reject(err instanceof Error ? err : new Error(String(err)));
      },
    });
  });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const template = readFileSync(resolve(distDir, "index.html"), "utf8");

  const manifest = await loadManifest();
  const modelId = manifest.defaultModel;
  const [model, teamCounts, speciesGraph] = await Promise.all([
    loadModel(modelId),
    loadTeamCounts(modelId),
    loadSpeciesGraph(modelId),
  ]);
  const modelValue: ModelContextValue = {
    modelId,
    setModelId: () => {},
    model,
    teamCounts,
    speciesGraph,
    corpusScoreIndex: buildCorpusScoreIndex(model, teamCounts),
    manifest,
    status: "ready",
    error: null,
  };
  console.log(`prerender model: ${modelId} (V=${model.V}, ${model.nCorpusTeams} teams)`);

  const stamped = stampModelPreloads(template, modelId);

  // One route per species in the default model (/pokemon/<slug>). Slugs must
  // be unique or two species would silently share (and overwrite) one page.
  const slugToSpecies = new Map<string, string>();
  for (const species of model.sites) {
    const slug = speciesPageSlug(species);
    const clash = slugToSpecies.get(slug);
    if (clash) throw new Error(`species page slug collision: "${species}" vs "${clash}" -> ${slug}`);
    slugToSpecies.set(slug, species);
  }
  const speciesRoutes: RouteMeta[] = model.sites.map((species) =>
    speciesRouteMeta(species, model.regulation),
  );
  const allRoutes: RouteMeta[] = [...ROUTE_META, ...speciesRoutes];

  for (const meta of allRoutes) {
    let html = applyMeta(stamped, meta);
    // Redirect routes (meta.canonical set) keep an empty body: rendering them
    // would just duplicate the target's content under the legacy URL.
    if (!meta.canonical) {
      html = injectHeadLinks(html, chunkPreloadLinks(stamped, routeChunkKeys(meta.path)));
      html = injectBody(html, await renderRouteBody(meta.path, modelValue));
    }
    const dir = meta.path ? resolve(distDir, meta.path) : distDir;
    mkdirSync(dir, { recursive: true });
    writeFileSync(resolve(dir, "index.html"), html);
    if (!meta.path.startsWith("pokemon/")) {
      console.log(`prerendered: ${meta.path ? `${meta.path}/index.html` : "index.html (home)"}`);
    }
  }
  console.log(`prerendered: ${speciesRoutes.length} species pages under pokemon/`);

  const sitemap =
    '<?xml version="1.0" encoding="UTF-8"?>\n' +
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' +
    allRoutes.filter((m) => !m.canonical)
      .map((m) => `  <url>\n    <loc>${canonicalUrl(m.path)}</loc>\n  </url>`)
      .join("\n") +
    "\n</urlset>\n";
  writeFileSync(resolve(distDir, "sitemap.xml"), sitemap);
  console.log("wrote: sitemap.xml");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
