// Post-build step: turn the single-page Vite output into per-route static HTML
// so GitHub Pages serves a real 200 (not the 404.html SPA-redirect) for every
// route, each with a unique <title>/description and a self-referencing
// canonical. Also (re)writes dist/sitemap.xml from the same route list.
//
// Without this, a client-routed SPA on static hosting is effectively
// un-indexable below the root: direct hits to /completer, /science, etc. 404
// for crawlers, and the one shared index.html names the homepage as everyone's
// canonical, collapsing the site to a single indexed URL.
//
// Run automatically as the last step of `npm run build` (see package.json).
// Reads the freshly built dist/index.html as the template; asset URLs in it are
// absolute (base-path prefixed) so copies in subfolders resolve correctly.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { ROUTE_META, canonicalUrl, type RouteMeta } from "../src/siteMeta";

const distDir = resolve(dirname(fileURLToPath(import.meta.url)), "../dist");

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

function applyMeta(template: string, meta: RouteMeta): string {
  const url = canonicalUrl(meta.path);
  let html = replaceTitle(template, meta.title);
  html = replaceMetaContent(html, "name", "description", meta.description);
  html = replaceMetaContent(html, "property", "og:title", meta.title);
  html = replaceMetaContent(html, "property", "og:description", meta.description);
  html = replaceMetaContent(html, "property", "og:url", url);
  html = replaceMetaContent(html, "name", "twitter:title", meta.title);
  html = replaceMetaContent(html, "name", "twitter:description", meta.description);
  return replaceCanonical(html, url);
}

const template = readFileSync(resolve(distDir, "index.html"), "utf8");

for (const meta of ROUTE_META) {
  const dir = meta.path ? resolve(distDir, meta.path) : distDir;
  mkdirSync(dir, { recursive: true });
  writeFileSync(resolve(dir, "index.html"), applyMeta(template, meta));
  console.log(`prerendered: ${meta.path ? `${meta.path}/index.html` : "index.html (home)"}`);
}

const sitemap =
  '<?xml version="1.0" encoding="UTF-8"?>\n' +
  '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' +
  ROUTE_META.map((m) => `  <url>\n    <loc>${canonicalUrl(m.path)}</loc>\n  </url>`).join("\n") +
  "\n</urlset>\n";
writeFileSync(resolve(distDir, "sitemap.xml"), sitemap);
console.log("wrote: sitemap.xml");
