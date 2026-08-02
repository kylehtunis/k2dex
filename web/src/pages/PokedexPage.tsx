// /pokemon — index of every species in the active model, sorted by usage.
// Each entry links to its species page (/pokemon/<slug>). This page is the
// crawlable hub for the per-Pokémon pages: it is prerendered with real links,
// so search engines can reach every species page from here. The search box is
// a client-side substring filter over the same list — it narrows what is
// rendered, so the unfiltered page is what the prerender writes.

import { useDeferredValue, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useModel } from "../state/ModelContext";
import { PageTitle } from "../render/atoms";
import { SpriteBox } from "../render/Sprite";
import { formatPct } from "../render/format";
import { speciesPageSlug } from "../siteMeta";

export function PokedexPage() {
  const { model, status } = useModel();
  const [query, setQuery] = useState("");
  // The grid is a few hundred sprite cards; deferring the filter keeps typing
  // responsive by letting React render the input ahead of the list.
  const deferredQuery = useDeferredValue(query);

  const rows = useMemo(() => {
    if (!model) return null;
    const out = model.sites.map((species, site) => {
      let usage = 0;
      for (const f of model.siteFeatures[site]) usage += model.m[f];
      return { species, slug: speciesPageSlug(species), usage };
    });
    out.sort((a, b) => b.usage - a.usage);
    return out;
  }, [model]);

  // Plain case-insensitive substring match on the species name.
  const filtered = useMemo(() => {
    if (rows === null) return null;
    const q = deferredQuery.trim().toLowerCase();
    if (q === "") return rows;
    return rows.filter((r) => r.species.toLowerCase().includes(q));
  }, [rows, deferredQuery]);

  const corpusCaption = model
    ? `Reg ${model.regulation} · ${model.nCorpusTeams.toLocaleString()} teams`
    : undefined;

  return (
    <>
      <PageTitle
        eyebrow="VGC Data"
        h1="Pokémon Index"
        subtitle="Every Pokémon in the model, ranked by tournament usage. Open one for its teammates, item builds, and synergies."
        rightCaption={corpusCaption}
      />
      {status === "loading" || rows === null || filtered === null ? (
        <p style={{ color: "var(--lab-ink-muted)" }}>Loading model…</p>
      ) : (
        <>
          <div className="lab-pokedex-search">
            <input
              type="search"
              className="lab-search-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search Pokémon…"
              aria-label="Search Pokémon"
              autoComplete="off"
            />
            <span className="lab-pokedex-count">
              {filtered.length} of {rows.length}
            </span>
          </div>
          {filtered.length === 0 ? (
            <p style={{ color: "var(--lab-ink-muted)" }}>
              No Pokémon match "{query.trim()}".
            </p>
          ) : (
            <ul className="lab-pokedex-grid">
              {filtered.map((r) => (
                <li key={r.slug}>
                  <Link to={`/pokemon/${r.slug}/`} className="lab-pokedex-card">
                    <SpriteBox name={r.species} size={40} />
                    <span className="lab-pokedex-name">{r.species}</span>
                    <span className="lab-pokedex-usage">
                      {formatPct(r.usage)}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </>
  );
}
export default PokedexPage;
