// /pokemon — index of every species in the active model, sorted by usage.
// Each entry links to its species page (/pokemon/<slug>). This page is the
// crawlable hub for the per-Pokémon pages: it is prerendered with real links,
// so search engines can reach every species page from here.

import { useMemo } from "react";
import { Link } from "react-router-dom";
import { useModel } from "../state/ModelContext";
import { PageTitle } from "../render/atoms";
import { SpriteBox } from "../render/Sprite";
import { formatPct } from "../render/format";
import { speciesPageSlug } from "../siteMeta";

export function PokedexPage() {
  const { model, status } = useModel();

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
      {status === "loading" || rows === null ? (
        <p style={{ color: "var(--lab-ink-muted)" }}>Loading model…</p>
      ) : (
        <ul className="lab-pokedex-grid">
          {rows.map((r) => (
            <li key={r.slug}>
              <Link to={`/pokemon/${r.slug}/`} className="lab-pokedex-card">
                <SpriteBox name={r.species} size={40} />
                <span className="lab-pokedex-name">{r.species}</span>
                <span className="lab-pokedex-usage">{formatPct(r.usage)}</span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
export default PokedexPage;
