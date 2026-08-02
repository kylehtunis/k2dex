// /pokemon/<slug> — a species' detail as a standalone page: the same
// SpeciesDetail body the feature modal shows, plus crawlable partner links
// (partnerHref) so the species pages cross-link into a walkable graph. Those
// links navigate only on modifier-click; a plain click matches the modal's
// behavior (row expands, and an expanded modulation row opens the feature
// modal via openSite rather than navigating the page). Prerendered per
// species by scripts/prerender-routes.tsx from the default model. It uses
// SpeciesDetail's "page" variant, which spans the full container width and
// renders corpus appearances as completer-style team cards. At runtime it
// reflects the active model, and a species missing from it (e.g. after
// switching regulation) gets a friendly fallback.

import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { useModel } from "../state/ModelContext";
import { useFeatureModal } from "../components/FeatureModalContext";
import { SpeciesDetail } from "../components/SpeciesDetail";
import { speciesPageSlug } from "../siteMeta";

export function PokemonPage() {
  const { slug = "" } = useParams();
  const { model, status } = useModel();
  const featureModal = useFeatureModal();

  const siteBySlug = useMemo(
    () =>
      model
        ? new Map(model.sites.map((s, i) => [speciesPageSlug(s), i] as const))
        : null,
    [model],
  );

  if (status === "loading" || !model || !siteBySlug) {
    return <p style={{ color: "var(--lab-ink-muted)" }}>Loading model…</p>;
  }

  const site = siteBySlug.get(slug);

  return (
    <div className="lab-pokemon-page">
      <nav className="lab-article-nav">
        <Link to="/pokemon/" className="lab-article-back">
          ← All Pokémon
        </Link>
      </nav>
      {site === undefined ? (
        <p style={{ color: "var(--lab-ink-muted)" }}>
          This Pokémon isn't in the currently selected model
          (Reg {model.regulation}). Pick it from the index instead.
        </p>
      ) : (
        <div className="lab-pokemon-detail">
          <SpeciesDetail
            model={model}
            site={site}
            headingLevel="h1"
            headExtra={
              <div className="lab-pokemon-caption">
                Reg {model.regulation} · {model.nCorpusTeams.toLocaleString()} teams
              </div>
            }
            variant="page"
            onDrillSite={(s) => featureModal?.openSite(s)}
            partnerHref={(sp) => `/pokemon/${speciesPageSlug(sp)}/`}
          />
        </div>
      )}
    </div>
  );
}
export default PokemonPage;
