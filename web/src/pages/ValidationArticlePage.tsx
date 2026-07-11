// Body of the "Why Not Just Count?" article (rendered by ArticlePage at
// /articles/model-vs-counting). Two sections, each making one argument about
// what the k2dex model captures that raw co-occurrence counting cannot:
//   §1 SpeciesComparison — the model separates real synergy from shared popularity
//   §2 ItemInteractions  — the model understands items change a Pokémon's identity
// Reuses the .lab-science article shell (see SciencePage.tsx).

import { SpeciesComparison } from "../validation/sections/SpeciesComparison";
import { ItemInteractions } from "../validation/sections/ItemInteractions";

export function ValidationArticlePage() {
  return (
    <div className="lab-science">
      <header className="lab-science-header">
        <p className="lab-science-lede">
          Every other teambuilder ranks partners by how often they appear
          together
          <span className="lab-science-lede-punch">(so why doesn't k2dex?)</span>
        </p>
        <h1>Why not just count?</h1>
      </header>
      <SpeciesComparison />
      <ItemInteractions />
    </div>
  );
}

export default ValidationArticlePage;
