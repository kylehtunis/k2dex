// §2: the model understands that an item can change what a Pokémon *is*.
//
// Co-occurrence works at the species level — "Charizard and Garchomp appear
// together a lot." The model works at the (species, item) level, so it can see
// that the synergy depends entirely on which item is held. Curated examples,
// each read live from the model's couplings.

import { useModel } from "../../state/ModelContext";
import {
  ItemModulationCard,
  type ModulationExample,
} from "../widgets/ItemModulationCard";

const EXAMPLES: ModulationExample[] = [
  {
    id: "charizard-garchomp",
    modulatedSpecies: "Charizard",
    partnerSpecies: "Garchomp",
    items: [
      { vocab: "Charizard @ Charizardite Y", label: "Charizardite Y" },
      { vocab: "Charizard @ Charizardite X", label: "Charizardite X" },
    ],
    headline: "One stone makes the pairing; the other breaks it",
    body: (
      <>
        Mega Charizard <strong>Y</strong> stays Fire/Flying, so it's immune to
        Garchomp's Earthquake and happily shares a team, spamming spread moves
        under its own sun. Mega Charizard <strong>X</strong> becomes Fire/Dragon
        and <em>loses</em> that Ground immunity, so the same Earthquake now hits
        it for super-effective damage. Same two Pokémon, opposite verdict, and
        it hinges entirely on the held item. Co-occurrence sees only "Charizard
        + Garchomp" and can't tell the two apart.
      </>
    ),
  },
  {
    id: "venusaur-charizard",
    modulatedSpecies: "Venusaur",
    partnerSpecies: "Charizard",
    items: [
      { vocab: "Venusaur @ Focus Sash", label: "Focus Sash" },
      { vocab: "Venusaur @ Sitrus Berry", label: "Sitrus Berry" },
      { vocab: "Venusaur @ Venusaurite", label: "Venusaurite (Mega)" },
    ],
    headline: "The item that deletes the synergy",
    body: (
      <>
        A held-item Venusaur keeps its Chlorophyll ability, doubling its Speed in
        the sun that Charizard-Y provides: the classic sun core. Give it its own
        mega stone and Chlorophyll is replaced by Thick Fat. The Speed boost is
        gone, the reason to pair them evaporates, and the model turns the synergy
        negative. The item didn't tweak Venusaur, it changed which Pokémon it is.
      </>
    ),
  },
];

export function ItemInteractions() {
  const { model, status } = useModel();
  const ready = model !== null && status === "ready";

  return (
    <section id="items" className="lab-science-section">
      <h3>Items aren't an afterthought</h3>
      <p>
        Most teambuilders treat item choice as the last step: pick your six
        Pokémon, then slap the most popular item on each. That order assumes the
        item doesn't affect who your Pokémon works with, which is often exactly
        wrong. An item can flip a Pokémon's type, swap its ability, or redefine
        its role, and with it, everything about who it wants as a teammate.
      </p>
      <p>
        The model treats a Pokémon and its item as a single unit, so it learns
        these interactions directly. Co-occurrence can't: it counts species, and
        a species is the same species whatever it's holding. Here are a few cases
        where the item makes all the difference.
      </p>
      {!ready ? (
        <p style={{ color: "var(--lab-ink-muted)" }}>Loading the live model…</p>
      ) : (
        <div className="lab-modulation-grid">
          {EXAMPLES.map((ex) => (
            <ItemModulationCard key={ex.id} model={model!} example={ex} />
          ))}
        </div>
      )}
      <p>
        This is the piece no co-occurrence teambuilder can replicate, because the
        information simply isn't in the counts. To know that Charizardite X
        quietly breaks the Garchomp pairing, you have to model the item as part
        of the Pokémon, which is exactly what k2dex does, and what the completer
        uses every time it fills a team.
      </p>
    </section>
  );
}
