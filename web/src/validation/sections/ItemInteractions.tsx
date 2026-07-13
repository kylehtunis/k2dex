// §2: the model understands that an item can change what a Pokémon *is*.
//
// Co-occurrence works at the species level — "Charizard and Garchomp appear
// together a lot." The model works at the (species, item) level, so it can see
// that the synergy depends entirely on which item is held. Curated examples,
// each read live from the model's couplings.

import { useModel } from "../../state/ModelContext";
import {
  ItemModulationCard,
  isExampleResolvable,
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
    headline: "Charizardite X ruins the synergy",
    body: (
      <>
        Mega Charizard <strong>Y</strong> stays Fire/Flying, so it's immune to
        Garchomp's Earthquake and happily shares a team.
        Mega Charizard <strong>X</strong> becomes Fire/Dragon
        and <em>loses</em> that Ground immunity, so the same Earthquake now hits
        it for super-effective damage. Same two Pokémon, opposite verdict, and
        it hinges entirely on the held item. Co-occurrence sees only "Charizard
        + Garchomp" and can't tell the two apart.
      </>
    ),
  },
  {
    id: "kingambit-aerodactyl",
    modulatedSpecies: "Kingambit",
    partnerSpecies: "Aerodactyl",
    items: [
      { vocab: "Kingambit @ Focus Sash", label: "Focus Sash" },
      { vocab: "Kingambit @ Black Glasses", label: "Black Glasses" },
      { vocab: "Kingambit @ Life Orb", label: "Life Orb" },
      { vocab: "Kingambit @ Occa Berry", label: "Occa Berry" },
    ],
    headline: "An item whose job is already taken",
    body: (
      <>
        Kingambit's Occa Berry has one purpose:
        survive a Fire attack, almost always Charizard's. Aerodactyl is one of
        the format's premier Charizard counters. On a team that already runs
        Aerodactyl, the berry is dead weight. The k2dex model
        sees it: usually Kingambit pairs wonderfully with
        Aerodactyl, but the Occa Berry set turns <em>negative</em>, because the
        two are solving the same problem. Co-occurrence only counts species, so
        it can't tell that one Pokémon makes the other's item redundant.
      </>
    ),
  },
  {
    id: "basculegion-whimsicott",
    modulatedSpecies: "Basculegion",
    partnerSpecies: "Whimsicott",
    items: [
      { vocab: "Basculegion @ Life Orb", label: "Life Orb" },
      { vocab: "Basculegion @ Mystic Water", label: "Mystic Water" },
      { vocab: "Basculegion @ Choice Scarf", label: "Choice Scarf" },
    ],
    headline: "A Choice Scarf softens the need for Tailwind",
    body: (
      <>
        The item doesn't have to flip a pairing to matter; it can just change the
        strength. Basculegion is a mid-speed powerhouse, so a Life Orb or Mystic Water set leans on
        Whimsicott's Tailwind to outrun the field and clean up. A Choice Scarf
        set already outspeeds most of what it needs to, so it wants Tailwind far
        less. The pairing stays positive either way since a little Tailwind is rarely a bad
        thing and because the two have synergy beyond Tailwind,
        but the model knows the Scarf build needs Whimsicott far less.
      </>
    ),
  },
];

export function ItemInteractions() {
  const { model, status } = useModel();
  const ready = model !== null && status === "ready";

  return (
    <section id="items" className="lab-science-section">
      <h3>Extended attributes aren't an afterthought</h3>
      <p>
        The problems with co-occurrence surface as soon as you begin considering more of a Pokémon's attributes:
        item, ability, nature, and moves.
        Usage stats don't consider how these choices interact across Pokémon, while the Boltzmann learning
        employed by k2dex is designed to understand exactly these effects.
        Looking at items specifically: item can often define a Pokémon's role,
        and with it, everything about who it wants as a teammate.
      </p>
      <p>
        For example, Charizard and Garchomp pair together quite well in Regulation M-B:
        Garchomp can spam its Earthquakes without fear, and Charizard can comfortably take Fairy or Ice moves
        intended to delete Garchomp, while Garchomp is a safe switch-in for the Rock, Water, and Electric moves
        that threaten Charizard. However, if Charizard is holding Charizardite X that relationship collapses:
        suddenly, Garchomp can't click Earthquake without threatening its partner with super-effective damage,
        and Charizard no longer resists Fairy moves and indeed now shares Garchomp's Dragon weakness.
        They also compete for the same slot: a strong, fast Dragon-type physical attacker. The k2dex model
        learns this relationship effortlessly, and the teams it builds reflect that understanding.
      </p>
      {!ready ? (
        <p style={{ color: "var(--lab-ink-muted)" }}>Loading the live model…</p>
      ) : (
        <div className="lab-modulation-grid">
          {EXAMPLES.filter((ex) => isExampleResolvable(model!, ex)).map((ex) => (
            <ItemModulationCard key={ex.id} model={model!} example={ex} />
          ))}
        </div>
      )}
      <p>
        This is the piece no co-occurrence teambuilder can replicate, because the
        information simply isn't in the counts. To know that Charizardite X
        quietly breaks the Garchomp pairing, you have to model the item as part
        of the Pokémon, which is exactly what k2dex does.
        Some teambuilders treat Mega Pokémon separately
        from their base formes which does mitigate this problem, but <em>only</em> for Mega stones.
        Understanding how other items affect a Pokémon's role, such as the difference between 
        Life Orb and Choice Scarf Basculegions, requires advanced modelling.
        That fact only becomes more true as you begin to consider abilities, natures, and moves,
        and is the driving philosophy behind k2dex.
      </p>
    </section>
  );
}
