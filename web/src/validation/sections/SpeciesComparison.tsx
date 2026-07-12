// §1: how far raw counting gets you at the species level.
//
// First shows the familiar "Teammates %" table (co-occurrence), then a mini
// team completer that fills a full team both ways — counting vs the real
// sampler. The honest finding is that both produce sensible teams: counting is
// a strong baseline for *which Pokémon*, which sets up §2, where it goes blind
// (attributes). This section is the on-ramp and the pivot, not the payoff.

import { useMemo } from "react";
import { useModel } from "../../state/ModelContext";
import { buildCooccurrence } from "../../sampler/cooccurrence";
import { aggregateToSpecies } from "../cooccurrenceSpecies";
import { TeammateTable } from "../widgets/TeammateTable";
import { MiniCompleter } from "../widgets/MiniCompleter";

export function SpeciesComparison() {
  const { model, teamCounts, status } = useModel();
  const ready = model !== null && teamCounts !== null && status === "ready";

  // Build the co-occurrence matrix once per model, folded to species level.
  const sc = useMemo(() => {
    if (!ready) return null;
    const cooc = buildCooccurrence(teamCounts!, model!.V);
    return aggregateToSpecies(model!, cooc);
  }, [ready, model, teamCounts]);

  return (
    <section id="species" className="lab-science-section">
      <h3>How far can counting get you?</h3>
      <p>
        If you've ever used Pikalytics, Smogon's usage stats, or almost any
        teambuilding site, you've seen a table like the one below. Pick a
        Pokémon, and it shows you the teammates it appears with most often. It's
        computed by the simplest possible method: go through every team in the
        data, and for every pair of Pokémon on it, add one to their shared
        count. That's <em>co-occurrence</em>, and it's the engine behind nearly
        every teambuilder except this one.
      </p>
      {!ready || sc === null ? (
        <p style={{ color: "var(--lab-ink-muted)" }}>Loading the live model…</p>
      ) : (
        <>
          <TeammateTable model={model!} sc={sc} />
          <p>
            It's a powerful tool, but in principle it has a blind spot:
            co-occurrence can't tell the difference between two Pokémon that
            genuinely work together and two Pokémon that are simply both popular.
            If Garchomp is on half of all teams and Sinistcha is on another 40%,
            they'll land on the same team constantly, purely by chance, even if
            neither cares about the other, and the table reports that as "synergy"
            all the same.
          </p>
          <p>
            The k2dex model is built to see past this. Instead of counting pairs, it
            fits a coupling between every pair of Pokémon that has to explain the
            team data as a whole, holding everything else constant. That's the
            difference between "how often do these two appear together?" and "do
            these two appear together <em>more</em> than their individual
            popularity would predict?" To see what that's worth, use the comparison
            below and let each method finish the team: counting fills the
            open slots with its top teammates, while k2dex runs the same sampler
            the completer uses and returns its single most-likely team.
          </p>
          <MiniCompleter model={model!} sc={sc} teamCounts={teamCounts} />
          <p>
            Try a few, and you'd be justified in raising an eyebrow: both
            teams are <em>usually</em> sensible. At the level of <em>which Pokémon</em>,
            popularity and real synergy mostly point the same way, so counting is
            a strong baseline. k2dex's team is often a little tighter, earning a
            higher coherence (the total strength of the couplings holding it
            together) and landing closer to a team someone actually brought to a
            tournament, but that difference alone isn't significant enough to justify the 
            advanced machinery of k2dex.
          </p>
          <p>
            So, if counting works this well, why bother going further? 
            Because co-occurrence isn't how real teams are built. The strongest teams aren't
            constructed by picking six synergistic Pokémon and slapping the most popular set on each.
            A good teambuilder, whether human or machine, needs to ensure that every Pokémon, item,
            ability, nature, move, and EV point is chosen holistically, each decision contributing
            to an overall goal.
          </p>
        </>
      )}
    </section>
  );
}
