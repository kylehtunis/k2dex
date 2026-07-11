// §1: the model separates real synergy from shared popularity.
//
// First shows the familiar "Teammates %" table (co-occurrence), then a mini
// team completer that fills a full team both ways — counting vs the real
// sampler — so the reader can compare finished teams, not abstract rankings.

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
      <h3>The teammates table, and why it isn't enough</h3>
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
            It's a useful table, but it has a blind spot: co-occurrence can't
            tell the difference between two Pokémon that genuinely work together
            and two Pokémon that are simply both popular. If Incineroar is on
            half of all teams and Flutter Mane is on another 40%, they'll land
            on the same team constantly, purely by chance, even if neither cares
            about the other. The table reports that as "synergy" all the same.
            Follow it far enough and it just walks you toward the most popular
            Pokémon in the format, whether or not they fit what you already have.
          </p>
          <p>
            The model is built to see past this. Instead of counting pairs, it
            fits a coupling between every pair of Pokémon that has to explain the
            team data as a whole, holding everything else constant. That's the
            difference between "how often do these two appear together?" and "do
            these two appear together <em>more</em> than their individual
            popularity would predict?" To see what that's worth, pick a Pokémon
            or two below and let each method finish the team: counting fills the
            open slots with its top teammates, while k2dex runs the same sampler
            the completer uses and returns its single most-likely team.
          </p>
          <MiniCompleter model={model!} sc={sc} teamCounts={teamCounts} />
          <p>
            The two teams are scored under the same fitted model. Counting's team
            is usually a pile of individually popular Pokémon that don't
            particularly want to be together; k2dex's team almost always earns a
            higher coherence, the total strength of the couplings holding it
            together, and lands closer to a team someone actually brought to a
            tournament. Better still, look at the Pokémon themselves: the
            sampler's team tends to read like a real, purposeful squad rather
            than a greatest-hits list.
          </p>
        </>
      )}
    </section>
  );
}
