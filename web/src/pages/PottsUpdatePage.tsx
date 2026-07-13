// Body of "The v2 Update" article (rendered by ArticlePage at
// /articles/the-v2-update). Release notes for the Boltzmann/Potts rebuild,
// written for existing users: leads with the visible changes (one model per
// regulation, the roster editor and site pins, Anchor Strength, the retired
// Bias Adjustment slider), then explains the modelling story behind them.
// Prose only; reuses the .lab-science article shell (see SciencePage.tsx).

export function PottsUpdatePage() {
  return (
    <div className="lab-science">
      <header className="lab-science-header">
        <p className="lab-science-lede">
          A rebuilt statistical engine, a rebuilt completer, and one slider
          fewer
          <span className="lab-science-lede-punch">
            (everything that changed, and why)
          </span>
        </p>
        <h1>The v2 update</h1>
      </header>

      <section id="overview" className="lab-science-section">
        <p>
          You may have noticed that k2dex looks and feels much different than it did previously.
          This update touches nearly every part of
          it: the model picker, the completer, the analysis page, and the meta
          page. The short version is that the model underneath is now fit with
          a fundamentally better method, and almost every visible change
          follows from that. Here's the tour, starting with what you'll notice
          first.
        </p>
      </section>

      <section id="one-model" className="lab-science-section">
        <h3>One model per regulation</h3>
        <p>
          The old picker offered four models: Species, Species @ Item
          (recommended), a weighted experimental variant, and an experimental
          Regulation M-B model. That menu is gone. There is now exactly one
          model per regulation, and it folds in everything the variants used
          to trade off against each other for a simplified and improved experience
          without any loss in model quality.
        </p>
        <p>
          If you preferred the old
          species-only view, it didn't disappear: it became a toggle. In the
          completer's advanced options you can exclude the item attribute
          entirely, and results aggregate by Pokémon set. Crucially this is
          not a separate, coarser model: the item dimension is collapsed
          from the full model exactly, so the species-level answers are
          consistent with the item-level ones <em>and</em> the old species-only model.
        </p>
      </section>

      <section id="roster-editor" className="lab-science-section">
        <h3>A completer built around your picks</h3>
        <p>
          The completer's input used to be a single search box of full builds
          like "Charizard @ Life Orb". It's now a proper six-slot roster
          editor. Each slot has a Pokémon picker and an optional item picker,
          and the item being optional is a new capability: leave it blank
          and you've locked in the Pokémon while letting the model choose the
          item as part of the completion. Set it and you've pinned the exact
          build, as before. The same editor now drives the analysis page, so
          moving a team between the two surfaces feels the same.
        </p>
        <p>
          Constraints grew a second half. Alongside the existing exclude list
          ("never suggest these") there's now an include list: give it a pool
          of Pokémon and completions are restricted to that pool plus your
          pins. This new constraint is perfect for newer players who may be limited
          by the Pokémon they have recruited thus far, creating teams from the Pokémon they
          have available instead of the full regulation roster.
        </p>
        <p>
          Results changed shape too. Instead of a ranked table of the
          completed members, each suggestion is a full six-Pokémon team card,
          with your pinned slots marked and the team's observables and actions
          underneath. The default stats strip now reports Variety (how
          different the shown teams are from each other) and Novelty (how far
          they sit from teams already in the tournament data), while the
          sampler diagnostics that used to occupy that space are still
          available behind a toggle. And anywhere a Score or Coherence value
          appears, hovering it shows the corpus percentile: how that number
          ranks against every real tournament team the model was trained on.
        </p>
      </section>

      <section id="anchor-strength" className="lab-science-section">
        <h3>Anchor Strength: no more awkward completions</h3>
        <p>
          The old completer had a problem that power users will certainly recognize: pin a
          niche Pokémon and ask for the rest of the team, and you'd often get
          a perfectly coherent five-Pokémon core with your pin bolted on as a
          passenger. This was the mathematically likeliest team containing your pick,
          but not a team built <em>around</em> it.
        </p>
        <p>
          The new Anchor Strength slider fixes this. It tilts the sampler
          toward teams whose members actually interact with your pins, by
          amplifying the couplings between pinned and free slots during the
          search. At the default of 1.0 nothing changes; slide it up and the
          suggestions become increasingly pin-centric. The math used to accomplish this
          retains the same level of scientific rigor that k2dex has always prided itself on.
        </p>
      </section>

      <section id="sliders" className="lab-science-section">
        <h3>Where did the Bias Adjustment slider go?</h3>
        <p>
          Gone, on purpose. The slider existed because the old model needed
          correcting: run as fitted, it leaned too hard on individual
          popularity, so the app shipped with the bias term dialed down (0.5
          by default) and let you tune it. The split between "raw" and
          "adjusted" scores existed for the same reason.
        </p>
        <p>
          The new model doesn't need that correction. Its fitting procedure
          guarantees the model's statistics match the tournament data (more on
          that below), so the webapp now runs it exactly as fitted: no bias
          correction, temperature 1. This was validated directly by sweeping the
          old knobs against the new correctionless model, and no hand-tuned setting
          beat it. So the slider is gone, there is a single Score, and the
          temperature control (now defaulting to 1.0) has moved into the
          advanced sampler options where it belongs: a diagnostic tool, not a
          setting you should ever <em>need</em> to touch.
        </p>
      </section>

      <section id="engine" className="lab-science-section">
        <h3>Under the hood: Boltzmann learning and the Potts view</h3>
        <p>
          The old models were fit by pseudo-likelihood: fast, standard, and
          subtly wrong for this use. Pseudo-likelihood matches the model's
          conditional distributions ("given these five, how likely is this
          sixth?") but not its overall statistics, and in practice it
          over-concentrated: the worst-case gap between how often the model
          used a build and how often tournaments actually did was about 0.34,
          which is enormous for a probability. This is what the old Bias Adjustment
          slider was built to address, and why it's no longer necessary.
          The new models are fit by
          Boltzmann learning, which iteratively samples the model and adjusts
          it until its usage rates and pairing rates match the data directly.
          That worst-case gap drops to about 0.05. When the completer says a
          team is likely, that now means likely under a model whose statistics
          agree with real tournament results.
        </p>
        <p>
          The second change is structural. The model now treats each team slot
          as a Pokémon whose held item is one of several states of that
          Pokémon (in physics terms, a Potts model: the site is the species,
          the state is the item). The sampler moves accordingly, swapping
          which Pokémon occupies a slot and, separately, rerolling which item
          it holds. That factoring is exactly what makes the new features
          possible: locking a Pokémon while the model picks its item is just
          freezing the site and sampling the state, and the item-free toggle
          is just summing the states.
        </p>
        <p>
          The Potts framing also makes the model easily extensible. The old
          representation treated every distinct build as its own vocabulary
          entry, which meant adding another attribute would have multiplied
          the vocabulary: every Pokémon crossed with every item crossed with
          every ability, and so on, with most combinations too rare to learn
          anything about. The new model instead stores each attribute as its
          own dimension of a Pokémon's state, so adding one grows the model
          additively rather than multiplicatively, and Boltzmann learning
          keeps the fit correct as the state space grows. That's not a
          hypothetical: items are the first attribute on this footing, and
          the plan is to bring abilities, natures, and eventually moves into
          the model the same way.
        </p>
        <p>
          The meta page and the Pokémon detail modal were rebuilt on the same
          view. Synergy is now reported between Pokémon, with each pairing
          expandable to show how specific items strengthen or undermine it,
          instead of one flat list of build-pair couplings. The synergy
          rankings also now weight each build by how often it's actually
          used, which removes a bias that previously favored Pokémon with
          only one common item.
        </p>
      </section>

      <section id="conclusion" className="lab-science-section">
        <h3>Summing up</h3>
        <p>
          This update addressed many of the issues with the old model and sets up
          k2dex to continue to improve. That said, this is an enormous update and likely
          to have some rough edges. If something on the
          new site looks wrong, behaves strangely, or is simply worse than
          what it replaced, or if you have an idea that would make k2dex more
          useful to you, come say so in the{" "}
          <a
            href="https://discord.gg/8xNjyn9yVP"
            target="_blank"
            rel="noopener noreferrer"
          >
            k2dex Discord server
          </a>
          . Bug reports, feature requests, and questions about how any of
          this works are all welcome. This is a true passion project for me, so I
          genuinely enjoy answering questions and appreciate any feedback you might have.
        </p>
        <p>
          Happy teambuilding!
        </p>
      </section>
    </div>
  );
}

export default PottsUpdatePage;
