// Closing references / sources footer for the /science page. Not a research
// paper — a curated list of the methods this project builds on plus the data
// and sprite acknowledgments. Inline (author, year) citations elsewhere on the
// page link to the matching `id` anchors here.

interface Reference {
  id: string;
  authors: string;
  year: string;
  title: string;
  venue: string;
  href?: string;
}

// Ordered roughly by where each idea appears on the page.
const REFERENCES: Reference[] = [
  {
    id: "ref-ising",
    authors: "E. Ising",
    year: "1925",
    title: "Beitrag zur Theorie des Ferromagnetismus",
    venue: "Zeitschrift für Physik 31, 253–258",
    href: "https://doi.org/10.1007/BF02980577",
  },
  {
    id: "ref-onsager",
    authors: "L. Onsager",
    year: "1944",
    title:
      "Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition",
    venue: "Physical Review 65, 117–149",
    href: "https://doi.org/10.1103/PhysRev.65.117",
  },
  {
    id: "ref-metropolis",
    authors: "N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, E. Teller",
    year: "1953",
    title: "Equation of State Calculations by Fast Computing Machines",
    venue: "The Journal of Chemical Physics 21, 1087–1092",
    href: "https://doi.org/10.1063/1.1699114",
  },
  {
    id: "ref-hukushima",
    authors: "K. Hukushima, K. Nemoto",
    year: "1996",
    title: "Exchange Monte Carlo Method and Application to Spin Glass Simulations",
    venue: "Journal of the Physical Society of Japan 65, 1604–1608",
    href: "https://doi.org/10.1143/JPSJ.65.1604",
  },
  {
    id: "ref-besag",
    authors: "J. Besag",
    year: "1975",
    title: "Statistical Analysis of Non-Lattice Data",
    venue: "The Statistician 24, 179–195",
    href: "https://doi.org/10.2307/2987782",
  },
  {
    id: "ref-schneidman",
    authors: "E. Schneidman, M. J. Berry II, R. Segev, W. Bialek",
    year: "2006",
    title:
      "Weak pairwise correlations imply strongly correlated network states in a neural population",
    venue: "Nature 440, 1007–1012",
    href: "https://doi.org/10.1038/nature04701",
  },
  {
    id: "ref-lee",
    authors: "E. D. Lee, C. P. Broedersz, W. Bialek",
    year: "2015",
    title: "Statistical Mechanics of the US Supreme Court",
    venue: "Journal of Statistical Physics 160, 275–301",
    href: "https://doi.org/10.1007/s10955-015-1253-6",
  },
];

export function References() {
  return (
    <section id="references" className="lab-science-section lab-science-refs">
      <h2 className="lab-science-act">References &amp; sources</h2>
      <p className="lab-science-note">
        This isn't a research paper, but the methods here are well-established
        and worth crediting. These are the works that the model and this
        explainer build on, in roughly the order their ideas appear above.
      </p>
      <ol className="lab-science-ref-list">
        {REFERENCES.map((r) => (
          <li key={r.id} id={r.id}>
            {r.authors} ({r.year}). {r.href ? (
              <a href={r.href} target="_blank" rel="noopener noreferrer">
                {r.title}
              </a>
            ) : (
              r.title
            )}
            . <span className="lab-science-ref-venue">{r.venue}</span>.
          </li>
        ))}
      </ol>
      <h4 className="lab-science-subhead">Data &amp; assets</h4>
      <ul className="lab-science-ref-list">
        <li>
          Tournament team data from{" "}
          <a
            href="https://limitlesstcg.com/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Limitless VGC
          </a>
          , whose public tournament results make the Pokémon model possible.
        </li>
        <li>
          Pokémon sprites courtesy of{" "}
          <a
            href="https://play.pokemonshowdown.com/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Pokémon Showdown
          </a>
          . Pokémon is © Nintendo / Creatures Inc. / GAME FREAK Inc.
        </li>
      </ul>
    </section>
  );
}
