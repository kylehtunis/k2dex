// Top-level shell for the /science page.
// Imports KaTeX CSS once; renders each section in order.

import { useEffect } from "react";
import { Magnets } from "../science/sections/Magnets";
import { Lattice } from "../science/sections/Lattice";
import { Graph } from "../science/sections/Graph";
import { Metropolis } from "../science/sections/Metropolis";
import { ParallelTempering } from "../science/sections/ParallelTempering";
import { MeanField } from "../science/sections/MeanField";
import { SCOTUS } from "../science/sections/SCOTUS";
import { Pokemon } from "../science/sections/Pokemon";
import { References } from "../science/sections/References";

export function SciencePage() {
  useEffect(() => {
    import("katex/dist/katex.min.css");
  }, []);
  return (
    <div className="lab-science">
      <header className="lab-science-header">
        <h1>The science behind k2dex</h1>
        <p className="lab-science-lede">
          Fitting a pairwise binary Markov Random Field via L2-regularized pseudolikelihood
          and sampling the resulting Boltzmann distribution with parallel-tempered Markov
          Chain Monte Carlo
          <span className="lab-science-lede-punch">(to get better at Pokémon)</span>
        </p>
      </header>
      <h2 className="lab-science-act">The Ising Model</h2>
      <Magnets />
      <Lattice />
      <Graph />
      <h2 className="lab-science-act">MCMC Sampling</h2>
      <Metropolis />
      <ParallelTempering />
      <MeanField />
      <h2 className="lab-science-act">The Inverse Problem</h2>
      <SCOTUS />
      <Pokemon />
      <References />
    </div>
  );
}
