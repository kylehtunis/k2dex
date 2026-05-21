// Top-level shell for the /science page.
// Imports KaTeX CSS once; renders each section in order.

import { useEffect } from "react";
import { Magnets } from "../science/sections/Magnets";
import { Lattice } from "../science/sections/Lattice";
import { Graph } from "../science/sections/Graph";
import { MCMC } from "../science/sections/MCMC";
import { PT } from "../science/sections/PT";
import { MeanField } from "../science/sections/MeanField";
import { SCOTUS } from "../science/sections/SCOTUS";
import { Pokemon } from "../science/sections/Pokemon";

export function SciencePage() {
  useEffect(() => {
    import("katex/dist/katex.min.css");
  }, []);
  return (
    <div className="lab-science">
      <header className="lab-science-header">
        <h1>The science behind k2dex</h1>
        <p className="lab-science-lede">
          The completer, analysis, and meta tools all rest on a single statistical model.
          This page uses interactive examples to explain what that model is and how it works.
        </p>
      </header>
      <Magnets />
      <Lattice />
      <Graph />
      <MCMC />
      <PT />
      <MeanField />
      <SCOTUS />
      <Pokemon />
    </div>
  );
}
