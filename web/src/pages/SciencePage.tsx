// Top-level shell for the /science page.
// Imports KaTeX CSS once; renders each section in order.

import { useEffect } from "react";
import { S1Magnets } from "../science/sections/S1Magnets";
import { S2Lattice } from "../science/sections/S2Lattice";
import { S3MCMC } from "../science/sections/S3MCMC";
import { S4PT } from "../science/sections/S4PT";
import { S5MeanField } from "../science/sections/S5MeanField";
import { S6SCOTUS } from "../science/sections/S6SCOTUS";

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
          This page walks through what that model is, why it works, and what it can't see.
        </p>
      </header>
      <S1Magnets />
      <S2Lattice />
      <S3MCMC />
      <S4PT />
      <S5MeanField />
      <S6SCOTUS />
    </div>
  );
}
