// Top-level shell for the /science page.
// Imports KaTeX CSS once; renders each section in order.

import { useEffect } from "react";
import { S1Magnets } from "../science/sections/S1Magnets";

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
    </div>
  );
}
