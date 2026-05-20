// S1: motivation. Magnets demagnetize when heated. No interaction —
// labeled diagram and one paragraph of prose.

import { SpinGrid } from "../widgets/SpinGrid";
import type { Lattice } from "../primitives/lattice";

function makeAligned(): Lattice {
  return Array.from({ length: 10 }, () =>
    Array.from({ length: 10 }, () => 1 as const),
  );
}

function makeRandom(seed: number): Lattice {
  let s = seed;
  const next = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  return Array.from({ length: 10 }, () =>
    Array.from({ length: 10 }, () => (next() < 0.5 ? -1 : 1)),
  ) as Lattice;
}

export function Magnets() {
  return (
    <section id="magnets" className="lab-science-section">
      <h2>Magnets and the Curie point</h2>
      <p>
        A bar magnet works because the tiny magnetic moments inside it line up. Heat the
        bar past its <em>Curie temperature</em> and the alignment breaks down — the moments
        point in random directions and the bulk magnetism vanishes. Cool it back down and
        order returns. The Ising model is the simplest mathematical object that captures
        this: a collection of two-state "spins" that prefer to agree with their neighbors,
        but get jostled by thermal noise.
      </p>
      <div className="lab-science-row">
        <figure>
          <SpinGrid lattice={makeAligned()} cell={16} />
          <figcaption>Cold: aligned (magnetized)</figcaption>
        </figure>
        <figure>
          <SpinGrid lattice={makeRandom(7)} cell={16} />
          <figcaption>Hot: disordered (no net magnetism)</figcaption>
        </figure>
      </div>
    </section>
  );
}
