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
      <h4>
        When you heat a magnet it loses its magnetism. 
        When it cools back down the magnetism returns, but sometimes the North and South poles swap.
        What causes that, and what on earth does it have to do with Pokemon? 
        
        </h4>
        <p>
        I'll walk you through understanding how the physics of magnets can be used to build competitive Pokemon teams 
        (or understand voting patterns of the United States Supreme Court).
        The math might be a little intimidating, but this page is filled with interactive widgets to help you understand 
        exactly what each equation is doing and how everything fits together.
      </p>

    <h2>Magnets and the Curie point</h2>
      <p>
        A any sort of magnet can be thought of as composed of individual tiny magnets, each with it's own magnetic field, called a "spin".
        These spins can point either up or down, also called +1 and -1.
        When the spins all align, their magnetic fields combine to create the strong magnetic effect we can see macroscopically.
        The more aligned the spins are, the lower the energy of the system, and the stronger the magnet. 
        Since nature prefers low energy states, these spins will align by themselves and create a macroscopic magnet.
      </p>
      <p>
        However, spins can be randomly flipped by thermal fluctuations. Flips into lower energy states are more likely, 
        but if you add enough heat then the spins will flip so often that they can't maintain their alignment. This is when the magnet "demagnetizes".
        The temperature this happens at is called the Curie point.
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
