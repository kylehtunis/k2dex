// SVG spin lattice. Up-spin = dark, down-spin = light. Optional highlight
// for the most-recently-considered cell (used by S3 in step-by-step mode).

import type { Lattice } from "../primitives/lattice";

export interface SpinGridProps {
  lattice: Lattice;
  cell?: number;
  highlight?: { i: number; j: number } | null;
}

export function SpinGrid({ lattice, cell = 14, highlight = null }: SpinGridProps) {
  const R = lattice.length;
  const C = lattice[0]?.length ?? 0;
  const W = C * cell;
  const H = R * cell;
  return (
    <svg width={W} height={H} className="lab-spingrid" shapeRendering="crispEdges">
      {lattice.flatMap((row, i) =>
        row.map((s, j) => (
          <rect
            key={`${i}-${j}`}
            x={j * cell}
            y={i * cell}
            width={cell}
            height={cell}
            fill={s === 1 ? "#1a1a1a" : "#e8e6df"}
            stroke="#aaa"
            strokeWidth={0.5}
          />
        )),
      )}
      {highlight && (
        <rect
          x={highlight.j * cell}
          y={highlight.i * cell}
          width={cell}
          height={cell}
          fill="none"
          stroke="#d97706"
          strokeWidth={2}
        />
      )}
    </svg>
  );
}
