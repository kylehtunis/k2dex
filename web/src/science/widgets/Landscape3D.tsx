// SVG isometric mesh for a 2D energy landscape, plus overlaid walker markers.
// Renders a precomputed grid of triangulated quads in painter's order (far-to-
// near by x+y), depth-shaded by energy. Walkers are drawn on top.

import { useMemo } from "react";
import {
  DOMAIN_MAX,
  DOMAIN_MIN,
  energyAt,
  project,
  type IsoProjection,
} from "../primitives/landscape";
import { useIsClient } from "./useIsClient";

export interface Walker {
  x: number;
  y: number;
  color: string;
  label?: string;
  trail?: ReadonlyArray<{ x: number; y: number }>;
}

export interface Landscape3DProps {
  width: number;
  height: number;
  walkers?: Walker[];
  grid?: number;
}

interface Quad {
  points: string;
  fill: string;
  depth: number;
}

function quadColor(t: number): string {
  // Valleys (t near 0) → muted blue; peaks (t near 1) → pale parchment.
  const r = Math.round(80 + (236 - 80) * t);
  const g = Math.round(120 + (228 - 120) * t);
  const b = Math.round(170 + (200 - 170) * t);
  return `rgb(${r},${g},${b})`;
}

export function Landscape3D({
  width,
  height,
  walkers = [],
  grid = 28,
}: Landscape3DProps) {
  const isClient = useIsClient();
  const proj: IsoProjection = useMemo(
    () => ({
      scaleXY: width / 8.4,
      scaleZ: height / 9.6,
      centerU: width / 2,
      centerV: height * 0.62,
    }),
    [width, height],
  );

  const { quads } = useMemo(() => {
    const dx = (DOMAIN_MAX - DOMAIN_MIN) / (grid - 1);
    const E: number[][] = [];
    let zmin = Infinity;
    let zmax = -Infinity;
    for (let i = 0; i < grid; i++) {
      const row: number[] = [];
      for (let j = 0; j < grid; j++) {
        const x = DOMAIN_MIN + i * dx;
        const y = DOMAIN_MIN + j * dx;
        const z = energyAt(x, y);
        if (z < zmin) zmin = z;
        if (z > zmax) zmax = z;
        row.push(z);
      }
      E.push(row);
    }
    const span = Math.max(zmax - zmin, 1e-9);
    const quads: Quad[] = [];
    for (let i = 0; i < grid - 1; i++) {
      for (let j = 0; j < grid - 1; j++) {
        const x0 = DOMAIN_MIN + i * dx;
        const y0 = DOMAIN_MIN + j * dx;
        const x1 = DOMAIN_MIN + (i + 1) * dx;
        const y1 = DOMAIN_MIN + (j + 1) * dx;
        const z00 = E[i][j];
        const z10 = E[i + 1][j];
        const z11 = E[i + 1][j + 1];
        const z01 = E[i][j + 1];
        const a = project({ x: x0, y: y0, z: z00 }, proj);
        const b = project({ x: x1, y: y0, z: z10 }, proj);
        const c = project({ x: x1, y: y1, z: z11 }, proj);
        const d = project({ x: x0, y: y1, z: z01 }, proj);
        const zavg = (z00 + z10 + z11 + z01) / 4;
        const t = (zavg - zmin) / span;
        const depth = (x0 + x1 + y0 + y1) / 4;
        quads.push({
          points: `${a.u},${a.v} ${b.u},${b.v} ${c.u},${c.v} ${d.u},${d.v}`,
          fill: quadColor(t),
          depth,
        });
      }
    }
    quads.sort((p, q) => p.depth - q.depth);
    return { quads };
  }, [grid, proj]);

  // After all hooks: skip the mesh during build-time prerendering.
  if (!isClient) {
    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="lab-landscape3d" />
    );
  }

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="lab-landscape3d">
      <rect width={width} height={height} fill="#fafaf6" />
      {quads.map((q, k) => (
        <polygon
          key={k}
          points={q.points}
          fill={q.fill}
          stroke="rgba(0,0,0,0.08)"
          strokeWidth={0.5}
        />
      ))}
      {walkers.map((w, k) => {
        if (!w.trail || w.trail.length < 2) return null;
        const pts = w.trail.map((q) =>
          project({ x: q.x, y: q.y, z: energyAt(q.x, q.y) }, proj),
        );
        const d = pts
          .map((p, i) => `${i === 0 ? "M" : "L"}${p.u.toFixed(1)},${p.v.toFixed(1)}`)
          .join(" ");
        return (
          <path
            key={`trail-${k}`}
            d={d}
            fill="none"
            stroke={w.color}
            strokeWidth={1.2}
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity={0.45}
          />
        );
      })}
      {walkers.map((w, k) => {
        const z = energyAt(w.x, w.y);
        const p = project({ x: w.x, y: w.y, z }, proj);
        return (
          <g key={k}>
            <circle
              cx={p.u}
              cy={p.v}
              r={6}
              fill={w.color}
              stroke="#fff"
              strokeWidth={1.6}
            />
            {w.label ? (
              <text
                x={p.u + 9}
                y={p.v - 7}
                fontSize={11}
                fontFamily="var(--lab-font-mono)"
                fill="#222"
              >
                {w.label}
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}
