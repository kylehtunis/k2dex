// SVG node-link diagram for small graphs (≤ ~15 nodes). Edge thickness
// scales with |weight|, color by sign. Caller is responsible for
// laying nodes out — positions are passed in as {x, y} pairs.

import { useEffect, useState } from "react";
import missingnoUrl from "../../assets/missingno.svg?url";

interface SpriteImageProps {
  href: string;
  x: number;
  y: number;
  size: number;
}

function SpriteImage({ href, x, y, size }: SpriteImageProps) {
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    setFailed(false);
  }, [href]);
  return (
    <image
      href={failed ? missingnoUrl : href}
      x={x}
      y={y}
      width={size}
      height={size}
      preserveAspectRatio="xMidYMid meet"
      onError={() => setFailed(true)}
    />
  );
}

export interface GraphNode {
  id: number;
  label: string;
  x: number;
  y: number;
  fill?: string;
  active?: boolean;
  /** Optional sprite URL. When set, the node renders as an image with a small
   * label below instead of the default circle + centered text. */
  sprite?: string;
}

export interface GraphEdge {
  i: number;
  j: number;
  weight: number;
}

export interface GraphViewProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  width: number;
  height: number;
  nodeRadius?: number;
  maxStrokeWidth?: number;
}

export function GraphView({
  nodes,
  edges,
  width,
  height,
  nodeRadius = 18,
  maxStrokeWidth = 5,
}: GraphViewProps) {
  const maxAbs = edges.reduce((m, e) => Math.max(m, Math.abs(e.weight)), 1e-9);
  const byId = new Map(nodes.map((n) => [n.id, n] as const));
  return (
    <svg width={width} height={height} className="lab-graphview">
      {edges.map((e, k) => {
        const a = byId.get(e.i);
        const b = byId.get(e.j);
        if (!a || !b) return null;
        const w = (Math.abs(e.weight) / maxAbs) * maxStrokeWidth;
        const color = e.weight >= 0 ? "#1f4e8c" : "#9c2a2a";
        return (
          <line
            key={k}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            stroke={color}
            strokeWidth={Math.max(0.5, w)}
            strokeLinecap="round"
            opacity={0.85}
          />
        );
      })}
      {nodes.map((n) => {
        if (n.sprite) {
          const size = nodeRadius * 2.2;
          return (
            <g key={n.id}>
              <SpriteImage
                href={n.sprite}
                x={n.x - size / 2}
                y={n.y - size / 2}
                size={size}
              />
              <text
                x={n.x}
                y={n.y + size / 2 + 12}
                textAnchor="middle"
                fontSize="11"
                fill={n.active ? "#1a1a1a" : "#333"}
                fontWeight={n.active ? 600 : 400}
              >
                {n.label}
              </text>
            </g>
          );
        }
        return (
          <g key={n.id}>
            <circle
              cx={n.x}
              cy={n.y}
              r={nodeRadius}
              fill={n.fill ?? (n.active ? "#1a1a1a" : "#e8e6df")}
              stroke="#444"
              strokeWidth={1}
            />
            <text
              x={n.x}
              y={n.y + 4}
              textAnchor="middle"
              fontSize="11"
              fill={n.active ? "#fff" : "#222"}
            >
              {n.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
