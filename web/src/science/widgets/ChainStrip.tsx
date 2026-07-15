// Replica-exchange timeline: K horizontal rows (one per temperature rung).
// At each sweep, color the cell by which original replica id occupies that
// rung — swap markers appear as color discontinuities. Used in S4.

export interface ChainStripProps {
  /** history[sweep][rungIndex] = originalReplicaId */
  history: number[][];
  K: number;
  width: number;
  height: number;
}

import { useIsClient } from "./useIsClient";

const COLORS = ["#1f4e8c", "#9c2a2a", "#2f7d4f", "#b3791c", "#6a3d9a", "#777"];

export function ChainStrip({ history, K, width, height }: ChainStripProps) {
  const isClient = useIsClient();
  const sweeps = history.length;
  if (!isClient) {
    return <svg width={width} height={height} className="lab-chainstrip" />;
  }
  const rowH = height / K;
  const colW = sweeps === 0 ? 0 : width / sweeps;
  return (
    <svg width={width} height={height} className="lab-chainstrip">
      {Array.from({ length: K }, (_, k) => (
        <g key={k} className="lab-chainstrip-rung">
          {history.map((row, t) => (
            <rect
              key={t}
              x={t * colW}
              y={k * rowH}
              width={Math.max(colW, 1)}
              height={rowH}
              fill={COLORS[row[k] % COLORS.length]}
            />
          ))}
        </g>
      ))}
    </svg>
  );
}
