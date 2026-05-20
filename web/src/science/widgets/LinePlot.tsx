// Minimal SVG line plot for /science widgets. One or more series, shared
// y-domain, optional axis labels. Axes are bare (no ticks beyond endpoints);
// keeps the lab-notebook aesthetic without pulling in a chart library.

export interface LinePlotSeries {
  data: number[];
  color: string;
  dashed?: boolean;
  label?: string;
}

export interface LinePlotProps {
  width: number;
  height: number;
  series: LinePlotSeries[];
  yDomain: [number, number];
  xLabel?: string;
  yLabel?: string;
}

const PAD = { left: 36, right: 8, top: 8, bottom: 24 };

function makePath(
  data: number[],
  xMax: number,
  yDomain: [number, number],
  W: number,
  H: number,
): string {
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const [y0, y1] = yDomain;
  const parts: string[] = [];
  for (let i = 0; i < data.length; i++) {
    const x = PAD.left + (xMax === 0 ? 0 : (i / xMax) * innerW);
    const y = PAD.top + innerH - ((data[i] - y0) / (y1 - y0)) * innerH;
    parts.push(`${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`);
  }
  return parts.join(" ");
}

export function LinePlot({ width, height, series, yDomain, xLabel, yLabel }: LinePlotProps) {
  const xMax = Math.max(0, ...series.map((s) => s.data.length - 1));
  return (
    <svg width={width} height={height} role="img" className="lab-lineplot">
      <line
        x1={PAD.left}
        x2={width - PAD.right}
        y1={height - PAD.bottom}
        y2={height - PAD.bottom}
        stroke="#666"
      />
      <line
        x1={PAD.left}
        x2={PAD.left}
        y1={PAD.top}
        y2={height - PAD.bottom}
        stroke="#666"
      />
      <text x={PAD.left - 4} y={PAD.top + 8} textAnchor="end" fontSize="10">
        {yDomain[1].toFixed(2)}
      </text>
      <text x={PAD.left - 4} y={height - PAD.bottom} textAnchor="end" fontSize="10">
        {yDomain[0].toFixed(2)}
      </text>
      {yLabel && (
        <text
          x={4}
          y={height / 2}
          fontSize="10"
          transform={`rotate(-90 4 ${height / 2})`}
        >
          {yLabel}
        </text>
      )}
      {xLabel && (
        <text x={width / 2} y={height - 4} fontSize="10" textAnchor="middle">
          {xLabel}
        </text>
      )}
      {series.map((s, k) => (
        <path
          key={k}
          d={makePath(s.data, xMax, yDomain, width, height)}
          stroke={s.color}
          strokeDasharray={s.dashed ? "4 3" : undefined}
          fill="none"
          strokeWidth={1.5}
        />
      ))}
    </svg>
  );
}
