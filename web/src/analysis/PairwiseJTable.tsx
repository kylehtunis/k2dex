// §03 of /analysis: C(6,2) = 15 pairwise couplings sorted by |J|.
//
// Mirrors app.py:_render_pairwise_j_html.

import { PairCell } from "../render/cells";
import { ScoreChip, SignedBar } from "../render/atoms";
import type { PairwiseJRow } from "../render/observables";

export interface PairwiseJTableProps {
  rows: readonly PairwiseJRow[];
}

const SIGNED_BAR_MAX = 2.5;

export function PairwiseJTable({ rows }: PairwiseJTableProps) {
  return (
    <table className="lab-comp-table">
      <thead>
        <tr>
          <th className="num">#</th>
          <th>pair</th>
          <th className="num">Coupling</th>
          <th className="num">% of total</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.rank}>
            <td className="rank">{r.rank.toString().padStart(2, "0")}</td>
            <td>
              <PairCell nameA={r.nameA} nameB={r.nameB} />
            </td>
            <td className="num">
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  justifyContent: "flex-end",
                }}
              >
                <SignedBar value={r.jValue} maxValue={SIGNED_BAR_MAX} width={80} />
                <ScoreChip value={r.jValue} />
              </div>
            </td>
            <td className="num">{(r.pctOfAbsSum * 100).toFixed(1)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
