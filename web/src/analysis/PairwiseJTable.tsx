// §03 of /analysis: C(6,2) = 15 pairwise couplings sorted by |J|.
//

import { PairCell } from "../render/cells";
import { ScoreChip, SignedBar } from "../render/atoms";
import type { PairwiseJRow } from "../render/observables";

export interface PairwiseJTableProps {
  rows: readonly PairwiseJRow[];
}

export function PairwiseJTable({ rows }: PairwiseJTableProps) {
  // Scale the bars to the strongest coupling on screen, like every other
  // signed-bar table. A fixed cap would flatten any pair beyond it, making a
  // very strong coupling indistinguishable from a merely strong one.
  const barMax = Math.max(...rows.map((r) => Math.abs(r.jValue)), 1e-9);
  return (
    <table className="lab-comp-table lab-table-pairs">
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
            <td className="pair">
              <PairCell nameA={r.nameA} nameB={r.nameB} />
            </td>
            <td className="num" data-label="coupling">
              <div className="lab-coupling-val">
                <SignedBar value={r.jValue} maxValue={barMax} width={80} />
                <ScoreChip value={r.jValue} />
              </div>
            </td>
            <td className="num" data-label="% of total">{(r.pctOfAbsSum * 100).toFixed(1)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
