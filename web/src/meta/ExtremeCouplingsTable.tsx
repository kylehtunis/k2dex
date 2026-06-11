// §03 of /meta: top pairs by Coupling (J) in the chosen direction.
// Pre-filtered upstream to drop same-species and same-item pairs on
// Species @ Item (those are mechanical mutual-exclusions, not real
// model structure). The +Coupling and −Coupling tables share `maxJ`
// so the signed bars are comparable.
//
// Mirrors app.py:_render_extreme_pairs_html.

import { ScoreChip, SignedBar } from "../render/atoms";
import { PairCell } from "../render/cells";
import { ScrollX } from "../components/ScrollX";
import type { IsingModel } from "../sampler/types";
import type { CouplingPair } from "./couplings";

export interface ExtremeCouplingsTableProps {
  /** Pre-sorted rows (top-K by ±J). */
  rows: readonly CouplingPair[];
  /** Shared bar scale across +Coupling and −Coupling tables. */
  maxJ: number;
  model: IsingModel;
}

export function ExtremeCouplingsTable({
  rows,
  maxJ,
  model,
}: ExtremeCouplingsTableProps) {
  return (
    <ScrollX>
    <table className="lab-comp-table lab-table-pairs">
      <thead>
        <tr>
          <th className="num">#</th>
          <th>pair</th>
          <th className="num">Coupling</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, rank) => (
          <tr key={`${r.i}-${r.j}`}>
            <td className="rank">{(rank + 1).toString().padStart(2, "0")}</td>
            <td className="pair">
              <PairCell nameA={model.vocab[r.i]} nameB={model.vocab[r.j]} />
            </td>
            <td className="num" data-label="coupling">
              <div className="lab-coupling-val">
                <SignedBar value={r.jValue} maxValue={maxJ} width={80} />
                <ScoreChip value={r.jValue} />
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
    </ScrollX>
  );
}
