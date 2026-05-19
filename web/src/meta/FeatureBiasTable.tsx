// §02 of /meta: top features by Bias (h) in the chosen direction. The
// +Bias and −Bias tables share the m̂ scale so the unipolar bars are
// directly comparable across columns.
//
// Mirrors app.py:_render_feature_h_table_html.

import { ScoreChip, MiniBar } from "../render/atoms";
import { InlineMon } from "../render/cells";
import type { IsingModel } from "../sampler/types";

export interface FeatureBiasTableProps {
  /** Vocab indices in row order (already sorted by ±h). */
  order: readonly number[];
  /** Shared max for the m̂ mini-bar so both columns calibrate together. */
  maxM: number;
  model: IsingModel;
}

export function FeatureBiasTable({ order, maxM, model }: FeatureBiasTableProps) {
  return (
    <table className="lab-comp-table">
      <thead>
        <tr>
          <th className="num">#</th>
          <th>feature</th>
          <th className="num">Bias</th>
          <th className="num">m̂</th>
        </tr>
      </thead>
      <tbody>
        {order.map((i, rank) => {
          const mI = model.m[i];
          return (
            <tr key={i}>
              <td className="rank">{(rank + 1).toString().padStart(2, "0")}</td>
              <td>
                <InlineMon name={model.vocab[i]} />
              </td>
              <td className="num">
                <ScoreChip value={model.h[i]} />
              </td>
              <td className="num">
                <div className="lab-comp-freq">
                  <span className="lab-comp-freq-pct">
                    {(mI * 100).toFixed(1)}%
                  </span>
                  <MiniBar value={mI} maxValue={maxM} width={80} />
                </div>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
