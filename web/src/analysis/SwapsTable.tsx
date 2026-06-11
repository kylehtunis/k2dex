// §04 of /analysis: top single-swap suggestions, evaluated independently
// from the starting team (each row is one applied swap; not a chain).
//
// Mirrors app.py:_render_swap_table_html.
//
// Δ columns use sign-flipped values (positive = improvement) since the
// model's ΔE is Hamiltonian-space (lower = better).

import { ScoreChip, CorpusCell } from "../render/atoms";
import { SwapCell, TeamMiniStrip } from "../render/cells";
import type { IsingModel, SingleSwapEntry, TeamCounts } from "../sampler/types";
import { nearestObserved } from "../render/corpus";

export interface SwapsTableProps {
  swaps: readonly SingleSwapEntry[];
  teamIdx: readonly number[];
  model: IsingModel;
  teamCounts: TeamCounts | null;
}

export function SwapsTable({
  swaps,
  teamIdx,
  model,
  teamCounts,
}: SwapsTableProps) {
  const teamSet = new Set(teamIdx);
  return (
    <table className="lab-comp-table lab-table-cards">
      <thead>
        <tr>
          <th className="num">#</th>
          <th>swap</th>
          <th className="num">Δ Score (adj)</th>
          <th className="num">Δ Score (raw)</th>
          <th className="num">Δ Coherence</th>
          <th className="num">corpus</th>
          <th>team after</th>
        </tr>
      </thead>
      <tbody>
        {swaps.map((sw, idx) => {
          const afterIdx = [...teamSet].filter((i) => i !== sw.outIdx);
          afterIdx.push(sw.inIdx);
          afterIdx.sort((a, b) => a - b);
          const corpus = nearestObserved(afterIdx, teamCounts);
          return (
            <tr key={`${sw.outIdx}-${sw.inIdx}`}>
              <td className="rank">{(idx + 1).toString().padStart(2, "0")}</td>
              <td className="swap">
                <SwapCell
                  nameOut={model.vocab[sw.outIdx]}
                  nameIn={model.vocab[sw.inIdx]}
                />
              </td>
              <td className="num" data-label="Δ score adj">
                <ScoreChip value={-sw.deltaEAdj} />
              </td>
              <td className="num" data-label="Δ score raw">
                <ScoreChip value={-sw.deltaERaw} />
              </td>
              <td className="num" data-label="Δ coherence">
                <ScoreChip value={sw.deltaSumJ} />
              </td>
              <td className="num" data-label="corpus">
                {corpus !== null && (
                  <CorpusCell delta={corpus.delta} count={corpus.count} />
                )}
              </td>
              <td className="team-after">
                <TeamMiniStrip names={afterIdx.map((i) => model.vocab[i])} />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
