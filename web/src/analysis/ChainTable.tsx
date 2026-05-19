// §05 of /analysis: greedy single-swap chain critique. Each row is the
// team's state AFTER that swap step. Row 0 is the starting team
// (italic "starting team" label, no swap cell).
//
// Mirrors app.py:_render_chain_table_html.

import { ScoreChip, CorpusCell } from "../render/atoms";
import { SwapCell, TeamMiniStrip } from "../render/cells";
import { nearestObserved } from "../render/corpus";
import { formatSigned } from "../render/format";
import type { GreedyChainEntry, IsingModel, TeamCounts } from "../sampler/types";

export interface ChainTableProps {
  startingTeam: readonly number[];
  chain: readonly GreedyChainEntry[];
  startScoreAdj: number;
  startScoreRaw: number;
  startSumJ: number;
  model: IsingModel;
  teamCounts: TeamCounts | null;
}

export function ChainTable({
  startingTeam,
  chain,
  startScoreAdj,
  startScoreRaw,
  startSumJ,
  model,
  teamCounts,
}: ChainTableProps) {
  const startCorpus = nearestObserved(startingTeam, teamCounts);
  return (
    <table className="lab-comp-table">
      <thead>
        <tr>
          <th className="num">#</th>
          <th>swap</th>
          <th className="num">Score (adj)</th>
          <th className="num">Score (raw)</th>
          <th className="num">Coherence</th>
          <th className="num">corpus</th>
          <th>team after</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="rank">00</td>
          <td>
            <span
              style={{
                fontFamily: "var(--lab-font-serif)",
                fontStyle: "italic",
                color: "var(--lab-ink-muted)",
              }}
            >
              starting team
            </span>
          </td>
          <td className="num">{formatSigned(startScoreAdj)}</td>
          <td className="num">{formatSigned(startScoreRaw)}</td>
          <td className="num">
            <ScoreChip value={startSumJ} />
          </td>
          <td className="num">
            {startCorpus !== null && (
              <CorpusCell delta={startCorpus.delta} count={startCorpus.count} />
            )}
          </td>
          <td>
            <TeamMiniStrip names={startingTeam.map((i) => model.vocab[i])} />
          </td>
        </tr>
        {chain.map((ev) => {
          const afterCorpus = nearestObserved(ev.teamAfter, teamCounts);
          // ev.energyAdjAfter and energyRawAfter are Hamiltonian-space
          // (lower = better). Sign-flip to Score (higher = better).
          const scoreAdj = -ev.energyAdjAfter;
          const scoreRaw = -ev.energyRawAfter;
          return (
            <tr key={ev.step}>
              <td className="rank">{ev.step.toString().padStart(2, "0")}</td>
              <td>
                <SwapCell
                  nameOut={model.vocab[ev.outIdx]}
                  nameIn={model.vocab[ev.inIdx]}
                />
              </td>
              <td className="num">{formatSigned(scoreAdj)}</td>
              <td className="num">{formatSigned(scoreRaw)}</td>
              <td className="num">
                <ScoreChip value={ev.sumJAfter} />
              </td>
              <td className="num">
                {afterCorpus !== null && (
                  <CorpusCell
                    delta={afterCorpus.delta}
                    count={afterCorpus.count}
                  />
                )}
              </td>
              <td>
                <TeamMiniStrip
                  names={ev.teamAfter.map((i) => model.vocab[i])}
                />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
