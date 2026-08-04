// §05 of /analysis: greedy single-swap chain critique. Each row is the
// team's state AFTER that swap step. Row 0 is the starting team
// (italic "starting team" label, no swap cell).
//

import { ScoreChip, CorpusCell } from "../render/atoms";
import { SwapCell, TeamMiniStrip } from "../render/cells";
import { nearestObserved } from "../render/corpus";
import {
  percentileTitle,
  type CorpusScoreIndex,
} from "../render/corpusScore";
import { formatSigned } from "../render/format";
import type { GreedyChainEntry, IsingModel, TeamCounts } from "../sampler/types";

export interface ChainTableProps {
  startingTeam: readonly number[];
  chain: readonly GreedyChainEntry[];
  startScore: number;
  startSumJ: number;
  model: IsingModel;
  teamCounts: TeamCounts | null;
  /** When provided, each row's Score and Coherence expose their corpus
   * percentile on hover. */
  scoreIndex?: CorpusScoreIndex | null;
}

export function ChainTable({
  startingTeam,
  chain,
  startScore,
  startSumJ,
  model,
  teamCounts,
  scoreIndex,
}: ChainTableProps) {
  const scoreDist = scoreIndex?.score;
  const coherenceDist = scoreIndex?.coherence;
  const startCorpus = nearestObserved(startingTeam, teamCounts);
  return (
    <table className="lab-comp-table lab-table-cards">
      <thead>
        <tr>
          <th className="num">#</th>
          <th>swap</th>
          <th className="num">Score</th>
          <th className="num">Coherence</th>
          <th className="num">corpus</th>
          <th>team after</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="rank">00</td>
          <td className="swap">
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
          <td className="num" data-label="score">
            <span
              title={scoreDist && percentileTitle(scoreDist, startScore)}
            >
              {formatSigned(startScore)}
            </span>
          </td>
          <td className="num" data-label="coherence">
            <ScoreChip
              value={startSumJ}
              title={coherenceDist && percentileTitle(coherenceDist, startSumJ)}
            />
          </td>
          <td className="num" data-label="corpus">
            {startCorpus !== null && (
              <CorpusCell delta={startCorpus.delta} count={startCorpus.count} />
            )}
          </td>
          <td className="team-after">
            <TeamMiniStrip names={startingTeam.map((i) => model.vocab[i])} />
          </td>
        </tr>
        {chain.map((ev) => {
          const afterCorpus = nearestObserved(ev.teamAfter, teamCounts);
          // ev.energyRawAfter is Hamiltonian-space (lower = better).
          // Sign-flip to Score (higher = better).
          const score = -ev.energyRawAfter;
          return (
            <tr key={ev.step}>
              <td className="rank">{ev.step.toString().padStart(2, "0")}</td>
              <td className="swap">
                <SwapCell
                  nameOut={model.vocab[ev.outIdx]}
                  nameIn={model.vocab[ev.inIdx]}
                />
              </td>
              <td className="num" data-label="score">
                <span title={scoreDist && percentileTitle(scoreDist, score)}>
                  {formatSigned(score)}
                </span>
              </td>
              <td className="num" data-label="coherence">
                <ScoreChip
                  value={ev.sumJAfter}
                  title={
                    coherenceDist &&
                    percentileTitle(coherenceDist, ev.sumJAfter)
                  }
                />
              </td>
              <td className="num" data-label="corpus">
                {afterCorpus !== null && (
                  <CorpusCell
                    delta={afterCorpus.delta}
                    count={afterCorpus.count}
                  />
                )}
              </td>
              <td className="team-after">
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
