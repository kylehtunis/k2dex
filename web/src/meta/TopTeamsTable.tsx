// §03 of /meta: the most-played exact rosters in the corpus, ranked by
// raw occurrence count. Teams render as a bare sprite strip — items
// show as corner-badge overlays and the full "Species @ Item" string is
// the hover title (see Sprite.tsx), so no per-member text is needed.
//
// No Streamlit counterpart yet; this section is webapp-only.

import { Link } from "react-router-dom";
import { ScoreChip, MiniBar } from "../render/atoms";
import { TeamMiniStrip } from "../render/cells";
import { ScrollX } from "../components/ScrollX";
import {
  percentileTitle,
  type CorpusScoreIndex,
} from "../render/corpusScore";
import { teamObservables } from "../render/observables";
import { encodeCore } from "../render/shareLink";
import type { IsingModel } from "../sampler/types";
import type { TopTeam } from "./topTeams";

export interface TopTeamsTableProps {
  /** Pre-ranked rows (top-K by occurrence count). */
  rows: readonly TopTeam[];
  /** Top team's count — calibrates the share mini-bar. */
  maxCount: number;
  /** Corpus size, for the share-of-corpus percentage. */
  nCorpusTeams: number;
  model: IsingModel;
  /** When provided, score/coherence chips show corpus percentiles on hover. */
  scoreIndex?: CorpusScoreIndex | null;
}

export function TopTeamsTable({
  rows,
  maxCount,
  nCorpusTeams,
  model,
  scoreIndex,
}: TopTeamsTableProps) {
  return (
    <ScrollX>
      <table className="lab-comp-table lab-table-cards">
        <thead>
          <tr>
            <th className="num">#</th>
            <th>team</th>
            <th />
            <th className="num">score</th>
            <th className="num">coherence</th>
            <th className="num">count</th>
            <th className="num">share</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, rank) => {
            const share = nCorpusTeams > 0 ? r.count / nCorpusTeams : 0;
            // scoreRaw / coherence don't depend on the fieldWeight
            // argument, so its value here is arbitrary.
            const obs = teamObservables(model, r.team);
            return (
              <tr key={r.team.join("-")} className={rank === 0 ? "top-row" : undefined}>
                <td className="rank">{(rank + 1).toString().padStart(2, "0")}</td>
                <td className="team">
                  <TeamMiniStrip
                    names={r.team.map((i) => model.vocab[i])}
                    size={40}
                    interactive
                  />
                </td>
                <td className="actions">
                  <Link
                    to={`/analysis?t=${encodeCore(model.id, r.team, model)}`}
                    className="lab-analyze-btn"
                  >
                    Send to Analysis
                  </Link>
                </td>
                <td className="num" data-label="score">
                  <ScoreChip
                    value={obs.scoreRaw}
                    title={
                      scoreIndex != null
                        ? percentileTitle(scoreIndex.score, obs.scoreRaw)
                        : undefined
                    }
                  />
                </td>
                <td className="num" data-label="coherence">
                  <ScoreChip
                    value={obs.coherence}
                    title={
                      scoreIndex != null
                        ? percentileTitle(scoreIndex.coherence, obs.coherence)
                        : undefined
                    }
                  />
                </td>
                <td className="num" data-label="count">
                  <ScoreChip value={r.count} fmt="count" />
                </td>
                <td className="num" data-label="share">
                  <div className="lab-comp-freq">
                    <span className="lab-comp-freq-pct">
                      {(share * 100).toFixed(2)}%
                    </span>
                    <MiniBar value={r.count} maxValue={maxCount} width={80} />
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </ScrollX>
  );
}
