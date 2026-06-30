// §03 of /meta: the most-played exact rosters in the corpus, ranked by
// raw occurrence count. Teams render as a bare sprite strip — items
// show as corner-badge overlays and the full "Species @ Item" string is
// the hover title (see Sprite.tsx), so no per-member text is needed.
//
// No Streamlit counterpart yet; this section is webapp-only.

import { ScoreChip, MiniBar } from "../render/atoms";
import { TeamMiniStrip } from "../render/cells";
import { ScrollX } from "../components/ScrollX";
import { teamObservables } from "../render/observables";
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
}

export function TopTeamsTable({
  rows,
  maxCount,
  nCorpusTeams,
  model,
}: TopTeamsTableProps) {
  return (
    <ScrollX>
      <table className="lab-comp-table lab-table-cards">
        <thead>
          <tr>
            <th className="num">#</th>
            <th>team</th>
            <th className="num">score (raw)</th>
            <th className="num">coherence</th>
            <th className="num">count</th>
            <th className="num">share</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, rank) => {
            const share = nCorpusTeams > 0 ? r.count / nCorpusTeams : 0;
            // Bias Adjustment is irrelevant here (no field-weight slider on
            // /meta), so scoreRaw / coherence are the only field-independent
            // observables — pass any field weight.
            const obs = teamObservables(model, r.team, 0);
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
                <td className="num" data-label="score (raw)">
                  <ScoreChip value={obs.scoreRaw} />
                </td>
                <td className="num" data-label="coherence">
                  <ScoreChip value={obs.coherence} />
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
