// "Teammates %" explorer — the familiar Smogon / Pikalytics display, computed
// live from the corpus. Pick a species; see its top teammates ranked by
// co-occurrence percentage. This is the recognizable hook for the article:
// it's exactly the table every other teambuilder shows, and the whole point of
// §1 is that this table alone is not enough.

import { useMemo, useState } from "react";
import Select, { type SingleValue } from "react-select";
import { MENU_PORTAL_TARGET } from "../../components/portalTarget";
import type { IsingModel } from "../../sampler/types";
import { SpriteBox } from "../../render/Sprite";
import {
  speciesTeammates,
  type SpeciesCooccurrence,
} from "../cooccurrenceSpecies";

const TOP_N = 12;
const portalStyles = {
  menuPortal: (base: Record<string, unknown>) => ({ ...base, zIndex: 9999 }),
};

interface Opt {
  label: string;
  value: number;
}

export function TeammateTable({
  model,
  sc,
}: {
  model: IsingModel;
  sc: SpeciesCooccurrence;
}) {
  // Species options sorted by popularity (appearance count).
  const options = useMemo<Opt[]>(() => {
    return Array.from({ length: sc.S }, (_, s) => s)
      .filter((s) => sc.siteAppear[s] > 0)
      .sort((a, b) => sc.siteAppear[b] - sc.siteAppear[a])
      .map((s) => ({ label: model.sites[s], value: s }));
  }, [model, sc]);

  // Default to the most popular species.
  const [anchor, setAnchor] = useState<number>(() => options[0]?.value ?? 0);

  const teammates = useMemo(
    () => speciesTeammates(sc, anchor, TOP_N),
    [sc, anchor],
  );
  const maxPct = teammates.length > 0 ? teammates[0].pct : 1;

  return (
    <div className="lab-teammate-widget">
      <label className="lab-teammate-picker">
        <span>Show teammates for</span>
        <Select<Opt>
          classNamePrefix="lab-select"
          options={options}
          value={options.find((o) => o.value === anchor) ?? null}
          onChange={(o: SingleValue<Opt>) => o && setAnchor(o.value)}
          menuPortalTarget={MENU_PORTAL_TARGET}
          styles={portalStyles}
          aria-label="Anchor species"
        />
      </label>
      <ol className="lab-teammate-table">
        {teammates.map((t, i) => (
          <li key={t.site} className="lab-teammate-row">
            <span className="lab-teammate-rank">{i + 1}</span>
            <SpriteBox name={model.sites[t.site]} size={32} />
            <span className="lab-teammate-name">{model.sites[t.site]}</span>
            <span className="lab-teammate-bar-wrap" aria-hidden="true">
              <span
                className="lab-teammate-bar"
                style={{ width: `${(t.pct / maxPct) * 100}%` }}
              />
            </span>
            <span className="lab-teammate-pct">
              {(t.pct * 100).toFixed(1)}%
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
