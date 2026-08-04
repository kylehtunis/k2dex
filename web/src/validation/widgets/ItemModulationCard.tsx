// A single curated item-interaction example. Given a "modulated" species whose
// item is varied and a partner species, it reads the live model's couplings and
// draws a diverging bar per item: how that specific build's synergy with the
// partner swings from strongly positive to negative. The numbers are read from
// the active model, never hardcoded, so they stay honest if the fit changes.

import { useMemo, type ReactNode } from "react";
import type { IsingModel } from "../../sampler/types";
import { SpriteBox } from "../../render/Sprite";

const POS_COLOR = "#1f4e8c"; // blue = positive coupling (matches the /science graph)
const NEG_COLOR = "#9c2a2a"; // red = negative coupling

export interface ModulationItem {
  /** Full vocab string, e.g. "Charizard @ Charizardite Y". */
  vocab: string;
  /** Short label shown on the row, e.g. "Charizardite Y". */
  label: string;
}

export interface ModulationExample {
  id: string;
  /** Species whose item is varied. */
  modulatedSpecies: string;
  /** Partner species the synergy is measured against. */
  partnerSpecies: string;
  /** Item builds to display, top to bottom. */
  items: ModulationItem[];
  headline: string;
  body: ReactNode;
}

/** Whether an example can render a real comparison in `model`: the partner
 * species must exist and at least two of its item builds must be in vocab. Lets
 * the section drop examples a given regulation's model can't support (e.g. a
 * build that appears in one regulation but not another) instead of showing a
 * degenerate single-bar card. */
export function isExampleResolvable(
  model: IsingModel,
  example: ModulationExample,
): boolean {
  if (!model.sites.includes(example.partnerSpecies)) return false;
  const resolvable = example.items.filter(
    (it) => model.indexOf.get(it.vocab) !== undefined,
  ).length;
  return resolvable >= 2;
}

/** Usage-weighted average coupling between one feature and every build of the
 * partner species — the partner's items weighted by how often they're actually
 * used, so the number reflects the real matchup rather than a flat average.
 *
 * Weighting matches `potts.py:FittedModel.item_weights`, including its fallback:
 * a partner with no marginal mass falls back to uniform weights (a flat mean),
 * not to zero, which would read as "no interaction" for a thin species. */
function weightedSynergy(
  model: IsingModel,
  feature: number,
  partnerSite: number,
): number {
  const partnerFeats = model.siteFeatures[partnerSite];
  if (partnerFeats.length === 0) return 0;
  let acc = 0;
  let wsum = 0;
  const base = feature * model.V;
  for (const g of partnerFeats) {
    acc += model.J[base + g] * model.m[g];
    wsum += model.m[g];
  }
  if (wsum > 0) return acc / wsum;
  let flat = 0;
  for (const g of partnerFeats) flat += model.J[base + g];
  return flat / partnerFeats.length;
}

export function ItemModulationCard({
  model,
  example,
}: {
  model: IsingModel;
  example: ModulationExample;
}) {
  const rows = useMemo(() => {
    const partnerSite = model.sites.indexOf(example.partnerSpecies);
    if (partnerSite < 0) return [];
    return example.items
      .map((it) => {
        const feature = model.indexOf.get(it.vocab);
        if (feature === undefined) return null;
        return { label: it.label, j: weightedSynergy(model, feature, partnerSite) };
      })
      .filter((r): r is { label: string; j: number } => r !== null);
  }, [model, example]);

  const maxAbs = useMemo(
    () => Math.max(0.1, ...rows.map((r) => Math.abs(r.j))),
    [rows],
  );

  return (
    <figure className="lab-modulation-card">
      <div className="lab-modulation-head">
        <div className="lab-modulation-pair">
          <SpriteBox name={example.modulatedSpecies} size={48} />
          <span className="lab-modulation-times">×</span>
          <SpriteBox name={example.partnerSpecies} size={48} />
        </div>
        <h4 className="lab-modulation-headline">{example.headline}</h4>
      </div>
      <div className="lab-modulation-bars">
        {rows.map((r) => {
          const frac = Math.abs(r.j) / maxAbs;
          const pos = r.j >= 0;
          return (
            <div key={r.label} className="lab-modulation-row">
              <span className="lab-modulation-label">{r.label}</span>
              <span className="lab-modulation-track">
                <span className="lab-modulation-axis" aria-hidden="true" />
                <span
                  className="lab-modulation-fill"
                  style={{
                    width: `${frac * 50}%`,
                    left: pos ? "50%" : undefined,
                    right: pos ? undefined : "50%",
                    background: pos ? POS_COLOR : NEG_COLOR,
                  }}
                />
              </span>
              <span
                className="lab-modulation-value"
                style={{ color: pos ? POS_COLOR : NEG_COLOR }}
              >
                {r.j >= 0 ? "+" : ""}
                {r.j.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
      <figcaption className="lab-modulation-body">{example.body}</figcaption>
    </figure>
  );
}
