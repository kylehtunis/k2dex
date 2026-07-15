// Species detail: a species-level mini-meta page, shared by two surfaces:
//   - FeatureModal.tsx renders it inside the modal/dock (drill-through stack)
//   - pages/PokemonPage.tsx renders it as the body of /pokemon/<slug>
// Shows species-level stats (weighted-average bias, total usage, corpus count),
// top synergy/antisynergy couplings with expandable item-modulation rows (same
// interaction pattern as /meta §02), corpus appearances with an optional item
// filter, and route-gated quick actions (completer: pin/exclude; analysis:
// add). When `partnerHref` is given (the page), partner species names render
// as real links so the species pages cross-link crawlably.

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import Select, { type SingleValue } from "react-select";
import { MENU_PORTAL_TARGET } from "./portalTarget";
import { topModulationEntries } from "../meta/couplings";
import { useModel } from "../state/ModelContext";
import { usePageState } from "../state/PageStateContext";
import { SpriteBox } from "../render/Sprite";
import { TeamMiniStrip } from "../render/cells";
import { ScoreChip, SignedBar, StatStrip } from "../render/atoms";
import { extractItem, formatPct, formatSigned } from "../render/format";
import {
  featureCorpusAppearances,
  siteCorpusAppearances,
  speciesCouplings,
} from "../render/featureDetail";
import { encodeCore } from "../render/shareLink";
import { TEAM_SIZE } from "../constants";
import type { IsingModel } from "../sampler/types";

export interface SpeciesDetailProps {
  model: IsingModel;
  site: number;
  /** Navigate to a partner species (modal: push the stack; page: route). */
  onDrillSite: (site: number) => void;
  /** Called when the user follows a link that leaves this surface (Analyze,
   * pin/exclude actions). The modal closes itself here; the page omits it. */
  onLeave?: () => void;
  /** Extra chrome rendered at the top of the header (the modal's back/close
   * button row). */
  headExtra?: ReactNode;
  /** id for the species heading, for the modal's aria-labelledby. */
  titleId?: string;
  /** Rendered inline next to the species name (the modal's open-full-page
   * link). */
  titleExtra?: ReactNode;
  /** Heading element for the species name: h2 in the modal, h1 on the page. */
  headingLevel?: "h1" | "h2";
  /** When set (the page), partner species names in the coupling lists render
   * as links to this href, so crawlers can walk the species graph. */
  partnerHref?: (species: string) => string;
}

export function SpeciesDetail({
  model,
  site,
  onDrillSite,
  onLeave,
  headExtra,
  titleId,
  titleExtra,
  headingLevel = "h2",
  partnerHref,
}: SpeciesDetailProps) {
  const { teamCounts, speciesGraph } = useModel();

  const species = model.sites[site];
  const Heading = headingLevel;

  // Item filter for corpus section only.
  const [corpusFeatureIdx, setCorpusFeatureIdx] = useState<number | null>(null);
  useEffect(() => setCorpusFeatureIdx(null), [site]);

  // Species-level couplings (always site-level, no item shift).
  const couplings = useMemo(
    () => speciesGraph ? speciesCouplings(model, speciesGraph, site, 10) : null,
    [model, speciesGraph, site],
  );

  // Corpus: scoped by corpusFeatureIdx when an item is selected.
  const corpus = useMemo(
    () =>
      corpusFeatureIdx !== null
        ? featureCorpusAppearances(model, teamCounts, corpusFeatureIdx, 10)
        : siteCorpusAppearances(model, teamCounts, site, 10),
    [model, teamCounts, corpusFeatureIdx, site],
  );

  // Species-level aggregates.
  const siteAggregates = useMemo(() => {
    const feats = model.siteFeatures[site];
    let totalM = 0;
    let weightedH = 0;
    for (const f of feats) {
      totalM += model.m[f];
      weightedH += model.m[f] * model.h[f];
    }
    const avgH = totalM > 0 ? weightedH / totalM : 0;
    return { totalM, avgH, nFeatures: feats.length };
  }, [model, site]);

  const maxSynergy = useMemo(() => {
    if (!couplings) return 1;
    let m = 0;
    for (const c of couplings.synergies) m = Math.max(m, Math.abs(c.synergy));
    for (const c of couplings.antisynergies) m = Math.max(m, Math.abs(c.synergy));
    return m || 1;
  }, [couplings]);

  return (
    <>
      <header className="lab-feature-modal-head">
        {headExtra}
        <div className="lab-feature-modal-identity">
          <SpriteBox name={species} size={64} />
          <div className="lab-feature-modal-title-wrap">
            <Heading id={titleId} className="lab-feature-modal-title">
              {species}
            </Heading>
            {titleExtra}
          </div>
        </div>
        <StatStrip
          cells={[
            {
              label: "Avg. Bias",
              value: formatSigned(siteAggregates.avgH),
              sub: `${siteAggregates.nFeatures} variants`,
            },
            {
              label: "Total usage",
              value: formatPct(siteAggregates.totalM),
            },
            {
              label: "In corpus",
              value: corpus.totalAppearances.toLocaleString(),
              sub: `${corpus.nTeams.toLocaleString()} rosters`,
            },
          ]}
        />
      </header>

      <div className="lab-feature-modal-body">
        <SpeciesActions model={model} site={site} species={species} onLeave={onLeave} />

        {couplings && (
        <div className="lab-feature-modal-couplings">
          <ExpandableCouplingList
            title="Top synergies"
            cls="lab-subheading-pos"
            rows={couplings.synergies}
            maxSynergy={maxSynergy}
            model={model}
            site={site}
            onDrillSite={onDrillSite}
            partnerHref={partnerHref}
            empty="No positive couplings."
          />
          <ExpandableCouplingList
            title="Top antisynergies"
            cls="lab-subheading-neg"
            rows={couplings.antisynergies}
            maxSynergy={maxSynergy}
            model={model}
            site={site}
            onDrillSite={onDrillSite}
            partnerHref={partnerHref}
            empty="No negative couplings."
          />
        </div>
        )}

        <div className="lab-feature-modal-section">
          <div className="lab-subheading">
            Top corpus appearances
          </div>
          <CorpusItemSelector
            model={model}
            site={site}
            featureIdx={corpusFeatureIdx}
            onChange={setCorpusFeatureIdx}
          />
          {corpus.teams.length === 0 ? (
            <p className="lab-feature-modal-empty">
              Not seen in any observed roster.
            </p>
          ) : (
            <ul className="lab-feature-corpus-list">
              {corpus.teams.map((t, i) => (
                <li key={i} className="lab-feature-corpus-row">
                  <TeamMiniStrip
                    names={t.team.map((x) => model.vocab[x])}
                    size={32}
                    interactive
                  />
                  <Link
                    to={`/analysis?t=${encodeCore(model.id, t.team, model)}`}
                    className="lab-analyze-btn lab-feature-corpus-analyze"
                    title="Send to Analysis"
                    onClick={onLeave}
                  >
                    Analyze
                  </Link>
                  <span className="lab-feature-corpus-meta">
                    <ScoreChip value={t.count} fmt="count" />
                    <span className="lab-feature-corpus-share">
                      {model.nCorpusTeams > 0
                        ? `${((t.count / model.nCorpusTeams) * 100).toFixed(2)}%`
                        : ""}
                    </span>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </>
  );
}

// ----- Expandable coupling rows (same pattern as /meta §02) -----

function ExpandableCouplingList({
  title,
  cls,
  rows,
  maxSynergy,
  model,
  site: mySite,
  onDrillSite,
  partnerHref,
  empty,
}: {
  title: string;
  cls: string;
  rows: readonly { species: string; synergy: number }[];
  maxSynergy: number;
  model: IsingModel;
  site: number;
  onDrillSite: (site: number) => void;
  partnerHref?: (species: string) => string;
  empty: string;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <div className="lab-feature-modal-section">
      <div className={`lab-subheading ${cls}`}>{title}</div>
      {rows.length === 0 ? (
        <p className="lab-feature-modal-empty">{empty}</p>
      ) : (
        <ul className="lab-feature-coupling-list">
          {rows.map((c) => {
            const partnerSite = model.sites.indexOf(c.species);
            const isOpen = expanded === c.species;
            // Clicking the row header toggles the expander. Drilling into the
            // partner species happens by clicking one of its modulation rows
            // (or, on the page, the linked partner name).
            const activate = () => setExpanded(isOpen ? null : c.species);
            const nameCell = (
              <>
                <SpriteBox name={c.species} size={32} />
                <span className="lab-comp-mon-name">{c.species}</span>
              </>
            );
            return (
              <li key={c.species} className="lab-feature-coupling-row-wrap">
                <div
                  className={`lab-feature-coupling-row${isOpen ? " lab-expanded" : ""}`}
                  role="button"
                  tabIndex={0}
                  style={{ cursor: "pointer" }}
                  onClick={activate}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      activate();
                    }
                  }}
                >
                  {partnerHref ? (
                    <Link
                      to={partnerHref(c.species)}
                      className="lab-feature-coupling-name"
                      onClick={(e) => e.stopPropagation()}
                    >
                      {nameCell}
                    </Link>
                  ) : (
                    <span className="lab-feature-coupling-name">{nameCell}</span>
                  )}
                  <span className="lab-feature-coupling-val">
                    <SignedBar value={c.synergy} maxValue={maxSynergy} width={70} />
                    <ScoreChip value={c.synergy} />
                  </span>
                </div>
                {isOpen && partnerSite >= 0 && (
                  <ModulationDetail
                    model={model}
                    siteA={mySite}
                    siteB={partnerSite}
                    synergy={c.synergy}
                    onDrillSite={onDrillSite}
                  />
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function ModulationDetail({
  model,
  siteA,
  siteB,
  synergy,
  onDrillSite,
}: {
  model: IsingModel;
  siteA: number;
  siteB: number;
  synergy: number;
  onDrillSite: (site: number) => void;
}) {
  const entries = useMemo(
    () => topModulationEntries(model, siteA, siteB, synergy),
    [model, siteA, siteB, synergy],
  );

  const maxJ = useMemo(() => {
    let m = 0;
    for (const e of entries) {
      const a = Math.abs(e.jValue);
      if (a > m) m = a;
    }
    return m || 1;
  }, [entries]);

  if (entries.length === 0) return null;

  return (
    <div className="lab-modulation-list">
      <table className="lab-modulation-table">
        <tbody>
          {entries.map((e) => (
            <tr
              key={`${e.featureA}-${e.featureB}`}
              className="lab-modulation-row"
              style={{ cursor: "pointer" }}
              onClick={() => onDrillSite(siteB)}
            >
              <td className="pair">
                <div className="lab-mod-pair">
                  <ModItem name={model.vocab[e.featureA]} />
                  <span className="lab-pair-sep">&times;</span>
                  <ModItem name={model.vocab[e.featureB]} />
                </div>
              </td>
              <td className="num">
                <div className="lab-coupling-val">
                  <SignedBar value={e.jValue} maxValue={maxJ} width={50} />
                  <ScoreChip value={e.jValue} />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Compact item cell for the modulation table: sprite + item label only. The
// two species are fixed for the whole table (the detail's species and the
// coupling partner), so repeating their names on every row is redundant.
function ModItem({ name }: { name: string }) {
  const item = extractItem(name);
  return (
    <span className="lab-mod-item">
      <SpriteBox name={name} size={22} />
      <span className="lab-mod-item-label">{item ?? "—"}</span>
    </span>
  );
}

// ----- Corpus item filter -----

interface AttrOpt {
  label: string;
  value: string;
}
const attrPortalStyles = {
  menuPortal: (base: Record<string, unknown>) => ({ ...base, zIndex: 9999 }),
};
const ALL_ITEMS_VALUE = "__all__";

function CorpusItemSelector({
  model,
  site,
  featureIdx,
  onChange,
}: {
  model: IsingModel;
  site: number;
  featureIdx: number | null;
  onChange: (featureIdx: number | null) => void;
}) {
  if (model.tracks.length === 0) return null;
  const track = model.tracks[0];
  const seen = new Set<string>();
  const rows: { opt: AttrOpt; m: number }[] = [];
  for (const f of model.siteFeatures[site]) {
    const val = model.trackValues[f][0];
    const key = val ?? " ";
    if (seen.has(key)) continue;
    seen.add(key);
    rows.push({ opt: { label: val ?? "—", value: String(f) }, m: model.m[f] });
  }
  rows.sort((a, b) => b.m - a.m);
  const options: AttrOpt[] = [
    { label: `All ${track.name}s`, value: ALL_ITEMS_VALUE },
    ...rows.map((r) => r.opt),
  ];
  const current = featureIdx !== null
    ? options.find((o) => o.value === String(featureIdx)) ?? options[0]
    : options[0];
  return (
    <div className="lab-feature-corpus-filter">
      <Select
        classNamePrefix="lab-select"
        className="lab-feature-attr-select"
        options={options}
        value={current}
        onChange={(o: SingleValue<AttrOpt>) => {
          if (!o) return;
          onChange(o.value === ALL_ITEMS_VALUE ? null : Number(o.value));
        }}
        isSearchable={options.length > 8}
        menuPortalTarget={MENU_PORTAL_TARGET}
        styles={attrPortalStyles}
        aria-label={`Filter by ${track.name}`}
      />
    </div>
  );
}

// ----- Route-gated actions (species-level) -----

function SpeciesActions({
  model,
  site,
  species,
  onLeave,
}: {
  model: IsingModel;
  site: number;
  species: string;
  onLeave?: () => void;
}) {
  const { pathname } = useLocation();
  const { completer, setCompleter, analysis, setAnalysis } = usePageState();

  if (pathname.includes("/completer")) {
    const speciesInRoster = completer.roster.some((s) => s.site === site);
    const rosterFull = completer.roster.length >= TEAM_SIZE;
    const excluded = completer.excludedSpecies.includes(species);
    const pinSite = () => {
      setCompleter({
        roster: [...completer.roster, { site, feature: null }],
        excludedSpecies: completer.excludedSpecies.filter((s) => s !== species),
      });
      onLeave?.();
    };
    const exclude = () => {
      setCompleter({
        excludedSpecies: [...completer.excludedSpecies, species],
        roster: completer.roster.filter((s) => model.sites[s.site] !== species),
      });
      onLeave?.();
    };
    return (
      <div className="lab-feature-modal-actions">
        <button
          type="button"
          className="lab-button-primary lab-feature-action"
          onClick={pinSite}
          disabled={speciesInRoster || rosterFull}
        >
          {speciesInRoster ? "On roster" : "Pin to roster"}
        </button>
        <button
          type="button"
          className="lab-analyze-btn lab-feature-action"
          onClick={exclude}
          disabled={excluded}
        >
          {excluded ? "Excluded" : "Exclude Pokémon"}
        </button>
      </div>
    );
  }

  if (pathname.includes("/analysis")) {
    const speciesOnTeam = analysis.roster.some((s) => s.site === site);
    const rosterFull = analysis.roster.length >= TEAM_SIZE;
    if (speciesOnTeam) {
      return (
        <div className="lab-feature-modal-actions">
          <span className="lab-feature-modal-note">Already on your team.</span>
        </div>
      );
    }
    if (!rosterFull) {
      return (
        <div className="lab-feature-modal-actions">
          <button
            type="button"
            className="lab-button-primary lab-feature-action"
            onClick={() => {
              setAnalysis({
                roster: [
                  ...analysis.roster,
                  { site, feature: null },
                ],
              });
              onLeave?.();
            }}
          >
            Add to team
          </button>
        </div>
      );
    }
    return null;
  }

  return null;
}
