// Feature detail modal: opens from any feature name/sprite across the app and
// shows the model's view of that single feature — Bias / usage / meta ranks,
// its strongest structural couplings (drill-through to a coupled feature, with
// Back), the most-played corpus rosters that ran it, and context-aware quick
// actions keyed off the active route.
//
// The provider is mounted once near the app root (App.tsx). The context/hook
// lives in FeatureModalContext.ts to avoid a cycle with render/cells.tsx.

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useLocation } from "react-router-dom";
import { Modal } from "./Modal";
import { FeatureModalContext } from "./FeatureModalContext";
import { useMediaQuery } from "./useMediaQuery";
import { useModel } from "../state/ModelContext";
import { usePageState } from "../state/PageStateContext";
import { SpriteBox } from "../render/Sprite";
import { InlineMon, TeamMiniStrip } from "../render/cells";
import { ScoreChip, SignedBar, StatStrip } from "../render/atoms";
import { extractItem, extractSpecies, formatPct, formatSigned } from "../render/format";
import {
  featureCorpusAppearances,
  featureCouplings,
  featureRanks,
  type FeatureCoupling,
} from "../render/featureDetail";
import { TEAM_SIZE } from "../constants";
import type { IsingModel } from "../sampler/types";

const TITLE_ID = "feature-modal-title";
const DOCK_QUERY = "(min-width: 1200px)";

export function FeatureModalProvider({ children }: { children: ReactNode }) {
  const { model } = useModel();
  // Navigation stack of vocab indices; last = currently shown. A fresh open
  // (page/cell click) resets to a single entry; in-panel drill-through pushes;
  // Back pops; close / model-switch clears.
  const [stack, setStack] = useState<number[]>([]);

  // Page/cell click → start a fresh inspection (no Back history). Used by the
  // outer context that the page's shared cells read.
  const openFeature = useCallback(
    (name: string) => {
      if (!model) return;
      const idx = model.indexOf.get(name);
      if (idx === undefined) return;
      setStack([idx]);
    },
    [model],
  );

  // Partner click *inside the panel* → drill, pushing onto Back history. The
  // panel re-provides this as its context value (see FeatureModalBody), so the
  // very same InlineMon pushes when rendered in-panel but resets on the page.
  const drillTo = useCallback(
    (name: string) => {
      if (!model) return;
      const idx = model.indexOf.get(name);
      if (idx === undefined) return;
      setStack((s) => (s.length > 0 && s[s.length - 1] === idx ? s : [...s, idx]));
    },
    [model],
  );

  const close = useCallback(() => setStack([]), []);
  const back = useCallback(() => setStack((s) => s.slice(0, -1)), []);

  // Indices are model-specific — drop any open feature when the model changes.
  useEffect(() => {
    setStack([]);
  }, [model?.id]);

  const value = useMemo(() => ({ openFeature }), [openFeature]);
  const currentIdx = stack.length > 0 ? stack[stack.length - 1] : null;

  return (
    <FeatureModalContext.Provider value={value}>
      {children}
      {currentIdx !== null && model && (
        <FeatureModalBody
          model={model}
          idx={currentIdx}
          canGoBack={stack.length > 1}
          onBack={back}
          onClose={close}
          onDrill={drillTo}
        />
      )}
    </FeatureModalContext.Provider>
  );
}

interface BodyProps {
  model: IsingModel;
  idx: number;
  canGoBack: boolean;
  onBack: () => void;
  onClose: () => void;
  /** Drill-through handler re-provided to in-panel partner links (pushes). */
  onDrill: (name: string) => void;
}

function FeatureModalBody({ model, idx, canGoBack, onBack, onClose, onDrill }: BodyProps) {
  const { teamCounts } = useModel();
  const { pathname } = useLocation();
  const { completer, setCompleter, analysis, setAnalysis } = usePageState();
  const docked = useMediaQuery(DOCK_QUERY);

  // In-panel context: partner InlineMons read this (push) instead of the outer
  // page context (fresh-select). Same component, location decides behavior.
  const drillValue = useMemo(() => ({ openFeature: onDrill }), [onDrill]);

  const name = model.vocab[idx];
  const species = extractSpecies(name);
  const item = extractItem(name);

  const couplings = useMemo(() => featureCouplings(model, idx, 10), [model, idx]);
  const corpus = useMemo(
    () => featureCorpusAppearances(model, teamCounts, idx, 10),
    [model, teamCounts, idx],
  );
  const ranks = useMemo(() => featureRanks(model, idx), [model, idx]);

  const maxJ = useMemo(() => {
    let m = 0;
    for (const c of couplings.synergies) m = Math.max(m, Math.abs(c.jValue));
    for (const c of couplings.antisynergies) m = Math.max(m, Math.abs(c.jValue));
    return m || 1;
  }, [couplings]);

  return (
    <Modal
      onClose={onClose}
      labelledBy={TITLE_ID}
      variant={docked ? "dock" : "modal"}
    >
      <FeatureModalContext.Provider value={drillValue}>
      <div className="lab-feature-modal">
        <header className="lab-feature-modal-head">
          <div className="lab-feature-modal-head-row">
            {canGoBack ? (
              <button type="button" className="lab-feature-modal-back" onClick={onBack}>
                ‹ Back
              </button>
            ) : (
              <span />
            )}
            <button
              type="button"
              className="lab-feature-modal-close"
              onClick={onClose}
              aria-label="Close"
            >
              ✕
            </button>
          </div>
          <div className="lab-feature-modal-identity">
            <SpriteBox name={name} size={64} />
            <div>
              <h2 id={TITLE_ID} className="lab-feature-modal-title">
                {species}
              </h2>
              {item && <div className="lab-feature-modal-item">@ {item}</div>}
            </div>
          </div>
          <StatStrip
            cells={[
              {
                label: "Bias",
                value: formatSigned(model.h[idx]),
                sub: `#${ranks.biasRank} of ${model.V}`,
              },
              {
                label: "Usage",
                value: formatPct(model.m[idx]),
                sub: `#${ranks.marginalRank} of ${model.V}`,
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
          <FeatureActions
            pathname={pathname}
            model={model}
            idx={idx}
            species={species}
            completer={completer}
            setCompleter={setCompleter}
            analysis={analysis}
            setAnalysis={setAnalysis}
            onClose={onClose}
          />

          <div className="lab-feature-modal-couplings">
            <CouplingList
              title="Top synergies"
              cls="lab-subheading-pos"
              rows={couplings.synergies}
              maxJ={maxJ}
              model={model}
              empty="No positive couplings."
            />
            <CouplingList
              title="Top antisynergies"
              cls="lab-subheading-neg"
              rows={couplings.antisynergies}
              maxJ={maxJ}
              model={model}
              empty="No negative couplings."
            />
          </div>

          <div className="lab-feature-modal-section">
            <div className="lab-subheading">Top corpus appearances</div>
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
      </div>
      </FeatureModalContext.Provider>
    </Modal>
  );
}

function CouplingList({
  title,
  cls,
  rows,
  maxJ,
  model,
  empty,
}: {
  title: string;
  cls: string;
  rows: readonly FeatureCoupling[];
  maxJ: number;
  model: IsingModel;
  empty: string;
}) {
  return (
    <div className="lab-feature-modal-section">
      <div className={`lab-subheading ${cls}`}>{title}</div>
      {rows.length === 0 ? (
        <p className="lab-feature-modal-empty">{empty}</p>
      ) : (
        <ul className="lab-feature-coupling-list">
          {rows.map((c) => (
            <li key={c.idx} className="lab-feature-coupling-row">
              <InlineMon name={model.vocab[c.idx]} size={32} />
              <span className="lab-feature-coupling-val">
                <SignedBar value={c.jValue} maxValue={maxJ} width={70} />
                <ScoreChip value={c.jValue} />
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

interface ActionsProps {
  pathname: string;
  model: IsingModel;
  idx: number;
  species: string;
  completer: ReturnType<typeof usePageState>["completer"];
  setCompleter: ReturnType<typeof usePageState>["setCompleter"];
  analysis: ReturnType<typeof usePageState>["analysis"];
  setAnalysis: ReturnType<typeof usePageState>["setAnalysis"];
  onClose: () => void;
}

/** Route-gated quick actions. Completer: pin / exclude. Analysis: add-to-team
 * or swap-in for a current member. Meta / science: nothing. */
function FeatureActions({
  pathname,
  model,
  idx,
  species,
  completer,
  setCompleter,
  analysis,
  setAnalysis,
  onClose,
}: ActionsProps) {
  if (pathname.includes("/completer")) {
    const site = model.siteOf[idx];
    const pinnedExact = completer.roster.some((s) => s.feature === idx);
    const speciesInRoster = completer.roster.some((s) => s.site === site);
    const rosterFull = completer.roster.length >= TEAM_SIZE;
    const excluded = completer.excludedSpecies.includes(species);
    const pin = () => {
      // Pin this exact build (species + item) as a roster slot.
      setCompleter({
        roster: [...completer.roster, { site, feature: idx }],
        excludedSpecies: completer.excludedSpecies.filter((s) => s !== species),
      });
      onClose();
    };
    const exclude = () => {
      setCompleter({
        excludedSpecies: [...completer.excludedSpecies, species],
        roster: completer.roster.filter((s) => model.sites[s.site] !== species),
      });
      onClose();
    };
    return (
      <div className="lab-feature-modal-actions">
        <button
          type="button"
          className="lab-button-primary lab-feature-action"
          onClick={pin}
          disabled={pinnedExact || rosterFull || speciesInRoster}
        >
          {pinnedExact ? "Pinned to roster" : "Pin to roster"}
        </button>
        <button
          type="button"
          className="lab-analyze-btn lab-feature-action"
          onClick={exclude}
          disabled={excluded}
        >
          {excluded ? "Excluded" : "Exclude species"}
        </button>
      </div>
    );
  }

  if (pathname.includes("/analysis")) {
    const team = analysis.teamIdxs;
    const onTeam = team.includes(idx);
    const teamFull = team.length >= TEAM_SIZE;
    if (onTeam) {
      return (
        <div className="lab-feature-modal-actions">
          <span className="lab-feature-modal-note">Already on your team.</span>
        </div>
      );
    }
    if (!teamFull) {
      return (
        <div className="lab-feature-modal-actions">
          <button
            type="button"
            className="lab-button-primary lab-feature-action"
            onClick={() => {
              setAnalysis({ teamIdxs: [...team, idx] });
              onClose();
            }}
          >
            Add to team
          </button>
        </div>
      );
    }
    // Full team: pick which member to swap out.
    return (
      <div className="lab-feature-modal-actions lab-feature-modal-swap">
        <span className="lab-feature-modal-note">Swap in for:</span>
        <div className="lab-feature-swap-targets">
          {team.map((memberIdx) => (
            <button
              key={memberIdx}
              type="button"
              className="lab-feature-swap-target"
              onClick={() => {
                setAnalysis({
                  teamIdxs: team.map((t) => (t === memberIdx ? idx : t)),
                });
                onClose();
              }}
            >
              <SpriteBox name={model.vocab[memberIdx]} size={32} />
              <span>{extractSpecies(model.vocab[memberIdx])}</span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  return null;
}
