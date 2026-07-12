// A miniature of the real team completer, built to run both methods head-to-
// head. Pick one to three Pokémon; "counting" fills the rest of the team by
// teammate co-occurrence (a species-level greedy, then each species' most-used
// item), while "k2dex" runs the actual parallel-tempered sampler and takes its
// single most-likely completion. Both finished teams are scored under the
// fitted model so the reader can compare them directly — and judge with their
// own eyes which one looks like the better team.

import { useMemo, useState } from "react";
import Select, { type SingleValue } from "react-select";
import type { IsingModel, TeamCounts } from "../../sampler/types";
import { SpriteBox } from "../../render/Sprite";
import { teamObservables } from "../../render/observables";
import { nearestObserved } from "../../render/corpus";
import { runPT } from "../../completer/ptDriver";
import {
  ANCHOR_ARTICLE_DEFAULT,
  ANCHOR_MAX,
  ANCHOR_MIN,
  ANCHOR_STEP,
  PT_HOT_T,
  PT_LADDER_LEVELS,
  PT_RUNS,
  PT_SWEEPS,
  PT_BURN_IN,
  PT_SWAP_INTERVAL,
} from "../../constants";
import {
  speciesCoocGreedy,
  topFeatureOfSite,
  type SpeciesCooccurrence,
} from "../cooccurrenceSpecies";

const EMPTY_SITES: ReadonlySet<number> = new Set();
const portalStyles = {
  menuPortal: (base: Record<string, unknown>) => ({ ...base, zIndex: 9999 }),
};

// The article's "k2dex team" IS the completer's output — same budget as
// /completer's default PT run (constants.ts), so the reader sees exactly what
// the real tool would give. A trimmed budget was tried and rejected: short
// chains mode-trap on rare mega clusters and return unrepresentative teams,
// which would misrepresent the model. The full run takes a few seconds; the
// button shows a spinner. `coldT = 1` / `fieldWeight = 1` is the calibrated
// operating point (see CLAUDE.md).
const PT_ARTICLE = {
  coldT: 1.0,
  hotT: PT_HOT_T,
  ladderLevels: PT_LADDER_LEVELS,
  nRuns: PT_RUNS,
  nSteps: PT_SWEEPS + PT_BURN_IN,
  burnIn: PT_BURN_IN,
  swapInterval: PT_SWAP_INTERVAL,
  seed: 0x5eed,
};

interface Opt {
  label: string;
  value: number;
}

interface Result {
  cooc: number[]; // feature indices
  model: number[];
}

export function MiniCompleter({
  model,
  sc,
  teamCounts,
}: {
  model: IsingModel;
  sc: SpeciesCooccurrence;
  teamCounts: TeamCounts | null;
}) {
  const options = useMemo<Opt[]>(() => {
    return Array.from({ length: sc.S }, (_, s) => s)
      .filter((s) => sc.siteAppear[s] > 0)
      .sort((a, b) => sc.siteAppear[b] - sc.siteAppear[a])
      .map((s) => ({ label: model.sites[s], value: s }));
  }, [model, sc]);

  const [slots, setSlots] = useState<(number | null)[]>([
    null,
    null,
    null,
    null,
    null,
  ]);
  // Anchor Strength (anchor-field tilt alpha): how strongly the sampler
  // commits to the picks. Defaults above neutral so a niche pick gets a team
  // built AROUND it — the slider is right here so the reader can see (and
  // undo) the thumb on the scale. Counting needs no equivalent: its greedy
  // re-anchors on the picks by construction.
  const [anchorStrength, setAnchorStrength] = useState(ANCHOR_ARTICLE_DEFAULT);
  const [result, setResult] = useState<Result | null>(null);
  const [running, setRunning] = useState(false);

  const picks = useMemo(
    () => slots.filter((s): s is number => s !== null),
    [slots],
  );

  const setSlot = (i: number, site: number | null) => {
    setSlots((prev) => prev.map((s, k) => (k === i ? site : s)));
    setResult(null);
  };

  const build = async () => {
    setRunning(true);
    setResult(null);
    // Counting: species-level greedy fill, then most-used item per species.
    const coocSites = speciesCoocGreedy(sc, picks, EMPTY_SITES, model.teamSize);
    const coocTeam = coocSites.map((s) => topFeatureOfSite(model, s));
    // Model: the real PT sampler, species pinned (item free to reroll in
    // context), take the single most-frequent completion.
    const r = await runPT(model, {
      fixed: [],
      fixedSites: picks,
      excluded: [],
      fieldWeight: 1,
      anchorStrength,
      ...PT_ARTICLE,
    });
    const modelTeam = r.ok && r.dist.length > 0 ? r.dist[0].team : [];
    setResult({ cooc: coocTeam, model: modelTeam });
    setRunning(false);
  };

  const usedInSlots = new Set(picks);

  return (
    <div className="lab-vs-widget">
      <div className="lab-vs-picks">
        <span className="lab-vs-picks-label">Start with:</span>
        {slots.map((slot, i) => (
          <Select<Opt>
            key={i}
            classNamePrefix="lab-select"
            className="lab-vs-pick"
            options={options.filter(
              (o) => !usedInSlots.has(o.value) || o.value === slot,
            )}
            value={slot !== null ? { label: model.sites[slot], value: slot } : null}
            onChange={(o: SingleValue<Opt>) => setSlot(i, o ? o.value : null)}
            isClearable
            placeholder={i === 0 ? "pick a Pokémon" : "add another"}
            menuPortalTarget={document.body}
            styles={portalStyles}
            aria-label={`Pick ${i + 1}`}
          />
        ))}
      </div>

      <div className="lab-vs-anchor">
        <label className="lab-form-label">
          Anchor Strength · {anchorStrength.toFixed(1)}
        </label>
        <div className="lab-form-caption">
          How strongly k2dex builds around your picks. 1.0 is the model&apos;s
          neutral distribution; we default higher so a niche pick gets a team
          built around it.
        </div>
        <input
          type="range"
          className="lab-slider"
          aria-label="Anchor Strength"
          min={ANCHOR_MIN}
          max={ANCHOR_MAX}
          step={ANCHOR_STEP}
          value={anchorStrength}
          onChange={(e) => {
            setAnchorStrength(Number(e.target.value));
            setResult(null);
          }}
        />
      </div>

      <div className="lab-vs-actions">
        <button
          type="button"
          className="lab-science-btn"
          onClick={build}
          disabled={picks.length === 0 || running}
        >
          {running ? "Building teams…" : "Build both teams →"}
        </button>
      </div>

      {result && (
        <div className="lab-comparison-columns lab-vs-teams">
          <TeamCard
            title="Counting's team"
            subtitle="teammate % + popular items"
            model={model}
            team={result.cooc}
            teamCounts={teamCounts}
          />
          <TeamCard
            title="k2dex's team"
            subtitle="parallel-tempered sampler"
            model={model}
            team={result.model}
            teamCounts={teamCounts}
          />
        </div>
      )}
    </div>
  );
}

function TeamCard({
  title,
  subtitle,
  model,
  team,
  teamCounts,
}: {
  title: string;
  subtitle: string;
  model: IsingModel;
  team: number[];
  teamCounts: TeamCounts | null;
}) {
  const obs = useMemo(
    () => (team.length > 0 ? teamObservables(model, team, 1) : null),
    [model, team],
  );
  const near = useMemo(
    () => (team.length > 0 ? nearestObserved(team, teamCounts) : null),
    [team, teamCounts],
  );

  return (
    <div className="lab-comparison-col">
      <h4 className="lab-comparison-col-head">
        {title}
        <span className="lab-comparison-col-sub">{subtitle}</span>
      </h4>
      <div className="lab-vs-team-grid">
        {team.map((f) => (
          <div key={f} className="lab-vs-team-mon">
            <SpriteBox name={model.vocab[f]} size={52} />
          </div>
        ))}
      </div>
      {obs && (
        <dl className="lab-vs-scorecard">
          <div>
            <dt>Score</dt>
            <dd>{obs.scoreRaw.toFixed(1)}</dd>
          </div>
          <div>
            <dt>Coherence</dt>
            <dd>{obs.coherence.toFixed(1)}</dd>
          </div>
          {near && (
            <div>
              <dt>Nearest real team</dt>
              <dd>
                {near.delta === 0
                  ? "exact match"
                  : `${near.delta} swap${near.delta === 1 ? "" : "s"} away`}
              </dd>
            </div>
          )}
        </dl>
      )}
    </div>
  );
}
