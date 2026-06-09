// Landing page — intro, model selector, tool index.
//
// The model selector here is the canonical place to pick a model on
// first visit; the header ModelPicker is hidden on this route (see
// Layout.tsx). After selecting, the choice is persisted via ModelContext
// (localStorage) so subsequent page visits remember it.

import { useState } from "react";
import { Link } from "react-router-dom";
import { CURRENT_REGULATION } from "../constants";
import { useModel } from "../state/ModelContext";
import type { ModelSummary } from "../state/manifest";

function formatDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}

function dimTag(fd: number): string {
  return fd === 1 ? "Species only" : "With held items";
}

function ModelCard({
  m,
  isActive,
  status,
  onSelect,
}: {
  m: ModelSummary;
  isActive: boolean;
  status: string;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`lab-home-model-card${isActive ? " is-active" : ""}`}
      aria-pressed={isActive}
    >
      <div className="lab-home-model-card-header">
        <span className="lab-home-model-card-label">{m.displayName}</span>
        <span className="lab-home-model-card-tag">{dimTag(m.featureDimensions)}</span>
      </div>
      <div className="lab-home-model-card-footer">
        <span className="lab-home-model-stats">
          {m.V.toLocaleString()} features
          <span className="lab-home-model-sep">·</span>
          {m.nCorpusTeams.toLocaleString()} teams
          {m.latestTournamentDate && (
            <>
              <span className="lab-home-model-sep">·</span>
              {formatDate(m.latestTournamentDate)}
            </>
          )}
        </span>
        {isActive && (
          <span className="lab-home-model-active-mark">
            {status === "loading" ? "Loading…" : "✓ active"}
          </span>
        )}
      </div>
    </button>
  );
}

const TOOLS = [
  {
    path: "/completer/",
    num: "§01",
    label: "Team Completer",
    desc: "Give a partial roster and let the model suggest optimal completions. Supports fast greedy fill and full parallel-tempered sampling.",
  },
  {
    path: "/analysis/",
    num: "§02",
    label: "Team Analysis",
    desc: "Detailed analysis of a full team of six, including energy, coupling, and suggested improvements.",
  },
  {
    path: "/meta/",
    num: "§03",
    label: "Metagame Model",
    desc: "Format-wide bias rankings, strongest/weakest coupling pairs, and a species-level view of the current metagame.",
  },
];

export function HomePage() {
  const { modelId, setModelId, manifest, status } = useModel();
  const [legacyOpen, setLegacyOpen] = useState(false);

  const currentModels: ModelSummary[] = [];
  const legacyGrouped = new Map<string, ModelSummary[]>();
  if (manifest) {
    for (const m of manifest.models) {
      const reg = m.regulation || "Other";
      if (reg === CURRENT_REGULATION) {
        currentModels.push(m);
      } else {
        if (!legacyGrouped.has(reg)) legacyGrouped.set(reg, []);
        legacyGrouped.get(reg)!.push(m);
      }
    }
  }

  return (
    <div className="lab-home">
      {/* Hero */}
      <section className="lab-home-hero">
        <span className="lab-eyebrow">VGC Teambuilding</span>
        <h1 className="lab-home-display">
          VGC teambuilding<br />
          with statistical<br />
          physics.
        </h1>
        <p className="lab-home-lede">
          k2dex learns teambuilding patterns from thousands of real tournament
          teams by fitting a statistical physics Maximum Entropy model that captures which Pokémon
          attract and repel each other in the teambuilding process. Use it to complete rosters, diagnose
          team synergy, and explore format-wide patterns.
        </p>
      </section>

      {/* Model selector */}
      <section className="lab-home-section">
        <div className="lab-home-section-head">
          <span className="lab-section-num">§</span>
          <span className="lab-section-title">Choose a model</span>
        </div>
        <p className="lab-home-section-note">
          Each model is fit on real team rosters from recent
          VGC tournaments (64+ players).
        </p>

        {manifest && currentModels.length > 0 && (
          <div className="lab-home-model-group">
            <div className="lab-home-model-cards">
              {currentModels.map((m) => (
                <ModelCard key={m.id} m={m} isActive={m.id === modelId} status={status} onSelect={() => setModelId(m.id)} />
              ))}
            </div>
          </div>
        )}

        {manifest && legacyGrouped.size > 0 && (
          <div className="lab-home-legacy">
            <button
              type="button"
              className="lab-home-legacy-toggle"
              onClick={() => setLegacyOpen((o) => !o)}
              aria-expanded={legacyOpen}
            >
              <span>Legacy regulations</span>
              <span className={`lab-home-legacy-chevron${legacyOpen ? " open" : ""}`}>
                &#9662;
              </span>
            </button>
            {legacyOpen && [...legacyGrouped.entries()].map(([reg, models]) => (
              <div key={reg} className="lab-home-model-group">
                <div className="lab-home-model-group-label">Reg {reg}</div>
                <div className="lab-home-model-cards">
                  {models.map((m) => (
                    <ModelCard key={m.id} m={m} isActive={m.id === modelId} status={status} onSelect={() => setModelId(m.id)} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Tools */}
      <section className="lab-home-section">
        <div className="lab-home-section-head">
          <span className="lab-section-num">§</span>
          <span className="lab-section-title">Tools</span>
        </div>
        <div className="lab-home-tool-grid">
          {TOOLS.map((t) => (
            <Link key={t.path} to={t.path} className="lab-home-tool-card">
              <span className="lab-home-tool-num">{t.num}</span>
              <span className="lab-home-tool-label">{t.label}</span>
              <p className="lab-home-tool-desc">{t.desc}</p>
              <span className="lab-home-tool-arrow">→</span>
            </Link>
          ))}
        </div>
      </section>

      {/* Science */}
      <section className="lab-home-section lab-home-section-last">
        <div className="lab-home-section-head">
          <span className="lab-section-num">§</span>
          <span className="lab-section-title">The science</span>
        </div>
        <Link to="/science/" className="lab-home-science-card">
          <div className="lab-home-science-inner">
            <div>
              <div className="lab-home-science-label">The Science of k2dex</div>
              <p className="lab-home-science-desc">
                An interactive explainer that will take you from ferromagnets and lattice models
                to the inverse Ising problem, explaining exactly how k2dex works under the hood.
              </p>
            </div>
            <span className="lab-home-science-arrow">→</span>
          </div>
        </Link>
      </section>
    </div>
  );
}
export default HomePage;
