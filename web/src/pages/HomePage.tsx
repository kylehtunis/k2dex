// Landing page — intro, corpus selector, tool index.
//
// The corpus selector here is the canonical place to pick a model on
// first visit; the header ModelPicker is hidden on this route (see
// Layout.tsx). After selecting, the choice is persisted via ModelContext
// (localStorage) so subsequent page visits remember it.

import { Link } from "react-router-dom";
import { useModel, type PhaseKey } from "../state/ModelContext";

const MODEL_OPTIONS: Array<{
  key: PhaseKey;
  label: string;
  tag: string;
  desc: string;
}> = [
  {
    key: "species",
    label: "Species",
    tag: "Pokémon only",
    desc: "Teams are represented only at the species level. Captures which Pokémon tend to appear together, independent of held items. Faster to load; good for team archetype analysis.",
  },
  {
    key: "species_item",
    label: "Species @ Item",
    tag: "With held items",
    desc: "Represents teams as Pokémon with their held items. Captures deeper relationships than the species-only model. This is the default.",
  },
];

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
    desc: "Detailed analysis of a full team of six, including enery, coupling, and suggested improvements.",
  },
  {
    path: "/meta/",
    num: "§03",
    label: "Meta Info",
    desc: "Format-wide bias rankings, strongest coupling pairs, and a species-level view of the current metagame.",
  },
];

export function HomePage() {
  const { phaseKey, setPhaseKey, model, status } = useModel();

  return (
    <div className="lab-home">
      {/* ── Hero ── */}
      <section className="lab-home-hero">
        <span className="lab-eyebrow">VGC Reg M-A</span>
        <h1 className="lab-home-display">
          VGC teambuilding<br />
          with statistical<br />
          physics.
        </h1>
        <p className="lab-home-lede">
          k2dex fits a pairwise maximum-entropy statistical physics model to competitive VGC tournament
          rosters, understanding which Pokémon attract and repel each other in
          the teambuilding process. Use it to complete rosters, diagnose team synergy,
          and explore format-wide patterns.
        </p>
      </section>

      {/* ── Corpus selector ── */}
      <section className="lab-home-section">
        <div className="lab-home-section-head">
          <span className="lab-section-num">§</span>
          <span className="lab-section-title">Choose a corpus</span>
        </div>
        <p className="lab-home-section-note">
          Both models are fit on the same Limitless tournament data.
          The <em>Species&nbsp;@&nbsp;Item</em> model is
          higher-resolution and the default.
        </p>
        <div className="lab-home-model-cards">
          {MODEL_OPTIONS.map((opt) => {
            const isActive = phaseKey === opt.key;
            return (
              <button
                key={opt.key}
                type="button"
                onClick={() => setPhaseKey(opt.key)}
                className={`lab-home-model-card${isActive ? " is-active" : ""}`}
                aria-pressed={isActive}
              >
                <div className="lab-home-model-card-header">
                  <span className="lab-home-model-card-label">{opt.label}</span>
                  <span className="lab-home-model-card-tag">{opt.tag}</span>
                </div>
                <p className="lab-home-model-card-desc">{opt.desc}</p>
                <div className="lab-home-model-card-footer">
                  {isActive && status === "ready" && model ? (
                    <span className="lab-home-model-stats">
                      {model.V.toLocaleString()} features
                      <span className="lab-home-model-sep">·</span>
                      {model.nCorpusTeams.toLocaleString()} teams
                    </span>
                  ) : isActive && status === "loading" ? (
                    <span className="lab-home-model-loading">Loading…</span>
                  ) : (
                    <span className="lab-home-model-select-hint">Click to select</span>
                  )}
                  {isActive && (
                    <span className="lab-home-model-active-mark">✓ active</span>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </section>

      {/* ── Tools ── */}
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

      {/* ── Science ── */}
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
                An interactive explainer — from ferromagnets and lattice models
                to the inverse Ising problem, illustrated with Pokémon and
                the Supreme Court.
              </p>
            </div>
            <span className="lab-home-science-arrow">→</span>
          </div>
        </Link>
      </section>
    </div>
  );
}
