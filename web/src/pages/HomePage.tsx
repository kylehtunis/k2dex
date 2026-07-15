// Landing page — intro, tool index, articles.
//
// Model selection lives in the header ModelPicker here just like every other
// route (Layout.tsx): nearly everyone stays on the default current-regulation
// model, so the page no longer carries its own selector.

import { Link } from "react-router-dom";
import { ArticleList } from "../components/ArticleList";

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
    desc: "General information about how the k2dex model sees the current metagame, with the most used teams and strongest synergies and anti-synergies.",
  },
  {
    path: "/pokemon/",
    num: "§04",
    label: "Pokémon Index",
    desc: "Per-Pokémon breakdowns: usage, best teammates, item builds, and the strongest synergies and anti-synergies for every Pokémon in the format.",
  },
];

export function HomePage() {
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

      {/* Articles */}
      <section className="lab-home-section lab-home-section-last">
        <div className="lab-home-section-head">
          <span className="lab-section-num">§</span>
          <span className="lab-section-title">Articles</span>
        </div>
        <p className="lab-home-section-note">
          Write-ups on the ideas, findings, and implementation details behind k2dex.
        </p>
        <ArticleList />
      </section>
    </div>
  );
}
export default HomePage;
