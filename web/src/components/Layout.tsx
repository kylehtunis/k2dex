// App shell: header (wordmark + model picker) + page tabs + routed
// page content + status banner if the active model failed to load.
//
// Nav has two groups: left-aligned teambuilding tabs (Team Completer / Team Analysis /
// Metagame Model) and a right-aligned Articles tab to signal it's a separate surface.
// Model picker is hidden on /articles only (those pages carry their own).

import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { useModel } from "../state/ModelContext";
import { usePageMeta } from "../usePageMeta";
import { AnnouncementBanner } from "./AnnouncementBanner";
import { ModelPicker } from "./ModelPicker";

const PRIMARY_TABS = [
  { path: "/completer/", label: "Team Completer" },
  { path: "/analysis/", label: "Team Analysis" },
  { path: "/meta/", label: "Metagame Model" },
];
const ARTICLES_TAB = { path: "/articles/", label: "Articles" };

export function Layout() {
  usePageMeta();
  const { status, error } = useModel();
  const location = useLocation();
  const isArticles = location.pathname.startsWith("/articles");
  return (
    <div className="lab-container">
      <AnnouncementBanner />
      <header className="lab-header">
        <Link to="/" className="lab-wordmark lab-wordmark-link">
          k2dex
        </Link>
        {!isArticles && <ModelPicker />}
      </header>
      <nav className="lab-tabs lab-tabs-split">
        <div className="lab-tabs-group">
          {PRIMARY_TABS.map((t) => (
            <NavLink
              key={t.path}
              to={t.path}
              className={({ isActive }) => `lab-tab${isActive ? " active" : ""}`}
            >
              {t.label}
            </NavLink>
          ))}
        </div>
        <div className="lab-tabs-group lab-tabs-group-end">
          <NavLink
            to={ARTICLES_TAB.path}
            className={({ isActive }) => `lab-tab${isActive ? " active" : ""}`}
          >
            {ARTICLES_TAB.label}
          </NavLink>
        </div>
      </nav>
      {status === "error" && !isArticles && (
        <div className="lab-form-error" style={{ marginBottom: 16 }}>
          Failed to load model: {error?.message ?? "unknown error"}
        </div>
      )}
      <Outlet />
      <footer className="lab-footer">
        <span>
          k2dex by{" "}
          <a href="https://github.com/kylehtunis" target="_blank" rel="noopener noreferrer">
            Kyle Tunis
          </a>
        </span>
        <span className="lab-footer-sep">·</span>
        <a href="https://discord.gg/8xNjyn9yVP" target="_blank" rel="noopener noreferrer">
          Discord
        </a>
        <span className="lab-footer-sep">·</span>
        <a href="https://github.com/kylehtunis/k2dex/" target="_blank" rel="noopener noreferrer">
          Source Code
        </a>
      </footer>
    </div>
  );
}
