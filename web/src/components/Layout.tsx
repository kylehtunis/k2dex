// App shell: header (wordmark + model picker) + page tabs + routed
// page content + status banner if the active model failed to load.
//
// Nav has two groups: left-aligned teambuilding tabs (Completer / Analysis /
// Meta) and a right-aligned Science tab to signal it's a separate surface.
// Model picker is hidden on /science since that page is not model-selectable.

import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useModel } from "../state/ModelContext";
import { ModelPicker } from "./ModelPicker";

const PRIMARY_TABS = [
  { path: "/completer", label: "Completer" },
  { path: "/analysis", label: "Analysis" },
  { path: "/meta", label: "Meta" },
];
const SCIENCE_TAB = { path: "/science", label: "Science" };

export function Layout() {
  const { status, error } = useModel();
  const location = useLocation();
  const isScience = location.pathname.startsWith("/science");
  return (
    <div className="lab-container">
      <header className="lab-header">
        <div className="lab-wordmark">
          k2dex<span className="lab-wordmark-mono">·science</span>
        </div>
        {!isScience && <ModelPicker />}
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
            to={SCIENCE_TAB.path}
            className={({ isActive }) => `lab-tab${isActive ? " active" : ""}`}
          >
            {SCIENCE_TAB.label}
          </NavLink>
        </div>
      </nav>
      {status === "error" && !isScience && (
        <div className="lab-form-error" style={{ marginBottom: 16 }}>
          Failed to load model: {error?.message ?? "unknown error"}
        </div>
      )}
      <Outlet />
    </div>
  );
}
