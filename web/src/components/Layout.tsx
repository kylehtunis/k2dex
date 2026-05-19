// App shell: header (wordmark + model picker) + page tabs + routed
// page content + status banner if the active model failed to load.

import { NavLink, Outlet } from "react-router-dom";
import { useModel } from "../state/ModelContext";
import { ModelPicker } from "./ModelPicker";

const TABS = [
  { path: "/completer", label: "Completer" },
  { path: "/analysis", label: "Analysis" },
  { path: "/meta", label: "Meta" },
];

export function Layout() {
  const { status, error } = useModel();
  return (
    <div className="lab-container">
      <header className="lab-header">
        <div className="lab-wordmark">
          k2dex<span className="lab-wordmark-mono">·science</span>
        </div>
        <ModelPicker />
      </header>
      <nav className="lab-tabs">
        {TABS.map((t) => (
          <NavLink
            key={t.path}
            to={t.path}
            className={({ isActive }) => `lab-tab${isActive ? " active" : ""}`}
          >
            {t.label}
          </NavLink>
        ))}
      </nav>
      {status === "error" && (
        <div className="lab-form-error" style={{ marginBottom: 16 }}>
          Failed to load model: {error?.message ?? "unknown error"}
        </div>
      )}
      <Outlet />
    </div>
  );
}
