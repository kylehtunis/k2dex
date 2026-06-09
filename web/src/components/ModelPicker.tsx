// Model selector dropdown. Collapsed: shows current model name + chevron.
// Expanded: groups models by regulation, showing metadata for each.
// Selecting a different model persists the choice and hard-reloads.

import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { useModel } from "../state/ModelContext";
import type { ModelSummary } from "../state/manifest";

function formatDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function dimLabel(fd: number): string {
  return fd === 1 ? "species" : "species + item";
}

export function ModelPicker() {
  const { modelId, setModelId, manifest } = useModel();
  const { pathname } = useLocation();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [open]);

  if (!manifest) return null;

  const current = manifest.models.find((m) => m.id === modelId);

  const grouped = new Map<string, ModelSummary[]>();
  for (const m of manifest.models) {
    const reg = m.regulation || "Other";
    if (!grouped.has(reg)) grouped.set(reg, []);
    grouped.get(reg)!.push(m);
  }

  return (
    <div className="lab-model-picker" ref={ref}>
      <button
        type="button"
        className="lab-model-picker-toggle"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        <span className="lab-model-picker-label">
          {current?.displayName ?? modelId}
        </span>
        <span className={`lab-model-picker-chevron${open ? " open" : ""}`}>
          &#9662;
        </span>
      </button>

      {open && (
        <div className="lab-model-picker-panel" role="listbox">
          {[...grouped.entries()].map(([reg, models]) => (
            <div key={reg} className="lab-model-picker-group">
              <div className="lab-model-picker-group-label">Reg {reg}</div>
              {models.map((m) => {
                const isActive = m.id === modelId;
                return (
                  <button
                    key={m.id}
                    type="button"
                    role="option"
                    aria-selected={isActive}
                    className={`lab-model-picker-option${isActive ? " active" : ""}`}
                    onClick={() => {
                      if (isActive) {
                        setOpen(false);
                        return;
                      }
                      setModelId(m.id);
                      setOpen(false);
                      const base = import.meta.env.BASE_URL.replace(/\/$/, "");
                      window.location.replace(base + pathname);
                    }}
                  >
                    <span className="lab-model-picker-option-name">
                      {m.displayName}
                    </span>
                    <span className="lab-model-picker-option-meta">
                      {m.V.toLocaleString()} {dimLabel(m.featureDimensions)}
                      {" · "}
                      {m.nCorpusTeams.toLocaleString()} teams
                      {m.latestTournamentDate && (
                        <> · {formatDate(m.latestTournamentDate)}</>
                      )}
                    </span>
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
