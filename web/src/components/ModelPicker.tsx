// Model selector dropdown. Collapsed: shows current model name + chevron.
// Expanded: groups models by regulation, showing metadata for each.
// Models from non-current regulations are collapsed under a "Legacy" toggle.
// Selecting a different model persists the choice and hard-reloads.

import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { CURRENT_REGULATION } from "../constants";
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

function ModelOption({
  m,
  isActive,
  onSelect,
}: {
  m: ModelSummary;
  isActive: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      key={m.id}
      type="button"
      role="option"
      aria-selected={isActive}
      className={`lab-model-picker-option${isActive ? " active" : ""}`}
      onClick={onSelect}
    >
      <span className="lab-model-picker-option-name">
        {m.displayName}
      </span>
      {m.description && (
        <span className="lab-model-picker-option-desc">
          {m.description}
        </span>
      )}
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
}

export function ModelPicker() {
  const { modelId, setModelId, manifest } = useModel();
  const { pathname } = useLocation();
  const [open, setOpen] = useState(false);
  const [legacyOpen, setLegacyOpen] = useState(false);
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

  const currentModels: ModelSummary[] = [];
  const legacyGrouped = new Map<string, ModelSummary[]>();
  for (const m of manifest.models) {
    const reg = m.regulation || "Other";
    if (reg === CURRENT_REGULATION) {
      currentModels.push(m);
    } else {
      if (!legacyGrouped.has(reg)) legacyGrouped.set(reg, []);
      legacyGrouped.get(reg)!.push(m);
    }
  }

  function handleSelect(m: ModelSummary) {
    if (m.id === modelId) {
      setOpen(false);
      return;
    }
    setModelId(m.id);
    setOpen(false);
    const base = import.meta.env.BASE_URL.replace(/\/$/, "");
    window.location.replace(base + pathname);
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
          {currentModels.length > 0 && (
            <div className="lab-model-picker-group">
              <div className="lab-model-picker-group-label">Reg {CURRENT_REGULATION}</div>
              {currentModels.map((m) => (
                <ModelOption
                  key={m.id}
                  m={m}
                  isActive={m.id === modelId}
                  onSelect={() => handleSelect(m)}
                />
              ))}
            </div>
          )}

          {legacyGrouped.size > 0 && (
            <div className="lab-model-picker-legacy">
              <button
                type="button"
                className="lab-model-picker-legacy-toggle"
                onClick={() => setLegacyOpen((o) => !o)}
                aria-expanded={legacyOpen}
              >
                <span>Legacy regulations</span>
                <span className={`lab-model-picker-chevron${legacyOpen ? " open" : ""}`}>
                  &#9662;
                </span>
              </button>
              {legacyOpen && [...legacyGrouped.entries()].map(([reg, models]) => (
                <div key={reg} className="lab-model-picker-group">
                  <div className="lab-model-picker-group-label">Reg {reg}</div>
                  {models.map((m) => (
                    <ModelOption
                      key={m.id}
                      m={m}
                      isActive={m.id === modelId}
                      onSelect={() => handleSelect(m)}
                    />
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
