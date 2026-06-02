// Segmented two-option picker for the model phase. Top-right of the
// header in the app shell. Persists selection in localStorage via the
// ModelContext.

import { useLocation } from "react-router-dom";
import { useModel, type PhaseKey } from "../state/ModelContext";

const OPTIONS: Array<{ key: PhaseKey; label: string }> = [
  { key: "species_item", label: "Species @ Item" },
  { key: "species", label: "Species" },
];

export function ModelPicker() {
  const { phaseKey, setPhaseKey } = useModel();
  const { pathname } = useLocation();
  return (
    <div className="lab-segmented" role="radiogroup" aria-label="Model">
      {OPTIONS.map((o) => (
        <button
          key={o.key}
          type="button"
          role="radio"
          aria-checked={phaseKey === o.key}
          className={`lab-segmented-option${phaseKey === o.key ? " active" : ""}`}
          onClick={() => {
            if (o.key === phaseKey) return;
            setPhaseKey(o.key);
            const base = import.meta.env.BASE_URL.replace(/\/$/, "");
            window.location.replace(base + pathname);
          }}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
