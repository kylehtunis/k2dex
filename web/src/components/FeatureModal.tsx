// Feature detail modal: hosts the shared SpeciesDetail body (see
// SpeciesDetail.tsx) inside the centered modal / docked inspector.
//
// Opens from any feature name/sprite across the app and resolves to the
// species (site). The provider is mounted once near the app root (App.tsx);
// the context/hook lives in FeatureModalContext.ts to avoid a cycle with
// render/cells.tsx. The head row carries a link to the species' full page
// (/pokemon/<slug>), which shows the same content as a standalone route.

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { Link } from "react-router-dom";
import { Modal } from "./Modal";
import { FeatureModalContext } from "./FeatureModalContext";
import { useMediaQuery } from "./useMediaQuery";
import { useModel } from "../state/ModelContext";
import { SpeciesDetail } from "./SpeciesDetail";
import { speciesPageSlug } from "../siteMeta";
import type { IsingModel } from "../sampler/types";

const TITLE_ID = "feature-modal-title";
const DOCK_QUERY = "(min-width: 1200px)";

export function FeatureModalProvider({ children }: { children: ReactNode }) {
  const { model } = useModel();
  // Navigation stack of site indices. A fresh open (page/cell click) resets to
  // a single entry; in-panel drill-through pushes; Back pops; close clears.
  const [stack, setStack] = useState<number[]>([]);

  const openFeature = useCallback(
    (name: string) => {
      if (!model) return;
      const idx = model.indexOf.get(name);
      if (idx === undefined) return;
      setStack([model.siteOf[idx]]);
    },
    [model],
  );

  const drillToSite = useCallback(
    (site: number) => {
      setStack((s) => {
        const top = s.length > 0 ? s[s.length - 1] : -1;
        return top === site ? s : [...s, site];
      });
    },
    [],
  );

  const close = useCallback(() => setStack([]), []);
  const back = useCallback(() => setStack((s) => s.slice(0, -1)), []);

  useEffect(() => {
    setStack([]);
  }, [model?.id]);

  const openSite = useCallback((site: number) => setStack([site]), []);

  const value = useMemo(() => ({ openFeature, openSite }), [openFeature, openSite]);
  const currentSite = stack.length > 0 ? stack[stack.length - 1] : null;

  return (
    <FeatureModalContext.Provider value={value}>
      {children}
      {currentSite !== null && model && (
        <FeatureModalShell
          model={model}
          site={currentSite}
          canGoBack={stack.length > 1}
          onBack={back}
          onClose={close}
          onDrillSite={drillToSite}
        />
      )}
    </FeatureModalContext.Provider>
  );
}

interface ShellProps {
  model: IsingModel;
  site: number;
  canGoBack: boolean;
  onBack: () => void;
  onClose: () => void;
  onDrillSite: (site: number) => void;
}

function FeatureModalShell({
  model,
  site,
  canGoBack,
  onBack,
  onClose,
  onDrillSite,
}: ShellProps) {
  const docked = useMediaQuery(DOCK_QUERY);

  // Drill-through context: clicking a partner species in the coupling list
  // pushes onto the stack (same InlineMon, behavior decided by render location).
  const drillValue = useMemo(
    () => ({
      openFeature: (name: string) => {
        const idx = model.indexOf.get(name);
        if (idx !== undefined) onDrillSite(model.siteOf[idx]);
      },
      openSite: onDrillSite,
    }),
    [model, onDrillSite],
  );

  const species = model.sites[site];

  const headRow = (
    <div className="lab-feature-modal-head-row">
      {canGoBack ? (
        <button type="button" className="lab-feature-modal-back" onClick={onBack}>
          ‹ Back
        </button>
      ) : (
        <span />
      )}
      <button
        type="button"
        className="lab-feature-modal-close"
        onClick={onClose}
        aria-label="Close"
      >
        ✕
      </button>
    </div>
  );

  const openPageLink = (
    <Link
      to={`/pokemon/${speciesPageSlug(species)}/`}
      target="_blank"
      rel="noopener"
      className="lab-feature-modal-open"
      title={`Open the full ${species} page in a new tab`}
      aria-label={`Open the full ${species} page in a new tab`}
    >
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
        <polyline points="15 3 21 3 21 9" />
        <line x1="10" y1="14" x2="21" y2="3" />
      </svg>
    </Link>
  );

  return (
    <Modal
      onClose={onClose}
      labelledBy={TITLE_ID}
      variant={docked ? "dock" : "modal"}
    >
      <FeatureModalContext.Provider value={drillValue}>
        <div className="lab-feature-modal">
          <SpeciesDetail
            model={model}
            site={site}
            onDrillSite={onDrillSite}
            onLeave={onClose}
            headExtra={headRow}
            titleExtra={openPageLink}
            titleId={TITLE_ID}
            headingLevel="h2"
          />
        </div>
      </FeatureModalContext.Provider>
    </Modal>
  );
}
