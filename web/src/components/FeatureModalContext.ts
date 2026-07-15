// Context + hook for the feature detail modal, split out from FeatureModal.tsx
// so the shared render cells (render/cells.tsx) can consume the hook without a
// circular import (FeatureModal.tsx itself imports those cells).
//
// useFeatureModal() returns null when no provider is mounted, so cells stay
// inert in any context that doesn't wrap them.

import { createContext, useContext } from "react";

export interface FeatureModalValue {
  /** Open the detail modal for a feature by its vocab string (resolved to an
   * index internally). No-op for unknown strings. */
  openFeature: (name: string) => void;
  /** Open the detail modal directly at a site (species) index. Inside the
   * modal's own drill context this pushes onto the navigation stack. */
  openSite: (site: number) => void;
}

export const FeatureModalContext = createContext<FeatureModalValue | null>(null);

export function useFeatureModal(): FeatureModalValue | null {
  return useContext(FeatureModalContext);
}
