// Horizontal-scroll wrapper with CSS-only scroll-shadow affordance.
// Use around tables that may overflow on narrow viewports.

import type { ReactNode } from "react";

export function ScrollX({ children }: { children: ReactNode }) {
  return <div className="lab-scrollx">{children}</div>;
}
