import { useEffect, useState } from "react";

/** False during build-time prerendering (effects never run in renderToString)
 * and the very first client frame; true from then on. Used by the heavy
 * simulation SVGs to emit an empty, correctly-sized placeholder in the static
 * HTML — thousands of animation <rect>s are dead weight for a crawler — while
 * rendering normally in the browser with no layout shift. */
export function useIsClient(): boolean {
  const [isClient, setIsClient] = useState(false);
  useEffect(() => setIsClient(true), []);
  return isClient;
}
