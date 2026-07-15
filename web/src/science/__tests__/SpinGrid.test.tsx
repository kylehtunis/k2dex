import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { SpinGrid } from "../widgets/SpinGrid";

// The widget SSR-skips its cells (useIsClient gate); this test checks the
// full client markup, so force the client path.
vi.mock("../widgets/useIsClient", () => ({ useIsClient: () => true }));

describe("SpinGrid", () => {
  it("renders one rect per spin", () => {
    const L: (-1 | 1)[][] = [
      [1, -1, 1],
      [-1, 1, -1],
    ];
    const html = renderToStaticMarkup(<SpinGrid lattice={L} cell={10} />);
    expect((html.match(/<rect /g) || []).length).toBe(6);
  });
});
