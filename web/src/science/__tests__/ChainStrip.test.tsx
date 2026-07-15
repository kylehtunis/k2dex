import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ChainStrip } from "../widgets/ChainStrip";

// The widget SSR-skips its cells (useIsClient gate); this test checks the
// full client markup, so force the client path.
vi.mock("../widgets/useIsClient", () => ({ useIsClient: () => true }));

describe("ChainStrip", () => {
  it("renders one group per ladder rung", () => {
    const history = [
      [0, 1, 2, 3],
      [1, 0, 2, 3],
      [1, 0, 3, 2],
    ];
    const html = renderToStaticMarkup(
      <ChainStrip history={history} K={4} width={300} height={120} />,
    );
    expect((html.match(/class="lab-chainstrip-rung"/g) || []).length).toBe(4);
  });
});
