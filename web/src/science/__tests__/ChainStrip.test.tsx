import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ChainStrip } from "../widgets/ChainStrip";

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
