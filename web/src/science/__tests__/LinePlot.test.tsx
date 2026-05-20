import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { LinePlot } from "../widgets/LinePlot";

describe("LinePlot", () => {
  it("renders an SVG with a path for one series", () => {
    const html = renderToStaticMarkup(
      <LinePlot
        width={400}
        height={200}
        series={[{ data: [0, 1, 0.5, 0.75], color: "#000" }]}
        yDomain={[0, 1]}
        xLabel="t"
        yLabel="m"
      />,
    );
    expect(html).toContain("<svg");
    expect(html).toContain("<path");
  });
});
