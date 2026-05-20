import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { GraphView } from "../widgets/GraphView";

describe("GraphView", () => {
  it("renders nodes and edges for a tiny graph", () => {
    const html = renderToStaticMarkup(
      <GraphView
        nodes={[
          { id: 0, label: "A", x: 50, y: 50 },
          { id: 1, label: "B", x: 150, y: 50 },
        ]}
        edges={[{ i: 0, j: 1, weight: 1.0 }]}
        width={200}
        height={100}
      />,
    );
    expect((html.match(/<line /g) || []).length).toBe(1);
    expect((html.match(/<circle /g) || []).length).toBe(2);
    expect(html).toContain(">A<");
    expect(html).toContain(">B<");
  });
});
