import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { SpinGrid } from "../widgets/SpinGrid";

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
