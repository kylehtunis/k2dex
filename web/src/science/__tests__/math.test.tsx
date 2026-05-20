import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { InlineMath, BlockMath } from "../widgets/Math";

describe("Math wrappers", () => {
  it("renders inline math without crashing", () => {
    const html = renderToStaticMarkup(<InlineMath formula="x = y + 1" />);
    expect(html).toContain("katex");
  });
  it("renders block math without crashing", () => {
    const html = renderToStaticMarkup(<BlockMath formula="H = -\\tfrac{1}{2} s^T J s" />);
    expect(html).toContain("katex");
  });
});
