// Thin wrapper around react-katex so the rest of /science doesn't need
// to know which library powers math rendering. Page-level KaTeX CSS is
// imported by SciencePage so other pages don't pay the cost.

import { InlineMath as KInline, BlockMath as KBlock } from "react-katex";

export function InlineMath({ formula }: { formula: string }) {
  return <KInline math={formula} />;
}

export function BlockMath({ formula }: { formula: string }) {
  return <KBlock math={formula} />;
}
