// Fixed circular layout for the 9 justices of the Rehnquist court.
// Order matches scotus_precompute.py:JUSTICES.

export const JUSTICE_POSITIONS: { x: number; y: number }[] = (() => {
  const cx = 220;
  const cy = 200;
  const r = 140;
  return Array.from({ length: 9 }, (_, i) => {
    const theta = (i / 9) * Math.PI * 2 - Math.PI / 2;
    return { x: cx + r * Math.cos(theta), y: cy + r * Math.sin(theta) };
  });
})();
