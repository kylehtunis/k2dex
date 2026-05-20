// Mean field as gradient descent on the same energy landscape used in the
// PT section. A single deterministic marker rolls downhill to whichever well
// it starts nearest — no thermal escape, no swap-driven exploration.

import { useCallback, useEffect, useRef, useState } from "react";
import { BlockMath, InlineMath } from "../widgets/Math";
import { Landscape3D, type Walker } from "../widgets/Landscape3D";
import {
  gradAt,
  gradientStep,
  type Point,
} from "../primitives/landscape";

const STEPS_PER_FRAME = 2;
const ETA = 0.04;
const CONVERGED_TOL = 1e-3;
const TRAIL_SUBSAMPLE = 1;
const TRAIL_LIMIT = 400;

interface Preset {
  key: string;
  label: string;
  start: Point;
}

const PRESETS: Preset[] = [
  { key: "nw", label: "NW", start: { x: -2.1, y: -1.9 } },
  { key: "ne", label: "NE", start: { x: 2.1, y: -1.9 } },
  { key: "s", label: "S", start: { x: -0.15, y: 2.1 } },
  { key: "saddle", label: "near ridge", start: { x: -0.05, y: 1.6 } },
];

interface DescentState {
  pos: Point;
  trail: Point[];
  steps: number;
  converged: boolean;
}

function freshState(start: Point): DescentState {
  return {
    pos: { ...start },
    trail: [{ ...start }],
    steps: 0,
    converged: false,
  };
}

function gradNorm(p: Point): number {
  const { gx, gy } = gradAt(p.x, p.y);
  return Math.hypot(gx, gy);
}

export function MeanField() {
  const [presetKey, setPresetKey] = useState<string>(PRESETS[0].key);
  const [running, setRunning] = useState(true);
  const [, forceTick] = useState(0);
  const stateRef = useRef<DescentState>(
    freshState(PRESETS[0].start),
  );

  const reset = useCallback((startKey: string) => {
    const preset = PRESETS.find((p) => p.key === startKey) ?? PRESETS[0];
    stateRef.current = freshState(preset.start);
    forceTick((t) => t + 1);
  }, []);

  useEffect(() => {
    reset(presetKey);
  }, [presetKey, reset]);

  useEffect(() => {
    if (!running) return;
    let raf = 0;
    const loop = () => {
      const st = stateRef.current;
      if (!st.converged) {
        for (let s = 0; s < STEPS_PER_FRAME; s++) {
          st.pos = gradientStep(st.pos, ETA);
          st.steps++;
          if (st.steps % TRAIL_SUBSAMPLE === 0) {
            st.trail.push({ x: st.pos.x, y: st.pos.y });
            if (st.trail.length > TRAIL_LIMIT) st.trail.shift();
          }
          if (gradNorm(st.pos) < CONVERGED_TOL) {
            st.converged = true;
            break;
          }
        }
        forceTick((t) => (t + 1) % 1_000_000);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running]);

  const st = stateRef.current;
  const walker: Walker = {
    x: st.pos.x,
    y: st.pos.y,
    color: "#cf8c2a",
    trail: st.trail,
  };
  const settledWell =
    st.converged ? (st.pos.x < 0 ? "left" : "right") : "—";

  return (
    <section id="mean-field" className="lab-science-section">
      <h2>Mean field: the cheap proxy</h2>
      <p>
        Parallel tempering is correct but expensive — thousands of sweeps across{" "}
        <InlineMath formula="K" /> chains. Mean field gives a much cheaper
        approximation: replace each spin's neighbors with their{" "}
        <em>average</em> values, then iterate until the averages stop changing.
      </p>
      <BlockMath formula="m_i \leftarrow \sigma\!\bigl(\beta(h_i + {\textstyle\sum_j} J_{ij} m_j)\bigr)" />
      <p>
        On the energy landscape, this is gradient descent: roll downhill with
        no thermal noise. The marker below converges to whichever local minimum
        it starts nearest and stops. No swap moves, no temperature, no escape.
      </p>
      <p>
        That's the whole tradeoff. For queries where one basin clearly
        dominates, MF lands in the right one and the answer matches PT for a
        fraction of the cost. For queries where multiple basins matter — or
        where a different basin is actually the global minimum — MF commits
        early and never finds out. On real Pokémon completions, MF and PT
        agree on the top-1 pick about <strong>85%</strong> of the time, which
        is why <em>/completer</em> uses MF by default and only switches to PT
        when the "Full statistical sampler" toggle is on.
      </p>
      <div className="lab-science-controls">
        <div
          className="lab-t-picker"
          role="radiogroup"
          aria-label="Start position"
        >
          {PRESETS.map((p) => (
            <button
              key={p.key}
              type="button"
              role="radio"
              aria-checked={presetKey === p.key}
              className={
                "lab-t-btn" + (presetKey === p.key ? " is-selected" : "")
              }
              onClick={() => setPresetKey(p.key)}
            >
              <span className="lab-t-btn-label">{p.label}</span>
              <span className="lab-t-btn-hint">start</span>
            </button>
          ))}
        </div>
        <button type="button" onClick={() => setRunning((r) => !r)}>
          {running ? "Pause" : "Play"}
        </button>
        <button type="button" onClick={() => reset(presetKey)}>
          Reset
        </button>
      </div>
      <div className="lab-science-row">
        <figure>
          <Landscape3D width={500} height={340} walkers={[walker]} />
          <figcaption>
            Deterministic gradient descent from the selected start. The marker
            stops as soon as the gradient vanishes — i.e., at the bottom of one
            well.
          </figcaption>
        </figure>
        <div className="lab-science-pt-readout">
          <table className="lab-science-mcmc-readout">
            <tbody>
              <tr>
                <th>iterations</th>
                <td>{st.steps}</td>
              </tr>
              <tr>
                <th>|gradient|</th>
                <td>{gradNorm(st.pos).toExponential(2)}</td>
              </tr>
              <tr>
                <th>status</th>
                <td>{st.converged ? "converged" : "descending"}</td>
              </tr>
              <tr>
                <th>settled in</th>
                <td>{settledWell} well</td>
              </tr>
            </tbody>
          </table>
          <p className="lab-science-note">
            Try each preset. The "NW" and "near ridge" starts both converge to
            the left well; "NE" to the right; the symmetric ridge means
            initialization decides everything. Compare with PT above, where the
            cold-chain visits <em>both</em> wells in proportion to their
            Boltzmann weight.
          </p>
        </div>
      </div>
    </section>
  );
}
