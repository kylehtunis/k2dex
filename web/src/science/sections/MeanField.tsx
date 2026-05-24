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

const STEPS_PER_FRAME = 1;
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
  { key: "nw", label: "Start NW", start: { x: -2.1, y: -1.9 } },
  { key: "ne", label: "Start NE", start: { x: 2.1, y: -1.9 } },
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
  const [running, setRunning] = useState(false);
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
      <h3>Mean field</h3>
      <p>
        Parallel tempering gives reliable results but it's computationally expensive, requiring thousands of sweeps across multiple chains. 
        Mean field theory gives us an alternative which is much simpler to compute, at the cost of being only an approximation.
        Rather than attempting to sample the full Boltzmann distribution, MF directly estimates the mean spin values by iteratively applying the MF update equation:
      </p>
      <BlockMath formula="m_i \leftarrow \sigma\!\bigl(\beta(h_i + {\textstyle\sum_j} J_{ij} m_j)\bigr)" />
      <p>
        On the energy landscape, this is gradient descent: just roll directly downhill.
        The marker below converges to whichever local minimum
        it starts nearest and stops. No swap moves, no temperature, no escape. 
        This is similar to (though not exactly the same as) running Metropolis or Parallel Tempering at <InlineMath formula="T=0" />.
      </p>
      <p>
        We also use it as the starting point for the Team Completer if you don't use full sampling. 
        We were talking about Pokemon, remember? 
        We'll get back to Pokemon and how this all ties together soon, but first I need to take <i>one more</i> detour and talk about the United States Supreme Court from 1994-2005.
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
              <span className="lab-t-btn-hint"></span>
            </button>
          ))}
        </div>
        <button type="button" className="lab-science-btn" onClick={() => setRunning((r) => !r)}>
          {running ? "Pause" : "Play"}
        </button>
        <button type="button" className="lab-science-btn" onClick={() => reset(presetKey)}>
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
        </div>
      </div>
    </section>
  );
}
