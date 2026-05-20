// Parallel tempering on a synthetic 2D energy landscape with two wells.
// Single-chain mode: one cold walker, gets trapped. PT mode: K walkers at
// increasing T with adjacent-rung swap proposals; the cold chain inherits
// the hot chain's mobility.

import { useCallback, useEffect, useRef, useState } from "react";
import { BlockMath, InlineMath } from "../widgets/Math";
import { Landscape3D, type Walker } from "../widgets/Landscape3D";
import { energyAt, metropolisStep, type Point } from "../primitives/landscape";
import { Rng } from "../../sampler/rng";

const STEPS_PER_FRAME = 18;
const SIGMA = 0.18;
const COLD_T = 0.25;
const PT_LADDER = [0.25, 0.8, 2.5];
const RUNG_COLORS = ["#1f4e8c", "#cf8c2a", "#9c2a2a"];
const SWAP_INTERVAL_STEPS = 25;
const TRAIL_SUBSAMPLE = 3;
const TRAIL_LIMIT = 220;

// Periphery starts so the descent is visible.
const SINGLE_START: Point = { x: -2.1, y: -1.9 };
const PT_STARTS: Point[] = [
  { x: -2.1, y: -1.9 },
  { x: 2.1, y: -1.9 },
  { x: 0.0, y: 2.1 },
];

type Mode = "single" | "pt";

interface SamplerState {
  single: Point;
  ptChains: Point[];
  singleTrail: Point[];
  ptTrails: Point[][];
  ptSwaps: number[];
  leftCount: number;
  rightCount: number;
  totalSteps: number;
}

function freshState(): SamplerState {
  return {
    single: { ...SINGLE_START },
    ptChains: PT_STARTS.map((p) => ({ ...p })),
    singleTrail: [{ ...SINGLE_START }],
    ptTrails: PT_STARTS.map((p) => [{ ...p }]),
    ptSwaps: [0, 0],
    leftCount: 0,
    rightCount: 0,
    totalSteps: 0,
  };
}

function pushTrail(trail: Point[], p: Point) {
  trail.push({ x: p.x, y: p.y });
  if (trail.length > TRAIL_LIMIT) trail.shift();
}

export function PT() {
  const [mode, setMode] = useState<Mode>("single");
  const [running, setRunning] = useState(true);
  const [, forceTick] = useState(0);
  const rngRef = useRef(new Rng(7));
  const stateRef = useRef<SamplerState>(freshState());
  const swapCounterRef = useRef(0);

  const reset = useCallback(() => {
    rngRef.current = new Rng(7);
    stateRef.current = freshState();
    swapCounterRef.current = 0;
    forceTick((t) => t + 1);
  }, []);

  useEffect(() => {
    reset();
  }, [mode, reset]);

  useEffect(() => {
    if (!running) return;
    let raf = 0;
    const loop = () => {
      const st = stateRef.current;
      const rng = rngRef.current;
      for (let s = 0; s < STEPS_PER_FRAME; s++) {
        if (mode === "single") {
          st.single = metropolisStep(st.single, COLD_T, SIGMA, rng);
          if (st.single.x < 0) st.leftCount++;
          else st.rightCount++;
          if (st.totalSteps % TRAIL_SUBSAMPLE === 0) {
            pushTrail(st.singleTrail, st.single);
          }
        } else {
          for (let k = 0; k < PT_LADDER.length; k++) {
            st.ptChains[k] = metropolisStep(
              st.ptChains[k],
              PT_LADDER[k],
              SIGMA,
              rng,
            );
          }
          swapCounterRef.current++;
          if (swapCounterRef.current % SWAP_INTERVAL_STEPS === 0) {
            for (let k = 0; k < PT_LADDER.length - 1; k++) {
              const a = st.ptChains[k];
              const b = st.ptChains[k + 1];
              const dE = energyAt(b.x, b.y) - energyAt(a.x, a.y);
              const dBeta = 1 / PT_LADDER[k] - 1 / PT_LADDER[k + 1];
              const logAcc = dBeta * dE;
              if (logAcc >= 0 || rng.random() < Math.exp(logAcc)) {
                [st.ptChains[k], st.ptChains[k + 1]] = [
                  st.ptChains[k + 1],
                  st.ptChains[k],
                ];
                st.ptSwaps[k]++;
              }
            }
          }
          if (st.ptChains[0].x < 0) st.leftCount++;
          else st.rightCount++;
          if (st.totalSteps % TRAIL_SUBSAMPLE === 0) {
            for (let k = 0; k < PT_LADDER.length; k++) {
              pushTrail(st.ptTrails[k], st.ptChains[k]);
            }
          }
        }
        st.totalSteps++;
      }
      forceTick((t) => (t + 1) % 1_000_000);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [mode, running]);

  const st = stateRef.current;
  const walkers: Walker[] =
    mode === "single"
      ? [
          {
            x: st.single.x,
            y: st.single.y,
            color: "#222",
            trail: st.singleTrail,
          },
        ]
      : st.ptChains.map((p, k) => ({
          x: p.x,
          y: p.y,
          color: RUNG_COLORS[k],
          trail: st.ptTrails[k],
        }));

  const leftFrac =
    st.totalSteps > 0 ? st.leftCount / (st.leftCount + st.rightCount) : 0;

  return (
    <section id="pt" className="lab-science-section">
      <h2>Parallel tempering: escaping basins at low T</h2>
      <p>
        We can't visualize the full Ising energy surface over{" "}
        <InlineMath formula="2^V" /> spin configurations, but its qualitative
        shape is what matters here: a high-dimensional landscape with multiple
        deep basins separated by ridges. The 2D analog below has the same key
        property — two wells of equal depth at{" "}
        <InlineMath formula="x = \pm 1" /> with a ridge in between.
      </p>
      <p>
        At low <InlineMath formula="T" />, a single Metropolis chain barely
        ever climbs that ridge. It samples one well densely and never sees the
        other; the marginals it reports reflect <em>where it got trapped</em>,
        not the true Boltzmann distribution.
      </p>
      <p>
        Parallel tempering runs <InlineMath formula="K" /> chains at temperatures{" "}
        <InlineMath formula="T_1 < T_2 < \dots < T_K" />. The hottest chain
        roams over the whole landscape; the coldest stays concentrated in
        whichever well it currently occupies. Periodically the algorithm
        proposes a swap between adjacent rungs:
      </p>
      <BlockMath formula="p_\text{swap} = \min\bigl(1,\ e^{(\beta_k - \beta_{k+1})(H_{k+1} - H_k)}\bigr)" />
      <p>
        The swap lets the cold chain "ride" the hot chain's mobility across
        ridges, then resume detailed exploration on the other side. This is why
        the "Full statistical sampler" toggle in <em>/completer</em> uses PT.
      </p>
      <div className="lab-science-controls">
        <div className="lab-t-picker" role="radiogroup" aria-label="Sampler">
          <button
            type="button"
            role="radio"
            aria-checked={mode === "single"}
            className={"lab-t-btn" + (mode === "single" ? " is-selected" : "")}
            onClick={() => setMode("single")}
          >
            <span className="lab-t-btn-label">Single</span>
            <span className="lab-t-btn-hint">cold chain only</span>
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={mode === "pt"}
            className={"lab-t-btn" + (mode === "pt" ? " is-selected" : "")}
            onClick={() => setMode("pt")}
          >
            <span className="lab-t-btn-label">PT</span>
            <span className="lab-t-btn-hint">3-rung ladder</span>
          </button>
        </div>
        <button type="button" onClick={() => setRunning((r) => !r)}>
          {running ? "Pause" : "Play"}
        </button>
        <button type="button" onClick={reset}>
          Reset
        </button>
      </div>
      <div className="lab-science-row">
        <figure>
          <Landscape3D width={500} height={340} walkers={walkers} />
          <figcaption>
            {mode === "single"
              ? "Single cold-T walker. Watch it stay in one well."
              : "PT walkers: blue = cold (T = 0.25), gold = mid (T = 0.8), red = hot (T = 2.5). The cold walker hitches rides across the ridge via swaps."}
          </figcaption>
        </figure>
        <div className="lab-science-pt-readout">
          <table className="lab-science-mcmc-readout">
            <tbody>
              <tr>
                <th>steps</th>
                <td>{st.totalSteps.toLocaleString()}</td>
              </tr>
              <tr>
                <th>cold chain in left well</th>
                <td>{(leftFrac * 100).toFixed(1)}%</td>
              </tr>
              <tr>
                <th>true (by symmetry)</th>
                <td>50.0%</td>
              </tr>
              {mode === "pt" && (
                <>
                  <tr>
                    <th>swaps cold↔mid</th>
                    <td>{st.ptSwaps[0]}</td>
                  </tr>
                  <tr>
                    <th>swaps mid↔hot</th>
                    <td>{st.ptSwaps[1]}</td>
                  </tr>
                </>
              )}
            </tbody>
          </table>
          <p className="lab-science-note">
            The two wells are symmetric, so the true fraction of time in the
            left well is exactly 50%. The single-chain estimate stays pinned
            near 0% or 100% indefinitely; the PT cold chain drifts toward 50%
            as swaps move it back and forth.
          </p>
        </div>
      </div>
    </section>
  );
}
