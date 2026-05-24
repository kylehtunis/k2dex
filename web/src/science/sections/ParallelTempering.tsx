// Parallel tempering on a synthetic 2D energy landscape with two wells.
// Single-chain mode: one cold walker, gets trapped. PT mode: K walkers at
// increasing T with adjacent-rung swap proposals; the cold chain inherits
// the hot chain's mobility.

import { useCallback, useEffect, useRef, useState } from "react";
import { BlockMath, InlineMath } from "../widgets/Math";
import { Landscape3D, type Walker } from "../widgets/Landscape3D";
import { energyAt, metropolisStep, type Point } from "../primitives/landscape";
import { Rng } from "../../sampler/rng";

const STEPS_PER_FRAME = 6;
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

export function ParallelTempering() {
  const [mode, setMode] = useState<Mode>("single");
  const [running, setRunning] = useState(false);
  const [showHotChains, setShowHotChains] = useState(false);
  const [, forceTick] = useState(0);
  const rngRef = useRef(new Rng(Math.floor(Math.random() * 2 ** 30)));
  const stateRef = useRef<SamplerState>(freshState());
  const swapCounterRef = useRef(0);

  const reset = useCallback(() => {
    rngRef.current = new Rng(Math.floor(Math.random() * 2 ** 30));
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
              const dE = energyAt(a.x, a.y) - energyAt(b.x, b.y);
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
      : st.ptChains
          .map((p, k) => ({
            x: p.x,
            y: p.y,
            color: RUNG_COLORS[k],
            trail: st.ptTrails[k],
          }))
          .filter((_, k) => k === 0 || showHotChains);

  const leftFrac =
    st.totalSteps > 0 ? st.leftCount / (st.leftCount + st.rightCount) : 0;

  return (
    <section id="pt" className="lab-science-section">
      <h3>Energy landscapes</h3>
      <p>
        Before we go further, it's important to understand the concept of an energy landscape.
        If you're already comfortable with the concept of energy landscapes then skip right on ahead to the next section.
        If not, I'll give you a quick primer here so that you can understand some of the more mind-bending concepts coming up.
      </p>
      <p>
        An <b>energy landscape</b> is a way of visualizing the energy function that defines a system. 
        Each point in the landscape corresponds to a particular configuration of the system, and the height of the landscape at that point corresponds to the energy of that configuration.
        A "configuration" here means a specific arrangement of all the spins, 
        so the small 32x32 lattice you played with earlier has <InlineMath formula="2^{32^2} = 2^{1024}" /> possible configurations.
      </p>
      <p>
        Imagine a large mountain range: Each of those <InlineMath formula="2^{1024}" /> configurations is a different spot in that mountain range, and the altitude at each point is the energy of that configuration.
        Valleys represent low-energy configurations (like a fully aligned magnet) and peaks represent high-energy configurations.
        Systems, including the Ising model, tend to "roll downhill" in this energy landscape as they look for low-energy configurations.
        The Metropolis algorithm can be thought of as a hiker exploring this landscape: every step, he points in a random direction. If it's downhill, he always takes it. If it's uphill, he might still take it, but he's less likely to if it's a steep climb. 
      </p>
      <p>
        (Technically, the landscape in our case would be in a 1024-dimensional space, but picturing a 2d landscape (with the 3rd dimension representing energy) is an easy mental image and the math is exactly the same.)
      </p>
      <p>
        Energy landscapes show up all over the place in science. 
        The "mountain range" analogy is so common that it's common to use geographical terminology to describe what's going on.
        When I use terms like "valley", "ridge", or "basin" in the next few sections, I'm referring to features of the energy landscape.
        There are some simulations below that will make this all make a lot more sense.
      </p>
      <h3>Parallel tempering</h3>
      <p>
        If you got upset when I said earlier that Metropolis will sample the Boltzmann distribution "given infinite time", good instincts.
        The random walk through the energy landscape can get stuck in a specific basin for longer than the duration of a finite simulation.
        You can see this in action by running the simulation below.
      </p>
      <p>
        Parallel tempering (<a href="#ref-hukushima">Hukushima &amp; Nemoto, 1996</a>){" "}
        runs <InlineMath formula="K" /> chains at temperatures{" "}
        <InlineMath formula="T_1 < T_2 < \dots < T_K" />. The hottest chain
        roams over the whole landscape; the coldest stays concentrated in
        whichever well it currently occupies. Periodically, the algorithm
        proposes that adjacent chains swap configurations, with an acceptance probability that preserves the Boltzmann distribution:
      </p>
      <BlockMath formula="p_\text{swap} = \min\bigl(1,\ e^{(T_k^{-1} - T_{k+1}^{-1})(H_{k} - H_{k+1})}\bigr)" />
      <p>
        The hotter chains explore the full landscape, and periodically swap with colder chains for detailed exploration.
        This gives a much more thorough sampling of the entire landscape relative to the single chain Metropolis version.
        Check for yourself by switching back and forth between single-chain and PT modes in the simulation.
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
            <span className="lab-t-btn-label">Metropolis</span>
            <span className="lab-t-btn-hint">single chain</span>
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
        {mode === "pt" && (
          <label className="lab-science-checkbox-label">
            <input
              type="checkbox"
              checked={showHotChains}
              onChange={(e) => setShowHotChains(e.target.checked)}
            />
            show hot chains
          </label>
        )}
      </div>
      <div className="lab-science-row">
        <figure>
          <Landscape3D width={500} height={340} walkers={walkers} />
          <figcaption>
            {mode === "single"
              ? "Single T=0.25 walker. Watch it struggle to escape its well."
              : showHotChains
                ? "PT walkers: blue = cold (T = 0.25), gold = mid (T = 0.8), red = hot (T = 2.5). The cold walker hitches rides across the ridge via swaps."
                : "PT cold walker (T = 0.25). The walker should now explore both landscapes, frequently switching without needing to explore high-energy areas. (Enable \"show hot chains\" to see the full ladder.)"}
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
        </div>
      </div>
    </section>
  );
}
