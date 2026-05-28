// Side-by-side species coupling networks for the Metagame Model page: one
// graph of pure synergies (positive J), one of pure antisynergies (negative J).
// Each graph has its own slider setting a minimum coupling strength — every
// coupling at or above the threshold is shown, and the species involved fall
// out as the visible node set. Splitting the signs keeps each view
// human-readable; mixing both colors on one dense graph isn't useful for
// teambuilding even though it's faithful to the model.

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CouplingGraph,
  type CouplingEdge,
  type SpeciesRep,
} from "../components/CouplingGraph";
import type { IsingModel } from "../sampler/types";

const VIEW_SIZE = 520;
const NODE_RADIUS = 16;
// Cap the candidate pool to the most-used features so dragging the slider to
// 0 can't enumerate every pair in the model (~228k edges) and freeze the
// browser. Bounds candidate edges to ~TOP_FEATURES²/2.
const TOP_FEATURES = 100;
// Initial threshold is set so roughly this many of the strongest couplings
// show by default — keeps the first render readable. Users drag toward 0 to
// reveal weaker couplings.
const DEFAULT_VISIBLE = 30;

interface Props {
  model: IsingModel;
}

export function CouplingNetwork({ model }: Props) {
  const [candidates, setCandidates] = useState<{
    reps: SpeciesRep[];
    edges: CouplingEdge[];
  }>({ reps: [], edges: [] });

  // Slider extrema + a sensible default threshold (the strength of the
  // DEFAULT_VISIBLE-th strongest coupling on each side).
  const { maxPos, maxNegMag, defaultPos, defaultNeg } = useMemo(() => {
    const posMag = candidates.edges
      .filter((e) => e.J > 0)
      .map((e) => e.J)
      .sort((a, b) => b - a);
    const negMag = candidates.edges
      .filter((e) => e.J < 0)
      .map((e) => -e.J)
      .sort((a, b) => b - a);
    const nth = (arr: number[]) =>
      arr.length === 0 ? 0 : arr[Math.min(DEFAULT_VISIBLE, arr.length) - 1];
    return {
      maxPos: posMag.length ? posMag[0] : 0.01,
      maxNegMag: negMag.length ? negMag[0] : 0.01,
      defaultPos: nth(posMag),
      defaultNeg: nth(negMag),
    };
  }, [candidates.edges]);

  const [posThreshold, setPosThreshold] = useState(0);
  const [negThreshold, setNegThreshold] = useState(0);

  // Reset to the default threshold whenever the candidate pool changes
  // (e.g. user swaps Species ↔ Species @ Item).
  useEffect(() => {
    setPosThreshold(defaultPos);
  }, [defaultPos]);
  useEffect(() => {
    setNegThreshold(defaultNeg);
  }, [defaultNeg]);

  const filterPos = useCallback(
    (e: CouplingEdge) => e.J > 0 && e.J >= posThreshold,
    [posThreshold],
  );
  const filterNeg = useCallback(
    (e: CouplingEdge) => e.J < 0 && -e.J >= negThreshold,
    [negThreshold],
  );

  // Both graphs share the same candidate set; record it from whichever
  // graph reports first.
  const onCandidates = useCallback(
    (info: { reps: SpeciesRep[]; edges: CouplingEdge[] }) => {
      setCandidates(info);
    },
    [],
  );

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 24,
        marginBottom: 12,
      }}
    >
      <div>
        <div className="lab-subheading lab-subheading-pos">
          Synergies · positive couplings
        </div>
        <div className="lab-science-controls">
          <label>
            Min strength {posThreshold.toFixed(2)}{" "}
            <input
              type="range"
              className="lab-slider"
              min={0}
              max={maxPos}
              step={0.01}
              value={posThreshold}
              onChange={(e) => setPosThreshold(Number(e.target.value))}
            />
          </label>
        </div>
        <CouplingGraph
          model={model}
          filterEdge={filterPos}
          topSpecies={TOP_FEATURES}
          viewSize={VIEW_SIZE}
          nodeRadius={NODE_RADIUS}
          onCandidates={onCandidates}
          renderCaption={({ visibleNodes, visibleEdges }) => (
            <>
              {visibleNodes} species, {visibleEdges} positive couplings shown.
              Thickness ∝ strength.
            </>
          )}
          emptyMessage="No couplings this strong. Slide left to lower the threshold."
        />
      </div>
      <div>
        <div className="lab-subheading lab-subheading-neg">
          Antisynergies · negative couplings
        </div>
        <div className="lab-science-controls">
          <label>
            Min strength {negThreshold.toFixed(2)}{" "}
            <input
              type="range"
              className="lab-slider"
              min={0}
              max={maxNegMag}
              step={0.01}
              value={negThreshold}
              onChange={(e) => setNegThreshold(Number(e.target.value))}
            />
          </label>
        </div>
        <CouplingGraph
          model={model}
          filterEdge={filterNeg}
          topSpecies={TOP_FEATURES}
          viewSize={VIEW_SIZE}
          nodeRadius={NODE_RADIUS}
          renderCaption={({ visibleNodes, visibleEdges }) => (
            <>
              {visibleNodes} species, {visibleEdges} negative couplings shown.
              Thickness ∝ |strength|.
            </>
          )}
          emptyMessage="No couplings this strong. Slide left to lower the threshold."
        />
      </div>
    </div>
  );
}
