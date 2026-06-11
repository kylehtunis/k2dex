// Force-directed species-coupling graph. One node per top species (collapsing
// Species @ Item models down to the highest-marginal build per species), edges
// are pairwise J among them. The caller supplies an edge filter predicate;
// the component re-runs spring layout each time the visible edge set changes
// and renders the result via GraphView.
//
// Used by both the /science Pokemon section and the Metagame Model page's
// coupling-network figure — the filter UI lives in those callers.

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { GraphView } from "../science/widgets/GraphView";
import { makeGraph, springLayout } from "../science/primitives/graph";
import { Rng } from "../sampler/rng";
import type { IsingModel } from "../sampler/types";
import { extractItem, extractSpecies } from "../render/format";
import { itemSpriteUrl, spriteUrl } from "../render/sprite-url";

export interface CouplingEdge {
  /** Local index into the reps array. */
  a: number;
  /** Local index into the reps array. */
  b: number;
  J: number;
}

export interface SpeciesRep {
  /** Display species name (extracted from vocab string for Phase 3). */
  species: string;
  /** Item string for Phase 3 features, null for Phase 2 / itemless features. */
  item: string | null;
  vocabIdx: number;
  m: number;
}

const LAYOUT_SEED = 0x5eed;
const LAYOUT_ITERS = 320;

interface Props {
  model: IsingModel;
  /** Filter applied to every candidate edge; only those returning true are drawn. */
  filterEdge: (edge: CouplingEdge) => boolean;
  topSpecies?: number;
  viewSize?: number;
  nodeRadius?: number;
  maxStrokeWidth?: number;
  /** Caption renderer; omit for no caption. */
  renderCaption?: (info: {
    reps: SpeciesRep[];
    visibleNodes: number;
    visibleEdges: number;
  }) => ReactNode;
  /** Shown when no edges pass the filter. */
  emptyMessage?: ReactNode;
  /** Optional callback exposing the candidate (pre-filter) edge set so callers
   * can compute slider ranges, extrema, etc. without recomputing themselves. */
  onCandidates?: (info: { reps: SpeciesRep[]; edges: CouplingEdge[] }) => void;
}

// Push apart any sprites closer than `minSep` pixels. A few relaxation passes
// after the spring layout: dense coupling cores otherwise collapse on top of
// each other, and sprites (unlike dots) can't usefully overlap.
function relaxOverlaps(
  pts: { x: number; y: number }[],
  minSep: number,
  iters: number,
  size: number,
  pad: number,
) {
  for (let it = 0; it < iters; it++) {
    for (let a = 0; a < pts.length; a++) {
      for (let b = a + 1; b < pts.length; b++) {
        let dx = pts[b].x - pts[a].x;
        let dy = pts[b].y - pts[a].y;
        let d = Math.hypot(dx, dy);
        if (d < 1e-6) {
          dx = a - b || 1;
          dy = 1;
          d = Math.hypot(dx, dy);
        }
        if (d < minSep) {
          const push = (minSep - d) / 2;
          pts[a].x -= (dx / d) * push;
          pts[a].y -= (dy / d) * push;
          pts[b].x += (dx / d) * push;
          pts[b].y += (dy / d) * push;
        }
      }
    }
    for (const p of pts) {
      p.x = Math.min(size - pad, Math.max(pad, p.x));
      p.y = Math.min(size - pad, Math.max(pad, p.y));
    }
  }
}

export function CouplingGraph({
  model,
  filterEdge,
  topSpecies = Infinity,
  viewSize = 600,
  nodeRadius = 18,
  maxStrokeWidth = 6,
  renderCaption,
  emptyMessage,
  onCandidates,
}: Props) {
  const spriteSize = nodeRadius * 2.2;
  const viewRadius = viewSize / 3;

  // One rep per vocab feature. For Phase 2 each rep is a bare species; for
  // Phase 3 each rep is a distinct (species, item) build. `topSpecies` caps
  // the candidate pool to the most-marginal features.
  const reps = useMemo<SpeciesRep[]>(() => {
    const all: SpeciesRep[] = [];
    for (let i = 0; i < model.V; i++) {
      all.push({
        species: model.speciesOf[i],
        item: model.itemOf[i],
        vocabIdx: i,
        m: model.m[i],
      });
    }
    all.sort((a, b) => b.m - a.m);
    return all.slice(0, Math.min(topSpecies, all.length));
  }, [model, topSpecies]);

  // Skip same-species and same-item pairs — those couplings exist purely
  // from mechanical mutual exclusion in VGC and don't reflect model
  // structure (mirrors meta/couplings.ts:filteredCouplings).
  const candidateEdges = useMemo<CouplingEdge[]>(() => {
    const V = model.V;
    const out: CouplingEdge[] = [];
    for (let x = 0; x < reps.length; x++) {
      const rx = reps[x];
      for (let y = x + 1; y < reps.length; y++) {
        const ry = reps[y];
        if (rx.species === ry.species) continue;
        if (rx.item !== null && ry.item !== null && rx.item === ry.item) continue;
        const w = model.J[rx.vocabIdx * V + ry.vocabIdx];
        if (w !== 0) out.push({ a: x, b: y, J: w });
      }
    }
    return out;
  }, [model, reps]);

  useEffect(() => {
    onCandidates?.({ reps, edges: candidateEdges });
  }, [reps, candidateEdges, onCandidates]);

  const { nodes, edges, visibleEdges, imageUrls } = useMemo(() => {
    const visible = candidateEdges.filter(filterEdge);
    const used = [...new Set(visible.flatMap((e) => [e.a, e.b]))].sort(
      (a, b) => a - b,
    );
    const compact = new Map(used.map((local, k) => [local, k]));
    const layoutEdges = visible.map(
      (e) => [compact.get(e.a)!, compact.get(e.b)!, e.J] as const,
    );
    const g = makeGraph(used.length, layoutEdges, new Array(used.length).fill(0));
    const pos = springLayout(g, new Rng(LAYOUT_SEED), LAYOUT_ITERS);
    const pts = pos.map((p) => ({
      x: viewSize / 2 + viewRadius * p.x,
      y: viewSize / 2 + viewRadius * p.y,
    }));
    relaxOverlaps(pts, spriteSize * 1.0, 120, viewSize, spriteSize / 2);
    const nodes = used.map((local, k) => ({
      id: local,
      label: "",
      x: pts[k].x,
      y: pts[k].y,
      feature: model.vocab[reps[local].vocabIdx],
    }));
    const edges = visible.map((e) => ({ i: e.a, j: e.b, weight: e.J }));
    // Collect sprite + item URLs for preloading. Preloading into the browser
    // cache before mounting the SVG eliminates the pop-in where lines render
    // instantly but `<object data=...>` / `<img src=...>` requests trickle in.
    const imageUrls: string[] = [];
    for (const n of nodes) {
      imageUrls.push(spriteUrl(extractSpecies(n.feature)));
      const itemUrl = itemSpriteUrl(extractItem(n.feature));
      if (itemUrl) imageUrls.push(itemUrl);
    }
    return { nodes, edges, visibleEdges: visible.length, imageUrls };
  }, [candidateEdges, filterEdge, reps, model, viewSize, viewRadius, spriteSize]);

  // Wait for all sprite URLs to finish loading (or erroring) before revealing
  // the graph. Cached URLs resolve synchronously on subsequent renders so
  // slider drags don't visibly re-flash.
  const [ready, setReady] = useState(false);
  useEffect(() => {
    if (imageUrls.length === 0) {
      setReady(true);
      return;
    }
    setReady(false);
    let cancelled = false;
    let remaining = imageUrls.length;
    const done = () => {
      if (cancelled) return;
      remaining -= 1;
      if (remaining <= 0) setReady(true);
    };
    for (const url of imageUrls) {
      const img = new Image();
      img.onload = done;
      img.onerror = done;
      img.src = url;
    }
    return () => {
      cancelled = true;
    };
  }, [imageUrls.join("|")]);

  if (nodes.length === 0) {
    return (
      <p style={{ color: "var(--lab-ink-muted)" }}>
        {emptyMessage ??
          "No couplings pass the current filter. Adjust the controls to bring species back."}
      </p>
    );
  }

  return (
    <figure style={{ margin: 0 }}>
      <div
        style={{
          position: "relative",
          width: "100%",
          maxWidth: viewSize,
          aspectRatio: "1 / 1",
        }}
      >
        <div
          aria-hidden={ready}
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            opacity: ready ? 0 : 1,
            transition: "opacity 0.2s ease-out",
            pointerEvents: ready ? "none" : "auto",
          }}
        >
          <div className="lab-spinner" />
        </div>
        <div
          style={{
            position: "absolute",
            inset: 0,
            opacity: ready ? 1 : 0,
            transition: "opacity 0.25s ease-out",
          }}
        >
          <GraphView
            nodes={nodes}
            edges={edges}
            width={viewSize}
            height={viewSize}
            nodeRadius={nodeRadius}
            maxStrokeWidth={maxStrokeWidth}
            showLabels={false}
            spriteOpacity={1.0}
          />
        </div>
      </div>
      {renderCaption && (
        <figcaption>
          {renderCaption({ reps, visibleNodes: nodes.length, visibleEdges })}
        </figcaption>
      )}
    </figure>
  );
}
