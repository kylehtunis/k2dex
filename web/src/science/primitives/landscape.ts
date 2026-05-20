// Synthetic 2D energy landscape with two wells, used as a visual analog for
// the high-dimensional Ising energy surface. Wells at (-1, 0) and (+1, 0),
// with a quadratic bowl so the domain stays finite.
//
// Used by the PT and MeanField sections to make basin-trapping intuitions
// concrete. NOT a faithful projection of any specific Ising graph — it's a
// pedagogical caricature.

import type { Rng } from "../../sampler/rng";

export interface Point {
  x: number;
  y: number;
}

const WELL_DEPTH = 3.0;
const WELL_WIDTH_SQ = 0.6;
const BOWL_K = 0.25;
export const DOMAIN_MIN = -2.4;
export const DOMAIN_MAX = 2.4;

export function energyAt(x: number, y: number): number {
  const w1 =
    -WELL_DEPTH * Math.exp(-(((x + 1) ** 2 + y * y) / WELL_WIDTH_SQ));
  const w2 =
    -WELL_DEPTH * Math.exp(-(((x - 1) ** 2 + y * y) / WELL_WIDTH_SQ));
  const bowl = BOWL_K * (x * x + y * y);
  return w1 + w2 + bowl;
}

/** Analytic gradient of energyAt. */
export function gradAt(x: number, y: number): { gx: number; gy: number } {
  const W1 =
    -WELL_DEPTH * Math.exp(-(((x + 1) ** 2 + y * y) / WELL_WIDTH_SQ));
  const W2 =
    -WELL_DEPTH * Math.exp(-(((x - 1) ** 2 + y * y) / WELL_WIDTH_SQ));
  const gx =
    (-2 * (x + 1) / WELL_WIDTH_SQ) * W1 +
    (-2 * (x - 1) / WELL_WIDTH_SQ) * W2 +
    2 * BOWL_K * x;
  const gy =
    (-2 * y / WELL_WIDTH_SQ) * (W1 + W2) + 2 * BOWL_K * y;
  return { gx, gy };
}

/**
 * One gradient-descent step. Used as the deterministic mean-field analog
 * on the continuous landscape: roll downhill, no thermal noise.
 */
export function gradientStep(p: Point, eta: number): Point {
  const { gx, gy } = gradAt(p.x, p.y);
  const nx = Math.max(DOMAIN_MIN, Math.min(DOMAIN_MAX, p.x - eta * gx));
  const ny = Math.max(DOMAIN_MIN, Math.min(DOMAIN_MAX, p.y - eta * gy));
  return { x: nx, y: ny };
}

/** Box-Muller standard normal sample. */
function randn(rng: Rng): number {
  const u1 = Math.max(rng.random(), 1e-12);
  const u2 = rng.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * One Metropolis step on the continuous landscape. Returns a new point
 * (caller decides whether to keep or discard the input).
 */
export function metropolisStep(
  p: Point,
  T: number,
  sigma: number,
  rng: Rng,
): Point {
  const nx = p.x + sigma * randn(rng);
  const ny = p.y + sigma * randn(rng);
  if (
    nx < DOMAIN_MIN ||
    nx > DOMAIN_MAX ||
    ny < DOMAIN_MIN ||
    ny > DOMAIN_MAX
  ) {
    return p;
  }
  const dE = energyAt(nx, ny) - energyAt(p.x, p.y);
  if (dE <= 0 || rng.random() < Math.exp(-dE / Math.max(T, 1e-9))) {
    return { x: nx, y: ny };
  }
  return p;
}

/** Isometric projection parameters for a Landscape3D render. */
export interface IsoProjection {
  scaleXY: number;
  scaleZ: number;
  centerU: number;
  centerV: number;
}

const COS30 = Math.cos(Math.PI / 6);
const SIN30 = Math.sin(Math.PI / 6);

/** Project a 3D point (x, y, z) to screen coords (u, v). */
export function project(
  p: { x: number; y: number; z: number },
  P: IsoProjection,
): { u: number; v: number } {
  return {
    u: P.centerU + (p.x - p.y) * COS30 * P.scaleXY,
    v: P.centerV + (p.x + p.y) * SIN30 * P.scaleXY - p.z * P.scaleZ,
  };
}
