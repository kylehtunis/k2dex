"""Feature classification by coupling structure.

Per-feature metrics derived from a fitted (J, m): the signed net coupling
to the meta (`J·m`), the unsigned magnitude (`|J|·m`), and the partner-
agnostic total (`Σ|J|`). These let us label features as glue, outcast,
archetype specialist, or flex filler. See `feature_classification.ipynb`
and the `/meta` page for the user-facing surfaces.

Pure functions only. No Streamlit imports; safe to import from notebooks
and from `app.py`. Matplotlib used only in `build_scatter_figure`.
"""
from __future__ import annotations

import numpy as np


def feature_metrics(J: np.ndarray, m: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-feature coupling summaries.

    Keys:
        j_dot_m     — signed net coupling to the meta: Σ_j J_ij m_j
        abs_j_dot_m — popularity-weighted total magnitude: Σ_j |J_ij| m_j
        sum_abs_j   — partner-agnostic total magnitude: Σ_j |J_ij|

    Diagonal of J should be zero (standard for Ising fits); function does
    not assume this but works correctly when it holds.
    """
    j_dot_m = J @ m
    abs_j_dot_m = np.abs(J) @ m
    sum_abs_j = np.abs(J).sum(axis=1)
    return {
        "j_dot_m": j_dot_m,
        "abs_j_dot_m": abs_j_dot_m,
        "sum_abs_j": sum_abs_j,
    }


def residual(h: np.ndarray, m: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Return r_i = h_i − logit(m_i).

    Under mean-field self-consistency: r ≈ −(J·m). PL doesn't enforce MF,
    so they differ — the gap is itself a diagnostic of fit quality. The
    notebook's MF-vs-PL sanity-check plot uses this directly.

    `m` is clipped to [eps, 1 − eps] before taking the logit so boundary
    marginals don't produce ±inf.
    """
    m_clipped = np.clip(m, eps, 1.0 - eps)
    logit_m = np.log(m_clipped / (1.0 - m_clipped))
    return h - logit_m


def partner_contributions(J: np.ndarray, m: np.ndarray, *, i: int) -> np.ndarray:
    """Return the length-V vector of per-partner contributions to (J·m)_i.

    The k-th entry is J[i, k] * m[k]. Summing across k recovers (J·m)_i.
    Caller is responsible for sorting / slicing / labeling.
    """
    return J[i, :] * m


def classify_features(
    *,
    j_dot_m: np.ndarray,
    abs_j_dot_m: np.ndarray,
    m: np.ndarray,
    m_floor: float,
    top_k: int,
) -> dict[str, list[int]]:
    """Sort surviving features (m ≥ m_floor) into four ranked top-K lists.

    Rules (all applied to the post-floor survivor set):
      glue       = top-K by  j_dot_m  (descending)
      outcast    = top-K by -j_dot_m  (ascending — most negative)
      central band = features with |j_dot_m| ≤ median(|j_dot_m|)
      specialist = top-K by  abs_j_dot_m  within the central band
      flex       = top-K by -abs_j_dot_m within the central band

    When the central band is smaller than 2 * top_k, the specialist and
    flex lists may overlap on at least one feature. That's by design —
    overlap is informative ("this feature is the only candidate", not
    "the rules contradict"). On real fits (V ≥ ~100 surviving) the
    central band is large enough that overlap doesn't occur.

    Returns a dict with keys 'glue', 'outcast', 'specialist', 'flex',
    each a list of int indices into the full V-length vectors (not into
    the surviving subset).
    """
    survivors = np.where(m >= m_floor)[0]
    if survivors.size == 0:
        return {"glue": [], "outcast": [], "specialist": [], "flex": []}

    jm_surv = j_dot_m[survivors]
    abs_jm_surv = abs_j_dot_m[survivors]

    glue_local = np.argsort(-jm_surv)[:top_k]
    outcast_local = np.argsort(jm_surv)[:top_k]

    median_abs_jm = float(np.median(np.abs(jm_surv)))
    in_band = np.where(np.abs(jm_surv) <= median_abs_jm)[0]
    band_abs_jm = abs_jm_surv[in_band]
    specialist_band = in_band[np.argsort(-band_abs_jm)[:top_k]]
    flex_band = in_band[np.argsort(band_abs_jm)[:top_k]]

    return {
        "glue": survivors[glue_local].tolist(),
        "outcast": survivors[outcast_local].tolist(),
        "specialist": survivors[specialist_band].tolist(),
        "flex": survivors[flex_band].tolist(),
    }


def build_scatter_figure(
    *,
    j_dot_m: np.ndarray,
    abs_j_dot_m: np.ndarray,
    m: np.ndarray,
    classification: dict[str, list[int]],
    vocab: list[str],
    label_top_k: int,
    figsize: tuple[float, float] = (10.0, 5.0),
):
    """Build the (J·m, |J|·m) scatter Figure.

    Color rules:
      glue       — warm green (#3a7d44)  matches positive-coupling color in app.py
      outcast    — warm red   (#b03030)
      specialist — purple     (#6a4ba8)
      flex       — slate      (#5a6b78)
      unclassified — light gray, low alpha, smaller marker

    `label_top_k` highest-m features (independent of color class) get text
    annotations using vocab names. No external label-deorbiting dep
    (e.g. adjustText) — overlap is tolerated; tune offsets if it becomes
    a problem.

    Returns a matplotlib Figure. Caller is responsible for `plt.close()`.
    """
    import matplotlib.pyplot as plt

    color_by_class = {
        "glue":       "#3a7d44",
        "outcast":    "#b03030",
        "specialist": "#6a4ba8",
        "flex":       "#5a6b78",
    }

    V = len(j_dot_m)
    label_of: list[str | None] = [None] * V
    for cname, idxs in classification.items():
        for i in idxs:
            label_of[int(i)] = cname

    fig, ax = plt.subplots(figsize=figsize)

    ax.axvline(0.0, color="black", lw=0.4, alpha=0.6)
    median_abs_jm = float(np.median(np.abs(j_dot_m)))
    ax.axhline(median_abs_jm, color="black", lw=0.4, alpha=0.6, linestyle="--")

    unc_idx = [i for i in range(V) if label_of[i] is None]
    if unc_idx:
        ax.scatter(
            j_dot_m[unc_idx], abs_j_dot_m[unc_idx],
            s=np.clip(m[unc_idx] * 600, 4, 80),
            color="#bfbfbf", alpha=0.5, linewidths=0,
        )

    for cname, color in color_by_class.items():
        cls_idx = classification.get(cname, [])
        if not cls_idx:
            continue
        cls_idx = [int(i) for i in cls_idx]
        ax.scatter(
            j_dot_m[cls_idx], abs_j_dot_m[cls_idx],
            s=np.clip(m[cls_idx] * 600, 8, 120),
            color=color, alpha=0.85, linewidths=0,
            label=cname,
        )

    label_order = np.argsort(-m)[:label_top_k]
    for i in label_order:
        i = int(i)
        ax.annotate(
            vocab[i],
            xy=(j_dot_m[i], abs_j_dot_m[i]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=8, color="#222",
        )

    ax.set_xlabel("J·m  (signed net coupling to the meta)")
    ax.set_ylabel("|J|·m  (popularity-weighted total coupling magnitude)")
    ax.set_title("Feature classification")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
