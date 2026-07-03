"""Offline precompute pipeline for the static web build.

Fits a single PL inverse-Ising model per invocation and serializes the
artifacts the JS client needs into `web/public/models/<slug>/`:

    meta.json         — vocab, sites, site_of, tracks, track_values,
                        scalars, fit hyperparams (schema v3, factored)
    J.bin             — float32 lower triangle, V*(V-1)/2 entries
                        ordering: [J[i,j] for i in range(1,V) for j in range(i)]
    h.bin             — float32, V entries
    m.bin             — float32, V entries
    team_counts.json  — sorted-index "-"-joined roster keys -> int count

Run locally; inspect artifacts; commit. Not invoked from CI per the
static-deploy plan: humans should eyeball every model refresh before it
goes live.

Float32 is sufficient for J/h/m at our scale (values O(1e-3..5),
regularized, no accumulating round-off). Lower-triangle packing halves
J's bytes; the JS loader reconstructs the symmetric matrix on init.

One artifact per (regulation, type) cell. `type` is a product-tier build recipe
(today only "standard": a species+item Boltzmann fit with the constants.py
defaults), applied uniformly across regulations. The slug is derived as
`reg-<regulation>` for the standard tier.

Usage:
    # Build one model (standard type implied):
    python precompute.py --build --regulation M-A
    python precompute.py --build --regulation M-B --new

    # Override the fit recipe if needed (advanced/dev only):
    python precompute.py --build --regulation M-A --lambda 5.0 --bz-iters 2000

    # Rebuild every committed model from its stored meta.json, then refresh manifest:
    python precompute.py --recompute

    # Generate manifest after models are built:
    python precompute.py --generate-manifest
    python precompute.py --generate-manifest --default-model reg-m-b
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.typing import NDArray

from k2dex.constants import (
    BOLTZMANN_AVG_LAST,
    BOLTZMANN_LR,
    BOLTZMANN_LR_FINAL,
    BOLTZMANN_N_BURN,
    BOLTZMANN_N_CHAINS,
    BOLTZMANN_N_ITERS,
    BOLTZMANN_N_SWEEPS,
    BOLTZMANN_N_TEMPS,
    BOLTZMANN_REG,
    BOLTZMANN_REG_LAMBDA,
    BOLTZMANN_SEED,
    BOLTZMANN_SUPPORT_MIN_COUNT,
    BOLTZMANN_SWAP_INTERVAL,
    BOLTZMANN_T_MAX,
    CURRENT_REGULATION,
    IN_PERSON_WEIGHT,
    PHASE2_MIN_TEAM_COUNT,
    RECENCY_TAU_DAYS,
    SPECIES_ITEM_LR_LAMBDA,
    SPECIES_LR_LAMBDA,
    TEAM_SIZE,
)
from k2dex.loaders import build_species_item_model, build_species_model

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "web" / "public" / "models"

MODEL_BUILDERS = {
    "species": build_species_model,
    "species_item": build_species_item_model,
}

DEFAULT_LAMBDA = {
    "species": SPECIES_LR_LAMBDA,
    "species_item": SPECIES_ITEM_LR_LAMBDA,
}

# Product-tier model types. A closed enum: each names a build *recipe* applied
# uniformly across every regulation, orthogonal to the regulation (= which sites
# and tracks are legal). Today only "standard" exists; a monetized "pro" tier may
# join later. This is distinct from the retired feature-dimension --type (species
# vs species_item), which the factored schema + attribute toggle made obsolete:
# the standard recipe is always the species+item build, and the species-only
# experience is a UI toggle, not a separate artifact.
PRODUCT_TYPES = ("standard",)

# Description of each type, shared across regulations. Surfaced in the manifest's
# `types` block (per-model provenance is derived from each meta.json instead).
TYPE_DESCRIPTIONS: dict[str, str] = {
    "standard": (
        "Fit on all recorded tournament rosters, weighting recent teams and "
        "those brought to in-person events. The recommended model."
    ),
}

# The builder + fit method each product type resolves to. The standard tier is a
# moment-matching Boltzmann fit on species+item features (see the plan: on-the-fly
# track marginalization is only faithful when the model's marginals match the data).
TYPE_BUILDER = {"standard": "species_item"}
TYPE_METHOD = {"standard": "boltzmann"}


def model_slug(regulation: str, product_type: str) -> str:
    """Artifact id/slug for a (regulation, type) cell.

    'reg-<regulation>' for the standard tier; 'reg-<regulation>-<type>' for any
    future tier, so the standard model keeps the clean per-regulation slug.
    """
    base = f"reg-{slugify(regulation)}"
    return base if product_type == "standard" else f"{base}-{slugify(product_type)}"


def default_boltzmann_opts() -> dict:
    """The standard Boltzmann recipe, read from the BOLTZMANN_* constants (the
    single source of truth). CLI --bz-* flags override individual knobs."""
    return {
        "n_iters": BOLTZMANN_N_ITERS,
        "lr": BOLTZMANN_LR,
        "lr_final": BOLTZMANN_LR_FINAL,
        "avg_last": BOLTZMANN_AVG_LAST,
        "n_chains": BOLTZMANN_N_CHAINS,
        "n_sweeps": BOLTZMANN_N_SWEEPS,
        "n_burn": BOLTZMANN_N_BURN,
        "n_temps": BOLTZMANN_N_TEMPS,
        "t_max": BOLTZMANN_T_MAX,
        "swap_interval": BOLTZMANN_SWAP_INTERVAL,
        "reg": BOLTZMANN_REG,
        "reg_lambda": BOLTZMANN_REG_LAMBDA,
        "seed": BOLTZMANN_SEED,
        "support_min_count": BOLTZMANN_SUPPORT_MIN_COUNT,
    }


def slugify(display_name: str) -> str:
    """Convert a human-readable display name to a URL-safe slug.

    Lowercases, replaces non-alphanumeric characters with hyphens,
    collapses runs, and strips leading/trailing hyphens.
    """
    s = display_name.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    if not s:
        raise ValueError(f"display name produces empty slug: {display_name!r}")
    return s


def pack_lower_triangle(J: NDArray[np.floating]) -> NDArray[np.float32]:
    """Extract J's strict lower triangle in row-major order as a flat
    float32 array of length V*(V-1)/2.

    Ordering: [J[1,0], J[2,0], J[2,1], J[3,0], J[3,1], J[3,2], ...].
    Equivalent to `J[np.tril_indices(V, k=-1)]`, which `np.tril_indices`
    returns in exactly this order.
    """
    V = J.shape[0]
    rows, cols = np.tril_indices(V, k=-1)
    return J[rows, cols].astype(np.float32)


def serialize_team_counts(
    team_counts: Counter,
    vocab: list[str],
) -> dict[str, int]:
    """Project frozenset[str] keys onto sorted-index lists, joined by "-".

    e.g. team {vocab[5], vocab[12], vocab[33], vocab[87], vocab[102], vocab[134]}
    -> "5-12-33-87-102-134".

    Skips rosters with any out-of-vocab member (defensive; the Phase 3
    builder already filters these, the Phase 2 builder doesn't filter
    explicitly but every roster member is by construction in vocab when
    the corpus indices are right).
    """
    name_to_idx = {name: i for i, name in enumerate(vocab)}
    out: dict[str, int] = {}
    skipped = 0
    for roster, count in team_counts.items():
        try:
            idxs = sorted(name_to_idx[name] for name in roster)
        except KeyError:
            skipped += 1
            continue
        out["-".join(str(i) for i in idxs)] = int(count)
    if skipped:
        print(f"  warn: skipped {skipped} rosters with out-of-vocab members")
    return out


def write_model(
    slug: str,
    display_name: str,
    regulation: str,
    builder_type: str,
    lam: float,
    out_dir: Path,
    *,
    product_type: str = "standard",
    description: str | None = None,
    min_team_count: int = PHASE2_MIN_TEAM_COUNT,
    recency_tau: float | None = RECENCY_TAU_DAYS,
    in_person_weight: float = IN_PERSON_WEIGHT,
    skip_team_counts: bool = False,
    force: bool = False,
    is_new: bool = False,
    prior_regulation: str | None = None,
    intercept_prior_weight: float = 1.0,
    method: str = "pseudo_likelihood",
    boltzmann_opts: dict | None = None,
) -> None:
    model_dir = out_dir / slug
    if model_dir.exists() and not force:
        print(f"error: {model_dir} already exists. Use --force to overwrite.")
        sys.exit(1)

    print(f"\n=== Building model: {slug} ===")
    print(f"  display_name: {display_name}")
    if description:
        print(f"  description: {description}")
    print(f"  regulation: {regulation}")
    print(f"  type: {product_type} (builder: {builder_type})")
    print(f"  method: {method}")
    print(f"  lambda: {lam}")
    if method == "boltzmann" and boltzmann_opts:
        print(f"  boltzmann: {boltzmann_opts}")
    print(f"  recency_tau_days: {recency_tau if recency_tau is not None else '(no decay)'}")
    print(f"  in_person_weight: {in_person_weight}")
    if prior_regulation is not None:
        print(f"  prior_regulation: {prior_regulation} "
              f"(intercept_prior_weight={intercept_prior_weight})")

    builder = MODEL_BUILDERS[builder_type]
    vocab, m, J, h, team_counts, species_of, item_of, latest_date = builder(
        regulation=regulation,
        min_team_count=min_team_count,
        lam=lam,
        recency_tau=recency_tau,
        in_person_multiplier=in_person_weight,
        prior_regulation=prior_regulation,
        intercept_prior_weight=intercept_prior_weight,
        method=method,
        boltzmann_opts=boltzmann_opts,
    )
    V = len(vocab)
    n_teams = int(sum(team_counts.values()))
    feature_dimensions = 1 if all(i is None for i in item_of) else 2
    print(f"  V = {V}, corpus teams = {n_teams:,}, latest tournament = {latest_date}")

    # Factored (sites + tracks) schema derivation. Sites are the distinct
    # species in first-appearance (vocab) order; site_of maps each feature to
    # its site. A species+item model carries one "item" track (unique per team);
    # a species-only model carries no tracks (each site has a single feature).
    # site_features is intentionally NOT stored -- it is a pure projection of
    # site_of that both loaders and the TS sampler rederive.
    sites: list[str] = []
    site_index: dict[str, int] = {}
    for sp in species_of:
        if sp not in site_index:
            site_index[sp] = len(sites)
            sites.append(sp)
    site_of = [site_index[sp] for sp in species_of]
    if feature_dimensions == 2:
        tracks = [{"name": "item", "unique": True}]
        track_values: list[list[str | None]] = [[it] for it in item_of]
    else:
        tracks = []
        track_values = [[] for _ in vocab]

    model_dir.mkdir(parents=True, exist_ok=True)

    j_flat = pack_lower_triangle(J)
    j_flat.tofile(model_dir / "J.bin")
    h.astype(np.float32).tofile(model_dir / "h.bin")
    m.astype(np.float32).tofile(model_dir / "m.bin")
    print(f"  J.bin: {j_flat.nbytes:,} bytes ({V}*{V-1}/2 = {len(j_flat):,} float32)")
    print(f"  h.bin: {V * 4:,} bytes")
    print(f"  m.bin: {V * 4:,} bytes")

    meta: dict = {
        "id": slug,
        "display_name": display_name,
        "regulation": regulation,
        "type": product_type,
        "feature_dimensions": feature_dimensions,
        "latest_tournament_date": latest_date,
        "V": V,
        "team_size": TEAM_SIZE,
        "n_corpus_teams": n_teams,
        "vocab": vocab,
        "sites": sites,
        "site_of": site_of,
        "tracks": tracks,
        "track_values": track_values,
        "fit": {
            "method": method,
            "lambda": lam,
            "min_team_count": min_team_count,
            "recency_tau_days": recency_tau,
            "in_person_weight": in_person_weight,
            **(
                {
                    "prior_regulation": prior_regulation,
                    "intercept_prior_weight": intercept_prior_weight,
                }
                if prior_regulation is not None
                else {}
            ),
            **(
                {"boltzmann": boltzmann_opts}
                if method == "boltzmann" and boltzmann_opts
                else {}
            ),
        },
        "schema_version": 3,
    }
    if description:
        meta["description"] = description
    if is_new:
        meta["new"] = True
    with open(model_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=None, separators=(",", ":"))
    print(f"  meta.json: {(model_dir / 'meta.json').stat().st_size:,} bytes")

    if skip_team_counts:
        print("  (skip-team-counts: not writing team_counts.json)")
        return

    tc_serialized = serialize_team_counts(team_counts, vocab)
    with open(model_dir / "team_counts.json", "w") as f:
        json.dump(tc_serialized, f, indent=None, separators=(",", ":"))
    tc_path = model_dir / "team_counts.json"
    print(f"  team_counts.json: {tc_path.stat().st_size:,} bytes "
          f"({len(tc_serialized):,} unique rosters)")


def generate_manifest(out_dir: Path, *, default_model: str | None = None) -> None:
    """Scan model directories and write manifest.json.

    Manifest schema v3 is a (regulation, type) grid: one entry per model, each
    tagged with its regulation and product `type`. Descriptions live once per
    type in a shared `types` block (not per model); everything else in a model
    entry is derived provenance read straight from its meta.json.
    """
    models = []
    types_present: set[str] = set()
    for meta_path in sorted(out_dir.glob("*/meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        slug = meta.get("id", meta.get("name", meta_path.parent.name))
        product_type = meta.get("type", "standard")
        types_present.add(product_type)
        entry: dict = {
            "id": slug,
            "display_name": meta.get("display_name", slug),
            "regulation": meta.get("regulation", ""),
            "type": product_type,
            "feature_dimensions": meta.get("feature_dimensions", 1),
            "V": meta["V"],
            "n_corpus_teams": meta["n_corpus_teams"],
            "latest_tournament_date": meta.get("latest_tournament_date", ""),
            "team_size": meta.get("team_size", TEAM_SIZE),
            "tracks": meta.get("tracks", []),
        }
        if meta.get("new"):
            entry["new"] = True
        models.append(entry)

    if not models:
        print(f"No models found in {out_dir}")
        return

    if default_model is not None and not any(m["id"] == default_model for m in models):
        valid = ", ".join(m["id"] for m in models)
        print(f"error: --default-model {default_model!r} not found. Valid slugs: {valid}")
        sys.exit(1)
    # Default to the current regulation's standard model when unspecified.
    if default_model is not None:
        resolved_default = default_model
    else:
        preferred = model_slug(CURRENT_REGULATION, "standard")
        resolved_default = preferred if any(m["id"] == preferred for m in models) else models[0]["id"]

    models.sort(key=lambda m: m["id"] != resolved_default)

    types = {
        t: {"description": TYPE_DESCRIPTIONS[t]}
        for t in sorted(types_present)
        if t in TYPE_DESCRIPTIONS
    }

    manifest = {
        "schema_version": 3,
        "default_model": resolved_default,
        "types": types,
        "models": models,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {manifest_path} ({len(models)} model(s))")
    for m in models:
        default_marker = " [default]" if m["id"] == resolved_default else ""
        print(f"  {m['id']}: V={m['V']}, teams={m['n_corpus_teams']:,}, "
              f"regulation={m['regulation']}{default_marker}")


def recompute_all(
    out_dir: Path,
    *,
    default_model: str | None = None,
    regulation_filter: str | None = None,
) -> None:
    """Rebuild models in `out_dir` from their stored `meta.json` parameters,
    then refresh the manifest.

    Used after a change to the fit (e.g. a fitter swap or a constant bump) to
    regenerate committed artifacts without re-specifying each model's CLI flags
    by hand. Each model is rebuilt with exactly the parameters it was last built
    with: lambda, weighting knobs, min-team-count, warm-start prior, description
    and 'new' badge all come from its `fit` block. `team_counts` are recomputed
    only for models that already had them.

    If `regulation_filter` is set, only models whose stored regulation matches
    it are rebuilt; the manifest is still refreshed for all models.
    """
    meta_paths = sorted(out_dir.glob("*/meta.json"))
    if not meta_paths:
        print(f"No models found in {out_dir}")
        return

    type_by_dim = {1: "species", 2: "species_item"}
    print(f"Output root: {out_dir}")
    if regulation_filter is not None:
        print(f"Regulation filter: {regulation_filter!r} (skipping others)\n")
    else:
        print(f"Recomputing {len(meta_paths)} model(s) from stored parameters.\n")
    for meta_path in meta_paths:
        with open(meta_path) as f:
            meta = json.load(f)
        reg = meta.get("regulation", CURRENT_REGULATION)
        if regulation_filter is not None and reg != regulation_filter:
            print(f"  skip {meta_path.parent.name}: regulation {reg!r} != {regulation_filter!r}")
            continue
        builder_type = type_by_dim.get(meta.get("feature_dimensions", 1))
        if builder_type is None:
            print(f"  skip {meta_path.parent.name}: unknown feature_dimensions")
            continue
        fit = meta.get("fit", {})
        slug = meta.get("id", meta_path.parent.name)
        has_team_counts = (meta_path.parent / "team_counts.json").exists()
        write_model(
            slug=slug,
            display_name=meta.get("display_name", slug),
            regulation=meta.get("regulation", CURRENT_REGULATION),
            builder_type=builder_type,
            lam=fit.get("lambda", DEFAULT_LAMBDA[builder_type]),
            out_dir=out_dir,
            product_type=meta.get("type", "standard"),
            description=meta.get("description"),
            min_team_count=fit.get("min_team_count", PHASE2_MIN_TEAM_COUNT),
            recency_tau=fit.get("recency_tau_days", RECENCY_TAU_DAYS),
            in_person_weight=fit.get("in_person_weight", IN_PERSON_WEIGHT),
            skip_team_counts=not has_team_counts,
            force=True,
            is_new=bool(meta.get("new", False)),
            prior_regulation=fit.get("prior_regulation"),
            intercept_prior_weight=fit.get("intercept_prior_weight", 1.0),
            method=fit.get("method", "pseudo_likelihood"),
            boltzmann_opts=fit.get("boltzmann"),
        )

    # V / n_corpus_teams can move, so refresh the manifest. Preserve the
    # existing default unless one was passed explicitly. Drop a stale default
    # (e.g. a retired split-model id after the schema collapse) so
    # generate_manifest re-resolves it instead of erroring.
    if default_model is None:
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                stored_default = json.load(f).get("default_model")
            if stored_default and (out_dir / stored_default / "meta.json").exists():
                default_model = stored_default
    print()
    generate_manifest(out_dir, default_model=default_model)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build PL inverse-Ising model artifacts for the static webapp.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output root directory (default: {DEFAULT_OUT_DIR})",
    )

    group = parser.add_argument_group("model build (one model per invocation)")
    group.add_argument(
        "--build",
        action="store_true",
        help="Build one model for --regulation / --type. The slug is derived from "
             "the (regulation, type) cell (e.g. 'reg-m-a'); no --display-name needed.",
    )
    group.add_argument(
        "--display-name",
        default=None,
        help="Optional display-name override (default: derived as 'Reg. <regulation>'). "
             "Does not affect the slug, which comes from (regulation, type).",
    )
    group.add_argument(
        "--regulation",
        default=CURRENT_REGULATION,
        help=f"Regulation string for tournament filtering (default: {CURRENT_REGULATION}).",
    )
    group.add_argument(
        "--type",
        choices=list(PRODUCT_TYPES),
        dest="product_type",
        default="standard",
        help="Product-tier model type (default: standard). A build recipe applied "
             "uniformly across regulations, distinct from the retired species vs "
             "species_item feature-dimension meaning.",
    )
    group.add_argument(
        "--lambda",
        type=float,
        dest="lam",
        default=None,
        help="L2 regularization strength (default: 10.0 for species, 1.0 for species_item). "
             "Converted to the logistic inverse-strength C = 1/lambda internally. "
             "For --method boltzmann this is the PL warm-start's lambda.",
    )
    group.add_argument(
        "--method",
        choices=["pseudo_likelihood", "boltzmann"],
        default=None,
        help="Fit method override. Defaults to the product type's method "
             "(standard -> boltzmann). 'boltzmann' warm-starts from the PL fit and "
             "refines it by constrained-MaxEnt moment matching. Recorded in meta.json:fit.",
    )
    group.add_argument(
        "--tau",
        type=float,
        dest="recency_tau",
        default=RECENCY_TAU_DAYS,
        help="Recency decay timescale in days for per-team fit weights "
             "(default: no decay). Recorded in meta.json:fit.",
    )
    group.add_argument(
        "--in-person-weight",
        type=float,
        dest="in_person_weight",
        default=IN_PERSON_WEIGHT,
        help=f"Fit-weight multiplier on in-person teams (default: {IN_PERSON_WEIGHT}). "
             "Recorded in meta.json:fit.",
    )
    group.add_argument(
        "--min-team-count",
        type=int,
        default=PHASE2_MIN_TEAM_COUNT,
        help=f"Vocab cutoff: feature must appear in >= N teams (default: {PHASE2_MIN_TEAM_COUNT}).",
    )
    group.add_argument(
        "--prior-regulation",
        default=None,
        help="Warm-start the fit from another regulation's model (a legality "
             "superset). Its full vocab is folded in and its (J, h) re-centers "
             "the L2 penalty, so thin features relax toward the prior instead "
             "of zero. Recorded in meta.json:fit.",
    )
    group.add_argument(
        "--intercept-prior-weight",
        type=float,
        default=1.0,
        help="With --prior-regulation, how hard the bias h is pulled toward "
             "the prior, as a multiple of the coupling-penalty strength "
             "(1.0 = same as couplings; 0.0 = bias free). Default: 1.0.",
    )
    group.add_argument(
        "--skip-team-counts",
        action="store_true",
        help="Skip team_counts.json (faster iteration during dev).",
    )
    group.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing model directory.",
    )
    group.add_argument(
        "--new",
        action="store_true",
        dest="is_new",
        help="Mark this model as new in the manifest (shows a 'New' badge in the webapp).",
    )

    bz_group = parser.add_argument_group(
        "boltzmann options (only used with --method boltzmann)")
    bz_group.add_argument("--bz-iters", type=int, default=BOLTZMANN_N_ITERS,
                          help=f"Gradient steps (default: {BOLTZMANN_N_ITERS}).")
    bz_group.add_argument("--bz-lr", type=float, default=BOLTZMANN_LR,
                          help=f"Adam learning rate (default: {BOLTZMANN_LR}).")
    bz_group.add_argument("--bz-lr-final", type=float, default=BOLTZMANN_LR_FINAL,
                          help="Cosine-anneal the step size to this value (shrinks the "
                               "stochastic-gradient noise ball -- the dominant error). "
                               f"Default: {BOLTZMANN_LR_FINAL} (BOLTZMANN_LR/20). Set "
                               "equal to --bz-lr to disable decay.")
    bz_group.add_argument("--bz-avg-last", type=int, default=BOLTZMANN_AVG_LAST,
                          help="Polyak iterate-averaging window: average (J,h) over the "
                               "last N steps (0 = off; alternative to lr decay, comparable).")
    bz_group.add_argument("--bz-chains", type=int, default=BOLTZMANN_N_CHAINS,
                          help=f"PCD chain-bank size (default: {BOLTZMANN_N_CHAINS}).")
    bz_group.add_argument("--bz-sweeps", type=int, default=BOLTZMANN_N_SWEEPS,
                          help=f"Swaps per chain per gradient step (default: {BOLTZMANN_N_SWEEPS}).")
    bz_group.add_argument("--bz-n-burn", type=int, default=BOLTZMANN_N_BURN,
                          help=f"Initial bank-mixing swaps per chain (default: {BOLTZMANN_N_BURN}).")
    bz_group.add_argument("--bz-temps", type=int, default=BOLTZMANN_N_TEMPS,
                          help=f"Parallel-tempering replicas per chain (default: "
                               f"{BOLTZMANN_N_TEMPS} = off; single-T PCD sufficed on the "
                               "M-A corpus). >1 only for genuinely multimodal models.")
    bz_group.add_argument("--bz-t-max", type=float, default=BOLTZMANN_T_MAX,
                          help=f"Hot temperature for tempering (default: {BOLTZMANN_T_MAX}; "
                               "unused if --bz-temps 1).")
    bz_group.add_argument("--bz-swap-interval", type=int, default=BOLTZMANN_SWAP_INTERVAL,
                          help=f"Sweeps between replica-exchange attempts (default: "
                               f"{BOLTZMANN_SWAP_INTERVAL}).")
    bz_group.add_argument("--bz-reg", choices=["l1", "l2"], default=BOLTZMANN_REG,
                          help=f"Regularizer toward zero (default: {BOLTZMANN_REG}).")
    bz_group.add_argument("--bz-reg-lambda", type=float, default=BOLTZMANN_REG_LAMBDA,
                          help=f"Regularization strength (default: {BOLTZMANN_REG_LAMBDA}).")
    bz_group.add_argument("--bz-support-min-count", type=int,
                          default=BOLTZMANN_SUPPORT_MIN_COUNT,
                          help="Only fit couplings for feature pairs that co-occur >= N "
                               "times in the corpus (freezes the rest at the PL warm-start). "
                               f"Default: {BOLTZMANN_SUPPORT_MIN_COUNT}. Pass 0 to fit all pairs.")
    bz_group.add_argument("--bz-seed", type=int, default=BOLTZMANN_SEED,
                          help=f"RNG seed for the PCD sampler (default: {BOLTZMANN_SEED}).")

    manifest_group = parser.add_argument_group("manifest generation")
    manifest_group.add_argument(
        "--generate-manifest",
        action="store_true",
        help="Scan model directories and write manifest.json. No model build.",
    )
    manifest_group.add_argument(
        "--recompute",
        nargs="?",
        const=True,
        default=None,
        metavar="REGULATION",
        help="Rebuild existing models from their stored meta.json parameters and "
             "refresh the manifest. No per-model flags needed. Optionally pass a "
             "regulation string (e.g. 'M-A') to rebuild only models for that "
             "regulation; omit the value to rebuild all.",
    )
    manifest_group.add_argument(
        "--default-model",
        help="Model slug to mark as default in the manifest.",
    )
    args = parser.parse_args()
    out_dir = args.out.resolve()

    if args.generate_manifest:
        generate_manifest(out_dir, default_model=args.default_model)
        return 0

    if args.recompute is not None:
        regulation_filter = None if args.recompute is True else args.recompute
        recompute_all(out_dir, default_model=args.default_model, regulation_filter=regulation_filter)
        print("\nDone. Inspect artifacts before committing.")
        return 0

    if not args.build:
        parser.error(
            "no action given. Use --build (single model), --recompute, or "
            "--generate-manifest.")

    product_type = args.product_type
    builder_type = TYPE_BUILDER[product_type]
    method = args.method if args.method is not None else TYPE_METHOD[product_type]
    slug = model_slug(args.regulation, product_type)
    display_name = args.display_name or f"Reg. {args.regulation}"
    lam = args.lam if args.lam is not None else DEFAULT_LAMBDA[builder_type]

    boltzmann_opts: dict | None = None
    if method == "boltzmann":
        boltzmann_opts = {
            "n_iters": args.bz_iters,
            "lr": args.bz_lr,
            # Cosine-decay the step size by default (kills the noise ball, ~7x
            # lower moment bias); set --bz-lr-final == --bz-lr to disable.
            "lr_final": args.bz_lr_final,
            "avg_last": args.bz_avg_last,
            "n_chains": args.bz_chains,
            "n_sweeps": args.bz_sweeps,
            "n_burn": args.bz_n_burn,
            "n_temps": args.bz_temps,
            "t_max": args.bz_t_max,
            "swap_interval": args.bz_swap_interval,
            "reg": args.bz_reg,
            "reg_lambda": args.bz_reg_lambda,
            "seed": args.bz_seed,
        }
        if args.bz_support_min_count and args.bz_support_min_count > 0:
            boltzmann_opts["support_min_count"] = args.bz_support_min_count

    print(f"Output root: {out_dir}")
    write_model(
        slug=slug,
        display_name=display_name,
        regulation=args.regulation,
        builder_type=builder_type,
        lam=lam,
        out_dir=out_dir,
        product_type=product_type,
        min_team_count=args.min_team_count,
        recency_tau=args.recency_tau,
        in_person_weight=args.in_person_weight,
        skip_team_counts=args.skip_team_counts,
        force=args.force,
        is_new=args.is_new,
        prior_regulation=args.prior_regulation,
        intercept_prior_weight=args.intercept_prior_weight,
        method=method,
        boltzmann_opts=boltzmann_opts,
    )

    print("\nDone. Inspect artifacts before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
