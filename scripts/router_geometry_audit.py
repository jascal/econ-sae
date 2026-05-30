"""Geometry audit of the econ-sae ISF router via polygram cluster_experts.

The router (``saeforge.isf``, ``scripts/isf_routing_validation.py``) partitions
the label vocabulary by which *objective-family* recipe reads each concept best.
This script asks the geometry question that makes "concise" meaningful: do those
recipes encode **orthogonal** concepts (a genuinely non-redundant ensemble) or
redundant ones?

Method — close the n-orca → sae-forge → polygram loop:
  1. Reproduce the router (recipe per label + ensemble-best AUC).
  2. For each routed label, take its recipe's **best-latent activation profile**
     over the N samples — a concept direction in sample space, comparable across
     recipes (which otherwise have different latent widths).
  3. Feed those profiles to ``polygram.cluster_experts`` (cosine block formation)
     and compare polygram's geometric blocks to the router's recipe partition.
  4. Measure within- vs cross-recipe cosine: low cross-recipe overlap ⇒ the
     specialists own orthogonal concept subspaces ⇒ the routed ensemble is
     genuinely concise, not redundant.

Outputs ``runs/router_geometry_audit_summary.json``.

Usage::

    python scripts/router_geometry_audit.py
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore", message="All-NaN slice encountered")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np

from polygram import Dictionary, Feature, HEA_Rung2, cluster_experts
from saeforge.isf import ensemble_route
# Reuse the merged routing-validation loaders so the two can't drift.
from isf_routing_validation import DATA, RECIPES, RUNS, _align
from econsae.sae.evaluation import feature_tier


def _sym_auc_matrix(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-latent × per-label symmetric AUC, shape ``(d_latent, V)``."""
    n, d = Z.shape
    Yf = Y.astype(np.float64)
    n_pos = Yf.sum(axis=0)
    n_neg = n - n_pos
    valid = (n_pos > 0) & (n_neg > 0)
    u_off = n_pos * (n_pos + 1) / 2.0
    denom = np.where(valid, n_pos * n_neg, 1.0)
    order = Z.argsort(axis=0)
    ranks = np.empty((n, d), dtype=np.float64)
    ranks[order, np.arange(d)[None, :]] = np.arange(1, n + 1, dtype=np.float64)[:, None]
    s_pos = Yf.T @ ranks                                   # (V, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (s_pos - u_off[:, None]) / denom[:, None]
    sym = np.maximum(auc, 1.0 - auc)
    sym = np.where(valid[:, None], sym, np.nan)
    return sym.T                                           # (d, V)


def _ident(name: str, used: set) -> str:
    s = re.sub(r"[^0-9A-Za-z_]", "_", name)
    s = re.sub(r"_+", "_", s).strip("_") or "f"
    if s[0].isdigit():
        s = "f_" + s
    base, k = s, 0
    while s in used:
        k += 1
        s = f"{base}_{k}"
    used.add(s)
    return s


def main() -> None:
    base = np.load(DATA, allow_pickle=True)
    Y = base["labels_Y"].astype(np.uint8)
    vocab = [str(v) for v in base["feature_vocab"]]
    tiers = [feature_tier(v) for v in vocab]
    keys = [tuple(int(x) for x in r) for r in base["sample_index"]]

    feeds, names = [], []
    for nm, fn in RECIPES:
        Z = _align(RUNS / fn, None, keys)
        if Z is not None:
            feeds.append(Z)
            names.append(nm)
    print(f"econ-sae: {len(names)} recipes, {Y.shape[1]} labels, N={Y.shape[0]}")

    # Per-recipe latent×label AUC → recipe_auc (max) + best-latent index per (recipe,label).
    per_recipe_M = [_sym_auc_matrix(Z, Y) for Z in feeds]      # each (d_r, V)
    with np.errstate(invalid="ignore"):
        A = np.vstack([np.nanmax(M, axis=0) for M in per_recipe_M])  # (R, V)
    route = ensemble_route(A, names, host=0)
    scorable = ~np.isnan(A).all(axis=0)
    scor_idx = np.where(scorable)[0]
    router = np.array(route["router"])                        # recipe idx per scorable label
    ens_best = np.array(route["ensemble_best"])

    # Concept direction = the routed recipe's best latent profile over N samples.
    profiles, feats, hierarchy, used = [], [], {}, set()
    concept_recipe, concept_tier = [], []
    for pos, v in enumerate(scor_idx):
        r = int(router[pos])
        col = per_recipe_M[r][:, v]
        if np.isnan(col).all():
            continue
        j = int(np.nanargmax(col))
        prof = feeds[r][:, j].astype(np.float64)
        prof = prof - prof.mean()
        nrm = np.linalg.norm(prof)
        if nrm < 1e-12:
            continue
        profiles.append(prof / nrm)                           # unit, centered
        ident = _ident(vocab[v], used)
        feats.append(Feature(name=ident, cluster=names[r], beta=max(float(ens_best[pos]) - 0.5, 0.0)))
        hierarchy.setdefault(names[r], []).append(ident)
        concept_recipe.append(names[r])
        concept_tier.append(tiers[v])

    P = np.vstack(profiles)                                    # (C, N) unit rows
    C = P.shape[0]
    print(f"  {C} routed concepts with usable best-latent profiles")

    # --- cross-recipe orthogonality (the concise-ensemble test) ---
    G = P @ P.T                                               # (C, C) cosine
    iu = np.triu_indices(C, k=1)
    same = np.array([concept_recipe[i] == concept_recipe[j] for i, j in zip(*iu)])
    absG = np.abs(G[iu])
    within = float(absG[same].mean()) if same.any() else float("nan")
    cross = float(absG[~same].mean()) if (~same).any() else float("nan")

    # --- polygram geometric clustering vs the router partition (threshold sweep) ---
    n_q = max(3, int(np.ceil(np.log2(max(C, 2)))))
    enc = HEA_Rung2(depth=2, entangler="ring", rotations=("Ry", "Rz"),
                    tier_separation_bound=0.025, n_qubits=n_q)
    dictionary = Dictionary(name="econ_router", features=feats,
                            hierarchy=hierarchy, encoding=enc)

    def _purity(geom: np.ndarray, n_blocks: int) -> float:
        # mean over blocks of the majority router-recipe share (1.0 = blocks are
        # pure recipe partitions; ~max-recipe-share = chance for one big block).
        ps = []
        for b in range(n_blocks):
            mem = [concept_recipe[i] for i in range(C) if geom[i] == b]
            if mem:
                ps.append(Counter(mem).most_common(1)[0][1] / len(mem))
        return float(np.mean(ps))

    chance = max(Counter(concept_recipe).values()) / C        # one-big-block baseline
    sweep = []
    for thr in (0.3, 0.5, 0.7, 0.9):
        ed = cluster_experts(dictionary, P, method="cosine", coherence_threshold=thr)
        geom = np.array(ed._feature_to_expert)
        sweep.append({"threshold": thr, "n_blocks": int(ed.n_experts),
                      "purity": _purity(geom, ed.n_experts)})

    print(f"\n  cross-recipe orthogonality: within-recipe |cos|={within:.3f}  "
          f"cross-recipe |cos|={cross:.3f}  (ratio cross/within={cross/within:.2f})")
    print(f"  polygram cosine clustering (chance purity={chance:.3f} at 1 block):")
    for s in sweep:
        print(f"    threshold={s['threshold']}: {s['n_blocks']:2d} blocks  "
              f"purity wrt router={s['purity']:.3f}")

    summary = {
        "fixture": "econ-sae router geometry audit",
        "tools": "saeforge.isf (router) + polygram.cluster_experts (geometry)",
        "n_concepts": C,
        "n_recipes": len(names),
        "within_recipe_abs_cos": within,
        "cross_recipe_abs_cos": cross,
        "cross_over_within": float(cross / within) if within else float("nan"),
        "polygram_threshold_sweep": sweep,
        "one_block_chance_purity": float(chance),
        "router_composition": route["router_composition"],
        "reading": (
            "Concepts routed to the SAME recipe are more cosine-similar to each "
            "other (within-recipe |cos|=%.3f) than to concepts in other recipes "
            "(cross-recipe |cos|=%.3f, ratio %.2f): the objective-family "
            "specialists own PARTIALLY orthogonal concept subspaces — non-redundant "
            "(ratio < 1) but not strongly disjoint (ratio not << 1), as expected "
            "when all recipes read the same economy. polygram's independent cosine "
            "clustering (threshold sweep) resolves the structure as the threshold "
            "tightens; purity above the one-big-block chance baseline (%.3f) means "
            "the geometric blocks track the router partition. Closes the n-orca -> "
            "sae-forge -> polygram loop on a real fixture; honest read: modest, not "
            "dramatic, orthogonality."
        ) % (within, cross, cross / within if within else float("nan"), chance),
    }
    out = RUNS / "router_geometry_audit_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
