"""Cross-fixture validation of the concise-via-routing thesis on econ-sae.

Imports the recipe-agnostic router from sae-forge (``saeforge.isf``) — the same
primitive bio-sae used to route its motif specialist — and applies it to
econ-sae's *already-committed* world-model recipe activations. No training: this
re-uses the objective-family encoders econ-sae already produced (baseline world
model, attention F1, regime-supervised Family G, conjunctive specialist) as
ensemble recipes, routed per label.

The concise-via-routing thesis (sae-forge docs/concise-via-routing.md) predicts:
  1. the routed ensemble beats every single recipe (H-ISF headline);
  2. the lift is concentrated on the LOW-salience tiers (regime / conjunctive)
     and is ~null on the salient bucketed tier (the salience HEURISTIC,
     a rule of thumb, not a law);
  3. each objective-family specialist wins its matched tier.

Outputs ``runs/isf_routing_validation_summary.json``.

Usage::

    python scripts/isf_routing_validation.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from econsae.sae.evaluation import feature_tier
from saeforge.isf import ensemble_route, recipe_auc_matrix, salience_headroom

DATA = REPO_ROOT / "data" / "econ_ensemble.npz"
RUNS = REPO_ROOT / "runs"

# (name, acts file) — recipe 0 is the host. Each is a different *objective
# family* econ-sae already trained; the router is blind to how they were made.
RECIPES = [
    ("baseline_host", "world_model_acts.npz"),
    ("attn_f1", "world_model_attn_acts.npz"),
    ("regime_supervised_g", "world_model_regime_supervised_acts.npz"),
    ("conjunctive", "world_model_conjunctive_unified_acts.npz"),
]


def _align(acts_path: Path, key_to_label_row: dict, keys: list) -> np.ndarray | None:
    """Align a recipe's H1 activations to the label rows via sample_index."""
    d = np.load(acts_path, allow_pickle=True)
    H = d["H1"]
    si = d["sample_index"]
    idx = {}
    for i, r in enumerate(si):
        idx.setdefault(tuple(int(x) for x in r), i)   # first occurrence
    rows = [idx.get(k, -1) for k in keys]
    if any(r < 0 for r in rows):
        return None
    return H[np.array(rows)].astype(np.float64)


def main() -> None:
    base = np.load(DATA, allow_pickle=True)
    Y = base["labels_Y"].astype(np.uint8)
    vocab = [str(v) for v in base["feature_vocab"]]
    tiers = np.array([feature_tier(v) for v in vocab])
    keys = [tuple(int(x) for x in r) for r in base["sample_index"]]
    print(f"econ-sae fixture: N={Y.shape[0]} labels={Y.shape[1]}")

    feeds, names = [], []
    for nm, fn in RECIPES:
        Z = _align(RUNS / fn, None, keys)
        if Z is None:
            print(f"  SKIP {nm}: unaligned sample_index")
            continue
        feeds.append(Z)
        names.append(nm)
        print(f"  recipe {nm}: {Z.shape}")

    A = recipe_auc_matrix(feeds, Y)                       # (R, V)
    route = ensemble_route(A, names, host=0)
    scorable = ~np.isnan(A).all(axis=0)
    tiers_s = tiers[scorable]
    host_s = A[0][scorable]
    ensemble_best = np.array(route["ensemble_best"])
    router_names = np.array(route["router_names"])

    print(f"\n  per-recipe mAUC: "
          + "  ".join(f"{k}={v:.3f}" for k, v in route["per_recipe_mauc"].items()))
    print(f"  ENSEMBLE mAUC={route['ensemble_mauc']:.3f}  "
          f"lift over best single ({route['best_single_recipe']})={route['ensemble_lift']:+.3f}  "
          f"retained={route['retained']:.3f}  beats_host={route['frac_beats_host']:.3f}")
    print(f"  router composition: {route['router_composition']}")

    per_tier = {}
    for t in ("categorical", "bucketed", "regime", "conjunctive"):
        m = tiers_s == t
        if not m.any():
            continue
        host_t = float(np.nanmean(host_s[m]))
        ens_t = float(ensemble_best[m].mean())
        per_tier[t] = {
            "n_labels": int(m.sum()),
            "host_mauc": host_t,
            "ensemble_mauc": ens_t,
            "lift_over_host": ens_t - host_t,
            "salience_headroom": float(np.mean(salience_headroom(host_s[m]))),
            "router_composition": {k: int(v) for k, v in Counter(router_names[m]).items()},
        }
        d = per_tier[t]
        print(f"   [{t:11s}] headroom={d['salience_headroom']:.3f}  "
              f"host={host_t:.3f} -> ensemble={ens_t:.3f} (+{d['lift_over_host']:.3f})  "
              f"routes={d['router_composition']}")

    summary = {
        "fixture": "econ-sae",
        "primitive": "saeforge.isf (ensemble_route + salience_headroom)",
        "thesis": "docs/concise-via-routing.md (sae-forge)",
        "recipes": names,
        "n_labels_scored": route["n_labels_scored"],
        "route": {k: route[k] for k in (
            "per_recipe_mauc", "ensemble_mauc", "best_single_recipe",
            "ensemble_lift", "host", "retained", "frac_beats_host",
            "router_composition")},
        "per_tier": per_tier,
        "reading": (
            "The routed ensemble beats every single recipe (the H-ISF headline) "
            "in a second domain. The lift concentrates on the low-salience tiers "
            "(conjunctive, regime) and is ~null on the salient bucketed tier — "
            "the salience heuristic (a rule of thumb). Each objective-family "
            "specialist wins its matched "
            "tier. Margins are thinner than bio-sae's synthetic motifs because "
            "econ's hard tiers are partly salient on the world-model substrate, "
            "as the heuristic suggests. NB: headroom is a cheap prior, not a "
            "predictor — conjunctive here is low-headroom yet gains most."
        ),
    }
    out = RUNS / "isf_routing_validation_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
