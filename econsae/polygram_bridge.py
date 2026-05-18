"""Phase 3.1: convert econ-sae's full ground-truth feature vocabulary into
a polygram Dictionary and run interference / cancellation experiments.

Unlike the sm-sae 8-particle slice (MPSRung1 cap), we use HEA_Rung2 with
n_qubits sized to fit *all* features in the vocabulary --- no arbitrary
truncation. econ-sae's GT vocab is ~51 features (categorical + bucketed
+ conjunctive + regime), comfortably within 2^6 = 64 feature slots.

Beta scalar choice (analogous to sm-sae's electric charge):
  Per-feature best-recovered AUC minus 0.5. Easy features (categorical
  at AUC = 1.0) get beta = 0.5; hard, unrecovered features (regime at
  AUC ~ 0.55) get beta ~ 0.05. This is the "interpretability strength"
  of each feature in the SAE substrate.

Cluster: the feature tier (categorical / bucketed / conjunctive /
regime / other).

Experiments:
  1. Build a Dictionary from the full feature vocabulary.
  2. InterferenceSweep on a structurally-related pair.
  3. Cancellation across a hand-picked set of within-tier and
     cross-tier pairs that reveal econ-sae's compositional geometry.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Iterable

import numpy as np

from polygram import (
    Cancellation, Dictionary, Experiment, Feature, HEA_Rung2,
)

from econsae.sae.evaluation import feature_tier


OUT_DIR = "runs/polygram"


# ---------------------------------------------------------------------------
# Identifier conversion: turn GT feature names into polygram-valid
# Python-style identifiers. Reversible roundtrip is not required (we
# carry the original name alongside in `summary["feature_aucs"]`).
# ---------------------------------------------------------------------------
def to_identifier(gt_name: str) -> str:
    ident = gt_name
    ident = ident.replace("phase:", "phase_")
    ident = ident.replace("sector:", "sector_")
    ident = ident.replace("cohort:", "cohort_")
    ident = ident.replace("firm_sector:", "firm_")
    ident = ident.replace("txn_period_has:", "txn_")
    ident = ident.replace("shock_period_has:", "shock_")
    ident = ident.replace("debt_bucket:high", "debt_high")
    ident = ident.replace("mpc_bucket:high", "mpc_high")
    ident = ident.replace("cash_bucket:rich", "cash_rich")
    ident = ident.replace("cash_bucket:poor", "cash_poor")
    ident = ident.replace("inventory_bucket:stockout", "stockout")
    ident = ident.replace("leverage:high", "lev_high")
    ident = ident.replace("productivity:high", "prod_high")
    # Anything left: replace non-id characters with underscore
    ident = re.sub(r"[^0-9A-Za-z_]", "_", ident)
    # Collapse multiple underscores; trim
    ident = re.sub(r"_+", "_", ident).strip("_")
    # Identifiers must not start with a digit
    if ident and ident[0].isdigit():
        ident = "f_" + ident
    return ident


def required_qubits(n_features: int) -> int:
    """How many qubits an HEA_Rung2 encoding needs for n_features feature slots."""
    return max(3, math.ceil(math.log2(max(n_features, 2))))


# ---------------------------------------------------------------------------
def build_dictionary(feature_aucs: dict[str, float],
                     name: str = "econ_sae_full",
                     depth: int = 2,
                     entangler: str = "ring") -> Dictionary:
    """Build a polygram Dictionary from the FULL feature vocabulary.

    `feature_aucs` maps GT-feature name -> best-recovered AUC. All
    entries are included; no 8-feature truncation. Encoding is
    HEA_Rung2 with n_qubits sized to fit the vocabulary.
    """
    features: list[Feature] = []
    hierarchy: dict[str, list[str]] = {}
    name_to_ident: dict[str, str] = {}

    # Sort for deterministic ordering (helps reproducible results).
    for gt_name in sorted(feature_aucs.keys()):
        auc = float(feature_aucs[gt_name])
        beta = auc - 0.5
        cluster = feature_tier(gt_name) or "other"
        ident = to_identifier(gt_name)
        # De-dup identifier collisions (rare; suffix with index)
        suffix = 0
        base = ident
        while ident in name_to_ident.values():
            suffix += 1
            ident = f"{base}_{suffix}"
        name_to_ident[gt_name] = ident
        features.append(Feature(name=ident, cluster=cluster, beta=beta))
        hierarchy.setdefault(cluster, []).append(ident)

    n_q = required_qubits(len(features))
    encoding = HEA_Rung2(
        depth=depth, entangler=entangler,
        rotations=("Ry", "Rz"),
        tier_separation_bound=0.025,
        n_qubits=n_q,
    )
    return Dictionary(
        name=name, features=features, hierarchy=hierarchy, encoding=encoding,
    ), name_to_ident


# ---------------------------------------------------------------------------
def run_interference_sweep(dictionary: Dictionary,
                            target_pair: tuple[str, str],
                            knob: str,
                            label: str) -> dict:
    """Sweep one feature's phi from 0 to 2*pi; record target-pair overlap."""
    print(f"\n--- Interference sweep:  {knob} from 0 to 2pi  "
          f"(target: <{target_pair[0]} | {target_pair[1]}>) ---")
    out_path = os.path.join(OUT_DIR, f"interference_{label}")
    os.makedirs(out_path, exist_ok=True)
    experiment = Experiment(
        name=f"econ_sae_{label}_sweep",
        dictionary=dictionary,
        target_pair=target_pair,
        sweep={knob: np.linspace(0.0, 2 * np.pi, 60)},
        measures=["overlap", "gram_matrix", "schmidt_rank"],
        assertions=["hierarchical_ordering_preserved"],
    )
    experiment.materialize(out_path)
    result = experiment.run()
    overlaps = list(result.overlaps)
    print(f"  swept {len(overlaps)} phi values")
    print(f"  target overlap: min={min(overlaps):.4f}  "
          f"max={max(overlaps):.4f}  mean={float(np.mean(overlaps)):.4f}")
    csv_path = os.path.join(OUT_DIR, f"interference_{label}.csv")
    result.to_csv(csv_path)
    print(f"  wrote {csv_path}  and {out_path}/")
    return {
        "label": label, "pair": list(target_pair), "knob": knob,
        "overlap_min": float(min(overlaps)),
        "overlap_max": float(max(overlaps)),
        "overlap_mean": float(np.mean(overlaps)),
        "n_samples": len(overlaps),
    }


def run_cancellation(dictionary: Dictionary,
                      pair: tuple[str, str],
                      label: str) -> dict:
    """Drive a target pair's overlap toward zero with phase alone."""
    print(f"\n--- Cancellation:  drive |<{pair[0]}|{pair[1]}>|^2 to ~0 ---")
    cancel = Cancellation(
        dictionary=dictionary,
        target_pair=pair,
        tolerance=0.05,
        preserve_tiers=True,
        optimize={"method": "grid", "max_steps": 40},
    )
    result = cancel.run()
    eff = (None if result.cancellation_efficiency is None
           else float(result.cancellation_efficiency))
    print(f"  before={result.before_overlap:.4f}  after={result.after_overlap:.4f}  "
          f"floor={result.structural_floor:.4f}  "
          f"eff={'N/A' if eff is None else f'{eff:.2%}'}  met={result.tolerance_met}")
    out_path = os.path.join(OUT_DIR, f"cancellation_{label}")
    os.makedirs(out_path, exist_ok=True)
    result.materialize(out_path)
    return {
        "label": label, "pair": list(pair),
        "before_overlap": float(result.before_overlap),
        "after_overlap": float(result.after_overlap),
        "structural_floor": float(result.structural_floor),
        "cancellation_efficiency": eff,
        "tolerance_met": bool(result.tolerance_met),
        "n_evaluations": int(len(result.trajectory)),
    }


# ---------------------------------------------------------------------------
def select_pairs(name_to_ident: dict[str, str],
                  feature_aucs: dict[str, float]) -> list[tuple[str, str, str]]:
    """Hand-picked pairs that probe econ-sae's compositional structure.

    Falls back gracefully if any feature is missing from the vocab.
    Returns (a_ident, b_ident, label) tuples.
    """
    def ident(gt_name: str) -> str | None:
        return name_to_ident.get(gt_name)

    candidates = [
        # within-tier (categorical) -- different concepts entirely
        ("sector:bank",              "firm_sector:food",          "cat_cat_distant"),
        # within-tier (categorical) -- same axis, different value
        ("firm_sector:food",         "firm_sector:durables",      "cat_cat_same_axis"),
        # within-tier (bucketed)
        ("debt_bucket:high",         "mpc_bucket:high",           "buck_buck"),
        # within-tier (conjunctive)
        ("food_firm_low_inv",        "young_AND_indebted",        "conj_conj"),
        # cross-tier: categorical embedded in a related conjunctive
        ("firm_sector:food",         "food_firm_low_inv",         "cat_in_conj"),
        # cross-tier: categorical vs unrelated regime
        ("sector:household",         "phase:high_leverage",       "cat_vs_regime"),
        # within-tier (regime)
        ("phase:high_leverage",     "phase:fiscal_active",        "regime_regime"),
        # cross-tier: regime vs conjunctive
        ("phase:high_leverage",     "young_AND_indebted",         "regime_vs_conj"),
        # cohort-axis sanity check: two different cohorts
        ("cohort:young",             "cohort:retiree",            "cohort_cohort"),
        # shock features: monetary vs fiscal (both rare period-level impulses)
        ("shock_period_has:monetary","shock_period_has:fiscal",   "shock_shock"),
    ]
    pairs = []
    for a_gt, b_gt, label in candidates:
        a = ident(a_gt); b = ident(b_gt)
        if a is None or b is None:
            continue
        # Drop pairs where either feature has trivial (near-chance) recovery,
        # since their beta is ~0 and the experiment is uninformative.
        if (feature_aucs.get(a_gt, 0.5) - 0.5) < 0.05: continue
        if (feature_aucs.get(b_gt, 0.5) - 0.5) < 0.05: continue
        pairs.append((a, b, label))
    return pairs


# ---------------------------------------------------------------------------
def main(feature_aucs: dict[str, float] | None = None,
         out_dir: str = OUT_DIR) -> dict:
    """Build full-vocab dictionary, run sweep + cancellations."""
    global OUT_DIR
    OUT_DIR = out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    if feature_aucs is None:
        feature_aucs = _default_feature_aucs()

    print("=" * 78)
    print(f"Build polygram Dictionary from full econ-sae vocabulary "
          f"({len(feature_aucs)} features)")
    print("=" * 78)
    dictionary, name_to_ident = build_dictionary(feature_aucs)
    print(f"  Dictionary: {dictionary.name}")
    print(f"  Encoding:   HEA_Rung2(depth=2, n_qubits="
          f"{required_qubits(len(feature_aucs))})  "
          f"(2^{required_qubits(len(feature_aucs))} = "
          f"{2 ** required_qubits(len(feature_aucs))} feature slots)")
    print(f"  Tier breakdown:")
    for cluster, members in dictionary.hierarchy.items():
        print(f"    {cluster:<12s} {len(members):>3d} features")
    print(f"  Feature betas summary:")
    betas_by_cluster: dict[str, list[float]] = {}
    for f in dictionary.features:
        betas_by_cluster.setdefault(f.cluster, []).append(f.beta)
    for cluster, betas in betas_by_cluster.items():
        arr = np.array(betas)
        print(f"    {cluster:<12s} n={len(betas):>3d}  "
              f"beta: min={arr.min():+.3f}  median={np.median(arr):+.3f}  "
              f"max={arr.max():+.3f}")

    # Interference sweep on a structurally-related pair: firm_sector:food
    # is a CATEGORICAL feature subsumed by food_firm_low_inv (a
    # CONJUNCTIVE feature requiring firm_sector=food AND low inventory).
    # Sweeping the categorical's phase should modulate the conjunctive's
    # overlap with it.
    a_id = name_to_ident.get("firm_sector:food")
    b_id = name_to_ident.get("food_firm_low_inv")
    sweep_result = None
    if a_id and b_id:
        sweep_result = run_interference_sweep(
            dictionary,
            target_pair=(a_id, b_id),
            knob=f"{a_id}.phi",
            label="firm_food_vs_low_inv",
        )

    # Cancellation pairings
    pairs = select_pairs(name_to_ident, feature_aucs)
    cancellations: list[dict] = []
    for a, b, label in pairs:
        try:
            cancellations.append(run_cancellation(dictionary, (a, b), label))
        except Exception as e:
            print(f"  cancellation {label} failed: {type(e).__name__}: {e}")
            cancellations.append({
                "label": label, "pair": [a, b], "error": f"{type(e).__name__}: {e}",
            })

    # Summary
    print("\n" + "=" * 90)
    print(f"CANCELLATION SUMMARY  ({len(cancellations)} pairs across the full vocabulary)")
    print("=" * 90)
    print(f"{'label':<24s} {'pair':<46s} {'before':>7s} {'after':>7s} "
          f"{'floor':>7s} {'eff':>7s} {'met':>5s}")
    print("-" * 110)
    for r in cancellations:
        if "error" in r:
            print(f"{r['label']:<24s} {' / '.join(r['pair']):<46s}  "
                  f"ERROR: {r['error'][:40]}")
            continue
        eff = "N/A" if r["cancellation_efficiency"] is None else f"{r['cancellation_efficiency']:.1%}"
        print(f"{r['label']:<24s} {' / '.join(r['pair']):<46s} "
              f"{r['before_overlap']:>7.4f} {r['after_overlap']:>7.4f} "
              f"{r['structural_floor']:>7.4f} {eff:>7s} "
              f"{str(r['tolerance_met']):>5s}")

    summary = {
        "dictionary": {
            "name": dictionary.name,
            "n_features": len(dictionary.features),
            "n_qubits": required_qubits(len(dictionary.features)),
            "encoding": "HEA_Rung2",
            "hierarchy": dict(dictionary.hierarchy),
            "betas_by_cluster": {
                cluster: {
                    "n": len(betas),
                    "min": float(min(betas)), "median": float(np.median(betas)),
                    "max": float(max(betas)),
                }
                for cluster, betas in betas_by_cluster.items()
            },
        },
        "interference_sweep": sweep_result,
        "cancellations": cancellations,
        "feature_aucs": feature_aucs,
        "name_to_ident": name_to_ident,
    }
    out_path = os.path.join(OUT_DIR, "polygram_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")
    return summary


def _default_feature_aucs() -> dict[str, float]:
    """Pull best-AUC-per-feature from on-disk experiment summaries."""
    from econsae.ground_truth import all_features

    aucs: dict[str, float] = {}
    candidates = [
        ("runs/temporal_sentiment_experiment_summary.json", "auc_per_feature"),
        ("runs/scale_experiment_summary.json", "conj_auc_per_feature"),
        ("runs/attn_experiment_summary.json", "conj_auc_per_feature"),
        ("runs/macro_feed_v2_experiment_summary.json", "regime_auc_per_feature"),
        ("runs/macro_feed_experiment_summary.json", "regime_auc_per_feature"),
    ]
    for path, key in candidates:
        if not os.path.exists(path):
            continue
        try:
            data = json.load(open(path))
        except Exception:
            continue
        rows: Iterable[dict] = data if isinstance(data, list) else [data]
        for row in rows:
            per = row.get(key) or {}
            for f_name, auc in per.items():
                aucs[f_name] = max(aucs.get(f_name, 0.0), float(auc))
    # Fill in any feature in the full vocab that didn't get scored anywhere
    # with a chance-level placeholder.
    for f in all_features():
        aucs.setdefault(f, 0.5)
    return aucs


if __name__ == "__main__":
    main()
