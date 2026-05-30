"""Phase B: ground-truth alignment scoring for every (feed, variant).

Loads each runs/{feed}__{variant}.pt checkpoint, regenerates its feed,
computes the (sae_features, gt_features) AUC matrix, aggregates into
coverage / mean-best-AUC / monosemanticity overall AND per tier
(categorical / bucketed / conjunctive / regime). The per-tier breakdown
is the headline metric for econ-sae --- it tells you whether the SAE
recovered the polysemantic-trap conjunctive features or only the easy
categorical ones.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import torch

from econsae.sae.data import feed_raw, feed_embedded, feed_acts
from econsae.sae.evaluation import (
    align, score_sae, format_report_table, report_to_dict, TIERS,
)
from econsae.sae.models import make_sae
from econsae.sae.world_model import WorldModel
from econsae.simulator.ensemble import generate_ensemble


RUNS_DIR = os.path.join(REPO_ROOT, "runs")


def _load_world_model() -> WorldModel:
    ckpt = torch.load(os.path.join(RUNS_DIR, "world_model.pt"),
                      map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = WorldModel(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _build_feed(name: str, ens, base_rate: float = 0.02, monetary_step: float = 0.01):
    if name == "raw":
        return feed_raw(ens, base_rate=base_rate, monetary_step=monetary_step)
    if name == "embedded":
        return feed_embedded(ens, embed_dim=12, seed=0,
                             base_rate=base_rate, monetary_step=monetary_step)
    if name == "acts":
        return feed_acts(ens, _load_world_model(),
                         base_rate=base_rate, monetary_step=monetary_step)
    raise ValueError(name)


def load_sae(ckpt_path: str):
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kind, in_dim, n_feat = obj["kind"], obj["input_dim"], obj["n_features"]
    cfg = obj.get("feed_config", {})
    if kind == "topk":
        sae = make_sae("topk", in_dim, n_feat, k=cfg.get("topk_k", 8))
    elif kind == "l1":
        sae = make_sae("l1", in_dim, n_feat, l1_coeff=cfg.get("l1_coeff", 1e-2))
    else:
        sae = make_sae("jumprelu", in_dim, n_feat,
                       l0_coeff=cfg.get("l0_coeff", 5e-3),
                       init_theta=cfg.get("init_theta", 0.05))
    sae.load_state_dict(obj["state_dict"])
    sae.eval()
    return sae, obj


def _per_tier_means(reports) -> dict:
    """Mean per-tier mAUC across all scored runs (one number per tier)."""
    import numpy as np
    out = {}
    for t in TIERS:
        vals = [r.per_tier[t]["mean_best_auc"] for r in reports]
        out[t] = float(np.mean(vals)) if vals else float("nan")
    return out


def _print_comparison(calibrated_reports):
    """If a baseline alignment summary exists, print baseline-vs-calibrated
    per-tier mAUC (averaged over runs). Both arms must have been run at
    matching n_traj / n_periods / seed for the delta to be meaningful."""
    base_path = os.path.join(RUNS_DIR, "alignment_summary.json")
    if not os.path.exists(base_path):
        print("\n(no baseline runs/alignment_summary.json -- run evaluate.py "
              "without --calibrated at matching settings to get a comparison)")
        return
    with open(base_path) as f:
        base = json.load(f)
    base_tier = {t: [] for t in TIERS}
    for r in base:
        for t in TIERS:
            if t in r.get("per_tier", {}):
                base_tier[t].append(r["per_tier"][t]["mean_best_auc"])
    import numpy as np
    base_means = {t: float(np.mean(v)) if v else float("nan") for t, v in base_tier.items()}
    cal_means = _per_tier_means(calibrated_reports)
    print("\n" + "=" * 60)
    print("BASELINE vs CALIBRATED  (mean per-tier mAUC across runs)")
    print("=" * 60)
    print(f"{'tier':<12s} {'baseline':>9s} {'calibrated':>11s} {'delta':>8s}")
    print("-" * 44)
    for t in TIERS:
        d = cal_means[t] - base_means[t]
        print(f"{t:<12s} {base_means[t]:>9.3f} {cal_means[t]:>11.3f} {d:>+8.3f}")
    print("(meaningful only if both arms used matching n_traj/n_periods/seed)")


def main(n_trajectories: int = 32, n_periods: int = 60, seed: int = 0,
         calibrated: str | None = None):
    from scripts._calibration_arm import resolve_arm
    arm = resolve_arm(calibrated)
    print("=" * 78)
    print(f"[{arm.label}] regenerating ensemble for evaluation "
          f"(must match training settings)")
    print("=" * 78)
    ens = generate_ensemble(n_trajectories=n_trajectories, n_periods=n_periods,
                            seed=seed, sim_config=arm.sim_config)

    reports = []
    for feed_name in ("raw", "embedded", "acts"):
        feed = _build_feed(feed_name, ens, base_rate=arm.base_rate,
                           monetary_step=arm.monetary_step)
        for variant in ("topk", "l1", "jumprelu"):
            run_id = f"{feed_name}__{variant}"
            ckpt_path = os.path.join(RUNS_DIR, f"{run_id}{arm.suffix}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  skip (no ckpt): {ckpt_path}")
                continue
            sae, _ = load_sae(ckpt_path)
            Z = score_sae(sae, feed.X)
            rep = align(Z, feed.Y, feed.feature_vocab)
            rep.run_id = run_id; rep.feed_name = feed_name; rep.variant = variant
            reports.append(rep)
            print(f"  scored {run_id}: cov95={rep.coverage_at_0_95:.1%}  "
                  f"mAUC={rep.mean_best_auc:.3f}  mono={rep.monosemanticity:.1%}  "
                  f"per-tier mAUC=" + ", ".join(
                      f"{t[:4]}={rep.per_tier[t]['mean_best_auc']:.2f}" for t in TIERS))

    print()
    print(format_report_table(reports))

    # Per-run detail: which GT features were recovered cleanly
    print("\n" + "=" * 78)
    print("Top recovered ground-truth features per run (AUC >= 0.95)")
    print("=" * 78)
    for r in reports:
        n_show = min(8, len(r.top_matches))
        print(f"\n--- {r.run_id} ({n_show}/{len(r.top_matches)} clean recoveries) ---")
        if not r.top_matches:
            print("    (none reached AUC >= 0.95)")
            continue
        for m in r.top_matches[:n_show]:
            print(f"    AUC={m['auc']:.4f}  [{m['tier']:<11s}]  "
                  f"GT={m['gt_feature']:<40s}  sae#{m['sae_feature_idx']}")

    out_name = "alignment_summary_calibrated.json" if arm.suffix else "alignment_summary.json"
    out_path = os.path.join(RUNS_DIR, out_name)
    with open(out_path, "w") as f:
        json.dump([report_to_dict(r) for r in reports], f, indent=2)
    print(f"\nWrote {out_path}")

    if arm.suffix:                       # calibrated arm -> show the delta
        _print_comparison(reports)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ground-truth alignment scoring per (feed, variant).")
    ap.add_argument("--calibrated", metavar="CONFIG.json", default=None,
                    help="score the calibrated arm (loads __calibrated checkpoints) "
                         "and print a baseline-vs-calibrated per-tier comparison")
    ap.add_argument("--n-traj", type=int, default=32)
    ap.add_argument("--n-periods", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(n_trajectories=args.n_traj, n_periods=args.n_periods, seed=args.seed,
         calibrated=args.calibrated)
