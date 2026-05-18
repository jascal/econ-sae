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


def _build_feed(name: str, ens):
    if name == "raw":      return feed_raw(ens)
    if name == "embedded": return feed_embedded(ens, embed_dim=12, seed=0)
    if name == "acts":     return feed_acts(ens, _load_world_model())
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


def main(n_trajectories: int = 32, n_periods: int = 60, seed: int = 0):
    print("=" * 78)
    print("Regenerating ensemble for evaluation (must match training settings)")
    print("=" * 78)
    ens = generate_ensemble(n_trajectories=n_trajectories, n_periods=n_periods, seed=seed)

    reports = []
    for feed_name in ("raw", "embedded", "acts"):
        feed = _build_feed(feed_name, ens)
        for variant in ("topk", "l1", "jumprelu"):
            run_id = f"{feed_name}__{variant}"
            ckpt_path = os.path.join(RUNS_DIR, f"{run_id}.pt")
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

    out_path = os.path.join(RUNS_DIR, "alignment_summary.json")
    with open(out_path, "w") as f:
        json.dump([report_to_dict(r) for r in reports], f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
