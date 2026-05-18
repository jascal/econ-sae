"""Sweep SAE width and training duration to test conjunctive-feature recovery.

Hypothesis under test (asked for by the user, phase 1.5):
    "Does giving the SAE more capacity (wider n_features) and/or more
    training (more epochs) improve recovery of the CONJUNCTIVE tier
    of ground-truth features?"

Fixed-feed setup: all configs use the `acts` feed (world-model h1
activations, 96-dim), which gave the strongest baseline (33% overall
cov95, 50% conjunctive). The sweep varies width and epochs along two
axes for JumpReLU, plus widens the other variants once for comparison.

Output:
    runs/sweep_widths/{config_name}.pt          -- SAE checkpoints
    runs/sweep_widths_summary.json              -- per-config metrics + per-tier
    stdout                                       -- comparison table

Usage:
    python scripts/sweep_widths.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import torch

from econsae.sae.data import feed_acts
from econsae.sae.evaluation import (
    align, score_sae, report_to_dict, TIERS,
)
from econsae.sae.models import make_sae
from econsae.sae.train import TrainConfig, train
from econsae.sae.world_model import WorldModel
from econsae.simulator.ensemble import generate_ensemble


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
SWEEP_DIR = os.path.join(RUNS_DIR, "sweep_widths")
os.makedirs(SWEEP_DIR, exist_ok=True)


@dataclass
class SweepConfig:
    name: str
    variant: str
    n_features: int
    epochs: int
    # Variant-specific
    topk_k: int = 12
    l1_coeff: float = 3e-3
    l0_coeff: float = 1.5e-3
    init_theta: float = 0.05


CONFIGS: list[SweepConfig] = [
    # 2x2 JumpReLU grid: {width 256, 1024} x {epochs 200, 500}
    SweepConfig(name="jr_w256_ep200",  variant="jumprelu", n_features=256,  epochs=200),
    SweepConfig(name="jr_w1024_ep200", variant="jumprelu", n_features=1024, epochs=200),
    SweepConfig(name="jr_w256_ep500",  variant="jumprelu", n_features=256,  epochs=500),
    SweepConfig(name="jr_w1024_ep500", variant="jumprelu", n_features=1024, epochs=500),
    # Wider TopK and L1 at 200 epochs for cross-variant comparison
    SweepConfig(name="topk_w1024_k20_ep200", variant="topk", n_features=1024,
                epochs=200, topk_k=20),
    SweepConfig(name="l1_w1024_loose_ep200", variant="l1", n_features=1024,
                epochs=200, l1_coeff=2e-3),
]


def load_world_model() -> WorldModel:
    ckpt = torch.load(os.path.join(RUNS_DIR, "world_model.pt"),
                      map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = WorldModel(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def build_sae(cfg: SweepConfig, input_dim: int):
    if cfg.variant == "topk":
        return make_sae("topk", input_dim, cfg.n_features, k=cfg.topk_k)
    if cfg.variant == "l1":
        return make_sae("l1", input_dim, cfg.n_features, l1_coeff=cfg.l1_coeff)
    if cfg.variant == "jumprelu":
        return make_sae("jumprelu", input_dim, cfg.n_features,
                        l0_coeff=cfg.l0_coeff, init_theta=cfg.init_theta)
    raise ValueError(cfg.variant)


def main(seed: int = 0, n_trajectories: int = 32, n_periods: int = 60):
    torch.manual_seed(seed)
    print("=" * 78)
    print(f"Sweep: acts feed only, {len(CONFIGS)} configs")
    print("=" * 78)

    ens = generate_ensemble(n_trajectories=n_trajectories,
                            n_periods=n_periods, seed=seed)
    feed = feed_acts(ens, load_world_model())
    print(f"  acts feed: X={tuple(feed.X.shape)} Y={feed.Y.shape}  "
          f"vocab={len(feed.feature_vocab)}")
    print()

    results: list[dict] = []

    for cfg in CONFIGS:
        print(f"--- {cfg.name}  ({cfg.variant}, n_features={cfg.n_features}, "
              f"epochs={cfg.epochs}) ---")
        torch.manual_seed(seed)
        sae = build_sae(cfg, feed.D)
        tcfg = TrainConfig(
            epochs=cfg.epochs, batch_size=256, lr=1e-3, warmup_steps=50,
            resample_every=max(100, cfg.epochs // 5),
            log_every=max(200, cfg.epochs * 10),  # quiet
        )
        t0 = time.time()
        hist = train(sae, feed.X, tcfg, verbose=False)
        train_time = time.time() - t0

        with torch.no_grad():
            out = sae(feed.X)
            final_recon = float(out.recon_loss)
            final_l0 = float((out.z.abs() > 1e-9).float().sum(dim=-1).mean())
            var_total = float(feed.X.var())
            var_resid = float((feed.X - out.x_hat).var())
            ve = 1.0 - var_resid / max(var_total, 1e-12)

        # save checkpoint
        ckpt_path = os.path.join(SWEEP_DIR, f"{cfg.name}.pt")
        torch.save({
            "state_dict": sae.state_dict(),
            "kind": cfg.variant, "feed_name": "acts",
            "input_dim": sae.input_dim, "n_features": sae.n_features,
            "feed_config": {
                "topk_k": cfg.topk_k, "l1_coeff": cfg.l1_coeff,
                "l0_coeff": cfg.l0_coeff, "init_theta": cfg.init_theta,
            },
        }, ckpt_path)

        # alignment
        Z = score_sae(sae, feed.X)
        rep = align(Z, feed.Y, feed.feature_vocab)
        rep.run_id = cfg.name; rep.feed_name = "acts"; rep.variant = cfg.variant

        row = report_to_dict(rep)
        row.update({
            "name": cfg.name,
            "n_features": cfg.n_features, "epochs": cfg.epochs,
            "recon_loss": final_recon, "l0": final_l0,
            "var_explained": ve, "wall_time_s": train_time,
        })
        results.append(row)
        print(f"   recon={final_recon:.4f}  L0={final_l0:.2f}  VE={ve:.4f}  "
              f"time={train_time:.1f}s")
        print(f"   cov95={rep.coverage_at_0_95:.1%}  mAUC={rep.mean_best_auc:.3f}  "
              f"mono={rep.monosemanticity:.1%}")
        for t in TIERS:
            pt = rep.per_tier[t]
            print(f"     {t:<11s} n={pt['n_features']:>2d}  "
                  f"cov95={pt['coverage_0.95']:>5.1%}  "
                  f"mAUC={pt['mean_best_auc']:.3f}")
        print()

    # Aggregate table
    print("=" * 100)
    print("SWEEP SUMMARY  (acts feed; per-tier cov95 / mean-best-AUC)")
    print("=" * 100)
    print(f"{'name':<24s} {'w':>5s} {'ep':>4s} {'time':>6s} {'L0':>5s} {'VE':>6s}  "
          f"{'cov95':>5s} {'mAUC':>5s} " + " ".join(f"{t[:4]+'/95':>8s}" for t in TIERS)
          + " " + " ".join(f"{t[:4]+'/A':>6s}" for t in TIERS))
    print("-" * 100)
    for r in results:
        line = (f"{r['name']:<24s} {r['n_features']:>5d} {r['epochs']:>4d} "
                f"{r['wall_time_s']:>5.0f}s {r['l0']:>5.1f} {r['var_explained']:>6.3f}  "
                f"{r['coverage_0.95']:>5.1%} {r['mean_best_auc']:>5.3f} "
                + " ".join(f"{r['per_tier'][t]['coverage_0.95']:>8.1%}" for t in TIERS)
                + " "
                + " ".join(f"{r['per_tier'][t]['mean_best_auc']:>6.3f}" for t in TIERS))
        print(line)

    out_path = os.path.join(RUNS_DIR, "sweep_widths_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
