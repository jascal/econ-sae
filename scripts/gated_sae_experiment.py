"""Phase 3.2: GatedSAE on the macro-feed v2 substrate.

The macro_feed_v2 experiment recovered 3 of 6 regime features cleanly
(AUC >= 0.97) but plateaued at AUC ~0.78-0.79 on three threshold-on-window
features: `phase:expansion`, `phase:contraction`, `phase:high_rate`.

The diagnosis there: smooth ReLU activations correlate with the
underlying continuous regime intensity but can't fire cleanly above a
hard threshold. Hypothesis: a Gated SAE with explicit Heaviside step
activations should match the threshold structure of these labels and
cross AUC 0.95.

This script:
  1. Rebuilds the macro_feed v2 inputs (z-scored macro + shock + GDP
     window + impulse flags + mean-pooled h1).
  2. Trains a GatedSAE on it.
  3. Reports per-regime-feature AUC and compares to the prior JumpReLU
     plateau.

Output:
  runs/gated_sae_experiment/{cfg}.pt
  runs/gated_sae_experiment_summary.json
"""

from __future__ import annotations

import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import torch

from econsae.sae.evaluation import (
    align, score_sae, report_to_dict, TIERS, feature_tier,
)
from econsae.sae.models import make_sae
from econsae.sae.train import TrainConfig, train
from econsae.simulator.ensemble import generate_ensemble

# Reuse the v2 feed builder + WM loader directly to avoid duplication.
from scripts.macro_feed_v2_experiment import (
    build_macro_feed_v2, load_temporal_sentiment_wm,
)


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
EXP_DIR = os.path.join(RUNS_DIR, "gated_sae_experiment")
os.makedirs(EXP_DIR, exist_ok=True)


SAE_CONFIGS = [
    # (name, variant, n_features, kwargs, epochs)
    ("gated_w256_l1_5e4",  "gated",    256,  {"l1_coeff": 5e-4}, 200),
    ("gated_w512_l1_5e4",  "gated",    512,  {"l1_coeff": 5e-4}, 300),
    ("gated_w1024_l1_3e4", "gated",    1024, {"l1_coeff": 3e-4}, 300),
    # JumpReLU baseline for direct comparison on the same feed
    ("jr_w512_baseline",   "jumprelu", 512,  {"l0_coeff": 1.5e-3,
                                              "init_theta": 0.05}, 300),
]


def main(seed: int = 0, n_trajectories: int = 128, n_periods: int = 100,
         sentiment_strength: float = 0.20):
    torch.manual_seed(seed)
    t_start = time.time()
    print("=" * 78)
    print("GATED-SAE EXPERIMENT  (macro-feed v2 substrate)")
    print("=" * 78)

    ens = generate_ensemble(n_trajectories=n_trajectories, n_periods=n_periods,
                            seed=seed, sentiment_strength=sentiment_strength)
    print(f"\n[1] Ensemble in {time.time() - t_start:.1f}s")

    wm = load_temporal_sentiment_wm()
    feed = build_macro_feed_v2(ens.trajectories, ens.shock_schedules, wm)
    print(f"[2] Feed: X={tuple(feed.X.shape)}  Y={feed.Y.shape}  "
          f"vocab={len(feed.feature_vocab)}")

    results: list[dict] = []
    for name, variant, n_feat, kw, epochs in SAE_CONFIGS:
        print(f"\n[3] SAE: {name}  ({variant}, n_features={n_feat}, epochs={epochs})")
        torch.manual_seed(seed)
        sae = make_sae(variant, feed.D, n_feat, **kw)
        tcfg = TrainConfig(
            epochs=epochs, batch_size=256, lr=1e-3, warmup_steps=50,
            resample_every=max(100, epochs // 5),
            log_every=10**6,
        )
        t0 = time.time()
        hist = train(sae, feed.X, tcfg, verbose=False)
        elapsed = time.time() - t0

        with torch.no_grad():
            out = sae(feed.X)
            recon = float(out.recon_loss)
            l0 = float((out.z.abs() > 1e-9).float().sum(dim=-1).mean())
            var_total = float(feed.X.var())
            var_resid = float((feed.X - out.x_hat).var())
            ve = 1.0 - var_resid / max(var_total, 1e-12)
        print(f"   recon={recon:.4f}  L0={l0:.2f}  VE={ve:.4f}  time={elapsed:.1f}s")

        torch.save({
            "state_dict": sae.state_dict(), "kind": variant,
            "feed_name": "macro_feed_v2",
            "input_dim": sae.input_dim, "n_features": sae.n_features,
            "feed_config": kw,
        }, os.path.join(EXP_DIR, f"{name}.pt"))

        Z = score_sae(sae, feed.X)
        rep = align(Z, feed.Y, feed.feature_vocab)
        rep.run_id = name; rep.feed_name = "macro_feed_v2"; rep.variant = variant
        print(f"   cov95={rep.coverage_at_0_95:.1%}  mAUC={rep.mean_best_auc:.3f}  "
              f"mono={rep.monosemanticity:.1%}")
        for tier in TIERS:
            pt = rep.per_tier[tier]
            print(f"     {tier:<11s} n={pt['n_features']:>2d}  "
                  f"cov95={pt['coverage_0.95']:>5.1%}  "
                  f"mAUC={pt['mean_best_auc']:.3f}")

        print("   regime AUC per feature:")
        for j, gname in enumerate(feed.feature_vocab):
            if feature_tier(gname) == "regime":
                best = float(rep.alignment[:, j].max())
                marker = " ***" if best >= 0.95 else ""
                print(f"     {best:.3f}  {gname:<48s}{marker}")

        row = report_to_dict(rep)
        row.update({
            "name": name, "n_features": n_feat, "epochs": epochs,
            "recon_loss": recon, "l0": l0, "var_explained": ve,
            "wall_time_s": elapsed,
            "regime_auc_per_feature": {
                feed.feature_vocab[j]: float(rep.alignment[:, j].max())
                for j in range(len(feed.feature_vocab))
                if feature_tier(feed.feature_vocab[j]) == "regime"
            },
        })
        results.append(row)

    print("\n" + "=" * 95)
    print("GATED-SAE SUMMARY  (macro-feed v2 substrate)")
    print("=" * 95)
    print(f"{'name':<24s} {'w':>5s} {'ep':>4s} {'time':>6s} {'L0':>5s} {'VE':>7s}  "
          f"{'cov95':>5s} {'mAUC':>5s}  {'regi/95':>8s}  {'regi/A':>6s}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<24s} {r['n_features']:>5d} {r['epochs']:>4d} "
              f"{r['wall_time_s']:>5.0f}s {r['l0']:>5.1f} {r['var_explained']:>7.3f}  "
              f"{r['coverage_0.95']:>5.1%} {r['mean_best_auc']:>5.3f}  "
              f"{r['per_tier']['regime']['coverage_0.95']:>8.1%}  "
              f"{r['per_tier']['regime']['mean_best_auc']:>6.3f}")

    out_path = os.path.join(RUNS_DIR, "gated_sae_experiment_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
