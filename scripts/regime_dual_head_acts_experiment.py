"""Phase 9.2.1: SAE on DualHeadRegimeWM's h1 substrate.

Trains an SAE directly on `DualHeadRegimeWM.h1` (192-d, the host's fc1
output) rather than on the engineered macro-feed v3 substrate (223-d)
that Phase 6.2 / 8.x used. The motivation is Phase 9.2:
`TemporalWMAdapter` bridges at `fc1` and therefore requires
`basis.d_model == host.h1_dim`. An SAE trained at this substrate is the
first one whose Phase 9.2 forge can produce a meaningful next-state MSE.

Reuses the already-trained `runs/world_model_regime_dual_head.pt`
checkpoint -- no world-model retraining. Just extracts h1 activations
from it (the same per-(period, agent) substrate `_build_acts_feed`
uses) and fits a JumpReLU SAE.

Output:
  runs/regime_dual_head_acts_experiment/jr_w512_ep100.pt
  runs/regime_dual_head_acts_experiment_summary.json

Usage:
    python scripts/regime_dual_head_acts_experiment.py
"""

from __future__ import annotations

import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import torch

from econsae.ground_truth import build_feature_matrix
from econsae.sae.data import Feed
from econsae.sae.evaluation import (
    align, score_sae, report_to_dict, TIERS, feature_tier,
)
from econsae.sae.models import make_sae
from econsae.sae.train import TrainConfig, train
from econsae.sae.world_model import extract_temporal_h1_activations
from econsae.simulator.ensemble import generate_ensemble
from scripts.regime_dual_head_experiment import DualHeadRegimeWM


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
EXP_DIR = os.path.join(RUNS_DIR, "regime_dual_head_acts_experiment")
os.makedirs(EXP_DIR, exist_ok=True)


SAE_CONFIGS = [
    ("jr_w512_ep100", "jumprelu", 512, {"l0_coeff": 1.5e-3, "init_theta": 0.05}, 100),
]


def main(
    seed: int = 0,
    n_trajectories: int = 128,
    n_periods: int = 100,
    sentiment_strength: float = 0.20,
    host_ckpt: str = os.path.join(RUNS_DIR, "world_model_regime_dual_head.pt"),
):
    torch.manual_seed(seed)
    t_start = time.time()
    print("=" * 78)
    print(f"REGIME DUAL-HEAD acts EXPERIMENT  (Phase 9.2.1)")
    print(f"  host ckpt = {host_ckpt}")
    print(f"  n_traj={n_trajectories}  n_periods={n_periods}  "
          f"sentiment={sentiment_strength}")
    print("=" * 78)

    # ---- 1. Ensemble (must match the sentiment used when the host was
    #         trained, so the h1 distribution matches the host's training
    #         distribution exactly) ----
    t0 = time.time()
    ens = generate_ensemble(n_trajectories=n_trajectories,
                            n_periods=n_periods, seed=seed,
                            sentiment_strength=sentiment_strength)
    print(f"\n[1] Ensemble in {time.time() - t0:.1f}s")

    # ---- 2. GT matrix ----
    t0 = time.time()
    fm = build_feature_matrix(ens.trajectories, ens.shock_schedules)
    print(f"\n[2] GT matrix in {time.time() - t0:.1f}s: "
          f"Y={fm.Y.shape}  vocab={len(fm.feature_vocab)}")

    # ---- 3. Load host (no retraining) ----
    t0 = time.time()
    ckpt = torch.load(host_ckpt, map_location="cpu", weights_only=False)
    host = DualHeadRegimeWM(**ckpt["config"])
    host.load_state_dict(ckpt["state_dict"])
    host.eval()
    print(f"\n[3] Host loaded in {time.time() - t0:.1f}s: "
          f"h1_dim={host.h1_dim}  n_regime_labels={host.n_regime_labels}")

    # ---- 4. Extract h1 ----
    t0 = time.time()
    H1, idx = extract_temporal_h1_activations(host, ens.trajectories,
                                                ens.shock_schedules)
    assert idx == fm.sample_index, "activation index mismatch"
    print(f"\n[4] h1 extraction in {time.time() - t0:.1f}s: "
          f"H1={H1.shape}  sparsity={(H1 == 0).mean():.1%}")

    feed = Feed(
        name=f"acts_regime_dual_head_d{H1.shape[1]}",
        X=torch.tensor(H1, dtype=torch.float32),
        Y=fm.Y,
        feature_vocab=fm.feature_vocab,
        sample_index=fm.sample_index,
        notes=("DualHeadRegimeWM h1 (192-d). Substrate aligned with "
               "TemporalWMAdapter's fc1 bridge for Phase 9.2 forge."),
    )

    # ---- 5. SAE training ----
    results: list[dict] = []
    for name, variant, n_feat, kw, epochs in SAE_CONFIGS:
        print(f"\n[5] SAE: {name}  ({variant}, n_features={n_feat}, "
              f"epochs={epochs})")
        torch.manual_seed(seed)
        sae = make_sae(variant, feed.D, n_feat, **kw)
        tcfg = TrainConfig(
            epochs=epochs, batch_size=512, lr=1e-3, warmup_steps=50,
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
        print(f"   recon={recon:.4f} L0={l0:.2f} VE={ve:.4f} time={elapsed:.1f}s")

        ckpt_path = os.path.join(EXP_DIR, f"{name}.pt")
        torch.save({
            "state_dict": sae.state_dict(), "kind": variant,
            "feed_name": "acts_regime_dual_head",
            "input_dim": sae.input_dim, "n_features": sae.n_features,
            "feed_config": kw,
        }, ckpt_path)
        print(f"   saved -> {ckpt_path}")

        Z = score_sae(sae, feed.X)
        rep = align(Z, feed.Y, feed.feature_vocab)
        rep.run_id = name; rep.feed_name = "acts_regime_dual_head"
        rep.variant = variant
        print(f"   cov95={rep.coverage_at_0_95:.1%}  mAUC={rep.mean_best_auc:.3f}  "
              f"mono={rep.monosemanticity:.1%}")
        for t in TIERS:
            pt = rep.per_tier[t]
            print(f"     {t:<11s} n={pt['n_features']:>2d}  "
                  f"cov95={pt['coverage_0.95']:>5.1%}  "
                  f"mAUC={pt['mean_best_auc']:.3f}")

        row = report_to_dict(rep)
        row.update({
            "name": name, "n_features": n_feat, "epochs": epochs,
            "recon_loss": recon, "l0": l0, "var_explained": ve,
            "wall_time_s": elapsed,
        })
        results.append(row)

    out_path = os.path.join(RUNS_DIR,
                             "regime_dual_head_acts_experiment_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"\nTotal wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
