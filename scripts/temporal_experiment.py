"""Temporal experiment: cross-agent attention + cross-period GRU.

Tests whether adding temporal context to the world model finally moves
the regime / phase tier of the ground-truth vocabulary. Regime features
(`phase:expansion`, `phase:contraction`, ...) are computed from a 5-period
trailing window of macros, so a model that only sees one period at a
time cannot encode them. A GRU across periods (per agent) carries
exactly that history.

Architecture: TemporalWorldModel = AttnWorldModel + GRU across periods.

Output:
  runs/world_model_temporal.pt
  runs/world_model_temporal_acts.npz
  runs/temporal_experiment/{cfg}.pt
  runs/temporal_experiment_summary.json

Usage:
    python scripts/temporal_experiment.py
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
from econsae.sae.world_model import (
    TemporalWorldModel, WMTrainConfig,
    build_temporal_data, extract_temporal_h1_activations,
    train_temporal_world_model,
)
from econsae.simulator.ensemble import generate_ensemble


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
EXP_DIR = os.path.join(RUNS_DIR, "temporal_experiment")
os.makedirs(EXP_DIR, exist_ok=True)


SAE_CONFIGS = [
    ("jr_w256_ep200",  "jumprelu", 256,  {"l0_coeff": 1.5e-3, "init_theta": 0.05}, 200),
    ("jr_w1024_ep200", "jumprelu", 1024, {"l0_coeff": 1.5e-3, "init_theta": 0.05}, 200),
]


def main(seed: int = 0, n_trajectories: int = 128, n_periods: int = 100,
         wm_epochs: int = 50, wm_batch_size: int = 16,
         embed_dim: int = 64, n_heads: int = 4, n_attn_layers: int = 1,
         gru_hidden: int = 128, n_gru_layers: int = 1,
         h1_dim: int = 192, h2_dim: int = 128):
    torch.manual_seed(seed)
    t_start = time.time()
    print("=" * 78)
    print(f"TEMPORAL EXPERIMENT  (n_traj={n_trajectories}, n_periods={n_periods})")
    print("=" * 78)

    # ---- 1. Ensemble ----
    t0 = time.time()
    ens = generate_ensemble(n_trajectories=n_trajectories,
                            n_periods=n_periods, seed=seed)
    print(f"\n[1] Ensemble in {time.time() - t0:.1f}s; "
          f"worst residual {max(ens.conservation_summary().values()):.2e}")

    # ---- 2. GT feature matrix ----
    t0 = time.time()
    fm = build_feature_matrix(ens.trajectories, ens.shock_schedules)
    print(f"\n[2] GT matrix in {time.time() - t0:.1f}s: "
          f"X={fm.X.shape} Y={fm.Y.shape} vocab={len(fm.feature_vocab)}")
    prev = fm.Y.mean(axis=0)
    print("    regime feature prevalence (this is the tier we want to crack):")
    for j, name in enumerate(fm.feature_vocab):
        if feature_tier(name) == "regime":
            print(f"      {prev[j]:>6.2%}  {name}")

    # ---- 3. Train TemporalWorldModel ----
    t0 = time.time()
    wm_data = build_temporal_data(ens.trajectories, ens.shock_schedules)
    print(f"\n[3] Temporal training trajectories in {time.time() - t0:.1f}s: "
          f"X={tuple(wm_data.X.shape)} S={tuple(wm_data.states.shape)}")
    wm = TemporalWorldModel(
        embed_dim=embed_dim, n_heads=n_heads, n_attn_layers=n_attn_layers,
        gru_hidden=gru_hidden, n_gru_layers=n_gru_layers,
        h1_dim=h1_dim, h2_dim=h2_dim,
    )
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"    TemporalWorldModel: embed={embed_dim} heads={n_heads} "
          f"attn_layers={n_attn_layers} gru_hidden={gru_hidden} "
          f"h1={h1_dim} h2={h2_dim}  params={n_params:,}")
    t0 = time.time()
    res = train_temporal_world_model(
        wm, wm_data,
        WMTrainConfig(epochs=wm_epochs, batch_size=wm_batch_size),
        verbose=False,
    )
    print(f"    trained {wm_epochs} epochs in {time.time() - t0:.1f}s. "
          f"final MSE (z-scored) = {res['history'][-1]:.4f}")
    torch.save({
        "state_dict": wm.state_dict(),
        "config": {
            "agent_dim": wm.agent_dim, "macro_dim": wm.macro_dim,
            "shock_dim": wm.shock_dim, "embed_dim": embed_dim,
            "n_heads": n_heads, "n_attn_layers": n_attn_layers,
            "gru_hidden": gru_hidden, "n_gru_layers": n_gru_layers,
            "h1_dim": h1_dim, "h2_dim": h2_dim,
        },
        "final_train_mse": res["history"][-1],
        "loss_history": res["history"],
    }, os.path.join(RUNS_DIR, "world_model_temporal.pt"))

    # ---- 4. Extract h1 ----
    t0 = time.time()
    H1, idx = extract_temporal_h1_activations(wm, ens.trajectories, ens.shock_schedules)
    assert idx == fm.sample_index, "activation index mismatch"
    print(f"\n[4] Temporal h1 extraction in {time.time() - t0:.1f}s: "
          f"H1={H1.shape} sparsity={(H1 == 0).mean():.1%}")
    np.savez_compressed(
        os.path.join(RUNS_DIR, "world_model_temporal_acts.npz"),
        H1=H1.astype(np.float32),
        sample_index=np.array(idx, dtype=np.int32),
    )

    # ---- 5. Build feed and run SAEs ----
    feed = Feed(
        name=f"acts_temporal_d{H1.shape[1]}",
        X=torch.tensor(H1, dtype=torch.float32),
        Y=fm.Y,
        feature_vocab=fm.feature_vocab,
        sample_index=fm.sample_index,
        notes=f"TemporalWorldModel h1 (d={H1.shape[1]}). "
              f"Sparsity {(H1 == 0).mean():.1%}.",
    )

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

        torch.save({
            "state_dict": sae.state_dict(), "kind": variant,
            "feed_name": "acts_temporal",
            "input_dim": sae.input_dim, "n_features": sae.n_features,
            "feed_config": kw,
        }, os.path.join(EXP_DIR, f"{name}.pt"))

        Z = score_sae(sae, feed.X)
        rep = align(Z, feed.Y, feed.feature_vocab)
        rep.run_id = name; rep.feed_name = "acts_temporal"; rep.variant = variant
        print(f"   cov95={rep.coverage_at_0_95:.1%}  mAUC={rep.mean_best_auc:.3f}  "
              f"mono={rep.monosemanticity:.1%}")
        for t in TIERS:
            pt = rep.per_tier[t]
            print(f"     {t:<11s} n={pt['n_features']:>2d}  "
                  f"cov95={pt['coverage_0.95']:>5.1%}  "
                  f"mAUC={pt['mean_best_auc']:.3f}")

        # Per-feature breakdown for the two interesting tiers
        for target_tier in ("conjunctive", "regime"):
            print(f"   {target_tier} AUC per feature:")
            for j, gname in enumerate(feed.feature_vocab):
                if feature_tier(gname) == target_tier:
                    best = float(rep.alignment[:, j].max())
                    marker = " (rare)" if prev[j] < 0.01 else ""
                    print(f"     {best:.3f}  {gname:<48s}{marker}")

        row = report_to_dict(rep)
        row.update({
            "name": name, "n_features": n_feat, "epochs": epochs,
            "recon_loss": recon, "l0": l0, "var_explained": ve,
            "wall_time_s": elapsed,
            "auc_per_feature": {
                feed.feature_vocab[j]: float(rep.alignment[:, j].max())
                for j in range(len(feed.feature_vocab))
            },
        })
        results.append(row)

    # ---- 6. Comparison table ----
    print("\n" + "=" * 100)
    print(f"TEMPORAL EXPERIMENT SUMMARY  ({n_trajectories}x{n_periods} ensemble)")
    print("=" * 100)
    print(f"{'name':<24s} {'w':>5s} {'ep':>4s} {'time':>6s} {'L0':>5s} {'VE':>6s}  "
          f"{'cov95':>5s} {'mAUC':>5s} " + " ".join(f"{t[:4]+'/95':>8s}" for t in TIERS)
          + " " + " ".join(f"{t[:4]+'/A':>6s}" for t in TIERS))
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<24s} {r['n_features']:>5d} {r['epochs']:>4d} "
              f"{r['wall_time_s']:>5.0f}s {r['l0']:>5.1f} {r['var_explained']:>6.3f}  "
              f"{r['coverage_0.95']:>5.1%} {r['mean_best_auc']:>5.3f} "
              + " ".join(f"{r['per_tier'][t]['coverage_0.95']:>8.1%}" for t in TIERS)
              + " "
              + " ".join(f"{r['per_tier'][t]['mean_best_auc']:>6.3f}" for t in TIERS))

    out_path = os.path.join(RUNS_DIR, "temporal_experiment_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"\nTotal wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
