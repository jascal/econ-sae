"""Macro-feed SAE experiment: per-period samples instead of per-agent.

Tests the Phase 1.8 structural diagnosis: regime / phase features are
per-period (global), but the standard SAE substrate is per-(period,
agent). A sparse decoder operating on per-agent rows naturally assigns
features to local concepts and spreads global signals across many
features at low individual correlation. The fix should be a feed where
each sample IS one period.

Setup:
  - Same 128 x 100 ensemble used in temporal_sentiment_experiment.
  - Same trained TemporalWorldModel (loaded from
    runs/world_model_temporal_sentiment.pt).
  - Each sample = one (trajectory, period) row, with input vector
        [macro_vec (10) + shock_vec (10) + mean_h1_over_agents (192)]
        = 212-dim
  - Total samples = 128 * 100 = 12,800
  - Labels = restricted vocabulary of per-period features only
        (phase:*, shock_period_has:*, txn_period_has:*); per-agent
        features (sector, cohort, conjunctive, bucketed) are dropped
        because they make no sense at period granularity.

If regime AUC climbs to >= 0.95 on this feed, the structural diagnosis
is confirmed: regime info IS in the substrate, just not at the
decoding granularity the per-agent SAE could see. If regime AUC stays
~0.6, the signal genuinely is weak in h1 and the fix has to come from
the world model.

Output:
  runs/macro_feed_experiment/{cfg}.pt
  runs/macro_feed_experiment_summary.json
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
    TemporalWorldModel, MACRO_DIM, SHOCK_DIM,
    encode_macros, encode_shock, extract_temporal_h1_activations,
)
from econsae.simulator.ensemble import generate_ensemble


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
EXP_DIR = os.path.join(RUNS_DIR, "macro_feed_experiment")
os.makedirs(EXP_DIR, exist_ok=True)


def load_temporal_sentiment_wm() -> TemporalWorldModel:
    ckpt = torch.load(
        os.path.join(RUNS_DIR, "world_model_temporal_sentiment.pt"),
        map_location="cpu", weights_only=False,
    )
    cfg = ckpt["config"]
    m = TemporalWorldModel(**cfg)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


PER_PERIOD_PREFIXES = ("phase:", "shock_period_has:", "txn_period_has:")


def build_macro_feed(trajectories, shock_schedules, wm) -> Feed:
    n_trajs = len(trajectories)
    T = trajectories[0].T
    N = trajectories[0].n_agents

    # 1) Extract per-agent h1 via the world model, reshape, mean-pool over agents.
    H1_flat, idx = extract_temporal_h1_activations(wm, trajectories, shock_schedules)
    # idx is in (traj-major, period-major, agent-major) order
    H1 = H1_flat.reshape(n_trajs, T, N, -1)
    h1_mean = H1.mean(axis=2)                                  # (n_trajs, T, h1_dim)

    # 2) Per-period macro + shock vectors
    macro_vecs = np.stack(
        [np.stack([encode_macros(m) for m in traj.macros], axis=0)
         for traj in trajectories],
        axis=0,
    )                                                          # (n_trajs, T, MACRO_DIM)
    shock_vecs = np.stack(
        [np.stack([encode_shock(sh) for sh in sched.shocks], axis=0)
         for sched in shock_schedules],
        axis=0,
    )                                                          # (n_trajs, T, SHOCK_DIM)

    X_pt = np.concatenate([macro_vecs, shock_vecs, h1_mean], axis=-1)
    X = X_pt.reshape(n_trajs * T, -1).astype(np.float32)       # (n_samples, in_dim)

    # 3) Per-period labels: pull the per-period features from the FULL
    #    feature matrix (any agent's row in a given period has the same
    #    per-period feature values).
    fm = build_feature_matrix(trajectories, shock_schedules)
    per_period_features = [
        f for f in fm.feature_vocab
        if any(f.startswith(p) for p in PER_PERIOD_PREFIXES)
    ]
    full_idx = {f: j for j, f in enumerate(fm.feature_vocab)}
    n_features = len(per_period_features)
    Y = np.zeros((n_trajs * T, n_features), dtype=np.uint8)
    for ti in range(n_trajs):
        for t in range(T):
            full_row = (ti * T + t) * N + 0    # agent 0 represents the period
            for j, f in enumerate(per_period_features):
                if fm.Y[full_row, full_idx[f]] > 0:
                    Y[ti * T + t, j] = 1

    return Feed(
        name="macro_feed",
        X=torch.tensor(X, dtype=torch.float32),
        Y=Y,
        feature_vocab=per_period_features,
        sample_index=[(ti, t) for ti in range(n_trajs) for t in range(T)],
        notes=(f"per-period feed: macro({MACRO_DIM}) + shock({SHOCK_DIM}) + "
               f"mean_h1({h1_mean.shape[-1]}) = {X.shape[1]}-dim, "
               f"{X.shape[0]} samples"),
    )


SAE_CONFIGS = [
    ("jr_w256_ep200",  "jumprelu", 256,  {"l0_coeff": 1.5e-3, "init_theta": 0.05}, 200),
    ("jr_w512_ep300",  "jumprelu", 512,  {"l0_coeff": 1.5e-3, "init_theta": 0.05}, 300),
    ("topk_w256_k12",  "topk",     256,  {"k": 12},                                  200),
]


def main(seed: int = 0, n_trajectories: int = 128, n_periods: int = 100,
         sentiment_strength: float = 0.20):
    torch.manual_seed(seed)
    t_start = time.time()
    print("=" * 78)
    print(f"MACRO-FEED EXPERIMENT  (n_traj={n_trajectories} x n_periods={n_periods})")
    print("=" * 78)

    ens = generate_ensemble(n_trajectories=n_trajectories, n_periods=n_periods,
                            seed=seed, sentiment_strength=sentiment_strength)
    print(f"\n[1] Ensemble in {time.time() - t_start:.1f}s")

    wm = load_temporal_sentiment_wm()
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"[2] Loaded TemporalWorldModel ({n_params:,} params)")

    feed = build_macro_feed(ens.trajectories, ens.shock_schedules, wm)
    print(f"\n[3] Macro feed: X={tuple(feed.X.shape)}  Y={feed.Y.shape}  "
          f"vocab={len(feed.feature_vocab)} per-period features")
    print(f"    {feed.notes}")
    prev = feed.Y.mean(axis=0)
    print(f"    feature prevalence: in [5%,95%] = "
          f"{int(((prev>=0.05)&(prev<=0.95)).sum())}/{len(prev)}")

    # Regime features specifically
    print("\n    regime feature prevalence (per-period feed):")
    for j, name in enumerate(feed.feature_vocab):
        if feature_tier(name) == "regime":
            print(f"      {prev[j]:>6.2%}  {name}")

    # ---- Train SAEs ----
    results: list[dict] = []
    for name, variant, n_feat, kw, epochs in SAE_CONFIGS:
        print(f"\n[4] SAE: {name}  ({variant}, n_features={n_feat}, epochs={epochs})")
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
        print(f"   recon={recon:.4f} L0={l0:.2f} VE={ve:.4f} time={elapsed:.1f}s")

        torch.save({
            "state_dict": sae.state_dict(), "kind": variant,
            "feed_name": "macro_feed",
            "input_dim": sae.input_dim, "n_features": sae.n_features,
            "feed_config": kw,
        }, os.path.join(EXP_DIR, f"{name}.pt"))

        Z = score_sae(sae, feed.X)
        rep = align(Z, feed.Y, feed.feature_vocab)
        rep.run_id = name; rep.feed_name = "macro_feed"; rep.variant = variant
        print(f"   cov95={rep.coverage_at_0_95:.1%}  mAUC={rep.mean_best_auc:.3f}  "
              f"mono={rep.monosemanticity:.1%}")
        for tier in TIERS:
            pt = rep.per_tier[tier]
            print(f"     {tier:<11s} n={pt['n_features']:>2d}  "
                  f"cov95={pt['coverage_0.95']:>5.1%}  "
                  f"mAUC={pt['mean_best_auc']:.3f}")

        # Per-feature regime AUC (the headline)
        print("   regime AUC per feature:")
        for j, gname in enumerate(feed.feature_vocab):
            if feature_tier(gname) == "regime":
                best = float(rep.alignment[:, j].max())
                marker = ""
                if prev[j] < 0.01:
                    marker = " (rare)"
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

    # ---- Compare ----
    print("\n" + "=" * 100)
    print("MACRO-FEED SUMMARY  (per-period samples)")
    print("=" * 100)
    print(f"{'name':<24s} {'w':>5s} {'ep':>4s} {'time':>6s} {'L0':>5s} {'VE':>6s}  "
          f"{'cov95':>5s} {'mAUC':>5s}  {'regi/95':>8s}  {'regi/A':>6s}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<24s} {r['n_features']:>5d} {r['epochs']:>4d} "
              f"{r['wall_time_s']:>5.0f}s {r['l0']:>5.1f} {r['var_explained']:>6.3f}  "
              f"{r['coverage_0.95']:>5.1%} {r['mean_best_auc']:>5.3f}  "
              f"{r['per_tier']['regime']['coverage_0.95']:>8.1%}  "
              f"{r['per_tier']['regime']['mean_best_auc']:>6.3f}")

    out_path = os.path.join(RUNS_DIR, "macro_feed_experiment_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
