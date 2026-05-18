"""Phase 4.1: macro-feed v3 with ratio inputs.

The Phase 2.0 macro-feed v2 closed `phase:high_leverage`,
`phase:fiscal_active`, and `phase:monetary_active` at AUC >= 0.95 but
plateaued at 0.78-0.82 on `phase:expansion`, `phase:contraction`, and
`phase:high_rate`. The Phase 3 GatedSAE experiment ruled out
activation-shape mismatch as the cause.

The remaining diagnosis: the plateaued features are **ratios** of input
dimensions:
  phase:expansion       <=>  GDP[t] / mean(GDP[t-4..t-1]) > 1.10
  phase:contraction     <=>  GDP[t] / mean(GDP[t-4..t-1]) < 0.90
  phase:high_rate       <=>  rate[t] - base_rate > 1.5 * step
  phase:high_leverage   <=>  debt_outstanding / money_stock > 0.10

ReLU / Heaviside features can't natively compute division, so even with
GDP[t-4..t] explicit in the input the SAE has to learn the ratio via
many feature combinations.

This script tests the obvious fix: pre-compute the ratios as additional
input dimensions, so the SAE only has to learn a threshold on a single
input dim. Per-period sample input now includes (on top of v2):

  + gdp_deviation = (GDP[t] - mean(GDP[t-4..t-1])) / mean(GDP[t-4..t-1])
  + rate_deviation = interest_rate[t] - base_rate
  + leverage_ratio = debt_outstanding[t] / money_stock[t]
  + inflation     = (price[t] - price[t-1]) / price[t-1]
                    (using mean firm price as a price index)

If the diagnosis is right, all three plateaued features should cross
AUC 0.95.

Output:
  runs/macro_feed_v3_experiment/{cfg}.pt
  runs/macro_feed_v3_experiment_summary.json
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
    MACRO_DIM, SHOCK_DIM, encode_macros, encode_shock,
    extract_temporal_h1_activations,
)
from econsae.simulator.ensemble import generate_ensemble
from scripts.macro_feed_v2_experiment import load_temporal_sentiment_wm


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
EXP_DIR = os.path.join(RUNS_DIR, "macro_feed_v3_experiment")
os.makedirs(EXP_DIR, exist_ok=True)

GDP_WINDOW = 5
PER_PERIOD_PREFIXES = ("phase:", "shock_period_has:", "txn_period_has:")
BASE_RATE = 0.02


def build_macro_feed_v3(trajectories, shock_schedules, wm) -> Feed:
    n_trajs = len(trajectories)
    T = trajectories[0].T
    N = trajectories[0].n_agents

    H1_flat, idx = extract_temporal_h1_activations(wm, trajectories, shock_schedules)
    H1 = H1_flat.reshape(n_trajs, T, N, -1)
    h1_mean = H1.mean(axis=2)                                  # (n_trajs, T, h1_dim)

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

    # ---- v2 extras: GDP window + impulse flags ----
    gdp_window = np.zeros((n_trajs, T, GDP_WINDOW), dtype=np.float32)
    monetary_flag = np.zeros((n_trajs, T, 1), dtype=np.float32)
    fiscal_flag = np.zeros((n_trajs, T, 1), dtype=np.float32)
    for ti, traj in enumerate(trajectories):
        gdps = np.array([m["GDP"] for m in traj.macros], dtype=np.float32)
        for t in range(T):
            window = []
            for k in range(GDP_WINDOW - 1, -1, -1):
                src = max(0, t - k)
                window.append(gdps[src])
            gdp_window[ti, t] = np.array(window, dtype=np.float32)
    for ti, sched in enumerate(shock_schedules):
        for t, kinds in enumerate(sched.kinds):
            if "monetary" in kinds:
                monetary_flag[ti, t, 0] = 1.0
            if "fiscal" in kinds:
                fiscal_flag[ti, t, 0] = 1.0

    # ---- v3 NEW: engineered ratios ----
    gdp_dev = np.zeros((n_trajs, T, 1), dtype=np.float32)
    rate_dev = np.zeros((n_trajs, T, 1), dtype=np.float32)
    leverage_ratio = np.zeros((n_trajs, T, 1), dtype=np.float32)
    inflation = np.zeros((n_trajs, T, 1), dtype=np.float32)
    # Need a per-period mean firm price for inflation. Reconstruct from
    # the per-agent state coords: firm.price lives at agent vec[COORD_IDX["price"]].
    from econsae.embeddings import COORD_IDX
    price_idx = COORD_IDX["price"]
    sector_idx = COORD_IDX["sector"]
    for ti, traj in enumerate(trajectories):
        states = traj.stack_states()                          # (T, N, DIM)
        # Mean firm price per period (sector == SECTOR_FIRM = 1)
        firm_mask = states[..., sector_idx] == 1.0
        prices_per_period = []
        for t in range(T):
            row_prices = states[t, firm_mask[t], price_idx]
            prices_per_period.append(float(row_prices.mean()) if len(row_prices) else 1.0)
        for t in range(T):
            # Output gap = (GDP[t] - mean(GDP[max(0,t-K)..t-1])) / mean(...)
            lo = max(0, t - GDP_WINDOW)
            window_data = [traj.macros[s]["GDP"] for s in range(lo, t)]
            trailing = float(np.mean(window_data)) if window_data else float(traj.macros[t]["GDP"])
            cur_gdp = traj.macros[t]["GDP"]
            gdp_dev[ti, t, 0] = (cur_gdp - trailing) / max(trailing, 1e-9)
            rate_dev[ti, t, 0] = traj.macros[t]["interest_rate"] - BASE_RATE
            leverage_ratio[ti, t, 0] = (
                traj.macros[t]["debt_outstanding"]
                / max(traj.macros[t]["money_stock"], 1e-9)
            )
            if t == 0:
                inflation[ti, t, 0] = 0.0
            else:
                inflation[ti, t, 0] = (
                    (prices_per_period[t] - prices_per_period[t - 1])
                    / max(prices_per_period[t - 1], 1e-9)
                )

    X_pt = np.concatenate(
        [macro_vecs, shock_vecs, gdp_window, monetary_flag, fiscal_flag,
         gdp_dev, rate_dev, leverage_ratio, inflation, h1_mean],
        axis=-1,
    )
    X = X_pt.reshape(n_trajs * T, -1).astype(np.float32)

    # Z-score normalization (same approach as v2, critical for mixed-scale dims)
    mu = X.mean(axis=0)
    sd = X.std(axis=0).clip(min=1e-6)
    X = ((X - mu) / sd).astype(np.float32)

    # Per-period labels (same as v2)
    fm = build_feature_matrix(trajectories, shock_schedules)
    per_period_features = [
        f for f in fm.feature_vocab
        if any(f.startswith(p) for p in PER_PERIOD_PREFIXES)
    ]
    full_idx = {f: j for j, f in enumerate(fm.feature_vocab)}
    Y = np.zeros((n_trajs * T, len(per_period_features)), dtype=np.uint8)
    for ti in range(n_trajs):
        for t in range(T):
            full_row = (ti * T + t) * N + 0
            for j, f in enumerate(per_period_features):
                if fm.Y[full_row, full_idx[f]] > 0:
                    Y[ti * T + t, j] = 1

    return Feed(
        name="macro_feed_v3",
        X=torch.tensor(X, dtype=torch.float32),
        Y=Y,
        feature_vocab=per_period_features,
        sample_index=[(ti, t) for ti in range(n_trajs) for t in range(T)],
        notes=(f"macro({MACRO_DIM}) + shock({SHOCK_DIM}) + gdp_window({GDP_WINDOW}) + "
               f"monetary/fiscal_flags(2) + gdp_dev + rate_dev + leverage + inflation(4)"
               f" + mean_h1({h1_mean.shape[-1]}) = {X.shape[1]}-dim, z-scored, "
               f"{X.shape[0]} samples"),
    )


SAE_CONFIGS = [
    ("jr_w256_ep200",  "jumprelu", 256,  {"l0_coeff": 1e-3, "init_theta": 0.05}, 200),
    ("jr_w512_ep300",  "jumprelu", 512,  {"l0_coeff": 8e-4, "init_theta": 0.05}, 300),
    ("jr_w1024_ep300", "jumprelu", 1024, {"l0_coeff": 6e-4, "init_theta": 0.05}, 300),
]


def main(seed: int = 0, n_trajectories: int = 128, n_periods: int = 100,
         sentiment_strength: float = 0.20):
    torch.manual_seed(seed)
    t_start = time.time()
    print("=" * 78)
    print("MACRO-FEED V3 EXPERIMENT  (engineered ratio inputs)")
    print("=" * 78)

    ens = generate_ensemble(n_trajectories=n_trajectories, n_periods=n_periods,
                            seed=seed, sentiment_strength=sentiment_strength)
    print(f"\n[1] Ensemble in {time.time() - t_start:.1f}s")

    wm = load_temporal_sentiment_wm()
    feed = build_macro_feed_v3(ens.trajectories, ens.shock_schedules, wm)
    print(f"[2] Feed: X={tuple(feed.X.shape)}  Y={feed.Y.shape}  "
          f"vocab={len(feed.feature_vocab)}")
    print(f"    {feed.notes}")

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
        train(sae, feed.X, tcfg, verbose=False)
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
            "feed_name": "macro_feed_v3",
            "input_dim": sae.input_dim, "n_features": sae.n_features,
            "feed_config": kw,
        }, os.path.join(EXP_DIR, f"{name}.pt"))

        Z = score_sae(sae, feed.X)
        rep = align(Z, feed.Y, feed.feature_vocab)
        rep.run_id = name; rep.feed_name = "macro_feed_v3"; rep.variant = variant
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
    print("MACRO-FEED V3 SUMMARY  (engineered ratios)")
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

    out_path = os.path.join(RUNS_DIR, "macro_feed_v3_experiment_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
