"""Phase 10: does the headline SAE-recovery result hold under *calibrated*
macro dynamics?

Runs the dual-head regime pipeline (Phase 6.2 / 9.2.1) twice under an
identical training configuration, varying ONLY the simulator dynamics:

  baseline   -- SimConfig.default()         (the synthetic-shock regime the
                                              Phase 1-9 benchmark used)
  calibrated -- configs/calibrated_macro.json (shock vols / impulse probs /
                                              policy-rate level fit to the
                                              vendored US macro moments)

Both arms hold the non-Markov regime mechanism fixed (sentiment_strength =
0.20, as in the headline runs); only the calibrated shock parameters differ.
Each arm: generate ensemble -> build GT labels (with that arm's rate
thresholds) -> train a DualHeadRegimeWM -> extract its h1 -> fit a JumpReLU
SAE on h1 -> score GT-AUC by tier. We then compare regime / conjunctive tier
mAUC across arms. The scientific question: when shock frequencies and
volatilities match history (which shifts regime-label prevalence), do the
supervised regime features stay recovered?

Outputs:
    runs/calibration/phase10_benchmark_summary.json

Usage:
    python scripts/phase10_calibrated_benchmark.py            # default budget
    python scripts/phase10_calibrated_benchmark.py --quick    # fast sanity run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import torch

from econsae.calibration import SimConfig, compute_moments
from econsae.calibration.moments import macros_from_ensemble
from econsae.ground_truth import build_feature_matrix
from econsae.sae.data import Feed
from econsae.sae.evaluation import align, score_sae, report_to_dict, TIERS, feature_tier
from econsae.sae.models import make_sae
from econsae.sae.train import TrainConfig, train
from econsae.sae.world_model import (
    WMTrainConfig, build_temporal_data, extract_temporal_h1_activations,
)
from econsae.simulator.ensemble import generate_ensemble
from scripts.regime_dual_head_experiment import DualHeadRegimeWM, train_dual_head_wm
from scripts.regime_supervised_experiment import REGIME_LABEL_ORDER

RUNS_DIR = os.path.join(REPO_ROOT, "runs")
REPORT_DIR = os.path.join(RUNS_DIR, "calibration")
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "calibrated_macro.json")

# Held fixed across both arms -- the non-Markov regime mechanism is part of
# the benchmark setup, not a thing we calibrate.
SENTIMENT_STRENGTH = 0.20


def _regime_labels_from_fm(fm, n_traj: int, T: int, N: int) -> torch.Tensor:
    """Extract (n_traj, T, n_regime) period-level labels from a GT matrix.

    Local copy of `build_regime_labels` that reuses an already-built feature
    matrix (so the calibrated rate thresholds baked into `fm` carry through).
    """
    full_idx = {f: j for j, f in enumerate(fm.feature_vocab)}
    Y = np.zeros((n_traj, T, len(REGIME_LABEL_ORDER)), dtype=np.float32)
    for ti in range(n_traj):
        for t in range(T):
            full_row = (ti * T + t) * N + 0          # labels identical across agents
            for j, name in enumerate(REGIME_LABEL_ORDER):
                if name in full_idx and fm.Y[full_row, full_idx[name]] > 0:
                    Y[ti, t, j] = 1.0
    return torch.tensor(Y, dtype=torch.float32)


def run_arm(label: str, cfg: SimConfig, *, seed: int, n_traj: int, n_periods: int,
            wm_epochs: int, sae_epochs: int) -> dict:
    print("\n" + "=" * 78)
    print(f"ARM: {label}")
    print("=" * 78)
    t_start = time.time()
    torch.manual_seed(seed)

    base_rate = cfg.shock.base_interest_rate
    monetary_step = cfg.shock.monetary_step

    # ---- 1. ensemble under this arm's dynamics ----
    ens = generate_ensemble(n_trajectories=n_traj, n_periods=n_periods,
                            seed=seed, sim_config=cfg)
    moments = compute_moments(macros_from_ensemble(ens))
    print(f"[1] ensemble {n_traj}x{n_periods}  "
          f"gdp_vol={moments['gdp_growth_vol']:.4f}  "
          f"rate_mean={moments['fedfunds_mean']:.4f}  "
          f"recession_freq={moments['recession_freq']:.3f}")

    # ---- 2. GT labels with this arm's rate thresholds ----
    fm = build_feature_matrix(ens.trajectories, ens.shock_schedules,
                              base_rate=base_rate, monetary_step=monetary_step)
    N = ens.trajectories[0].n_agents
    Yreg = _regime_labels_from_fm(fm, n_traj, n_periods, N)
    reg_prev = {name: float(Yreg[:, :, j].mean())
                for j, name in enumerate(REGIME_LABEL_ORDER)}
    print(f"[2] regime label prevalence: "
          + "  ".join(f"{n.split(':')[-1]}={p:.2%}" for n, p in reg_prev.items()))

    # ---- 3. train DualHeadRegimeWM (rate thresholds -> encode_shock too) ----
    wm_data = build_temporal_data(ens.trajectories, ens.shock_schedules,
                                  base_rate=base_rate)
    model = DualHeadRegimeWM(embed_dim=64, n_heads=4, n_attn_layers=1,
                             gru_hidden=128, n_gru_layers=1, h1_dim=192, h2_dim=128)
    X = wm_data.X; S = wm_data.states
    X_flat = X.reshape(-1, X.shape[-1]); S_flat = S.reshape(-1, S.shape[-1])
    x_mean = X_flat.mean(dim=0); x_std = X_flat.std(dim=0).clamp_min(1e-3)
    y_mean = S_flat.mean(dim=0); y_std = S_flat.std(dim=0).clamp_min(1e-3)
    model.x_mean.copy_(x_mean); model.x_std.copy_(x_std)
    model.y_mean.copy_(y_mean); model.y_std.copy_(y_std)
    Xn = (X - x_mean) / x_std; Sn = (S - y_mean) / y_std
    t0 = time.time()
    train_dual_head_wm(model, Xn, Sn, Yreg,
                       WMTrainConfig(epochs=wm_epochs, batch_size=16),
                       alpha_channel=1.0, beta_pooled=1.0, focal_gamma=2.0,
                       verbose=False)
    print(f"[3] trained DualHeadRegimeWM ({wm_epochs} ep) in {time.time()-t0:.1f}s")

    # ---- 4. extract h1, fit SAE ----
    H1, idx = extract_temporal_h1_activations(model, ens.trajectories,
                                              ens.shock_schedules, base_rate=base_rate)
    assert idx == fm.sample_index, "activation index mismatch"
    feed = Feed(name=f"phase10_{label}", X=torch.tensor(H1, dtype=torch.float32),
                Y=fm.Y, feature_vocab=fm.feature_vocab, sample_index=fm.sample_index,
                notes=f"Phase 10 {label} arm: DualHeadRegimeWM h1 (192-d).")
    torch.manual_seed(seed)
    sae = make_sae("jumprelu", feed.D, 512, l0_coeff=1.5e-3, init_theta=0.05)
    t0 = time.time()
    train(sae, feed.X, TrainConfig(epochs=sae_epochs, batch_size=512, lr=1e-3,
                                   warmup_steps=50, resample_every=max(100, sae_epochs // 5),
                                   log_every=10**6), verbose=False)
    Z = score_sae(sae, feed.X)
    rep = align(Z, feed.Y, feed.feature_vocab)
    print(f"[4] SAE fit ({sae_epochs} ep) in {time.time()-t0:.1f}s  "
          f"cov95={rep.coverage_at_0_95:.1%}  mAUC={rep.mean_best_auc:.3f}")
    for tier in TIERS:
        pt = rep.per_tier[tier]
        print(f"      {tier:<11s} n={pt['n_features']:>2d}  "
              f"cov95={pt['coverage_0.95']:>5.1%}  mAUC={pt['mean_best_auc']:.3f}")

    regime_auc = {feed.feature_vocab[j]: float(rep.alignment[:, j].max())
                  for j in range(len(feed.feature_vocab))
                  if feature_tier(feed.feature_vocab[j]) == "regime"}

    return {
        "label": label,
        "config": cfg.to_dict(),
        "base_rate": base_rate, "monetary_step": monetary_step,
        "moments": moments,
        "regime_label_prevalence": reg_prev,
        "per_tier": report_to_dict(rep)["per_tier"],
        "coverage_0.95": rep.coverage_at_0_95,
        "mean_best_auc": rep.mean_best_auc,
        "regime_auc_per_feature": regime_auc,
        "wall_seconds": time.time() - t_start,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fast sanity run (smaller ensemble + fewer epochs)")
    ap.add_argument("--config", default=CONFIG_PATH)
    args = ap.parse_args()

    if args.quick:
        budget = dict(n_traj=24, n_periods=80, wm_epochs=20, sae_epochs=60)
    else:
        budget = dict(n_traj=64, n_periods=100, wm_epochs=40, sae_epochs=100)

    if not os.path.exists(args.config):
        sys.exit(f"calibrated config not found: {args.config}\n"
                 f"run `python scripts/calibrate.py` first.")
    calibrated = SimConfig.from_json(args.config)

    # Hold the regime mechanism fixed across arms.
    base_arm = SimConfig.default().with_overrides(sentiment_strength=SENTIMENT_STRENGTH)
    calib_arm = calibrated.with_overrides(sentiment_strength=SENTIMENT_STRENGTH)

    print("=" * 78)
    print(f"PHASE 10 CALIBRATED BENCHMARK  {'(quick)' if args.quick else ''}")
    print(f"  budget: {budget}")
    print("=" * 78)

    t_start = time.time()
    results = [
        run_arm("baseline", base_arm, seed=0, **budget),
        run_arm("calibrated", calib_arm, seed=0, **budget),
    ]

    # ---- comparison ----
    base_r, calib_r = results
    tier_delta = {
        tier: {
            "baseline_mAUC": base_r["per_tier"][tier]["mean_best_auc"],
            "calibrated_mAUC": calib_r["per_tier"][tier]["mean_best_auc"],
            "delta": calib_r["per_tier"][tier]["mean_best_auc"]
                     - base_r["per_tier"][tier]["mean_best_auc"],
        }
        for tier in TIERS
    }

    print("\n" + "=" * 78)
    print("PHASE 10 COMPARISON  (tier mean-best-AUC: baseline -> calibrated)")
    print("=" * 78)
    print(f"{'tier':<12s} {'baseline':>9s} {'calibrated':>11s} {'delta':>8s}")
    print("-" * 44)
    for tier in TIERS:
        d = tier_delta[tier]
        print(f"{tier:<12s} {d['baseline_mAUC']:>9.3f} {d['calibrated_mAUC']:>11.3f} "
              f"{d['delta']:>+8.3f}")

    summary = {
        "budget": budget,
        "sentiment_strength": SENTIMENT_STRENGTH,
        "arms": results,
        "tier_comparison": tier_delta,
        "wall_seconds": time.time() - t_start,
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, "phase10_benchmark_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nwrote {out_path}")
    print(f"total wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
