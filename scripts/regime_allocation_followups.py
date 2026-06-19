"""Two closing checks for the regime allocation-gap verdict.

A. l0 SWEEP — the last label-free SAE knob the width/TopK/whiten sweep didn't try.
   Does relaxing (or tightening) the JumpReLU sparsity penalty let the unsupervised
   SAE allocate a latent to the (present) regime signal? On the granularity feed (C3)
   and the engineered feed (Cref).

B. SUPERVISED-WM CONTROL — locates WHERE supervision acts. Train the regime-SUPERVISED
   world model (regime head + BCE), extract its h1, run it through the SAME granularity
   feeds + the SAME baseline SAE. If regime cov95 jumps to ~6/6 here while the
   label-free-but-present feeds plateau at <=2/6, supervision's role is confirmed to be
   WM-level REPRESENTATION SHAPING (making regime a high-variance, allocatable
   direction), not scoring-time answer injection.

Run:  .venv/bin/python scripts/regime_allocation_followups.py [--seeds 0 1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from econsae.ground_truth import build_feature_matrix
from econsae.sae.evaluation import feature_tier
from econsae.sae.world_model import WMTrainConfig, build_temporal_data
from econsae.simulator.ensemble import generate_ensemble
from macro_feed_v3_experiment import build_macro_feed_v3
from regime_granularity_experiment import WINDOW, per_period_Y, temporal_pool
from regime_sae_allocation_experiment import build_feeds, regime_scores
from regime_supervised_experiment import (
    RegimeSupervisedTemporalWM, build_regime_labels, extract_h1_from_supervised,
    train_regime_supervised_wm,
)

REGIME = "regime"
L0_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def part_a_l0_sweep(n_trajs, n_periods, wm_epochs, seeds):
    print(f"\n=== A. l0 (sparsity) sweep on label-free feeds "
          f"({len(seeds)} seeds, jumprelu w512) ===")
    out = {}
    feeds_by_seed = {s: build_feeds(n_trajs, n_periods, wm_epochs, s) for s in seeds}
    for fname in feeds_by_seed[seeds[0]]:
        print(f"\n[{fname}]  {'l0_coeff':<12}{'mAUC':>14}{'cov95':>14}")
        out[fname] = {}
        for l0 in L0_GRID:
            ms, cs = [], []
            for s in seeds:
                X, Y, vocab = feeds_by_seed[s][fname]
                m, c = regime_scores(X, Y, vocab, "jumprelu", 512,
                                     {"l0_coeff": l0, "init_theta": 0.05}, False, s)
                ms.append(m); cs.append(c)
            ms, cs = np.array(ms), np.array(cs)
            out[fname][f"{l0:g}"] = {"mauc": ms.tolist(), "cov95": cs.tolist()}
            print(f"  {l0:<12g}{ms.mean():>6.3f}±{ms.std():<4.2f}  "
                  f"{cs.mean():>6.1%}±{cs.std():<5.1%}")
    return out


def supervised_h1(n_trajs, n_periods, wm_epochs, seed):
    """Train the regime-supervised WM (published recipe) and return its h1 +
    the ensemble, so we can feed it through the SAME granularity pipeline."""
    torch.manual_seed(seed)
    ens = generate_ensemble(n_trajectories=n_trajs, n_periods=n_periods, seed=seed,
                            sentiment_strength=0.20)
    trajs, scheds = ens.trajectories, ens.shock_schedules
    wm_data = build_temporal_data(trajs, scheds)
    Yreg, _ = build_regime_labels(trajs, scheds)
    model = RegimeSupervisedTemporalWM(embed_dim=64, n_heads=4, n_attn_layers=1,
                                       gru_hidden=128, n_gru_layers=1,
                                       h1_dim=192, h2_dim=128)
    X, S = wm_data.X, wm_data.states
    xm = X.reshape(-1, X.shape[-1]).mean(0); xs = X.reshape(-1, X.shape[-1]).std(0).clamp_min(1e-3)
    ym = S.reshape(-1, S.shape[-1]).mean(0); ys = S.reshape(-1, S.shape[-1]).std(0).clamp_min(1e-3)
    model.x_mean.copy_(xm); model.x_std.copy_(xs)
    model.y_mean.copy_(ym); model.y_std.copy_(ys)
    train_regime_supervised_wm(model, (X - xm) / xs, (S - ym) / ys, Yreg,
                               WMTrainConfig(epochs=wm_epochs, batch_size=16),
                               regime_weight=1.0, verbose=False)
    H1, _ = extract_h1_from_supervised(model, trajs, scheds)
    return ens, H1


def part_b_supervised_control(n_trajs, n_periods, wm_epochs, seed):
    print(f"\n=== B. supervised-WM control (regime-supervised h1 through the SAME "
          f"pipeline, seed {seed}) ===")
    ens, H1 = supervised_h1(n_trajs, n_periods, wm_epochs, seed)
    trajs, scheds = ens.trajectories, ens.shock_schedules
    T, N = trajs[0].T, trajs[0].n_agents
    d = H1.shape[1]
    h1_mean = H1.reshape(n_trajs, T, N, d).mean(axis=2)
    fm = build_feature_matrix(trajs, scheds)
    Ypp, vocab_pp = per_period_Y(fm, n_trajs, T, N)
    feeds = {
        "C0_raw_sup": (H1, fm.Y, list(fm.feature_vocab)),
        "C3_timecat_sup": (temporal_pool(h1_mean, WINDOW, "concat"), Ypp, vocab_pp),
    }
    print(f"  {'feed (supervised h1)':<22}{'mAUC':>10}{'cov95':>10}")
    out = {}
    for fname, (X, Y, vocab) in feeds.items():
        m, c = regime_scores(X, Y, vocab, "jumprelu", 512,
                             {"l0_coeff": 1e-3, "init_theta": 0.05}, False, seed)
        out[fname] = {"mauc": m, "cov95": c}
        print(f"  {fname:<22}{m:>6.3f}    {c:>6.1%}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_trajs, n_periods, wm_epochs = (16, 48, 20) if args.quick else (48, 80, 40)

    a = part_a_l0_sweep(n_trajs, n_periods, wm_epochs, args.seeds)
    b = part_b_supervised_control(n_trajs, n_periods, wm_epochs, args.seeds[0])

    out = Path(__file__).resolve().parents[1] / "runs" / "regime_allocation_followups_summary.json"
    out.write_text(json.dumps({"l0_sweep": a, "supervised_control": b,
                               "config": {"n_trajs": n_trajs, "n_periods": n_periods,
                                          "wm_epochs": wm_epochs, "seeds": args.seeds}},
                              indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
