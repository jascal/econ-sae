"""Follow-up to regime_granularity_experiment: is the residual SAE-ALLOCATION gap
closable LABEL-FREE?

The granularity experiment found regime is linearly PRESENT in a label-free
granularity-matched representation (probe cov95 up to 72% on C3_timecat, 100% on the
engineered Cref_macro feed) yet the vanilla unsupervised JumpReLU SAE surfaces 0%
(C3) / 17% (Cref) at cov95. That residual is an SAE-objective/allocation problem, not
a substrate-absence one. This sweeps LABEL-FREE SAE-side knobs (width, TopK vs
JumpReLU, input whitening) on the SAME two feeds and asks: does any unsupervised SAE
cross cov95 ≥ 0.95 for regime — closing the gap WITHOUT the labels?

  yes -> the regime unsupervised ceiling is fully a method artifact (granularity +
         SAE capacity), and supervision is incidental on BOTH axes.
  no  -> even with the signal provably present, unsupervised reconstruction won't
         allocate it; supervision plays a genuine allocation role (the honest residual).

Run:  .venv/bin/python scripts/regime_sae_allocation_experiment.py [--seeds 0 1]
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
from econsae.sae.evaluation import align, feature_tier, score_sae
from econsae.sae.models import make_sae
from econsae.sae.train import TrainConfig, train
from econsae.sae.world_model import (
    TemporalWorldModel, WMTrainConfig, build_temporal_data,
    extract_temporal_h1_activations, train_temporal_world_model,
)
from econsae.simulator.ensemble import generate_ensemble
from macro_feed_v3_experiment import build_macro_feed_v3
from regime_granularity_experiment import (
    WINDOW, per_period_Y, ridge_lda_probe_auc, temporal_pool, zscore,
)

REGIME = "regime"

# label-free SAE-side configs: (label, kind, width, kwargs, whiten)
CONFIGS = [
    ("jr_w256",        "jumprelu", 256,  {"l0_coeff": 1e-3, "init_theta": 0.05}, False),
    ("jr_w512",        "jumprelu", 512,  {"l0_coeff": 1e-3, "init_theta": 0.05}, False),
    ("jr_w1024",       "jumprelu", 1024, {"l0_coeff": 1e-3, "init_theta": 0.05}, False),
    ("topk_w512_k8",   "topk",     512,  {"k": 8},  False),
    ("topk_w512_k16",  "topk",     512,  {"k": 16}, False),
    ("topk_w1024_k16", "topk",     1024, {"k": 16}, False),
    ("jr_w512_whiten", "jumprelu", 512,  {"l0_coeff": 1e-3, "init_theta": 0.05}, True),
    ("topk_w512_k16_whiten", "topk", 512, {"k": 16}, True),
]


def whiten(X: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """PCA-whiten (label-free): equalise variance across directions so a
    lower-variance signal like regime competes for SAE latents."""
    Xc = X - X.mean(0)
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(cov)
    W = V @ np.diag(1.0 / np.sqrt(np.clip(w, eps, None))) @ V.T
    return (Xc @ W).astype(np.float32)


def regime_scores(X, Y, vocab, kind, width, kwargs, do_whiten, seed):
    torch.manual_seed(seed)
    Xz = whiten(zscore(X)) if do_whiten else zscore(X)
    Xt = torch.tensor(Xz, dtype=torch.float32)
    sae = make_sae(kind, Xt.shape[1], width, **kwargs)
    train(sae, Xt, TrainConfig(epochs=300, batch_size=256, lr=1e-3, log_every=10**9),
          verbose=False)
    rep = align(score_sae(sae, Xt), Y, vocab)
    pt = rep.per_tier[REGIME]
    return float(pt["mean_best_auc"]), float(pt["coverage_0.95"])


def build_feeds(n_trajs, n_periods, wm_epochs, seed):
    ens = generate_ensemble(n_trajectories=n_trajs, n_periods=n_periods, seed=seed)
    trajs, scheds = ens.trajectories, ens.shock_schedules
    T, N = trajs[0].T, trajs[0].n_agents
    torch.manual_seed(seed)
    wm = TemporalWorldModel()
    train_temporal_world_model(wm, build_temporal_data(trajs, scheds),
                               WMTrainConfig(epochs=wm_epochs), verbose=False)
    H1_flat, _ = extract_temporal_h1_activations(wm, trajs, scheds)
    d = H1_flat.shape[1]
    h1_mean = H1_flat.reshape(n_trajs, T, N, d).mean(axis=2)
    fm = build_feature_matrix(trajs, scheds)
    Ypp, vocab_pp = per_period_Y(fm, n_trajs, T, N)
    feeds = {
        "C3_timecat": (temporal_pool(h1_mean, WINDOW, "concat"), Ypp, vocab_pp),
    }
    feed = build_macro_feed_v3(trajs, scheds, wm)
    feeds["Cref_macro"] = (feed.X.numpy(), feed.Y, list(feed.feature_vocab))
    return feeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_trajs, n_periods, wm_epochs = (16, 48, 20) if args.quick else (48, 80, 40)

    # probe ceiling per feed (label-using diagnostic of what's present), seed 0
    feeds0 = build_feeds(n_trajs, n_periods, wm_epochs, args.seeds[0])
    ceiling = {}
    for fname, (X, Y, vocab) in feeds0.items():
        Xz = zscore(X)
        reg_idx = [j for j, f in enumerate(vocab) if feature_tier(f) == REGIME]
        pf = [ridge_lda_probe_auc(Xz, Y[:, j], seed=args.seeds[0]) for j in reg_idx]
        ceiling[fname] = (float(np.nanmean(pf)),
                          float(np.mean([v >= 0.95 for v in pf if not np.isnan(v)])))

    results = {f: {c[0]: {"mauc": [], "cov95": []} for c in CONFIGS} for f in feeds0}
    for seed in args.seeds:
        feeds = feeds0 if seed == args.seeds[0] else build_feeds(n_trajs, n_periods, wm_epochs, seed)
        for fname, (X, Y, vocab) in feeds.items():
            for label, kind, width, kwargs, dw in CONFIGS:
                m, c = regime_scores(X, Y, vocab, kind, width, kwargs, dw, seed)
                results[fname][label]["mauc"].append(m)
                results[fname][label]["cov95"].append(c)

    print(f"\n=== label-free SAE-side sweep: can it close the regime allocation gap? "
          f"({len(args.seeds)} seeds, n_trajs={n_trajs}, n_periods={n_periods}) ===")
    for fname in feeds0:
        pm, pc = ceiling[fname]
        print(f"\n[{fname}]  probe ceiling: mAUC={pm:.3f}  cov95={pc:.1%}")
        print(f"  {'sae config':<22}{'mAUC':>14}  {'cov95':>14}")
        for label, *_ in CONFIGS:
            m = np.array(results[fname][label]["mauc"])
            c = np.array(results[fname][label]["cov95"])
            star = "  <-- crosses" if c.mean() > 0.0 else ""
            print(f"  {label:<22}{m.mean():>6.3f}±{m.std():<4.2f}  "
                  f"{c.mean():>6.1%}±{c.std():<5.1%}{star}")

    out = Path(__file__).resolve().parents[1] / "runs" / "regime_sae_allocation_summary.json"
    out.write_text(json.dumps(
        {"config": {"n_trajs": n_trajs, "n_periods": n_periods, "wm_epochs": wm_epochs,
                    "seeds": args.seeds, "window": WINDOW},
         "probe_ceiling": ceiling,
         "results": {f: {c: {"mauc": results[f][c]["mauc"], "cov95": results[f][c]["cov95"]}
                         for c in results[f]} for f in results}}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
