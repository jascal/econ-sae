"""N1 mechanism ablation for the cov95 forge tax — econ-sae port.

Cross-substrate replication of bio-sae's N1 (bio-sae/scripts/forge_cov_mechanism.py).
bio-sae found the cov95 tax is an EMERGENT forward-pass distortion: rank /
over-completeness was EXONERATED (the frozen ESM-2 forge keeps full rank yet
smears; a rank-128 projection of host keeps cov95 0.685 vs host 0.717). The
prediction for econ-sae (docs note + the manifesto cross-substrate read) is the
OPPOSITE: econ's host is a *trainable* DualHeadRegimeWM whose SAE sits on a single
dense fc1/fc2 bridge at a 192-d bottleneck, and its own forge MSE blows up with
over-completeness (w256 MSE 5.3 -> w1024 MSE 443). So rank/over-completeness should
be the DOMINANT knob here, not exonerated.

Probes (per tier: categorical / regime), host = DualHeadRegimeWM + jr SAE on the
macro feed (per-period, 223-d, attention + LayerNorm + GRU host):
  - widths : host cov95 vs SAE over-completeness (jr_w512/1024/2048 = 2.3x/4.6x/9.2x)
  - rank   : project host onto top-r decoder-atom subspace, sweep r
  - LN     : one LayerNorm on the host activation
(TopK sweep is bio-only: these SAEs are JumpReLU, no k.)

Training-free, deterministic. Run from the econ-sae repo root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)


def _build_feed(n_traj, n_periods):
    import torch
    from econsae.simulator.ensemble import generate_ensemble
    from scripts.regime_dual_head_experiment import DualHeadRegimeWM
    from scripts.macro_feed_v3_experiment import build_macro_feed_v3

    ck = torch.load("runs/world_model_regime_dual_head.pt", map_location="cpu", weights_only=False)
    wm = DualHeadRegimeWM(**ck["config"])
    wm.load_state_dict(ck["state_dict"])
    wm.eval()
    ens = generate_ensemble(n_trajectories=n_traj, n_periods=n_periods, seed=0,
                            sentiment_strength=0.20)
    return build_macro_feed_v3(ens.trajectories, ens.shock_schedules, wm)


def _per_tier(sae, X, feed):
    from econsae.sae.evaluation import align, score_sae
    import torch

    rep = align(score_sae(sae, torch.as_tensor(X, dtype=torch.float32)), feed.Y, feed.feature_vocab)
    out = {"all": {"cov95": rep.coverage_at_0_95, "mauc": rep.mean_best_auc}}
    for t, d in rep.per_tier.items():
        if d["n_features"]:
            out[t] = {"cov95": d["coverage_0.95"], "mauc": d["mean_best_auc"], "n": d["n_features"]}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-traj", type=int, default=64)
    p.add_argument("--n-periods", type=int, default=100)
    p.add_argument("--sae", default="runs/regime_dual_head_experiment/jr_w512_ep300.pt")
    p.add_argument("--width-ckpts",
                   default="runs/regime_dual_head_experiment/jr_w512_ep300.pt,"
                           "runs/regime_dual_head_experiment/jr_w1024_ep300.pt,"
                           "runs/regime_dual_head_experiment/jr_w2048_ep300.pt")
    p.add_argument("--output", default="runs/cov_mechanism_summary.json")
    args = p.parse_args(argv)

    from econsae.sae.forge_bridge import load_sae

    print(f"[1] build macro feed ({args.n_traj}x{args.n_periods} ensemble)")
    feed = _build_feed(args.n_traj, args.n_periods)
    X = feed.X.detach().cpu().numpy().astype(np.float32)
    print(f"    X={X.shape}  Y={feed.Y.shape}  vocab={len(feed.feature_vocab)}")

    out = {"experiment": "N1 mechanism ablation (econ-sae, DualHeadRegimeWM + jr SAE, macro feed)",
           "n_samples": int(X.shape[0]), "d_model": int(X.shape[1])}

    # ---- over-completeness sweep: host cov95 vs SAE width ----
    print("[N1-width] host cov95 vs SAE over-completeness")
    width_rows = []
    for ck in args.width_ckpts.split(","):
        sae, meta = load_sae(ck)
        st = _per_tier(sae, X, feed)
        oc = meta["n_features"] / meta["input_dim"]
        width_rows.append({"ckpt": os.path.basename(ck), "n_features": meta["n_features"],
                           "over_complete": round(oc, 2), **st})
        print(f"    F={meta['n_features']:>4} ({oc:.1f}x)  cov95={st['all']['cov95']:.3f} "
              f"mAUC={st['all']['mauc']:.3f}  "
              + "  ".join(f"{t}={st[t]['cov95']:.2f}" for t in st if t != "all"))
    out["N1_width"] = width_rows

    # ---- rank + LN probes on the reference SAE ----
    sae, meta = load_sae(args.sae)
    W_dec = sae.W_dec.detach().cpu().numpy().astype(np.float64)   # (d_model, n_features); atoms = columns
    d_model = W_dec.shape[0]
    host = _per_tier(sae, X, feed)
    out["host"] = host
    print(f"[host] {os.path.basename(args.sae)}  cov95={host['all']['cov95']:.3f} "
          f"mAUC={host['all']['mauc']:.3f}")

    print("[N1-rank] project host onto top-r decoder-atom subspace")
    norms = np.linalg.norm(W_dec, axis=0)
    order = np.argsort(-norms)
    rank_rows = []
    for r in [4, 8, 16, 32, 64, 128, d_model]:
        A = W_dec[:, order[:r]]                                   # (d_model, r)
        Q, _ = np.linalg.qr(A)
        Xp = (X @ (Q @ Q.T)).astype(np.float32)
        st = _per_tier(sae, Xp, feed)
        rank_rows.append({"r": r, "rank": int(Q.shape[1]), **st})
        print(f"    r={r:>4} (rank {Q.shape[1]:>3})  cov95={st['all']['cov95']:.3f}  "
              + "  ".join(f"{t}={st[t]['cov95']:.2f}" for t in st if t != "all"))
    out["N1_rank"] = rank_rows

    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    ln = _per_tier(sae, ((X - mu) / (sd + 1e-5)).astype(np.float32), feed)
    out["N1_layernorm"] = ln
    print(f"[N1-LN]   one LayerNorm on host   cov95={ln['all']['cov95']:.3f}")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
