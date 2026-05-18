"""Phase 6.1: unified-recipe regime supervision -- per-channel BCE with
class-balanced pos_weight.

Phase 5.1 + 5.2 together covered 6/6 regime features but needed two
separate training runs with different supervision shapes:
  - 5.1 (pooled BCE)     was best on impulse + current-state features.
  - 5.2 (per-channel BCE) was best on windowed features.

In 5.2 the rare-impulse channels (fiscal_active, monetary_active at
~10% prevalence) never learned: each channel's BCE gradient was
dominated by the negative class, the bias term went large-negative,
and the channel reduced to "always predict 0" -- training-time AUC
0.50.

The textbook fix is `pos_weight` in BCEWithLogitsLoss: each positive
example contributes `pos_weight[j]` times as much loss as a negative
example. Setting `pos_weight[j] = N_neg[j] / N_pos[j] = (1-prev[j]) / prev[j]`
balances the gradient signal across the prevalence axis.

This experiment tests whether per-channel BCE + class-balanced
pos_weight produces a SINGLE training recipe that hits 6/6 regime
features at AUC >= 0.95 -- removing the need for the Phase 5.1 + 5.2
union.

Predictions:
  - All 6 channels learn during training (no more 0.50 AUC on rare ones).
  - phase:expansion / contraction stay at the Phase 5.2 levels (~0.97).
  - phase:fiscal_active / monetary_active climb from the Phase 5.2 dip
    (0.92-0.99) back up to Phase 5.1 levels (~0.99-1.000).
  - phase:high_leverage / high_rate stay at Phase 5.1 levels (~0.99-1.0).

Output:
  runs/world_model_regime_unified.pt
  runs/world_model_regime_unified_acts.npz
  runs/regime_unified_experiment/{cfg}.pt
  runs/regime_unified_experiment_summary.json
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
import torch.nn as nn
import torch.nn.functional as F

from econsae.ground_truth import build_feature_matrix
from econsae.sae.data import Feed
from econsae.sae.evaluation import (
    align, score_sae, report_to_dict, TIERS, feature_tier, _auc_one_vs_many,
)
from econsae.sae.models import make_sae
from econsae.sae.train import TrainConfig, train
from econsae.sae.world_model import (
    WMTrainConfig, build_temporal_data,
)
from econsae.simulator.ensemble import generate_ensemble
from scripts.macro_feed_v3_experiment import build_macro_feed_v3
from scripts.regime_supervised_experiment import (
    REGIME_LABEL_ORDER, build_regime_labels,
)
from scripts.regime_bottleneck_experiment import (
    RegimeBottleneckedTemporalWM, extract_h1_from_bottlenecked,
)


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
EXP_DIR = os.path.join(RUNS_DIR, "regime_unified_experiment")
os.makedirs(EXP_DIR, exist_ok=True)


def train_unified_wm(
    model: RegimeBottleneckedTemporalWM,
    Xn: torch.Tensor,
    Sn: torch.Tensor,
    Yreg: torch.Tensor,
    cfg: WMTrainConfig,
    regime_weight: float = 1.0,
    pos_weight_clip: tuple[float, float] = (1.0, 20.0),
    verbose: bool = True,
) -> dict:
    """Same architecture as RegimeBottleneckedTemporalWM but with class-balanced
    per-channel BCE: pos_weight[j] = (1 - prev[j]) / prev[j], clipped."""
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)
    history_mse: list[float] = []
    history_bce: list[float] = []

    # Per-channel pos_weight derived from training-set prevalence.
    prev_per_label = Yreg.mean(dim=(0, 1)).clamp_min(1e-9)
    pos_weight = ((1.0 - prev_per_label) / prev_per_label).clamp(
        min=pos_weight_clip[0], max=pos_weight_clip[1]
    ).to(device)
    if verbose:
        prev_list = prev_per_label.cpu().tolist()
        pw_list = pos_weight.cpu().tolist()
        for name, p, w in zip(REGIME_LABEL_ORDER, prev_list, pw_list):
            print(f"  {name:<24s} prevalence={p:.3f}  pos_weight={w:.2f}")

    B_total = Xn.shape[0]
    N = Xn.shape[2]
    bs = max(1, cfg.batch_size)

    # Broadcast period-level labels to per-(period, agent)
    Yreg_per_agent = Yreg.unsqueeze(2).expand(-1, -1, N, -1).contiguous()

    for ep in range(cfg.epochs):
        perm = torch.randperm(B_total, device=device)
        mse_losses = []
        bce_losses = []
        for i in range(0, B_total, bs):
            bi = perm[i:i + bs]
            pred, regime_logits = model(Xn[bi], return_regime=True)
            mse = F.mse_loss(pred[:, :-1], Sn[bi][:, 1:])
            bce = F.binary_cross_entropy_with_logits(
                regime_logits, Yreg_per_agent[bi],
                pos_weight=pos_weight,
            )
            loss = mse + regime_weight * bce
            opt.zero_grad(); loss.backward(); opt.step()
            mse_losses.append(float(mse))
            bce_losses.append(float(bce))
        history_mse.append(float(np.mean(mse_losses)))
        history_bce.append(float(np.mean(bce_losses)))
        if verbose and (ep == 0 or (ep + 1) % max(1, cfg.epochs // 10) == 0):
            print(f"  ep {ep + 1:>3d}/{cfg.epochs}  mse={history_mse[-1]:.4f}  "
                  f"bce={history_bce[-1]:.4f}")
    return {"history_mse": history_mse, "history_bce": history_bce,
            "pos_weight": pos_weight.cpu().tolist()}


SAE_CONFIGS = [
    ("jr_w256_ep200",  "jumprelu", 256,  {"l0_coeff": 1e-3, "init_theta": 0.05}, 200),
    ("jr_w512_ep300",  "jumprelu", 512,  {"l0_coeff": 8e-4, "init_theta": 0.05}, 300),
]


def main(seed: int = 0, n_trajectories: int = 128, n_periods: int = 100,
         sentiment_strength: float = 0.20,
         wm_epochs: int = 50, wm_batch_size: int = 16,
         regime_weight: float = 1.0):
    torch.manual_seed(seed)
    t_start = time.time()
    print("=" * 78)
    print(f"REGIME-UNIFIED EXPERIMENT  (per-channel + pos_weight BCE)")
    print("=" * 78)

    ens = generate_ensemble(n_trajectories=n_trajectories, n_periods=n_periods,
                            seed=seed, sentiment_strength=sentiment_strength)
    print(f"\n[1] Ensemble in {time.time() - t_start:.1f}s")

    wm_data = build_temporal_data(ens.trajectories, ens.shock_schedules)
    Yreg, label_order = build_regime_labels(ens.trajectories, ens.shock_schedules)
    print(f"[2] Data: X={tuple(wm_data.X.shape)}  Yreg={tuple(Yreg.shape)}")

    model = RegimeBottleneckedTemporalWM(
        embed_dim=64, n_heads=4, n_attn_layers=1,
        gru_hidden=128, n_gru_layers=1, h1_dim=192, h2_dim=128,
    )
    print(f"\n[3] RegimeBottleneckedTemporalWM (pos_weight BCE): "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # Normalize
    X = wm_data.X; S = wm_data.states
    X_flat = X.reshape(-1, X.shape[-1]); S_flat = S.reshape(-1, S.shape[-1])
    x_mean = X_flat.mean(dim=0); x_std = X_flat.std(dim=0).clamp_min(1e-3)
    y_mean = S_flat.mean(dim=0); y_std = S_flat.std(dim=0).clamp_min(1e-3)
    model.x_mean.copy_(x_mean.detach()); model.x_std.copy_(x_std.detach())
    model.y_mean.copy_(y_mean.detach()); model.y_std.copy_(y_std.detach())
    Xn = (X - x_mean) / x_std
    Sn = (S - y_mean) / y_std

    t0 = time.time()
    res = train_unified_wm(
        model, Xn, Sn, Yreg,
        WMTrainConfig(epochs=wm_epochs, batch_size=wm_batch_size),
        regime_weight=regime_weight, verbose=True,
    )
    print(f"    trained {wm_epochs} epochs in {time.time() - t0:.1f}s.  "
          f"final mse={res['history_mse'][-1]:.4f}  "
          f"final bce={res['history_bce'][-1]:.4f}")
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "agent_dim": model.agent_dim, "macro_dim": model.macro_dim,
            "shock_dim": model.shock_dim, "embed_dim": model.embed_dim,
            "n_heads": model.n_heads, "n_attn_layers": model.n_attn_layers,
            "gru_hidden": model.gru_hidden, "n_gru_layers": model.n_gru_layers,
            "h1_dim": model.h1_dim, "h2_dim": model.h2_dim,
            "n_regime_labels": model.n_regime_labels,
        },
        "label_order": label_order, "regime_weight": regime_weight,
        "history": res,
    }, os.path.join(RUNS_DIR, "world_model_regime_unified.pt"))

    # Training-time sanity check
    with torch.no_grad():
        _, regime_logits = model(Xn, return_regime=True)
        probs_pp = torch.sigmoid(regime_logits.mean(dim=2)).numpy().reshape(
            -1, len(label_order))
        labels_pp = Yreg.numpy().reshape(-1, len(label_order))
    print("\n[4] Per-channel training-time recovery:")
    for j, name in enumerate(label_order):
        auc = float(_auc_one_vs_many(probs_pp[:, j], labels_pp[:, j:j + 1])[0])
        print(f"     {auc:.3f}  {name}")

    # Extract h1, run macro-feed SAE
    H1, idx = extract_h1_from_bottlenecked(model, ens.trajectories,
                                            ens.shock_schedules)
    fm = build_feature_matrix(ens.trajectories, ens.shock_schedules)
    assert idx == fm.sample_index
    print(f"\n[5] H1: {H1.shape}  sparsity {(H1 == 0).mean():.1%}")
    np.savez_compressed(
        os.path.join(RUNS_DIR, "world_model_regime_unified_acts.npz"),
        H1=H1.astype(np.float32),
        sample_index=np.array(idx, dtype=np.int32),
    )
    feed = build_macro_feed_v3(ens.trajectories, ens.shock_schedules, model)
    print(f"[6] Macro feed: X={tuple(feed.X.shape)}")

    results: list[dict] = []
    for name, variant, n_feat, kw, epochs in SAE_CONFIGS:
        print(f"\n[7] SAE: {name}")
        torch.manual_seed(seed)
        sae = make_sae(variant, feed.D, n_feat, **kw)
        tcfg = TrainConfig(epochs=epochs, batch_size=256, lr=1e-3,
                            warmup_steps=50,
                            resample_every=max(100, epochs // 5),
                            log_every=10**6)
        t0 = time.time()
        train(sae, feed.X, tcfg, verbose=False)
        elapsed = time.time() - t0
        with torch.no_grad():
            out = sae(feed.X)
            recon = float(out.recon_loss); l0 = float((out.z.abs() > 1e-9).float().sum(dim=-1).mean())
            var_total = float(feed.X.var()); var_resid = float((feed.X - out.x_hat).var())
            ve = 1.0 - var_resid / max(var_total, 1e-12)
        torch.save({
            "state_dict": sae.state_dict(), "kind": variant,
            "feed_name": "macro_feed_v3_regime_unified",
            "input_dim": sae.input_dim, "n_features": sae.n_features,
            "feed_config": kw,
        }, os.path.join(EXP_DIR, f"{name}.pt"))
        Z = score_sae(sae, feed.X)
        rep = align(Z, feed.Y, feed.feature_vocab)
        rep.run_id = name; rep.variant = variant
        print(f"   recon={recon:.4f}  L0={l0:.2f}  VE={ve:.4f}  time={elapsed:.1f}s")
        print(f"   cov95={rep.coverage_at_0_95:.1%}  mAUC={rep.mean_best_auc:.3f}")
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
        row.update({"name": name, "n_features": n_feat, "epochs": epochs,
                    "recon_loss": recon, "l0": l0, "var_explained": ve,
                    "wall_time_s": elapsed,
                    "regime_auc_per_feature": {
                        feed.feature_vocab[j]: float(rep.alignment[:, j].max())
                        for j in range(len(feed.feature_vocab))
                        if feature_tier(feed.feature_vocab[j]) == "regime"
                    }})
        results.append(row)

    print("\n" + "=" * 90)
    print(f"REGIME-UNIFIED SUMMARY")
    print("=" * 90)
    print(f"{'name':<22s} {'w':>5s} {'ep':>4s} {'time':>6s} {'L0':>5s} {'VE':>7s}  "
          f"{'cov95':>5s} {'mAUC':>5s}  {'regi/95':>8s}  {'regi/A':>6s}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<22s} {r['n_features']:>5d} {r['epochs']:>4d} "
              f"{r['wall_time_s']:>5.0f}s {r['l0']:>5.1f} {r['var_explained']:>7.3f}  "
              f"{r['coverage_0.95']:>5.1%} {r['mean_best_auc']:>5.3f}  "
              f"{r['per_tier']['regime']['coverage_0.95']:>8.1%}  "
              f"{r['per_tier']['regime']['mean_best_auc']:>6.3f}")

    out_path = os.path.join(RUNS_DIR, "regime_unified_experiment_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
