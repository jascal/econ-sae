"""Phase 8.2: per-channel pos_weight for the rare conjunctive features.

Phase 8.1 reached 7/8 conjunctive features at AUC >= 0.95 with the
dual-head architecture, but `young_AND_indebted` (0.35% prevalence,
rarest conjunctive feature) stuck at AUC 0.944. The training-time
diagnostic showed:

  - deep head (with focal loss):   AUC 0.952  -- substrate DID encode it
  - per-channel head (plain BCE):  AUC 0.510  -- dedicated channel
                                                  collapsed under class
                                                  imbalance

So the SAE downstream couldn't allocate a dedicated feature for
young_AND_indebted because the dedicated h1 channel was uninformative
(plain BCE on a single dim at 0.35% positive class can't beat the
"always predict 0" minimizer).

Fix: add per-channel pos_weight to the BCE for the per-channel head.
With pos_weight[j] = (1 - prev[j]) / prev[j] (clipped to [1, 50]), the
rare-feature channels get up-weighted positive-class gradients and can
actually learn the label on a single dim. The deep head's focal loss
stays as-is (focal already handles class imbalance smoothly).

Hypothesis: per-channel pos_weight clears the substrate-side
bottleneck. The downstream SAE then has 8 clean dedicated channels
(one per conjunctive label) and recovers 8/8 in a single run.

Output:
  runs/world_model_conjunctive_unified.pt
  runs/world_model_conjunctive_unified_acts.npz
  runs/conjunctive_unified_experiment/{cfg}.pt
  runs/conjunctive_unified_experiment_summary.json
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
from econsae.sae.world_model import WMTrainConfig, build_temporal_data
from econsae.simulator.ensemble import generate_ensemble
from scripts.conjunctive_dual_head_experiment import (
    CONJ_LABEL_ORDER, ConjunctiveDualHeadWM, build_conj_labels,
    extract_h1_from_conj_wm,
)
from scripts.regime_dual_head_experiment import focal_bce_with_logits


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
EXP_DIR = os.path.join(RUNS_DIR, "conjunctive_unified_experiment")
os.makedirs(EXP_DIR, exist_ok=True)


def train_unified_wm(
    model: ConjunctiveDualHeadWM,
    Xn: torch.Tensor,
    Sn: torch.Tensor,
    Yconj: torch.Tensor,
    cfg: WMTrainConfig,
    alpha_channel: float = 1.0,
    beta_deep: float = 1.0,
    focal_gamma: float = 2.0,
    pos_weight_clip: tuple[float, float] = (1.0, 50.0),
    verbose: bool = True,
) -> dict:
    """Train with per-channel pos_weight BCE + deep focal BCE + next-state MSE."""
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)
    history = {"mse": [], "channel_bce": [], "deep_focal": []}
    B_total = Xn.shape[0]
    bs = max(1, cfg.batch_size)

    # Per-channel pos_weight from training-set prevalence
    prev_per_label = Yconj.mean(dim=(0, 1, 2)).clamp_min(1e-9)
    pos_weight = ((1.0 - prev_per_label) / prev_per_label).clamp(
        min=pos_weight_clip[0], max=pos_weight_clip[1]
    ).to(device)
    if verbose:
        print("  pos_weight per label:")
        for name, p, w in zip(CONJ_LABEL_ORDER,
                               prev_per_label.cpu().tolist(),
                               pos_weight.cpu().tolist()):
            print(f"    {name:<40s}  prev={p:.4f}  pos_weight={w:.2f}")

    for ep in range(cfg.epochs):
        perm = torch.randperm(B_total, device=device)
        losses = {"mse": [], "channel_bce": [], "deep_focal": []}
        for i in range(0, B_total, bs):
            bi = perm[i:i + bs]
            pred, per_channel_logits, deep_logits = model(
                Xn[bi], return_conj=True
            )
            mse = F.mse_loss(pred[:, :-1], Sn[bi][:, 1:])
            channel_bce = F.binary_cross_entropy_with_logits(
                per_channel_logits, Yconj[bi], pos_weight=pos_weight
            )
            deep_focal = focal_bce_with_logits(
                deep_logits, Yconj[bi], gamma=focal_gamma
            )
            loss = mse + alpha_channel * channel_bce + beta_deep * deep_focal
            opt.zero_grad(); loss.backward(); opt.step()
            losses["mse"].append(float(mse))
            losses["channel_bce"].append(float(channel_bce))
            losses["deep_focal"].append(float(deep_focal))
        for k, v in losses.items():
            history[k].append(float(np.mean(v)))
        if verbose and (ep == 0 or (ep + 1) % max(1, cfg.epochs // 10) == 0):
            print(f"  ep {ep + 1:>3d}/{cfg.epochs}  "
                  f"mse={history['mse'][-1]:.4f}  "
                  f"ch_bce={history['channel_bce'][-1]:.4f}  "
                  f"deep_focal={history['deep_focal'][-1]:.4f}")
    return history


SAE_CONFIGS = [
    ("jr_w1024_ep200", "jumprelu", 1024, {"l0_coeff": 8e-4, "init_theta": 0.05}, 200),
]


def main(seed: int = 0, n_trajectories: int = 128, n_periods: int = 100,
         sentiment_strength: float = 0.20,
         wm_epochs: int = 50, wm_batch_size: int = 16,
         alpha_channel: float = 1.0, beta_deep: float = 1.0,
         focal_gamma: float = 2.0,
         io_network: bool = True):
    torch.manual_seed(seed)
    t_start = time.time()
    print("=" * 78)
    print(f"CONJUNCTIVE-UNIFIED EXPERIMENT  (per-channel pos_weight + deep focal)")
    print("=" * 78)

    ens = generate_ensemble(n_trajectories=n_trajectories, n_periods=n_periods,
                            seed=seed, sentiment_strength=sentiment_strength,
                            io_network=io_network)
    print(f"\n[1] Ensemble in {time.time() - t_start:.1f}s")

    wm_data = build_temporal_data(ens.trajectories, ens.shock_schedules)
    Yconj, label_order = build_conj_labels(ens.trajectories, ens.shock_schedules)
    print(f"[2] Data: X={tuple(wm_data.X.shape)}  Yconj={tuple(Yconj.shape)}")

    model = ConjunctiveDualHeadWM(
        embed_dim=64, n_heads=4, n_attn_layers=1,
        gru_hidden=128, n_gru_layers=1, h1_dim=192, h2_dim=128,
    )
    print(f"\n[3] ConjunctiveDualHeadWM with pos_weight: "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    X = wm_data.X; S = wm_data.states
    X_flat = X.reshape(-1, X.shape[-1]); S_flat = S.reshape(-1, S.shape[-1])
    x_mean = X_flat.mean(dim=0); x_std = X_flat.std(dim=0).clamp_min(1e-3)
    y_mean = S_flat.mean(dim=0); y_std = S_flat.std(dim=0).clamp_min(1e-3)
    model.x_mean.copy_(x_mean.detach()); model.x_std.copy_(x_std.detach())
    model.y_mean.copy_(y_mean.detach()); model.y_std.copy_(y_std.detach())
    Xn = (X - x_mean) / x_std
    Sn = (S - y_mean) / y_std

    t0 = time.time()
    history = train_unified_wm(
        model, Xn, Sn, Yconj,
        WMTrainConfig(epochs=wm_epochs, batch_size=wm_batch_size),
        alpha_channel=alpha_channel, beta_deep=beta_deep,
        focal_gamma=focal_gamma, verbose=True,
    )
    print(f"    trained {wm_epochs} epochs in {time.time() - t0:.1f}s")

    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "agent_dim": model.agent_dim, "macro_dim": model.macro_dim,
            "shock_dim": model.shock_dim, "embed_dim": model.embed_dim,
            "n_heads": model.n_heads, "n_attn_layers": model.n_attn_layers,
            "gru_hidden": model.gru_hidden, "n_gru_layers": model.n_gru_layers,
            "h1_dim": model.h1_dim, "h2_dim": model.h2_dim,
            "n_conj_labels": model.n_conj_labels,
        },
        "label_order": label_order,
        "alpha_channel": alpha_channel, "beta_deep": beta_deep,
        "focal_gamma": focal_gamma, "io_network": io_network,
        "history": history,
    }, os.path.join(RUNS_DIR, "world_model_conjunctive_unified.pt"))

    # Training-time AUCs
    with torch.no_grad():
        _, per_channel_logits, deep_logits = model(Xn, return_conj=True)
    ch_probs = torch.sigmoid(per_channel_logits).numpy().reshape(-1, len(label_order))
    dp_probs = torch.sigmoid(deep_logits).numpy().reshape(-1, len(label_order))
    labels_flat = Yconj.numpy().reshape(-1, len(label_order))
    print("\n[4] Training-time head AUCs:")
    print(f"     {'label':<40s}  {'channel':>8s}  {'deep':>8s}")
    for j, name in enumerate(label_order):
        ch_auc = float(_auc_one_vs_many(ch_probs[:, j], labels_flat[:, j:j + 1])[0])
        dp_auc = float(_auc_one_vs_many(dp_probs[:, j], labels_flat[:, j:j + 1])[0])
        marker = " <- pos_weight target" if name == "young_AND_indebted" else ""
        print(f"     {name:<40s}  {ch_auc:>8.3f}  {dp_auc:>8.3f}{marker}")

    H1, idx = extract_h1_from_conj_wm(model, ens.trajectories,
                                        ens.shock_schedules)
    fm = build_feature_matrix(ens.trajectories, ens.shock_schedules)
    assert idx == fm.sample_index
    print(f"\n[5] H1: {H1.shape}  sparsity {(H1 == 0).mean():.1%}")
    np.savez_compressed(
        os.path.join(RUNS_DIR, "world_model_conjunctive_unified_acts.npz"),
        H1=H1.astype(np.float32),
        sample_index=np.array(idx, dtype=np.int32),
    )
    feed = Feed(
        name="acts_conj_unified", X=torch.tensor(H1, dtype=torch.float32),
        Y=fm.Y, feature_vocab=fm.feature_vocab, sample_index=fm.sample_index,
        notes="ConjunctiveDualHeadWM h1 with per-channel pos_weight.",
    )

    results: list[dict] = []
    for name, variant, n_feat, kw, epochs in SAE_CONFIGS:
        print(f"\n[6] SAE: {name}")
        torch.manual_seed(seed)
        sae = make_sae(variant, feed.D, n_feat, **kw)
        tcfg = TrainConfig(epochs=epochs, batch_size=512, lr=1e-3,
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
            "feed_name": "acts_conj_unified",
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
        print("   conjunctive AUC per feature:")
        for j, gname in enumerate(feed.feature_vocab):
            if feature_tier(gname) == "conjunctive":
                best = float(rep.alignment[:, j].max())
                marker = " ***" if best >= 0.95 else ""
                print(f"     {best:.3f}  {gname:<48s}{marker}")
        row = report_to_dict(rep)
        row.update({"name": name, "n_features": n_feat, "epochs": epochs,
                    "recon_loss": recon, "l0": l0, "var_explained": ve,
                    "wall_time_s": elapsed,
                    "conj_auc_per_feature": {
                        feed.feature_vocab[j]: float(rep.alignment[:, j].max())
                        for j in range(len(feed.feature_vocab))
                        if feature_tier(feed.feature_vocab[j]) == "conjunctive"
                    }})
        results.append(row)

    print("\n" + "=" * 90)
    print(f"CONJ-UNIFIED SUMMARY")
    print("=" * 90)
    print(f"{'name':<22s} {'w':>5s} {'ep':>4s} {'time':>6s} {'L0':>5s} {'VE':>7s}  "
          f"{'cov95':>5s} {'mAUC':>5s}  {'conj/95':>8s}  {'conj/A':>6s}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<22s} {r['n_features']:>5d} {r['epochs']:>4d} "
              f"{r['wall_time_s']:>5.0f}s {r['l0']:>5.1f} {r['var_explained']:>7.3f}  "
              f"{r['coverage_0.95']:>5.1%} {r['mean_best_auc']:>5.3f}  "
              f"{r['per_tier']['conjunctive']['coverage_0.95']:>8.1%}  "
              f"{r['per_tier']['conjunctive']['mean_best_auc']:>6.3f}")

    out_path = os.path.join(RUNS_DIR, "conjunctive_unified_experiment_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
