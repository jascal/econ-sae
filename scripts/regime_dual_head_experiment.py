"""Phase 6.2: dual-head + focal loss + wider SAE.

Phase 6.1 (per-channel + pos_weight) achieved 3/6 in a single run; the
union with Phase 5.1 + 5.2 stayed at 6/6 but no single recipe got
there. Diagnosis: pos_weight rebalances per-channel BCE so rare
channels learn, but it over-weights rare-class loss and steals SAE
allocation from the windowed features.

Phase 6.2 tries the principled fix:

  1. **Dual supervision head**. Pooled BCE (Phase 5.1 style) + per-channel
     BCE (Phase 5.2 style) trained simultaneously. The pooled path
     handles impulse / current-state features (good when input flags
     exist); the per-channel path handles windowed features (good when
     a specific h1 dim must internalize a multi-period statistic).
  2. **Focal loss** on the pooled head instead of pos_weight.
     Focal loss = (1 - p_t)^gamma * BCE down-weights well-classified
     examples specifically. Class imbalance is handled smoothly rather
     than via constant per-class scaling, so it doesn't over-pump rare
     classes when they're easy.
  3. **Wider downstream SAE** (n_features=2048). The macro-feed v3 input
     is 223-dim; at width 2048 the L0 budget has room to allocate one
     feature per regime label without competing with the input's other
     217 dims.

Predictions:
  - Per-channel training-time AUCs >= 0.95 across all 6 labels.
  - Downstream SAE recovers 6/6 in a single run.

Output:
  runs/world_model_regime_dual_head.pt
  runs/world_model_regime_dual_head_acts.npz
  runs/regime_dual_head_experiment/{cfg}.pt
  runs/regime_dual_head_experiment_summary.json
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
    TemporalWorldModel, WMTrainConfig, build_temporal_data,
)
from econsae.simulator.ensemble import generate_ensemble
from scripts.macro_feed_v3_experiment import build_macro_feed_v3
from scripts.regime_supervised_experiment import (
    REGIME_LABEL_ORDER, build_regime_labels,
)


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
EXP_DIR = os.path.join(RUNS_DIR, "regime_dual_head_experiment")
os.makedirs(EXP_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# DualHeadRegimeWM: per-channel + pooled, both trained jointly.
# ---------------------------------------------------------------------------
class DualHeadRegimeWM(TemporalWorldModel):
    def __init__(self, *args, n_regime_labels: int = len(REGIME_LABEL_ORDER),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_regime_labels = n_regime_labels
        # Per-channel path: reserve last n_regime_labels h1 dims (Phase 5.2 style)
        self.regime_scale = nn.Parameter(torch.ones(n_regime_labels))
        self.regime_bias = nn.Parameter(torch.zeros(n_regime_labels))
        # Pooled path: separate linear head off mean-pooled h1 (Phase 5.1 style)
        self.regime_head_pooled = nn.Linear(self.h1_dim, n_regime_labels)

    def forward(self, x: torch.Tensor, return_h1: bool = False,
                 return_regime: bool = False):
        B, T, N, _ = x.shape
        h = x.reshape(B * T, N, -1)
        h = self.input_proj(h)
        for attn, norm in zip(self.attn_layers, self.attn_norms):
            attended, _ = attn(h, h, h, need_weights=False)
            h = norm(h + attended)
        h = h.reshape(B, T, N, -1)
        h = h.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, -1)
        h, _ = self.gru(h)
        h = h.reshape(B, N, T, -1).permute(0, 2, 1, 3).contiguous()
        h1 = F.relu(self.fc1(h))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        per_channel_logits = (
            h1[..., -self.n_regime_labels:] * self.regime_scale + self.regime_bias
        )                                                       # (B, T, N, n_regime)
        pooled_logits = self.regime_head_pooled(h1.mean(dim=2))  # (B, T, n_regime)
        if return_h1 and return_regime:
            return out, per_channel_logits, pooled_logits, h1
        if return_regime:
            return out, per_channel_logits, pooled_logits
        if return_h1:
            return out, h1
        return out


# ---------------------------------------------------------------------------
def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                           gamma: float = 2.0) -> torch.Tensor:
    """Focal BCE: down-weights well-classified examples by (1 - p_t)^gamma.

    Handles class imbalance smoothly without the over-pumping behavior of
    constant pos_weight. p_t is the probability assigned to the *correct*
    class; the focal weight (1 - p_t)^gamma is near 1 for hard examples
    and near 0 for easy ones, so optimization concentrates on the
    misclassified cases regardless of class.
    """
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    focal_weight = (1.0 - p_t).clamp(min=1e-9) ** gamma
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
def train_dual_head_wm(
    model: DualHeadRegimeWM,
    Xn: torch.Tensor,
    Sn: torch.Tensor,
    Yreg: torch.Tensor,           # (B, T, n_regime_labels) period-level labels
    cfg: WMTrainConfig,
    alpha_channel: float = 1.0,   # weight on per-channel BCE
    beta_pooled: float = 1.0,     # weight on pooled focal BCE
    focal_gamma: float = 2.0,
    verbose: bool = True,
) -> dict:
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)
    history = {"mse": [], "channel_bce": [], "pooled_focal": []}
    B_total = Xn.shape[0]
    N = Xn.shape[2]
    bs = max(1, cfg.batch_size)
    Yreg_per_agent = Yreg.unsqueeze(2).expand(-1, -1, N, -1).contiguous()

    for ep in range(cfg.epochs):
        perm = torch.randperm(B_total, device=device)
        for key in history:
            history.setdefault(f"_ep_{key}", []).clear() if False else None
        ep_losses = {"mse": [], "channel_bce": [], "pooled_focal": []}
        for i in range(0, B_total, bs):
            bi = perm[i:i + bs]
            pred, per_channel_logits, pooled_logits = model(
                Xn[bi], return_regime=True
            )
            mse = F.mse_loss(pred[:, :-1], Sn[bi][:, 1:])
            channel_bce = F.binary_cross_entropy_with_logits(
                per_channel_logits, Yreg_per_agent[bi]
            )
            pooled_focal = focal_bce_with_logits(
                pooled_logits, Yreg[bi], gamma=focal_gamma
            )
            loss = mse + alpha_channel * channel_bce + beta_pooled * pooled_focal
            opt.zero_grad(); loss.backward(); opt.step()
            ep_losses["mse"].append(float(mse))
            ep_losses["channel_bce"].append(float(channel_bce))
            ep_losses["pooled_focal"].append(float(pooled_focal))
        for k, v in ep_losses.items():
            history[k].append(float(np.mean(v)))
        if verbose and (ep == 0 or (ep + 1) % max(1, cfg.epochs // 10) == 0):
            print(f"  ep {ep + 1:>3d}/{cfg.epochs}  "
                  f"mse={history['mse'][-1]:.4f}  "
                  f"ch_bce={history['channel_bce'][-1]:.4f}  "
                  f"pl_focal={history['pooled_focal'][-1]:.4f}")
    return history


def extract_h1_from_dualhead(model: DualHeadRegimeWM, trajectories, shock_schedules,
                              base_rate: float = 0.02
                              ) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    from econsae.sae.world_model import build_temporal_data
    model.eval()
    data = build_temporal_data(trajectories, shock_schedules, base_rate=base_rate)
    rows: list[np.ndarray] = []
    idx: list[tuple[int, int, int]] = []
    with torch.no_grad():
        x_mean = model.x_mean; x_std = model.x_std
        for ti in range(data.X.shape[0]):
            xt = (data.X[ti:ti + 1] - x_mean) / x_std
            _, h1 = model(xt, return_h1=True)
            h1 = h1.squeeze(0).cpu().numpy()
            T, N, _ = h1.shape
            for t in range(T):
                for a in range(N):
                    rows.append(h1[t, a])
                    idx.append((ti, t, a))
    return np.stack(rows, axis=0).astype(np.float32), idx


# Wider SAE for the macro-feed v3 substrate (223 input dims)
SAE_CONFIGS = [
    ("jr_w512_ep300",  "jumprelu", 512,  {"l0_coeff": 8e-4, "init_theta": 0.05}, 300),
    ("jr_w1024_ep300", "jumprelu", 1024, {"l0_coeff": 5e-4, "init_theta": 0.05}, 300),
    ("jr_w2048_ep300", "jumprelu", 2048, {"l0_coeff": 3e-4, "init_theta": 0.05}, 300),
]


def main(seed: int = 0, n_trajectories: int = 128, n_periods: int = 100,
         sentiment_strength: float = 0.20,
         wm_epochs: int = 50, wm_batch_size: int = 16,
         alpha_channel: float = 1.0, beta_pooled: float = 1.0,
         focal_gamma: float = 2.0):
    torch.manual_seed(seed)
    t_start = time.time()
    print("=" * 78)
    print(f"REGIME-DUAL-HEAD EXPERIMENT  "
          f"(alpha_channel={alpha_channel}, beta_pooled={beta_pooled}, "
          f"focal_gamma={focal_gamma})")
    print("=" * 78)

    ens = generate_ensemble(n_trajectories=n_trajectories, n_periods=n_periods,
                            seed=seed, sentiment_strength=sentiment_strength)
    print(f"\n[1] Ensemble in {time.time() - t_start:.1f}s")

    wm_data = build_temporal_data(ens.trajectories, ens.shock_schedules)
    Yreg, label_order = build_regime_labels(ens.trajectories, ens.shock_schedules)
    print(f"[2] Data: X={tuple(wm_data.X.shape)}  Yreg={tuple(Yreg.shape)}")
    prev = Yreg.mean(dim=(0, 1)).tolist()
    for name, p in zip(label_order, prev):
        print(f"      {p:>6.2%}  {name}")

    model = DualHeadRegimeWM(
        embed_dim=64, n_heads=4, n_attn_layers=1,
        gru_hidden=128, n_gru_layers=1, h1_dim=192, h2_dim=128,
    )
    print(f"\n[3] DualHeadRegimeWM: params={sum(p.numel() for p in model.parameters()):,}")

    X = wm_data.X; S = wm_data.states
    X_flat = X.reshape(-1, X.shape[-1]); S_flat = S.reshape(-1, S.shape[-1])
    x_mean = X_flat.mean(dim=0); x_std = X_flat.std(dim=0).clamp_min(1e-3)
    y_mean = S_flat.mean(dim=0); y_std = S_flat.std(dim=0).clamp_min(1e-3)
    model.x_mean.copy_(x_mean.detach()); model.x_std.copy_(x_std.detach())
    model.y_mean.copy_(y_mean.detach()); model.y_std.copy_(y_std.detach())
    Xn = (X - x_mean) / x_std
    Sn = (S - y_mean) / y_std

    t0 = time.time()
    history = train_dual_head_wm(
        model, Xn, Sn, Yreg,
        WMTrainConfig(epochs=wm_epochs, batch_size=wm_batch_size),
        alpha_channel=alpha_channel, beta_pooled=beta_pooled,
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
            "n_regime_labels": model.n_regime_labels,
        },
        "label_order": label_order,
        "alpha_channel": alpha_channel, "beta_pooled": beta_pooled,
        "focal_gamma": focal_gamma, "history": history,
    }, os.path.join(RUNS_DIR, "world_model_regime_dual_head.pt"))

    # Training-time sanity AUC on both heads
    with torch.no_grad():
        _, per_channel_logits, pooled_logits = model(Xn, return_regime=True)
    channel_probs = torch.sigmoid(per_channel_logits.mean(dim=2)).numpy().reshape(-1, len(label_order))
    pooled_probs = torch.sigmoid(pooled_logits).numpy().reshape(-1, len(label_order))
    labels_flat = Yreg.numpy().reshape(-1, len(label_order))
    print("\n[4] Training-time head AUCs (mean over period-pooled samples):")
    print(f"     {'label':<24s}  {'channel':>8s}  {'pooled':>8s}")
    for j, name in enumerate(label_order):
        ch_auc = float(_auc_one_vs_many(channel_probs[:, j], labels_flat[:, j:j + 1])[0])
        pl_auc = float(_auc_one_vs_many(pooled_probs[:, j], labels_flat[:, j:j + 1])[0])
        print(f"     {name:<24s}  {ch_auc:>8.3f}  {pl_auc:>8.3f}")

    # Extract h1, run macro-feed v3 SAE
    H1, idx = extract_h1_from_dualhead(model, ens.trajectories,
                                         ens.shock_schedules)
    fm = build_feature_matrix(ens.trajectories, ens.shock_schedules)
    assert idx == fm.sample_index
    print(f"\n[5] H1: {H1.shape}  sparsity {(H1 == 0).mean():.1%}")
    np.savez_compressed(
        os.path.join(RUNS_DIR, "world_model_regime_dual_head_acts.npz"),
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
            "feed_name": "macro_feed_v3_regime_dual_head",
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
    print(f"DUAL-HEAD SUMMARY")
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

    out_path = os.path.join(RUNS_DIR, "regime_dual_head_experiment_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
