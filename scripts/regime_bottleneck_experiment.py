"""Phase 5.2: feature-bottlenecked regime supervision.

Phase 5.1 pushed regime mAUC to 0.972 with a pooled linear regime head,
but `phase:expansion` and `phase:contraction` plateaued at AUC 0.92.
Diagnosis: the supervised gradient flowed through a Linear that pooled
h1 over agents, so the substrate encoded each regime label in a
DISTRIBUTED way across many h1 components -- and the SAE's L0-budget
sparse compression couldn't localize the distributed signal into a
single feature for those two windowed-regime targets.

Phase 5.2 tests the obvious fix: bottleneck the supervision through a
SINGLE h1 dimension per regime label. We reserve the last 6 dimensions
of h1 (indices h1_dim-6 .. h1_dim-1) as "regime channels". Each
(period, agent) row's regime-channel dim j is directly used as the
logit for regime label j (with optional learnable scale + bias). BCE
loss is applied per-(period, agent) so every agent's regime-channel j
gets pressured to track the period-level label j.

Predictions:
  - phase:expansion / contraction should cross AUC 0.95.
  - Other regime features stay at or above the Phase 5.1 numbers.
  - Per-agent SAEs find one feature per regime channel trivially
    (and the macro-feed SAE picks them up the same way via pooled h1).

This is the most aggressive supervision used yet -- the substrate is
LITERALLY told which h1 dim should carry each label. The point is to
identify the absolute theoretical ceiling of regime recovery under the
econ-sae benchmark, not to claim unsupervised recovery.

Output:
  runs/world_model_regime_bottleneck.pt
  runs/world_model_regime_bottleneck_acts.npz
  runs/regime_bottleneck_experiment/{cfg}.pt
  runs/regime_bottleneck_experiment_summary.json
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
EXP_DIR = os.path.join(RUNS_DIR, "regime_bottleneck_experiment")
os.makedirs(EXP_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature-bottlenecked regime-supervised TemporalWorldModel.
#
# Reserve the LAST `n_regime_labels` dimensions of h1 as direct regime
# channels. Each (period, agent) row's regime-channel j is supervised
# (via BCE) to track regime label j for that period (broadcast across
# agents). No pooling, no linear head; the substrate is forced to
# allocate one h1 channel per label.
# ---------------------------------------------------------------------------
class RegimeBottleneckedTemporalWM(TemporalWorldModel):
    def __init__(self, *args, n_regime_labels: int = len(REGIME_LABEL_ORDER),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_regime_labels = n_regime_labels
        # Optional per-channel calibration so the substrate can rescale the
        # final logits without fighting the next-state MSE for absolute
        # magnitude on those h1 dims.
        self.regime_scale = nn.Parameter(torch.ones(n_regime_labels))
        self.regime_bias = nn.Parameter(torch.zeros(n_regime_labels))

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
        h1 = F.relu(self.fc1(h))                            # (B, T, N, h1_dim)
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        # Per-(period, agent) regime logits = last n_regime_labels h1 dims
        # times learnable scale + bias.
        regime_logits = (
            h1[..., -self.n_regime_labels:] * self.regime_scale
            + self.regime_bias
        )                                                   # (B, T, N, n_regime)
        if return_h1 and return_regime:
            return out, regime_logits, h1
        if return_regime:
            return out, regime_logits
        if return_h1:
            return out, h1
        return out


# ---------------------------------------------------------------------------
def train_bottlenecked_wm(
    model: RegimeBottleneckedTemporalWM,
    Xn: torch.Tensor,         # (B, T, N, in_dim) normalized
    Sn: torch.Tensor,         # (B, T, N, agent_dim) normalized
    Yreg: torch.Tensor,       # (B, T, n_regime_labels) binary
    cfg: WMTrainConfig,
    regime_weight: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Train with L = MSE(next_state) + regime_weight * BCE(per-agent regime)."""
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)
    history_mse: list[float] = []
    history_bce: list[float] = []
    B_total = Xn.shape[0]
    N = Xn.shape[2]
    bs = max(1, cfg.batch_size)

    # Broadcast period-level regime labels to per-(period, agent) targets.
    Yreg_per_agent = Yreg.unsqueeze(2).expand(-1, -1, N, -1).contiguous()

    for ep in range(cfg.epochs):
        perm = torch.randperm(B_total, device=device)
        mse_losses = []
        bce_losses = []
        for i in range(0, B_total, bs):
            bi = perm[i:i + bs]
            pred, regime_logits = model(Xn[bi], return_regime=True)
            mse = F.mse_loss(pred[:, :-1], Sn[bi][:, 1:])
            bce = F.binary_cross_entropy_with_logits(regime_logits, Yreg_per_agent[bi])
            loss = mse + regime_weight * bce
            opt.zero_grad(); loss.backward(); opt.step()
            mse_losses.append(float(mse))
            bce_losses.append(float(bce))
        history_mse.append(float(np.mean(mse_losses)))
        history_bce.append(float(np.mean(bce_losses)))
        if verbose and (ep == 0 or (ep + 1) % max(1, cfg.epochs // 10) == 0):
            print(f"  ep {ep + 1:>3d}/{cfg.epochs}  mse={history_mse[-1]:.4f}  "
                  f"bce={history_bce[-1]:.4f}")
    return {"history_mse": history_mse, "history_bce": history_bce}


def extract_h1_from_bottlenecked(model: RegimeBottleneckedTemporalWM,
                                  trajectories, shock_schedules,
                                  base_rate: float = 0.02
                                  ) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Per-(traj, t, agent) h1 (same flat layout as the standard extractor)."""
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
    print(f"REGIME-BOTTLENECK EXPERIMENT  (regime_weight={regime_weight})")
    print("=" * 78)

    ens = generate_ensemble(n_trajectories=n_trajectories, n_periods=n_periods,
                            seed=seed, sentiment_strength=sentiment_strength)
    print(f"\n[1] Ensemble in {time.time() - t_start:.1f}s; "
          f"worst residual {max(ens.conservation_summary().values()):.2e}")

    wm_data = build_temporal_data(ens.trajectories, ens.shock_schedules)
    Yreg, label_order = build_regime_labels(ens.trajectories, ens.shock_schedules)
    print(f"[2] Data: X={tuple(wm_data.X.shape)}  Yreg={tuple(Yreg.shape)}")
    print(f"    Regime label order: {label_order}")

    model = RegimeBottleneckedTemporalWM(
        embed_dim=64, n_heads=4, n_attn_layers=1,
        gru_hidden=128, n_gru_layers=1, h1_dim=192, h2_dim=128,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[3] RegimeBottleneckedTemporalWM: params={n_params:,}  "
          f"regime channels = h1[{model.h1_dim - model.n_regime_labels}:{model.h1_dim}]")

    # Normalize
    device = torch.device("cpu")
    X = wm_data.X.to(device); S = wm_data.states.to(device)
    X_flat = X.reshape(-1, X.shape[-1]); S_flat = S.reshape(-1, S.shape[-1])
    x_mean = X_flat.mean(dim=0); x_std = X_flat.std(dim=0).clamp_min(1e-3)
    y_mean = S_flat.mean(dim=0); y_std = S_flat.std(dim=0).clamp_min(1e-3)
    model.x_mean.copy_(x_mean.detach()); model.x_std.copy_(x_std.detach())
    model.y_mean.copy_(y_mean.detach()); model.y_std.copy_(y_std.detach())
    Xn = (X - x_mean) / x_std
    Sn = (S - y_mean) / y_std

    t0 = time.time()
    res = train_bottlenecked_wm(
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
    }, os.path.join(RUNS_DIR, "world_model_regime_bottleneck.pt"))

    # ---- 4. Sanity check: how well do the regime channels themselves
    #     predict the labels (training-time AUC)?
    with torch.no_grad():
        _, regime_logits = model(Xn, return_regime=True)
        # Mean-pool over agents to get per-period predictions
        probs_pp = torch.sigmoid(regime_logits.mean(dim=2)).numpy().reshape(
            -1, len(label_order))
        labels_pp = Yreg.numpy().reshape(-1, len(label_order))
    print("\n[4] Regime-channel training-time recovery "
          "(AUC of period-mean-pooled probs vs labels):")
    for j, name in enumerate(label_order):
        auc = float(_auc_one_vs_many(probs_pp[:, j], labels_pp[:, j:j + 1])[0])
        print(f"     {auc:.3f}  {name}")

    # ---- 5. Extract h1 for SAE training ----
    H1, idx = extract_h1_from_bottlenecked(model, ens.trajectories,
                                            ens.shock_schedules)
    fm = build_feature_matrix(ens.trajectories, ens.shock_schedules)
    assert idx == fm.sample_index
    print(f"\n[5] H1: {H1.shape}  sparsity {(H1 == 0).mean():.1%}")
    np.savez_compressed(
        os.path.join(RUNS_DIR, "world_model_regime_bottleneck_acts.npz"),
        H1=H1.astype(np.float32),
        sample_index=np.array(idx, dtype=np.int32),
    )

    # ---- 6. Macro-feed v3 SAE on the bottlenecked substrate ----
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
            "feed_name": "macro_feed_v3_regime_bottleneck",
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
    print(f"REGIME-BOTTLENECK SUMMARY")
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

    out_path = os.path.join(RUNS_DIR, "regime_bottleneck_experiment_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
