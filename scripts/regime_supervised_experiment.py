"""Phase 5.1: regime-supervised TemporalWorldModel + macro-feed v3 SAE.

The Phase 4.1 ratio-engineered macro-feed v3 left three windowed-regime
features plateaued at AUC 0.82-0.87 (`phase:expansion`,
`phase:contraction`, `phase:high_rate`). The diagnosis was that the
underlying signal is encoded smoothly in the substrate but the SAE
can't fire a clean step function at the label's threshold without
supervised feature allocation.

This experiment supplies exactly that supervision. We add an auxiliary
regime-prediction head to the TemporalWorldModel:

  per-(period, agent) h1 (shape: B, T, N, h1_dim)
      -> mean-pool over agents -> (B, T, h1_dim)
      -> Linear -> (B, T, 6 regime logits)

The world model is trained with combined loss:

    L = MSE(next_state_pred, state[t+1])
        + lambda * BCE_logits(regime_logits, regime_labels)

where the regime labels are the same per-period binary indicators used
to grade SAE alignment. Gradient pressure now actively wants the h1
substrate to encode step-shaped features at the regime thresholds.

After training, the standard macro-feed v3 pipeline (mean-pooled h1 as
part of the per-period SAE input) should let the SAE recover the
plateaued features. If `phase:expansion / contraction / high_rate`
cross AUC 0.95, the diagnosis is confirmed: it was a supervision gap,
not an architecture or input-engineering gap.

Note: this is a moderately aggressive intervention. The world model
now sees the regime labels at training time, which is information that
would not be available in a pure mechanistic-interpretability setting.
The point is to identify the *theoretical* recovery ceiling under the
econ-sae benchmark, not to claim unsupervised recovery.

Output:
  runs/world_model_regime_supervised.pt
  runs/world_model_regime_supervised_acts.npz
  runs/regime_supervised_experiment/{cfg}.pt
  runs/regime_supervised_experiment_summary.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

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
    align, score_sae, report_to_dict, TIERS, feature_tier,
)
from econsae.sae.models import make_sae
from econsae.sae.train import TrainConfig, train
from econsae.sae.world_model import (
    TemporalWorldModel, WMTrainConfig, build_temporal_data,
)
from econsae.simulator.ensemble import generate_ensemble
from scripts.macro_feed_v3_experiment import build_macro_feed_v3


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
EXP_DIR = os.path.join(RUNS_DIR, "regime_supervised_experiment")
os.makedirs(EXP_DIR, exist_ok=True)


# Fixed ordering for the regime supervised head's output logits.
# Order: alphabetical by name for determinism.
REGIME_LABEL_ORDER = [
    "phase:contraction",
    "phase:expansion",
    "phase:fiscal_active",
    "phase:high_leverage",
    "phase:high_rate",
    "phase:monetary_active",
]


# ---------------------------------------------------------------------------
# Regime-supervised TemporalWorldModel: inherits, adds head + altered forward.
# ---------------------------------------------------------------------------
class RegimeSupervisedTemporalWM(TemporalWorldModel):
    def __init__(self, *args, n_regime_labels: int = len(REGIME_LABEL_ORDER),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_regime_labels = n_regime_labels
        # Period-level regime head: mean-pool h1 over agents -> n_regime logits.
        self.regime_head = nn.Linear(self.h1_dim, n_regime_labels)

    def forward(self, x: torch.Tensor, return_h1: bool = False,
                 return_regime: bool = False):
        # Replicate parent forward but capture h1 and produce regime logits.
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
        h1 = F.relu(self.fc1(h))                         # (B, T, N, h1_dim)
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)                               # (B, T, N, agent_dim)
        # Regime head: per-period prediction via mean-pool over agents
        regime_logits = self.regime_head(h1.mean(dim=2))  # (B, T, n_regime_labels)
        # Variable returns based on flags
        if return_h1 and return_regime:
            return out, regime_logits, h1
        if return_regime:
            return out, regime_logits
        if return_h1:
            return out, h1
        return out


# ---------------------------------------------------------------------------
# Regime labels per (trajectory, period).
# ---------------------------------------------------------------------------
def build_regime_labels(trajectories, shock_schedules) -> tuple[torch.Tensor, list[str]]:
    """Return (n_trajs, T, n_regime_labels) binary tensor + ordered label names."""
    n_trajs = len(trajectories)
    T = trajectories[0].T
    N = trajectories[0].n_agents
    fm = build_feature_matrix(trajectories, shock_schedules)
    full_idx = {f: j for j, f in enumerate(fm.feature_vocab)}
    n_labels = len(REGIME_LABEL_ORDER)
    Y = np.zeros((n_trajs, T, n_labels), dtype=np.float32)
    for ti in range(n_trajs):
        for t in range(T):
            # Per-period labels are identical across agents; take agent 0
            full_row = (ti * T + t) * N + 0
            for j, name in enumerate(REGIME_LABEL_ORDER):
                if name in full_idx and fm.Y[full_row, full_idx[name]] > 0:
                    Y[ti, t, j] = 1.0
    return torch.tensor(Y, dtype=torch.float32), REGIME_LABEL_ORDER


# ---------------------------------------------------------------------------
# Training loop with combined next-state MSE + regime BCE loss.
# ---------------------------------------------------------------------------
def train_regime_supervised_wm(
    model: RegimeSupervisedTemporalWM,
    Xn: torch.Tensor,        # normalized (B, T, N, in_dim)
    Sn: torch.Tensor,        # normalized (B, T, N, agent_dim) -- targets for state pred
    Yreg: torch.Tensor,      # binary  (B, T, n_regime_labels)
    cfg: WMTrainConfig,
    regime_weight: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Train with L = MSE(next_state) + regime_weight * BCE(regime)."""
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)
    history_mse: list[float] = []
    history_bce: list[float] = []
    B_total = Xn.shape[0]
    bs = max(1, cfg.batch_size)
    for ep in range(cfg.epochs):
        perm = torch.randperm(B_total, device=device)
        mse_losses = []
        bce_losses = []
        for i in range(0, B_total, bs):
            bi = perm[i:i + bs]
            pred, regime_logits = model(Xn[bi], return_regime=True)
            # Next-state MSE on periods 0..T-2 -> 1..T-1
            mse = F.mse_loss(pred[:, :-1], Sn[bi][:, 1:])
            # Regime BCE on all periods (the labels exist for every t)
            bce = F.binary_cross_entropy_with_logits(regime_logits, Yreg[bi])
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


# ---------------------------------------------------------------------------
def extract_h1_from_supervised(model: RegimeSupervisedTemporalWM,
                                 trajectories, shock_schedules,
                                 base_rate: float = 0.02
                                 ) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Identical to extract_temporal_h1_activations but uses the supervised
    model's forward."""
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
            h1 = h1.squeeze(0).cpu().numpy()                   # (T, N, h1_dim)
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
    print(f"REGIME-SUPERVISED EXPERIMENT  (regime_weight={regime_weight})")
    print("=" * 78)

    # ---- 1. Ensemble ----
    ens = generate_ensemble(n_trajectories=n_trajectories, n_periods=n_periods,
                            seed=seed, sentiment_strength=sentiment_strength)
    print(f"\n[1] Ensemble in {time.time() - t_start:.1f}s; "
          f"worst residual {max(ens.conservation_summary().values()):.2e}")

    # ---- 2. Temporal data + regime labels ----
    wm_data = build_temporal_data(ens.trajectories, ens.shock_schedules)
    Yreg, label_order = build_regime_labels(ens.trajectories, ens.shock_schedules)
    print(f"[2] Data: X={tuple(wm_data.X.shape)}  S={tuple(wm_data.states.shape)}  "
          f"Yreg={tuple(Yreg.shape)}")
    print(f"    Regime label order: {label_order}")
    prev_per_label = Yreg.mean(dim=(0, 1)).tolist()
    for name, p in zip(label_order, prev_per_label):
        print(f"      {p:>6.2%}  {name}")

    # ---- 3. Train supervised model ----
    model = RegimeSupervisedTemporalWM(
        embed_dim=64, n_heads=4, n_attn_layers=1,
        gru_hidden=128, n_gru_layers=1, h1_dim=192, h2_dim=128,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[3] RegimeSupervisedTemporalWM: params={n_params:,}")

    # Normalize (same convention as parent train function)
    device = torch.device("cpu")
    X = wm_data.X.to(device); S = wm_data.states.to(device)
    X_flat = X.reshape(-1, X.shape[-1])
    S_flat = S.reshape(-1, S.shape[-1])
    x_mean = X_flat.mean(dim=0); x_std = X_flat.std(dim=0).clamp_min(1e-3)
    y_mean = S_flat.mean(dim=0); y_std = S_flat.std(dim=0).clamp_min(1e-3)
    model.x_mean.copy_(x_mean.detach())
    model.x_std.copy_(x_std.detach())
    model.y_mean.copy_(y_mean.detach())
    model.y_std.copy_(y_std.detach())
    Xn = (X - x_mean) / x_std
    Sn = (S - y_mean) / y_std

    t0 = time.time()
    res = train_regime_supervised_wm(
        model, Xn, Sn, Yreg,
        WMTrainConfig(epochs=wm_epochs, batch_size=wm_batch_size),
        regime_weight=regime_weight,
        verbose=True,
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
        "label_order": label_order,
        "regime_weight": regime_weight,
        "history": res,
    }, os.path.join(RUNS_DIR, "world_model_regime_supervised.pt"))

    # ---- 4. Verify the auxiliary head itself recovered the regime ----
    # As a sanity check: how well does the supervised regime_head predict
    # the labels it was trained on?
    with torch.no_grad():
        _, logits = model(Xn, return_regime=True)
        probs = torch.sigmoid(logits).numpy().reshape(-1, len(label_order))
        labels = Yreg.numpy().reshape(-1, len(label_order))
    print("\n[4] Regime-head training-time recovery (AUC of probs vs labels):")
    from econsae.sae.evaluation import _auc_one_vs_many
    for j, name in enumerate(label_order):
        auc = float(_auc_one_vs_many(probs[:, j], labels[:, j:j + 1])[0])
        print(f"     {auc:.3f}  {name}")

    # ---- 5. Extract h1, run macro-feed v3 SAE ----
    H1, idx = extract_h1_from_supervised(model, ens.trajectories,
                                          ens.shock_schedules)
    print(f"\n[5] H1 extraction: {H1.shape}  sparsity {(H1 == 0).mean():.1%}")
    np.savez_compressed(
        os.path.join(RUNS_DIR, "world_model_regime_supervised_acts.npz"),
        H1=H1.astype(np.float32),
        sample_index=np.array(idx, dtype=np.int32),
    )
    feed = build_macro_feed_v3(ens.trajectories, ens.shock_schedules, model)
    print(f"[6] Macro feed: X={tuple(feed.X.shape)}  Y={feed.Y.shape}  "
          f"vocab={len(feed.feature_vocab)}")

    # ---- 6. SAE training ----
    results: list[dict] = []
    for name, variant, n_feat, kw, epochs in SAE_CONFIGS:
        print(f"\n[7] SAE: {name}  ({variant}, n_features={n_feat}, epochs={epochs})")
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
            "feed_name": "macro_feed_v3_regime_supervised",
            "input_dim": sae.input_dim, "n_features": sae.n_features,
            "feed_config": kw,
        }, os.path.join(EXP_DIR, f"{name}.pt"))
        Z = score_sae(sae, feed.X)
        rep = align(Z, feed.Y, feed.feature_vocab)
        rep.run_id = name; rep.feed_name = "macro_feed_v3_regime_supervised"
        rep.variant = variant
        print(f"   recon={recon:.4f}  L0={l0:.2f}  VE={ve:.4f}  time={elapsed:.1f}s")
        print(f"   cov95={rep.coverage_at_0_95:.1%}  mAUC={rep.mean_best_auc:.3f}")
        for tier in TIERS:
            pt = rep.per_tier[tier]
            print(f"     {tier:<11s} n={pt['n_features']:>2d}  "
                  f"cov95={pt['coverage_0.95']:>5.1%}  mAUC={pt['mean_best_auc']:.3f}")
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
    print(f"REGIME-SUPERVISED SUMMARY  (regime_weight={regime_weight})")
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

    out_path = os.path.join(RUNS_DIR, "regime_supervised_experiment_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
