"""Phase 8.1: dual-head conjunctive supervision -- push the conjunctive
tier to 8/8 in a single run.

The Phase 6.2 dual-head recipe closed the regime tier at 6/6 in one
training run. Conjunctive features were left at 6/8 best (from the union
of Phase 1.6 attn and Phase 4.2). The two stubborn features
(`young_AND_indebted`, sometimes also `prime_AND_high_cash` /
`retiree_AND_decumulating` depending on the run) cap at AUC ~0.92-0.94
unsupervised. The hypothesis: same dual-head structure that closed
regime will close conjunctive, with two key adaptations:

  1. Conjunctive labels are **per-(period, agent)**, not per-period.
     Each agent has its own conjunctive label (e.g., this particular
     young HH is indebted right now). So both the per-channel and the
     "broad" supervision heads operate per-(period, agent); we don't
     pool over agents.
  2. The downstream SAE evaluated is the **per-agent** SAE, not the
     macro-feed SAE. Conjunctive features live at per-(period, agent)
     granularity.

Architecture: ConjunctiveDualHeadWM = TemporalWorldModel + two heads:

  - Per-channel: last 8 h1 dims used directly as logits for the 8
    conjunctive labels, with learnable per-channel scale + bias.
  - Per-(period, agent) "deep" head: separate Linear off the full h1
    -> 8 logits, with focal loss for class imbalance.

Loss: MSE(next_state) + alpha * BCE(per_channel) + beta * focal_BCE(deep_head).
Both use Yconj broadcast to per-(period, agent).

Output:
  runs/world_model_conjunctive_dual_head.pt
  runs/world_model_conjunctive_dual_head_acts.npz
  runs/conjunctive_dual_head_experiment/{cfg}.pt
  runs/conjunctive_dual_head_experiment_summary.json
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
from scripts.regime_dual_head_experiment import focal_bce_with_logits


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
EXP_DIR = os.path.join(RUNS_DIR, "conjunctive_dual_head_experiment")
os.makedirs(EXP_DIR, exist_ok=True)


# Fixed ordering for the conjunctive supervision targets. Mirrors the
# alphabetical ordering convention from REGIME_LABEL_ORDER.
CONJ_LABEL_ORDER = [
    "durables_firm_high_inv",
    "firm_AND_indebted_AND_high_inventory",
    "food_firm_low_inv",
    "prime_AND_high_cash",
    "retiree_AND_decumulating",
    "services_firm_high_output",
    "young_AND_high_mpc_AND_expansion",
    "young_AND_indebted",
]


# ---------------------------------------------------------------------------
class ConjunctiveDualHeadWM(TemporalWorldModel):
    """TemporalWorldModel + per-channel + deep conjunctive supervision."""

    def __init__(self, *args, n_conj_labels: int = len(CONJ_LABEL_ORDER),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_conj_labels = n_conj_labels
        # Per-channel: dedicate last n_conj_labels h1 dims to conjunctive labels
        self.conj_scale = nn.Parameter(torch.ones(n_conj_labels))
        self.conj_bias = nn.Parameter(torch.zeros(n_conj_labels))
        # Per-(period, agent) "deep" head off the full h1 vector
        self.conj_head_full = nn.Linear(self.h1_dim, n_conj_labels)

    def forward(self, x: torch.Tensor, return_h1: bool = False,
                 return_conj: bool = False):
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
        h1 = F.relu(self.fc1(h))                              # (B, T, N, h1_dim)
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        per_channel_logits = (
            h1[..., -self.n_conj_labels:] * self.conj_scale + self.conj_bias
        )                                                     # (B, T, N, n_conj)
        deep_logits = self.conj_head_full(h1)                 # (B, T, N, n_conj)
        if return_h1 and return_conj:
            return out, per_channel_logits, deep_logits, h1
        if return_conj:
            return out, per_channel_logits, deep_logits
        if return_h1:
            return out, h1
        return out


# ---------------------------------------------------------------------------
def build_conj_labels(trajectories, shock_schedules) -> tuple[torch.Tensor, list[str]]:
    """Return (n_trajs, T, N, n_conj_labels) per-(period, agent) binary tensor."""
    n_trajs = len(trajectories)
    T = trajectories[0].T
    N = trajectories[0].n_agents
    fm = build_feature_matrix(trajectories, shock_schedules)
    full_idx = {f: j for j, f in enumerate(fm.feature_vocab)}
    n_labels = len(CONJ_LABEL_ORDER)
    Y = np.zeros((n_trajs, T, N, n_labels), dtype=np.float32)
    for ti in range(n_trajs):
        for t in range(T):
            for ai in range(N):
                row = (ti * T + t) * N + ai
                for j, name in enumerate(CONJ_LABEL_ORDER):
                    if name in full_idx and fm.Y[row, full_idx[name]] > 0:
                        Y[ti, t, ai, j] = 1.0
    return torch.tensor(Y, dtype=torch.float32), CONJ_LABEL_ORDER


# ---------------------------------------------------------------------------
def train_conj_dual_head_wm(
    model: ConjunctiveDualHeadWM,
    Xn: torch.Tensor,
    Sn: torch.Tensor,
    Yconj: torch.Tensor,        # (B, T, N, n_conj_labels) per-(period, agent)
    cfg: WMTrainConfig,
    alpha_channel: float = 1.0,
    beta_deep: float = 1.0,
    focal_gamma: float = 2.0,
    verbose: bool = True,
) -> dict:
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)
    history = {"mse": [], "channel_bce": [], "deep_focal": []}
    B_total = Xn.shape[0]
    bs = max(1, cfg.batch_size)
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
                per_channel_logits, Yconj[bi]
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


def extract_h1_from_conj_wm(model: ConjunctiveDualHeadWM, trajectories, shock_schedules,
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


# Per-agent SAEs trained on the conj-supervised h1 substrate
SAE_CONFIGS = [
    ("jr_w512_ep200",  "jumprelu", 512,  {"l0_coeff": 1e-3, "init_theta": 0.05}, 200),
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
    print(f"CONJUNCTIVE-DUAL-HEAD EXPERIMENT  "
          f"(alpha_channel={alpha_channel}, beta_deep={beta_deep}, "
          f"focal_gamma={focal_gamma}, io_network={io_network})")
    print("=" * 78)

    ens = generate_ensemble(n_trajectories=n_trajectories, n_periods=n_periods,
                            seed=seed, sentiment_strength=sentiment_strength,
                            io_network=io_network)
    print(f"\n[1] Ensemble in {time.time() - t_start:.1f}s")

    wm_data = build_temporal_data(ens.trajectories, ens.shock_schedules)
    Yconj, label_order = build_conj_labels(ens.trajectories, ens.shock_schedules)
    print(f"[2] Data: X={tuple(wm_data.X.shape)}  Yconj={tuple(Yconj.shape)}")
    prev = Yconj.mean(dim=(0, 1, 2)).tolist()
    for name, p in zip(label_order, prev):
        print(f"      {p:>6.2%}  {name}")

    model = ConjunctiveDualHeadWM(
        embed_dim=64, n_heads=4, n_attn_layers=1,
        gru_hidden=128, n_gru_layers=1, h1_dim=192, h2_dim=128,
    )
    print(f"\n[3] ConjunctiveDualHeadWM: params={sum(p.numel() for p in model.parameters()):,}")

    X = wm_data.X; S = wm_data.states
    X_flat = X.reshape(-1, X.shape[-1]); S_flat = S.reshape(-1, S.shape[-1])
    x_mean = X_flat.mean(dim=0); x_std = X_flat.std(dim=0).clamp_min(1e-3)
    y_mean = S_flat.mean(dim=0); y_std = S_flat.std(dim=0).clamp_min(1e-3)
    model.x_mean.copy_(x_mean.detach()); model.x_std.copy_(x_std.detach())
    model.y_mean.copy_(y_mean.detach()); model.y_std.copy_(y_std.detach())
    Xn = (X - x_mean) / x_std
    Sn = (S - y_mean) / y_std

    t0 = time.time()
    history = train_conj_dual_head_wm(
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
    }, os.path.join(RUNS_DIR, "world_model_conjunctive_dual_head.pt"))

    # Training-time sanity AUC on both heads (averaged across agents per period)
    with torch.no_grad():
        _, per_channel_logits, deep_logits = model(Xn, return_conj=True)
    # Per-(period, agent) prediction -> per-agent flat
    ch_probs_flat = torch.sigmoid(per_channel_logits).numpy().reshape(-1, len(label_order))
    deep_probs_flat = torch.sigmoid(deep_logits).numpy().reshape(-1, len(label_order))
    labels_flat = Yconj.numpy().reshape(-1, len(label_order))
    print("\n[4] Training-time head AUCs (per-(period, agent)):")
    print(f"     {'label':<40s}  {'channel':>8s}  {'deep':>8s}")
    for j, name in enumerate(label_order):
        ch_auc = float(_auc_one_vs_many(ch_probs_flat[:, j], labels_flat[:, j:j + 1])[0])
        dp_auc = float(_auc_one_vs_many(deep_probs_flat[:, j], labels_flat[:, j:j + 1])[0])
        print(f"     {name:<40s}  {ch_auc:>8.3f}  {dp_auc:>8.3f}")

    # Extract h1, build per-agent feed
    H1, idx = extract_h1_from_conj_wm(model, ens.trajectories,
                                       ens.shock_schedules)
    fm = build_feature_matrix(ens.trajectories, ens.shock_schedules)
    assert idx == fm.sample_index
    print(f"\n[5] H1: {H1.shape}  sparsity {(H1 == 0).mean():.1%}")
    np.savez_compressed(
        os.path.join(RUNS_DIR, "world_model_conjunctive_dual_head_acts.npz"),
        H1=H1.astype(np.float32),
        sample_index=np.array(idx, dtype=np.int32),
    )
    feed = Feed(
        name="acts_conj_dual_head", X=torch.tensor(H1, dtype=torch.float32),
        Y=fm.Y, feature_vocab=fm.feature_vocab, sample_index=fm.sample_index,
        notes="ConjunctiveDualHeadWM h1.",
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
            "feed_name": "acts_conj_dual_head",
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
    print(f"CONJ-DUAL-HEAD SUMMARY")
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

    out_path = os.path.join(RUNS_DIR, "conjunctive_dual_head_experiment_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
