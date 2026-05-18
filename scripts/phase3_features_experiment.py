"""Phase 4.2: full Phase 3 simulator features in an SAE experiment.

Turns on all three non-Markov simulator channels simultaneously:
  - sentiment-driven MPC (Phase 1.7)
  - Taylor-rule central bank (Phase 3.3)
  - input-output firm network (Phase 3.4)

Then runs the complete world-model + macro-feed-v3 SAE pipeline, scoring
per-tier AUC against the standard 51-feature ground-truth vocabulary.

This experiment serves two purposes:
  1. Validate that the Phase 3 simulator features compose cleanly under
     the full pipeline (conservation, training stability).
  2. Compare per-tier AUCs against the sentiment-only baseline to see
     whether the new cross-sector cascades + endogenous monetary policy
     produce richer or harder ground-truth feature behavior.

Output:
  runs/world_model_phase3.pt
  runs/world_model_phase3_acts.npz
  runs/phase3_features_experiment/{cfg}.pt
  runs/phase3_features_experiment_summary.json
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

from econsae.ground_truth import build_feature_matrix
from econsae.sae.data import Feed
from econsae.sae.evaluation import (
    align, score_sae, report_to_dict, TIERS, feature_tier,
)
from econsae.sae.models import make_sae
from econsae.sae.train import TrainConfig, train
from econsae.sae.world_model import (
    TemporalWorldModel, WMTrainConfig,
    build_temporal_data, extract_temporal_h1_activations,
    train_temporal_world_model,
    MACRO_DIM, SHOCK_DIM, encode_macros, encode_shock,
)
from econsae.simulator.ensemble import generate_ensemble
from scripts.macro_feed_v3_experiment import build_macro_feed_v3  # ratio-engineered feed


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
EXP_DIR = os.path.join(RUNS_DIR, "phase3_features_experiment")
os.makedirs(EXP_DIR, exist_ok=True)


SAE_CONFIGS_PER_AGENT = [
    ("jr_acts_w1024_ep200", "jumprelu", 1024, {"l0_coeff": 1.5e-3, "init_theta": 0.05}, 200),
]

SAE_CONFIGS_MACRO = [
    ("jr_macro_w512_ep300", "jumprelu", 512,  {"l0_coeff": 8e-4, "init_theta": 0.05}, 300),
]


def main(seed: int = 0, n_trajectories: int = 128, n_periods: int = 100,
         sentiment_strength: float = 0.20,
         taylor_rule: bool = True, io_network: bool = True,
         wm_epochs: int = 50, wm_batch_size: int = 16):
    torch.manual_seed(seed)
    t_start = time.time()
    print("=" * 78)
    print("PHASE 3 FEATURES EXPERIMENT  (sentiment + taylor + io_network)")
    print("=" * 78)
    print(f"  sentiment_strength={sentiment_strength}  "
          f"taylor_rule={taylor_rule}  io_network={io_network}")

    # ---- 1. Ensemble with all Phase 3 features enabled ----
    t0 = time.time()
    ens = generate_ensemble(
        n_trajectories=n_trajectories, n_periods=n_periods, seed=seed,
        sentiment_strength=sentiment_strength,
        taylor_rule=taylor_rule, io_network=io_network,
    )
    cons = ens.conservation_summary()
    print(f"\n[1] Ensemble in {time.time() - t0:.1f}s; "
          f"worst residual {max(cons.values()):.2e}")

    # ---- 2. Quick GT vocab sanity check ----
    fm = build_feature_matrix(ens.trajectories, ens.shock_schedules)
    prev = fm.Y.mean(axis=0)
    print(f"\n[2] GT vocab: {len(fm.feature_vocab)} features")
    print("    Regime prevalence:")
    for j, name in enumerate(fm.feature_vocab):
        if feature_tier(name) == "regime":
            print(f"      {prev[j]:>6.2%}  {name}")
    print("    txn_period_has:b2b_purchase prevalence:")
    for j, name in enumerate(fm.feature_vocab):
        if name == "txn_period_has:b2b_purchase":
            print(f"      {prev[j]:>6.2%}  {name}  (NEW with io_network)")
            break

    # ---- 3. Train fresh TemporalWorldModel on the Phase-3 ensemble ----
    t0 = time.time()
    wm_data = build_temporal_data(ens.trajectories, ens.shock_schedules)
    print(f"\n[3] Temporal training trajectories in {time.time() - t0:.1f}s: "
          f"X={tuple(wm_data.X.shape)}")
    wm = TemporalWorldModel(
        embed_dim=64, n_heads=4, n_attn_layers=1,
        gru_hidden=128, n_gru_layers=1,
        h1_dim=192, h2_dim=128,
    )
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"    TemporalWorldModel params={n_params:,}")
    t0 = time.time()
    res = train_temporal_world_model(
        wm, wm_data, WMTrainConfig(epochs=wm_epochs, batch_size=wm_batch_size),
        verbose=False,
    )
    print(f"    trained {wm_epochs} epochs in {time.time() - t0:.1f}s. "
          f"final MSE (z-scored) = {res['history'][-1]:.4f}")
    torch.save({
        "state_dict": wm.state_dict(),
        "config": {
            "agent_dim": wm.agent_dim, "macro_dim": wm.macro_dim,
            "shock_dim": wm.shock_dim, "embed_dim": wm.embed_dim,
            "n_heads": wm.n_heads, "n_attn_layers": wm.n_attn_layers,
            "gru_hidden": wm.gru_hidden, "n_gru_layers": wm.n_gru_layers,
            "h1_dim": wm.h1_dim, "h2_dim": wm.h2_dim,
        },
        "sentiment_strength": sentiment_strength,
        "taylor_rule": taylor_rule, "io_network": io_network,
        "final_train_mse": res["history"][-1],
        "loss_history": res["history"],
    }, os.path.join(RUNS_DIR, "world_model_phase3.pt"))

    # ---- 4. Extract h1 + build per-agent feed ----
    t0 = time.time()
    H1, idx = extract_temporal_h1_activations(wm, ens.trajectories, ens.shock_schedules)
    assert idx == fm.sample_index, "activation index mismatch"
    print(f"\n[4] H1 extraction in {time.time() - t0:.1f}s: "
          f"H1={H1.shape}  sparsity={(H1 == 0).mean():.1%}")
    np.savez_compressed(
        os.path.join(RUNS_DIR, "world_model_phase3_acts.npz"),
        H1=H1.astype(np.float32),
        sample_index=np.array(idx, dtype=np.int32),
    )
    feed_acts = Feed(
        name="acts_phase3", X=torch.tensor(H1, dtype=torch.float32),
        Y=fm.Y, feature_vocab=fm.feature_vocab, sample_index=fm.sample_index,
        notes=f"TemporalWorldModel h1 with all Phase 3 features on.",
    )

    # ---- 5. Build macro-feed-v3 (per-period, with engineered ratios) ----
    feed_macro = build_macro_feed_v3(ens.trajectories, ens.shock_schedules, wm)
    print(f"[5] Macro feed: X={tuple(feed_macro.X.shape)}  Y={feed_macro.Y.shape}  "
          f"vocab={len(feed_macro.feature_vocab)}")

    # ---- 6. Train per-agent SAEs (categorical / bucketed / conjunctive) ----
    results: list[dict] = []
    for name, variant, n_feat, kw, epochs in SAE_CONFIGS_PER_AGENT:
        print(f"\n[6] SAE on PER-AGENT feed: {name}")
        torch.manual_seed(seed)
        sae = make_sae(variant, feed_acts.D, n_feat, **kw)
        tcfg = TrainConfig(epochs=epochs, batch_size=512, lr=1e-3,
                            warmup_steps=50,
                            resample_every=max(100, epochs // 5),
                            log_every=10**6)
        t0 = time.time()
        train(sae, feed_acts.X, tcfg, verbose=False)
        elapsed = time.time() - t0
        with torch.no_grad():
            out = sae(feed_acts.X)
            recon = float(out.recon_loss); l0 = float((out.z.abs() > 1e-9).float().sum(dim=-1).mean())
            var_total = float(feed_acts.X.var()); var_resid = float((feed_acts.X - out.x_hat).var())
            ve = 1.0 - var_resid / max(var_total, 1e-12)
        torch.save({
            "state_dict": sae.state_dict(), "kind": variant, "feed_name": "acts_phase3",
            "input_dim": sae.input_dim, "n_features": sae.n_features, "feed_config": kw,
        }, os.path.join(EXP_DIR, f"{name}.pt"))
        Z = score_sae(sae, feed_acts.X)
        rep = align(Z, feed_acts.Y, feed_acts.feature_vocab)
        rep.run_id = name; rep.feed_name = "acts_phase3"; rep.variant = variant
        print(f"   recon={recon:.4f}  L0={l0:.2f}  VE={ve:.4f}  time={elapsed:.1f}s")
        print(f"   cov95={rep.coverage_at_0_95:.1%}  mAUC={rep.mean_best_auc:.3f}")
        for tier in TIERS:
            pt = rep.per_tier[tier]
            print(f"     {tier:<11s} n={pt['n_features']:>2d}  "
                  f"cov95={pt['coverage_0.95']:>5.1%}  mAUC={pt['mean_best_auc']:.3f}")
        # Conjunctive per-feature
        print("   conjunctive AUC per feature:")
        for j, gname in enumerate(feed_acts.feature_vocab):
            if feature_tier(gname) == "conjunctive":
                best = float(rep.alignment[:, j].max())
                marker = " ***" if best >= 0.95 else ""
                print(f"     {best:.3f}  {gname:<48s}{marker}")
        row = report_to_dict(rep)
        row.update({"name": name, "n_features": n_feat, "epochs": epochs,
                    "recon_loss": recon, "l0": l0, "var_explained": ve,
                    "wall_time_s": elapsed, "feed": "acts_phase3",
                    "auc_per_feature": {
                        feed_acts.feature_vocab[j]: float(rep.alignment[:, j].max())
                        for j in range(len(feed_acts.feature_vocab))
                    }})
        results.append(row)

    # ---- 7. Train macro-feed SAEs (regime) ----
    for name, variant, n_feat, kw, epochs in SAE_CONFIGS_MACRO:
        print(f"\n[7] SAE on MACRO-FEED v3: {name}")
        torch.manual_seed(seed)
        sae = make_sae(variant, feed_macro.D, n_feat, **kw)
        tcfg = TrainConfig(epochs=epochs, batch_size=256, lr=1e-3,
                            warmup_steps=50,
                            resample_every=max(100, epochs // 5),
                            log_every=10**6)
        t0 = time.time()
        train(sae, feed_macro.X, tcfg, verbose=False)
        elapsed = time.time() - t0
        with torch.no_grad():
            out = sae(feed_macro.X)
            recon = float(out.recon_loss); l0 = float((out.z.abs() > 1e-9).float().sum(dim=-1).mean())
            var_total = float(feed_macro.X.var()); var_resid = float((feed_macro.X - out.x_hat).var())
            ve = 1.0 - var_resid / max(var_total, 1e-12)
        torch.save({
            "state_dict": sae.state_dict(), "kind": variant, "feed_name": "macro_feed_v3_phase3",
            "input_dim": sae.input_dim, "n_features": sae.n_features, "feed_config": kw,
        }, os.path.join(EXP_DIR, f"{name}.pt"))
        Z = score_sae(sae, feed_macro.X)
        rep = align(Z, feed_macro.Y, feed_macro.feature_vocab)
        rep.run_id = name; rep.feed_name = "macro_feed_v3_phase3"; rep.variant = variant
        print(f"   recon={recon:.4f}  L0={l0:.2f}  VE={ve:.4f}  time={elapsed:.1f}s")
        print(f"   cov95={rep.coverage_at_0_95:.1%}  mAUC={rep.mean_best_auc:.3f}")
        for tier in TIERS:
            pt = rep.per_tier[tier]
            print(f"     {tier:<11s} n={pt['n_features']:>2d}  "
                  f"cov95={pt['coverage_0.95']:>5.1%}  mAUC={pt['mean_best_auc']:.3f}")
        print("   regime AUC per feature:")
        for j, gname in enumerate(feed_macro.feature_vocab):
            if feature_tier(gname) == "regime":
                best = float(rep.alignment[:, j].max())
                marker = " ***" if best >= 0.95 else ""
                print(f"     {best:.3f}  {gname:<48s}{marker}")
        row = report_to_dict(rep)
        row.update({"name": name, "n_features": n_feat, "epochs": epochs,
                    "recon_loss": recon, "l0": l0, "var_explained": ve,
                    "wall_time_s": elapsed, "feed": "macro_feed_v3_phase3",
                    "auc_per_feature": {
                        feed_macro.feature_vocab[j]: float(rep.alignment[:, j].max())
                        for j in range(len(feed_macro.feature_vocab))
                    }})
        results.append(row)

    # ---- 8. Save summary ----
    out_path = os.path.join(RUNS_DIR, "phase3_features_experiment_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
