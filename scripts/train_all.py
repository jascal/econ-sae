"""Phase 1 SAE runner: train 3 SAE variants on 3 feeds.

Feeds:
    raw       -- agent state vectors (R^23)
    embedded  -- raw projected through a random 23x12 linear map (superposition)
    acts      -- h1 activations of the trained world model (R^96)

Variants: TopK, L1, JumpReLU.

Per-(feed, variant) checkpoint goes to runs/{feed}__{variant}.pt; a global
runs/summary.json captures recon / L0 / dead / variance-explained.

Usage:
    python scripts/train_world_model.py   # one-time: produces runs/world_model*.pt
    python scripts/train_all.py
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

from econsae.sae.data import feed_raw, feed_embedded, feed_acts
from econsae.sae.models import make_sae
from econsae.sae.train import TrainConfig, train
from econsae.sae.world_model import WorldModel
from econsae.simulator.ensemble import generate_ensemble


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)


FEED_CONFIGS = {
    # n_features: how many learned features to allocate; expectation is roughly
    # n_features >= 2x ground-truth vocabulary so good SAEs have headroom.
    "raw":      dict(n_features=128, epochs=120, batch_size=256,
                     topk_k=8, l1_coeff=4e-3, l0_coeff=2e-3, init_theta=0.05),
    "embedded": dict(n_features=128, epochs=200, batch_size=256,
                     topk_k=8, l1_coeff=4e-3, l0_coeff=2e-3, init_theta=0.05),
    "acts":     dict(n_features=256, epochs=200, batch_size=256,
                     topk_k=12, l1_coeff=3e-3, l0_coeff=1.5e-3, init_theta=0.05),
}


def load_world_model() -> WorldModel:
    ckpt_path = os.path.join(RUNS_DIR, "world_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} not found. Run scripts/train_world_model.py first."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = WorldModel(
        agent_dim=cfg["agent_dim"], macro_dim=cfg["macro_dim"],
        shock_dim=cfg["shock_dim"], h1_dim=cfg["h1_dim"], h2_dim=cfg["h2_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def build_feed(name: str, ens, base_rate: float = 0.02, monetary_step: float = 0.01):
    if name == "raw":
        return feed_raw(ens, base_rate=base_rate, monetary_step=monetary_step)
    if name == "embedded":
        return feed_embedded(ens, embed_dim=12, seed=0,
                             base_rate=base_rate, monetary_step=monetary_step)
    if name == "acts":
        return feed_acts(ens, load_world_model(),
                         base_rate=base_rate, monetary_step=monetary_step)
    raise ValueError(name)


def build_sae(variant: str, input_dim: int, feed_name: str):
    cfg = FEED_CONFIGS[feed_name]
    n_features = cfg["n_features"]
    if variant == "topk":
        return make_sae("topk", input_dim, n_features, k=cfg["topk_k"])
    if variant == "l1":
        return make_sae("l1", input_dim, n_features, l1_coeff=cfg["l1_coeff"])
    if variant == "jumprelu":
        return make_sae("jumprelu", input_dim, n_features,
                        l0_coeff=cfg["l0_coeff"], init_theta=cfg["init_theta"])
    raise ValueError(variant)


def main(n_trajectories: int = 32, n_periods: int = 60, seed: int = 0,
         calibrated: str | None = None, quick: bool = False,
         device: str | None = None):
    from scripts._calibration_arm import resolve_arm
    from scripts._device import resolve_device
    arm = resolve_arm(calibrated)
    device = resolve_device(device)
    print(f"  device: {device}")
    epoch_cap = 20 if quick else None

    print("=" * 78)
    print(f"[{arm.label}] ensemble (n_traj={n_trajectories}, n_periods={n_periods})"
          + (f"  config={calibrated}" if calibrated else ""))
    if arm.sim_config is not None:
        print("  note: the 'acts' feed runs the baseline world_model.pt on "
              "calibrated\n        data (encoder transfer). For a matched-encoder "
              "A/B use\n        scripts/phase10_calibrated_benchmark.py.")
    print("=" * 78)
    torch.manual_seed(seed)
    ens = generate_ensemble(n_trajectories=n_trajectories, n_periods=n_periods,
                            seed=seed, sim_config=arm.sim_config)

    rows: list[dict] = []
    summary: dict = {}

    for feed_name in ("raw", "embedded", "acts"):
        print("\n" + "=" * 78)
        print(f"FEED: {feed_name}")
        print("=" * 78)
        feed = build_feed(feed_name, ens, base_rate=arm.base_rate,
                          monetary_step=arm.monetary_step)
        print(f"  X: {tuple(feed.X.shape)}  Y: {feed.Y.shape}  "
              f"vocab: {len(feed.feature_vocab)} GT features")
        print(f"  notes: {feed.notes}")

        fcfg = FEED_CONFIGS[feed_name]
        for variant in ("topk", "l1", "jumprelu"):
            print(f"\n--- {variant} on {feed_name} ---")
            torch.manual_seed(seed)
            sae = build_sae(variant, feed.D, feed_name)
            epochs = min(fcfg["epochs"], epoch_cap) if epoch_cap else fcfg["epochs"]
            tcfg = TrainConfig(
                epochs=epochs, batch_size=fcfg["batch_size"],
                lr=1e-3, warmup_steps=50,
                resample_every=max(100, epochs // 5),
                log_every=max(50, epochs // 4),
                device=device,
            )
            t0 = time.time()
            hist = train(sae, feed.X, tcfg, verbose=False)
            elapsed = time.time() - t0

            with torch.no_grad():
                X_eval = feed.X.to(device)
                out = sae(X_eval)
                final_recon = float(out.recon_loss)
                final_l0 = float((out.z.abs() > 1e-9).float().sum(dim=-1).mean())
                dead = float((sae.steps_dead >= tcfg.resample_threshold).float().mean())
                var_total = float(X_eval.var())
                var_resid = float((X_eval - out.x_hat).var())
                ve = 1.0 - var_resid / max(var_total, 1e-12)

            run_id = f"{feed_name}__{variant}"
            ckpt_path = os.path.join(RUNS_DIR, f"{run_id}{arm.suffix}.pt")
            torch.save({
                "state_dict": sae.state_dict(),
                "kind": variant, "feed_name": feed_name,
                "input_dim": sae.input_dim, "n_features": sae.n_features,
                "feed_config": fcfg,
            }, ckpt_path)

            print(f"  recon={final_recon:.4f}  L0={final_l0:.2f}  "
                  f"dead={dead:.2%}  VE={ve:.4f}  "
                  f"resamples={len(hist.resamples)}  time={elapsed:.1f}s")

            row = {
                "feed": feed_name, "variant": variant, "run_id": run_id,
                "input_dim": sae.input_dim, "n_features": sae.n_features,
                "recon_loss": final_recon, "l0": final_l0,
                "dead_fraction": dead, "var_explained": ve,
                "resamples_count": len(hist.resamples),
                "n_train_steps": hist.step[-1] if hist.step else 0,
                "wall_time_s": elapsed,
            }
            rows.append(row); summary[run_id] = row

    print("\n" + "=" * 78)
    print("SUMMARY: 3 variants x 3 feeds")
    print("=" * 78)
    print(f"{'feed':<14s} {'variant':<10s} {'recon':>8s} {'L0':>6s} "
          f"{'dead':>6s} {'VE':>8s} {'time':>7s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['feed']:<14s} {r['variant']:<10s} {r['recon_loss']:>8.4f} "
              f"{r['l0']:>6.2f} {r['dead_fraction']:>6.1%} {r['var_explained']:>8.4f} "
              f"{r['wall_time_s']:>6.1f}s")

    summary_name = "summary_calibrated.json" if arm.suffix else "summary.json"
    with open(os.path.join(RUNS_DIR, summary_name), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote runs/{summary_name}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Phase 1 SAE runner (3 feeds x 3 variants).")
    ap.add_argument("--calibrated", metavar="CONFIG.json", default=None,
                    help="run the calibrated arm from a fitted SimConfig "
                         "(writes __calibrated-suffixed artifacts)")
    ap.add_argument("--quick", action="store_true", help="cap epochs at 20 for a fast run")
    ap.add_argument("--n-traj", type=int, default=32)
    ap.add_argument("--n-periods", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"],
                    help="compute device; 'auto' (default) uses CUDA when available")
    args = ap.parse_args()
    main(n_trajectories=args.n_traj, n_periods=args.n_periods, seed=args.seed,
         calibrated=args.calibrated, quick=args.quick, device=args.device)
