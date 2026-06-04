"""Train the per-agent predictive world model and dump h1 activations.

Output:
    runs/world_model.pt          -- state_dict + config + normalization stats
    runs/world_model_acts.npz    -- (N_samples, h1_dim) activation matrix +
                                    sample_index alignment

Usage:
    python scripts/train_world_model.py
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import torch

from econsae.sae.world_model import (
    WorldModel, WMTrainConfig, build_world_model_data, train_world_model,
    extract_h1_activations,
)
from econsae.simulator.ensemble import generate_ensemble


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)


def main(n_trajectories: int = 32, n_periods: int = 60, seed: int = 0,
         h1_dim: int = 96, h2_dim: int = 48, epochs: int = 40,
         device: str | None = None):
    from scripts._device import resolve_device
    device = resolve_device(device)
    print("=" * 78)
    print(f"Training world model: n_traj={n_trajectories}, n_periods={n_periods}, "
          f"h1={h1_dim}, h2={h2_dim}, epochs={epochs}, device={device}")
    print("=" * 78)
    torch.manual_seed(seed)

    ens = generate_ensemble(n_trajectories=n_trajectories, n_periods=n_periods, seed=seed)
    data = build_world_model_data(ens.trajectories, ens.shock_schedules)
    print(f"  training pairs: X={tuple(data.X.shape)}  Y={tuple(data.Y.shape)}")

    model = WorldModel(h1_dim=h1_dim, h2_dim=h2_dim)
    res = train_world_model(model, data, WMTrainConfig(epochs=epochs, device=device))
    final_loss = res["history"][-1]
    print(f"  final train MSE (z-scored): {final_loss:.4f}")

    # Activation extraction below builds per-sample CPU inputs; move the
    # trained model (and its normalization buffers) back to CPU to match.
    model.to("cpu")

    # Save the model
    ckpt = os.path.join(RUNS_DIR, "world_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "agent_dim": model.agent_dim, "macro_dim": model.macro_dim,
            "shock_dim": model.shock_dim, "h1_dim": model.h1_dim, "h2_dim": model.h2_dim,
        },
        "final_train_mse": final_loss,
        "loss_history": res["history"],
    }, ckpt)
    print(f"  saved {ckpt}")

    # Extract h1 activations across all (traj, period, agent)
    H1, idx = extract_h1_activations(model, ens.trajectories, ens.shock_schedules)
    acts_path = os.path.join(RUNS_DIR, "world_model_acts.npz")
    np.savez_compressed(
        acts_path,
        H1=H1.astype(np.float32),
        sample_index=np.array(idx, dtype=np.int32),
    )
    print(f"  saved {acts_path}: H1 shape={H1.shape}, "
          f"mean sparsity={(H1 == 0).mean():.1%}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train the per-agent world model.")
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"],
                    help="compute device; 'auto' (default) uses CUDA when available")
    args = ap.parse_args()
    main(device=args.device)
