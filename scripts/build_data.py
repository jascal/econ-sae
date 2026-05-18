"""Generate the Phase 0.5 ensemble bundle under data/.

Output:
    data/econ_ensemble.npz with arrays:
        states         (n_traj, T, n_agents, DIM) float32  -- agent state trajectories
        macros_<key>   (n_traj, T) float32                -- per-period macro aggregates
        labels_X       (N_samples, DIM) float32           -- flat training inputs for SAE
        labels_Y       (N_samples, V) uint8               -- ground-truth feature labels
        sample_index   (N_samples, 3) int32               -- (traj, period, agent) for each row
        feature_vocab  (V,) str                           -- feature name strings
        coord_names    (DIM,) str                         -- agent coord names
        agent_names    (n_agents,) str
        agent_kinds    (n_agents,) str
        agent_cohorts  (n_agents,) str
        agent_firm_sectors (n_agents,) str
        shock_kinds_per_period (n_traj, T) object        -- list of active shock kinds

Phase 0.5 defaults: 32 trajectories x 60 periods x 17 agents x 23 coords
gives ~32k SAE samples, 51 ground-truth features, conservation residuals
at machine precision on every trajectory.

Usage:
    python scripts/build_data.py
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np

from econsae.embeddings import COORDS, DIM
from econsae.simulator.ensemble import generate_ensemble
from econsae.simulator.core import check_conservation
from econsae.ground_truth import build_feature_matrix


DATA_DIR = os.path.join(REPO_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def main(n_trajectories: int = 32, n_periods: int = 60, seed: int = 0):
    print("=" * 78)
    print(f"Phase 0.5 ensemble: n_traj={n_trajectories}, n_periods={n_periods}, seed={seed}")
    print("=" * 78)

    ens = generate_ensemble(
        n_trajectories=n_trajectories, n_periods=n_periods, seed=seed,
    )
    cons = ens.conservation_summary()
    print(f"  conservation (worst over ensemble): "
          f"max residual = {max(cons.values()):.2e}")
    for k, v in cons.items():
        print(f"    {k:<22s} {v:.2e}")

    states = ens.stack_all_states().astype(np.float32)   # (n_traj, T, N, DIM)
    print(f"\n  states tensor: {states.shape}")

    # Collect per-period macros into per-key (n_traj, T) arrays
    macro_keys = list(ens.trajectories[0].macros[0].keys())
    macros = {
        k: np.array(
            [[m[k] for m in t.macros] for t in ens.trajectories], dtype=np.float32
        )
        for k in macro_keys
    }
    print(f"  macros: {len(macro_keys)} keys, shape per key {macros[macro_keys[0]].shape}")

    # Build SAE-ready (X, Y) feature matrix
    fm = build_feature_matrix(ens.trajectories, ens.shock_schedules)
    print(f"  feature matrix: X={fm.X.shape}, Y={fm.Y.shape}, "
          f"vocab={len(fm.feature_vocab)} GT features")
    prev = fm.Y.mean(axis=0)
    print(f"  feature prevalence stats: "
          f"in [5%,95%]: {int(((prev>=0.05)&(prev<=0.95)).sum())}/{len(prev)}; "
          f"never observed: {int((prev==0).sum())}")

    # Pack shock kind sets as 'kind1|kind2|...' strings per period
    shock_kinds_grid = np.empty((n_trajectories, n_periods), dtype=object)
    for ti, sched in enumerate(ens.shock_schedules):
        for t, ks in enumerate(sched.kinds):
            shock_kinds_grid[ti, t] = "|".join(sorted(ks))

    out_path = os.path.join(DATA_DIR, "econ_ensemble.npz")
    np.savez_compressed(
        out_path,
        states=states,
        labels_X=fm.X.astype(np.float32),
        labels_Y=fm.Y.astype(np.uint8),
        sample_index=np.array(fm.sample_index, dtype=np.int32),
        feature_vocab=np.array(fm.feature_vocab),
        coord_names=np.array(COORDS),
        agent_names=np.array(ens.trajectories[0].agent_names),
        agent_kinds=np.array(ens.trajectories[0].agent_kinds),
        agent_cohorts=np.array([c or "" for c in ens.trajectories[0].agent_cohorts]),
        agent_firm_sectors=np.array([s or "" for s in ens.trajectories[0].agent_firm_sectors]),
        shock_kinds_per_period=shock_kinds_grid,
        **{f"macro_{k}": v for k, v in macros.items()},
    )
    print(f"\n  wrote {out_path}  ({os.path.getsize(out_path):,} bytes)")


if __name__ == "__main__":
    main()
