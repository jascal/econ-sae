"""Multi-trajectory ensemble runner.

Generates N independent trajectories under stochastic shock schedules
drawn from `econsae.simulator.shocks.draw_shock_schedule`. Used by the
SAE pipeline to build a training distribution: each (period, agent)
pair becomes one SAE training sample, with the shock kinds active in
that period as part of the ground-truth feature vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from econsae.simulator.core import Economy, Trajectory, check_conservation
from econsae.simulator.shocks import draw_shock_schedule, ShockSchedule


@dataclass
class Ensemble:
    trajectories: list[Trajectory] = field(default_factory=list)
    shock_schedules: list[ShockSchedule] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)

    def __len__(self) -> int: return len(self.trajectories)
    def __iter__(self):       return iter(self.trajectories)
    def __getitem__(self, i): return self.trajectories[i]

    def stack_all_states(self) -> np.ndarray:
        """(n_trajs, T, n_agents, DIM)"""
        return np.stack([t.stack_states() for t in self.trajectories], axis=0)

    def conservation_summary(self) -> dict[str, float]:
        """Return the max residual across all trajectories for each identity."""
        if not self.trajectories:
            return {}
        per_traj = [check_conservation(t) for t in self.trajectories]
        keys = list(per_traj[0].keys())
        return {k: max(r[k] for r in per_traj) for k in keys}


def generate_ensemble(
    n_trajectories: int = 32,
    n_periods: int = 60,
    households_per_cohort: int = 4,
    firms_per_sector: int = 1,
    seed: int = 0,
    interest_rate: float = 0.02,
    sentiment_strength: float = 0.0,
) -> Ensemble:
    """Generate an ensemble of trajectories with reproducible seeded shocks.

    `sentiment_strength` (default 0.0) controls a behavioral non-Markov
    feedback: when > 0, household consumption shifts +/- this fraction in
    detected expansion / contraction regimes (computed from a 5-period
    trailing GDP buffer that lives outside agent state). With 0.0 the
    simulator is fully Markovian conditional on agent state -- backward
    compatible with all Phase 1 / 1.5 experiments.

    Each trajectory uses a derived seed (seed + traj_index) so that
    re-running with the same outer seed gives identical results, and
    trajectories within a run are independent.
    """
    ens = Ensemble()
    for k in range(n_trajectories):
        sub_seed = seed * 7919 + k * 17 + 1
        econ = Economy.small(
            households_per_cohort=households_per_cohort,
            firms_per_sector=firms_per_sector,
            seed=sub_seed, interest_rate=interest_rate,
        )
        econ.sentiment_strength = sentiment_strength
        sched = draw_shock_schedule(n_periods=n_periods, seed=sub_seed)
        traj = econ.rollout(n_periods=n_periods, shocks=sched.shocks)
        ens.trajectories.append(traj)
        ens.shock_schedules.append(sched)
        ens.seeds.append(sub_seed)
    return ens


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ens = generate_ensemble(n_trajectories=8, n_periods=40, seed=0)
    print(f"ensemble: {len(ens)} trajectories, "
          f"states shape: {ens.stack_all_states().shape}")
    print(f"conservation (worst over ensemble): {ens.conservation_summary()}")
    print("\nGDP distribution at t=20:")
    gdps = np.array([t.macros[20]['GDP'] for t in ens])
    print(f"  mean={gdps.mean():.2f}  std={gdps.std():.2f}  "
          f"min={gdps.min():.2f}  max={gdps.max():.2f}")
    print("\nShock kinds across all trajectories:")
    all_kinds: dict[str, int] = {}
    for s in ens.shock_schedules:
        for kinds_t in s.kinds:
            for k in kinds_t:
                all_kinds[k] = all_kinds.get(k, 0) + 1
    for k in sorted(all_kinds):
        print(f"  {k:<28s} {all_kinds[k]}")
